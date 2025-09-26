"""
genius_client.py

A minimal, robust client for the official Genius developer API (api.genius.com)
focused on METADATA only (no lyrics). Suitable for academic/portfolio use.

Features:
- Search songs by free text (title + artist) and pick best match.
- Retrieve structured song metadata by song ID.
- Enrich a pandas DataFrame of (title, artist) with Genius metadata.
- Gentle rate limiting + retry for 429s and transient errors.
- Strictly avoids scraping or obtaining lyrics text.

Usage:
    from genius_client import GeniusClient, enrich_with_genius

    client = GeniusClient(os.environ.get("GENIUS_TOKEN"))
    df = pd.read_csv("billboard.csv")  # columns: title, artist
    df_out = enrich_with_genius(client, df)
    df_out.to_parquet("billboard_with_genius.parquet", index=False)

Notes:
- The official API does NOT provide full lyrics. Do not scrape web pages;
  that would violate Genius ToS and potentially copyright.
- This client is metadata-only (IDs, URLs, stats, etc.).
"""

from __future__ import annotations

import os
import time
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests


@dataclass
class GeniusClientConfig:
    base_url: str = "https://api.genius.com"
    user_agent: str = "academic-research/1.0 (contact: you@example.com)"
    timeout: int = 10
    max_retries: int = 3
    backoff: float = 1.5
    per_request_sleep: float = 0.0  # set e.g. to 0.25 if you batch many requests


class GeniusClient:
    def __init__(self, access_token: Optional[str], config: Optional[GeniusClientConfig] = None):
        if not access_token:
            raise ValueError("Genius access token is required. Set GENIUS_TOKEN env var or pass explicitly.")
        self.token = access_token
        self.config = config or GeniusClientConfig()
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.token}",
            "User-Agent": self.config.user_agent
        })

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.config.base_url.rstrip('/')}/{path.lstrip('/')}"
        for i in range(self.config.max_retries):
            r = self.session.get(url, params=params, timeout=self.config.timeout)
            if r.status_code == 429:
                # Rate limited, exponential backoff
                time.sleep((self.config.backoff ** i))
                continue
            # Raise for other error codes
            try:
                r.raise_for_status()
            except requests.HTTPError as e:
                # If transient server error (5xx), backoff and retry
                if 500 <= r.status_code < 600 and i < self.config.max_retries - 1:
                    time.sleep((self.config.backoff ** i))
                    continue
                raise e
            data = r.json()
            if self.config.per_request_sleep > 0:
                time.sleep(self.config.per_request_sleep)
            return data
        # If we exhausted retries on 429s, raise
        r.raise_for_status()  # type: ignore

    def search(self, q: str, per_page: int = 5) -> Dict[str, Any]:
        return self._get("/search", {"q": q, "per_page": per_page})

    def song(self, song_id: int) -> Dict[str, Any]:
        return self._get(f"/songs/{song_id}")

    @staticmethod
    def _normalize(s: str) -> str:
        s2 = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
        s2 = re.sub(r"[^\w\s]", " ", s2.lower())
        return re.sub(r"\s+", " ", s2).strip()

    @staticmethod
    def pick_best_hit(title: str, artist: str, hits: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], float]:
        """Pick the best 'hit' result by fuzzy similarity of normalized title+artist"""
        target = f"{GeniusClient._normalize(title)} {GeniusClient._normalize(artist)}"
        best = None
        best_score = 0.0
        for h in hits:
            res = h.get("result", {})
            cand_title = res.get("title", "")
            cand_artist = (res.get("primary_artist") or {}).get("name", "")
            cand = f"{GeniusClient._normalize(cand_title)} {GeniusClient._normalize(cand_artist)}"
            score = difflib.SequenceMatcher(None, target, cand).ratio()
            if score > best_score:
                best, best_score = res, score
        return best, best_score

    def resolve_song(self, title: str, artist: str, per_page: int = 5, min_score: float = 0.60) -> Optional[Tuple[Dict[str, Any], float]]:
        query = f"{artist} {title}"
        data = self.search(query, per_page=per_page)
        hits = (data.get("response") or {}).get("hits", [])
        if not hits:
            return None
        best, score = self.pick_best_hit(title, artist, hits)
        if best and score >= min_score:
            return best, score
        return None


# --- Feature helpers (language hints without lyrics) -----

SPANISH_CHARS = set("áéíóúüñ¿¡")
SPANISH_TOKENS = {"qué","cómo","por qué","amor","corazón","hasta","contigo","bailar","mañana","niña","niño","señor","señora",
                  "tú","tu","te","mi","con","sin","para","porque","cuando","dónde","sólo","solo","más","mas"}
TRANSLATION_HINTS = ("spanish-translation","traduccion","traducci%C3%B3n","en-espa%C3%B1ol","letra")

def spanish_surface_hints(title: str, path: Optional[str]) -> Dict[str, bool]:
    """Heuristics over title + Genius path. Weak signals; do NOT use as ground-truth."""
    title_lc = (title or "").lower()
    has_spanish_char = any(c in SPANISH_CHARS for c in title_lc)
    has_spanish_token = any(tok in title_lc for tok in SPANISH_TOKENS)
    has_translation_page = bool(path and any(tag in path.lower() for tag in TRANSLATION_HINTS))
    return {
        "title_has_spanish_char": has_spanish_char,
        "title_has_spanish_word": has_spanish_token,
        "genius_has_spanish_translation_hint": has_translation_page
    }


# --- Pandas enrichment -----

def enrich_with_genius(client: GeniusClient, df, title_col: str = "title", artist_col: str = "artist",
                       sleep: float = 0.25):
    """
    For each (title, artist) row, resolve to a Genius song and append metadata columns.

    Args:
        client: GeniusClient
        df: input dataframe
        title_col: column name in df with song title
        artist_col: column name in df with artist
        sleep: seconds to sleep between API calls

    Returns:
        DataFrame with original cols + Genius metadata + hints
    """
    import pandas as pd, time
    from typing import List, Dict, Any

    out_rows: List[Dict[str, Any]] = []
    it = df[[title_col, artist_col]].itertuples(index=False, name=None)

    for title, artist in it:
        try:
            match = client.resolve_song(title, artist)
            if match:
                res, score = match
                song_id = res["id"]
                song = (client.song(song_id) or {}).get("response", {}).get("song", {})
                stats = song.get("stats") or {}
                hints = spanish_surface_hints(song.get("title", ""), song.get("path"))
                out_rows.append({
                    title_col: title,
                    artist_col: artist,
                    "genius_id": song.get("id"),
                    "genius_url": song.get("url"),
                    "genius_path": song.get("path"),
                    "genius_full_title": song.get("full_title"),
                    "genius_primary_artist": (song.get("primary_artist") or {}).get("name"),
                    "genius_pageviews": stats.get("pageviews"),
                    "genius_pyongs": stats.get("pyongs_count"),
                    "genius_annotation_count": song.get("annotation_count"),
                    "genius_lyrics_state": song.get("lyrics_state"),
                    "match_score": score,
                    **hints
                })
            else:
                out_rows.append({
                    title_col: title,
                    artist_col: artist,
                    "genius_id": None,
                    "match_score": 0.0,
                    "genius_url": None,
                    "genius_path": None,
                    "genius_full_title": None,
                    "genius_primary_artist": None,
                    "genius_pageviews": None,
                    "genius_pyongs": None,
                    "genius_annotation_count": None,
                    "genius_lyrics_state": None,
                    "title_has_spanish_char": False,
                    "title_has_spanish_word": False,
                    "genius_has_spanish_translation_hint": False,
                })
        except Exception as e:
            out_rows.append({
                title_col: title,
                artist_col: artist,
                "error": str(e)
            })
        if sleep > 0:
            time.sleep(sleep)

    return pd.DataFrame(out_rows)



# --- Optional provider hook: lyrics language without storing lyrics -----

class LyricsLanguageProvider:
    """
    Interface for plugging in a licensed provider that returns ONLY language code
    (e.g., 'es', 'en') for a (title, artist) or track ID, without returning lyric text.
    Implementations must respect provider ToS and licensing.
    """
    def detect_language(self, title: str, artist: str) -> Optional[str]:
        raise NotImplementedError


def attach_lyrics_language(df, provider: LyricsLanguageProvider,
                           title_col: str = "title", artist_col: str = "artist",
                           out_col: str = "lyrics_language"):
    """
    Adds a column with ISO language code as reported by a licensed provider.
    No lyrics are fetched or stored.
    """
    langs: List[Optional[str]] = []
    for t, a in df[[title_col, artist_col]].itertuples(index=False, name=None):
        try:
            langs.append(provider.detect_language(t, a))
        except Exception:
            langs.append(None)
    df[out_col] = langs
    return df
