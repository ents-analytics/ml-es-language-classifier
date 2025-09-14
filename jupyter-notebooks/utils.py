
# utils.py

import os
import time
from typing import Iterable, List, Tuple, Dict, Any, Optional

import json
import hashlib
from pathlib import Path

from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException


# --- Auth -----

def get_spotify_client() -> spotipy.Spotify:
	""" 
	Load credentials from .env file and returns an authenticated spotipy client
	Uses client credentials (no user data): spotipy auto-refreshes the token
	"""

	load_dotenv() # loads credential variables from .env in os.environ
	client_id = os.environ["SPOTIFY_CLIENT_ID"]
	client_secret = os.environ["SPOTIFY_CLIENT_SECRET"]

	return spotipy.Spotify(
		auth_manager=SpotifyClientCredentials(
			client_id = client_id,
			client_secret = client_secret,
		)
	)


# --- Search helpers -----

def search_tracks(sp, query: str, limit: int = 5) -> List[Tuple[str, str, str]]:
	"""
	Search for tracks on Spotify and return a clean list of tuples:
	(track_id, track_name, artist_name), ... tracks only

	Parameters
	----------
	sp : spotipy.Spotify
		An authenticated Spotify client
	query :str
		The search query (track/artist name, etc)
	limit : int, optional (default=5)
		Number of results to return (1-50)

	Returns
	-------
	List[Tuple[str, str, str]]
		A list of (id, name, artist) tuples for each matched track
		Returns an empty list if no results are found

	Notes
	-----
	Uses defensive JSON access to avoid KeyError if fields are missing
	"""

	results = sp.search(q=query, type="track", limit=limit)
	items = results.get("tracks", {}).get("items", []) or []
	rows: List[Tuple[str, str, str]] = []
	for item in items:
		if item.get("type") != "track": # --- this for safety
			continue
		track_id = item.get("id")
		track_name = item.get("name")
		artists = item.get("artists", [])
		artist_name = artists[0]["name"] if artists else None
		if track_id and track_name and artist_name:
			rows.append((track_id, track_name, artist_name))
	return rows


def search_track_id(sp, query: str) -> Optional[str]:
	"""
	Return the first track ID for a search query, or None if not found
	"""

	result = sp.search(q=query, type="track", limit=1)
	items = result.get("tracks", {}).get("items", [])
	return items[0]["id"] if items else None




def paginate_search_tracks(sp, query: str, max_items: int = 200) -> List[Dict[str, Any]]:
	"""
	Return up to max_items track objects by paging through search results
	"""

	items: List[Dict[str, Any]] = []
	limit = 50
	offset = 0
	while len(items) < max_items:
		result = sp.search(q=query, type = "track", limit=min(limit, max_items - len(items)), offset=offset)
		page = result.get("tracks", {}).get("items", []) or []
		if not page:
			break
		items.extend(page)
		offset += len(page)
		if len(page) < limit:
			break
	return items 


# --- Metadata and features -----


def get_track_core(sp, track_id: str) -> Dict[str, Any]:
	"""
	Return a compact dictionary with key fields for a track
	"""

	track = sp.track(track_id)
	return {
		"track_id": track["id"],
		"track_name": track["name"],
		"artist_id": track["artists"][0]["id"],
		"artist_name": track["artists"][0]["name"],
		"album_id": track["album"]["id"],
		"album_name": track["album"]["name"],
		"release_date": track["album"].get("release_date"),
		"popularity": track.get("popularity"),
		"duration_ms": track.get("duration_ms"),
		"explicit": track.get("explicit"),
	}


def get_audio_features(sp, track_ids):
    """
    Batch in chunks of 100. If a batch 403s, fall back to single-item calls
    and skip any IDs that still fail.
    """
    ids = [i for i in dict.fromkeys(track_ids) if i]  # de-dup & drop falsy
    feats = []

    for i in range(0, len(ids), 100):
        chunk = ids[i:i+100]
        try:
            got = sp.audio_features(chunk)
            feats.extend([f for f in got if f])  # filter None
        except SpotifyException as e:
            # If a whole batch fails, try one-by-one to isolate bad IDs
            if getattr(e, "http_status", None) == 403:
                for tid in chunk:
                    try:
                        got1 = sp.audio_features([tid])
                        if got1 and got1[0]:
                            feats.append(got1[0])
                    except SpotifyException as e1:
                        # still forbidden; skip this ID
                        continue
            else:
                # re-raise other errors
                raise
    return feats




def search_tracks_with_features(sp, query: str, limit: int = 10) -> List[Dict[str, Any]]:
	"""
	Return a list of dicts merging (id, name, artist) with audio features
	"""

	rows = search_tracks(sp, query, limit=limit)
	ids = [r[0] for r in rows]
	feats = {f["id"]: f for f in get_audio_features(sp, ids)}
	merged = []
	for tid, name, artist in rows:
		f = feats.get(tid, {})
		merged.append({
			"track_id": tid,
			"track_name": name,
			"artist_name": artist,
			"danceability": f.get("danceability"),
			"energy": f.get("energy"),
			"speechiness": f.get("speechiness"),
			"acousticness": f.get("acousticness"),
			"instrumentalness": f.get("instrumentalness"),
			"liveness": f.get("liveness"),
			"valence": f.get("valence"),
			"tempo": f.get("tempo"),
			"time_signature": f.get("time_signature"),
			"key": f.get("key"),
			"mode": f.get("mode"),
			"duration_ms": f.get("duration_ms"),
			})
	return merged


# --- Resilience -----


def call_with_retry(fn, *args, retries: int = 3, **kwargs):
	"""
	Call a Spotify function with basic 429 handling
	Retries if Spotify asks to back off via Retry-After header
	"""

	for attempt in range(retries + 1):
		try:
			return fn(*args, **kwargs)
		except Exception as e:
			# Spotify attaches HTTP status; check for rate limit
			msg = str(e).lower()
			if "status code: 429" in msg and attempt < retries:
				# Try to parse Retry-After seconds if present
				retry_after = 2 + attempt # default small backoff
				# If the exception exposes headers, try to read them
				h = getattr(e, "headers", {}) or {}
				retry_after = int(h.get("Retry-After", retry_after))
				time.sleep(retry_after)
				continue
			raise # re-raise other errors or after retries



def to_dataframe(rows: List[Dict[str, Any]]):
	"""
	Convert a list of dictionaries to a pandas dataframe if pandas is installed
	"""

	try:
		import pandas as pd #import to keep local utils lightweight
		return pd.DataFrame(rows)
	except Exception:
		return rows # graceful fallback: just return the list


# --- Tiny disk cache -----

# Add the cache file to .gitignore
CACHE_PATH = Path(".spotify_cache.json")

def _now() -> int:
    import time as _t
    return int(_t.time())

def _hash_key(kind: str, payload: Dict[str, Any]) -> str:
    """Stable key for (kind, payload)."""
    raw = json.dumps({"k": kind, "p": payload}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

def _load_cache() -> Dict[str, Any]:
    if CACHE_PATH.exists():
        try:
            with CACHE_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_cache(cache: Dict[str, Any]) -> None:
    try:
        with CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        # best-effort: caching should never crash your pipeline
        pass

def _cache_get(kind: str, payload: Dict[str, Any], ttl: Optional[int]) -> Optional[Any]:
    cache = _load_cache()
    key = _hash_key(kind, payload)
    entry = cache.get(key)
    if not entry:
        return None
    if ttl is not None and _now() > entry.get("created", 0) + int(ttl):
        return None  # expired
    return entry.get("value")

def _cache_set(kind: str, payload: Dict[str, Any], value: Any) -> None:
    cache = _load_cache()
    key = _hash_key(kind, payload)
    cache[key] = {"created": _now(), "value": value}
    _save_cache(cache)

def prune_cache(max_keys: int = 5000) -> int:
    """Keep only the newest `max_keys` entries. Returns number of entries removed."""
    cache = _load_cache()
    if len(cache) <= max_keys: 
        return 0
    items = sorted(cache.items(), key=lambda kv: kv[1].get("created", 0), reverse=True)
    keep = dict(items[:max_keys])
    removed = len(cache) - len(keep)
    _save_cache(keep)
    return removed


# --- Cached wrappers -----

def search_tracks_cached(sp, query: str, limit: int = 5, ttl: int = 24 * 3600):
    """
    Cached version of search_tracks(sp, ...). Default TTL = 24h.
    Pass ttl=0 to bypass reads (still writes).
    """

    kind = "search_tracks"
    payload = {"q": query, "limit": int(limit)}
    hit = _cache_get(kind, payload, ttl)
    if hit is not None:
        return hit
    rows = search_tracks(sp, query, limit=limit)  # your existing helper
    _cache_set(kind, payload, rows)
    return rows

def get_track_core_cached(sp, track_id: str, ttl: int = 30 * 24 * 3600):
    """Cached compact track metadata. Default TTL = 30 days."""

    kind = "track_core"
    payload = {"id": track_id}
    hit = _cache_get(kind, payload, ttl)
    if hit is not None:
        return hit
    data = get_track_core(sp, track_id)  # your existing helper
    _cache_set(kind, payload, data)
    return data

def get_audio_features_cached(sp, track_ids: Iterable[str], ttl: int = 7 * 24 * 3600):
    """
    Cached audio features. Default TTL = 7 days.
    Only requests the IDs missing/expired in cache, then merges results.
    """

    ids = list(dict.fromkeys([i for i in track_ids if i]))  # de-dup & drop falsy
    got: Dict[str, Dict[str, Any]] = {}
    missing: List[str] = []

    # 1) Try cache
    for tid in ids:
        hit = _cache_get("audio_features", {"id": tid}, ttl)
        if hit is not None:
            got[tid] = hit
        else:
            missing.append(tid)

    # 2) Fetch only the missing (uses your existing get_audio_features)
    if missing:
        fresh = get_audio_features(sp, missing)
        for f in fresh:
            if not f:
                continue
            tid = f.get("id")
            if tid:
                _cache_set("audio_features", {"id": tid}, f)
                got[tid] = f

    # 3) Return aligned to original order
    return [got[tid] for tid in ids if tid in got]




# --- Validation: check which track IDs are usable -----

def validate_track_ids(
    sp,
    track_ids: Iterable[str],
    *,
    retry: bool = True,
    quiet: bool = True,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Validate a list of Spotify track IDs by calling audio_features() one-by-one.

    Returns
    -------
    ok_ids : List[str]
        IDs that returned a valid audio-features record.
    report : List[Dict[str, Any]]
        Per-ID diagnostics with keys: id, ok (bool), status (Optional[int]),
        reason (Optional[str]), msg (Optional[str]).

    Notes
    -----
    - One-by-one calls are slower but isolate problem IDs precisely.
    - Set retry=True to re-attempt transient errors once.
    - This only checks the audio-features endpoint; if your pipeline
      depends on other endpoints, consider similar validators for those.
    """

    seen = set()
    ok_ids: List[str] = []
    report: List[Dict[str, Any]] = []

    ids = [i for i in track_ids if i]  # drop falsy
    for tid in ids:
        if tid in seen:
            continue
        seen.add(tid)

        def _try_once() -> Tuple[bool, Optional[int], Optional[str], Optional[str]]:
            try:
                res = sp.audio_features([tid])
                if res and res[0]:
                    return True, 200, None, None
                # Returned None (e.g., invalid/unavailable track)
                return False, 200, "NoFeatures", "audio_features returned None"
            except SpotifyException as e:
                status = getattr(e, "http_status", None)
                return False, status, "SpotifyException", str(e)
            except Exception as e:
                return False, None, "Exception", str(e)

        ok, status, reason, msg = _try_once()

        # Optional single retry for transient issues
        if not ok and retry and (status in (429, 500, 502, 503, 504) or reason in ("Exception",)):
            ok2, status2, reason2, msg2 = _try_once()
            ok, status, reason, msg = ok2, status2, reason2, msg2

        if ok:
            ok_ids.append(tid)
            if not quiet:
                print(f"[OK] {tid}")
            report.append({"id": tid, "ok": True, "status": status, "reason": None, "msg": None})
        else:
            if not quiet:
                print(f"[BAD] {tid} status={status} reason={reason} msg={msg}")
            report.append({"id": tid, "ok": False, "status": status, "reason": reason, "msg": msg})

    return ok_ids, report


def summarise_id_report(report: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarise the output of validate_track_ids() for quick inspection.
    """
    total = len(report)
    bad = sum(1 for r in report if not r["ok"])
    by_status: Dict[str, int] = {}
    for r in report:
        key = str(r.get("status"))
        by_status[key] = by_status.get(key, 0) + 1
    return {"total_checked": total, "bad": bad, "ok": total - bad, "by_status": by_status}


# --- Module exports (public API) -----
# Added so that only these functions are imported with a import * wildcard

__all__ = [
    "get_spotify_client",
    "search_tracks",
    "search_track_id",
    "search_tracks_cached",
    "get_track_core",
    "get_track_core_cached",
    "get_audio_features",
    "get_audio_features_cached",
    "validate_track_ids",
    "summarise_id_report",
    "prune_cache",
]



"""
	ON THE USEFULNESS OF THESE FUNCTIONS

	- search_track_id / search_tracks
		No more repeating raw JSON indexing in notebooks - get clean tuples quickly.

	- paginate_search_tracks
		When you need more than 50 results for broader retrieval or candidate sets.

	- get_track_details
		Pulls the most commonly needed fields without the full payload clutter.

	- get_audio_features
		Handles batching up to 100 IDs per call (Spotify limit) and filters Nones.

	- search_tracks_with_features
		One-liner in a notebook to go from query --> IDs --> features --> tidy dictionaries.

	- call_with_retry
		Drop-in wrapper that sleeps on 429 Too Many Requests using Retry-After.

	- to_dataframe
		Quick step to land results into a DataFrame when pandas is available (but keeps utils.py import-light).
"""
















