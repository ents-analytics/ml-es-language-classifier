# config.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict

from dotenv import load_dotenv

def _load_env(dotenv_path: Optional[str] = None) -> None:
    """
    Load .env. If no path is provided, search upwards from CWD for a '.env' file.
    """
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)
        return
    # Walk up from CWD to find a .env
    here = Path.cwd()
    for p in [here, *here.parents]:
        env_file = p / ".env"
        if env_file.exists():
            load_dotenv(str(env_file), override=False)
            return
    load_dotenv(override=False)  # fallback (no-op if none found)

def get_genius_token(dotenv_path: Optional[str] = None) -> Optional[str]:
    _load_env(dotenv_path)
    return os.getenv("GENIUS_TOKEN")

def get_spotify_config(dotenv_path: Optional[str] = None) -> Dict[str, Optional[str]]:
    _load_env(dotenv_path)
    return {
        "client_id": os.getenv("SPOTIFY_CLIENT_ID"),
        "client_secret": os.getenv("SPOTIFY_CLIENT_SECRET"),
        "redirect_uri": os.getenv("SPOTIFY_REDIRECT_URI"),
    }

def require_env(var_name: str) -> str:
    val = os.getenv(var_name)
    if not val:
        raise RuntimeError(
            f"Missing required environment variable: {var_name}. "
            "Set it in your .env or shell env."
        )
    return val
