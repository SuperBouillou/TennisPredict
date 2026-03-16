# src/odds_api_client.py
"""
Odds API client — fetch + cache + normalize today's ATP tennis odds.

One API call per day, cached to:
    data/odds_cache/{tour}/odds_YYYY-MM-DD.json

Usage:
    from odds_api_client import fetch_odds_today, merge_odds
    result  = fetch_odds_today("atp")
    matches = merge_odds(matches, result.odds)

Requires ODDS_API_KEY in .env or environment variable.
"""

import os
import json
import unicodedata
import logging
import warnings

warnings.filterwarnings("ignore")

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()  # no-op if .env absent; also picks up system env vars

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
_API_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds"

# Pre-map characters that NFD cannot decompose to ASCII equivalents.
# Built once at module load, reused by every normalize_player_name() call.
_STROKE_MAP = str.maketrans({
    "\u0110": "D",  # Đ  LATIN CAPITAL LETTER D WITH STROKE
    "\u0111": "d",  # đ  LATIN SMALL LETTER D WITH STROKE
    "\u0141": "L",  # Ł  LATIN CAPITAL LETTER L WITH STROKE
    "\u0142": "l",  # ł  LATIN SMALL LETTER L WITH STROKE
    "\u00d8": "O",  # Ø  LATIN CAPITAL LETTER O WITH STROKE
    "\u00f8": "o",  # ø  LATIN SMALL LETTER O WITH STROKE
    "\u0166": "T",  # Ŧ  LATIN CAPITAL LETTER T WITH STROKE
    "\u0167": "t",  # ŧ  LATIN SMALL LETTER T WITH STROKE
})


# ─────────────────────────────────────────────────────────────────────────────
# Data Types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OddsResult:
    """Return type for fetch_odds_today()."""
    odds: dict = field(default_factory=dict)  # normalized_key -> (odd_p1, odd_p2)
    fetched_at: str | None = None             # ISO timestamp or None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def normalize_player_name(name: str) -> str:
    """
    Public. Lowercase + strip accents.
    Example: 'Björn Borg' -> 'bjorn borg', 'SINNER' -> 'sinner'
    Used by both cache construction (API names) and merge (ESPN names).

    Characters that do not decompose via NFD (e.g. Đ/đ — D-with-stroke)
    are mapped explicitly before NFD normalization.
    """
    if not name:
        return ""
    name = name.translate(_STROKE_MAP)
    nfkd = unicodedata.normalize("NFD", name)
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    return ascii_str.strip().lower()


def fetch_odds_today(tour: str) -> OddsResult:
    """
    Fetch today's ATP odds. Returns OddsResult (never raises).

    Returns empty OddsResult if:
      - tour != 'atp'
      - ODDS_API_KEY missing or empty
      - API request fails
      - No tennis matches found

    NOT decorated with @st.cache_data — file cache is the only cache layer.
    Deleting the cache file (Rafraîchir button) forces a real API call.
    """
    if tour != "atp":
        return OddsResult()

    api_key = os.environ.get("ODDS_API_KEY", "").strip()
    if not api_key:
        return OddsResult()

    today = date.today()
    cached = _load_cache(tour, today)
    if cached is not None:
        return cached

    # Step 1: discover all active tennis_{tour}_* sport keys
    try:
        sports_resp = requests.get(
            "https://api.the-odds-api.com/v4/sports",
            params={"apiKey": api_key}, timeout=10,
        )
        sports_resp.raise_for_status()
        sport_keys = [
            s["key"] for s in sports_resp.json()
            if s.get("key", "").startswith(f"tennis_{tour}") and s.get("active")
        ]
    except Exception as exc:
        log.warning("Odds API sports list failed: %s", exc)
        return OddsResult()

    if not sport_keys:
        log.info("No active tennis_%s tournaments found in Odds API", tour)
        return OddsResult()

    # Step 2: fetch odds for each active tournament and aggregate
    odds_dict = {}
    for sport in sport_keys:
        try:
            url  = _API_URL.format(sport=sport)
            resp = requests.get(url, params={
                "apiKey"     : api_key,
                "regions"    : "eu",
                "markets"    : "h2h",
                "oddsFormat" : "decimal",
                "dateFormat" : "iso",
            }, timeout=10)
            resp.raise_for_status()
            events = resp.json()
        except Exception as exc:
            log.warning("Odds API request failed for %s: %s", sport, exc)
            continue

        for event in events:
            bks  = event.get("bookmakers", [])
            pair = _extract_pinnacle_odds(bks) or _extract_avg_odds(bks)
            if pair is None:
                continue
            home = normalize_player_name(event.get("home_team", ""))
            away = normalize_player_name(event.get("away_team", ""))
            if home and away:
                odds_dict[f"{home} vs {away}"] = pair

    result = OddsResult(
        odds=odds_dict,
        fetched_at=datetime.now().isoformat(timespec="seconds"),
    )
    _save_cache(tour, today, result)
    return result


def merge_odds(matches: list, odds: dict) -> list:
    """
    Enriches each match dict with odd_p1 / odd_p2 from the odds dict.

    Normalization applied to BOTH sides:
      - Keys in `odds` are normalized at cache-write time (API names)
      - ESPN p1_name / p2_name are normalized here at lookup time

    Tries "p1 vs p2" AND "p2 vs p1" to handle reversed API order.
    Sets odd_p1=None, odd_p2=None for unmatched matches.
    Returns a new list — does not mutate input dicts.
    """
    enriched = []
    for m in matches:
        m2  = dict(m)
        p1  = normalize_player_name(m.get("p1_name", ""))
        p2  = normalize_player_name(m.get("p2_name", ""))
        fwd = f"{p1} vs {p2}"
        rev = f"{p2} vs {p1}"

        if fwd in odds:
            pair         = odds[fwd]
            m2["odd_p1"] = float(pair[0])
            m2["odd_p2"] = float(pair[1])
        elif rev in odds:
            pair         = odds[rev]          # API home/away reversed
            m2["odd_p1"] = float(pair[1])     # swap: API p2 is our p1
            m2["odd_p2"] = float(pair[0])
        else:
            m2["odd_p1"] = None
            m2["odd_p2"] = None

        enriched.append(m2)
    return enriched


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(tour: str, today: date) -> Path:
    p = ROOT / "data" / "odds_cache" / tour
    p.mkdir(parents=True, exist_ok=True)
    return p / f"odds_{today.isoformat()}.json"


def _load_cache(tour: str, today: date) -> OddsResult | None:
    """
    Returns None if file missing OR corrupted.
    fetch_odds_today() treats None as a cache miss and re-fetches in both cases.
    """
    path = _cache_path(tour, today)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return OddsResult(
            odds={k: tuple(v) for k, v in data.get("odds", {}).items()},
            fetched_at=data.get("fetched_at"),
        )
    except Exception:
        log.warning("Cache file corrupted: %s — will re-fetch", path)
        return None


def _save_cache(tour: str, today: date, result: OddsResult) -> None:
    """Write OddsResult to data/odds_cache/{tour}/odds_YYYY-MM-DD.json."""
    path    = _cache_path(tour, today)
    payload = {
        "fetched_at" : result.fetched_at,
        "tour"       : tour,
        "odds"       : {k: list(v) for k, v in result.odds.items()},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _extract_pinnacle_odds(bookmakers: list) -> tuple | None:
    """Find Pinnacle h2h odds. Returns (odd_p1, odd_p2) or None if absent."""
    for bk in bookmakers:
        if bk.get("key") == "pinnacle":
            for market in bk.get("markets", []):
                if market.get("key") == "h2h":
                    outcomes = market.get("outcomes", [])
                    if len(outcomes) >= 2:
                        return (float(outcomes[0]["price"]), float(outcomes[1]["price"]))
    return None


def _extract_avg_odds(bookmakers: list) -> tuple | None:
    """Average h2h odds across all bookmakers. Returns None if no data."""
    p1_prices, p2_prices = [], []
    for bk in bookmakers:
        for market in bk.get("markets", []):
            if market.get("key") == "h2h":
                outcomes = market.get("outcomes", [])
                if len(outcomes) >= 2:
                    p1_prices.append(float(outcomes[0]["price"]))
                    p2_prices.append(float(outcomes[1]["price"]))
    if not p1_prices:
        return None
    return (sum(p1_prices) / len(p1_prices), sum(p2_prices) / len(p2_prices))
