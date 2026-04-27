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
            home = normalize_player_name(event.get("home_team", ""))
            away = normalize_player_name(event.get("away_team", ""))
            if not home or not away:
                continue
            bks  = event.get("bookmakers", [])
            pair = _extract_pinnacle_odds(bks, home, away) or _extract_avg_odds(bks, home, away)
            if pair is None:
                continue
            odds_dict[f"{home} vs {away}"] = pair

    result = OddsResult(
        odds=odds_dict,
        fetched_at=datetime.now().isoformat(timespec="seconds"),
    )
    _save_cache(tour, today, result)
    return result


def _name_tokens(name: str) -> frozenset:
    """
    Tokenize a normalized player name into a frozenset of words.
    Hyphens are treated as spaces so "Elena-Gabriela" → {"elena", "gabriela"}.
    This makes matching order-independent and hyphen-insensitive.
    """
    return frozenset(normalize_player_name(name).replace("-", " ").split())


def merge_odds(matches: list, odds: dict) -> list:
    """
    Enriches each match dict with odd_p1 / odd_p2 from the odds dict.

    Four-level lookup (most to least strict):
      1. Exact key "p1 vs p2"
      2. Reversed key "p2 vs p1"
      3. Token-set match — handles word-order differences (e.g. "Shuai Zhang" ↔
         "Zhang Shuai") and hyphenated names ("Elena-Gabriela Ruse")
      4. Last-name pair match — fallback for nicknames ("Caty" ↔ "Catherine")

    Sets odd_p1=None, odd_p2=None for unmatched matches.
    Returns a new list — does not mutate input dicts.
    """
    # Pre-build token-set index and last-name index from odds dict
    token_index: dict[tuple, tuple] = {}    # (frozenset, frozenset) → (odd1, odd2)
    lastname_index: dict[tuple, tuple] = {} # (last1, last2) → (odd1, odd2) — for nicknames

    for key, pair in odds.items():
        parts = key.split(" vs ", 1)
        if len(parts) != 2:
            continue
        t1 = _name_tokens(parts[0])
        t2 = _name_tokens(parts[1])
        o1, o2 = float(pair[0]), float(pair[1])
        l1 = parts[0].split()[-1] if parts[0].split() else ""  # last word = surname
        l2 = parts[1].split()[-1] if parts[1].split() else ""
        # Store both orientations
        for ta, tb, oa, ob, la, lb in (
            (t1, t2, o1, o2, l1, l2),
            (t2, t1, o2, o1, l2, l1),
        ):
            token_index[(ta, tb)] = (oa, ob)
            if la and lb:
                lastname_index.setdefault((la, lb), (oa, ob))

    enriched = []
    for m in matches:
        m2 = dict(m)
        p1 = normalize_player_name(m.get("p1_name", ""))
        p2 = normalize_player_name(m.get("p2_name", ""))
        fwd = f"{p1} vs {p2}"
        rev = f"{p2} vs {p1}"

        if fwd in odds:
            pair = odds[fwd]
            m2["odd_p1"] = float(pair[0])
            m2["odd_p2"] = float(pair[1])
        elif rev in odds:
            pair = odds[rev]           # API home/away reversed
            m2["odd_p1"] = float(pair[1])
            m2["odd_p2"] = float(pair[0])
        else:
            # Level 3: token-set match (word order + hyphens)
            t1 = _name_tokens(p1)
            t2 = _name_tokens(p2)
            pair = token_index.get((t1, t2))
            if pair:
                m2["odd_p1"] = pair[0]
                m2["odd_p2"] = pair[1]
            else:
                # Level 4: last-name pair fallback (nickname tolerance)
                l1 = p1.split()[-1] if p1 else ""
                l2 = p2.split()[-1] if p2 else ""
                pair = lastname_index.get((l1, l2)) if l1 and l2 else None
                if pair:
                    m2["odd_p1"] = pair[0]
                    m2["odd_p2"] = pair[1]
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


def _match_outcome_to_player(outcome_name: str, player_key: str) -> bool:
    """
    Flexible player name matching between Pinnacle outcome names and event player keys.
    Handles: full name, last name only, "F. Lastname", "Lastname F." (API format).
    Both inputs are already normalized (lowercase, no accents).
    """
    on = normalize_player_name(outcome_name)
    if not on:
        return False
    # Exact match
    if on == player_key:
        return True
    # Last name only (e.g. outcome "lehecka" vs key "jiri lehecka")
    player_last = player_key.split()[-1]
    if on == player_last:
        return True
    # "F. Lastname" format — outcome ends with player last name
    if on.endswith(player_last) and len(on) > len(player_last):
        return True
    # Key ends with outcome last name
    outcome_last = on.split()[-1] if on else ""
    if outcome_last and len(outcome_last) > 1 and player_key.endswith(outcome_last):
        return True
    # "Lastname F." format — first word of outcome matches a word in player key
    outcome_first = on.split()[0] if on else ""
    if outcome_first and len(outcome_first) > 1 and outcome_first in player_key.split():
        return True
    return False


def _pair_outcomes(outcomes: list, home_key: str, away_key: str) -> tuple | None:
    """Match two outcomes to home/away player and return (home_price, away_price)."""
    if len(outcomes) < 2:
        return None
    prices = [(normalize_player_name(o["name"]), float(o["price"])) for o in outcomes]
    home_price = away_price = None
    for name, price in prices:
        if _match_outcome_to_player(name, home_key):
            home_price = price
        elif _match_outcome_to_player(name, away_key):
            away_price = price
    if home_price is not None and away_price is not None:
        return (home_price, away_price)
    # Name matching failed — return None rather than silently assigning wrong odds
    log.warning("Outcome name mismatch for %s / %s — skipping", home_key, away_key)
    return None


def _extract_pinnacle_odds(bookmakers: list, home_key: str, away_key: str) -> tuple | None:
    """Find Pinnacle h2h odds. Returns (home_odd, away_odd) matched by player name."""
    for bk in bookmakers:
        if bk.get("key") == "pinnacle":
            for market in bk.get("markets", []):
                if market.get("key") == "h2h":
                    return _pair_outcomes(market.get("outcomes", []), home_key, away_key)
    return None


def _extract_avg_odds(bookmakers: list, home_key: str, away_key: str) -> tuple | None:
    """Average h2h odds across all bookmakers, matched by player name."""
    home_prices, away_prices = [], []
    for bk in bookmakers:
        for market in bk.get("markets", []):
            if market.get("key") == "h2h":
                pair = _pair_outcomes(market.get("outcomes", []), home_key, away_key)
                if pair:
                    home_prices.append(pair[0])
                    away_prices.append(pair[1])
    if not home_prices:
        return None
    return (sum(home_prices) / len(home_prices), sum(away_prices) / len(away_prices))
