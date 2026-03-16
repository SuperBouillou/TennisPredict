# Odds API Integration — Design Spec
**Date:** 2026-03-15
**Status:** Approved

---

## Goal

Automatically fetch today's ATP tennis match odds from The Odds API (one request per day maximum), cache them locally, and enrich the dashboard's Today tab so that:
1. Each match card shows a GO/PASS signal + edge/EV without any manual input
2. The "+Pari" popover has odds pre-filled (still editable)

---

## Architecture

```
Dashboard load
    │
    ├─ ESPN API → matchs du jour (p1_name, p2_name, tournament, surface...)
    │
    └─ odds_api_client.fetch_odds_today(tour)
           │
           ├─ tour != "atp" → return OddsResult(odds={}, fetched_at=None)
           │
           ├─ Cache hit? data/odds_cache/{tour}/odds_YYYY-MM-DD.json
           │      └─ YES → charge JSON, retourne OddsResult
           │
           └─ NO → GET https://api.the-odds-api.com/v4/sports/tennis_{tour}/odds
                      └─ sauvegarde JSON → retourne OddsResult

odds_api_client.merge_odds(matches_espn, odds_result.odds)
    │   matching via normalize_player_name() (public, shared)
    │   cherche "p1 vs p2" ET "p2 vs p1"
    │   non-matché → odd_p1=None, odd_p2=None
    │
    └─ matches enrichis : chaque match a optional (odd_p1, odd_p2)

render_today_tab()
    ├─ Status bar : "● X matchs enrichis · mis à jour HH:MM  [↻ Rafraîchir]"
    ├─ Carte match : GO/PASS + edge/EV si odds disponibles, badge gris sinon
    └─ Popover "+Pari" : number_input value=odd_p1 si disponible, sinon 2.0
```

---

## New File: `src/odds_api_client.py`

### API Endpoint

```
GET https://api.the-odds-api.com/v4/sports/tennis_{tour}/odds
    ?apiKey={ODDS_API_KEY}
    &regions=eu
    &markets=h2h
    &oddsFormat=decimal
    &dateFormat=iso
```

Sport key is `tennis_atp` for ATP. WTA would use `tennis_wta` but is out of scope for now.

### Return Type

```python
from dataclasses import dataclass

@dataclass
class OddsResult:
    odds: dict[str, tuple[float, float]]  # normalized_key -> (odd_p1, odd_p2)
    fetched_at: str | None                # ISO timestamp string, None if no data
```

### Public Interface

```python
def fetch_odds_today(tour: str) -> OddsResult:
    """
    Fetches today's odds for the given tour (ATP only for now).
    Returns OddsResult with empty odds dict if:
      - tour != "atp"
      - ODDS_API_KEY missing
      - API request fails
      - No tennis matches found
    Never raises — always returns a valid OddsResult.
    NOT decorated with @st.cache_data (file cache is the only cache layer).
    """

def merge_odds(matches: list[dict], odds: dict) -> list[dict]:
    """
    Enriches each match dict with odd_p1 / odd_p2 from odds dict.
    Normalization is applied to BOTH sides:
      - Keys in `odds` are already normalized at cache-write time (API names → normalize_player_name)
      - ESPN p1_name / p2_name are normalized at lookup time via normalize_player_name()
    Lookup key: f"{normalize(p1)} vs {normalize(p2)}", also tries reversed order.
    Adds odd_p1=None, odd_p2=None for unmatched matches.
    Returns a new list (does not mutate input).
    """

def normalize_player_name(name: str) -> str:
    """
    Public. Lowercase + strip accents via unicodedata.normalize('NFD').
    Used by both odds_api_client and dashboard for consistent matching.
    Example: "Björn Borg" -> "bjorn borg"
    """
```

### Internal Functions

```python
def _load_cache(tour: str, today: date) -> OddsResult | None:
    """
    Read data/odds_cache/{tour}/odds_YYYY-MM-DD.json.
    Returns None if file missing OR if JSON is corrupted.
    fetch_odds_today() treats None as "cache miss" and re-fetches from API in both cases.
    """

def _save_cache(tour: str, today: date, result: OddsResult) -> None:
    """Write OddsResult to data/odds_cache/{tour}/odds_YYYY-MM-DD.json."""

def _extract_pinnacle_odds(bookmakers: list) -> tuple[float, float] | None:
    """Extract h2h (odd_p1, odd_p2) from Pinnacle bookmaker entry. None if absent."""

def _extract_avg_odds(bookmakers: list) -> tuple[float, float] | None:
    """Average h2h odds across all bookmakers. None if no bookmakers."""
```

### Bookmaker Priority

Pinnacle first (sharp market, same source as backtests). Fallback to average across all available bookmakers if Pinnacle absent.

### API Key Loading

```python
import os
from dotenv import load_dotenv

load_dotenv()  # no-op if .env absent
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
```

`load_dotenv()` is a no-op if `.env` doesn't exist — the function checks system environment variables first, which allows CI/Docker usage without a `.env` file.

Developer must run `pip install -r requirements.txt` after `python-dotenv` is added.

### Cache Format

Path: `data/odds_cache/{tour}/odds_YYYY-MM-DD.json`

```json
{
  "fetched_at": "2026-03-15T09:32:00",
  "tour": "atp",
  "odds": {
    "jannik sinner vs carlos alcaraz": [2.10, 1.75],
    "novak djokovic vs rafael nadal": [1.85, 2.05]
  }
}
```

One file per tour per day. Cache is valid for the entire day — the API is not called again unless the user explicitly refreshes (which deletes the file).

---

## Modified File: `src/dashboard.py`

### 1. Import `fetch_odds_today`, `merge_odds` from `odds_api_client`

Add to imports at top of `dashboard.py`:
```python
from odds_api_client import fetch_odds_today, merge_odds
```

### 2. `render_today_tab()` — odds fetch + merge + thresholds

At the start of `render_today_tab()`, before the per-match loop (alongside the existing ESPN fetch):
```python
odds_result = fetch_odds_today(tour)         # NOT @st.cache_data — file cache only
matches     = merge_odds(matches, odds_result.odds)
thresholds  = load_optimal_thresholds(tour)  # called once here, same as predictions tab
opt_edge    = thresholds.get("best_roi", {}).get("min_edge", 0.03)
opt_prob    = thresholds.get("best_roi", {}).get("min_prob", 0.55)
```
`opt_edge` and `opt_prob` are then used inside the per-match loop for GO/PASS evaluation.

**`fetch_odds_today` must NOT be wrapped in `@st.cache_data`.** The file-based cache is the only cache layer. Wrapping it in Streamlit cache would prevent the "Rafraîchir" button from working.

### 3. Status bar

Displayed above the match list when `odds_result.fetched_at` is not None:
```
● Cotes Odds API — 14 matchs enrichis · mis à jour 09:32  [↻ Rafraîchir]
```
- Count = number of matches where `odd_p1 is not None`
- Time = `odds_result.fetched_at` formatted as HH:MM
- "Rafraîchir" button: deletes `data/odds_cache/{tour}/odds_YYYY-MM-DD.json`, calls `st.rerun()`

### 4. Match card

**When odds available** (`odd_p1 is not None`):
- Compute `edge1, ev1, imp1 = compute_edge(prob_p1, odd_p1, odd_p2)`
- Display GO/PASS badge using `opt_edge`/`opt_prob` from `load_optimal_thresholds(tour)`
- Same badge HTML styling as predictions tab

**When odds missing** (`odd_p1 is None`):
- Display grey "Cotes manquantes" badge
- No edge/EV shown

### 5. Popover "+Pari" — pre-filled odds

```python
o1 = st.number_input(
    f"Cote {m['p1_name'].split()[0]}",
    min_value=1.01,
    value=float(m.get('odd_p1') or 2.0),
    step=0.05
)
o2 = st.number_input(
    f"Cote {m['p2_name'].split()[0]}",
    min_value=1.01,
    value=float(m.get('odd_p2') or 2.0),
    step=0.05
)
```

User can still edit freely. Edge/EV table updates live as before.

---

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `src/odds_api_client.py` | **Create** | Fetch + cache + normalize + merge odds |
| `src/dashboard.py` | **Modify** | Import module, fetch on load, status bar, card GO/PASS, popover pre-fill |
| `.env` | **Create** (user) | Store `ODDS_API_KEY=...` |
| `.gitignore` | **Create or modify** | Add `.env`, `data/odds_cache/`, `venv/`, `__pycache__/` |
| `requirements.txt` | **Modify** | Add `python-dotenv>=1.0.0` |
| `data/odds_cache/{tour}/` | **Auto-created** | Daily JSON cache files |

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| `tour != "atp"` | Return `OddsResult({}, None)` immediately, no API call |
| API key missing | Return `OddsResult({}, None)`, no crash |
| API request fails (timeout/500) | Log warning, return `OddsResult({}, None)` |
| No tennis matches in API response | Return `OddsResult({}, None)` |
| Player name not matched | `odd_p1=None`, `odd_p2=None` — manual entry as before |
| Cache file corrupted (bad JSON) | Catch parse error, re-fetch from API |
| Both Pinnacle and avg odds absent | Skip that match entry in the odds dict |

---

## Out of Scope

- WTA odds (guard in `fetch_odds_today` returns empty immediately for `tour != "atp"`)
- Historical odds storage (cache is today-only)
- Automatic scheduled fetch (fetch happens on dashboard load)
