# Odds API Integration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Auto-fetch today's ATP tennis odds from The Odds API (one call/day, file-cached), merge with ESPN match data, display GO/PASS badges on match cards, and pre-fill odds in the "+Pari" popover.

**Architecture:** `src/odds_api_client.py` owns all API/cache/merge logic; `src/dashboard.py` calls it at render time. File cache (`data/odds_cache/atp/odds_YYYY-MM-DD.json`) prevents repeated API calls. No `@st.cache_data` — file cache is the only cache layer so "Rafraîchir" button works correctly.

**Tech Stack:** `requests` (already installed), `python-dotenv>=1.0.0` (new), `unicodedata` (stdlib), `dataclasses` (stdlib), `pytest` (existing tests in `tests/`)

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/odds_api_client.py` | **Create** | OddsResult dataclass, normalize, cache, API fetch, merge |
| `tests/test_odds_api_client.py` | **Create** | Unit tests for all public + internal functions |
| `src/dashboard.py` | **Modify** | Import module, fetch on load, status bar, GO/PASS card badge, popover pre-fill |
| `requirements.txt` | **Modify** | Add `python-dotenv>=1.0.0` |
| `.gitignore` | **Create** | Add `.env`, `data/odds_cache/`, `venv/`, `__pycache__/` |

---

## Chunk 1: odds_api_client.py + tests

### Task 1: Create src/odds_api_client.py

**Files:**
- Create: `src/odds_api_client.py`

- [ ] **Step 1: Write the file**

```python
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
    """
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

    try:
        url  = _API_URL.format(sport=f"tennis_{tour}")
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
        log.warning("Odds API request failed: %s", exc)
        return OddsResult()

    odds_dict = {}
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
```

- [ ] **Step 2: Verify syntax**

```bash
cd E:/Claude/botbet/tennis/tennis_ml
venv/Scripts/python -c "import ast; ast.parse(open('src/odds_api_client.py').read()); print('OK')"
```

Expected: `OK`

---

### Task 2: Create tests/test_odds_api_client.py

**Files:**
- Create: `tests/test_odds_api_client.py`

- [ ] **Step 1: Write failing tests first**

```python
# tests/test_odds_api_client.py
"""
Unit tests for src/odds_api_client.py
Run: cd E:/Claude/botbet/tennis/tennis_ml && venv/Scripts/pytest tests/test_odds_api_client.py -v
"""
import json
import sys
from pathlib import Path
from datetime import date

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from odds_api_client import (
    normalize_player_name,
    merge_odds,
    _extract_pinnacle_odds,
    _extract_avg_odds,
    _load_cache,
    _save_cache,
    OddsResult,
)


# ── normalize_player_name ─────────────────────────────────────────────────────

def test_normalize_strips_accents():
    assert normalize_player_name("Björn Borg") == "bjorn borg"

def test_normalize_lowercases():
    assert normalize_player_name("JANNIK SINNER") == "jannik sinner"

def test_normalize_strips_whitespace():
    assert normalize_player_name("  Carlos Alcaraz  ") == "carlos alcaraz"

def test_normalize_mixed():
    assert normalize_player_name("Novak Đoković") == "novak dokovic"


# ── merge_odds ────────────────────────────────────────────────────────────────

def test_merge_forward_match():
    matches = [{"p1_name": "Jannik Sinner", "p2_name": "Carlos Alcaraz"}]
    odds    = {"jannik sinner vs carlos alcaraz": (1.85, 2.05)}
    result  = merge_odds(matches, odds)
    assert result[0]["odd_p1"] == 1.85
    assert result[0]["odd_p2"] == 2.05

def test_merge_reversed_match():
    """API may return home/away in reversed order — odds must be swapped."""
    matches = [{"p1_name": "Jannik Sinner", "p2_name": "Carlos Alcaraz"}]
    odds    = {"carlos alcaraz vs jannik sinner": (2.05, 1.85)}
    result  = merge_odds(matches, odds)
    assert result[0]["odd_p1"] == 1.85   # swapped
    assert result[0]["odd_p2"] == 2.05   # swapped

def test_merge_no_match_returns_none():
    matches = [{"p1_name": "Unknown Player", "p2_name": "Also Unknown"}]
    odds    = {"jannik sinner vs carlos alcaraz": (1.85, 2.05)}
    result  = merge_odds(matches, odds)
    assert result[0]["odd_p1"] is None
    assert result[0]["odd_p2"] is None

def test_merge_does_not_mutate_input():
    matches  = [{"p1_name": "Jannik Sinner", "p2_name": "Carlos Alcaraz"}]
    original = dict(matches[0])
    merge_odds(matches, {})
    assert matches[0] == original

def test_merge_returns_new_list():
    matches = [{"p1_name": "A", "p2_name": "B"}]
    result  = merge_odds(matches, {})
    assert result is not matches

def test_merge_accent_normalization():
    """ESPN and API may format accented names differently."""
    matches = [{"p1_name": "Björn Borg", "p2_name": "Rafael Nadal"}]
    odds    = {"bjorn borg vs rafael nadal": (1.50, 2.60)}
    result  = merge_odds(matches, odds)
    assert result[0]["odd_p1"] == 1.50


# ── _extract_pinnacle_odds ────────────────────────────────────────────────────

def test_extract_pinnacle_present():
    bookmakers = [
        {"key": "bet365",  "markets": [{"key": "h2h", "outcomes": [{"price": 2.0}, {"price": 1.8}]}]},
        {"key": "pinnacle","markets": [{"key": "h2h", "outcomes": [{"price": 1.85},{"price": 2.05}]}]},
    ]
    assert _extract_pinnacle_odds(bookmakers) == (1.85, 2.05)

def test_extract_pinnacle_absent():
    bookmakers = [
        {"key": "bet365", "markets": [{"key": "h2h", "outcomes": [{"price": 2.0}, {"price": 1.8}]}]},
    ]
    assert _extract_pinnacle_odds(bookmakers) is None

def test_extract_pinnacle_empty():
    assert _extract_pinnacle_odds([]) is None


# ── _extract_avg_odds ─────────────────────────────────────────────────────────

def test_extract_avg_two_bookmakers():
    bookmakers = [
        {"key": "bk1", "markets": [{"key": "h2h", "outcomes": [{"price": 2.0}, {"price": 2.0}]}]},
        {"key": "bk2", "markets": [{"key": "h2h", "outcomes": [{"price": 1.8}, {"price": 2.2}]}]},
    ]
    result = _extract_avg_odds(bookmakers)
    assert result is not None
    assert abs(result[0] - 1.9) < 0.001
    assert abs(result[1] - 2.1) < 0.001

def test_extract_avg_empty():
    assert _extract_avg_odds([]) is None


# ── cache roundtrip ───────────────────────────────────────────────────────────

def test_cache_roundtrip(tmp_path, monkeypatch):
    """Write then read returns identical data."""
    import odds_api_client
    monkeypatch.setattr(odds_api_client, "ROOT", tmp_path)
    today  = date(2026, 3, 15)
    result = OddsResult(
        odds={"jannik sinner vs carlos alcaraz": (1.85, 2.05)},
        fetched_at="2026-03-15T09:32:00",
    )
    _save_cache("atp", today, result)
    loaded = _load_cache("atp", today)
    assert loaded is not None
    assert loaded.fetched_at == "2026-03-15T09:32:00"
    assert loaded.odds["jannik sinner vs carlos alcaraz"] == (1.85, 2.05)

def test_cache_missing_returns_none(tmp_path, monkeypatch):
    import odds_api_client
    monkeypatch.setattr(odds_api_client, "ROOT", tmp_path)
    assert _load_cache("atp", date(2026, 1, 1)) is None

def test_cache_corrupted_returns_none(tmp_path, monkeypatch):
    import odds_api_client
    monkeypatch.setattr(odds_api_client, "ROOT", tmp_path)
    today     = date(2026, 3, 15)
    cache_dir = tmp_path / "data" / "odds_cache" / "atp"
    cache_dir.mkdir(parents=True)
    (cache_dir / f"odds_{today.isoformat()}.json").write_text("NOT_VALID_JSON")
    assert _load_cache("atp", today) is None
```

- [ ] **Step 2: Run tests — expect failures (module doesn't exist yet)**

```bash
cd E:/Claude/botbet/tennis/tennis_ml
venv/Scripts/pytest tests/test_odds_api_client.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError` or import error (file not created yet — confirms test harness works)

- [ ] **Step 3: Run tests against the implementation from Task 1**

```bash
cd E:/Claude/botbet/tennis/tennis_ml
venv/Scripts/pytest tests/test_odds_api_client.py -v
```

Expected: **ALL GREEN** — 19 tests pass

If `python-dotenv` is not yet installed and causes import error:
```bash
venv/Scripts/pip install python-dotenv
venv/Scripts/pytest tests/test_odds_api_client.py -v
```

- [ ] **Step 4: Commit**

```bash
cd E:/Claude/botbet/tennis/tennis_ml
git add src/odds_api_client.py tests/test_odds_api_client.py
git commit -m "feat: add odds_api_client module with cache, normalize, merge"
```

---

## Chunk 2: Dashboard + Setup

### Task 3: Add import + odds fetch setup to render_today_tab()

**Files:**
- Modify: `src/dashboard.py`

- [ ] **Step 1: Add import at top of dashboard.py**

Find the existing imports block (around line 6). After:
```python
import json
import glob
import warnings
warnings.filterwarnings('ignore')
```

Add:
```python
from odds_api_client import fetch_odds_today, merge_odds
```

Exact edit — find:
```python
import json
import glob
import warnings
warnings.filterwarnings('ignore')

import numpy as np
```

Replace with:
```python
import json
import glob
import warnings
warnings.filterwarnings('ignore')

from odds_api_client import fetch_odds_today, merge_odds

import numpy as np
```

- [ ] **Step 2: Add odds fetch + merge + thresholds after the early-return guard**

The `if not matches: return` guard must stay BEFORE the odds fetch — this avoids calling
`merge_odds` on a potentially empty list and avoids a wasted API call when there are no matches.

In `render_today_tab()`, find:
```python
    if not matches:
        st.info("Aucun match programme aujourd'hui pour ce circuit.")
        return

    try:
        model, imputer, features, platt = load_model(tour)
```

Replace with:
```python
    if not matches:
        st.info("Aucun match programme aujourd'hui pour ce circuit.")
        return

    odds_result = fetch_odds_today(tour)
    matches     = merge_odds(matches, odds_result.odds)
    thresholds  = load_optimal_thresholds(tour)
    opt_edge    = thresholds.get("best_roi", {}).get("min_edge", 0.03)
    opt_prob    = thresholds.get("best_roi", {}).get("min_prob", 0.55)

    try:
        model, imputer, features, platt = load_model(tour)
```

- [ ] **Step 3: Add status bar after the bankroll caption**

Find (around line 1389):
```python
    st.caption(f"{len(matches)} match(s) programme(s) — bankroll : **{bankroll:.0f} €**")

    for i, m in enumerate(matches):
```

Replace with:
```python
    st.caption(f"{len(matches)} match(s) programme(s) — bankroll : **{bankroll:.0f} €**")

    # ── Status bar Odds API ───────────────────────────────────────────────────
    n_enriched = sum(1 for m in matches if m.get("odd_p1") is not None)
    if odds_result.fetched_at:
        fetched_time = odds_result.fetched_at[11:16]
        _sb1, _sb2 = st.columns([5, 1])
        with _sb1:
            st.html(
                f'<span style="font-family:\'DM Mono\',monospace;font-size:0.72rem;color:#3DFFA0;">'
                f'● Cotes Odds API</span>'
                f'<span style="font-size:0.72rem;color:#64748b;"> — {n_enriched} match'
                f'{"s" if n_enriched != 1 else ""} enrichi'
                f'{"s" if n_enriched != 1 else ""} · mis à jour {fetched_time}</span>'
            )
        with _sb2:
            if st.button("↻ Rafraîchir", key=f"refresh_odds_{tour}",
                         use_container_width=True):
                _cache_file = (ROOT / "data" / "odds_cache" / tour
                               / f"odds_{date.today().isoformat()}.json")
                if _cache_file.exists():
                    _cache_file.unlink()
                st.rerun()

    for i, m in enumerate(matches):
```

- [ ] **Step 4: Verify syntax**

```bash
cd E:/Claude/botbet/tennis/tennis_ml
venv/Scripts/python -c "import ast; ast.parse(open('src/dashboard.py').read()); print('OK')"
```

Expected: `OK`

---

### Task 4: Add GO/PASS badge to match cards

**Files:**
- Modify: `src/dashboard.py`

- [ ] **Step 1: Add GO/PASS badge inside the match container**

Find the exact text at the end of the `with c2:` block (unique in file):
```python
                    f'Favori : <b style="color:#f1f5f9">{fav}</b> ({fav_prob:.0%})'
                    f'</div>'
                )
            with c3:
```

Replace with:
```python
                    f'Favori : <b style="color:#f1f5f9">{fav}</b> ({fav_prob:.0%})'
                    f'</div>'
                )

            # ── GO/PASS badge (auto-computed from Odds API) ───────────────────
            _op1 = m.get("odd_p1")
            _op2 = m.get("odd_p2")
            if _op1 is not None and _op2 is not None:
                _e1, _, _i1 = compute_edge(prob_p1, _op1, _op2)
                _e2, _, _i2 = compute_edge(prob_p2, _op2, _op1)
                _go_p1 = _e1 >= opt_edge and prob_p1 >= opt_prob
                _go_p2 = _e2 >= opt_edge and prob_p2 >= opt_prob
                if _go_p1:
                    _badge = (f"GO · {m['p1_name'].split()[0]} @ {_op1:.2f}"
                              f" · edge {_e1:+.1%}")
                    _bcol, _bbg, _bbd = "#3DFFA0", "rgba(61,255,160,0.07)", "rgba(61,255,160,0.25)"
                elif _go_p2:
                    _badge = (f"GO · {m['p2_name'].split()[0]} @ {_op2:.2f}"
                              f" · edge {_e2:+.1%}")
                    _bcol, _bbg, _bbd = "#3DFFA0", "rgba(61,255,160,0.07)", "rgba(61,255,160,0.25)"
                else:
                    _badge = "PASS"
                    _bcol, _bbg, _bbd = "#64748b", "rgba(30,40,60,0.4)", "rgba(80,100,140,0.15)"
                st.html(
                    f'<div style="margin-top:4px;padding:5px 14px;'
                    f'background:{_bbg};border:1px solid {_bbd};'
                    f'border-radius:6px;font-family:\'DM Mono\',monospace;'
                    f'font-size:0.7rem;color:{_bcol};font-weight:700;'
                    f'letter-spacing:0.5px">{_badge}</div>'
                )

            with c3:
```

- [ ] **Step 2: Verify syntax**

```bash
cd E:/Claude/botbet/tennis/tennis_ml
venv/Scripts/python -c "import ast; ast.parse(open('src/dashboard.py').read()); print('OK')"
```

Expected: `OK`

---

### Task 5: Pre-fill odds in the "+Pari" popover

**Files:**
- Modify: `src/dashboard.py`

- [ ] **Step 1: Replace hardcoded value=2.0 in both number_inputs**

Find (inside the `with st.popover("+ Pari", ...)` block):
```python
                        o1 = st.number_input(
                            f"Cote {m['p1_name'].split()[0]}",
                            min_value=1.01, value=2.0, step=0.05,
                            key=f"today_o1_{tour}_{i}",
                        )
                    with cb:
                        o2 = st.number_input(
                            f"Cote {m['p2_name'].split()[0]}",
                            min_value=1.01, value=2.0, step=0.05,
                            key=f"today_o2_{tour}_{i}",
                        )
```

Replace with:
```python
                        o1 = st.number_input(
                            f"Cote {m['p1_name'].split()[0]}",
                            min_value=1.01,
                            value=float(m.get("odd_p1") or 2.0),
                            step=0.05,
                            key=f"today_o1_{tour}_{i}",
                        )
                    with cb:
                        o2 = st.number_input(
                            f"Cote {m['p2_name'].split()[0]}",
                            min_value=1.01,
                            value=float(m.get("odd_p2") or 2.0),
                            step=0.05,
                            key=f"today_o2_{tour}_{i}",
                        )
```

- [ ] **Step 2: Verify syntax**

```bash
cd E:/Claude/botbet/tennis/tennis_ml
venv/Scripts/python -c "import ast; ast.parse(open('src/dashboard.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit dashboard changes**

```bash
cd E:/Claude/botbet/tennis/tennis_ml
git add src/dashboard.py
git commit -m "feat: integrate Odds API into today tab — GO/PASS badge + pre-filled odds"
```

---

### Task 6: Setup — requirements.txt, .gitignore, install

**Files:**
- Modify: `requirements.txt`
- Create: `.gitignore`

- [ ] **Step 1: Add python-dotenv to requirements.txt**

Find the end of `requirements.txt`. Append the line:
```
python-dotenv>=1.0.0
```

Verify it's not already there first:
```bash
cd E:/Claude/botbet/tennis/tennis_ml
grep -i "dotenv" requirements.txt || echo "not found — safe to add"
```

- [ ] **Step 2: Create .gitignore**

```bash
cd E:/Claude/botbet/tennis/tennis_ml
cat .gitignore 2>/dev/null || echo "(none)"
```

Create `.gitignore` with:
```
# Secrets
.env

# Odds API cache (auto-generated daily, no need to commit)
data/odds_cache/

# Python
venv/
__pycache__/
*.pyc
*.pyo
.pytest_cache/

# Data files (large, not committed)
data/raw/
data/processed/
data/models/
data/predictions/
data/odds/

# Jupyter
.ipynb_checkpoints/
```

- [ ] **Step 3: Install python-dotenv**

```bash
cd E:/Claude/botbet/tennis/tennis_ml
venv/Scripts/pip install python-dotenv
```

Expected: `Successfully installed python-dotenv-X.X.X` (or "already satisfied")

- [ ] **Step 4: Create .env template (user-facing instruction)**

Create `.env.example` at project root (safe to commit, shows required keys):
```bash
# Rename to .env and fill in your key
ODDS_API_KEY=your_key_here
```

- [ ] **Step 5: Run full test suite to confirm no regressions**

```bash
cd E:/Claude/botbet/tennis/tennis_ml
venv/Scripts/pytest tests/test_odds_api_client.py -v
```

Expected: All 19 tests PASS

- [ ] **Step 6: Syntax check dashboard one final time**

```bash
cd E:/Claude/botbet/tennis/tennis_ml
venv/Scripts/python -c "import ast; ast.parse(open('src/dashboard.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 7: Commit setup files**

```bash
cd E:/Claude/botbet/tennis/tennis_ml
git add requirements.txt .gitignore .env.example
git commit -m "chore: add python-dotenv, .gitignore, .env.example for Odds API setup"
```

---

## Post-Implementation: User Action Required

Before the dashboard will fetch real odds, the user must:

1. Create `.env` at project root:
   ```
   ODDS_API_KEY=your_actual_key_here
   ```
2. Launch dashboard: `streamlit run src/dashboard.py`
3. Open Today tab → ATP
4. If matches scheduled today: verify status bar shows "● Cotes Odds API — X matchs enrichis"
5. If no matches today: verify dashboard loads without errors (badge and status bar simply don't appear)

---

## Summary

| What | Files | Verification |
|------|-------|-------------|
| Odds client (fetch + cache + merge) | `src/odds_api_client.py` | 19 pytest tests all green |
| Dashboard integration | `src/dashboard.py` | `ast.parse` OK + visual check |
| Dependencies | `requirements.txt`, `.gitignore`, `.env.example` | `pip install` succeeds |
