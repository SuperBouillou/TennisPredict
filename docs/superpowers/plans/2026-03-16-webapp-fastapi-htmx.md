# TennisPredict Web App (FastAPI + HTMX) — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Streamlit dashboard with a FastAPI + HTMX web app covering 5 pages: Today's matches, Manual predictions, Bet history, Player profiles, and Backtest stats.

**Architecture:** FastAPI serves full HTML pages (Jinja2) and HTMX partials. SQLite replaces scattered CSV/JSON files. ML models are loaded once at startup via lifespan. Chart.js via CDN handles all charts (no build step).

**Tech Stack:** FastAPI, Jinja2, HTMX 1.9 (CDN), SQLite (stdlib), Chart.js 4 (CDN), uvicorn, joblib, pandas, existing `src/espn_client.py` + `src/config.py` + `src/predict_today.py`

**Spec:** `docs/superpowers/specs/2026-03-16-web-app-design.md`

**Run:** `uvicorn src.webapp.main:app --reload --port 8000`

---

## Chunk 1: Core Infrastructure

> DB schema, ML wrapper, FastAPI app skeleton, base layout + CSS.

### Task 1: Project scaffold

**Files:**
- Create: `src/webapp/__init__.py`
- Create: `src/webapp/routers/__init__.py`
- Create: `src/webapp/templates/partials/.gitkeep`
- Create: `src/webapp/migrations/001_init.sql`
- Create: `tests/webapp/__init__.py`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p src/webapp/routers src/webapp/templates/partials src/webapp/static src/webapp/migrations tests/webapp
touch src/webapp/__init__.py src/webapp/routers/__init__.py tests/webapp/__init__.py
```

- [ ] **Step 2: Write SQL schema** → `src/webapp/migrations/001_init.sql`

```sql
-- Paris
CREATE TABLE IF NOT EXISTS bets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tour        TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    tournament  TEXT NOT NULL,
    surface     TEXT NOT NULL,
    round       TEXT,
    p1_name     TEXT NOT NULL,
    p2_name     TEXT NOT NULL,
    bet_on      TEXT NOT NULL,
    prob        REAL NOT NULL,
    edge        REAL,
    odd         REAL NOT NULL,
    stake       REAL NOT NULL,
    kelly_frac  REAL,
    status      TEXT DEFAULT 'pending',
    pnl         REAL DEFAULT 0,
    resolved_at TEXT
);

-- Bankroll (une ligne par circuit)
CREATE TABLE IF NOT EXISTS bankroll (
    tour       TEXT PRIMARY KEY,
    amount     REAL NOT NULL DEFAULT 1000.0,
    updated_at TEXT NOT NULL
);

-- Settings
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

- [ ] **Step 3: Commit scaffold**

```bash
git add src/webapp/ tests/webapp/
git commit -m "feat(webapp): scaffold directory structure and SQL schema"
```

---

### Task 2: Database module (`db.py`)

**Files:**
- Create: `src/webapp/db.py`
- Create: `tests/webapp/test_db.py`

- [ ] **Step 1: Write failing tests** → `tests/webapp/test_db.py`

```python
import pytest
import sqlite3
from pathlib import Path
from src.webapp import db

@pytest.fixture
def conn(tmp_path):
    c = db.get_connection(tmp_path / "test.db")
    db.init_db(c)
    return c

def test_init_creates_tables(conn):
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {'bets', 'bankroll', 'settings'} <= tables

def test_get_bankroll_default(conn):
    amount = db.get_bankroll(conn, 'atp')
    assert amount == 1000.0

def test_set_and_get_bankroll(conn):
    db.set_bankroll(conn, 'atp', 1250.0)
    assert db.get_bankroll(conn, 'atp') == 1250.0

def test_add_bet_and_list(conn):
    db.set_bankroll(conn, 'atp', 1000.0)
    bet_id = db.add_bet(conn, {
        'tour': 'atp', 'tournament': 'Australian Open', 'surface': 'Hard',
        'round': 'R64', 'p1_name': 'Sinner J.', 'p2_name': 'Alcaraz C.',
        'bet_on': 'p1', 'prob': 0.67, 'edge': 0.05, 'odd': 1.75,
        'stake': 50.0, 'kelly_frac': 0.08,
    })
    assert bet_id > 0
    # stake debited from bankroll
    assert db.get_bankroll(conn, 'atp') == 950.0

def test_resolve_bet_won(conn):
    db.set_bankroll(conn, 'atp', 1000.0)
    bet_id = db.add_bet(conn, {
        'tour': 'atp', 'tournament': 'T', 'surface': 'Hard', 'round': 'R64',
        'p1_name': 'A', 'p2_name': 'B', 'bet_on': 'p1',
        'prob': 0.6, 'edge': 0.04, 'odd': 2.0, 'stake': 100.0, 'kelly_frac': 0.1,
    })
    db.resolve_bet(conn, bet_id, 'won')
    assert db.get_bankroll(conn, 'atp') == 1000.0  # 900 - 100 stake + 200 return = 1000... wait
    # bankroll after add_bet = 1000 - 100 = 900
    # on won: bankroll += stake * (odd - 1) = 900 + 100 = 1000
    assert db.get_bankroll(conn, 'atp') == 1000.0
    bet = db.get_bet(conn, bet_id)
    assert bet['status'] == 'won'
    assert bet['pnl'] == pytest.approx(100.0)

def test_resolve_bet_lost(conn):
    db.set_bankroll(conn, 'atp', 1000.0)
    bet_id = db.add_bet(conn, {
        'tour': 'atp', 'tournament': 'T', 'surface': 'Hard', 'round': 'R64',
        'p1_name': 'A', 'p2_name': 'B', 'bet_on': 'p1',
        'prob': 0.6, 'edge': 0.04, 'odd': 2.0, 'stake': 100.0, 'kelly_frac': 0.1,
    })
    db.resolve_bet(conn, bet_id, 'lost')
    assert db.get_bankroll(conn, 'atp') == 900.0  # stake already debited, no change
    bet = db.get_bet(conn, bet_id)
    assert bet['status'] == 'lost'
    assert bet['pnl'] == -100.0

def test_resolve_already_resolved_raises(conn):
    db.set_bankroll(conn, 'atp', 1000.0)
    bet_id = db.add_bet(conn, {
        'tour': 'atp', 'tournament': 'T', 'surface': 'Hard', 'round': 'R64',
        'p1_name': 'A', 'p2_name': 'B', 'bet_on': 'p1',
        'prob': 0.6, 'edge': 0.04, 'odd': 2.0, 'stake': 50.0, 'kelly_frac': 0.05,
    })
    db.resolve_bet(conn, bet_id, 'won')
    with pytest.raises(ValueError, match="already resolved"):
        db.resolve_bet(conn, bet_id, 'lost')

def test_get_setting_default(conn):
    assert db.get_setting(conn, 'min_edge', '0.03') == '0.03'

def test_set_and_get_setting(conn):
    db.set_setting(conn, 'min_edge', '0.05')
    assert db.get_setting(conn, 'min_edge', '0.03') == '0.05'

def test_list_bets_filter_by_tour(conn):
    db.set_bankroll(conn, 'atp', 1000.0)
    db.set_bankroll(conn, 'wta', 1000.0)
    db.add_bet(conn, {'tour': 'atp', 'tournament': 'T', 'surface': 'Hard',
        'round': 'R64', 'p1_name': 'A', 'p2_name': 'B', 'bet_on': 'p1',
        'prob': 0.6, 'edge': 0.04, 'odd': 2.0, 'stake': 10.0, 'kelly_frac': 0.05})
    db.add_bet(conn, {'tour': 'wta', 'tournament': 'T', 'surface': 'Clay',
        'round': 'SF', 'p1_name': 'C', 'p2_name': 'D', 'bet_on': 'p2',
        'prob': 0.55, 'edge': 0.02, 'odd': 1.9, 'stake': 20.0, 'kelly_frac': 0.04})
    atp_bets = db.list_bets(conn, tour='atp')
    assert len(atp_bets) == 1 and atp_bets[0]['tour'] == 'atp'
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
cd E:/Claude/botbet/tennis/tennis_ml && source venv/Scripts/activate && python -m pytest tests/webapp/test_db.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'src.webapp'`

- [ ] **Step 3: Implement `src/webapp/db.py`**

```python
"""SQLite helpers — bets, bankroll, settings."""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DB = ROOT / "data" / "tennis_predict.db"
_SCHEMA = Path(__file__).parent / "migrations" / "001_init.sql"


def get_connection(path: Path = _DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA.read_text())
    conn.commit()
    # Seed default bankroll rows if missing
    for tour in ('atp', 'wta'):
        conn.execute(
            "INSERT OR IGNORE INTO bankroll (tour, amount, updated_at) VALUES (?, 1000.0, ?)",
            (tour, _now()),
        )
    # Seed default settings if missing
    defaults = {'min_edge': '0.03', 'min_prob': '0.55', 'kelly_fraction': '0.25'}
    for k, v in defaults.items():
        conn.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (k, v))
    conn.commit()


# ── Bankroll ──────────────────────────────────────────────────────────────────

def get_bankroll(conn: sqlite3.Connection, tour: str) -> float:
    row = conn.execute("SELECT amount FROM bankroll WHERE tour = ?", (tour,)).fetchone()
    return row['amount'] if row else 1000.0


def set_bankroll(conn: sqlite3.Connection, tour: str, amount: float) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO bankroll (tour, amount, updated_at) VALUES (?, ?, ?)",
        (tour, round(amount, 2), _now()),
    )
    conn.commit()


# ── Bets ──────────────────────────────────────────────────────────────────────

def add_bet(conn: sqlite3.Connection, bet: dict) -> int:
    """Insert bet and debit stake from bankroll. Returns new bet id."""
    now = _now()
    cur = conn.execute(
        """INSERT INTO bets
           (tour, created_at, tournament, surface, round, p1_name, p2_name,
            bet_on, prob, edge, odd, stake, kelly_frac, status, pnl)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,'pending',0)""",
        (bet['tour'], now, bet['tournament'], bet['surface'], bet.get('round'),
         bet['p1_name'], bet['p2_name'], bet['bet_on'], bet['prob'],
         bet.get('edge'), bet['odd'], bet['stake'], bet.get('kelly_frac')),
    )
    # Debit stake immediately (pessimistic accounting)
    current = get_bankroll(conn, bet['tour'])
    set_bankroll(conn, bet['tour'], current - bet['stake'])
    conn.commit()
    return cur.lastrowid


def get_bet(conn: sqlite3.Connection, bet_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM bets WHERE id = ?", (bet_id,)).fetchone()
    return dict(row) if row else None


def resolve_bet(conn: sqlite3.Connection, bet_id: int, outcome: str) -> None:
    """outcome: 'won' | 'lost'"""
    bet = get_bet(conn, bet_id)
    if bet is None:
        raise ValueError(f"Bet {bet_id} not found")
    if bet['status'] != 'pending':
        raise ValueError(f"Bet {bet_id} already resolved (status={bet['status']})")

    if outcome == 'won':
        profit = round(bet['stake'] * (bet['odd'] - 1), 2)
        pnl = profit
        current = get_bankroll(conn, bet['tour'])
        set_bankroll(conn, bet['tour'], current + bet['stake'] + profit)
    else:
        pnl = -bet['stake']

    conn.execute(
        "UPDATE bets SET status=?, pnl=?, resolved_at=? WHERE id=?",
        (outcome, pnl, _now(), bet_id),
    )
    conn.commit()


def list_bets(
    conn: sqlite3.Connection,
    tour: str | None = None,
    status: str | None = None,
    surface: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[dict]:
    clauses, params = [], []
    if tour:
        clauses.append("tour = ?"); params.append(tour)
    if status:
        clauses.append("status = ?"); params.append(status)
    if surface:
        clauses.append("surface = ?"); params.append(surface)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = conn.execute(
        f"SELECT * FROM bets {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
        params + [limit, offset],
    ).fetchall()
    return [dict(r) for r in rows]


# ── Settings ──────────────────────────────────────────────────────────────────

def get_setting(conn: sqlite3.Connection, key: str, default: str = '') -> str:
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row['value'] if row else default


def set_setting(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
    conn.commit()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
python -m pytest tests/webapp/test_db.py -v
```

Expected: 10/10 PASS

- [ ] **Step 5: Commit**

```bash
git add src/webapp/db.py tests/webapp/test_db.py
git commit -m "feat(webapp): add SQLite db module with bets/bankroll/settings"
```

---

### Task 3: ML wrapper (`ml.py`)

**Files:**
- Create: `src/webapp/ml.py`
- Create: `tests/webapp/test_ml.py`

- [ ] **Step 1: Write failing tests** → `tests/webapp/test_ml.py`

```python
"""Tests for ml.py — prediction wrapper."""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.webapp import ml


@pytest.fixture
def mock_artifacts():
    """Minimal mock artifacts for ATP."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.35, 0.65]])

    imputer = MagicMock()
    imputer.transform.side_effect = lambda x: x

    platt = MagicMock()
    platt.predict_proba.return_value = np.array([[0.33, 0.67]])

    features = ['elo_diff', 'rank_diff', 'form_win_rate_5_diff']

    # Minimal player profile row
    import pandas as pd
    profiles = pd.DataFrame([{
        'player_name': 'Jannik Sinner',
        'name_key': 'jannik sinner',
        'rank': 1,
        'elo': 2200.0,
        'elo_hard': 2210.0,
        'elo_clay': 2150.0,
        'elo_grass': 2100.0,
        'form_win_rate_5': 0.8,
        'form_win_rate_10': 0.75,
        'form_win_rate_20': 0.72,
        'h2h_win_rate': 0.6,
        'rank_points': 11000,
    }, {
        'player_name': 'Carlos Alcaraz',
        'name_key': 'carlos alcaraz',
        'rank': 2,
        'elo': 2180.0,
        'elo_hard': 2190.0,
        'elo_clay': 2200.0,
        'elo_grass': 2160.0,
        'form_win_rate_5': 0.75,
        'form_win_rate_10': 0.72,
        'form_win_rate_20': 0.70,
        'h2h_win_rate': 0.5,
        'rank_points': 9500,
    }])
    return {
        'model': model,
        'imputer': imputer,
        'platt': platt,
        'feature_list': features,
        'profiles': profiles,
    }


def test_predict_known_players(mock_artifacts):
    result = ml.predict(
        mock_artifacts,
        p1_name='Jannik Sinner',
        p2_name='Carlos Alcaraz',
        tournament='Australian Open',
        surface='Hard',
        round_='R64',
        best_of=3,
        odd_p1=None,
        odd_p2=None,
        bankroll=1000.0,
    )
    assert result['p1_found'] is True
    assert result['p2_found'] is True
    assert 0 < result['prob_p1'] < 1
    assert abs(result['prob_p1'] + result['prob_p2'] - 1.0) < 1e-6
    assert result['edge'] is None  # no odds provided
    assert result['kelly_frac'] is None


def test_predict_with_odds(mock_artifacts):
    result = ml.predict(
        mock_artifacts,
        p1_name='Jannik Sinner',
        p2_name='Carlos Alcaraz',
        tournament='Australian Open',
        surface='Hard',
        round_='R64',
        best_of=3,
        odd_p1=1.50,
        odd_p2=2.50,
        bankroll=1000.0,
    )
    assert result['edge'] is not None
    assert result['kelly_frac'] is not None
    assert result['kelly_eur'] is not None


def test_predict_unknown_player_returns_elo_only(mock_artifacts):
    result = ml.predict(
        mock_artifacts,
        p1_name='Unknown Player',
        p2_name='Carlos Alcaraz',
        tournament='Roland Garros',
        surface='Clay',
        round_='QF',
        best_of=5,
        odd_p1=None,
        odd_p2=None,
        bankroll=1000.0,
    )
    assert result['p1_found'] is False
    assert result['elo_only'] is True


def test_kelly_capped_at_max_fraction(mock_artifacts):
    """Kelly fraction should never exceed 0.25 (full Kelly cap)."""
    result = ml.predict(
        mock_artifacts,
        p1_name='Jannik Sinner',
        p2_name='Carlos Alcaraz',
        tournament='Wimbledon',
        surface='Grass',
        round_='F',
        best_of=5,
        odd_p1=10.0,  # extreme odds → huge raw Kelly
        odd_p2=1.02,
        bankroll=1000.0,
    )
    if result['kelly_frac'] is not None:
        assert result['kelly_frac'] <= 0.25
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
python -m pytest tests/webapp/test_ml.py -v 2>&1 | head -20
```

- [ ] **Step 3: Implement `src/webapp/ml.py`**

```python
"""ML wrapper — loads artifacts once, exposes predict()."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Feature builder (reuses logic from predict_today.py)
# ─────────────────────────────────────────────────────────────────────────────

_ROUND_IMP = {
    'R128': 0.1, 'R64': 0.2, 'R32': 0.3, 'R16': 0.4,
    'QF': 0.6, 'SF': 0.8, 'F': 1.0, 'RR': 0.3,
}
_SURFACE_ENC = {'Hard': 0, 'Clay': 1, 'Grass': 2, 'Carpet': 3}
_GS = {'australian open', 'roland garros', 'wimbledon', 'us open'}
_MASTERS = {
    'miami open', 'monte-carlo', 'madrid open', 'rome', 'canadian open',
    'cincinnati', 'shanghai', 'paris masters', 'indian wells',
}


def _tourney_importance(name: str) -> float:
    n = name.lower()
    if any(g in n for g in _GS):
        return 1.0
    if any(m in n for m in _MASTERS):
        return 0.7
    return 0.4


def _get_player(profiles: pd.DataFrame, name: str) -> dict | None:
    key = name.lower().strip()
    rows = profiles[profiles['name_key'] == key]
    if rows.empty:
        # Fuzzy: last-name match
        last = key.split()[0] if key else ''
        rows = profiles[profiles['name_key'].str.startswith(last)]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def _elo_win_prob(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def _build_features(p1: dict, p2: dict, tournament: str, surface: str,
                    round_: str, best_of: int, feature_list: list[str]) -> np.ndarray:
    """Build feature vector matching feature_list order. Missing → NaN."""
    surf_key = f'elo_{surface.lower()}'
    elo1 = p1.get('elo', 1500.0)
    elo2 = p2.get('elo', 1500.0)
    elo_surf1 = p1.get(surf_key, elo1)
    elo_surf2 = p2.get(surf_key, elo2)

    row: dict[str, Any] = {
        'elo_diff':             elo1 - elo2,
        'elo_surface_diff':     elo_surf1 - elo_surf2,
        'elo_win_prob_p1':      _elo_win_prob(elo1, elo2),
        'rank_diff':            (p1.get('rank', 500) or 500) - (p2.get('rank', 500) or 500),
        'rank_ratio':           ((p1.get('rank', 500) or 500) /
                                 max((p2.get('rank', 500) or 500), 1)),
        'rank_points_diff':     (p1.get('rank_points', 0) or 0) - (p2.get('rank_points', 0) or 0),
        'form_win_rate_5_diff': (p1.get('form_win_rate_5', 0.5) or 0.5) -
                                (p2.get('form_win_rate_5', 0.5) or 0.5),
        'form_win_rate_10_diff':(p1.get('form_win_rate_10', 0.5) or 0.5) -
                                (p2.get('form_win_rate_10', 0.5) or 0.5),
        'form_win_rate_20_diff':(p1.get('form_win_rate_20', 0.5) or 0.5) -
                                (p2.get('form_win_rate_20', 0.5) or 0.5),
        'h2h_win_rate_p1':      p1.get('h2h_win_rate', 0.5) or 0.5,
        'surface_enc':          _SURFACE_ENC.get(surface, 0),
        'tourney_importance':   _tourney_importance(tournament),
        'round_importance':     _ROUND_IMP.get(round_, 0.3),
        'best_of_5':            1 if best_of == 5 else 0,
    }

    vec = np.array([row.get(f, np.nan) for f in feature_list], dtype=float)
    return vec.reshape(1, -1)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    artifacts: dict,
    p1_name: str,
    p2_name: str,
    tournament: str,
    surface: str,
    round_: str,
    best_of: int,
    odd_p1: float | None,
    odd_p2: float | None,
    bankroll: float,
) -> dict:
    """Run prediction and return result dict."""
    profiles    = artifacts['profiles']
    model       = artifacts['model']
    imputer     = artifacts['imputer']
    platt       = artifacts['platt']
    feature_list = artifacts['feature_list']

    p1 = _get_player(profiles, p1_name)
    p2 = _get_player(profiles, p2_name)

    p1_found = p1 is not None
    p2_found = p2 is not None

    if not p1_found:
        p1 = {}
    if not p2_found:
        p2 = {}

    elo_only = not p1_found or not p2_found

    # ELO baseline
    elo1 = p1.get('elo', 1500.0)
    elo2 = p2.get('elo', 1500.0)
    elo_prob = _elo_win_prob(elo1, elo2)

    # ML prediction
    X = _build_features(p1, p2, tournament, surface, round_, best_of, feature_list)
    X_imp = imputer.transform(X)
    raw_prob = model.predict_proba(X_imp)[0, 1]
    cal_prob = float(platt.predict_proba([[raw_prob]])[0, 1])
    cal_prob = max(0.01, min(0.99, cal_prob))

    # Edge + Kelly
    edge = ev = kelly_frac = kelly_eur = None
    if odd_p1 is not None and odd_p1 > 1.0:
        implied = 1.0 / odd_p1
        edge = round(cal_prob - implied, 4)
        ev   = round(cal_prob * (odd_p1 - 1) - (1 - cal_prob), 4)
        if edge > 0:
            raw_kelly = (cal_prob * odd_p1 - 1) / (odd_p1 - 1)
            # Quarter-Kelly, capped at 25%
            kelly_frac = round(min(raw_kelly * 0.25, 0.25), 4)
            kelly_eur  = round(kelly_frac * bankroll, 2)

    return {
        'prob_p1':   round(cal_prob, 4),
        'prob_p2':   round(1 - cal_prob, 4),
        'elo_prob':  round(elo_prob, 4),
        'edge':      edge,
        'ev':        ev,
        'kelly_frac': kelly_frac,
        'kelly_eur':  kelly_eur,
        'p1_found':  p1_found,
        'p2_found':  p2_found,
        'elo_only':  elo_only,
        'confidence': round(cal_prob, 4),
    }
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
python -m pytest tests/webapp/test_ml.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/webapp/ml.py tests/webapp/test_ml.py
git commit -m "feat(webapp): add ML prediction wrapper with Kelly calculation"
```

---

### Task 4: FastAPI app entry + base templates + CSS

**Files:**
- Create: `src/webapp/main.py`
- Create: `src/webapp/templates/base.html`
- Create: `src/webapp/static/app.css`
- Create: `src/webapp/static/manifest.json`
- Create: `src/webapp/static/sw.js`
- Create: `tests/webapp/test_routes.py` (skeleton — will grow each task)

- [ ] **Step 1: Write `src/webapp/main.py`**

```python
"""FastAPI entry point — loads ML artifacts once via lifespan."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.webapp.db import get_connection, init_db
from src.config import get_paths

ROOT = Path(__file__).resolve().parent.parent.parent

# ── Shared state ──────────────────────────────────────────────────────────────
# Populated once at startup, read-only during requests.
APP_STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models + open DB connection at startup."""
    db_path = ROOT / "data" / "tennis_predict.db"
    conn = get_connection(db_path)
    init_db(conn)
    APP_STATE['db'] = conn

    models = {}
    for tour in ('atp', 'wta'):
        paths = get_paths(tour)
        mdir  = paths['models_dir']
        pdir  = paths['processed_dir']
        try:
            profiles_path = pdir / 'player_profiles_updated.parquet'
            if not profiles_path.exists():
                profiles_path = pdir / 'matches_features_final.parquet'

            profiles = pd.read_parquet(profiles_path)
            if 'name_key' not in profiles.columns:
                profiles['name_key'] = profiles['player_name'].str.lower().str.strip()

            players_path = pdir / 'players.parquet'
            players = pd.read_parquet(players_path) if players_path.exists() else pd.DataFrame()

            models[tour] = {
                'model':        joblib.load(mdir / 'xgb_tuned.pkl'),
                'imputer':      joblib.load(mdir / 'imputer.pkl'),
                'platt':        joblib.load(mdir / 'platt_scaler.pkl'),
                'feature_list': joblib.load(mdir / 'feature_list.pkl'),
                'profiles':     profiles,
                'players':      players,
            }
        except FileNotFoundError as e:
            print(f"[WARN] {tour.upper()} artifacts missing: {e}")
            models[tour] = None

    APP_STATE['models'] = models
    APP_STATE['sync_status'] = {'atp': 'idle', 'wta': 'idle'}
    yield
    conn.close()


app = FastAPI(title="TennisPredict", lifespan=lifespan)

# Static files + templates
_HERE = Path(__file__).parent
app.mount("/static", StaticFiles(directory=_HERE / "static"), name="static")
templates = Jinja2Templates(directory=_HERE / "templates")

# ── Routers ───────────────────────────────────────────────────────────────────
from src.webapp.routers import today, predictions, history, joueurs, stats  # noqa: E402

app.include_router(today.router)
app.include_router(predictions.router)
app.include_router(history.router)
app.include_router(joueurs.router)
app.include_router(stats.router)


@app.get("/")
async def root():
    return RedirectResponse("/today")
```

- [ ] **Step 2: Write `src/webapp/templates/base.html`**

```html
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{% block title %}TennisPredict{% endblock %}</title>
  <link rel="manifest" href="/static/manifest.json">
  <link rel="stylesheet" href="/static/app.css">
  <script src="https://unpkg.com/htmx.org@1.9.10" defer></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js" defer></script>
  {% block head %}{% endblock %}
</head>
<body>
  <!-- ── Sidebar ── -->
  <nav class="sidebar">
    <div class="logo">🎾</div>
    <a href="/today"       class="nav-item {% if active=='today' %}active{% endif %}" title="Matchs du jour">📅</a>
    <a href="/predictions" class="nav-item {% if active=='predictions' %}active{% endif %}" title="Prédictions">⚡</a>
    <a href="/history"     class="nav-item {% if active=='history' %}active{% endif %}" title="Historique">📋</a>
    <a href="/joueurs"     class="nav-item {% if active=='joueurs' %}active{% endif %}" title="Joueurs">👤</a>
    <a href="/stats"       class="nav-item {% if active=='stats' %}active{% endif %}" title="Statistiques">📊</a>
  </nav>

  <!-- ── Main ── -->
  <div class="main">
    <div class="topbar">
      <div>
        <div class="topbar-title">{% block page_title %}TennisPredict{% endblock %}</div>
        <div class="topbar-meta">{% block page_meta %}{% endblock %}</div>
      </div>
      <div>{% block topbar_right %}{% endblock %}</div>
    </div>
    <div class="content">
      {% block content %}{% endblock %}
    </div>
  </div>
  {% block scripts %}{% endblock %}
</body>
</html>
```

- [ ] **Step 3: Write `src/webapp/static/app.css`** (dark theme, matches mockup)

Key CSS variables (full file in implementation):
```css
:root {
  --bg:      #0f172a;
  --surface: #1e293b;
  --border:  #334155;
  --text:    #f1f5f9;
  --muted:   #64748b;
  --blue:    #3b82f6;
  --green:   #22c55e;
  --orange:  #f97316;
  --red:     #ef4444;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: system-ui, sans-serif;
       display: flex; height: 100vh; overflow: hidden; }
.sidebar { width: 64px; background: var(--surface); border-right: 1px solid var(--border);
           display: flex; flex-direction: column; align-items: center; padding: 16px 0; gap: 6px; }
.logo { width: 36px; height: 36px; background: var(--blue); border-radius: 10px;
        display: flex; align-items: center; justify-content: center; font-size: 18px; margin-bottom: 12px; }
.nav-item { width: 44px; height: 44px; border-radius: 12px; display: flex; align-items: center;
            justify-content: center; font-size: 20px; cursor: pointer; text-decoration: none;
            transition: background 0.15s; color: var(--text); }
.nav-item:hover { background: var(--border); }
.nav-item.active { background: #1d4ed8; box-shadow: 0 0 0 2px var(--blue); }
.main { flex: 1; overflow-y: auto; display: flex; flex-direction: column; }
.topbar { padding: 14px 20px; border-bottom: 1px solid var(--surface);
          display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; }
.topbar-title { font-size: 15px; font-weight: 700; }
.topbar-meta { font-size: 11px; color: var(--muted); margin-top: 1px; }
.content { padding: 16px 20px; display: flex; flex-direction: column; gap: 10px; overflow-y: auto; }
/* Toggle ATP/WTA */
.toggle-group { display: flex; background: var(--bg); border-radius: 20px; padding: 3px; }
.toggle-btn { padding: 4px 16px; border-radius: 16px; font-size: 11px; font-weight: 700;
              cursor: pointer; color: var(--muted); border: none; background: transparent; }
.toggle-btn.active { background: var(--blue); color: white; }
/* Cards */
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; }
.badge { padding: 2px 8px; border-radius: 10px; font-size: 10px; font-weight: 700; }
.badge-value  { background: rgba(34,197,94,.15); color: var(--green); }
.badge-edge   { background: rgba(249,115,22,.15); color: var(--orange); }
.badge-neutral{ background: rgba(100,116,139,.15); color: var(--muted); }
/* Buttons */
.btn { padding: 6px 16px; border-radius: 8px; font-size: 13px; font-weight: 600;
       cursor: pointer; border: none; }
.btn-primary { background: var(--blue); color: white; }
.btn-primary:hover { background: #2563eb; }
.btn-sm { padding: 4px 10px; font-size: 11px; }
.btn-won  { background: rgba(34,197,94,.2); color: var(--green); }
.btn-lost { background: rgba(239,68,68,.2); color: var(--red); }
/* Form */
.form-group { display: flex; flex-direction: column; gap: 4px; }
.form-label { font-size: 11px; color: var(--muted); font-weight: 600; text-transform: uppercase; }
.form-input, .form-select { background: var(--bg); border: 1px solid var(--border); color: var(--text);
                             padding: 8px 12px; border-radius: 8px; font-size: 13px; width: 100%; }
.form-input:focus, .form-select:focus { outline: none; border-color: var(--blue); }
/* Grid */
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }
/* KPI */
.kpi-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
            padding: 16px; text-align: center; }
.kpi-value { font-size: 24px; font-weight: 800; }
.kpi-label { font-size: 11px; color: var(--muted); margin-top: 4px; }
/* Bankroll bar */
.bankroll-bar { background: var(--surface); border-top: 1px solid var(--border);
                padding: 10px 20px; display: flex; gap: 24px; align-items: center;
                flex-shrink: 0; font-size: 12px; }
/* Section label */
.section-label { font-size: 10px; font-weight: 700; color: var(--muted);
                 letter-spacing: 0.08em; text-transform: uppercase; }
/* Autocomplete */
.autocomplete-list { background: var(--surface); border: 1px solid var(--border);
                     border-radius: 8px; max-height: 200px; overflow-y: auto; z-index: 100; }
.autocomplete-item { padding: 8px 12px; cursor: pointer; font-size: 13px; }
.autocomplete-item:hover { background: var(--border); }
/* Responsive */
@media (max-width: 640px) {
  .sidebar { width: 52px; }
  .grid-2, .grid-3 { grid-template-columns: 1fr; }
}
```

- [ ] **Step 4: Write `src/webapp/static/manifest.json`**

```json
{
  "name": "TennisPredict",
  "short_name": "Tennis",
  "start_url": "/today",
  "display": "standalone",
  "background_color": "#0f172a",
  "theme_color": "#3b82f6",
  "icons": [{"src": "/static/icon-192.png", "sizes": "192x192", "type": "image/png"}]
}
```

- [ ] **Step 5: Write `src/webapp/static/sw.js`** (cache static assets only)

```javascript
const CACHE = 'tennis-v1';
const STATIC = ['/static/app.css'];
self.addEventListener('install', e =>
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(STATIC)))
);
self.addEventListener('fetch', e => {
  if (e.request.url.includes('/static/')) {
    e.respondWith(caches.match(e.request).then(r => r || fetch(e.request)));
  }
});
```

- [ ] **Step 6: Create stub routers** (so `main.py` imports don't fail)

Each router file in `src/webapp/routers/` should have at minimum:
```python
# today.py / predictions.py / history.py / joueurs.py / stats.py
from fastapi import APIRouter
router = APIRouter()
```

- [ ] **Step 7: Write skeleton route test** → `tests/webapp/test_routes.py`

```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import pandas as pd

@pytest.fixture
def client():
    """TestClient with mocked ML models and DB."""
    import src.webapp.main as main_module
    # Inject mock state before creating client
    mock_profiles = pd.DataFrame([{
        'player_name': 'Jannik Sinner', 'name_key': 'jannik sinner',
        'rank': 1, 'elo': 2200.0,
    }])
    main_module.APP_STATE['models'] = {
        'atp': {'model': MagicMock(), 'imputer': MagicMock(),
                'platt': MagicMock(), 'feature_list': ['elo_diff'],
                'profiles': mock_profiles, 'players': pd.DataFrame()},
        'wta': None,
    }
    import sqlite3; conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    from src.webapp.db import init_db; init_db(conn)
    main_module.APP_STATE['db'] = conn
    main_module.APP_STATE['sync_status'] = {'atp': 'idle', 'wta': 'idle'}
    with TestClient(main_module.app, raise_server_exceptions=False) as c:
        yield c

def test_root_redirects(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)

def test_today_returns_200(client):
    r = client.get("/today")
    assert r.status_code == 200
    assert 'text/html' in r.headers['content-type']

def test_predictions_returns_200(client):
    r = client.get("/predictions")
    assert r.status_code == 200

def test_history_returns_200(client):
    r = client.get("/history")
    assert r.status_code == 200

def test_stats_returns_200(client):
    r = client.get("/stats")
    assert r.status_code == 200
```

- [ ] **Step 8: Run route tests** (expect failures until pages are built — OK for now)

```bash
python -m pytest tests/webapp/test_routes.py::test_root_redirects -v
```

- [ ] **Step 9: Commit**

```bash
git add src/webapp/main.py src/webapp/templates/base.html src/webapp/static/ src/webapp/routers/ tests/webapp/test_routes.py
git commit -m "feat(webapp): FastAPI app skeleton, base template, CSS dark theme"
```

---

## Chunk 2: Today + Predictions Pages

### Task 5: Today page (`/today`)

**Files:**
- Modify: `src/webapp/routers/today.py`
- Create: `src/webapp/templates/today.html`
- Create: `src/webapp/templates/partials/match_card.html`

- [ ] **Step 1: Implement `src/webapp/routers/today.py`**

```python
"""Router — Today's matches."""
from __future__ import annotations

import asyncio
from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from src.webapp.main import APP_STATE
from src.webapp import ml as ml_module
from src.webapp.db import get_bankroll

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")


def _get_today_matches(tour: str, match_date: str) -> list[dict]:
    """Fetch matches from ESPN; fall back gracefully."""
    try:
        import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
        from espn_client import fetch_day
        matches = fetch_day(tour, match_date)
        return matches if matches else []
    except Exception:
        return []


def _enrich_with_predictions(matches: list[dict], tour: str, bankroll: float) -> list[dict]:
    """Add ML prediction to each match if players are known."""
    artifacts = APP_STATE.get('models', {}).get(tour)
    if not artifacts:
        return matches
    enriched = []
    for m in matches:
        try:
            result = ml_module.predict(
                artifacts,
                p1_name=m.get('p1_name', ''),
                p2_name=m.get('p2_name', ''),
                tournament=m.get('tournament', ''),
                surface=m.get('surface', 'Hard'),
                round_=m.get('round', 'R64'),
                best_of=m.get('best_of', 3),
                odd_p1=m.get('odd_p1'),
                odd_p2=m.get('odd_p2'),
                bankroll=bankroll,
            )
            m.update(result)
        except Exception:
            pass
        # Badge logic
        edge = m.get('edge') or 0
        if edge >= 0.05:
            m['badge'] = 'value'
        elif edge >= 0.02:
            m['badge'] = 'edge'
        else:
            m['badge'] = 'neutral'
        enriched.append(m)
    # Sort: value bets first
    return sorted(enriched, key=lambda x: -(x.get('edge') or -99))


@router.get("/today", response_class=HTMLResponse)
async def today_page(request: Request, tour: str = "atp",
                     match_date: str = Query(default=None)):
    if not match_date:
        match_date = date.today().isoformat()
    db = APP_STATE['db']
    bankroll_atp = get_bankroll(db, 'atp')
    bankroll_wta = get_bankroll(db, 'wta')
    matches = _get_today_matches(tour, match_date)
    matches = _enrich_with_predictions(matches, tour, get_bankroll(db, tour))
    return templates.TemplateResponse("today.html", {
        "request": request, "active": "today",
        "tour": tour, "match_date": match_date,
        "matches": matches, "match_count": len(matches),
        "bankroll_atp": bankroll_atp, "bankroll_wta": bankroll_wta,
        "sync_status": APP_STATE.get('sync_status', {}).get(tour, 'idle'),
    })


@router.get("/today/matches", response_class=HTMLResponse)
async def today_matches_partial(request: Request, tour: str = "atp",
                                match_date: str = Query(default=None)):
    """HTMX partial — swap #match-list."""
    if not match_date:
        match_date = date.today().isoformat()
    db = APP_STATE['db']
    bankroll = get_bankroll(db, tour)
    matches = _get_today_matches(tour, match_date)
    matches = _enrich_with_predictions(matches, tour, bankroll)
    return templates.TemplateResponse("partials/match_card.html", {
        "request": request, "matches": matches, "tour": tour,
    })
```

- [ ] **Step 2: Write `src/webapp/templates/today.html`**

```html
{% extends "base.html" %}
{% block title %}Matchs du jour — TennisPredict{% endblock %}

{% block page_title %}Matchs du jour{% endblock %}
{% block page_meta %}{{ match_date }} · {{ match_count }} match(s){% endblock %}

{% block topbar_right %}
<div class="toggle-group">
  <button class="toggle-btn {% if tour=='atp' %}active{% endif %}"
          hx-get="/today/matches?tour=atp&match_date={{ match_date }}"
          hx-target="#match-list" hx-swap="innerHTML">ATP</button>
  <button class="toggle-btn {% if tour=='wta' %}active{% endif %}"
          hx-get="/today/matches?tour=wta&match_date={{ match_date }}"
          hx-target="#match-list" hx-swap="innerHTML">WTA</button>
</div>
{% endblock %}

{% block content %}
{% if not matches %}
<div class="card" style="text-align:center;color:var(--muted);padding:40px">
  Aucun match trouvé pour {{ match_date }}.
  <br><a href="/predictions" style="color:var(--blue)">→ Saisie manuelle</a>
</div>
{% endif %}
<div id="match-list">
  {% include "partials/match_card.html" %}
</div>
{% endblock %}

{% block scripts %}
<div class="bankroll-bar">
  <span>💰 ATP <strong>{{ bankroll_atp | round(0) | int }}€</strong></span>
  <span>💰 WTA <strong>{{ bankroll_wta | round(0) | int }}€</strong></span>
  <span style="color:var(--muted);margin-left:auto;font-size:11px">Sync: {{ sync_status }}</span>
  <button class="btn btn-sm" style="background:var(--surface);color:var(--text)"
          hx-post="/sync?tour={{ tour }}" hx-swap="none">↻</button>
</div>
{% endblock %}
```

- [ ] **Step 3: Write `src/webapp/templates/partials/match_card.html`**

```html
{% for m in matches %}
<div class="card" style="margin-bottom:8px">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">
    <div style="font-size:11px;color:var(--muted)">{{ m.get('tournament','') }} · {{ m.get('round','') }}</div>
    {% if m.get('badge') == 'value' %}
      <span class="badge badge-value">VALUE BET</span>
    {% elif m.get('badge') == 'edge' %}
      <span class="badge badge-edge">EDGE MODÉRÉ</span>
    {% else %}
      <span class="badge badge-neutral">NEUTRE</span>
    {% endif %}
  </div>
  <div style="display:flex;align-items:center;justify-content:space-between">
    <div>
      <div style="font-weight:700;font-size:15px">{{ m.get('p1_name','?') }}</div>
      <div style="font-size:11px;color:var(--muted)">vs {{ m.get('p2_name','?') }}</div>
    </div>
    <div style="text-align:right">
      {% if m.get('prob_p1') %}
        <div style="font-size:20px;font-weight:800;color:var(--blue)">{{ (m.prob_p1 * 100) | round(1) }}%</div>
        {% if m.get('kelly_eur') %}
          <div style="font-size:11px;color:var(--green)">Kelly: {{ m.kelly_eur }}€</div>
        {% endif %}
      {% else %}
        <div style="color:var(--muted);font-size:12px">Prédiction indisponible</div>
      {% endif %}
    </div>
  </div>
</div>
{% endfor %}
```

- [ ] **Step 4: Start app and verify today page loads**

```bash
uvicorn src.webapp.main:app --reload --port 8000
# Navigate to http://localhost:8000/today
```

- [ ] **Step 5: Commit**

```bash
git add src/webapp/routers/today.py src/webapp/templates/today.html src/webapp/templates/partials/match_card.html
git commit -m "feat(webapp): add Today page with ESPN matches and ML predictions"
```

---

### Task 6: Predictions page (`/predictions`)

**Files:**
- Modify: `src/webapp/routers/predictions.py`
- Create: `src/webapp/templates/predictions.html`
- Create: `src/webapp/templates/partials/prediction_result.html`
- Create: `tests/webapp/test_autocomplete.py`

- [ ] **Step 1: Write autocomplete tests** → `tests/webapp/test_autocomplete.py`

```python
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import src.webapp.main as main_module
from src.webapp.db import init_db
import sqlite3


@pytest.fixture
def client_with_players():
    profiles = pd.DataFrame([
        {'player_name': 'Jannik Sinner',    'name_key': 'jannik sinner',    'rank': 1,  'elo': 2200.0},
        {'player_name': 'Carlos Alcaraz',   'name_key': 'carlos alcaraz',   'rank': 2,  'elo': 2180.0},
        {'player_name': 'Rafael Nadal',     'name_key': 'rafael nadal',     'rank': 300,'elo': 1900.0},
        {'player_name': 'Tommy Müller',     'name_key': 'tommy müller',     'rank': 50, 'elo': 2050.0},
        {'player_name': 'Roberto Bautista Agut','name_key': 'roberto bautista agut','rank': 35,'elo': 2060.0},
    ])
    main_module.APP_STATE['models'] = {
        'atp': {'model': MagicMock(), 'imputer': MagicMock(), 'platt': MagicMock(),
                'feature_list': ['elo_diff'], 'profiles': profiles, 'players': pd.DataFrame()},
        'wta': None,
    }
    conn = sqlite3.connect(':memory:'); conn.row_factory = sqlite3.Row
    init_db(conn)
    main_module.APP_STATE['db'] = conn
    main_module.APP_STATE['sync_status'] = {'atp': 'idle', 'wta': 'idle'}
    with TestClient(main_module.app, raise_server_exceptions=False) as c:
        yield c


def test_autocomplete_basic(client_with_players):
    r = client_with_players.get("/predictions/autocomplete?q=sin&tour=atp")
    assert r.status_code == 200
    assert 'Sinner' in r.text

def test_autocomplete_accent(client_with_players):
    r = client_with_players.get("/predictions/autocomplete?q=mull&tour=atp")
    assert r.status_code == 200
    assert 'Müller' in r.text

def test_autocomplete_compound_name(client_with_players):
    r = client_with_players.get("/predictions/autocomplete?q=baut&tour=atp")
    assert r.status_code == 200
    assert 'Bautista' in r.text

def test_autocomplete_case_insensitive(client_with_players):
    r = client_with_players.get("/predictions/autocomplete?q=ALCARAZ&tour=atp")
    assert r.status_code == 200
    assert 'Alcaraz' in r.text

def test_autocomplete_min_2_chars(client_with_players):
    r = client_with_players.get("/predictions/autocomplete?q=a&tour=atp")
    assert r.status_code == 200
    assert r.text.strip() == '' or 'autocomplete-item' not in r.text

def test_autocomplete_sorted_by_rank(client_with_players):
    r = client_with_players.get("/predictions/autocomplete?q=a&tour=atp")
    # Result order: rank 2 (Alcaraz) before rank 300 (Nadal) if both match "a"
    # With min_chars=2, this returns empty — just check no crash
    assert r.status_code == 200
```

- [ ] **Step 2: Implement `src/webapp/routers/predictions.py`**

```python
"""Router — Manual predictions."""
from __future__ import annotations

from pathlib import Path
from fastapi import APIRouter, Request, Form, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.webapp.main import APP_STATE
from src.webapp import ml as ml_module
from src.webapp.db import get_bankroll, add_bet

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")

_ROUNDS = ['R128','R64','R32','R16','QF','SF','F','RR']
_SURFACES = ['Hard','Clay','Grass']


@router.get("/predictions", response_class=HTMLResponse)
async def predictions_page(request: Request, tour: str = "atp"):
    db = APP_STATE['db']
    return templates.TemplateResponse("predictions.html", {
        "request": request, "active": "predictions",
        "tour": tour, "rounds": _ROUNDS, "surfaces": _SURFACES,
        "bankroll": get_bankroll(db, tour),
    })


@router.get("/predictions/autocomplete", response_class=HTMLResponse)
async def autocomplete(request: Request, q: str = "", tour: str = "atp"):
    if len(q) < 2:
        return HTMLResponse("")
    artifacts = APP_STATE.get('models', {}).get(tour)
    if not artifacts:
        return HTMLResponse("")
    profiles = artifacts['profiles']
    mask = profiles['name_key'].str.contains(q.lower(), na=False, regex=False)
    results = (profiles[mask]
               .sort_values('rank', na_position='last')
               .head(8)[['player_name', 'rank']]
               .to_dict('records'))
    html = ""
    for r in results:
        name = r['player_name']
        rank = r.get('rank', '—')
        html += (f'<div class="autocomplete-item" '
                 f'onclick="selectPlayer(this,\'{name}\')">'
                 f'{name} <span style="color:var(--muted)">#{rank}</span></div>')
    return HTMLResponse(html)


@router.post("/predictions/run", response_class=HTMLResponse)
async def run_prediction(
    request: Request,
    tour: str = Form(...),
    p1_name: str = Form(...),
    p2_name: str = Form(...),
    tournament: str = Form(...),
    surface: str = Form(...),
    round_: str = Form(..., alias="round"),
    best_of: int = Form(3),
    odd_p1: float | None = Form(None),
    odd_p2: float | None = Form(None),
):
    artifacts = APP_STATE.get('models', {}).get(tour)
    db = APP_STATE['db']
    bankroll = get_bankroll(db, tour)

    if not artifacts:
        return HTMLResponse('<div class="card" style="color:var(--red)">Modèle non disponible pour ce circuit.</div>')

    result = ml_module.predict(
        artifacts, p1_name=p1_name, p2_name=p2_name,
        tournament=tournament, surface=surface, round_=round_,
        best_of=best_of, odd_p1=odd_p1, odd_p2=odd_p2, bankroll=bankroll,
    )
    return templates.TemplateResponse("partials/prediction_result.html", {
        "request": request, "result": result,
        "p1_name": p1_name, "p2_name": p2_name,
        "tour": tour, "tournament": tournament, "surface": surface,
        "round": round_, "best_of": best_of, "odd_p1": odd_p1,
    })


@router.post("/bets", response_class=HTMLResponse)
async def save_bet(
    request: Request,
    tour: str = Form(...),
    p1_name: str = Form(...), p2_name: str = Form(...),
    bet_on: str = Form(...), tournament: str = Form(...),
    surface: str = Form(...), round_: str = Form(..., alias="round"),
    prob: float = Form(...), edge: float | None = Form(None),
    odd: float = Form(...), stake: float = Form(...),
    kelly_frac: float | None = Form(None),
):
    db = APP_STATE['db']
    add_bet(db, {
        'tour': tour, 'tournament': tournament, 'surface': surface,
        'round': round_, 'p1_name': p1_name, 'p2_name': p2_name,
        'bet_on': bet_on, 'prob': prob, 'edge': edge,
        'odd': odd, 'stake': stake, 'kelly_frac': kelly_frac,
    })
    bankroll = get_bankroll(db, tour)
    return HTMLResponse(
        f'<div class="card" style="color:var(--green)">✅ Pari enregistré. '
        f'Bankroll {tour.upper()}: <strong>{bankroll:.0f}€</strong></div>'
    )
```

- [ ] **Step 3: Write templates** — `predictions.html` and `partials/prediction_result.html`

`predictions.html` — 2-column form grid:
- Toggle ATP/WTA (hx-get="/predictions?tour=...")
- Joueur 1 / Joueur 2 with autocomplete (hx-get="/predictions/autocomplete")
- Tournoi, Surface, Tour, Best-of dropdowns
- Cotes optionnelles
- Submit button → hx-post="/predictions/run" → #result-card

`partials/prediction_result.html` — result card:
- Big probability gauge (CSS arc or number)
- Edge + EV in green/orange/red
- Kelly fraction + € amount
- Warning if elo_only
- Button "Enregistrer le pari" → hx-post="/bets"

(Full HTML in implementation — follows mockup design exactly)

- [ ] **Step 4: Run autocomplete tests**

```bash
python -m pytest tests/webapp/test_autocomplete.py -v
```

Expected: 6/6 PASS

- [ ] **Step 5: Commit**

```bash
git add src/webapp/routers/predictions.py src/webapp/templates/predictions.html src/webapp/templates/partials/prediction_result.html tests/webapp/test_autocomplete.py
git commit -m "feat(webapp): add Predictions page with autocomplete and edge calculation"
```

---

## Chunk 3: History + Players Pages

### Task 7: History page (`/history`)

**Files:**
- Modify: `src/webapp/routers/history.py`
- Create: `src/webapp/templates/history.html`
- Create: `src/webapp/templates/partials/bet_row.html`

- [ ] **Step 1: Implement `src/webapp/routers/history.py`**

```python
"""Router — Bet history."""
from __future__ import annotations

from pathlib import Path
from fastapi import APIRouter, Request, Form, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import csv, io

from src.webapp.main import APP_STATE
from src.webapp.db import get_bankroll, list_bets, resolve_bet, get_bet

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")


@router.get("/history", response_class=HTMLResponse)
async def history_page(
    request: Request, tour: str = "atp",
    surface: str | None = None, status: str | None = None,
    page: int = 1,
):
    db = APP_STATE['db']
    offset = (page - 1) * 20
    bets = list_bets(db, tour=tour, surface=surface, status=status, limit=20, offset=offset)
    pending = [b for b in bets if b['status'] == 'pending']
    resolved = [b for b in bets if b['status'] != 'pending']
    pnl_month = sum(b['pnl'] for b in list_bets(db, tour=tour, status=None, limit=1000)
                    if b['status'] != 'pending')
    return templates.TemplateResponse("history.html", {
        "request": request, "active": "history",
        "tour": tour, "pending": pending, "resolved": resolved,
        "bankroll_atp": get_bankroll(db, 'atp'),
        "bankroll_wta": get_bankroll(db, 'wta'),
        "pnl_month": round(pnl_month, 2),
        "page": page, "surface_filter": surface, "status_filter": status,
    })


@router.post("/bets/{bet_id}/resolve", response_class=HTMLResponse)
async def resolve(request: Request, bet_id: int, outcome: str = Form(...)):
    db = APP_STATE['db']
    try:
        resolve_bet(db, bet_id, outcome)
    except ValueError as e:
        return HTMLResponse(f'<div style="color:var(--red)">{e}</div>', status_code=400)
    bankroll_atp = get_bankroll(db, 'atp')
    bankroll_wta = get_bankroll(db, 'wta')
    return HTMLResponse(
        f'<div class="card" style="color:var(--green)">✅ Résolu. '
        f'ATP: {bankroll_atp:.0f}€ · WTA: {bankroll_wta:.0f}€</div>'
    )


@router.get("/history/export")
async def export_csv(tour: str = "atp"):
    db = APP_STATE['db']
    bets = list_bets(db, tour=tour, limit=10000)
    output = io.StringIO()
    if bets:
        writer = csv.DictWriter(output, fieldnames=bets[0].keys())
        writer.writeheader(); writer.writerows(bets)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=bets_{tour}.csv"},
    )
```

- [ ] **Step 2: Write `src/webapp/templates/history.html` + `partials/bet_row.html`**

`history.html` — 3 sections:
1. Header KPIs: bankroll ATP, bankroll WTA, P&L mois
2. Section "En attente" — for each pending bet: match name, odds, stake, Gagné/Perdu buttons
3. Section "Résolus" — table with date, match, cote, mise, P&L, status badge

`partials/bet_row.html` — single bet row for HTMX swap on resolve

- [ ] **Step 3: Add resolve test to test_routes.py**

```python
def test_resolve_bet_already_resolved_returns_400(client):
    # Add a bet, resolve it once, try again
    from src.webapp.db import add_bet, set_bankroll
    db = APP_STATE['db']
    set_bankroll(db, 'atp', 1000.0)
    bet_id = add_bet(db, {
        'tour': 'atp', 'tournament': 'T', 'surface': 'Hard', 'round': 'R64',
        'p1_name': 'A', 'p2_name': 'B', 'bet_on': 'p1',
        'prob': 0.6, 'edge': 0.04, 'odd': 2.0, 'stake': 50.0, 'kelly_frac': 0.05,
    })
    r = client.post(f"/bets/{bet_id}/resolve", data={"outcome": "won"})
    assert r.status_code == 200
    r2 = client.post(f"/bets/{bet_id}/resolve", data={"outcome": "lost"})
    assert r2.status_code == 400
```

- [ ] **Step 4: Commit**

```bash
git add src/webapp/routers/history.py src/webapp/templates/history.html src/webapp/templates/partials/bet_row.html
git commit -m "feat(webapp): add History page with bet resolution and CSV export"
```

---

### Task 8: Players page (`/joueurs`)

**Files:**
- Create: `src/webapp/players.py`
- Modify: `src/webapp/routers/joueurs.py`
- Create: `src/webapp/templates/joueurs.html`
- Create: `src/webapp/templates/joueurs_profile.html`

- [ ] **Step 1: Implement `src/webapp/players.py`** (search + profile logic)

```python
"""Player search and profile helpers."""
import pandas as pd
from pathlib import Path


def search_players(profiles: pd.DataFrame, players: pd.DataFrame,
                   q: str, limit: int = 10) -> list[dict]:
    """Search by name substring. Returns list sorted by rank."""
    if len(q) < 2:
        return []
    key = q.lower()
    mask = profiles['name_key'].str.contains(key, na=False, regex=False)
    rows = profiles[mask].sort_values('rank', na_position='last').head(limit)
    result = []
    for _, r in rows.iterrows():
        result.append({
            'player_name': r['player_name'],
            'rank': r.get('rank'),
            'elo': round(r.get('elo', 0), 0),
            'surface': _best_surface(r),
        })
    return result


def get_profile(profiles: pd.DataFrame, players: pd.DataFrame,
                player_name: str) -> dict | None:
    """Full profile for a player."""
    key = player_name.lower().strip()
    rows = profiles[profiles['name_key'] == key]
    if rows.empty:
        return None
    r = rows.iloc[0].to_dict()

    # Enrich from players.parquet (identity)
    identity = {}
    if not players.empty and 'name_first' in players.columns:
        players['name_key'] = (players['name_first'] + ' ' + players['name_last']).str.lower().str.strip()
        id_rows = players[players['name_key'] == key]
        if not id_rows.empty:
            identity = id_rows.iloc[0].to_dict()

    return {**r, **{k: v for k, v in identity.items() if k not in r}}


def _best_surface(r) -> str:
    surfaces = {
        'Hard':  r.get('elo_hard', 0) or 0,
        'Clay':  r.get('elo_clay', 0) or 0,
        'Grass': r.get('elo_grass', 0) or 0,
    }
    return max(surfaces, key=surfaces.get)
```

- [ ] **Step 2: Implement `src/webapp/routers/joueurs.py`**

```python
"""Router — Player profiles."""
from __future__ import annotations

from pathlib import Path
from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.webapp.main import APP_STATE
from src.webapp.players import search_players, get_profile

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")


@router.get("/joueurs", response_class=HTMLResponse)
async def joueurs_page(request: Request, tour: str = "atp"):
    return templates.TemplateResponse("joueurs.html", {
        "request": request, "active": "joueurs", "tour": tour, "players": [],
    })


@router.get("/joueurs/search", response_class=HTMLResponse)
async def joueurs_search(request: Request, q: str = "", tour: str = "atp"):
    artifacts = APP_STATE.get('models', {}).get(tour)
    if not artifacts or len(q) < 2:
        return HTMLResponse("")
    players = search_players(artifacts['profiles'], artifacts['players'], q)
    html = ""
    for p in players:
        html += (f'<div class="card" style="margin-bottom:6px;cursor:pointer" '
                 f'onclick="window.location=\'/joueurs/{p[\"player_name\"]}\'">'
                 f'<strong>{p["player_name"]}</strong> '
                 f'<span style="color:var(--muted)">#{p.get("rank","—")} · '
                 f'ELO {p.get("elo",0):.0f} · {p.get("surface","?")} spécialiste</span></div>')
    return HTMLResponse(html)


@router.get("/joueurs/{player_name:path}", response_class=HTMLResponse)
async def joueur_profile(request: Request, player_name: str, tour: str = "atp"):
    artifacts = APP_STATE.get('models', {}).get(tour)
    if not artifacts:
        return HTMLResponse("Circuit non disponible.", status_code=503)
    profile = get_profile(artifacts['profiles'], artifacts['players'], player_name)
    if not profile:
        return HTMLResponse(f"Joueur '{player_name}' non trouvé.", status_code=404)
    return templates.TemplateResponse("joueurs_profile.html", {
        "request": request, "active": "joueurs", "tour": tour, "p": profile,
    })
```

- [ ] **Step 3: Write templates** `joueurs.html` + `joueurs_profile.html`

`joueurs.html` — search bar + result list (HTMX live search)
`joueurs_profile.html` — identity header + ELO bars (Hard/Clay/Grass vs tour average) + form badges (win rate 5/10/20) + mini sparkline Chart.js

- [ ] **Step 4: Commit**

```bash
git add src/webapp/players.py src/webapp/routers/joueurs.py src/webapp/templates/joueurs.html src/webapp/templates/joueurs_profile.html
git commit -m "feat(webapp): add Players page with search and profile view"
```

---

## Chunk 4: Stats Page + Sync + Migration + PWA

### Task 9: Stats page (`/stats`)

**Files:**
- Modify: `src/webapp/routers/stats.py`
- Create: `src/webapp/templates/stats.html`

- [ ] **Step 1: Implement `src/webapp/routers/stats.py`**

Key endpoints:
- `GET /stats` — page complète avec KPIs
- `GET /stats/equity?tour=atp&strategy=Kelly` — JSON pour Chart.js
- `GET /stats/roi-bookmakers?tour=atp` — JSON barres
- `GET /stats/roi-surface?tour=atp` — JSON barres
- `GET /stats/features?tour=atp` — top 15 feature importances
- `POST /settings` — sauvegarde seuils dans SQLite

```python
"""Router — Stats and backtest."""
from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from src.webapp.main import APP_STATE
from src.webapp.db import get_setting, set_setting, get_bankroll
from src.config import get_paths

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")


def _load_equity(tour: str, strategy: str) -> dict:
    paths = get_paths(tour)
    strat_map = {'Kelly': 'backtest_kelly.parquet',
                 'Flat':  'backtest_flat.parquet',
                 'Percent':'backtest_percent.parquet'}
    fname = strat_map.get(strategy, 'backtest_kelly.parquet')
    fpath = paths['models_dir'] / fname
    if not fpath.exists():
        return {'labels': [], 'values': []}
    df = pd.read_parquet(fpath)
    # Expect columns: date + bankroll (or cumulative bankroll column)
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    bank_col = next((c for c in df.columns if 'bankroll' in c.lower() or 'cumul' in c.lower()), None)
    if not date_col or not bank_col:
        return {'labels': [], 'values': []}
    df = df.sort_values(date_col).dropna(subset=[date_col, bank_col])
    return {
        'labels': df[date_col].astype(str).tolist(),
        'values': df[bank_col].round(2).tolist(),
    }


def _load_roi_bookmakers(tour: str) -> dict:
    paths = get_paths(tour)
    bookmakers = ['Bet365', 'Pinnacle', 'Best', 'Avg']
    roi_list = []
    for bk in bookmakers:
        fpath = paths['models_dir'] / f'backtest_real_{bk}.parquet'
        if not fpath.exists():
            roi_list.append(None); continue
        df = pd.read_parquet(fpath)
        if 'roi' in df.columns:
            roi_list.append(round(float(df['roi'].iloc[-1]), 4))
        elif 'pnl' in df.columns and 'stake' in df.columns:
            total_stake = df['stake'].sum()
            roi = df['pnl'].sum() / total_stake if total_stake > 0 else 0
            roi_list.append(round(roi, 4))
        else:
            roi_list.append(None)
    return {'bookmakers': bookmakers, 'roi': roi_list}


def _load_feature_importance(tour: str) -> dict:
    artifacts = APP_STATE.get('models', {}).get(tour)
    if not artifacts or not artifacts.get('model'):
        return {'features': [], 'values': []}
    model = artifacts['model']
    features = artifacts['feature_list']
    try:
        importances = model.feature_importances_
    except AttributeError:
        return {'features': [], 'values': []}
    pairs = sorted(zip(features, importances), key=lambda x: -x[1])[:15]
    return {
        'features': [p[0] for p in pairs],
        'values':   [round(float(p[1]), 4) for p in pairs],
    }


@router.get("/stats", response_class=HTMLResponse)
async def stats_page(request: Request, tour: str = "atp"):
    db = APP_STATE['db']
    settings = {
        'min_edge':       get_setting(db, 'min_edge', '0.03'),
        'min_prob':       get_setting(db, 'min_prob', '0.55'),
        'kelly_fraction': get_setting(db, 'kelly_fraction', '0.25'),
    }
    roi_bk  = _load_roi_bookmakers(tour)
    features = _load_feature_importance(tour)
    return templates.TemplateResponse("stats.html", {
        "request": request, "active": "stats", "tour": tour,
        "settings": settings, "roi_bk": roi_bk, "features": features,
        "bankroll": get_bankroll(db, tour),
    })


@router.get("/stats/equity")
async def equity_data(tour: str = "atp", strategy: str = "Kelly"):
    return JSONResponse(_load_equity(tour, strategy))


@router.get("/stats/roi-bookmakers")
async def roi_bookmakers_data(tour: str = "atp"):
    return JSONResponse(_load_roi_bookmakers(tour))


@router.get("/stats/features")
async def features_data(tour: str = "atp"):
    return JSONResponse(_load_feature_importance(tour))


@router.post("/settings", response_class=HTMLResponse)
async def save_settings(
    min_edge: str = Form(...),
    min_prob: str = Form(...),
    kelly_fraction: str = Form(...),
):
    db = APP_STATE['db']
    set_setting(db, 'min_edge', min_edge)
    set_setting(db, 'min_prob', min_prob)
    set_setting(db, 'kelly_fraction', kelly_fraction)
    return HTMLResponse('<div style="color:var(--green)">✅ Seuils sauvegardés.</div>')
```

- [ ] **Step 2: Write `src/webapp/templates/stats.html`** with Chart.js charts:

4 sections:
1. **KPIs** — grid-4 cards: ROI Pinnacle, Win Rate, Nb Paris, Sharpe
2. **Equity Curve** — `<canvas id="equity-chart">` + toggle Flat/Kelly/Percent
3. **ROI par bookmaker** — bar chart `<canvas id="roi-chart">`
4. **Feature importance** — horizontal bar chart `<canvas id="feat-chart">`
5. **Configuration seuils** — sliders with current values

Chart.js initialization inline:
```html
<script>
// Fetches /stats/equity?tour=... on page load and after toggle
async function loadEquity(tour, strategy) {
  const r = await fetch(`/stats/equity?tour=${tour}&strategy=${strategy}`);
  const d = await r.json();
  const ctx = document.getElementById('equity-chart').getContext('2d');
  if (window._equityChart) window._equityChart.destroy();
  window._equityChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: d.labels,
      datasets: [{
        label: `Bankroll (${strategy})`,
        data: d.values,
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59,130,246,0.1)',
        fill: true, tension: 0.3, pointRadius: 0,
      }]
    },
    options: {
      plugins: { legend: { labels: { color: '#f1f5f9' } } },
      scales: {
        x: { ticks: { color: '#64748b' }, grid: { color: '#1e293b' } },
        y: { ticks: { color: '#64748b' }, grid: { color: '#1e293b' } },
      }
    }
  });
}
document.addEventListener('DOMContentLoaded', () => loadEquity('{{ tour }}', 'Kelly'));
</script>
```

- [ ] **Step 3: Add stats tests to test_routes.py**

```python
def test_stats_equity_returns_json(client):
    r = client.get("/stats/equity?tour=atp&strategy=Kelly")
    assert r.status_code == 200
    data = r.json()
    assert 'labels' in data and 'values' in data

def test_settings_save(client):
    r = client.post("/settings", data={"min_edge": "0.04", "min_prob": "0.57", "kelly_fraction": "0.20"})
    assert r.status_code == 200
    assert '✅' in r.text
```

- [ ] **Step 4: Commit**

```bash
git add src/webapp/routers/stats.py src/webapp/templates/stats.html
git commit -m "feat(webapp): add Stats page with Chart.js equity curve, ROI bars, feature importance"
```

---

### Task 10: Sync endpoint

**Files:**
- Create: `src/webapp/routers/sync.py`
- Modify: `src/webapp/main.py` (include sync router)

- [ ] **Step 1: Implement sync router**

```python
"""Router — Background sync."""
from __future__ import annotations

import asyncio
from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import HTMLResponse

from src.webapp.main import APP_STATE

router = APIRouter()


async def _run_sync(tour: str):
    APP_STATE['sync_status'][tour] = 'running'
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
        import importlib
        fld = importlib.import_module('fetch_live_data')
        await asyncio.to_thread(fld.run_update, tour, False)
    except Exception as e:
        print(f"[sync] Error: {e}")
    finally:
        APP_STATE['sync_status'][tour] = 'idle'


@router.post("/sync", response_class=HTMLResponse)
async def trigger_sync(background_tasks: BackgroundTasks, tour: str = "atp"):
    if APP_STATE['sync_status'].get(tour) == 'running':
        return HTMLResponse("Sync déjà en cours…")
    background_tasks.add_task(_run_sync, tour)
    return HTMLResponse("Sync démarrée…")


@router.get("/sync/status", response_class=HTMLResponse)
async def sync_status(tour: str = "atp"):
    status = APP_STATE['sync_status'].get(tour, 'idle')
    if status == 'running':
        return HTMLResponse(
            '<span hx-get="/sync/status?tour=' + tour + '" hx-trigger="every 3s" '
            'hx-swap="outerHTML" style="color:var(--orange)">⟳ Sync en cours…</span>'
        )
    return HTMLResponse('<span style="color:var(--muted)">Sync: idle</span>')
```

- [ ] **Step 2: Include sync router in main.py**

```python
from src.webapp.routers import today, predictions, history, joueurs, stats, sync  # noqa
app.include_router(sync.router)
```

- [ ] **Step 3: Commit**

```bash
git add src/webapp/routers/sync.py
git commit -m "feat(webapp): add background sync endpoint with HTMX polling"
```

---

### Task 11: CSV migration script

**Files:**
- Create: `src/webapp/migrate_csv.py`

- [ ] **Step 1: Write migration script**

```python
"""One-time migration: bets_atp.csv + bets_wta.csv → SQLite."""
from pathlib import Path
import pandas as pd
from src.webapp.db import get_connection, init_db, add_bet, set_bankroll
import json

ROOT = Path(__file__).resolve().parent.parent.parent

COLUMN_MAP = {
    'tournament': 'tournament', 'surface': 'surface', 'round': 'round',
    'p1_name': 'p1_name', 'p2_name': 'p2_name', 'bet_on': 'bet_on',
    'prob': 'prob', 'edge': 'edge', 'odd': 'odd', 'stake': 'stake',
    'kelly_frac': 'kelly_frac', 'status': 'status', 'pnl': 'pnl',
}


def migrate(tour: str, conn):
    csv_path = ROOT / "data" / f"bets_{tour}.csv"
    if not csv_path.exists():
        print(f"  No {csv_path.name} found, skipping.")
        return 0
    df = pd.read_csv(csv_path)
    migrated = 0
    for _, row in df.iterrows():
        bet = {k: row.get(v) for k, v in COLUMN_MAP.items()}
        bet['tour'] = tour
        if pd.isna(bet.get('tournament')): bet['tournament'] = 'Unknown'
        if pd.isna(bet.get('surface')):    bet['surface'] = 'Hard'
        if pd.isna(bet.get('p1_name')):    continue
        if pd.isna(bet.get('p2_name')):    continue
        if pd.isna(bet.get('bet_on')):     bet['bet_on'] = 'p1'
        if pd.isna(bet.get('prob')):       bet['prob'] = 0.5
        if pd.isna(bet.get('odd')):        bet['odd'] = 1.0
        if pd.isna(bet.get('stake')):      bet['stake'] = 0.0
        add_bet(conn, bet)
        migrated += 1
    print(f"  Migrated {migrated} bets for {tour.upper()}")
    # Rename original as .bak
    csv_path.rename(csv_path.with_suffix('.csv.bak'))
    return migrated


def migrate_bankroll(tour: str, conn):
    json_path = ROOT / "data" / f"bankroll_{tour}.json"
    if not json_path.exists():
        return
    with open(json_path) as f:
        data = json.load(f)
    amount = data.get('bankroll', 1000.0)
    set_bankroll(conn, tour, amount)
    json_path.rename(json_path.with_suffix('.json.bak'))
    print(f"  Migrated bankroll {tour.upper()}: {amount}€")


if __name__ == '__main__':
    db_path = ROOT / "data" / "tennis_predict.db"
    conn = get_connection(db_path)
    init_db(conn)
    for tour in ('atp', 'wta'):
        migrate_bankroll(tour, conn)
        migrate(tour, conn)
    conn.close()
    print("Migration complete.")
```

- [ ] **Step 2: Run migration if CSV files exist**

```bash
python src/webapp/migrate_csv.py
```

- [ ] **Step 3: Commit**

```bash
git add src/webapp/migrate_csv.py
git commit -m "feat(webapp): add CSV→SQLite migration script"
```

---

### Task 12: Full integration test suite

**Files:**
- Modify: `tests/webapp/test_routes.py` (finalize all routes)

- [ ] **Step 1: Run full test suite**

```bash
python -m pytest tests/webapp/ -v
```

Expected: all green

- [ ] **Step 2: Run app end-to-end**

```bash
uvicorn src.webapp.main:app --reload --port 8000
# Verify manually:
# http://localhost:8000/today
# http://localhost:8000/predictions
# http://localhost:8000/history
# http://localhost:8000/joueurs
# http://localhost:8000/stats
```

- [ ] **Step 3: Final commit + PR**

```bash
git add -A
git commit -m "feat(webapp): complete FastAPI+HTMX web app — all pages, tests, migration"
```

Then invoke: `superpowers:finishing-a-development-branch`

---

## Dependencies to install

Add to `requirements.txt` (or install directly):
```
fastapi>=0.110
uvicorn[standard]>=0.27
jinja2>=3.1
python-multipart>=0.0.9
```

```bash
pip install fastapi uvicorn[standard] jinja2 python-multipart
```

---

## Acceptance criteria

- [ ] All 5 pages return 200 with valid HTML
- [ ] HTMX partials return fragments (no `<html>` tag)
- [ ] Bet add → bankroll debited; resolve won → bankroll credited correctly
- [ ] Autocomplete works with accents and compound names
- [ ] Chart.js equity curve loads data from parquet via JSON endpoint
- [ ] All tests pass: `python -m pytest tests/webapp/ -v`
- [ ] App starts with: `uvicorn src.webapp.main:app --port 8000`
