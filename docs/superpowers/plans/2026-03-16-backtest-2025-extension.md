# 2025 Backtest Extension — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Inject ~2,491 completed 2025 ATP matches from `data/odds/atp/atp_2025.xlsx` into `matches_consolidated.parquet`, re-run the feature pipeline, and execute `backtest_real.py` against TEST_YEARS=[2025] for a true out-of-sample evaluation — no retraining.

**Architecture:** A new script `src/inject_2025_data.py` translates tennis-data.co.uk format to the Sackmann `matches_consolidated.parquet` schema (name translation, ID lookup, round/level mapping, synthetic IDs for unmatched players). The full feature pipeline is then re-run (`--tour atp`) to populate `X_test`. A one-line change to `backtest_real.py` targets TEST_YEARS=[2025].

**Tech Stack:** Python 3.x, pandas, openpyxl, joblib, pytest; existing pipeline scripts and `update_database.py` helper functions reused directly.

---

## Chunk 1: Script Design and Helper Functions

### File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/inject_2025_data.py` | Translate + inject 2025 tennis-data rows into consolidated parquet |
| Create | `tests/test_inject_2025_data.py` | Unit tests (no network, mock XLSX) |
| Modify | `src/backtest_real.py` line 531 | Change TEST_YEARS to [2025] |

### Task 1: Bootstrap — failing test for `_synthetic_id` and `_build_player_name_to_id`

**Files:**
- Create: `tests/test_inject_2025_data.py`

- [ ] **Step 1.1: Write the failing tests**

```python
# tests/test_inject_2025_data.py
"""Tests for inject_2025_data.py — no network required."""
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestSyntheticId:
    """_synthetic_id returns a stable int in [1_000_001, 2_900_000]."""

    def test_returns_int_in_range(self):
        from inject_2025_data import _synthetic_id
        sid = _synthetic_id("Random Player")
        assert isinstance(sid, int)
        assert 1_000_001 <= sid <= 2_900_000

    def test_deterministic(self):
        from inject_2025_data import _synthetic_id
        assert _synthetic_id("Sinner J.") == _synthetic_id("Sinner J.")

    def test_different_names_different_ids(self):
        from inject_2025_data import _synthetic_id
        assert _synthetic_id("Alpha") != _synthetic_id("Beta")

    def test_case_insensitive(self):
        """'sinner j.' and 'SINNER J.' should produce the same ID."""
        from inject_2025_data import _synthetic_id
        assert _synthetic_id("sinner j.") == _synthetic_id("SINNER J.")


class TestBuildPlayerNameToId:
    """_build_player_name_to_id parses atp_players.csv correctly."""

    def _make_csv(self, tmp_path) -> Path:
        csv_path = tmp_path / "atp_players.csv"
        csv_path.write_text(
            "player_id,name_first,name_last,hand,dob,ioc,height,wikidata_id\n"
            "207989,Jannik,Sinner,R,20011116.0,ITA,188.0,Q123\n"
            "104925,Novak,Djokovic,R,19870522.0,SRB,188.0,Q456\n"
            "invalid,,name,R,,,,\n"
        )
        return csv_path

    def test_returns_full_name_to_id(self, tmp_path):
        from inject_2025_data import _build_player_name_to_id
        csv_path = self._make_csv(tmp_path)
        mapping = _build_player_name_to_id(csv_path)
        assert mapping["Jannik Sinner"] == 207989
        assert mapping["Novak Djokovic"] == 104925

    def test_skips_invalid_id_rows(self, tmp_path):
        from inject_2025_data import _build_player_name_to_id
        csv_path = self._make_csv(tmp_path)
        mapping = _build_player_name_to_id(csv_path)
        assert all(isinstance(v, int) for v in mapping.values())

    def test_returns_dict(self, tmp_path):
        from inject_2025_data import _build_player_name_to_id
        csv_path = self._make_csv(tmp_path)
        result = _build_player_name_to_id(csv_path)
        assert isinstance(result, dict)
```

- [ ] **Step 1.2: Run tests to verify they fail (module not yet created)**

```bash
cd E:/Claude/botbet/tennis/tennis_ml
venv/Scripts/python -m pytest tests/test_inject_2025_data.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'inject_2025_data'`

- [ ] **Step 1.3: Create `src/inject_2025_data.py` with the two helper functions**

```python
# src/inject_2025_data.py
"""
inject_2025_data.py
-------------------
Injects 2025 ATP match data from tennis-data.co.uk into
matches_consolidated.parquet so the feature pipeline can produce
a non-empty X_test (year >= 2025) for backtest_real.py.

Usage:
    venv/Scripts/python src/inject_2025_data.py --tour atp

Idempotent: exits early if year==2025 rows already exist.
Always creates a .bak2024 backup before modifying the parquet.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from config import get_tour_config, get_paths, make_dirs
from update_database import load_new_matches, build_name_mapping


# ── Round mapping: tennis-data → Sackmann ───────────────────────────────────
ROUND_MAP = {
    '1st Round'      : 'R64',
    '2nd Round'      : 'R32',
    '3rd Round'      : 'R16',
    '4th Round'      : 'R16',
    'Quarterfinals'  : 'QF',
    'Semifinals'     : 'SF',
    'The Final'      : 'F',
    '1st Qualifying' : 'Q1',
    '2nd Qualifying' : 'Q2',
    'Round Robin'    : 'RR',
}

# ── Surface normalisation ────────────────────────────────────────────────────
SURFACE_MAP = {
    'Hard'        : 'Hard',
    'Clay'        : 'Clay',
    'Grass'       : 'Grass',
    'Indoor Hard' : 'Hard',
    'Carpet'      : 'Carpet',
}


def _synthetic_id(name: str) -> int:
    """
    Returns a deterministic integer ID in [1_000_001, 2_900_000] for a
    player name not found in atp_players.csv.

    Uses abs(hash(normalised_name)) % 1_900_000 + 1_000_001.
    All existing Sackmann IDs are <= 213_704, so there is no overlap.
    """
    return abs(hash(name.lower().strip())) % 1_900_000 + 1_000_001


def _build_player_name_to_id(players_csv: Path) -> dict:
    """
    Parses atp_players.csv and returns {"Jannik Sinner": 207989, ...}.

    Skips rows where player_id is not a valid integer.
    """
    df = pd.read_csv(players_csv, dtype=str, low_memory=False)
    df['player_id'] = pd.to_numeric(df['player_id'], errors='coerce')
    df = df.dropna(subset=['player_id', 'name_first', 'name_last'])
    df['player_id'] = df['player_id'].astype(int)

    return {
        f"{row['name_first']} {row['name_last']}": int(row['player_id'])
        for _, row in df.iterrows()
    }
```

- [ ] **Step 1.4: Run tests — should pass for the two helper classes**

```bash
venv/Scripts/python -m pytest tests/test_inject_2025_data.py::TestSyntheticId tests/test_inject_2025_data.py::TestBuildPlayerNameToId -v
```

Expected: All 7 tests PASS.

- [ ] **Step 1.5: Commit**

```bash
git add src/inject_2025_data.py tests/test_inject_2025_data.py
git commit -m "feat: add inject_2025_data skeleton with _synthetic_id and _build_player_name_to_id"
```

---

### Task 2: `convert_2025_to_consolidated` — core conversion logic

- [ ] **Step 2.1: Write failing tests for convert_2025_to_consolidated**

Add the following class to `tests/test_inject_2025_data.py`:

```python
class TestConvert2025ToConsolidated:
    """convert_2025_to_consolidated maps XLSX rows to consolidated schema."""

    def _make_raw_df(self):
        """Minimal 2-row tennis-data XLSX-style DataFrame."""
        return pd.DataFrame({
            'Date'       : [pd.Timestamp('2025-01-06'), pd.Timestamp('2025-01-07')],
            'Tournament' : ['Brisbane',                  'Brisbane'],
            'Series'     : ['ATP250',                    'ATP250'],
            'Surface'    : ['Hard',                      'Hard'],
            'Round'      : ['1st Round',                 '2nd Round'],
            'Best of'    : [3,                           3],
            'Winner'     : ['Vukic A.',                  'Michelsen A.'],
            'Loser'      : ['Djokovic N.',               'Sinner J.'],
            'WRank'      : [77.0,                        18.0],
            'LRank'      : [7.0,                         1.0],
            'WPts'       : [800.0,                       3400.0],
            'LPts'       : [9960.0,                      10900.0],
            'Comment'    : ['Completed',                 'Completed'],
            'year'       : [2025,                        2025],
        })

    def _make_name_to_id(self):
        return {
            "Novak Djokovic": 104925,
            "Jannik Sinner": 207989,
        }

    def _make_name_mapping(self):
        """tennis-data name → Sackmann name."""
        return {
            "Djokovic N.": "Novak Djokovic",
            "Sinner J.": "Jannik Sinner",
            "Vukic A.": "Vukic A.",
            "Michelsen A.": "Michelsen A.",
        }

    def test_required_columns_present(self):
        from inject_2025_data import convert_2025_to_consolidated
        df_raw = self._make_raw_df()
        name_to_id = self._make_name_to_id()
        name_mapping = self._make_name_mapping()
        result = convert_2025_to_consolidated(df_raw, name_to_id, name_mapping, 'atp')

        required = [
            'tourney_id', 'tourney_name', 'tourney_date', 'tourney_level',
            'surface', 'draw_size', 'best_of', 'round', 'year',
            'winner_id', 'winner_name', 'winner_rank', 'winner_rank_points',
            'loser_id', 'loser_name', 'loser_rank', 'loser_rank_points',
        ]
        for col in required:
            assert col in result.columns, f"Missing column: {col}"

    def test_round_mapping(self):
        from inject_2025_data import convert_2025_to_consolidated
        df_raw = self._make_raw_df()
        result = convert_2025_to_consolidated(
            df_raw, self._make_name_to_id(), self._make_name_mapping(), 'atp'
        )
        assert result.iloc[0]['round'] == 'R64'   # '1st Round'
        assert result.iloc[1]['round'] == 'R32'   # '2nd Round'

    def test_level_mapping(self):
        from inject_2025_data import convert_2025_to_consolidated
        df_raw = self._make_raw_df()
        result = convert_2025_to_consolidated(
            df_raw, self._make_name_to_id(), self._make_name_mapping(), 'atp'
        )
        assert (result['tourney_level'] == 'A').all()  # ATP250 → A

    def test_known_player_gets_sackmann_id(self):
        from inject_2025_data import convert_2025_to_consolidated
        df_raw = self._make_raw_df()
        result = convert_2025_to_consolidated(
            df_raw, self._make_name_to_id(), self._make_name_mapping(), 'atp'
        )
        # Djokovic is the loser in row 0 → loser_id should be 104925
        assert int(result.iloc[0]['loser_id']) == 104925
        # Sinner is the loser in row 1 → loser_id should be 207989
        assert int(result.iloc[1]['loser_id']) == 207989

    def test_unknown_player_gets_synthetic_id_in_range(self):
        from inject_2025_data import convert_2025_to_consolidated
        df_raw = self._make_raw_df()
        result = convert_2025_to_consolidated(
            df_raw, self._make_name_to_id(), self._make_name_mapping(), 'atp'
        )
        # Vukic and Michelsen are unmapped → synthetic IDs
        vukic_id = int(result.iloc[0]['winner_id'])
        assert 1_000_001 <= vukic_id <= 2_900_000

    def test_tourney_id_format(self):
        """tourney_id must match pattern YYYY-NNNN."""
        from inject_2025_data import convert_2025_to_consolidated
        df_raw = self._make_raw_df()
        result = convert_2025_to_consolidated(
            df_raw, self._make_name_to_id(), self._make_name_mapping(), 'atp'
        )
        for tid in result['tourney_id']:
            parts = tid.split('-')
            assert parts[0] == '2025'
            assert parts[1].isdigit()

    def test_year_column_is_2025(self):
        from inject_2025_data import convert_2025_to_consolidated
        df_raw = self._make_raw_df()
        result = convert_2025_to_consolidated(
            df_raw, self._make_name_to_id(), self._make_name_mapping(), 'atp'
        )
        assert (result['year'] == 2025).all()

    def test_best_of_dtype_is_Int64(self):
        from inject_2025_data import convert_2025_to_consolidated
        df_raw = self._make_raw_df()
        result = convert_2025_to_consolidated(
            df_raw, self._make_name_to_id(), self._make_name_mapping(), 'atp'
        )
        assert str(result['best_of'].dtype) == 'Int64'

    def test_stat_columns_are_nan(self):
        """Service stats (w_ace etc.) should be NaN — not available from tennis-data."""
        from inject_2025_data import convert_2025_to_consolidated
        df_raw = self._make_raw_df()
        result = convert_2025_to_consolidated(
            df_raw, self._make_name_to_id(), self._make_name_mapping(), 'atp'
        )
        stat_cols = ['w_ace', 'w_df', 'l_ace', 'l_df']
        for col in stat_cols:
            if col in result.columns:
                assert result[col].isna().all(), f"{col} should be all NaN"
```

- [ ] **Step 2.2: Run tests to verify they fail**

```bash
venv/Scripts/python -m pytest tests/test_inject_2025_data.py::TestConvert2025ToConsolidated -v 2>&1 | head -20
```

Expected: `AttributeError` or `ImportError` — `convert_2025_to_consolidated` not yet defined.

- [ ] **Step 2.3: Implement `convert_2025_to_consolidated` in `src/inject_2025_data.py`**

Add the following function after `_build_player_name_to_id`:

```python
def convert_2025_to_consolidated(
    df_raw: pd.DataFrame,
    name_to_id: dict,
    name_mapping: dict,
    tour: str,
) -> pd.DataFrame:
    """
    Converts a tennis-data 2025 DataFrame (already filtered to Completed rows)
    into the matches_consolidated.parquet column schema.

    Parameters
    ----------
    df_raw      : DataFrame from load_new_matches([2025], cfg, odds_dir)
    name_to_id  : {"Jannik Sinner": 207989, ...} from _build_player_name_to_id()
    name_mapping: {"Sinner J.": "Jannik Sinner", ...} from build_name_mapping()
    tour        : "atp" (used only for level mapping via config)
    """
    from config import get_tour_config
    cfg = get_tour_config(tour)
    level_map = cfg['td_level_map']

    df = df_raw.copy()
    df['tourney_date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['year'] = df['tourney_date'].dt.year.astype('int32')

    # ── Tournament name & level ──────────────────────────────────────────────
    series_col = 'Series' if 'Series' in df.columns else 'Tier'
    df['tourney_name']  = df['Tournament'].str.strip()
    df['tourney_level'] = (
        df[series_col].map(level_map).fillna('A')
        if series_col in df.columns else 'A'
    )

    # ── tourney_id: "2025-<zero_padded_hash>" unique per tournament name ────
    def _tourney_suffix(name: str) -> str:
        suffix = abs(hash(name.lower().strip())) % 9899 + 1
        return str(suffix).zfill(4)

    df['tourney_id'] = df['tourney_name'].apply(
        lambda n: f"2025-{_tourney_suffix(n)}"
    )

    # ── Surface ──────────────────────────────────────────────────────────────
    df['surface'] = df['Surface'].map(SURFACE_MAP).fillna('Hard')

    # ── Round ────────────────────────────────────────────────────────────────
    df['round'] = df['Round'].map(ROUND_MAP).fillna(df['Round'])

    # ── best_of ───────────────────────────────────────────────────────────────
    df['best_of'] = pd.to_numeric(df['Best of'], errors='coerce').fillna(3).astype('Int64')

    # ── Player names: translate tennis-data → Sackmann then look up IDs ─────
    def _resolve_name(td_name: str) -> str:
        return name_mapping.get(td_name, td_name)

    def _resolve_id(sackmann_name: str) -> int:
        known = name_to_id.get(sackmann_name)
        return known if known is not None else _synthetic_id(sackmann_name)

    df['winner_name'] = df['Winner'].str.strip().apply(_resolve_name)
    df['loser_name']  = df['Loser'].str.strip().apply(_resolve_name)
    df['winner_id']   = df['winner_name'].apply(_resolve_id)
    df['loser_id']    = df['loser_name'].apply(_resolve_id)
    df['winner_id']   = df['winner_id'].astype('Int64')
    df['loser_id']    = df['loser_id'].astype('Int64')

    # ── Rankings ──────────────────────────────────────────────────────────────
    df['winner_rank']        = pd.to_numeric(df['WRank'], errors='coerce')
    df['loser_rank']         = pd.to_numeric(df['LRank'], errors='coerce')
    df['winner_rank_points'] = pd.to_numeric(df.get('WPts', np.nan), errors='coerce')
    df['loser_rank_points']  = pd.to_numeric(df.get('LPts', np.nan), errors='coerce')

    # ── Columns absent from tennis-data → NaN ────────────────────────────────
    for col in ['draw_size', 'winner_hand', 'winner_age', 'loser_hand', 'loser_age',
                'score', 'minutes',
                'w_ace','w_df','w_svpt','w_1stIn','w_1stWon','w_2ndWon',
                'w_SvGms','w_bpSaved','w_bpFaced',
                'l_ace','l_df','l_svpt','l_1stIn','l_1stWon','l_2ndWon',
                'l_SvGms','l_bpSaved','l_bpFaced']:
        df[col] = np.nan

    # ── Select and order final columns to match consolidated schema ──────────
    SCHEMA_COLS = [
        'tourney_id', 'tourney_name', 'tourney_date', 'tourney_level',
        'surface', 'draw_size', 'best_of', 'round', 'year',
        'winner_id', 'winner_name', 'winner_hand', 'winner_age',
        'winner_rank', 'winner_rank_points',
        'loser_id', 'loser_name', 'loser_hand', 'loser_age',
        'loser_rank', 'loser_rank_points',
        'score', 'minutes',
        'w_ace','w_df','w_svpt','w_1stIn','w_1stWon','w_2ndWon',
        'w_SvGms','w_bpSaved','w_bpFaced',
        'l_ace','l_df','l_svpt','l_1stIn','l_1stWon','l_2ndWon',
        'l_SvGms','l_bpSaved','l_bpFaced',
    ]

    out = df[SCHEMA_COLS].copy()
    out['draw_size'] = out['draw_size'].astype(float)

    return out.reset_index(drop=True)
```

- [ ] **Step 2.4: Run the conversion tests**

```bash
venv/Scripts/python -m pytest tests/test_inject_2025_data.py::TestConvert2025ToConsolidated -v
```

Expected: All 9 tests PASS.

- [ ] **Step 2.5: Commit**

```bash
git add src/inject_2025_data.py tests/test_inject_2025_data.py
git commit -m "feat: implement convert_2025_to_consolidated with round/level/ID mapping"
```

---

### Task 3: Idempotency check and `main()` function

- [ ] **Step 3.1: Write the idempotency test**

Add this class to `tests/test_inject_2025_data.py`:

```python
class TestIdempotency:
    """Running inject_2025_data when 2025 data is already present must be a no-op."""

    def test_already_present_returns_true(self, tmp_path):
        """
        _check_already_injected returns True (= already done) when year==2025
        rows exist in the parquet.
        """
        from inject_2025_data import _check_already_injected

        df = pd.DataFrame({'year': [2024, 2025], 'tourney_name': ['X', 'Y']})
        p = tmp_path / "matches_consolidated.parquet"
        df.to_parquet(p, index=False)

        assert _check_already_injected(p) is True

    def test_not_present_returns_false(self, tmp_path):
        from inject_2025_data import _check_already_injected

        df = pd.DataFrame({'year': [2023, 2024], 'tourney_name': ['X', 'Y']})
        p = tmp_path / "matches_consolidated.parquet"
        df.to_parquet(p, index=False)

        assert _check_already_injected(p) is False
```

- [ ] **Step 3.2: Run tests to verify they fail**

```bash
venv/Scripts/python -m pytest tests/test_inject_2025_data.py::TestIdempotency -v 2>&1 | head -15
```

Expected: `ImportError` — `_check_already_injected` not yet defined.

- [ ] **Step 3.3: Implement `_check_already_injected` and `main()` in `src/inject_2025_data.py`**

Add after `convert_2025_to_consolidated`:

```python
def _check_already_injected(consolidated_path: Path) -> bool:
    """
    Returns True if matches_consolidated.parquet already contains
    rows with year == 2025 — in which case injection should be skipped.
    """
    df = pd.read_parquet(consolidated_path, columns=['year'])
    return int((df['year'] == 2025).sum()) > 0


def main():
    parser = argparse.ArgumentParser(
        description="Inject 2025 ATP matches into matches_consolidated.parquet"
    )
    parser.add_argument('--tour', default='atp', choices=['atp', 'wta'],
                        help="Tour (default: atp)")
    args = parser.parse_args()

    tour  = args.tour.lower()
    cfg   = get_tour_config(tour)
    paths = get_paths(tour)
    make_dirs(tour)

    processed_dir = paths['processed_dir']
    raw_dir       = paths['raw_dir']
    odds_dir      = paths['odds_dir']
    consolidated  = processed_dir / "matches_consolidated.parquet"
    players_csv   = raw_dir / cfg['player_file']

    print("=" * 55)
    print(f"INJECT 2025 DATA — {tour.upper()}")
    print("=" * 55)

    # ── Idempotency guard ────────────────────────────────────────────────────
    if _check_already_injected(consolidated):
        print("\n  2025 rows already present in matches_consolidated.parquet.")
        print("  Nothing to do. Exiting.")
        return

    # ── Load name-to-id mapping from players CSV ─────────────────────────────
    print("\n── Building player ID lookup ─────────────────────────")
    name_to_id = _build_player_name_to_id(players_csv)
    print(f"  {len(name_to_id):,} players found in {players_csv.name}")

    # ── Load existing consolidated parquet for name-building ─────────────────
    print("\n── Loading consolidated parquet ──────────────────────")
    df_base = pd.read_parquet(consolidated)
    print(f"  {len(df_base):,} rows, {df_base['tourney_date'].max().date()} max date")

    sackmann_names = set(
        df_base['winner_name'].dropna().tolist() +
        df_base['loser_name'].dropna().tolist()
    )
    print(f"  {len(sackmann_names):,} unique Sackmann player names available")

    # ── Load 2025 XLSX via shared loader ────────────────────────────────────
    print("\n── Loading 2025 tennis-data XLSX ─────────────────────")
    df_raw = load_new_matches([2025], cfg, odds_dir)

    if df_raw.empty:
        print("  No 2025 data found. Check data/odds/atp/atp_2025.xlsx exists.")
        return

    print(f"  {len(df_raw):,} completed 2025 matches loaded")

    # ── Build name mapping (tennis-data → Sackmann) ──────────────────────────
    print("\n── Building name mapping ─────────────────────────────")
    elo_fake = {name: 0 for name in sackmann_names}
    name_mapping = build_name_mapping(df_raw, elo_fake)

    # ── Convert to consolidated schema ───────────────────────────────────────
    print("\n── Converting to consolidated schema ─────────────────")
    df_2025 = convert_2025_to_consolidated(df_raw, name_to_id, name_mapping, tour)
    print(f"  {len(df_2025):,} rows converted")
    print(f"  Date range: {df_2025['tourney_date'].min().date()} "
          f"→ {df_2025['tourney_date'].max().date()}")

    # ── Backup ───────────────────────────────────────────────────────────────
    backup = consolidated.with_suffix('.parquet.bak2024')
    print(f"\n── Creating backup → {backup.name} ──────────────────")
    import shutil
    shutil.copy2(consolidated, backup)
    print(f"  Backup written ({backup.stat().st_size / 1024 / 1024:.1f} MB)")

    # ── Append and save ───────────────────────────────────────────────────────
    print("\n── Appending 2025 rows and saving ────────────────────")
    df_combined = pd.concat([df_base, df_2025], ignore_index=True)
    df_combined = df_combined.sort_values('tourney_date').reset_index(drop=True)

    df_combined['best_of']   = df_combined['best_of'].astype('Int64')
    df_combined['winner_id'] = df_combined['winner_id'].astype('Int64')
    df_combined['loser_id']  = df_combined['loser_id'].astype('Int64')
    df_combined['year']      = df_combined['year'].astype('int32')

    df_combined.to_parquet(consolidated, index=False)

    n_2025_check = int((df_combined['year'] == 2025).sum())
    print(f"\n  Saved: {len(df_combined):,} total rows")
    print(f"  2025 rows: {n_2025_check:,}")
    print(f"\n  Now run the feature pipeline (--tour {tour}):")
    print(f"    venv/Scripts/python src/restructure_data.py --tour {tour}")
    print(f"    venv/Scripts/python src/compute_elo.py --tour {tour}")
    print(f"    venv/Scripts/python src/compute_rolling_features.py --tour {tour}")
    print(f"    venv/Scripts/python src/compute_h2h.py --tour {tour}")
    print(f"    venv/Scripts/python src/compute_contextual_features.py --tour {tour}")
    print(f"    venv/Scripts/python src/compute_glicko.py --tour {tour}")
    print(f"    venv/Scripts/python src/prepare_ml_dataset.py --tour {tour}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3.4: Run all tests to ensure no regressions**

```bash
venv/Scripts/python -m pytest tests/test_inject_2025_data.py -v
```

Expected: All tests PASS.

- [ ] **Step 3.5: Commit**

```bash
git add src/inject_2025_data.py tests/test_inject_2025_data.py
git commit -m "feat: add _check_already_injected and main() for inject_2025_data"
```

---

## Chunk 2: Execution — Injection + Pipeline + Backtest

### Task 4: Run `inject_2025_data.py`

- [ ] **Step 4.1: Verify 2025 XLSX is present and has expected row count**

```bash
cd E:/Claude/botbet/tennis/tennis_ml
venv/Scripts/python -c "
import pandas as pd
df = pd.read_excel('data/odds/atp/atp_2025.xlsx', engine='openpyxl')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df_2025 = df[(df['Date'].dt.year == 2025) & (df['Comment'] == 'Completed')]
print(f'2025 Completed rows: {len(df_2025)}')
print(f'Date range: {df_2025[\"Date\"].min().date()} to {df_2025[\"Date\"].max().date()}')
"
```

Expected: ~2,491 completed rows from 2025-01-01 to 2025-11-16.

- [ ] **Step 4.2: Run the injection script**

```bash
venv/Scripts/python src/inject_2025_data.py --tour atp
```

Expected output (approximately):
```
=======================================================
INJECT 2025 DATA — ATP
=======================================================
── Building player ID lookup ─────────────────────────
  65989 players found in atp_players.csv
── Loading consolidated parquet ──────────────────────
  193714 rows, 2024-12-18 max date
── Loading 2025 tennis-data XLSX ─────────────────────
  Total : 2,491 matchs | 2025-01-06 → 2025-11-16
── Building name mapping ─────────────────────────────
── Converting to consolidated schema ─────────────────
  2,491 rows converted
── Creating backup → matches_consolidated.parquet.bak2024
── Appending 2025 rows and saving ────────────────────
  Saved: ~196,205 total rows
  2025 rows: 2,491
```

- [ ] **Step 4.3: Verify injection succeeded**

```bash
venv/Scripts/python -c "
import pandas as pd
df = pd.read_parquet('data/processed/atp/matches_consolidated.parquet')
print('Total rows:', len(df))
print('Max date:', df['tourney_date'].max().date())
print('Year 2025 rows:', int((df['year'] == 2025).sum()))
print('Dtypes:', df['best_of'].dtype, df['winner_id'].dtype)
"
```

Expected: Total rows ~196,205, year 2025 rows ~2,491, dtypes `Int64`.

- [ ] **Step 4.4: Verify idempotency — running again must exit early**

```bash
venv/Scripts/python src/inject_2025_data.py --tour atp
```

Expected: `2025 rows already present in matches_consolidated.parquet. Nothing to do. Exiting.`

---

### Task 5: Re-run the feature pipeline

All commands run from the project root. Each step reads the output of the previous.

- [ ] **Step 5.1: restructure_data.py — winner/loser → neutral p1/p2**

```bash
venv/Scripts/python src/restructure_data.py --tour atp
```

Expected: prints match count, `target (p1 wins) ≈ 0.500`, writes `matches_ml_ready.parquet`.

- [ ] **Step 5.2: compute_elo.py**

```bash
venv/Scripts/python src/compute_elo.py --tour atp
```

Expected: completes without error, writes `matches_with_elo.parquet`.

- [ ] **Step 5.3: compute_rolling_features.py**

```bash
venv/Scripts/python src/compute_rolling_features.py --tour atp
```

Expected: completes without error.

- [ ] **Step 5.4: compute_h2h.py**

```bash
venv/Scripts/python src/compute_h2h.py --tour atp
```

Expected: completes without error.

- [ ] **Step 5.5: compute_contextual_features.py**

```bash
venv/Scripts/python src/compute_contextual_features.py --tour atp
```

Expected: completes without error, writes `matches_features_final.parquet`.

- [ ] **Step 5.6: compute_glicko.py**

```bash
venv/Scripts/python src/compute_glicko.py --tour atp
```

Expected: completes without error.

- [ ] **Step 5.7: prepare_ml_dataset.py**

```bash
venv/Scripts/python src/prepare_ml_dataset.py --tour atp
```

Expected output includes:
```
    Test   (>= 2025) :   X,XXX matchs
```

- [ ] **Step 5.8: Verify X_test is now populated**

```bash
venv/Scripts/python -c "
import joblib
splits = joblib.load('data/models/atp/splits.pkl')
for k in ['X_train', 'X_valid', 'X_test']:
    print(f'{k}: {splits[k].shape}')
print('meta_test year:', splits['meta_test']['year'].unique().tolist())
"
```

Expected: `X_test` shape is `(~1800+, N_features)`, meta_test year is `[2025]`.

---

### Task 6: Modify `backtest_real.py` TEST_YEARS and run

- [ ] **Step 6.1: Edit line 531 of `src/backtest_real.py`**

Change:
```python
TEST_YEARS = [2023, 2024]
```
To:
```python
TEST_YEARS = [2025]
```

Verify:
```bash
grep -n "TEST_YEARS" src/backtest_real.py
```

Expected: `531: TEST_YEARS = [2025]`

- [ ] **Step 6.2: Run the backtest**

```bash
venv/Scripts/python src/backtest_real.py --tour atp 2>&1 | tee backtest_2025_output.txt
```

Expected: no crash, outputs ROI and bet statistics for 2025.

- [ ] **Step 6.3: Verify jointure quality**

In `backtest_2025_output.txt` look for:
- "Jointure normale" and "Jointure inversée" row counts — both non-zero
- Accuracy numbers in range 55%–80%
- At least 200 bets matched with PSW odds

If Pinnacle coverage is unexpectedly low (< 30%), check the atp_2025.xlsx PSW column:
```bash
venv/Scripts/python -c "
import pandas as pd
df = pd.read_excel('data/odds/atp/atp_2025.xlsx', engine='openpyxl')
df = df[df['Comment'] == 'Completed']
print('PSW non-null:', df['PSW'].notna().sum(), '/', len(df))
"
```

- [ ] **Step 6.4: Commit**

```bash
git add src/backtest_real.py backtest_2025_output.txt
git commit -m "feat: run 2025 out-of-sample backtest (TEST_YEARS=[2025])"
```

---

## Troubleshooting Reference

**X_test still empty after prepare_ml_dataset.py**
Check that 2025 rows exist in `matches_features_final.parquet`:
```python
df = pd.read_parquet('data/processed/atp/matches_features_final.parquet')
print((df['year'] == 2025).sum())
```
If 0: re-run `compute_contextual_features.py` (it writes the final parquet).

**Very low match count in backtest join (< 50 matches)**
`normalize_name_for_join` handles both "Sinner J." and "Jannik Sinner". If coverage is poor, print a sample:
```python
print(df_pred['p1_name'].head(10).tolist())
print(df_odds['winner_clean'].head(10).tolist())
```
Both should produce `"lastname_initial"` keys when normalized.

**Pipeline step crashes on dtype mismatch**
The concat in `inject_2025_data.main()` re-applies `Int64` dtypes. If a later script still crashes, check with:
```python
df = pd.read_parquet('data/processed/atp/matches_consolidated.parquet')
print(df[['winner_id','loser_id','best_of']].dtypes)
```

**`Masters Cup` or `ATP Finals` level mapping**
`config.py` td_level_map includes `'Masters Cup': 'F'` and `'ATP Finals': 'F'`. If new Series values appear, they fall back to `'A'` (conservative and safe).
