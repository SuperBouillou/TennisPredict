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
    # build_name_mapping expects 'winner_name' / 'loser_name' columns
    df_raw_mapped = df_raw.rename(columns={'Winner': 'winner_name', 'Loser': 'loser_name'})
    name_mapping = build_name_mapping(df_raw_mapped, elo_fake)

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
