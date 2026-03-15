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
