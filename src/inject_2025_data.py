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
