"""Player search and profile helpers."""
from __future__ import annotations

import pandas as pd


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
            'rank':        int(r['rank']) if pd.notna(r.get('rank')) else None,
            'elo':         round(float(r.get('elo', 0) or 0), 0),
            'surface':     _best_surface(r),
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

    # Enrich from players.parquet (identity: nationality, hand, DOB)
    identity = {}
    if not players.empty and 'name_first' in players.columns:
        if 'name_key' not in players.columns:
            players = players.copy()
            players['name_key'] = (
                players['name_first'].fillna('') + ' ' + players['name_last'].fillna('')
            ).str.lower().str.strip()
        id_rows = players[players['name_key'] == key]
        if not id_rows.empty:
            identity = id_rows.iloc[0].to_dict()

    merged = {**r, **{k: v for k, v in identity.items() if k not in r}}
    return merged


def _best_surface(r) -> str:
    surfaces = {
        'Hard':  float(r.get('elo_hard', 0) or 0),
        'Clay':  float(r.get('elo_clay', 0) or 0),
        'Grass': float(r.get('elo_grass', 0) or 0),
    }
    return max(surfaces, key=surfaces.get)
