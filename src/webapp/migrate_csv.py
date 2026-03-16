"""One-time migration: bets_atp.csv + bets_wta.csv → SQLite.

Run with:
    python src/webapp/migrate_csv.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

COLUMN_MAP = {
    'tournament': 'tournament', 'surface': 'surface', 'round': 'round',
    'p1_name': 'p1_name', 'p2_name': 'p2_name', 'bet_on': 'bet_on',
    'prob': 'prob', 'edge': 'edge', 'odd': 'odd', 'stake': 'stake',
    'kelly_frac': 'kelly_frac', 'status': 'status', 'pnl': 'pnl',
}


def _safe(v, default=None):
    """Return None if pandas NA, else the value."""
    try:
        if pd.isna(v):
            return default
    except (TypeError, ValueError):
        pass
    return v


def migrate_bets(tour: str, conn) -> int:
    from src.webapp.db import add_bet, set_bankroll, get_bankroll
    csv_path = ROOT / "data" / f"bets_{tour}.csv"
    if not csv_path.exists():
        print(f"  {csv_path.name} not found — skipping.")
        return 0
    df = pd.read_csv(csv_path)
    migrated = 0
    for _, row in df.iterrows():
        bet = {k: _safe(row.get(v)) for k, v in COLUMN_MAP.items()}
        bet['tour'] = tour
        if not bet.get('tournament'): bet['tournament'] = 'Unknown'
        if not bet.get('surface'):    bet['surface'] = 'Hard'
        if not bet.get('p1_name') or not bet.get('p2_name'):
            continue
        if not bet.get('bet_on'):  bet['bet_on'] = 'p1'
        if bet.get('prob') is None: bet['prob'] = 0.5
        if bet.get('odd') is None:  bet['odd'] = 1.0
        if bet.get('stake') is None: bet['stake'] = 0.0
        add_bet(conn, bet)
        migrated += 1
    print(f"  Migrated {migrated} bets for {tour.upper()}")
    csv_path.rename(csv_path.with_suffix('.csv.bak'))
    return migrated


def migrate_bankroll(tour: str, conn) -> None:
    from src.webapp.db import set_bankroll
    json_path = ROOT / "data" / f"bankroll_{tour}.json"
    if not json_path.exists():
        return
    with open(json_path) as f:
        data = json.load(f)
    amount = data.get('bankroll', 1000.0)
    set_bankroll(conn, tour, amount)
    print(f"  Migrated bankroll {tour.upper()}: {amount}€")
    json_path.rename(json_path.with_suffix('.json.bak'))


if __name__ == '__main__':
    from src.webapp.db import get_connection, init_db

    db_path = ROOT / "data" / "tennis_predict.db"
    conn = get_connection(db_path)
    init_db(conn)
    for tour in ('atp', 'wta'):
        migrate_bankroll(tour, conn)
        migrate_bets(tour, conn)
    conn.close()
    print("Migration complete.")
