# src/optimize_thresholds.py
"""
Optimisation des seuils de pari (min_edge × min_prob) sur données Pinnacle.

Usage:
    python src/optimize_thresholds.py --tour atp
    python src/optimize_thresholds.py --tour wta --min-bets 20

Prérequis : avoir lancé backtest_real.py pour générer backtest_all_candidates.parquet

Output:
    data/models/{tour}/optimal_thresholds.json
"""

import sys
import json
import argparse
import warnings
import itertools
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from pathlib import Path

from config import get_paths, make_dirs


def compute_threshold_grid(df: pd.DataFrame,
                            min_edge_range: list,
                            min_prob_range: list,
                            min_bets: int = 30) -> pd.DataFrame:
    """
    Grid search (min_edge × min_prob) → tableau ROI / N_bets / Sharpe.

    df doit avoir les colonnes : edge, our_prob, pnl, stake, won
    """
    rows = []
    for min_edge, min_prob in itertools.product(min_edge_range, min_prob_range):
        sub = df[(df['edge'] >= min_edge) & (df['our_prob'] >= min_prob)]
        n = len(sub)
        if n < min_bets:
            roi = np.nan
            sharpe = np.nan
        else:
            roi = sub['pnl'].sum() / sub['stake'].sum()
            sharpe = (sub['pnl'].mean() / sub['pnl'].std() * np.sqrt(252)
                      if sub['pnl'].std() > 0 else 0.0)
        rows.append({
            'min_edge': round(min_edge, 3),
            'min_prob': round(min_prob, 3),
            'n_bets'  : n,
            'roi'     : roi,
            'sharpe'  : sharpe,
        })
    return pd.DataFrame(rows)


def analyse_by_group(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """ROI / n_bets / win_rate groupé par une colonne (level, surface)."""
    if col not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby(col)
        .apply(lambda g: pd.Series({
            'n_bets'  : len(g),
            'win_rate': round(g['won'].mean(), 3),
            'roi'     : round(g['pnl'].sum() / g['stake'].sum(), 3),
            'pnl'     : round(g['pnl'].sum(), 1),
            'avg_odd' : round(g['odd'].mean(), 2),
            'avg_clv' : round(g['clv'].mean(), 3) if 'clv' in g.columns else np.nan,
        }))
        .sort_values('roi', ascending=False)
    )


def main():
    parser = argparse.ArgumentParser(description="Optimisation seuils de pari")
    parser.add_argument('--tour',     default='atp', choices=['atp', 'wta'])
    parser.add_argument('--min-bets', type=int, default=30,
                        help="Minimum de paris requis pour valider un combo (défaut: 30)")
    args = parser.parse_args()

    tour  = args.tour.lower()
    paths = get_paths(tour)
    MODELS_DIR = paths['models_dir']

    print("=" * 55)
    print(f"THRESHOLD OPTIMIZER — {tour.upper()}")
    print("=" * 55)

    # ── Chargement des candidats bruts ───────────────────────────────────────
    raw_path = MODELS_DIR / "backtest_all_candidates.parquet"
    if not raw_path.exists():
        print(f"\n  ERREUR : {raw_path} non trouvé.")
        print("  Lance d'abord : python src/backtest_real.py")
        sys.exit(1)

    df = pd.read_parquet(raw_path)
    print(f"\n  Candidats chargés : {len(df):,} paris")
    print(f"  Période           : {df['date'].min()} → {df['date'].max()}")
    print(f"  Edge moyen        : {df['edge'].mean():+.1%}")
    if 'clv' in df.columns:
        print(f"  CLV moyen         : {df['clv'].mean():+.1%}")

    # ── Grid search ──────────────────────────────────────────────────────────
    min_edge_range = [i / 100 for i in range(0, 12)]          # 0% → 11%
    min_prob_range = [0.50 + i * 0.01 for i in range(17)]     # 0.50 → 0.66

    n_combos = len(min_edge_range) * len(min_prob_range)
    print(f"\n  Grid search : {len(min_edge_range)} × {len(min_prob_range)} = {n_combos} combos")
    print(f"  Minimum bets requis : {args.min_bets}")

    df_grid = compute_threshold_grid(df, min_edge_range, min_prob_range, args.min_bets)

    valid = df_grid.dropna(subset=['roi'])
    if len(valid) == 0:
        print("\n  Aucun combo valide (trop peu de paris partout).")
        print("  Diminue --min-bets ou relance backtest_real.py avec des données supplémentaires.")
        sys.exit(1)

    best_roi    = valid.loc[valid['roi'].idxmax()].to_dict()
    best_sharpe = valid.loc[valid['sharpe'].idxmax()].to_dict()

    # ── Résultats ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("RÉSULTATS")
    print("=" * 55)

    print(f"\n  Meilleur ROI :")
    print(f"    min_edge : {best_roi['min_edge']:.0%}")
    print(f"    min_prob : {best_roi['min_prob']:.2f}")
    print(f"    n_bets   : {int(best_roi['n_bets'])}")
    print(f"    ROI      : {best_roi['roi']:+.2%}")
    print(f"    Sharpe   : {best_roi['sharpe']:.2f}")

    print(f"\n  Meilleur Sharpe :")
    print(f"    min_edge : {best_sharpe['min_edge']:.0%}")
    print(f"    min_prob : {best_sharpe['min_prob']:.2f}")
    print(f"    n_bets   : {int(best_sharpe['n_bets'])}")
    print(f"    ROI      : {best_sharpe['roi']:+.2%}")
    print(f"    Sharpe   : {best_sharpe['sharpe']:.2f}")

    # ── Analyse par niveau et surface au threshold optimal ────────────────────
    sub_opt = df[
        (df['edge'] >= best_roi['min_edge']) &
        (df['our_prob'] >= best_roi['min_prob'])
    ]

    print(f"\n  Paris au threshold optimal : {len(sub_opt):,}")

    print("\n  ROI par niveau de tournoi :")
    by_level = analyse_by_group(sub_opt, 'level')
    if not by_level.empty:
        print(by_level[['n_bets', 'win_rate', 'roi', 'pnl', 'avg_odd']].to_string())
        profitable_levels = by_level[by_level['roi'] > 0].index.tolist()
    else:
        profitable_levels = []

    print("\n  ROI par surface :")
    by_surface = analyse_by_group(sub_opt, 'surface')
    if not by_surface.empty:
        print(by_surface[['n_bets', 'win_rate', 'roi', 'pnl', 'avg_odd']].to_string())
        profitable_surfaces = by_surface[by_surface['roi'] > 0].index.tolist()
    else:
        profitable_surfaces = []

    print("\n  Top 15 combos par ROI :")
    print(valid.sort_values('roi', ascending=False).head(15).to_string(index=False))

    # ── Sauvegarde JSON ───────────────────────────────────────────────────────
    out = MODELS_DIR / "optimal_thresholds.json"
    result = {
        'tour'       : tour,
        'n_candidates': len(df),
        'best_roi'   : {
            'min_edge'  : float(best_roi['min_edge']),
            'min_prob'  : float(best_roi['min_prob']),
            'n_bets'    : int(best_roi['n_bets']),
            'roi'       : float(best_roi['roi']),
            'sharpe'    : float(best_roi['sharpe']),
        },
        'best_sharpe': {
            'min_edge'  : float(best_sharpe['min_edge']),
            'min_prob'  : float(best_sharpe['min_prob']),
            'n_bets'    : int(best_sharpe['n_bets']),
            'roi'       : float(best_sharpe['roi']),
            'sharpe'    : float(best_sharpe['sharpe']),
        },
        'profitable_levels'  : profitable_levels,
        'profitable_surfaces': profitable_surfaces,
    }
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n  Thresholds sauvegardés → {out}")
    print(f"\nOptimisation terminée.")


if __name__ == "__main__":
    main()
