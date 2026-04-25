# src/calibrate_thresholds.py
# Sweep systématique des seuils edge par surface × niveau de tournoi
# Usage: python src/calibrate_thresholds.py --tour atp
#
# Charge le test set (≥2025), joint avec les cotes Pinnacle,
# et pour chaque combinaison (surface, level, edge_min) calcule :
#   - nombre de bets, win rate, ROI flat, ROI Kelly
# Sortie : tableau + recommandations de seuils pour today.py

import sys, argparse, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import get_tour_config, get_paths, make_dirs
from backtest_real import (
    load_real_odds, build_compound_lastnames,
    join_odds_to_predictions
)


# ─────────────────────────────────────────────────────────────────────────────
# BETS BRUTS
# ─────────────────────────────────────────────────────────────────────────────

def collect_all_bets(df: pd.DataFrame,
                     odds_col_w: str = 'PSW',
                     odds_col_l: str = 'PSL') -> pd.DataFrame:
    """
    Pour chaque match avec cotes, génère DEUX lignes (côté P1 et côté P2)
    avec : edge, odd, won, surface, level.
    Pas de filtre — on garde tout pour le sweep.
    """
    rows = []
    for _, row in df.iterrows():
        p1_prob = float(row['p1_prob'])
        p2_prob = 1.0 - p1_prob
        won_p1  = int(row['target']) == 1

        odd_p1 = row.get(odds_col_w) if row.get('p1_is_winner_odds', True) else row.get(odds_col_l)
        odd_p2 = row.get(odds_col_l) if row.get('p1_is_winner_odds', True) else row.get(odds_col_w)

        surf  = row.get('surface', 'Hard') or 'Hard'
        level = row.get('tourney_level', '') or ''
        # Normaliser level: G=GS, M=Masters, A=250/500, reste=other
        if level in ('G',):       lvl = 'G'
        elif level in ('M', 'F'): lvl = 'M'
        elif level in ('A', 'D'): lvl = 'A'
        else:                     lvl = 'other'

        for prob, odd, won in [(p1_prob, odd_p1, won_p1),
                               (p2_prob, odd_p2, not won_p1)]:
            if pd.isna(odd) or odd <= 1.0:
                continue
            bk_impl = 1.0 / odd
            edge    = prob - bk_impl
            rows.append({
                'edge':    edge,
                'prob':    prob,
                'odd':     odd,
                'won':     int(won),
                'surface': surf,
                'level':   lvl,
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# SWEEP DES SEUILS
# ─────────────────────────────────────────────────────────────────────────────

def roi_at_threshold(bets: pd.DataFrame, edge_min: float,
                     min_bets: int = 20) -> dict | None:
    sub = bets[bets['edge'] >= edge_min]
    n   = len(sub)
    if n < min_bets:
        return None
    wr  = sub['won'].mean()
    # ROI flat (mise 1 unité)
    pnl = (sub['won'] * (sub['odd'] - 1) - (1 - sub['won'])).sum()
    roi = pnl / n
    return {'n': n, 'wr': wr, 'roi': roi, 'avg_odd': sub['odd'].mean()}


def sweep(bets: pd.DataFrame,
          thresholds: np.ndarray = np.arange(0.00, 0.55, 0.01),
          min_bets: int = 25) -> pd.DataFrame:
    rows = []
    groups = [('ALL', 'ALL', bets)]
    for surf in bets['surface'].unique():
        groups.append((surf, 'ALL', bets[bets['surface'] == surf]))
    for lvl in bets['level'].unique():
        groups.append(('ALL', lvl, bets[bets['level'] == lvl]))
    for surf in bets['surface'].unique():
        for lvl in bets['level'].unique():
            sub = bets[(bets['surface'] == surf) & (bets['level'] == lvl)]
            if len(sub) >= min_bets:
                groups.append((surf, lvl, sub))

    for surf, lvl, grp in groups:
        for thr in thresholds:
            r = roi_at_threshold(grp, thr, min_bets)
            if r:
                rows.append({
                    'surface': surf, 'level': lvl,
                    'edge_min': round(thr, 2),
                    'n': r['n'], 'wr': r['wr'],
                    'roi': r['roi'], 'avg_odd': r['avg_odd'],
                })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMANDATIONS
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_threshold(df_sweep: pd.DataFrame,
                           surf: str, lvl: str,
                           target_roi: float = 0.02,
                           min_bets: int = 25) -> dict:
    """
    Pour (surf, lvl), trouve le seuil minimum tel que ROI >= target_roi
    en s'assurant que le ROI reste positif au-delà (pas juste un pic).
    """
    sub = df_sweep[(df_sweep['surface'] == surf) & (df_sweep['level'] == lvl)]
    if sub.empty:
        return {'thr': None, 'roi': None, 'n': 0}

    sub = sub.sort_values('edge_min')
    # Chercher le premier seuil où ROI >= target_roi avec au moins min_bets
    for _, row in sub.iterrows():
        if row['roi'] >= target_roi and row['n'] >= min_bets:
            return {'thr': row['edge_min'], 'roi': row['roi'], 'n': row['n']}
    return {'thr': None, 'roi': sub.iloc[-1]['roi'] if len(sub) else None, 'n': 0}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tour', default='atp', choices=['atp', 'wta'])
    parser.add_argument('--min-bets', type=int, default=25,
                        help='Minimum de bets pour qu\'un seuil soit valide')
    parser.add_argument('--save', action='store_true',
                        help='Sauvegarder les bets bruts en CSV')
    args = parser.parse_args()

    tour = args.tour.lower()
    cfg  = get_tour_config(tour)
    paths = get_paths(tour)
    make_dirs(tour)

    MODELS_DIR = paths['models_dir']
    ODDS_DIR   = paths['odds_dir']

    print("=" * 60)
    print(f"CALIBRATION DES SEUILS — {tour.upper()}")
    print("=" * 60)

    # ── Charger modèle + test set ────────────────────────────────────────────
    splits   = joblib.load(MODELS_DIR / "splits.pkl")
    features = joblib.load(MODELS_DIR / "feature_list.pkl")
    imputer  = joblib.load(MODELS_DIR / "imputer.pkl")
    model    = joblib.load(MODELS_DIR / "xgb_tuned.pkl")
    platt_path = MODELS_DIR / "platt_scaler.pkl"
    platt = joblib.load(platt_path) if platt_path.exists() else None

    if len(splits.get('X_test', [])) > 0:
        X_test = splits['X_test']
        y_test = splits['y_test']
        meta   = splits['meta_test']
        period = "test (≥2025)"
    else:
        X_test = splits['X_valid']
        y_test = splits['y_valid']
        meta   = splits['meta_valid']
        period = "valid (2023-2024)"

    X_imp    = imputer.transform(X_test)
    raw_prob = model.predict_proba(X_imp)[:, 1]
    p1_prob  = platt.predict_proba(raw_prob.reshape(-1, 1))[:, 1] if platt else raw_prob

    df_pred = meta.copy().reset_index(drop=True)
    df_pred['p1_prob'] = p1_prob
    df_pred['target']  = y_test.values
    print(f"  Set        : {period} ({len(df_pred):,} matchs)")

    # ── Charger cotes Pinnacle ───────────────────────────────────────────────
    years = list(range(2023, 2027))
    df_odds = load_real_odds(years, ODDS_DIR, cfg['odds_filename'])
    compound = build_compound_lastnames(df_odds)

    # ── Jointure ─────────────────────────────────────────────────────────────
    df_joined = join_odds_to_predictions(df_pred, df_odds, compound)
    df_joined = df_joined[df_joined['PSW'].notna()].copy()
    print(f"  Jointure   : {len(df_joined):,} matchs avec cotes Pinnacle")

    # ── Collecter tous les bets bruts ────────────────────────────────────────
    bets = collect_all_bets(df_joined, odds_col_w='PSW', odds_col_l='PSL')
    print(f"  Bets bruts : {len(bets):,} (2 côtés × {len(df_joined):,} matchs)")

    if args.save:
        out = MODELS_DIR / "bets_raw.csv"
        bets.to_csv(out, index=False)
        print(f"  Bets sauvés : {out}")

    # ── Sweep ────────────────────────────────────────────────────────────────
    print("\n── Sweep des seuils ─────────────────────────────────────")
    thresholds = np.arange(0.00, 0.51, 0.01)
    df_sweep = sweep(bets, thresholds, min_bets=args.min_bets)

    # ── Tableau surface × niveau ─────────────────────────────────────────────
    print("\n── ROI par surface × niveau (edge ≥ 0%) ─────────────────")
    print(f"  {'Surface':<8} {'Level':<7} {'N':>6} {'WR':>6} {'ROI':>7}  (baseline avant seuil)")
    print("  " + "-" * 42)
    for surf in ['Hard', 'Clay', 'Grass', 'ALL']:
        for lvl in ['G', 'M', 'A', 'ALL']:
            sub = df_sweep[(df_sweep['surface'] == surf) &
                           (df_sweep['level']   == lvl)  &
                           (df_sweep['edge_min'] == 0.00)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            print(f"  {surf:<8} {lvl:<7} {r['n']:>6,.0f} {r['wr']:>5.1%} {r['roi']:>+7.1%}")

    # ── Tableau sweep détaillé par surface+level ─────────────────────────────
    print("\n── ROI selon le seuil d'edge par combinaison ───────────────")
    header = f"  {'Surface':<7} {'Level':<6} {'Seuil':>6}  {'N':>5}  {'WR':>5}  {'ROI':>7}  {'AvgOdd':>7}"
    print(header)
    print("  " + "-" * 52)
    for surf in ['Hard', 'Clay', 'Grass']:
        for lvl in ['G', 'M', 'A']:
            sub = df_sweep[(df_sweep['surface'] == surf) &
                           (df_sweep['level']   == lvl)]
            if sub.empty:
                continue
            # Afficher seuils clés: 0%, 10%, 15%, 20%, 25%, 30%
            for thr in [0.00, 0.10, 0.15, 0.20, 0.25, 0.30]:
                row = sub[sub['edge_min'] == thr]
                if row.empty:
                    continue
                r = row.iloc[0]
                marker = " ◄" if r['roi'] >= 0.02 else ""
                print(f"  {surf:<7} {lvl:<6} {thr:>5.0%}  {r['n']:>5.0f}  "
                      f"{r['wr']:>4.1%}  {r['roi']:>+7.1%}  "
                      f"{r['avg_odd']:>6.2f}{marker}")
            print()

    # ── Recommandations finales ───────────────────────────────────────────────
    print("=" * 60)
    print("RECOMMANDATIONS DE SEUILS (VALUE threshold)")
    print("  (1er seuil avec ROI ≥ +2%, min 25 bets)")
    print("=" * 60)
    print(f"\n  {'Surface':<8} {'Level':<7} {'VALUE thr':>10}  {'ROI@thr':>8}  {'N@thr':>6}")
    print("  " + "-" * 46)

    recs = {}
    for surf in ['Hard', 'Clay', 'Grass']:
        for lvl in ['G', 'M', 'A']:
            r = find_optimal_threshold(df_sweep, surf, lvl, target_roi=0.02,
                                       min_bets=args.min_bets)
            recs[(surf, lvl)] = r
            thr_str = f"{r['thr']:.0%}" if r['thr'] is not None else "N/A"
            roi_str = f"{r['roi']:+.1%}" if r['roi'] is not None else "N/A"
            print(f"  {surf:<8} {lvl:<7} {thr_str:>10}  {roi_str:>8}  {r['n']:>6}")

    # ── Base par surface (médiane des levels) ─────────────────────────────────
    print("\n── Seuils base suggérés (à ajuster manuellement) ───────────")
    for surf in ['Hard', 'Clay', 'Grass']:
        thrs = [recs[(surf, lvl)]['thr'] for lvl in ['G', 'M', 'A']
                if recs[(surf, lvl)]['thr'] is not None]
        if thrs:
            median_thr = float(np.median(thrs))
            print(f"  {surf:<8} base VALUE ~ {median_thr:.0%}  "
                  f"(G={recs[(surf,'G')]['thr']}, "
                  f"M={recs[(surf,'M')]['thr']}, "
                  f"A={recs[(surf,'A')]['thr']})")

    print("\nNote: ces seuils sont pour DQ='high'. Ajouter +3pp pour DQ='medium'.")
    print("      Le seuil EDGE est typiquement base_VALUE - 5pp.")


if __name__ == "__main__":
    main()
