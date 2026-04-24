"""
Signal Selection Optimization — Data-driven threshold analysis
Run: python src/analyze_signals.py
"""
import warnings
warnings.filterwarnings('ignore')

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def roi(df):
    s = df['stake'].sum()
    return round(df['pnl'].sum() / s * 100, 2) if s > 0 else 0.0


def win_rate(df):
    return round(df['won'].mean() * 100, 1) if len(df) else 0.0


def avg_odd(df):
    return round(df['odd'].mean(), 2) if len(df) else 0.0


def section(title):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print('='*62)


def analyse_tour(tour: str):
    mdir = ROOT / 'data' / 'models' / tour

    # Prefer candidates file (all bets before filter) for threshold analysis
    candidates_path = mdir / 'backtest_all_candidates.parquet'
    real_path       = mdir / 'backtest_real_Pinnacle.parquet'

    if candidates_path.exists():
        df_all = pd.read_parquet(candidates_path)
        print(f"\n[{tour.upper()}] backtest_all_candidates: {len(df_all)} rows, cols: {list(df_all.columns)}")
    else:
        print(f"[{tour.upper()}] backtest_all_candidates.parquet NOT FOUND — skipping")
        df_all = None

    if real_path.exists():
        df_real = pd.read_parquet(real_path)
        print(f"[{tour.upper()}] backtest_real_Pinnacle:    {len(df_real)} rows")
    else:
        print(f"[{tour.upper()}] backtest_real_Pinnacle.parquet NOT FOUND")
        df_real = None

    df = df_all if df_all is not None else df_real
    if df is None:
        return

    # ── 0. Baseline ───────────────────────────────────────────────────────────
    section(f"{tour.upper()} — BASELINE (all candidates)")
    print(f"  Bets:      {len(df)}")
    print(f"  Win rate:  {win_rate(df)}%")
    print(f"  ROI:       {roi(df)}%")
    print(f"  Avg edge:  {df['edge'].mean()*100:.1f}%")
    print(f"  Avg odd:   {avg_odd(df)}")
    if 'clv' in df.columns:
        print(f"  Avg CLV:   {df['clv'].mean()*100:.2f}%")

    # ── 1. Edge threshold grid search ─────────────────────────────────────────
    section(f"{tour.upper()} — EDGE THRESHOLD GRID SEARCH")
    print(f"  {'Threshold':>10}  {'N':>6}  {'Win%':>7}  {'ROI%':>8}  {'AvgOdd':>8}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*7}  {'-'*8}  {'-'*8}")
    best_roi, best_thr = -99, 0
    for thr in np.arange(0.02, 0.30, 0.01):
        sub = df[df['edge'] >= thr]
        if len(sub) < 30:
            break
        r = roi(sub)
        print(f"  {thr*100:>9.0f}%  {len(sub):>6}  {win_rate(sub):>6.1f}%  {r:>+8.2f}%  {avg_odd(sub):>8.2f}")
        if r > best_roi:
            best_roi, best_thr = r, thr
    print(f"\n  ★ Best single threshold: edge ≥ {best_thr*100:.0f}%  →  ROI {best_roi:+.2f}%")

    # ── 2. ROI by surface (above 5% edge) ─────────────────────────────────────
    if 'surface' in df.columns:
        section(f"{tour.upper()} — BY SURFACE (edge ≥ 5%)")
        filt = df[df['edge'] >= 0.05]
        print(f"  {'Surface':>10}  {'N':>6}  {'Win%':>7}  {'ROI%':>8}  {'AvgOdd':>8}  {'AvgEdge':>9}")
        for surf, g in sorted(filt.groupby('surface'), key=lambda x: roi(x[1]), reverse=True):
            print(f"  {surf:>10}  {len(g):>6}  {win_rate(g):>6.1f}%  {roi(g):>+8.2f}%  {avg_odd(g):>8.2f}  {g['edge'].mean()*100:>8.1f}%")

    # ── 3. ROI by odds range (above 5% edge) ──────────────────────────────────
    section(f"{tour.upper()} — BY ODDS RANGE (edge ≥ 5%)")
    filt = df[df['edge'] >= 0.05].copy()
    bins   = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 100.0]
    labels = ['1.0–1.5', '1.5–2.0', '2.0–2.5', '2.5–3.0', '3.0–4.0', '4.0–6.0', '6.0+']
    filt['odds_bin'] = pd.cut(filt['odd'], bins=bins, labels=labels)
    print(f"  {'Odds':>10}  {'N':>6}  {'Win%':>7}  {'ROI%':>8}  {'AvgEdge':>9}")
    for ob, g in filt.groupby('odds_bin', observed=True):
        print(f"  {str(ob):>10}  {len(g):>6}  {win_rate(g):>6.1f}%  {roi(g):>+8.2f}%  {g['edge'].mean()*100:>8.1f}%")

    # ── 4. ROI by tournament level ────────────────────────────────────────────
    if 'level' in df.columns:
        section(f"{tour.upper()} — BY TOURNAMENT LEVEL (edge ≥ 5%)")
        filt = df[df['edge'] >= 0.05]
        print(f"  {'Level':>12}  {'N':>6}  {'Win%':>7}  {'ROI%':>8}  {'AvgOdd':>8}")
        for lvl, g in sorted(filt.groupby('level'), key=lambda x: roi(x[1]), reverse=True):
            print(f"  {str(lvl):>12}  {len(g):>6}  {win_rate(g):>6.1f}%  {roi(g):>+8.2f}%  {avg_odd(g):>8.2f}")

    # ── 5. Surface × Edge tier cross-tab ─────────────────────────────────────
    if 'surface' in df.columns:
        section(f"{tour.upper()} — SURFACE × EDGE TIER (N ≥ 20)")
        df2 = df.copy()
        df2['edge_tier'] = pd.cut(df2['edge'],
            bins=[-1, 0.05, 0.08, 0.12, 0.16, 0.20, 1.0],
            labels=['<5%', '5–8%', '8–12%', '12–16%', '16–20%', '>20%'])
        tbl = (df2.groupby(['surface', 'edge_tier'], observed=True)
                  .apply(lambda g: pd.Series({'n': len(g), 'roi': roi(g), 'wr': win_rate(g)}))
                  .reset_index())
        tbl = tbl[tbl['n'] >= 20].sort_values('roi', ascending=False)
        print(f"  {'Surface':>8}  {'Edge':>7}  {'N':>6}  {'Win%':>7}  {'ROI%':>8}")
        for _, r2 in tbl.head(20).iterrows():
            print(f"  {r2['surface']:>8}  {r2['edge_tier']:>7}  {int(r2['n']):>6}  {r2['wr']:>6.1f}%  {r2['roi']:>+8.2f}%")

    # ── 6. Favourite vs underdog ─────────────────────────────────────────────
    if 'p1_is_winner_odds' in df.columns or 'odd' in df.columns:
        section(f"{tour.upper()} — FAVOURITE vs UNDERDOG (edge ≥ 5%)")
        filt = df[df['edge'] >= 0.05].copy()
        filt['is_fav'] = filt['odd'] < 2.0
        for label, g in [('Favori (odd < 2)', filt[filt['is_fav']]),
                         ('Outsider (odd ≥ 2)', filt[~filt['is_fav']])]:
            print(f"  {label:<22}  N={len(g):>5}  Win={win_rate(g):>5.1f}%  ROI={roi(g):>+7.2f}%  AvgOdd={avg_odd(g):.2f}")

    # ── 7. Optimal combined filter recommendation ────────────────────────────
    section(f"{tour.upper()} — OPTIMAL FILTER SEARCH (edge + odds range)")
    best_combos = []
    for thr in np.arange(0.05, 0.22, 0.01):
        for odd_max in [3.0, 4.0, 5.0, 6.0, None]:
            sub = df[df['edge'] >= thr]
            if odd_max:
                sub = sub[sub['odd'] < odd_max]
            if len(sub) < 40:
                continue
            r = roi(sub)
            best_combos.append({'edge_min': thr, 'odd_max': odd_max or 99, 'n': len(sub), 'roi': r, 'wr': win_rate(sub)})

    combos_df = pd.DataFrame(best_combos).sort_values('roi', ascending=False)
    print(f"  {'Edge≥':>7}  {'Odd<':>6}  {'N':>6}  {'Win%':>7}  {'ROI%':>8}")
    for _, row in combos_df.head(15).iterrows():
        odd_str = f"{row['odd_max']:.0f}" if row['odd_max'] < 90 else "∞"
        print(f"  {row['edge_min']*100:>6.0f}%  {odd_str:>6}  {int(row['n']):>6}  {row['wr']:>6.1f}%  {row['roi']:>+8.2f}%")

    print()


if __name__ == '__main__':
    tours = sys.argv[1:] if len(sys.argv) > 1 else ['atp', 'wta']
    for t in tours:
        analyse_tour(t)
    print("\nDone.")
