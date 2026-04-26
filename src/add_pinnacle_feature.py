"""
Inject Pinnacle no-vig probability as a feature into matches_features_final.parquet.

The Pinnacle implied probability (no-vig) is the market consensus estimate of P(p1 wins).
Adding it as a feature lets XGBoost learn the systematic disagreements between our
model and the market — the core signal for value betting.

Matches without Pinnacle coverage (pre-2010, qualifiers, etc.) receive NaN,
which the imputer handles with fill_value=0.5 (neutral prior).

Usage:
    python src/add_pinnacle_feature.py --tour atp
    python src/add_pinnacle_feature.py --tour atp --years 2010 2025
    python src/add_pinnacle_feature.py --tour wta
"""
from __future__ import annotations

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import get_paths
from backtest_real import _read_excel_auto, build_compound_lastnames, join_odds_to_predictions


def _load_odds_robust(years: list[int], odds_dir: Path, odds_filename) -> pd.DataFrame:
    """
    Load odds files year by year, auto-detecting OLE (.xls) vs OOXML (.xlsx) format.
    Skips missing or unreadable files instead of crashing.
    """
    dfs = []
    for year in years:
        fname = odds_filename(year)
        path  = odds_dir / fname
        if not path.exists():
            continue
        try:
            df = _read_excel_auto(path)
            df['year'] = year
            dfs.append(df)
            print(f"  {year}: {len(df):,} rows")
        except Exception as e:
            print(f"  {year}: SKIP ({e})")
    if not dfs:
        raise ValueError(f"No odds files loaded from {odds_dir}")
    combined = pd.concat(dfs, ignore_index=True)
    # Minimal cleaning needed by join_odds_to_predictions
    combined['Date'] = pd.to_datetime(combined['Date'], errors='coerce')
    combined = combined.dropna(subset=['Date'])
    combined = combined[combined.get('Comment', pd.Series('Completed', index=combined.index)) == 'Completed'] \
        if 'Comment' in combined.columns else combined
    for col in ['B365W', 'B365L', 'PSW', 'PSL', 'MaxW', 'MaxL', 'AvgW', 'AvgL']:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors='coerce')
    combined['winner_clean'] = combined['Winner'].str.strip() if 'Winner' in combined.columns else ''
    combined['loser_clean']  = combined['Loser'].str.strip()  if 'Loser'  in combined.columns else ''
    return combined


def add_pinnacle_feature(tour: str, years: list[int]) -> None:
    paths         = get_paths(tour)
    processed_dir = paths['processed_dir']
    odds_dir      = paths['odds_dir']

    print(f"{'='*60}")
    print(f"INJECT PINNACLE FEATURE — {tour.upper()}")
    print(f"{'='*60}\n")

    # ── 1. Load matches dataset ───────────────────────────────────────────────
    parquet_path = processed_dir / "matches_features_final.parquet"
    if not parquet_path.exists():
        print(f"[ERROR] Not found: {parquet_path}")
        return

    df = pd.read_parquet(parquet_path)
    print(f"Dataset: {len(df):,} matches × {df.shape[1]} columns")

    required = {'tourney_date', 'p1_name', 'p2_name', 'target'}
    missing  = required - set(df.columns)
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        return

    # ── 2. Load Pinnacle odds (all available years) ───────────────────────────
    odds_filename = lambda y: f"{tour}_{y}.xlsx"
    available_years = [y for y in years if (odds_dir / odds_filename(y)).exists()]
    if not available_years:
        print(f"[ERROR] No odds files found in {odds_dir}")
        return

    print(f"Loading odds for years: {available_years}")
    try:
        df_odds = _load_odds_robust(available_years, odds_dir, odds_filename)
    except Exception as e:
        print(f"[ERROR] Could not load odds: {e}")
        return

    if 'PSW' not in df_odds.columns or 'PSL' not in df_odds.columns:
        print("[ERROR] Pinnacle odds columns PSW/PSL not found")
        return

    # ── 3. Join odds to matches ───────────────────────────────────────────────
    # Use only the columns needed for the join to keep memory manageable
    meta_cols = ['tourney_date', 'p1_name', 'p2_name', 'target']
    df_meta   = df[meta_cols].copy().reset_index(drop=True)

    compound_lastnames = build_compound_lastnames(df_odds)

    print(f"\nJoining {len(df_meta):,} matches to {len(df_odds):,} odds rows...")
    df_joined = join_odds_to_predictions(df_meta, df_odds, compound_lastnames)

    # ── 4. Compute Pinnacle no-vig probability for p1 ─────────────────────────
    # After join_odds_to_predictions, PSW is ALWAYS p1's odd regardless of who won.
    valid = (
        df_joined['PSW'].notna() &
        df_joined['PSL'].notna() &
        (df_joined['PSW'] > 1.01) &
        (df_joined['PSL'] > 1.01)
    )
    n_joined = int(valid.sum())
    pct      = n_joined / len(df_joined) * 100
    print(f"Joined: {n_joined:,} / {len(df_joined):,} ({pct:.1f}%)")

    psw        = df_joined.loc[valid, 'PSW'].values.astype(float)
    psl        = df_joined.loc[valid, 'PSL'].values.astype(float)
    total_impl = 1.0 / psw + 1.0 / psl
    probs_novid = (1.0 / psw) / total_impl  # P(p1 wins) — no-vig

    pinnacle_col = np.full(len(df), np.nan)
    valid_positions = np.where(valid.values)[0]
    pinnacle_col[valid_positions] = probs_novid

    print(f"\nPinnacle p1 prob (joined rows):")
    print(f"  mean = {probs_novid.mean():.4f}  (expected ~0.50)")
    print(f"  std  = {probs_novid.std():.4f}")
    print(f"  min  = {probs_novid.min():.4f}  max = {probs_novid.max():.4f}")

    # Sanity check: correlation with target (p1 wins)
    mask      = ~np.isnan(pinnacle_col)
    corr      = np.corrcoef(pinnacle_col[mask], df['target'].values[mask])[0, 1]
    print(f"  corr(pinnacle_prob, target) = {corr:.4f}  (expected ~0.20-0.35)")

    # ── 5. Write enriched parquet ─────────────────────────────────────────────
    df['pinnacle_p1_prob'] = pinnacle_col
    df.to_parquet(parquet_path, index=False)

    n_non_nan = int(mask.sum())
    print(f"\nSaved: {parquet_path}")
    print(f"  pinnacle_p1_prob: {n_non_nan:,} values, {len(df)-n_non_nan:,} NaN")
    print(f"\n→ Next: python src/prepare_ml_dataset.py --tour {tour}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject Pinnacle no-vig prob into training data")
    parser.add_argument('--tour',  default='atp', choices=['atp', 'wta'])
    parser.add_argument('--years', nargs=2, type=int, default=[2010, 2025],
                        metavar=('START', 'END'),
                        help="Year range [START, END] inclusive (default: 2010 2025)")
    args = parser.parse_args()

    years = list(range(args.years[0], args.years[1] + 1))
    add_pinnacle_feature(args.tour, years)


if __name__ == "__main__":
    main()
