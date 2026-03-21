"""
Recalibrate Platt scaler against Pinnacle no-vig probabilities.

Replaces the current LogisticRegression (calibrated on historical outcomes)
with a LinearRegression mapping raw XGBoost scores → Pinnacle no-vig probability.

After recalibration: edge = cal_prob - pinnacle_novid = true market disagreement.
Mean edge across all matches drops to ~0; VALUE BET only flags genuine outliers.

Usage:
    python src/recalibrate_platt.py --tour atp
    python src/recalibrate_platt.py --tour atp --years 2023 2024
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from config import get_paths
from backtest_real import (
    load_real_odds,
    build_compound_lastnames,
    join_odds_to_predictions,
)


def recalibrate(tour: str, years: list[int]) -> None:
    paths = get_paths(tour)
    models_dir = paths['models_dir']
    odds_dir   = paths['odds_dir']

    print(f"\n=== Recalibrating Platt scaler: {tour.upper()} (years={years}) ===\n")

    # ── 1. Load splits ────────────────────────────────────────────────────────
    splits_path = models_dir / "splits.pkl"
    if not splits_path.exists():
        print(f"[ERROR] splits.pkl not found: {splits_path}")
        return

    splits = joblib.load(splits_path)

    X_valid   = splits.get('X_valid')
    y_valid   = splits.get('y_valid')
    meta_valid = splits.get('meta_valid')

    if X_valid is None or y_valid is None or meta_valid is None:
        print("[ERROR] splits.pkl missing X_valid/y_valid/meta_valid")
        return

    print(f"Validation set: {len(X_valid)} rows")

    # ── 2. Load model + imputer ───────────────────────────────────────────────
    model   = joblib.load(models_dir / "xgb_tuned.pkl")
    imputer = joblib.load(models_dir / "imputer.pkl")
    feature_list = joblib.load(models_dir / "feature_list.pkl")

    # ── 3. Compute raw XGBoost probabilities on validation set ────────────────
    X_df   = pd.DataFrame(X_valid, columns=feature_list) if not isinstance(X_valid, pd.DataFrame) else X_valid
    X_imp  = imputer.transform(X_df)
    raw_probs = model.predict_proba(X_imp)[:, 1]  # P(p1 wins)

    print(f"Raw prob range: [{raw_probs.min():.3f}, {raw_probs.max():.3f}]")

    # ── 4. Load Pinnacle odds ─────────────────────────────────────────────────
    odds_filename = lambda y: f"{tour}_{y}.xlsx"
    try:
        df_odds = load_real_odds(years, odds_dir, odds_filename)
    except Exception as e:
        print(f"[ERROR] Could not load odds: {e}")
        return

    if 'PSW' not in df_odds.columns or 'PSL' not in df_odds.columns:
        print("[ERROR] Pinnacle odds (PSW/PSL) not found in odds file")
        return

    # ── 5. Join odds → validation predictions ────────────────────────────────
    meta = meta_valid.copy() if isinstance(meta_valid, pd.DataFrame) else pd.DataFrame(meta_valid)

    # join_odds_to_predictions needs a 'target' column for its diagnostic print
    if 'target' not in meta.columns:
        meta['target'] = y_valid.values if hasattr(y_valid, 'values') else y_valid

    # Ensure meta has p1_name / p2_name columns
    if 'p1_name' not in meta.columns or 'p2_name' not in meta.columns:
        print("[ERROR] meta_valid missing p1_name/p2_name columns")
        return

    compound_lastnames = build_compound_lastnames(df_odds)
    meta_joined = join_odds_to_predictions(meta, df_odds, compound_lastnames)

    # ── 6. Build target: Pinnacle no-vig prob for p1 ─────────────────────────
    # PSW = Pinnacle odd for winner, PSL = Pinnacle odd for loser
    # p1_is_winner_odds = True → PSW corresponds to p1
    valid_mask = (
        meta_joined['PSW'].notna() &
        meta_joined['PSL'].notna() &
        (meta_joined['PSW'] > 1.0) &
        (meta_joined['PSL'] > 1.0) &
        meta_joined['p1_is_winner_odds'].notna()
    )

    n_joined = valid_mask.sum()
    join_rate = n_joined / len(meta_joined) * 100
    print(f"Joined: {n_joined}/{len(meta_joined)} ({join_rate:.1f}%)")

    if n_joined < 100:
        print("[ERROR] Too few joined samples — check odds files and years")
        return

    psw = meta_joined.loc[valid_mask, 'PSW'].values.astype(float)
    psl = meta_joined.loc[valid_mask, 'PSL'].values.astype(float)

    # After join_odds_to_predictions' swap, PSW is ALWAYS p1's odd and PSL is p2's odd,
    # regardless of who won. No flip needed — just compute P(p1 wins) directly.
    total_impl = 1.0 / psw + 1.0 / psl
    y_target = (1.0 / psw) / total_impl  # P(p1 wins) per Pinnacle no-vig

    # Corresponding raw probs
    valid_idx = np.where(valid_mask)[0]
    X_sub = raw_probs[valid_idx].reshape(-1, 1)

    print(f"\ny_target range: [{y_target.min():.3f}, {y_target.max():.3f}]")
    print(f"X_sub range:    [{X_sub.min():.3f}, {X_sub.max():.3f}]")

    # ── 7. Fit LinearRegression ───────────────────────────────────────────────
    lr = LinearRegression()
    lr.fit(X_sub, y_target)

    y_pred = lr.predict(X_sub)
    mae  = mean_absolute_error(y_target, y_pred)
    r2   = r2_score(y_target, y_pred)
    print(f"\nLinearRegression fit:")
    print(f"  Coef:      {lr.coef_[0]:.4f}")
    print(f"  Intercept: {lr.intercept_:.4f}")
    print(f"  R²:        {r2:.4f}")
    print(f"  MAE:       {mae:.4f}")

    # Evaluate calibration: mean edge across all matches (should be ~0)
    cal_probs = np.clip(lr.predict(raw_probs.reshape(-1, 1)), 0.01, 0.99)
    print(f"\nAll-match cal_prob distribution:")
    print(f"  mean={cal_probs.mean():.3f}  std={cal_probs.std():.3f}  "
          f"min={cal_probs.min():.3f}  max={cal_probs.max():.3f}")

    # ── 8. Save ───────────────────────────────────────────────────────────────
    out_path = models_dir / "platt_pinnacle.pkl"
    joblib.dump(lr, out_path)
    print(f"\nSaved: {out_path}")
    print("→ Restart webapp to use new scaler (main.py auto-detects platt_pinnacle.pkl)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recalibrate Platt scaler against Pinnacle")
    parser.add_argument("--tour",  default="atp", choices=["atp", "wta"])
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024])
    args = parser.parse_args()
    recalibrate(args.tour, args.years)


if __name__ == "__main__":
    main()
