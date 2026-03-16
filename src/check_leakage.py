# check_leakage.py — v2
# Diagnostic data leakage — placer dans tennis_ml/src/

import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "data" / "models"
sys.path.append(str(ROOT / "src"))

splits   = joblib.load(MODELS_DIR / "splits.pkl")
features = joblib.load(MODELS_DIR / "feature_list.pkl")
imputer  = joblib.load(MODELS_DIR / "imputer.pkl")
model    = joblib.load(MODELS_DIR / "xgb_default.pkl")

X_test = splits['X_test']
y_test = splits['y_test']
meta   = splits['meta_test']

# Imputer AVANT les calculs pour éviter les NaN
X_imp = imputer.transform(X_test)
probs = model.predict_proba(X_imp)[:, 1]

# ── Test 1 : distribution des probabilités ───────────────────────────────
print("── Distribution des probabilités ────────────────────")
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
hist, _ = np.histogram(probs, bins=bins)
for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
    print(f"  [{lo:.1f}-{hi:.1f}] : {hist[i]:4d} ({hist[i]/len(probs):.1%})")

print(f"\n  Médiane  : {np.median(probs):.3f}  ← doit être ~0.500")
print(f"  % > 0.90 : {(probs > 0.90).mean():.1%}  ← doit être <5%")
print(f"  % > 0.95 : {(probs > 0.95).mean():.1%}  ← doit être <2%")

# ── Test 2 : corrélations sur X imputé (pas de NaN) ──────────────────────
print("\n── Corrélations features/target (X imputé) ─────────")
corrs = []
for i, feat in enumerate(features):
    col = X_imp[:, i]
    r   = np.corrcoef(col, y_test.values)[0, 1] if col.std() > 0 else 0.0
    corrs.append((feat, abs(r) if not np.isnan(r) else 0.0))

corrs.sort(key=lambda x: x[1], reverse=True)
print("  Top 15 |r| avec target :")
for feat, corr in corrs[:15]:
    flag = " ⚠️  SUSPECT" if corr > 0.5 else ""
    print(f"    {feat:<35} : {corr:.4f}{flag}")

# ── Test 3 : calibration ──────────────────────────────────────────────────
print("\n── Calibration ──────────────────────────────────────")
for lo in np.arange(0.5, 1.0, 0.05):
    hi   = lo + 0.05
    mask = (probs >= lo) & (probs < hi)
    if mask.sum() >= 10:
        wr = y_test.values[mask].mean()
        ok = abs(wr - (lo+hi)/2) < 0.07
        print(f"  [{lo:.2f}-{hi:.2f}] → {mask.sum():4d} paris, "
              f"win rate : {wr:.3f} {'✅' if ok else '⚠️'}")

# ── Test 4 : rolling features ─────────────────────────────────────────────
print("\n── Rolling features (sur X imputé) ──────────────────")
for feat in ['winrate_diff_5', 'p1_streak', 'streak_diff']:
    if feat in features:
        idx = features.index(feat)
        col = X_imp[:, idx]
        r   = np.corrcoef(col, y_test.values)[0, 1]
        print(f"  {feat:<25} max={col.max():.3f} min={col.min():.3f} r={r:.4f}")
