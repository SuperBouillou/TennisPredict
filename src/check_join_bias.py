# check_join_bias.py — v2
# Analyse biais de sélection jointure — placer dans tennis_ml/src/

import argparse
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from config import get_tour_config, get_paths
from backtest_real import load_real_odds, normalize_name_for_join, join_odds_to_predictions, build_compound_lastnames

parser = argparse.ArgumentParser()
parser.add_argument('--tour', default='atp', choices=['atp', 'wta'])
args = parser.parse_args()

cfg        = get_tour_config(args.tour)
paths      = get_paths(args.tour)
MODELS_DIR = paths['models_dir']
ODDS_DIR   = paths['odds_dir']

# ── Chargement ────────────────────────────────────────────────────────────
splits  = joblib.load(MODELS_DIR / "splits.pkl")
imputer = joblib.load(MODELS_DIR / "imputer.pkl")
model   = joblib.load(MODELS_DIR / "xgb_tuned.pkl")

platt_path = MODELS_DIR / "platt_scaler.pkl"
platt = joblib.load(platt_path) if platt_path.exists() else None

X_imp    = imputer.transform(splits['X_test'])
raw_prob = model.predict_proba(X_imp)[:, 1]
probs    = platt.predict_proba(raw_prob.reshape(-1, 1))[:, 1] if platt else raw_prob

df_pred = splits['meta_test'].copy().reset_index(drop=True)
df_pred['p1_prob'] = probs
df_pred['target']  = splits['y_test'].values

# Ajouter elo_diff depuis meta si disponible, sinon depuis features
features = joblib.load(MODELS_DIR / "feature_list.pkl")
if 'elo_diff' in features:
    idx = features.index('elo_diff')
    df_pred['elo_diff'] = X_imp[:, idx]
elif 'elo_diff' in splits['meta_test'].columns:
    df_pred['elo_diff'] = splits['meta_test']['elo_diff'].values
else:
    # Recalculer depuis le dataset complet
    df_full = pd.read_parquet(ROOT / "data" / "processed" / "matches_features_final.parquet")
    df_pred = df_pred.merge(
        df_full[['match_key','elo_diff']],
        on='match_key', how='left'
    ) if 'match_key' in df_pred.columns else df_pred.assign(elo_diff=0)

# ── Jointure ──────────────────────────────────────────────────────────────
df_odds              = load_real_odds([2023, 2024], ODDS_DIR, cfg['odds_filename'])
compound_lastnames   = build_compound_lastnames(df_odds)
df_joined            = join_odds_to_predictions(df_pred, df_odds, compound_lastnames)

# ── Analyse biais ─────────────────────────────────────────────────────────
print("\n" + "="*55)
print("ANALYSE DU BIAIS DE SÉLECTION")
print("="*55)

joined   = df_joined[df_joined['PSW'].notna()].copy()
unjoined = df_joined[df_joined['PSW'].isna()].copy()

# 1. Taux victoire du favori ELO
if 'elo_diff' in joined.columns and joined['elo_diff'].notna().any():
    j_fav  = ((joined['elo_diff']   > 0) == (joined['target'] == 1)).mean()
    u_fav  = ((unjoined['elo_diff'] > 0) == (unjoined['target'] == 1)).mean()
    print(f"\n  Taux victoire favori ELO :")
    print(f"    Joints    : {j_fav:.3f}")
    print(f"    Non-joints: {u_fav:.3f}")
    print(f"  ← Si joints >> non-joints : biais confirmé")

    print(f"\n  ELO diff absolu :")
    print(f"    Joints    : {joined['elo_diff'].abs().mean():.1f}")
    print(f"    Non-joints: {unjoined['elo_diff'].abs().mean():.1f}")

# 2. Confiance modèle
print(f"\n  Confiance modèle |prob - 0.5| :")
print(f"    Joints    : {(joined['p1_prob'] - 0.5).abs().mean():.3f}")
print(f"    Non-joints: {(unjoined['p1_prob'] - 0.5).abs().mean():.3f}")
print(f"  ← Si joints > non-joints : biais vers matchs faciles")

# 3. Distribution cotes PSW
print(f"\n  Distribution cotes PSW :")
bins   = [1.0, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0, 5.0, 100.0]
labels = ['1.0-1.3','1.3-1.5','1.5-1.75','1.75-2.0',
          '2.0-2.5','2.5-3.0','3.0-5.0','5.0+']
joined['cote_bin'] = pd.cut(joined['PSW'], bins=bins, labels=labels)
for lbl in labels:
    sub = joined[joined['cote_bin'] == lbl]
    if len(sub) > 0:
        wr  = sub['target'].mean()
        pct = len(sub) / len(joined)
        print(f"    {lbl:<12}: {len(sub):4d} ({pct:.1%}) | win rate p1: {wr:.3f}")

# 4. Value bets résumé
print(f"\n  Résumé value bets Pinnacle (prob>0.55, edge>3%) :")
margin  = 1/joined['PSW'] + 1/joined['PSL'] - 1
bk_imp  = (1/joined['PSW']) / (1 + margin)
vb      = joined[(joined['p1_prob'] > 0.55) & ((joined['p1_prob'] - bk_imp) > 0.03)]
if len(vb) > 0:
    print(f"    N value bets          : {len(vb):,}")
    print(f"    Win rate réel         : {vb['target'].mean():.3f}")
    print(f"    Cote PSW moyenne      : {vb['PSW'].mean():.2f}")
    print(f"    % p1=favori bookmaker : {(vb['PSW'] < vb['PSL']).mean():.1%}")

# 5. Benchmark naïf : toujours parier sur le favori
print(f"\n  BENCHMARK : parier systématiquement sur le favori PSW")
naive_pnl = []
for _, row in joined.iterrows():
    if pd.isna(row['PSW']) or pd.isna(row['PSL']):
        continue
    if row['PSW'] <= row['PSL']:
        won = (row['target'] == 1)
        odd = float(row['PSW'])
    else:
        won = (row['target'] == 0)
        odd = float(row['PSL'])
    naive_pnl.append(10*(odd-1) if won else -10)

naive_roi = sum(naive_pnl) / (len(naive_pnl) * 10)
naive_wr  = sum(1 for p in naive_pnl if p > 0) / len(naive_pnl)
print(f"    Paris    : {len(naive_pnl):,}")
print(f"    Win rate : {naive_wr:.3f}")
print(f"    ROI      : {naive_roi:+.2%}")

# 6. Conclusion
print(f"\n" + "="*55)
print("CONCLUSION")
print("="*55)
if naive_roi > 0.05:
    print(f"  ❌ BIAIS CONFIRMÉ  (ROI naïf = {naive_roi:+.2%})")
    print(f"     La jointure sélectionne les matchs favoris faciles.")
    print(f"     Objectif : améliorer la jointure pour atteindre >85%.")
elif naive_roi > -0.01:
    print(f"  ⚠️  BIAIS LÉGER    (ROI naïf = {naive_roi:+.2%})")
    print(f"     Jointure légèrement biaisée mais partiellement exploitable.")
else:
    print(f"  ✅ PAS DE BIAIS   (ROI naïf = {naive_roi:+.2%})")
    print(f"     Le sous-ensemble joint est représentatif.")
    print(f"     Notre edge modèle est potentiellement réel → Pilier 5 !")
