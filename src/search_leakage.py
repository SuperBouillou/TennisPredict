# src/check_leakage.py

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "data" / "models"

splits   = joblib.load(MODELS_DIR / "splits.pkl")
features = joblib.load(MODELS_DIR / "feature_list.pkl")

X_test = splits['X_test']
y_test = splits['y_test']
meta   = splits['meta_test']

df = meta.copy()
df['target']  = y_test.values

# ── Test 1 : distribution des probabilités ────────────────────────────────
imputer = joblib.load(MODELS_DIR / "imputer.pkl")
model   = joblib.load(MODELS_DIR / "xgb_default.pkl")
X_imp   = imputer.transform(X_test)
probs   = model.predict_proba(X_imp)[:, 1]

df['p1_prob'] = probs
print("── Distribution des probabilités (dataset complet) ──")
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
hist, _ = np.histogram(probs, bins=bins)
for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
    print(f"  [{lo:.1f}-{hi:.1f}] : {hist[i]:4d} matchs ({hist[i]/len(probs):.1%})")

# Un modèle sans leakage doit avoir une distribution symétrique
# autour de 0.5 (car p1/p2 assignés aléatoirement)
print(f"\n  Médiane  : {np.median(probs):.3f}  ← doit être ~0.500")
print(f"  % > 0.90 : {(probs > 0.90).mean():.1%}  ← doit être <5%")
print(f"  % > 0.95 : {(probs > 0.95).mean():.1%}  ← doit être <2%")

# ── Test 2 : features suspectes ────────────────────────────────────────────
print("\n── Features les plus corrélées au target ────────────")
df_feat = pd.DataFrame(X_test, columns=features)
df_feat['target'] = y_test.values

corrs = df_feat.corr()['target'].abs().sort_values(ascending=False)
print("  Top 15 corrélations |r| avec target :")
for feat, corr in corrs.head(15).items():
    flag = " ⚠️  SUSPECT" if corr > 0.5 else ""
    print(f"    {feat:<35} : {corr:.4f}{flag}")

# ── Test 3 : ELO look-ahead ────────────────────────────────────────────────
print("\n── Test ELO : pré-match vs post-match ───────────────")
# Charger le dataset complet pour vérifier les ELO
try:
    df_full = pd.read_parquet(ROOT / "data" / "processed" / "matches_features_final.parquet")
    
    # Vérifier si elo_p1 change après le match
    # Si elo_p1 après == elo_p1 avant + delta, c'est bon
    # Si elo_p1 = ELO final (post toutes saisons), c'est du leakage
    
    sample = df_full[df_full['tourney_date'] >= '2023-01-01'].head(5)
    print("  Exemple d'ELOs dans les données 2023 :")
    elo_cols = [c for c in df_full.columns if 'elo' in c.lower()]
    print(f"  Colonnes ELO : {elo_cols}")
    print(sample[['tourney_date','p1_name','p2_name'] + elo_cols[:4]].to_string())
    
except Exception as e:
    print(f"  Erreur : {e}")

# Ajoute ce diagnostic dans check_leakage.py, après la section ELO :

print("\n── Analyse du biais structurel de jointure ──────────")

# Recharger le dataset joint
import joblib
splits = joblib.load(MODELS_DIR / "splits.pkl")
meta   = splits['meta_test']

# Recharger les cotes
import sys
sys.path.append(str(ROOT / "src"))
from backtest_real import load_real_odds, normalize_name_for_join

df_odds = load_real_odds([2023, 2024])
df_odds['key_w'] = df_odds['winner_clean'].apply(normalize_name_for_join)
df_odds['key_l'] = df_odds['loser_clean'].apply(normalize_name_for_join)

# Vérifier : dans Sackmann, p1 est assigné aléatoirement ?
df_test = meta.copy()
df_test['target'] = splits['y_test'].values
df_test['key_p1'] = df_test['p1_name'].apply(normalize_name_for_join)
df_test['key_p2'] = df_test['p2_name'].apply(normalize_name_for_join)

# Combien de fois p1 = winner dans Sackmann ?
print(f"  target=1 (p1 gagne) dans Sackmann : {df_test['target'].mean():.3f}")
print(f"  ← Si ~0.500 : assignation aléatoire OK")

# Vérifier dans tennis-data : winner_key vs nos key_p1
odds_winners = set(df_odds['key_w'].unique())
pred_p1s     = set(df_test['key_p1'].unique())
pred_p2s     = set(df_test['key_p2'].unique())

# Pour les matchs joints, p1 est-il plus souvent le winner odds ?
df_odds['Date'] = pd.to_datetime(df_odds['Date'])
df_odds['date_exact'] = df_odds['Date'].dt.date
df_test['date_exact'] = pd.to_datetime(df_test['tourney_date']).dt.date

# Jointure test : p1 = winner_odds ?
merged_w = df_test.merge(
    df_odds[['date_exact','key_w','key_l']].rename(columns={'key_w':'key_p1','key_l':'key_p2'}),
    on=['date_exact','key_p1','key_p2'], how='inner'
)
merged_l = df_test.merge(
    df_odds[['date_exact','key_w','key_l']].rename(columns={'key_w':'key_p2','key_l':'key_p1'}),
    on=['date_exact','key_p1','key_p2'], how='inner'
)

print(f"\n  Matchs où p1=winner_odds (date exacte) : {len(merged_w):,} | target=1 : {merged_w['target'].mean():.3f}")
print(f"  Matchs où p1=loser_odds  (date exacte) : {len(merged_l):,} | target=1 : {merged_l['target'].mean():.3f}")
print(f"\n  ← Si target=1≈0.995 dans merged_w : la source Sackmann")
print(f"    encode winner_id en premier → p1 = toujours le vainqueur")
print(f"    Le flag 'aléatoire' dans restructure_data.py n'a pas fonctionné")

# ── Test 4 : Rolling features — shift correct ? ───────────────────────────
print("\n── Test rolling features ─────────────────────────────")
roll_cols = [c for c in features if 'winrate' in c or 'streak' in c]
print(f"  Features rolling : {roll_cols}")

# Le winrate_5 de p1 AVANT le match doit être calculé sur les 5 matchs PRÉCÉDENTS
# Vérifier : pour une grosse victoire upsets, le winrate AVANT = ?
if 'winrate_diff_5' in features:
    idx_feat = features.index('winrate_diff_5')
    wd5 = X_test[:, idx_feat]
    print(f"  winrate_diff_5 — max : {wd5.max():.3f} | min : {wd5.min():.3f}")
    print(f"  Si max ≈ 1.0 → le favori avait 100% de winrate → SUSPECT")
    print(f"  Corrélation winrate_diff_5 / target : {np.corrcoef(wd5, y_test.values)[0,1]:.4f}")