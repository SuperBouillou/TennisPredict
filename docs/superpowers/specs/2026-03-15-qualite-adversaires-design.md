# Design Spec — Win Rate Pondéré par Qualité Adversaire

**Date :** 2026-03-15
**Statut :** Approuvé
**Scope :** `compute_rolling_features.py`, `prepare_ml_dataset.py`

---

## Contexte

Le modèle XGBoost actuel (~78% accuracy) utilise des win rates bruts (5/10/20 matchs) qui traitent chaque victoire de manière identique. Cette feature vise à capturer la **qualité du calendrier récent** d'un joueur.

---

## Fonction de pondération

```
w = 1 / log2(opponent_rank + 2)
```

- Rang #1 → w ≈ 1.000, Rang #10 → w ≈ 0.585, Rang #50 → w ≈ 0.373, Rang #200 → w ≈ 0.263

### Normalisation (win rate pondéré vrai, ∈ [0, 1])

```
winrate_quality_N = rolling_sum(won × w, N) / rolling_sum(w, N)
```

Constantes : `RANK_FALLBACK = 300` (rang pour adversaires sans classement, supérieur au seuil ATP ~250), `MIN_PERIODS = 3`.

---

## Features générées (9)

`p1/p2/diff` × `winrate_quality_5/10/20`. Plage [0, 1], moyenne attendue ~0.45-0.55.

---

## Propagation dans le pipeline de fichiers

Le pipeline de fichiers est le suivant (ordre strict) :

```
matches_with_elo.parquet          ← lu par compute_rolling_features.py
    ↓ (ajoute rolling + quality columns)
matches_with_features.parquet     ← écrit par compute_rolling_features.py
    ↓ (lu par compute_h2h.py, qui ajoute colonnes h2h et passe toutes les colonnes existantes)
matches_with_h2h.parquet          ← écrit par compute_h2h.py
    ↓ (lu par compute_contextual_features.py, qui ajoute colonnes contextuelles)
matches_features_final.parquet    ← écrit par compute_contextual_features.py
    ↓
prepare_ml_dataset.py             ← lit matches_features_final.parquet
```

Les scripts `compute_h2h.py` et `compute_contextual_features.py` font des `df.merge()` / `df[new_cols] = ...` sans supprimer les colonnes existantes. Les 9 nouvelles colonnes `winrate_quality_*` survivent donc à travers toute la chaîne jusqu'à `matches_features_final.parquet`, qui est la source de `prepare_ml_dataset.py`.

**Succès criterion 1 doit donc être vérifié dans `matches_features_final.parquet`**, pas seulement `matches_with_features.parquet`.

---

## Implémentation détaillée

### Fichier 1 : `src/compute_rolling_features.py`

#### `build_player_match_history()` — Ajout de `opponent_rank`

```python
def build_player_match_history(df: pd.DataFrame) -> pd.DataFrame:
    stat_cols = ROLLING_STATS + ['rank', 'rank_points', 'age']

    # Vue joueur 1
    p1_cols = {f'p1_{s}': s for s in stat_cols if f'p1_{s}' in df.columns}
    df_p1 = df[['tourney_date', 'tourney_level', 'surface',
                 'p1_id', 'target'] + list(p1_cols.keys())].copy()
    df_p1 = df_p1.rename(columns={'p1_id': 'player_id', 'target': 'won'})
    df_p1 = df_p1.rename(columns=p1_cols)
    # NOUVEAU : rang de l'adversaire = rang de p2 dans la vue p1
    df_p1['opponent_rank'] = df['p2_rank'].values

    # Vue joueur 2
    p2_cols = {f'p2_{s}': s for s in stat_cols if f'p2_{s}' in df.columns}
    df_p2 = df[['tourney_date', 'tourney_level', 'surface',
                 'p2_id', 'target'] + list(p2_cols.keys())].copy()
    df_p2 = df_p2.rename(columns={'p2_id': 'player_id'})
    df_p2['won'] = 1 - df_p2['target']
    df_p2 = df_p2.drop(columns=['target'])
    df_p2 = df_p2.rename(columns=p2_cols)
    # NOUVEAU : rang de l'adversaire = rang de p1 dans la vue p2
    df_p2['opponent_rank'] = df['p1_rank'].values

    df_history = pd.concat([df_p1, df_p2], ignore_index=True)
    df_history = df_history.sort_values(['player_id', 'tourney_date']).reset_index(drop=True)
    return df_history
```

Note : `df['p1_rank']` et `df['p2_rank']` existent dans `matches_with_elo.parquet` (colonnes issues de `winner_rank`/`loser_rank` Sackmann, renumérotées lors de `restructure_data.py`).

#### `compute_rolling_stats()` — Calcul du win rate pondéré

Ajouter après le bloc "Win rate glissant global" :

```python
# Win rate pondéré par qualité adversaire
RANK_FALLBACK = 300
MIN_PERIODS   = 3

opp_weight = 1.0 / np.log2(df_pid['opponent_rank'].fillna(RANK_FALLBACK) + 2)
win_quality = df_pid['won'] * opp_weight

for w in WINDOWS:
    num = win_quality.shift(1).rolling(w, min_periods=MIN_PERIODS).sum()
    den = opp_weight.shift(1).rolling(w, min_periods=MIN_PERIODS).sum()
    row[f'winrate_quality_{w}'] = (num / den).values
```

#### `join_rolling_to_ml()` — Propagation des colonnes

La détection des colonnes rolling se fait via :
```python
rolling_cols = [c for c in df_history.columns
                if any(x in c for x in ['winrate', 'roll', 'streak'])]
```
Les colonnes `winrate_quality_*` seront automatiquement incluses (contiennent `'winrate'`). Ajouter les diffs après jointure :

```python
for w in WINDOWS:
    df_ml[f'winrate_quality_diff_{w}'] = (df_ml[f'p1_winrate_quality_{w}']
                                           - df_ml[f'p2_winrate_quality_{w}'])
```

---

### Fichier 2 : `src/prepare_ml_dataset.py`

#### `define_feature_sets()` — Nouveau groupe

```python
'qualite_adversaires': [
    'p1_winrate_quality_5',        'p2_winrate_quality_5',        'winrate_quality_diff_5',
    'p1_winrate_quality_10',       'p2_winrate_quality_10',       'winrate_quality_diff_10',
    'p1_winrate_quality_20',       'p2_winrate_quality_20',       'winrate_quality_diff_20',
],
```

#### `prepare_dataset()` — Insertion dans le groupe actif

```python
groups = ['elo', 'glicko', 'ranking', 'forme', 'qualite_adversaires',
          'surface_forme', 'h2h', 'fatigue', 'contexte']
if use_stats:
    groups.append('stats_service')
```

---

## Garanties anti-leakage

- `.shift(1)` avant le rolling : le match courant n'est jamais inclus dans sa propre feature
- `opponent_rank` = rang ATP à la date du match (publié avant le match, déjà dans les données)
- `build_player_match_history()` opère sur le format p1/p2 restructuré (pas winner/loser)
- Jointure via `(player_id, tourney_date)`, identique aux features rolling existantes

---

## Limitations connues

- `min_periods=3` : les 2 premiers matchs d'un joueur → NaN (imputés à 0.5 par l'imputer)
- Pré-1973 : pas de classement ATP → `RANK_FALLBACK=300` (poids faible, conservateur)

---

## Pipeline à relancer

```bash
python src/compute_rolling_features.py --tour atp   # modifié
python src/compute_h2h.py --tour atp                # passe les nouvelles colonnes
python src/compute_contextual_features.py --tour atp
python src/prepare_ml_dataset.py --tour atp         # modifié
python src/train_xgboost.py --tour atp
```

---

## Critères de succès

1. 9 nouvelles colonnes `winrate_quality_*` présentes dans **`matches_features_final.parquet`**
2. Couverture ≥ 80% non-NaN sur le train set (vérifier avec `audit_rolling()`)
3. `p1_winrate_quality_10` ∈ [0, 1], moyenne ~0.45-0.55
4. Feature importance XGBoost : au moins une `winrate_quality_*` dans le top 20
5. Accuracy sur le **validation set 2023-2024** (split `valid_mask = (year >= 2023) & (year <= 2024)`) ≥ 78.1%
