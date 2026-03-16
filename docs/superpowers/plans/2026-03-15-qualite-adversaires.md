# Qualité Adversaires Récents — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ajouter 9 features de win rate pondéré par qualité adversaire au pipeline ML de prédiction tennis.

**Architecture:** On modifie `compute_rolling_features.py` pour (1) inclure le rang de l'adversaire dans l'historique par joueur et (2) calculer un win rate normalisé `sum(won×w)/sum(w)` avec `w=1/log2(rank+2)`. On enregistre le groupe de features dans `prepare_ml_dataset.py` pour qu'il soit inclus dans le dataset ML.

**Tech Stack:** Python, pandas, numpy, pytest, XGBoost (pipeline existant)

**Spec:** `docs/superpowers/specs/2026-03-15-qualite-adversaires-design.md`

---

## Chunk 1: Tests & `build_player_match_history()`

### Task 1: Créer les tests unitaires

**Files:**
- Create: `tests/test_qualite_adversaires.py`

- [ ] **Step 1 : Créer le fichier de tests**

```python
# tests/test_qualite_adversaires.py
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

ROOT = Path(__file__).parent.parent


def make_ml_df(n_extra=0):
    """
    Dataset ML minimal (format p1/p2 avec index 0-based) pour les tests.
    n_extra : matchs supplémentaires à ajouter pour le joueur 1 (pour tester min_periods).
    """
    base = {
        'tourney_date': pd.to_datetime([
            '2020-01-06', '2020-01-13', '2020-01-20',
            '2020-01-27', '2020-02-03',
        ]),
        'tourney_level': ['G', 'G', 'M', 'M', 'A'],
        'surface':       ['Hard', 'Hard', 'Clay', 'Clay', 'Grass'],
        'p1_id':   [1, 1, 1, 2, 2],
        'p2_id':   [2, 3, 2, 1, 3],
        'target':  [1, 0, 1, 0, 1],
        'p1_rank': [5.0,  5.0,  5.0,  20.0, 20.0],
        'p2_rank': [20.0, 50.0, 20.0, 5.0,  80.0],
        'p1_rank_points': [3000, 3000, 3000, 500, 500],
        'p2_rank_points': [500,  400,  500,  3000, 200],
        'p1_age': [25.0, 25.0, 25.0, 28.0, 28.0],
        'p2_age': [28.0, 30.0, 28.0, 25.0, 22.0],
    }
    df = pd.DataFrame(base)

    if n_extra > 0:
        extras = pd.DataFrame({
            'tourney_date': pd.date_range('2020-02-10', periods=n_extra, freq='7D'),
            'tourney_level': ['A'] * n_extra,
            'surface': ['Hard'] * n_extra,
            'p1_id':   [1] * n_extra,
            'p2_id':   [4] * n_extra,
            'target':  [1] * n_extra,
            'p1_rank': [5.0] * n_extra,
            'p2_rank': [100.0] * n_extra,
            'p1_rank_points': [3000] * n_extra,
            'p2_rank_points': [300] * n_extra,
            'p1_age': [25.0] * n_extra,
            'p2_age': [26.0] * n_extra,
        })
        df = pd.concat([df, extras], ignore_index=True)

    # Index 0-based propre — précondition requise par build_player_match_history()
    return df.reset_index(drop=True)


# ── Tests build_player_match_history ──────────────────────────────────────────

def test_opponent_rank_column_exists():
    """opponent_rank doit être présente dans df_history."""
    from compute_rolling_features import build_player_match_history
    hist = build_player_match_history(make_ml_df())
    assert 'opponent_rank' in hist.columns


def test_opponent_rank_p1_view():
    """Vue p1 : opponent_rank doit être p2_rank du match."""
    from compute_rolling_features import build_player_match_history
    df = make_ml_df()
    hist = build_player_match_history(df)
    # Match du 2020-01-06 : p1=1 vs p2=2 (p2_rank=20)
    row = hist[(hist['player_id'] == 1) &
               (hist['tourney_date'] == pd.Timestamp('2020-01-06'))].iloc[0]
    assert row['opponent_rank'] == 20.0


def test_opponent_rank_p2_view():
    """Vue p2 : opponent_rank doit être p1_rank du match."""
    from compute_rolling_features import build_player_match_history
    df = make_ml_df()
    hist = build_player_match_history(df)
    # Match du 2020-01-06 : p2=2 vs p1=1 (p1_rank=5)
    row = hist[(hist['player_id'] == 2) &
               (hist['tourney_date'] == pd.Timestamp('2020-01-06'))].iloc[0]
    assert row['opponent_rank'] == 5.0


# ── Tests compute_rolling_stats ────────────────────────────────────────────────

def test_winrate_quality_columns_present():
    """winrate_quality_5/10/20 doivent apparaître dans les stats glissantes."""
    from compute_rolling_features import build_player_match_history, compute_rolling_stats
    hist = build_player_match_history(make_ml_df())
    hist = compute_rolling_stats(hist)
    for w in [5, 10, 20]:
        assert f'winrate_quality_{w}' in hist.columns, f"Colonne manquante : winrate_quality_{w}"


def test_winrate_quality_bounded():
    """winrate_quality doit être dans [0, 1] (win rate pondéré normalisé)."""
    from compute_rolling_features import build_player_match_history, compute_rolling_stats
    # Besoin d'assez de matchs pour passer min_periods=3
    hist = build_player_match_history(make_ml_df(n_extra=5))
    hist = compute_rolling_stats(hist)
    col = hist['winrate_quality_10'].dropna()
    assert len(col) > 0, "Aucune valeur non-NaN produite"
    assert (col >= 0.0).all(), f"Valeurs < 0 trouvées : {col[col < 0]}"
    assert (col <= 1.0).all(), f"Valeurs > 1 trouvées : {col[col > 1]}"


def test_winrate_quality_no_leakage():
    """
    Test anti-leakage : vérifie que .shift(1) est bien appliqué.
    Stratégie : un joueur avec exactement 4 matchs.
    - Ligne 0 (shift → regarde avant match 0) : NaN car aucun historique avant match 0
    - Ligne 3 (shift → regarde les matchs 0, 1, 2) : doit valoir la qualité des 3 premiers matchs
    Si .shift(1) est absent, la ligne 0 contiendrait une valeur (le match lui-même).
    """
    from compute_rolling_features import build_player_match_history, compute_rolling_stats
    # 4 matchs pour le joueur 1 (3 de base + 1 extra) → min_periods=3 satisfait au 4ème
    df = make_ml_df(n_extra=1)
    hist = build_player_match_history(df)
    hist = compute_rolling_stats(hist)

    p1 = hist[hist['player_id'] == 1].sort_values('tourney_date').reset_index(drop=True)
    # 1er match du joueur : pas d'historique avant lui → NaN obligatoire (shift correct)
    assert pd.isna(p1.iloc[0]['winrate_quality_5']), \
        "Fuite détectée : le 1er match ne devrait avoir aucune valeur quality"
    # 4ème match : 3 matchs précédents disponibles → valeur attendue
    assert not pd.isna(p1.iloc[3]['winrate_quality_5']), \
        "La 4ème ligne devrait avoir une valeur quality (3 matchs précédents)"


# ── Tests join_rolling_to_ml ──────────────────────────────────────────────────

def test_diff_columns_present():
    """winrate_quality_diff_5/10/20 doivent être dans le dataset ML après jointure."""
    from compute_rolling_features import (build_player_match_history,
                                          compute_rolling_stats, join_rolling_to_ml)
    df = make_ml_df(n_extra=5)
    hist = build_player_match_history(df)
    hist = compute_rolling_stats(hist)
    df_out = join_rolling_to_ml(df, hist)
    for w in [5, 10, 20]:
        assert f'winrate_quality_diff_{w}' in df_out.columns


def test_diff_equals_p1_minus_p2():
    """winrate_quality_diff_10 = p1_winrate_quality_10 - p2_winrate_quality_10."""
    from compute_rolling_features import (build_player_match_history,
                                          compute_rolling_stats, join_rolling_to_ml)
    df = make_ml_df(n_extra=5)
    hist = build_player_match_history(df)
    hist = compute_rolling_stats(hist)
    df_out = join_rolling_to_ml(df, hist)
    valid = df_out.dropna(subset=['p1_winrate_quality_10', 'p2_winrate_quality_10',
                                   'winrate_quality_diff_10'])
    if len(valid) > 0:
        expected = valid['p1_winrate_quality_10'] - valid['p2_winrate_quality_10']
        pd.testing.assert_series_equal(
            valid['winrate_quality_diff_10'].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False
        )


# ── Tests prepare_ml_dataset ──────────────────────────────────────────────────

def test_qualite_adversaires_group_defined():
    """Le groupe 'qualite_adversaires' doit exister dans define_feature_sets()."""
    from prepare_ml_dataset import define_feature_sets
    fs = define_feature_sets()
    assert 'qualite_adversaires' in fs, "Groupe 'qualite_adversaires' absent"


def test_qualite_adversaires_features_count():
    """Le groupe doit contenir exactement 9 features."""
    from prepare_ml_dataset import define_feature_sets
    fs = define_feature_sets()
    n = len(fs['qualite_adversaires'])
    assert n == 9, f"Attendu 9 features, trouvé {n}: {fs['qualite_adversaires']}"
```

- [ ] **Step 2 : Vérifier que les tests échouent (état initial)**

```bash
cd E:\Claude\botbet\tennis\tennis_ml && source venv/Scripts/activate
pytest tests/test_qualite_adversaires.py -v 2>&1 | head -40
```

Résultat attendu : tous les tests `FAILED` ou `ERROR`.

---

### Task 2 : Modifier `build_player_match_history()`

**Files:**
- Modify: `src/compute_rolling_features.py` (fonction `build_player_match_history`, lignes ~25-54)

**Prérequis :** `build_player_match_history()` reçoit `df` depuis `matches_with_elo.parquet`, qui est lu avec `pd.read_parquet()` → index 0-based propre par défaut. L'assignment `.values` est donc safe. La fonction `make_ml_df()` dans les tests appelle `.reset_index(drop=True)` pour garantir la même précondition.

- [ ] **Step 3 : Remplacer la fonction `build_player_match_history()` entière**

Remplacer le corps de la fonction (lignes 26-54) par :

```python
def build_player_match_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme le dataset ML (p1/p2) en vue par joueur.
    Chaque match apparaît 2 fois : une fois pour p1, une fois pour p2.
    Permet de calculer l'historique individuel de chaque joueur.

    Précondition : df doit avoir un index 0-based contigu (pd.read_parquet garantit cela).
    """
    stat_cols = ROLLING_STATS + ['rank', 'rank_points', 'age']

    # Vue joueur 1
    p1_cols = {f'p1_{s}': s for s in stat_cols if f'p1_{s}' in df.columns}
    df_p1 = df[['tourney_date', 'tourney_level', 'surface',
                 'p1_id', 'target'] + list(p1_cols.keys())].copy()
    df_p1 = df_p1.rename(columns={'p1_id': 'player_id', 'target': 'won'})
    df_p1 = df_p1.rename(columns=p1_cols)
    # NOUVEAU : rang de l'adversaire (p2 est l'adversaire de p1)
    # .values requis car df_p1 est une copie dont l'index peut différer de df
    df_p1['opponent_rank'] = df['p2_rank'].values

    # Vue joueur 2
    p2_cols = {f'p2_{s}': s for s in stat_cols if f'p2_{s}' in df.columns}
    df_p2 = df[['tourney_date', 'tourney_level', 'surface',
                 'p2_id', 'target'] + list(p2_cols.keys())].copy()
    df_p2 = df_p2.rename(columns={'p2_id': 'player_id'})
    df_p2['won'] = 1 - df_p2['target']
    df_p2 = df_p2.drop(columns=['target'])
    df_p2 = df_p2.rename(columns=p2_cols)
    # NOUVEAU : rang de l'adversaire (p1 est l'adversaire de p2)
    df_p2['opponent_rank'] = df['p1_rank'].values

    # Concaténation et tri chronologique
    df_history = pd.concat([df_p1, df_p2], ignore_index=True)
    df_history = df_history.sort_values(['player_id', 'tourney_date']).reset_index(drop=True)

    print(f"✅ Historique joueur construit : {len(df_history):,} entrées")
    return df_history
```

- [ ] **Step 4 : Vérifier les 3 tests `build_player_match_history`**

```bash
cd E:\Claude\botbet\tennis\tennis_ml && source venv/Scripts/activate
pytest tests/test_qualite_adversaires.py::test_opponent_rank_column_exists \
       tests/test_qualite_adversaires.py::test_opponent_rank_p1_view \
       tests/test_qualite_adversaires.py::test_opponent_rank_p2_view -v
```

Résultat attendu : 3 `PASSED`.

---

## Chunk 2: `compute_rolling_stats()` & `join_rolling_to_ml()`

### Task 3 : Ajouter le calcul de `winrate_quality_*`

**Files:**
- Modify: `src/compute_rolling_features.py` (constantes + fonction `compute_rolling_stats`)

- [ ] **Step 5 : Ajouter les constantes au niveau module (après `WINDOWS = [5, 10, 20]`)**

```python
# Paramètres pour le win rate pondéré par qualité adversaire
RANK_FALLBACK = 300       # Rang conservateur pour adversaires sans classement ATP (seuil ~250)
MIN_QUALITY_PERIODS = 3   # Minimum de matchs précédents pour calculer le quality win rate
```

- [ ] **Step 6 : Ajouter le bloc qualité dans `compute_rolling_stats()`**

Le nouveau bloc doit être placé **à l'intérieur de la boucle `for pid in tqdm(players, ...)`**,
au même niveau d'indentation que le bloc "Win rate glissant global" (8 espaces),
**après** la boucle `for w in WINDOWS: row[f'winrate_{w}'] = ...` et **avant** le bloc "Stats de service glissantes".

```python
        # Win rate pondéré par qualité adversaire
        # Formule : sum(won * w) / sum(w), avec w = 1/log2(opponent_rank + 2)
        # Résultat dans [0, 1] : vrai win rate pondéré par la qualité des adversaires
        opp_weight = 1.0 / np.log2(
            df_pid['opponent_rank'].fillna(RANK_FALLBACK) + 2
        )
        win_quality = df_pid['won'] * opp_weight

        for w in WINDOWS:
            num = (win_quality
                   .shift(1)
                   .rolling(w, min_periods=MIN_QUALITY_PERIODS)
                   .sum())
            den = (opp_weight
                   .shift(1)
                   .rolling(w, min_periods=MIN_QUALITY_PERIODS)
                   .sum())
            row[f'winrate_quality_{w}'] = (num / den).values
```

Note : `opponent_rank` est une colonne intermédiaire de `df_history` — elle sert uniquement au
calcul dans cette fonction et ne doit PAS apparaître dans `matches_with_features.parquet` (elle
n'est pas dans `rolling_cols` qui filtre sur `['winrate', 'roll', 'streak']`).

- [ ] **Step 7 : Vérifier les tests `compute_rolling_stats`**

```bash
cd E:\Claude\botbet\tennis\tennis_ml && source venv/Scripts/activate
pytest tests/test_qualite_adversaires.py::test_winrate_quality_columns_present \
       tests/test_qualite_adversaires.py::test_winrate_quality_bounded \
       tests/test_qualite_adversaires.py::test_winrate_quality_no_leakage -v
```

Résultat attendu : 3 `PASSED`.

---

### Task 4 : Ajouter les colonnes `diff` dans `join_rolling_to_ml()`

**Files:**
- Modify: `src/compute_rolling_features.py` (fonction `join_rolling_to_ml`, lignes ~134-182)

- [ ] **Step 8 : Ajouter les diffs après la boucle `streak_diff`**

Dans `join_rolling_to_ml()`, après la ligne `df_ml['streak_diff'] = df_ml['p1_streak'] - df_ml['p2_streak']`, ajouter :

```python
    # Différences win rate qualité adversaire p1 - p2
    for w in WINDOWS:
        col_p1 = f'p1_winrate_quality_{w}'
        col_p2 = f'p2_winrate_quality_{w}'
        if col_p1 in df_ml.columns and col_p2 in df_ml.columns:
            df_ml[f'winrate_quality_diff_{w}'] = df_ml[col_p1] - df_ml[col_p2]
```

Note : `winrate_quality_*` est déjà inclus dans `rolling_cols` via le filtre `'winrate' in c`
(ligne existante dans `join_rolling_to_ml`). Les colonnes `p1_winrate_quality_*` et
`p2_winrate_quality_*` sont donc jointes automatiquement. Il reste seulement à calculer les diffs.

- [ ] **Step 9 : Vérifier les tests `join_rolling_to_ml`**

```bash
cd E:\Claude\botbet\tennis\tennis_ml && source venv/Scripts/activate
pytest tests/test_qualite_adversaires.py::test_diff_columns_present \
       tests/test_qualite_adversaires.py::test_diff_equals_p1_minus_p2 -v
```

Résultat attendu : 2 `PASSED`.

- [ ] **Step 10 : Commit Chunk 2**

```bash
cd E:\Claude\botbet\tennis\tennis_ml
git add src/compute_rolling_features.py tests/test_qualite_adversaires.py
git commit -m "feat: add quality-weighted win rate features (opponent rank)"
```

---

## Chunk 3: `prepare_ml_dataset.py` & validation pipeline

### Task 5 : Enregistrer le groupe de features

**Files:**
- Modify: `src/prepare_ml_dataset.py`

- [ ] **Step 11 : Ajouter le groupe `qualite_adversaires` dans `define_feature_sets()`**

Dans le dictionnaire `features`, après le groupe `'forme'` (ligne ~39), ajouter :

```python
        'qualite_adversaires': [
            'p1_winrate_quality_5',  'p2_winrate_quality_5',  'winrate_quality_diff_5',
            'p1_winrate_quality_10', 'p2_winrate_quality_10', 'winrate_quality_diff_10',
            'p1_winrate_quality_20', 'p2_winrate_quality_20', 'winrate_quality_diff_20',
        ],
```

- [ ] **Step 12 : Insérer le groupe dans `prepare_dataset()`**

Modifier la liste `groups` dans `prepare_dataset()` (ligne ~116) :

```python
    groups = ['elo', 'glicko', 'ranking', 'forme', 'qualite_adversaires',
              'surface_forme', 'h2h', 'fatigue', 'contexte']
```

- [ ] **Step 13 : Vérifier les tests `prepare_ml_dataset`**

```bash
cd E:\Claude\botbet\tennis\tennis_ml && source venv/Scripts/activate
pytest tests/test_qualite_adversaires.py::test_qualite_adversaires_group_defined \
       tests/test_qualite_adversaires.py::test_qualite_adversaires_features_count -v
```

Résultat attendu : 2 `PASSED`.

- [ ] **Step 14 : Vérifier que TOUS les tests passent**

```bash
cd E:\Claude\botbet\tennis\tennis_ml && source venv/Scripts/activate
pytest tests/test_qualite_adversaires.py -v
```

Résultat attendu : **10/10 PASSED**, 0 failed.

---

### Task 6 : Validation pipeline end-to-end (ATP)

- [ ] **Step 15 : Relancer `compute_rolling_features.py`**

```bash
cd E:\Claude\botbet\tennis\tennis_ml && source venv/Scripts/activate
python src/compute_rolling_features.py --tour atp
```

Vérifier dans la sortie :
- `✅ Historique joueur construit : X entrées`
- `✅ Features glissantes calculées : N colonnes` (N ~9 de plus qu'avant)
- Aucun `KeyError` sur `opponent_rank`

- [ ] **Step 16 : Vérifier les nouvelles colonnes dans `matches_with_features.parquet`**

```bash
cd E:\Claude\botbet\tennis\tennis_ml && source venv/Scripts/activate
python -c "
import pandas as pd
from pathlib import Path
df = pd.read_parquet(Path('data/processed/atp/matches_with_features.parquet'))
quality_cols = [c for c in df.columns if 'winrate_quality' in c]
print(f'Colonnes quality: {quality_cols}')
print(f'Nombre: {len(quality_cols)} (attendu: 6 — p1/p2 × 3 fenêtres)')
# Note: les diff sont ajoutées lors de la jointure, pas dans df_history
print(df[quality_cols].describe().round(3))
"
```

Attendu : 6 colonnes `p1/p2_winrate_quality_5/10/20`, valeurs dans [0, 1].

- [ ] **Step 17 : Relancer `compute_h2h.py` et `compute_contextual_features.py`**

```bash
cd E:\Claude\botbet\tennis\tennis_ml && source venv/Scripts/activate
python src/compute_h2h.py --tour atp
python src/compute_contextual_features.py --tour atp
```

- [ ] **Step 18 : Vérifier propagation dans `matches_features_final.parquet`**

```bash
cd E:\Claude\botbet\tennis\tennis_ml && source venv/Scripts/activate
python -c "
import pandas as pd
from pathlib import Path
df = pd.read_parquet(Path('data/processed/atp/matches_features_final.parquet'))
quality_cols = [c for c in df.columns if 'winrate_quality' in c]
print(f'Colonnes propagées: {len(quality_cols)} (attendu: 9)')
coverage = df[quality_cols].notna().mean()
print(coverage.round(3))
assert len(quality_cols) == 9, f'ERREUR: {len(quality_cols)} colonnes, attendu 9'
assert (coverage >= 0.8).all(), f'ERREUR: couverture < 80% pour {coverage[coverage < 0.8]}'
print('Critères 1 et 2 validés')
"
```

- [ ] **Step 19 : Relancer `prepare_ml_dataset.py`**

```bash
cd E:\Claude\botbet\tennis\tennis_ml && source venv/Scripts/activate
python src/prepare_ml_dataset.py --tour atp
```

Vérifier : `Features sélectionnées : N` (N ≈ 71, soit +9 par rapport à la baseline 62).

- [ ] **Step 20 : Relancer `train_xgboost.py` et vérifier l'accuracy**

```bash
cd E:\Claude\botbet\tennis\tennis_ml && source venv/Scripts/activate
python src/train_xgboost.py --tour atp 2>&1 | grep -E -i "accuracy|valid|test|top"
```

Critère de succès : accuracy sur le validation set 2023-2024 ≥ **78.1%**.
Vérifier aussi que des features `winrate_quality_*` apparaissent dans le top 20 (feature importance).

- [ ] **Step 21 : Commit final**

```bash
cd E:\Claude\botbet\tennis\tennis_ml
git add src/prepare_ml_dataset.py
git commit -m "feat: register qualite_adversaires feature group in ML dataset"
```

---

## Résumé des fichiers modifiés

| Fichier | Modifications |
|---|---|
| `src/compute_rolling_features.py` | `build_player_match_history()` +4 lignes, 2 constantes module-level, `compute_rolling_stats()` +12 lignes, `join_rolling_to_ml()` +5 lignes |
| `src/prepare_ml_dataset.py` | `define_feature_sets()` +11 lignes, `prepare_dataset()` +1 groupe dans la liste |
| `tests/test_qualite_adversaires.py` | Nouveau fichier — 10 tests |

**Aucun autre fichier ne doit être touché.** `compute_h2h.py`, `compute_contextual_features.py`
et `train_xgboost.py` sont relancés mais non modifiés.
