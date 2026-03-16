# Optuna Hyperparameter Tuning — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace manual XGBoost hyperparameters with an Optuna-driven search (100 trials) that minimizes validation log-loss, then integrate best params into `train_xgboost.py` via a `--optuna` flag.

**Architecture:** A standalone `src/tune_optuna.py` script runs the Optuna study and saves best params to `data/models/{tour}/optuna_best_params.json`. `train_xgboost.py` gains an `--optuna` flag: when set, it loads those params instead of the hardcoded tuned config. The imputer + splits are already saved in `data/models/{tour}/splits.pkl` — the tuning script reuses them directly.

**Tech Stack:** `optuna==3.6.*`, `xgboost==2.0.3`, `joblib`, `json`, Python 3.11 (venv at `venv/`)

---

## Chunk 1: Setup + tune_optuna.py

### Task 1: Install Optuna and update requirements.txt

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Install optuna into the venv**

```bash
cd E:/Claude/botbet/tennis/tennis_ml
venv/Scripts/pip install "optuna==3.6.*"
```

Expected output: `Successfully installed optuna-3.6.x ...`

- [ ] **Step 2: Verify installation**

```bash
venv/Scripts/python -c "import optuna; print(optuna.__version__)"
```

Expected: prints `3.6.x`

- [ ] **Step 3: Add optuna to requirements.txt**

Add this line to `requirements.txt` (after the existing entries):
```
optuna==3.6.0
```

---

### Task 2: Create src/tune_optuna.py

**Files:**
- Create: `src/tune_optuna.py`

This script:
1. Loads `splits.pkl` + `feature_list.pkl` from `data/models/{tour}/`
2. Fits the imputer on the training set
3. Runs an Optuna study (N trials) to minimize validation log-loss
4. Saves the best params to `data/models/{tour}/optuna_best_params.json`
5. Prints a summary table of best params and improvement vs. baseline

- [ ] **Step 1: Create src/tune_optuna.py**

```python
# src/tune_optuna.py
"""
Optuna hyperparameter search for XGBoost.

Usage:
    python src/tune_optuna.py --tour atp --n-trials 100
    python src/tune_optuna.py --tour wta --n-trials 50

Output:
    data/models/{tour}/optuna_best_params.json
"""

import sys
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import joblib
import optuna
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss

from config import get_paths, make_dirs

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial, X_train_imp, y_train, X_valid_imp, y_valid):
    """Optuna objective: minimize validation log-loss."""
    params = {
        'max_depth'        : trial.suggest_int  ('max_depth',         2,     6),
        'min_child_weight' : trial.suggest_int  ('min_child_weight',  1,    30),
        'gamma'            : trial.suggest_float('gamma',             0.0,   3.0),
        'subsample'        : trial.suggest_float('subsample',         0.5,   1.0),
        'colsample_bytree' : trial.suggest_float('colsample_bytree',  0.4,   1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4,   1.0),
        'reg_alpha'        : trial.suggest_float('reg_alpha',         0.0,   2.0),
        'reg_lambda'       : trial.suggest_float('reg_lambda',        0.5,   5.0),
        'learning_rate'    : trial.suggest_float('learning_rate',     0.005, 0.1, log=True),
    }

    model = xgb.XGBClassifier(
        n_estimators          = 1000,
        eval_metric           = 'logloss',
        early_stopping_rounds = 50,
        random_state          = 42,
        verbosity             = 0,
        **params,
    )

    model.fit(
        X_train_imp, y_train,
        eval_set=[(X_valid_imp, y_valid)],
        verbose=False,
    )

    # best_score is the best validation logloss found during early stopping
    return model.best_score


def main():
    parser = argparse.ArgumentParser(description="Optuna tuning XGBoost")
    parser.add_argument('--tour',     default='atp', choices=['atp', 'wta'])
    parser.add_argument('--n-trials', type=int, default=100,
                        help="Nombre de trials Optuna (défaut: 100)")
    parser.add_argument('--timeout',  type=int, default=None,
                        help="Durée max en secondes (optionnel)")
    args = parser.parse_args()

    tour  = args.tour.lower()
    paths = get_paths(tour)
    make_dirs(tour)

    MODELS_DIR = paths['models_dir']

    print("=" * 55)
    print(f"OPTUNA TUNING — {tour.upper()} — {args.n_trials} trials")
    print("=" * 55)

    # ── Chargement des données ───────────────────────────────────
    splits   = joblib.load(MODELS_DIR / "splits.pkl")
    features = joblib.load(MODELS_DIR / "feature_list.pkl")

    X_train = splits['X_train']
    X_valid = splits['X_valid']
    y_train = splits['y_train']
    y_valid = splits['y_valid']

    print(f"\n  Train : {len(X_train):,} matchs")
    print(f"  Valid : {len(X_valid):,} matchs")
    print(f"  Features : {len(features)}")

    # ── Imputation ───────────────────────────────────────────────
    imputer = SimpleImputer(strategy='constant', fill_value=0.5)
    X_train_imp = imputer.fit_transform(X_train)
    X_valid_imp = imputer.transform(X_valid)

    # ── Baseline : params manuels actuels ───────────────────────
    print("\n  Calcul baseline (params manuels)...")
    baseline_model = xgb.XGBClassifier(
        n_estimators          = 1000,
        learning_rate         = 0.02,
        max_depth             = 3,
        min_child_weight      = 10,
        subsample             = 0.75,
        colsample_bytree      = 0.7,
        colsample_bylevel     = 0.7,
        gamma                 = 1.0,
        reg_alpha             = 0.1,
        reg_lambda            = 2.0,
        eval_metric           = 'logloss',
        early_stopping_rounds = 50,
        random_state          = 42,
        verbosity             = 0,
    )
    baseline_model.fit(X_train_imp, y_train,
                       eval_set=[(X_valid_imp, y_valid)], verbose=False)
    baseline_ll = baseline_model.best_score
    print(f"  Baseline log-loss (valid) : {baseline_ll:.6f}")

    # ── Étude Optuna ─────────────────────────────────────────────
    print(f"\n  Lancement étude Optuna ({args.n_trials} trials)...\n")

    study = optuna.create_study(
        direction   = 'minimize',
        study_name  = f'xgb_{tour}',
        sampler     = optuna.samplers.TPESampler(seed=42),
        pruner      = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0),
    )

    study.optimize(
        lambda trial: objective(trial, X_train_imp, y_train, X_valid_imp, y_valid),
        n_trials = args.n_trials,
        timeout  = args.timeout,
        show_progress_bar = True,
    )

    # ── Résultats ────────────────────────────────────────────────
    best_params = study.best_params
    best_value  = study.best_value
    improvement = baseline_ll - best_value

    print("\n" + "=" * 55)
    print("RÉSULTATS OPTUNA")
    print("=" * 55)
    print(f"\n  Meilleur log-loss (valid) : {best_value:.6f}")
    print(f"  Baseline log-loss         : {baseline_ll:.6f}")
    print(f"  Amélioration              : {improvement:+.6f}")
    print(f"\n  Meilleurs hyperparamètres :")
    for k, v in sorted(best_params.items()):
        print(f"    {k:<25}: {v}")

    # ── Sauvegarde JSON ──────────────────────────────────────────
    out = MODELS_DIR / "optuna_best_params.json"
    with open(out, 'w') as f:
        json.dump({
            'tour'        : tour,
            'n_trials'    : args.n_trials,
            'best_value'  : best_value,
            'baseline_ll' : baseline_ll,
            'improvement' : improvement,
            'params'      : best_params,
        }, f, indent=2)

    print(f"\n  Params sauvegardés → {out}")
    print(f"\nOptuna tuning termine.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

```bash
cd E:/Claude/botbet/tennis/tennis_ml
venv/Scripts/python -c "import ast; ast.parse(open('src/tune_optuna.py').read()); print('OK')"
```

Expected: `OK`

---

## Chunk 2: Modify train_xgboost.py + Integration

### Task 3: Add --optuna flag to train_xgboost.py

**Files:**
- Modify: `src/train_xgboost.py`

When `--optuna` is passed, the script loads `optuna_best_params.json` and uses those params instead of the hardcoded tuned config. The rest of the training pipeline (calibration, evaluation, saving) stays identical.

- [ ] **Step 1: Read current train_xgboost.py to identify exact lines to modify**

Lines to change in `src/train_xgboost.py`:
- Around line 107-109 (argparse block): add `--optuna` argument
- Around line 165-180 (XGBoost tuned params block): add conditional loading from JSON

- [ ] **Step 2: Add --optuna argument to the argparse block**

In `src/train_xgboost.py`, find:
```python
    parser.add_argument('--tour', default='atp', choices=['atp', 'wta'],
                        help="Tour à traiter : atp ou wta (défaut: atp)")
    args = parser.parse_args()
```

Replace with:
```python
    parser.add_argument('--tour', default='atp', choices=['atp', 'wta'],
                        help="Tour à traiter : atp ou wta (défaut: atp)")
    parser.add_argument('--optuna', action='store_true',
                        help="Charger les hyperparamètres depuis optuna_best_params.json")
    args = parser.parse_args()
```

- [ ] **Step 3: Add JSON loading + conditional params block**

In `src/train_xgboost.py`, add this import at the top (after `import warnings`):
```python
import json
```

Then find the XGBoost tuned block (around line 163):
```python
    # ── Étape 2 : XGBoost tuné ───────────────────────────────────────────────
    print("\n── XGBoost tuné ────────────────────────────────────")

    xgb_tuned = xgb.XGBClassifier(
        n_estimators          = 1000,
        learning_rate         = 0.02,
        max_depth             = 3,
        min_child_weight      = 10,
        subsample             = 0.75,
        colsample_bytree      = 0.7,
        colsample_bylevel     = 0.7,
        gamma                 = 1.0,
        reg_alpha             = 0.1,
        reg_lambda            = 2.0,
        eval_metric           = 'logloss',
        early_stopping_rounds = 50,
        random_state          = 42,
        verbosity             = 0,
    )
```

Replace with:
```python
    # ── Étape 2 : XGBoost tuné ───────────────────────────────────────────────
    print("\n── XGBoost tuné ────────────────────────────────────")

    # Charger les params Optuna si disponibles et --optuna passé
    optuna_json = MODELS_DIR / "optuna_best_params.json"
    tuned_params = {}
    if args.optuna and optuna_json.exists():
        with open(optuna_json) as f:
            optuna_data = json.load(f)
        tuned_params = optuna_data['params']
        print(f"  [Optuna] Params chargés depuis {optuna_json}")
        print(f"  [Optuna] Best val log-loss    : {optuna_data['best_value']:.6f}")
        print(f"  [Optuna] Amelioration baseline: {optuna_data['improvement']:+.6f}")
    else:
        tuned_params = {
            'learning_rate'    : 0.02,
            'max_depth'        : 3,
            'min_child_weight' : 10,
            'subsample'        : 0.75,
            'colsample_bytree' : 0.7,
            'colsample_bylevel': 0.7,
            'gamma'            : 1.0,
            'reg_alpha'        : 0.1,
            'reg_lambda'       : 2.0,
        }
        if args.optuna:
            print(f"  [Optuna] JSON non trouvé — params manuels utilisés")

    xgb_tuned = xgb.XGBClassifier(
        n_estimators          = 1000,
        eval_metric           = 'logloss',
        early_stopping_rounds = 50,
        random_state          = 42,
        verbosity             = 0,
        **tuned_params,
    )
```

- [ ] **Step 4: Verify syntax**

```bash
cd E:/Claude/botbet/tennis/tennis_ml
venv/Scripts/python -c "import ast; ast.parse(open('src/train_xgboost.py').read()); print('OK')"
```

Expected: `OK`

---

### Task 4: Run Optuna tuning for ATP + validate

- [ ] **Step 1: Run Optuna study ATP (100 trials)**

```bash
cd E:/Claude/botbet/tennis/tennis_ml
venv/Scripts/python src/tune_optuna.py --tour atp --n-trials 100
```

Expected output (approximate):
```
=======================================================
OPTUNA TUNING — ATP — 100 trials
=======================================================

  Train : 95,192 matchs
  Valid : 2,796 matchs
  Features : 55

  Calcul baseline (params manuels)...
  Baseline log-loss (valid) : 0.47xxxx

  Lancement étude Optuna (100 trials)...
[progress bar...]

=======================================================
RÉSULTATS OPTUNA
=======================================================

  Meilleur log-loss (valid) : 0.46xxxx
  Baseline log-loss         : 0.47xxxx
  Amélioration              : +0.00xxxx
  ...
  Params sauvegardés → data/models/atp/optuna_best_params.json
```

- [ ] **Step 2: Verify JSON was created**

```bash
cat E:/Claude/botbet/tennis/tennis_ml/data/models/atp/optuna_best_params.json
```

Expected: valid JSON with keys `tour`, `best_value`, `baseline_ll`, `improvement`, `params`

- [ ] **Step 3: Retrain XGBoost ATP avec params Optuna**

```bash
cd E:/Claude/botbet/tennis/tennis_ml
venv/Scripts/python src/train_xgboost.py --tour atp --optuna
```

Check that the output shows:
- `[Optuna] Params chargés depuis ...`
- Valid/test accuracy and log-loss printed at the end
- Model saved to `data/models/atp/xgb_tuned.pkl`

- [ ] **Step 4: Vérifier que le modèle sauvegardé est bien rechargé**

```bash
venv/Scripts/python -c "
import joblib, numpy as np
m = joblib.load('data/models/atp/xgb_tuned.pkl')
imp = joblib.load('data/models/atp/imputer.pkl')
splits = joblib.load('data/models/atp/splits.pkl')
X = imp.transform(splits['X_valid'])
y = splits['y_valid']
proba = m.predict_proba(X)[:,1]
from sklearn.metrics import log_loss, accuracy_score
print(f'Accuracy: {accuracy_score(y, m.predict(X)):.4f}')
print(f'Log-loss: {log_loss(y, proba):.6f}')
"
```

Expected: Accuracy ≥ 0.780, Log-loss ≤ 0.475

---

### Task 5: Run Optuna tuning for WTA + validate

- [ ] **Step 1: Run Optuna study WTA (100 trials)**

```bash
cd E:/Claude/botbet/tennis/tennis_ml
venv/Scripts/python src/tune_optuna.py --tour wta --n-trials 100
```

- [ ] **Step 2: Retrain XGBoost WTA avec params Optuna**

```bash
venv/Scripts/python src/train_xgboost.py --tour wta --optuna
```

- [ ] **Step 3: Vérifier résultats WTA**

```bash
venv/Scripts/python -c "
import joblib
from sklearn.metrics import log_loss, accuracy_score
m = joblib.load('data/models/wta/xgb_tuned.pkl')
imp = joblib.load('data/models/wta/imputer.pkl')
splits = joblib.load('data/models/wta/splits.pkl')
X = imp.transform(splits['X_valid'])
y = splits['y_valid']
proba = m.predict_proba(X)[:,1]
print(f'WTA Accuracy: {accuracy_score(y, m.predict(X)):.4f}')
print(f'WTA Log-loss: {log_loss(y, proba):.6f}')
"
```

Expected: Accuracy ≥ 0.770, Log-loss ≤ 0.480

---

## Résumé des fichiers

| Fichier | Action | Rôle |
|---|---|---|
| `requirements.txt` | Modifier | Ajouter `optuna==3.6.0` |
| `src/tune_optuna.py` | Créer | Étude Optuna → JSON best params |
| `src/train_xgboost.py` | Modifier | Flag `--optuna`, chargement JSON |
| `data/models/atp/optuna_best_params.json` | Généré | Best params ATP |
| `data/models/wta/optuna_best_params.json` | Généré | Best params WTA |

## Commandes pipeline post-tuning

Pour ré-entraîner avec les meilleurs params à l'avenir :
```bash
# 1. Tuner (une seule fois, ou après nouvelle saison)
python src/tune_optuna.py --tour atp --n-trials 100

# 2. Entraîner avec les params Optuna
python src/train_xgboost.py --tour atp --optuna

# 3. Backtester
python src/backtest_real.py --tour atp
```
