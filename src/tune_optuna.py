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
