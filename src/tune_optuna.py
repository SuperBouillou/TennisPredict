# src/tune_optuna.py
"""
Optuna hyperparameter search for XGBoost with walk-forward cross-validation.

Walk-forward folds (default, --no-walk-forward to disable):
    Fold 1 : train ≤2021, valid 2022  (~80K train / ~3K valid)
    Fold 2 : train ≤2022, valid 2023  (~83K train / ~3K valid)
    Fold 3 : train ≤2023, valid 2024  (~95K train / ~2.8K valid)
    Objective = mean log-loss across 3 folds → params robustes multi-années

Single-fold (--no-walk-forward):
    Utilise splits.pkl standard (train ≤2023, valid 2024)

Usage:
    python src/tune_optuna.py --tour atp --n-trials 200
    python src/tune_optuna.py --tour atp --n-trials 100 --no-walk-forward

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
import pandas as pd
import joblib
import optuna
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss

from config import get_paths, make_dirs

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Walk-forward fold definitions: (train_end_year, valid_year)
WF_FOLDS = [
    (2021, 2022),
    (2022, 2023),
    (2023, 2024),
]

BASELINE_PARAMS = {
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


def _build_fold(df_full: pd.DataFrame,
                features: list,
                train_end: int,
                valid_year: int) -> tuple:
    """
    Build one walk-forward fold from the full dataset.
    Returns (X_train_imp, y_train, X_valid_imp, y_valid, n_train, n_valid).
    Imputer is fitted on train only — no data leakage.
    """
    year = df_full['year']
    train_mask = year <= train_end
    valid_mask = year == valid_year

    # Use only features present in the dataframe
    feats = [f for f in features if f in df_full.columns]

    X_tr = df_full.loc[train_mask, feats].values
    y_tr = df_full.loc[train_mask, 'target'].values
    X_va = df_full.loc[valid_mask, feats].values
    y_va = df_full.loc[valid_mask, 'target'].values

    imp = SimpleImputer(strategy='constant', fill_value=0.5)
    X_tr_imp = imp.fit_transform(X_tr)
    X_va_imp = imp.transform(X_va)

    return X_tr_imp, y_tr, X_va_imp, y_va, int(train_mask.sum()), int(valid_mask.sum())


def _train_and_score(params: dict,
                     X_tr: np.ndarray, y_tr: np.ndarray,
                     X_va: np.ndarray, y_va: np.ndarray) -> float:
    """Train XGBoost on one fold and return best validation log-loss."""
    model = xgb.XGBClassifier(
        n_estimators          = 1000,
        eval_metric           = 'logloss',
        early_stopping_rounds = 50,
        random_state          = 42,
        verbosity             = 0,
        **params,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model.best_score


def objective_wf(trial, folds: list) -> float:
    """Walk-forward objective: mean log-loss across all folds."""
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
    scores = [_train_and_score(params, *fold) for fold in folds]
    return float(np.mean(scores))


def objective_single(trial,
                     X_train_imp: np.ndarray, y_train: np.ndarray,
                     X_valid_imp: np.ndarray, y_valid: np.ndarray) -> float:
    """Single-fold objective (legacy mode)."""
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
    return _train_and_score(params, X_train_imp, y_train, X_valid_imp, y_valid)


def main():
    parser = argparse.ArgumentParser(description="Optuna tuning XGBoost")
    parser.add_argument('--tour',            default='atp', choices=['atp', 'wta'])
    parser.add_argument('--n-trials',        type=int, default=100)
    parser.add_argument('--timeout',         type=int, default=None,
                        help="Durée max en secondes (optionnel)")
    parser.add_argument('--no-walk-forward', action='store_true',
                        help="Désactiver le walk-forward — utilise splits.pkl standard")
    args = parser.parse_args()

    tour  = args.tour.lower()
    paths = get_paths(tour)
    make_dirs(tour)

    MODELS_DIR    = paths['models_dir']
    PROCESSED_DIR = paths['processed_dir']

    walk_forward = not args.no_walk_forward

    print("=" * 60)
    mode = "WALK-FORWARD (3 folds)" if walk_forward else "SINGLE FOLD"
    print(f"OPTUNA TUNING — {tour.upper()} — {args.n_trials} trials — {mode}")
    print("=" * 60)

    features = joblib.load(MODELS_DIR / "feature_list.pkl")
    print(f"\n  Features : {len(features)}")

    # ── Walk-forward : construire les 3 folds ────────────────────
    if walk_forward:
        print(f"\n  Chargement dataset complet pour walk-forward...")
        df_full = pd.read_parquet(
            PROCESSED_DIR / "matches_features_final.parquet"
        )
        # Même filtre que prepare_ml_dataset.py
        if 'has_stats' in df_full.columns:
            df_full = df_full[df_full['has_stats'] == 1].copy()
        if 'year' not in df_full.columns:
            df_full['year'] = pd.to_datetime(df_full['tourney_date']).dt.year

        print(f"  Dataset filtré : {len(df_full):,} matchs "
              f"({df_full['year'].min()}-{df_full['year'].max()})")

        print(f"\n  Construction des folds :")
        folds = []
        baseline_scores = []
        for train_end, valid_year in WF_FOLDS:
            X_tr, y_tr, X_va, y_va, n_tr, n_va = _build_fold(
                df_full, features, train_end, valid_year
            )
            folds.append((X_tr, y_tr, X_va, y_va))
            bl = _train_and_score(BASELINE_PARAMS, X_tr, y_tr, X_va, y_va)
            baseline_scores.append(bl)
            print(f"    Fold train≤{train_end}/valid={valid_year} : "
                  f"{n_tr:,} train / {n_va:,} valid | baseline ll={bl:.6f}")

        baseline_ll = float(np.mean(baseline_scores))
        print(f"\n  Baseline log-loss moyen (3 folds) : {baseline_ll:.6f}")

        print(f"\n  Lancement étude Optuna ({args.n_trials} trials × 3 folds)...\n")
        study = optuna.create_study(
            direction  = 'minimize',
            study_name = f'xgb_{tour}_wf',
            sampler    = optuna.samplers.TPESampler(seed=42),
            pruner     = optuna.pruners.MedianPruner(n_startup_trials=10,
                                                     n_warmup_steps=0),
        )
        study.optimize(
            lambda trial: objective_wf(trial, folds),
            n_trials          = args.n_trials,
            timeout           = args.timeout,
            show_progress_bar = True,
        )

    # ── Single fold (legacy) ─────────────────────────────────────
    else:
        splits  = joblib.load(MODELS_DIR / "splits.pkl")
        X_train = splits['X_train']
        X_valid = splits['X_valid']
        y_train = splits['y_train']
        y_valid = splits['y_valid']

        print(f"\n  Train : {len(X_train):,} matchs")
        print(f"  Valid : {len(X_valid):,} matchs")

        imp = SimpleImputer(strategy='constant', fill_value=0.5)
        X_tr_imp = imp.fit_transform(X_train)
        X_va_imp = imp.transform(X_valid)

        print("\n  Calcul baseline (params manuels)...")
        baseline_ll = _train_and_score(BASELINE_PARAMS, X_tr_imp,
                                       y_train.values if hasattr(y_train, 'values') else y_train,
                                       X_va_imp,
                                       y_valid.values if hasattr(y_valid, 'values') else y_valid)
        print(f"  Baseline log-loss (valid) : {baseline_ll:.6f}")

        print(f"\n  Lancement étude Optuna ({args.n_trials} trials)...\n")
        study = optuna.create_study(
            direction  = 'minimize',
            study_name = f'xgb_{tour}',
            sampler    = optuna.samplers.TPESampler(seed=42),
            pruner     = optuna.pruners.MedianPruner(n_startup_trials=10,
                                                     n_warmup_steps=0),
        )
        y_tr_arr = y_train.values if hasattr(y_train, 'values') else y_train
        y_va_arr = y_valid.values if hasattr(y_valid, 'values') else y_valid
        study.optimize(
            lambda trial: objective_single(trial, X_tr_imp, y_tr_arr,
                                           X_va_imp, y_va_arr),
            n_trials          = args.n_trials,
            timeout           = args.timeout,
            show_progress_bar = True,
        )

    # ── Résultats ────────────────────────────────────────────────
    best_params = study.best_params
    best_value  = study.best_value
    improvement = baseline_ll - best_value

    print("\n" + "=" * 60)
    print("RÉSULTATS OPTUNA")
    print("=" * 60)
    print(f"\n  Meilleur log-loss moyen : {best_value:.6f}")
    print(f"  Baseline log-loss moyen : {baseline_ll:.6f}")
    print(f"  Amélioration            : {improvement:+.6f}")
    print(f"\n  Meilleurs hyperparamètres :")
    for k, v in sorted(best_params.items()):
        print(f"    {k:<25}: {v}")

    # ── Sauvegarde JSON ──────────────────────────────────────────
    out = MODELS_DIR / "optuna_best_params.json"
    with open(out, 'w') as f:
        json.dump({
            'tour'         : tour,
            'n_trials'     : args.n_trials,
            'walk_forward' : walk_forward,
            'best_value'   : best_value,
            'baseline_ll'  : baseline_ll,
            'improvement'  : improvement,
            'params'       : best_params,
        }, f, indent=2)

    print(f"\n  Params sauvegardés → {out}")
    print(f"\nOptuna tuning termine.")


if __name__ == "__main__":
    main()
