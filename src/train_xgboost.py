# src/train_xgboost.py

import sys
import argparse
import pandas as pd
import numpy as np
import joblib
import warnings
import json
warnings.filterwarnings('ignore')
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import xgboost as xgb
from sklearn.impute          import SimpleImputer
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import accuracy_score, log_loss, brier_score_loss
from sklearn.calibration     import calibration_curve, CalibratedClassifierCV
import matplotlib.pyplot as plt

from config import get_tour_config, get_paths, make_dirs


def evaluate_model(name, model, X, y, split_name) -> dict:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    acc    = accuracy_score(y, y_pred)
    ll     = log_loss(y, y_prob)
    brier  = brier_score_loss(y, y_prob)

    prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=10)
    cal_err = np.mean(np.abs(prob_true - prob_pred))

    print(f"\n  [{split_name}] {name}")
    print(f"    Accuracy          : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"    Log-loss          : {ll:.4f}")
    print(f"    Brier score       : {brier:.4f}")
    print(f"    Calibration error : {cal_err:.4f}")

    return {'model': name, 'split': split_name,
            'accuracy': acc, 'log_loss': ll,
            'brier': brier, 'calibration_error': cal_err}


def plot_feature_importance(model, features, top_n=20, models_dir: Path = None) -> None:
    """Feature importance XGBoost — gain moyen par feature."""
    booster    = model.named_steps['model']
    importance = booster.get_booster().get_score(importance_type='gain')

    # Mapper les noms f0, f1... → noms réels
    feat_map = {f'f{i}': name for i, name in enumerate(features)}
    importance_named = {feat_map.get(k, k): v for k, v in importance.items()}

    df_imp = (pd.DataFrame.from_dict(importance_named, orient='index', columns=['gain'])
              .sort_values('gain', ascending=False)
              .head(top_n))

    plt.figure(figsize=(10, 7))
    plt.barh(df_imp.index[::-1], df_imp['gain'][::-1], color='steelblue')
    plt.xlabel('Gain moyen')
    plt.title(f'Top {top_n} Features — XGBoost')
    plt.tight_layout()
    out = models_dir / "xgb_feature_importance.png" if models_dir else Path("xgb_feature_importance.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  Top 10 features :")
    print(df_imp.head(10).to_string())


def plot_calibration_comparison(models_preds, y_valid,
                                models_dir: Path = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', label='Parfaitement calibré')
    for name, y_prob in models_preds:
        prob_true, prob_pred = calibration_curve(y_valid, y_prob, n_bins=10)
        ax.plot(prob_pred, prob_true, marker='o', label=name)
    ax.set_xlabel('Probabilité prédite')
    ax.set_ylabel('Fréquence réelle')
    ax.set_title('Calibration — Comparaison modèles')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for name, y_prob in models_preds:
        ax.hist(y_prob, bins=30, alpha=0.5, label=name)
    ax.set_xlabel('Probabilité prédite p1 gagne')
    ax.set_ylabel('Nombre de matchs')
    ax.set_title('Distribution des probabilités')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = models_dir / "calibration_xgb.png" if models_dir else Path("calibration_xgb.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Entraînement XGBoost par tour")
    parser.add_argument('--tour', default='atp', choices=['atp', 'wta'],
                        help="Tour à traiter : atp ou wta (défaut: atp)")
    parser.add_argument('--optuna', action='store_true',
                        help="Charger les hyperparamètres depuis optuna_best_params.json")
    args = parser.parse_args()

    tour  = args.tour.lower()
    cfg   = get_tour_config(tour)
    paths = get_paths(tour)
    make_dirs(tour)

    MODELS_DIR = paths['models_dir']

    print("=" * 55)
    print(f"XGBOOST — ENTRAINEMENT & TUNING — {tour.upper()}")
    print("=" * 55)

    splits   = joblib.load(MODELS_DIR / "splits.pkl")
    features = joblib.load(MODELS_DIR / "feature_list.pkl")
    lr_full  = joblib.load(MODELS_DIR / "lr_full.pkl")

    X_train = splits['X_train']
    X_valid = splits['X_valid']
    X_test  = splits['X_test']
    y_train = splits['y_train']
    y_valid = splits['y_valid']
    y_test  = splits['y_test']

    imputer = SimpleImputer(strategy='constant', fill_value=0.5)
    X_train_imp = imputer.fit_transform(X_train)
    X_valid_imp = imputer.transform(X_valid)
    X_test_imp  = imputer.transform(X_test) if len(X_test) > 0 else X_test

    # ── Étape 1 : XGBoost par défaut ─────────────────────────────────────────
    print("\n── XGBoost défaut ──────────────────────────────────")

    xgb_default = xgb.XGBClassifier(
        n_estimators      = 500,
        learning_rate     = 0.05,
        max_depth         = 4,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        eval_metric       = 'logloss',
        early_stopping_rounds = 30,
        random_state      = 42,
        verbosity         = 0,
    )

    xgb_default.fit(
        X_train_imp, y_train,
        eval_set=[(X_valid_imp, y_valid)],
        verbose=False
    )

    best_iter = xgb_default.best_iteration
    print(f"  Entraine | Best iteration : {best_iter}")

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

    xgb_tuned.fit(
        X_train_imp, y_train,
        eval_set=[(X_valid_imp, y_valid)],
        verbose=False
    )

    best_iter_tuned = xgb_tuned.best_iteration
    print(f"  Entraine | Best iteration : {best_iter_tuned}")

    # ── Étape 3 : Calibration Platt Scaling ──────────────────────────────────
    print("\n── Calibration (Platt Scaling) ─────────────────────")

    from sklearn.linear_model import LogisticRegression as LR

    # Récupérer les probabilités brutes du modèle tuné
    y_prob_train_raw = xgb_tuned.predict_proba(X_train_imp)[:, 1].astype(np.float64)
    y_prob_valid_raw = xgb_tuned.predict_proba(X_valid_imp)[:, 1].astype(np.float64)

    # Platt scaling manuel : LR sur les log-odds
    platt_scaler = LR()
    platt_scaler.fit(y_prob_valid_raw.reshape(-1, 1), y_valid.astype(np.float64))

    class XGBCalibrated:
        """Wrapper XGBoost + Platt scaling manuel."""
        def __init__(self, xgb_model, platt, imputer):
            self.xgb_model = xgb_model
            self.platt     = platt
            self.imputer   = imputer

        def predict_proba(self, X):
            X_imp  = self.imputer.transform(X)
            probs  = self.xgb_model.predict_proba(X_imp)[:, 1].astype(np.float64)
            probs_cal = self.platt.predict_proba(probs.reshape(-1, 1))
            return probs_cal

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    xgb_calibrated = XGBCalibrated(xgb_tuned, platt_scaler, imputer)
    print("  Calibre (Platt scaling manuel)")

    # ── Évaluation ───────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("EVALUATION")
    print("=" * 55)

    # Wrapper predict_proba pour les modèles non-pipeline
    class ArrayModel:
        def __init__(self, model, imputer):
            self.model   = model
            self.imputer = imputer
        def predict(self, X):
            return self.model.predict(self.imputer.transform(X))
        def predict_proba(self, X):
            return self.model.predict_proba(self.imputer.transform(X))

    xgb_def_wrapped = ArrayModel(xgb_default, imputer)
    xgb_tun_wrapped = ArrayModel(xgb_tuned,   imputer)
    # xgb_calibrated a déjà son propre wrapper — on l'utilise directement

    results = []
    eval_sets = [('VALID', X_valid, y_valid)]
    if len(X_test) > 0:
        eval_sets.append(('TEST', X_test, y_test))
    for name, model in [
        ('XGB_default',   xgb_def_wrapped),
        ('XGB_tuned',     xgb_tun_wrapped),
        ('XGB_calibrated',xgb_calibrated),
    ]:
        for split_name, Xs, ys in eval_sets:
            results.append(evaluate_model(name, model, Xs, ys, split_name))

    # ── Tableau comparatif final ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("TABLEAU COMPARATIF COMPLET")
    print("=" * 55)

    # Ajouter LR_full pour comparaison
    lr_results = []
    for split_name, Xs, ys in eval_sets:
        y_prob = lr_full.predict_proba(Xs)[:, 1]
        y_pred = lr_full.predict(Xs)
        prob_true, prob_pred = calibration_curve(ys, y_prob, n_bins=10)
        lr_results.append({
            'model': 'LR_full', 'split': split_name,
            'accuracy': accuracy_score(ys, y_pred),
            'log_loss': log_loss(ys, y_prob),
            'brier': brier_score_loss(ys, y_prob),
            'calibration_error': np.mean(np.abs(prob_true - prob_pred))
        })

    df_all = pd.DataFrame(lr_results + results)
    for split_name in df_all['split'].unique():
        sub = df_all[df_all['split'] == split_name].sort_values('log_loss')
        print(f"\n  {split_name} :")
        print(sub[['model','accuracy','log_loss','brier','calibration_error']]
              .to_string(index=False))

    # ── Feature importance ───────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("FEATURE IMPORTANCE — XGBoost tuné")
    print("=" * 55)
    plot_feature_importance(
        type('obj', (object,), {
            'named_steps': {'model': xgb_tuned},
            'get_booster': xgb_tuned.get_booster
        })(),
        features,
        models_dir=MODELS_DIR,
    )

    # ── Calibration plot ────────────────────────────────────────────────────
    models_preds = [
        ('LR Full',        lr_full.predict_proba(X_valid)[:, 1]),
        ('XGB default',    xgb_def_wrapped.predict_proba(X_valid)[:, 1]),
        ('XGB tuned',      xgb_tun_wrapped.predict_proba(X_valid)[:, 1]),
        ('XGB calibrated', xgb_calibrated.predict_proba(X_valid)[:, 1]),
    ]
    plot_calibration_comparison(models_preds, y_valid, models_dir=MODELS_DIR)

    # ── Sauvegarde ──────────────────────────────────────────────────────────
    joblib.dump(imputer,        MODELS_DIR / "imputer.pkl")
    joblib.dump(xgb_default,    MODELS_DIR / "xgb_default.pkl")
    joblib.dump(xgb_tuned,      MODELS_DIR / "xgb_tuned.pkl")
    joblib.dump(platt_scaler,   MODELS_DIR / "platt_scaler.pkl")

    print(f"\nModeles sauvegardes dans {MODELS_DIR}")
    print(f"\nXGBoost termine.")
