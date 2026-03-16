# src/train_baseline.py

import sys
import argparse
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.impute           import SimpleImputer
from sklearn.metrics         import (accuracy_score, log_loss,
                                      classification_report,
                                      brier_score_loss)
from sklearn.calibration      import calibration_curve
import matplotlib.pyplot as plt

from config import get_tour_config, get_paths, make_dirs


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRIQUES
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(name: str,
                   model,
                   X: pd.DataFrame,
                   y: pd.Series,
                   split_name: str) -> dict:
    """
    Évalue un modèle sur les 3 métriques fondamentales du Pilier 4.
    """
    y_pred      = model.predict(X)
    y_prob      = model.predict_proba(X)[:, 1]

    acc         = accuracy_score(y, y_pred)
    ll          = log_loss(y, y_prob)
    brier       = brier_score_loss(y, y_prob)

    # Calibration : compare probabilités prédites vs fréquences réelles
    prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=10)
    calibration_error    = np.mean(np.abs(prob_true - prob_pred))

    print(f"\n  [{split_name}] {name}")
    print(f"    Accuracy          : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"    Log-loss          : {ll:.4f}  (baseline naive = {log_loss(y, np.full(len(y), 0.5)):.4f})")
    print(f"    Brier score       : {brier:.4f}")
    print(f"    Calibration error : {calibration_error:.4f}")

    return {
        'model'             : name,
        'split'             : split_name,
        'accuracy'          : acc,
        'log_loss'          : ll,
        'brier'             : brier,
        'calibration_error' : calibration_error,
    }


def plot_calibration(models_preds: list, y_valid: pd.Series,
                     models_dir: Path) -> None:
    """
    Courbe de calibration — nos probs estimées vs fréquences réelles.
    Une courbe proche de la diagonale = modèle bien calibré.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Calibration curve ───────────────────────────────────────────────────
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', label='Parfaitement calibré')

    for name, y_prob in models_preds:
        prob_true, prob_pred = calibration_curve(y_valid, y_prob, n_bins=10)
        ax.plot(prob_pred, prob_true, marker='o', label=name)

    ax.set_xlabel('Probabilité prédite')
    ax.set_ylabel('Fréquence réelle')
    ax.set_title('Courbe de calibration (Validation)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Distribution des probabilités ───────────────────────────────────────
    ax = axes[1]
    for name, y_prob in models_preds:
        ax.hist(y_prob, bins=30, alpha=0.5, label=name)

    ax.set_xlabel('Probabilité prédite p1 gagne')
    ax.set_ylabel('Nombre de matchs')
    ax.set_title('Distribution des probabilités')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(models_dir / "calibration_baseline.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nGraphique sauvegardé : calibration_baseline.png")


# ─────────────────────────────────────────────────────────────────────────────
# MODÈLES
# ─────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Entraînement baseline Logistic Regression par tour")
    parser.add_argument('--tour', default='atp', choices=['atp', 'wta'],
                        help="Tour à traiter : atp ou wta (défaut: atp)")
    args = parser.parse_args()

    tour  = args.tour.lower()
    cfg   = get_tour_config(tour)
    paths = get_paths(tour)
    make_dirs(tour)

    MODELS_DIR = paths['models_dir']

    print("=" * 55)
    print(f"BASELINE — LOGISTIC REGRESSION — {tour.upper()}")
    print("=" * 55)

    # Chargement
    splits   = joblib.load(MODELS_DIR / "splits.pkl")
    features = joblib.load(MODELS_DIR / "feature_list.pkl")

    X_train = splits['X_train']
    X_valid = splits['X_valid']
    X_test  = splits['X_test']
    y_train = splits['y_train']
    y_valid = splits['y_valid']
    y_test  = splits['y_test']

    print(f"\nTrain : {X_train.shape} | Valid : {X_valid.shape} | Test : {X_test.shape}")

    # ── Features ELO uniquement (baseline naïve) ────────────────────────────
    elo_features = [
        'elo_diff', 'elo_surface_diff', 'elo_win_prob_p1',
        'p1_surface_specialization', 'p2_surface_specialization',
        'surface_specialization_diff'
    ]
    elo_features = [f for f in elo_features if f in features]
    baseline_label = "ELO"

    # ── Entraînement ────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("ENTRAINEMENT")
    print("=" * 55)

    # 1. ELO Only — notre baseline de référence
    print(f"\n── LR {baseline_label} Only ─────────────────────────────────")
    lr_elo = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0.5)),
        ('scaler',  StandardScaler()),
        ('model',   LogisticRegression(max_iter=1000, C=1.0, random_state=42))
    ])
    lr_elo.fit(X_train[elo_features], y_train)
    print("  Entraine")

    # 2. Full features
    print("\n── LR Full (62 features) ───────────────────────────")
    lr_full = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0.5)),
        ('scaler',  StandardScaler()),
        ('model',   LogisticRegression(max_iter=1000, C=1.0, random_state=42))
    ])
    lr_full.fit(X_train[features], y_train)
    print("  Entraine")

    # ── Évaluation ──────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("EVALUATION")
    print("=" * 55)

    results = []

    # Validation
    results.append(evaluate_model(
        'LR_elo_only', lr_elo, X_valid[elo_features], y_valid, 'VALID'
    ))
    results.append(evaluate_model(
        'LR_full', lr_full, X_valid[features], y_valid, 'VALID'
    ))

    # Test — on regarde mais on n'optimise pas dessus
    if len(X_test) > 0:
        results.append(evaluate_model(
            'LR_elo_only', lr_elo, X_test[elo_features], y_test, 'TEST'
        ))
        results.append(evaluate_model(
            'LR_full', lr_full, X_test[features], y_test, 'TEST'
        ))

    # ── Tableau comparatif ──────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("TABLEAU COMPARATIF")
    print("=" * 55)
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

    # ── Coefficients les plus importants ─────────────────────────────────────
    print("\n" + "=" * 55)
    print("TOP 15 FEATURES (LR Full — coefficients)")
    print("=" * 55)
    coefs = pd.DataFrame({
        'feature': features,
        'coef'   : lr_full.named_steps['model'].coef_[0]
    }).reindex(columns=['feature', 'coef'])
    coefs['abs_coef'] = coefs['coef'].abs()
    coefs = coefs.sort_values('abs_coef', ascending=False)
    print(coefs[['feature', 'coef']].head(15).to_string(index=False))

    # ── Calibration plot ────────────────────────────────────────────────────
    models_preds = [
        (f'LR {baseline_label} only', lr_elo.predict_proba(X_valid[elo_features])[:, 1]),
        ('LR Full',                    lr_full.predict_proba(X_valid[features])[:, 1]),
    ]
    plot_calibration(models_preds, y_valid, MODELS_DIR)

    # ── Sauvegarde ──────────────────────────────────────────────────────────
    joblib.dump(lr_elo,  MODELS_DIR / "lr_elo_only.pkl")
    joblib.dump(lr_full, MODELS_DIR / "lr_full.pkl")
    joblib.dump(elo_features, MODELS_DIR / "elo_feature_list.pkl")

    print(f"\nModeles sauvegardes :")
    print(f"   {MODELS_DIR / 'lr_elo_only.pkl'}")
    print(f"   {MODELS_DIR / 'lr_full.pkl'}")
    print(f"\nBaseline terminee.")
