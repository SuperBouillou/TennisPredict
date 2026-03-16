# src/prepare_ml_dataset.py

import sys
import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config import get_tour_config, get_paths, make_dirs, TEMPORAL_SPLIT


def define_feature_sets() -> dict:
    """
    Définit les groupes de features utilisables pour le ML.
    On exclut tout ce qui causerait du data leakage.
    """
    features = {

        # ── Disponibles sur TOUS les matchs ────────────────────────────────
        'elo': [
            'elo_diff',
            'elo_surface_diff',
            'elo_win_prob_p1',
            'p1_surface_specialization',
            'p2_surface_specialization',
            'surface_specialization_diff',
        ],

        'ranking': [
            'rank_diff',
            'rank_ratio',
            'rank_points_diff',
        ],

        'forme': [
            'p1_winrate_5',  'p2_winrate_5',  'winrate_diff_5',
            'p1_winrate_10', 'p2_winrate_10', 'winrate_diff_10',
            'p1_winrate_20', 'p2_winrate_20', 'winrate_diff_20',
            'p1_streak',     'p2_streak',     'streak_diff',
        ],

        'qualite_adversaires': [
            'p1_winrate_quality_5',  'p2_winrate_quality_5',  'winrate_quality_diff_5',
            'p1_winrate_quality_10', 'p2_winrate_quality_10', 'winrate_quality_diff_10',
            'p1_winrate_quality_20', 'p2_winrate_quality_20', 'winrate_quality_diff_20',
        ],

        'surface_forme': [
            'p1_winrate_surf_Hard',  'p2_winrate_surf_Hard',
            'p1_winrate_surf_Clay',  'p2_winrate_surf_Clay',
            'p1_winrate_surf_Grass', 'p2_winrate_surf_Grass',
            'winrate_surf_diff_Hard',
            'winrate_surf_diff_Clay',
            'winrate_surf_diff_Grass',
        ],

        'h2h': [
            'h2h_p1_winrate',
            'h2h_surf_p1_winrate',
            'h2h_total',
            'h2h_played',
        ],

        'fatigue': [
            'p1_matches_7d',  'p2_matches_7d',  'fatigue_diff_7d',
            'p1_matches_14d', 'p2_matches_14d', 'fatigue_diff_14d',
            'p1_days_since',  'p2_days_since',
        ],

        'contexte': [
            'tourney_importance',
            'round_importance',
            'is_best_of_5',
            'age_diff',
            'surface_Hard', 'surface_Clay', 'surface_Grass', 'surface_Carpet',
        ],

        # Seules les features UNIQUES à Glicko-2 (non redondantes avec ELO) :
        # glicko_diff / glicko_win_prob_p1 / glicko_surface_diff sont écartées
        # car corrélées à ~0.95 avec elo_diff / elo_win_prob_p1 / elo_surface_diff.
        'glicko': [
            'glicko_rd_diff',        # Différence d'incertitude entre joueurs
            'glicko_rd_surface_diff',# Idem sur surface spécifique
            'p1_glicko_rd',          # Incertitude P1 (haut = peu de matchs récents)
            'p2_glicko_rd',          # Incertitude P2
        ],

        # ── Disponibles uniquement post-1991 ───────────────────────────────
        'stats_service': [
            'p1_1stIn_pct_roll10',   'p2_1stIn_pct_roll10',
            'p1_1stWon_pct_roll10',  'p2_1stWon_pct_roll10',
            'p1_2ndWon_pct_roll10',  'p2_2ndWon_pct_roll10',
            'p1_bpSaved_pct_roll10', 'p2_bpSaved_pct_roll10',
            'p1_ace_ratio_roll10',   'p2_ace_ratio_roll10',
            'p1_df_ratio_roll10',    'p2_df_ratio_roll10',
        ],
    }

    return features


def prepare_dataset(df: pd.DataFrame,
                    feature_sets: dict,
                    use_stats: bool = True) -> tuple:
    """
    Prépare le dataset final pour le ML.

    Paramètres :
    - use_stats : inclure les stats de service (réduit le dataset à post-1991)

    Retourne : X, y, df_meta (colonnes non-features pour le backtest)
    """

    # ── Sélection de la période ─────────────────────────────────────────────
    if use_stats:
        df = df[df['has_stats'] == 1].copy()
        print(f"  Filtrage post-1991 : {len(df):,} matchs")
    else:
        print(f"  Dataset complet    : {len(df):,} matchs")

    # ── Construction de la liste de features ────────────────────────────────
    # ELO : force relative + spécialisation de surface.
    # Glicko : uniquement les 4 features d'incertitude (RD) absentes d'ELO.
    groups = ['elo', 'glicko', 'ranking', 'forme', 'qualite_adversaires',
              'surface_forme', 'h2h', 'fatigue', 'contexte']
    if use_stats:
        groups.append('stats_service')

    all_features = []
    for g in groups:
        cols = [c for c in feature_sets[g] if c in df.columns]
        all_features.extend(cols)

    # Dédoublonnage (une feature peut apparaître dans plusieurs groupes)
    all_features = list(dict.fromkeys(all_features))

    print(f"  Features sélectionnées : {len(all_features)}")

    # ── X et y ──────────────────────────────────────────────────────────────
    X = df[all_features].copy()
    y = df['target'].copy()

    # ── Métadonnées pour le backtest ─────────────────────────────────────────
    meta_cols = ['tourney_date', 'tourney_name', 'tourney_level',
                 'surface', 'round', 'year',
                 'p1_id', 'p1_name', 'p2_id', 'p2_name',
                 'p1_rank', 'p2_rank']
    meta_cols = [c for c in meta_cols if c in df.columns]
    df_meta   = df[meta_cols].copy()

    return X, y, df_meta, all_features


def temporal_split(X: pd.DataFrame,
                   y: pd.Series,
                   df_meta: pd.DataFrame) -> dict:
    """
    Split temporel strict — pas de random split pour les séries temporelles.

    Train  : ≤ 2022  (TEMPORAL_SPLIT['train_end'])
    Valid  : 2023-2024
    Test   : ≥ 2025  (ne pas toucher avant évaluation finale)
    """
    year = df_meta['year']

    train_mask = year <= TEMPORAL_SPLIT['train_end'].year
    valid_mask = (year >= TEMPORAL_SPLIT['valid_start'].year) & (year <= TEMPORAL_SPLIT['valid_end'].year)
    test_mask  = year >= TEMPORAL_SPLIT['test_start'].year

    splits = {
        'X_train' : X[train_mask],
        'X_valid' : X[valid_mask],
        'X_test'  : X[test_mask],
        'y_train' : y[train_mask],
        'y_valid' : y[valid_mask],
        'y_test'  : y[test_mask],
        'meta_train': df_meta[train_mask],
        'meta_valid': df_meta[valid_mask],
        'meta_test' : df_meta[test_mask],
    }

    print(f"\n  Split temporel :")
    train_end_y  = TEMPORAL_SPLIT['train_end'].year
    valid_s_y    = TEMPORAL_SPLIT['valid_start'].year
    valid_e_y    = TEMPORAL_SPLIT['valid_end'].year
    test_start_y = TEMPORAL_SPLIT['test_start'].year
    print(f"    Train  (<=  {train_end_y}) : {train_mask.sum():>7,} matchs")
    print(f"    Valid  ({valid_s_y}-{valid_e_y}) : {valid_mask.sum():>7,} matchs")
    print(f"    Test   (>= {test_start_y}) : {test_mask.sum():>7,} matchs")

    return splits


def audit_splits(splits: dict, all_features: list) -> None:

    print("\n" + "=" * 55)
    print("AUDIT DATASET ML")
    print("=" * 55)

    for name in ['train', 'valid', 'test']:
        X = splits[f'X_{name}']
        y = splits[f'y_{name}']
        nan_pct = X.isna().mean().mean()
        print(f"\n  {name.upper()}")
        print(f"    Shape     : {X.shape}")
        print(f"    Target=1  : {y.mean():.3f} (attendu ~0.500)")
        print(f"    NaN moyen : {nan_pct:.1%}")

    print(f"\n  Features ({len(all_features)}) :")
    # Afficher les NaN par groupe
    X_train = splits['X_train']
    nan_by_feature = X_train.isna().mean().sort_values(ascending=False)
    problematic = nan_by_feature[nan_by_feature > 0.1]
    if len(problematic) > 0:
        print(f"\n  Features avec >10% NaN dans le train :")
        for feat, pct in problematic.items():
            print(f"    {feat:<45}: {pct:.1%}")
    else:
        print(f"  Aucune feature avec >10% NaN")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Préparation dataset ML par tour")
    parser.add_argument('--tour', default='atp', choices=['atp', 'wta'],
                        help="Tour à traiter : atp ou wta (défaut: atp)")
    args = parser.parse_args()

    tour  = args.tour.lower()
    cfg   = get_tour_config(tour)
    paths = get_paths(tour)
    make_dirs(tour)

    PROCESSED_DIR = paths['processed_dir']
    MODELS_DIR    = paths['models_dir']

    print("=" * 55)
    print(f"PREPARATION DATASET ML — {tour.upper()}")
    print("=" * 55)

    df = pd.read_parquet(PROCESSED_DIR / "matches_features_final.parquet")
    print(f"\nDataset brut : {len(df):,} matchs, {df.shape[1]} colonnes\n")

    feature_sets = define_feature_sets()

    # ── Version complète (avec stats service) ───────────────────────────────
    print("-- Dataset avec stats service (post-1991) ----------")
    X, y, df_meta, features = prepare_dataset(df, feature_sets, use_stats=True)

    splits = temporal_split(X, y, df_meta)
    audit_splits(splits, features)

    # Sauvegarde
    joblib.dump(splits,   MODELS_DIR / "splits.pkl")
    joblib.dump(features, MODELS_DIR / "feature_list.pkl")

    print(f"\nSauvegardes :")
    print(f"   splits.pkl       → train/valid/test")
    print(f"   feature_list.pkl → {len(features)} features")
    print(f"\nDonnees pretes pour la modelisation.")
