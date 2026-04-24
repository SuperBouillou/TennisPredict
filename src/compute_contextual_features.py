# src/compute_contextual_features.py

import re
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import get_tour_config, get_paths, make_dirs


def _parse_sets(score_str) -> int:
    """
    Retourne le nombre de sets disputés dans un score brut.
    Exemples : '6-4 3-6 7-5' → 3,  '6-2 6-1' → 2,  NaN → 0
    """
    if not isinstance(score_str, str) or not score_str.strip():
        return 0
    # Supprimer les suffixes (RET, W/O, DEF, etc.)
    s = re.sub(r'\s*(RET|W/O|DEF|Def\.?|ret\.?|ABD).*$', '', score_str.strip(),
               flags=re.IGNORECASE)
    # Compter les blocs score du type '6-4', '7-6(3)', '1-0'
    return len(re.findall(r'\d+-\d+', s))


def compute_fatigue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la fatigue calendrier pour chaque joueur avant chaque match :
    - Nombre de matchs joués sur les 7, 14, et 21 derniers jours
    - Nombre de jours depuis le dernier match
    - Nombre de sets joués sur les 7 et 14 derniers jours  [NOUVEAU]
    - Minutes jouées sur les 7 et 14 derniers jours        [NOUVEAU]
    """
    df = df.sort_values('tourney_date').reset_index(drop=True)

    # Pré-calculer sets et minutes pour chaque ligne
    if 'score' in df.columns:
        df['_n_sets'] = df['score'].apply(_parse_sets)
    else:
        df['_n_sets'] = 0

    has_minutes = 'minutes' in df.columns

    for col_id, prefix in [('p1_id', 'p1'), ('p2_id', 'p2')]:

        matches_7d  = np.zeros(len(df))
        matches_14d = np.zeros(len(df))
        matches_21d = np.zeros(len(df))
        days_since  = np.full(len(df), np.nan)
        sets_7d     = np.zeros(len(df))
        sets_14d    = np.zeros(len(df))
        minutes_7d  = np.zeros(len(df))
        minutes_14d = np.zeros(len(df))

        # Historique par joueur : {player_id: [(date, n_sets, minutes)]}
        history: dict = {}

        for i, row in df.iterrows():
            pid  = row[col_id]
            date = row['tourney_date']

            if pid in history:
                past = history[pid]
                past_dates   = np.array([x[0] for x in past])
                past_sets    = np.array([x[1] for x in past])
                past_minutes = np.array([x[2] for x in past])

                delta = (date - past_dates).astype('timedelta64[D]').astype(int)

                mask_7  = delta <= 7
                mask_14 = delta <= 14

                matches_7d[i]  = mask_7.sum()
                matches_14d[i] = mask_14.sum()
                matches_21d[i] = (delta <= 21).sum()
                days_since[i]  = delta.min()
                sets_7d[i]     = past_sets[mask_7].sum()
                sets_14d[i]    = past_sets[mask_14].sum()
                minutes_7d[i]  = past_minutes[mask_7].sum()
                minutes_14d[i] = past_minutes[mask_14].sum()
            else:
                history[pid] = []

            n_sets   = int(row['_n_sets'])
            mins     = float(row['minutes']) if has_minutes and not pd.isna(row.get('minutes')) else 0.0
            history.setdefault(pid, []).append((date, n_sets, mins))

        df[f'{prefix}_matches_7d']  = matches_7d
        df[f'{prefix}_matches_14d'] = matches_14d
        df[f'{prefix}_matches_21d'] = matches_21d
        df[f'{prefix}_days_since']  = days_since
        df[f'{prefix}_sets_7d']     = sets_7d
        df[f'{prefix}_sets_14d']    = sets_14d
        df[f'{prefix}_minutes_7d']  = minutes_7d
        df[f'{prefix}_minutes_14d'] = minutes_14d

    # Différences de fatigue
    df['fatigue_diff_7d']       = df['p1_matches_7d']  - df['p2_matches_7d']
    df['fatigue_diff_14d']      = df['p1_matches_14d'] - df['p2_matches_14d']
    df['fatigue_sets_diff_7d']  = df['p1_sets_7d']     - df['p2_sets_7d']
    df['fatigue_sets_diff_14d'] = df['p1_sets_14d']    - df['p2_sets_14d']
    df['fatigue_min_diff_7d']   = df['p1_minutes_7d']  - df['p2_minutes_7d']
    df['fatigue_min_diff_14d']  = df['p1_minutes_14d'] - df['p2_minutes_14d']

    # Nettoyage colonne intermédiaire
    df = df.drop(columns=['_n_sets'])

    print(f"Features fatigue calculees (+ sets/minutes)")
    return df


def compute_tournament_features(df: pd.DataFrame,
                                level_importance: dict,
                                bo5_levels: set) -> pd.DataFrame:
    """
    Encode les features contextuelles du tournoi :
    - Importance du tournoi (numérique ordinal, depuis la config du tour)
    - Format best_of encodé (depuis la config du tour)
    - Round encodé (profondeur dans le tableau)
    - Surface encodée
    """

    # Importance tournoi — depuis la config du tour
    df['tourney_importance'] = df['tourney_level'].map(level_importance).fillna(1)

    # Round — profondeur dans le tableau (plus le chiffre est élevé, plus c'est tard)
    round_order = {
        'R128': 1, 'R64': 2, 'R32': 3, 'R16': 4,
        'QF'  : 5, 'SF' : 6, 'F'  : 7, 'RR'  : 3,
        'BR'  : 6,  # Bronze/3rd place
    }
    df['round_importance'] = df['round'].map(round_order).fillna(2)

    # Surface encodée (one-hot)
    for surface in ['Hard', 'Clay', 'Grass', 'Carpet']:
        df[f'surface_{surface}'] = (df['surface'] == surface).astype(int)

    # Best of encodé — on vérifie si le niveau de tournoi peut être best-of-5
    # ET si la colonne best_of indique effectivement 5
    if bo5_levels:
        df['is_best_of_5'] = (
            (df['best_of'] == 5) &
            (df['tourney_level'].isin(bo5_levels))
        ).astype(int)
    else:
        # WTA : jamais de best-of-5
        df['is_best_of_5'] = 0

    # Rang différentiel (feature simple mais puissante)
    df['rank_diff'] = df['p1_rank'] - df['p2_rank']
    df['rank_ratio'] = np.where(
        df['p2_rank'] > 0,
        df['p1_rank'] / df['p2_rank'].replace(0, np.nan),
        np.nan
    )

    # Points de ranking différentiel
    df['rank_points_diff'] = df['p1_rank_points'] - df['p2_rank_points']

    # Différence d'âge
    df['age_diff'] = df['p1_age'] - df['p2_age']

    print(f"Features contextuelles calculees")
    return df


def compute_surface_affinity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule l'affinité ELO par surface normalisée :
    ratio ELO surface / ELO global — mesure la spécialisation.
    """
    for p in ['p1', 'p2']:
        elo_g = df[f'{p}_elo']
        elo_s = df[f'{p}_elo_surface']

        # Spécialisation surface (>1 = meilleur sur cette surface que globalement)
        df[f'{p}_surface_specialization'] = np.where(
            elo_g > 0,
            elo_s / elo_g.replace(0, np.nan),
            1.0
        )

    df['surface_specialization_diff'] = (
        df['p1_surface_specialization'] - df['p2_surface_specialization']
    )

    print(f"Features affinite surface calculees")
    return df


def final_feature_summary(df: pd.DataFrame) -> None:
    """Résumé complet de toutes les features disponibles."""

    print("\n" + "=" * 55)
    print("RESUME FEATURES — PILIER 2 COMPLET")
    print("=" * 55)

    groups = {
        'ELO'          : [c for c in df.columns if 'elo' in c.lower()],
        'Win Rate'     : [c for c in df.columns if 'winrate' in c],
        'Stats service': [c for c in df.columns if 'roll' in c],
        'H2H'          : [c for c in df.columns if 'h2h' in c],
        'Fatigue'      : [c for c in df.columns if any(x in c for x in ['7d', '14d', '21d', 'days_since'])],
        'Contexte'     : [c for c in df.columns if any(x in c for x in
                          ['importance', 'surface_', 'best_of_5', 'rank_diff',
                           'rank_ratio', 'age_diff', 'streak'])],
        'Specialisation': [c for c in df.columns if 'specialization' in c],
    }

    total_features = 0
    for group, cols in groups.items():
        print(f"\n  {group} ({len(cols)}) :")
        for c in cols[:5]:
            missing = df[c].isna().mean()
            print(f"    {c:<40} NaN: {missing:.1%}")
        if len(cols) > 5:
            print(f"    ... et {len(cols)-5} autres")
        total_features += len(cols)

    print(f"\n  {'─'*50}")
    print(f"  TOTAL features ML          : {total_features}")
    print(f"  Shape dataset final        : {df.shape}")
    print(f"  Periode                    : "
          f"{df['tourney_date'].min().date()} → {df['tourney_date'].max().date()}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calcul features contextuelles par tour")
    parser.add_argument('--tour', default='atp', choices=['atp', 'wta'],
                        help="Tour à traiter : atp ou wta (défaut: atp)")
    args = parser.parse_args()

    tour  = args.tour.lower()
    cfg   = get_tour_config(tour)
    paths = get_paths(tour)
    make_dirs(tour)

    PROCESSED_DIR = paths['processed_dir']

    print("=" * 55)
    print(f"CALCUL FEATURES CONTEXTUELLES — {tour.upper()}")
    print("=" * 55)

    df = pd.read_parquet(PROCESSED_DIR / "matches_with_h2h.parquet")
    print(f"\nDataset : {len(df):,} matchs, {df.shape[1]} colonnes\n")

    print("── Fatigue calendrier ───────────────────────────────")
    df = compute_fatigue(df)

    print("── Features tournoi ─────────────────────────────────")
    df = compute_tournament_features(
        df,
        level_importance=cfg['level_importance'],
        bo5_levels=cfg['bo5_levels'],
    )

    print("── Affinite surface ─────────────────────────────────")
    df = compute_surface_affinity(df)

    final_feature_summary(df)

    # Sauvegarde dataset final Pilier 2
    output = PROCESSED_DIR / "matches_features_final.parquet"
    df.to_parquet(output, index=False)
    print(f"\nSauvegarde : matches_features_final.parquet")
    print(f"\nPilier 2 termine.")
