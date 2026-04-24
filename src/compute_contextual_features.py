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
    Calcule la fatigue calendrier pour chaque joueur avant chaque match.

    Approche vectorisée par joueur :
    - groupby player_id → dates triées par joueur
    - searchsorted (O log n) pour trouver les bornes des fenêtres 7/14/21j
    - cumsum sur sets et minutes → somme de fenêtre en O(1)

    Nettement plus rapide que la boucle row-by-row originale,
    et produit les mêmes résultats.
    """
    df = df.sort_values('tourney_date').reset_index(drop=True)

    # Pré-calculer sets et minutes (vectorisé)
    if 'score' in df.columns:
        df['_n_sets'] = df['score'].apply(_parse_sets).astype(np.int16)
    else:
        df['_n_sets'] = np.int16(0)

    if 'minutes' in df.columns:
        df['_minutes'] = pd.to_numeric(df['minutes'], errors='coerce').fillna(0.0)
    else:
        df['_minutes'] = 0.0

    TD = {7: np.timedelta64(7, 'D'), 14: np.timedelta64(14, 'D'),
          21: np.timedelta64(21, 'D')}

    for col_id, prefix in [('p1_id', 'p1'), ('p2_id', 'p2')]:

        out_m7   = np.zeros(len(df), dtype=np.int16)
        out_m14  = np.zeros(len(df), dtype=np.int16)
        out_m21  = np.zeros(len(df), dtype=np.int16)
        out_ds   = np.full(len(df), np.nan)
        out_s7   = np.zeros(len(df), dtype=np.float32)
        out_s14  = np.zeros(len(df), dtype=np.float32)
        out_min7  = np.zeros(len(df), dtype=np.float32)
        out_min14 = np.zeros(len(df), dtype=np.float32)

        for _pid, grp in df.groupby(col_id, sort=False):
            idx   = grp.index.values               # positions in df
            dates = grp['tourney_date'].values      # already sorted (df is sorted)
            sets  = grp['_n_sets'].values.astype(np.float32)
            mins  = grp['_minutes'].values.astype(np.float32)
            n     = len(dates)

            # Cumulative sums (include current match, so use exclusive prefix)
            sets_cs = np.concatenate([[0.0], np.cumsum(sets)])
            mins_cs = np.concatenate([[0.0], np.cumsum(mins)])

            for j in range(1, n):
                cur = dates[j]

                # searchsorted: first match strictly within window (before current)
                lo7  = np.searchsorted(dates[:j], cur - TD[7],  side='left')
                lo14 = np.searchsorted(dates[:j], cur - TD[14], side='left')
                lo21 = np.searchsorted(dates[:j], cur - TD[21], side='left')

                out_m7[idx[j]]  = j - lo7
                out_m14[idx[j]] = j - lo14
                out_m21[idx[j]] = j - lo21

                # days since last match
                out_ds[idx[j]] = (cur - dates[j - 1]) / np.timedelta64(1, 'D')

                # sets and minutes in windows (using prefix sums)
                out_s7[idx[j]]    = sets_cs[j] - sets_cs[lo7]
                out_s14[idx[j]]   = sets_cs[j] - sets_cs[lo14]
                out_min7[idx[j]]  = mins_cs[j] - mins_cs[lo7]
                out_min14[idx[j]] = mins_cs[j] - mins_cs[lo14]

        df[f'{prefix}_matches_7d']  = out_m7
        df[f'{prefix}_matches_14d'] = out_m14
        df[f'{prefix}_matches_21d'] = out_m21
        df[f'{prefix}_days_since']  = out_ds
        df[f'{prefix}_sets_7d']     = out_s7
        df[f'{prefix}_sets_14d']    = out_s14
        df[f'{prefix}_minutes_7d']  = out_min7
        df[f'{prefix}_minutes_14d'] = out_min14

    # Différences de fatigue
    df['fatigue_diff_7d']       = df['p1_matches_7d']  - df['p2_matches_7d']
    df['fatigue_diff_14d']      = df['p1_matches_14d'] - df['p2_matches_14d']
    df['fatigue_sets_diff_7d']  = df['p1_sets_7d']     - df['p2_sets_7d']
    df['fatigue_sets_diff_14d'] = df['p1_sets_14d']    - df['p2_sets_14d']
    df['fatigue_min_diff_7d']   = df['p1_minutes_7d']  - df['p2_minutes_7d']
    df['fatigue_min_diff_14d']  = df['p1_minutes_14d'] - df['p2_minutes_14d']

    df = df.drop(columns=['_n_sets', '_minutes'])

    print("Features fatigue calculees (+ sets/minutes, vectorise)")
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
