# src/compute_elo.py — VERSION CORRIGÉE

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import get_tour_config, get_paths, make_dirs

ELO_CONFIG = {
    'initial_rating' : 1500,
    'k_base'         : 32,
    'd_scale'        : 400,
}

SURFACE_LIST = ['Hard', 'Clay', 'Grass', 'Carpet']


def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / ELO_CONFIG['d_scale']))


def update_elo(rating_a, rating_b, score_a, k):
    exp_a = expected_score(rating_a, rating_b)
    new_a = rating_a + k * (score_a - exp_a)
    new_b = rating_b + k * ((1 - score_a) - (1 - exp_a))
    return new_a, new_b


def compute_elo(df: pd.DataFrame, level_k: dict) -> pd.DataFrame:
    """
    Calcule les ELO sur le dataset original (winner/loser).
    Crée une clé unique match_key pour jointure sans explosion.
    """
    # Clé unique par match : évite le produit cartésien à la jointure
    df = df.copy()
    df['match_key'] = (
        df['tourney_id'].astype(str) + '__' +
        df['winner_id'].astype(str)  + '__' +
        df['loser_id'].astype(str)
    )

    elo_global  = {}
    elo_surface = {s: {} for s in SURFACE_LIST}
    init        = ELO_CONFIG['initial_rating']

    w_elo_before   = np.full(len(df), np.nan)
    l_elo_before   = np.full(len(df), np.nan)
    w_elo_s_before = np.full(len(df), np.nan)
    l_elo_s_before = np.full(len(df), np.nan)

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Calcul ELO"):
        w_id    = row['winner_id']
        l_id    = row['loser_id']
        surface = row['surface']
        k       = level_k.get(row['tourney_level'], ELO_CONFIG['k_base'])

        # ELO Global — enregistrer AVANT
        wg = elo_global.get(w_id, init)
        lg = elo_global.get(l_id, init)
        w_elo_before[i] = wg
        l_elo_before[i] = lg
        elo_global[w_id], elo_global[l_id] = update_elo(wg, lg, 1.0, k)

        # ELO Surface — enregistrer AVANT
        if surface in elo_surface:
            ws = elo_surface[surface].get(w_id, init)
            ls = elo_surface[surface].get(l_id, init)
            w_elo_s_before[i] = ws
            l_elo_s_before[i] = ls
            elo_surface[surface][w_id], elo_surface[surface][l_id] = \
                update_elo(ws, ls, 1.0, k)

    df['winner_elo']         = w_elo_before
    df['loser_elo']          = l_elo_before
    df['winner_elo_surface'] = w_elo_s_before
    df['loser_elo_surface']  = l_elo_s_before

    return df, elo_global, elo_surface


def join_elo_to_ml(df_ml: pd.DataFrame, df_elo: pd.DataFrame) -> pd.DataFrame:
    """
    Jointure propre via match_key unique.
    """
    # Créer la même clé dans df_ml
    # df_ml a p1_id/p2_id — on reconstruit winner_id/loser_id via target
    df_ml = df_ml.copy()

    df_ml['winner_id'] = np.where(df_ml['target'] == 1, df_ml['p1_id'], df_ml['p2_id'])
    df_ml['loser_id']  = np.where(df_ml['target'] == 1, df_ml['p2_id'], df_ml['p1_id'])

    df_ml['match_key'] = (
        df_ml['tourney_id'].astype(str) + '__' +
        df_ml['winner_id'].astype(str)  + '__' +
        df_ml['loser_id'].astype(str)
    )

    # Sous-ensemble ELO à joindre
    elo_sub = df_elo[[
        'match_key',
        'winner_elo', 'loser_elo',
        'winner_elo_surface', 'loser_elo_surface'
    ]].drop_duplicates('match_key')  # Sécurité anti-doublon

    df_ml = df_ml.merge(elo_sub, on='match_key', how='left')

    # Remapper en p1/p2
    is_p1_winner = df_ml['target'] == 1

    df_ml['p1_elo']         = np.where(is_p1_winner, df_ml['winner_elo'],         df_ml['loser_elo'])
    df_ml['p2_elo']         = np.where(is_p1_winner, df_ml['loser_elo'],          df_ml['winner_elo'])
    df_ml['p1_elo_surface'] = np.where(is_p1_winner, df_ml['winner_elo_surface'], df_ml['loser_elo_surface'])
    df_ml['p2_elo_surface'] = np.where(is_p1_winner, df_ml['loser_elo_surface'],  df_ml['winner_elo_surface'])

    df_ml['elo_diff']         = df_ml['p1_elo']         - df_ml['p2_elo']
    df_ml['elo_surface_diff'] = df_ml['p1_elo_surface'] - df_ml['p2_elo_surface']
    df_ml['elo_win_prob_p1']  = 1 / (1 + 10 ** (-df_ml['elo_diff'] / ELO_CONFIG['d_scale']))

    # Nettoyage
    df_ml = df_ml.drop(columns=[
        'winner_elo', 'loser_elo',
        'winner_elo_surface', 'loser_elo_surface',
        'winner_id', 'loser_id', 'match_key'
    ], errors='ignore')

    return df_ml


def snapshot_final_ratings(elo_global, elo_surface, df_players):
    """Snapshot des ratings finaux avec noms des joueurs."""

    # Uniformiser le type player_id
    df_players = df_players.copy()
    df_players['player_id'] = pd.to_numeric(df_players['player_id'], errors='coerce')

    rows = []
    for pid in elo_global:
        rows.append({
            'player_id'  : pid,
            'elo_global' : elo_global.get(pid,              ELO_CONFIG['initial_rating']),
            'elo_Hard'   : elo_surface['Hard'].get(pid,     ELO_CONFIG['initial_rating']),
            'elo_Clay'   : elo_surface['Clay'].get(pid,     ELO_CONFIG['initial_rating']),
            'elo_Grass'  : elo_surface['Grass'].get(pid,    ELO_CONFIG['initial_rating']),
            'elo_Carpet' : elo_surface['Carpet'].get(pid,   ELO_CONFIG['initial_rating']),
        })

    df_snap = pd.DataFrame(rows)
    df_snap['player_id'] = pd.to_numeric(df_snap['player_id'], errors='coerce')

    df_snap = df_snap.merge(
        df_players[['player_id', 'full_name']],
        on='player_id', how='left'
    )

    return df_snap.sort_values('elo_global', ascending=False).reset_index(drop=True)


def audit_elo(df_ml, df_snap):
    print("\n" + "=" * 55)
    print("AUDIT ELO")
    print("=" * 55)
    print(f"\n  Shape dataset ML           : {df_ml.shape}")
    print(f"  Matchs avec ELO global     : {df_ml['p1_elo'].notna().sum():,} "
          f"({df_ml['p1_elo'].notna().mean():.1%})")
    print(f"  Matchs avec ELO surface    : {df_ml['p1_elo_surface'].notna().sum():,} "
          f"({df_ml['p1_elo_surface'].notna().mean():.1%})")
    print(f"  elo_win_prob_p1 moyenne    : {df_ml['elo_win_prob_p1'].mean():.3f} (attendu ~0.500)")

    higher_elo_wins = (
        ((df_ml['elo_diff'] > 0) & (df_ml['target'] == 1)) |
        ((df_ml['elo_diff'] < 0) & (df_ml['target'] == 0))
    ).mean()
    print(f"  Accuracy ELO brut          : {higher_elo_wins:.1%} (attendu 60-70%)")

    print(f"\n  Top 10 joueurs (ELO global final) :")
    print(df_snap[['full_name', 'elo_global', 'elo_Hard', 'elo_Clay', 'elo_Grass']]
          .head(10).to_string(index=False))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calcul ELO par tour")
    parser.add_argument('--tour', default='atp', choices=['atp', 'wta'],
                        help="Tour à traiter : atp ou wta (défaut: atp)")
    args = parser.parse_args()

    tour  = args.tour.lower()
    cfg   = get_tour_config(tour)
    paths = get_paths(tour)
    make_dirs(tour)

    PROCESSED_DIR = paths['processed_dir']

    # Construire LEVEL_K depuis la config du tour
    LEVEL_K = cfg['k_factor_map']

    print("=" * 55)
    print(f"CALCUL ELO {tour.upper()} — VERSION CORRIGÉE")
    print("=" * 55)

    # Dataset original winner/loser pour le calcul
    df_raw = pd.read_parquet(PROCESSED_DIR / "matches_consolidated.parquet")
    df_raw = df_raw[df_raw['tourney_level'] != 'D'].copy()
    df_raw = df_raw[df_raw['surface'] != 'Unknown'].copy()
    df_raw = df_raw.sort_values('tourney_date').reset_index(drop=True)
    print(f"\nDataset source : {len(df_raw):,} matchs")

    # Calcul ELO
    df_raw, elo_global, elo_surface = compute_elo(df_raw, LEVEL_K)

    # Snapshot ratings finaux
    df_players = pd.read_parquet(PROCESSED_DIR / "players.parquet")
    df_snap    = snapshot_final_ratings(elo_global, elo_surface, df_players)

    # Jointure sur dataset ML restructuré
    df_ml = pd.read_parquet(PROCESSED_DIR / "matches_ml_ready.parquet")
    print(f"Dataset ML     : {len(df_ml):,} matchs")

    df_ml = join_elo_to_ml(df_ml, df_raw)

    # Audit
    audit_elo(df_ml, df_snap)

    # Sauvegarde
    df_ml.to_parquet(PROCESSED_DIR / "matches_with_elo.parquet", index=False)
    df_snap.to_parquet(PROCESSED_DIR / "elo_ratings_final.parquet", index=False)

    print(f"\nSauvegardes :")
    print(f"   matches_with_elo.parquet  → {len(df_ml):,} matchs")
    print(f"   elo_ratings_final.parquet → {len(df_snap):,} joueurs")
