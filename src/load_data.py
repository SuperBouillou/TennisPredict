# src/load_data.py

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from config import get_tour_config, get_paths, make_dirs


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def load_matches(raw_dir: Path, cfg: dict) -> pd.DataFrame:
    """
    Charge tous les fichiers {prefix}_matches_YYYY.csv (tour principal).
    Trie par date et ajoute la colonne year.
    """
    prefix = cfg['file_prefix']
    files = sorted(raw_dir.glob(f"{prefix}_matches_[0-9][0-9][0-9][0-9].csv"))

    if not files:
        raise FileNotFoundError(f"Aucun fichier {prefix}_matches_*.csv dans {raw_dir}")

    dfs = []
    for f in files:
        df = pd.read_csv(f, dtype=str)   # Tout en str pour éviter les conflits de types
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # Typage explicite colonne par colonne
    df_all['tourney_date']      = pd.to_datetime(df_all['tourney_date'],      format='%Y%m%d', errors='coerce')
    df_all['winner_id']         = pd.to_numeric(df_all['winner_id'],          errors='coerce').astype('Int64')
    df_all['loser_id']          = pd.to_numeric(df_all['loser_id'],           errors='coerce').astype('Int64')
    df_all['winner_rank']       = pd.to_numeric(df_all['winner_rank'],        errors='coerce')
    df_all['loser_rank']        = pd.to_numeric(df_all['loser_rank'],         errors='coerce')
    df_all['winner_rank_points']= pd.to_numeric(df_all['winner_rank_points'], errors='coerce')
    df_all['loser_rank_points'] = pd.to_numeric(df_all['loser_rank_points'],  errors='coerce')
    df_all['winner_age']        = pd.to_numeric(df_all['winner_age'],         errors='coerce')
    df_all['loser_age']         = pd.to_numeric(df_all['loser_age'],          errors='coerce')
    df_all['minutes']           = pd.to_numeric(df_all['minutes'],            errors='coerce')
    df_all['draw_size']         = pd.to_numeric(df_all['draw_size'],          errors='coerce')
    df_all['best_of']           = pd.to_numeric(df_all['best_of'],            errors='coerce').astype('Int64')

    # Colonnes stats numériques
    stat_cols = [
        'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon',
        'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced',
        'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon',
        'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced',
    ]
    for col in stat_cols:
        if col in df_all.columns:
            df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

    df_all['year'] = df_all['tourney_date'].dt.year
    df_all = df_all.sort_values('tourney_date').reset_index(drop=True)

    print(f"✅ Matchs chargés       : {len(df_all):>8,}")
    print(f"   Période              : {df_all['tourney_date'].min().date()} "
          f"→ {df_all['tourney_date'].max().date()}")
    print(f"   Colonnes             : {df_all.shape[1]}")

    return df_all


def load_players(raw_dir: Path, cfg: dict) -> pd.DataFrame:
    """
    Charge le fichier joueurs (atp_players.csv ou wta_players.csv).
    Colonnes réelles : player_id, name_first, name_last, hand, dob, ioc, height, wikidata_id
    """
    path = raw_dir / cfg['player_file']

    df = pd.read_csv(path, dtype=str)  # Header détecté automatiquement

    # Typage player_id
    df['player_id'] = pd.to_numeric(df['player_id'], errors='coerce').astype('Int64')

    # Nettoyage noms
    df['name_first'] = df['name_first'].fillna('').str.strip()
    df['name_last']  = df['name_last'].fillna('').str.strip()
    df['full_name']  = (df['name_first'] + ' ' + df['name_last']).str.strip()

    # Date de naissance et taille
    df['dob']    = pd.to_datetime(df['dob'], format='%Y%m%d', errors='coerce')
    df['height'] = pd.to_numeric(df['height'], errors='coerce')

    # Retirer les lignes sans ID valide
    df = df.dropna(subset=['player_id']).reset_index(drop=True)

    print(f"✅ Joueurs chargés      : {len(df):>8,}")

    return df


def load_rankings(raw_dir: Path, cfg: dict) -> pd.DataFrame:
    """
    Charge les fichiers de classement définis dans cfg['ranking_files'].
    Colonnes : ranking_date (datetime), rank (int), player_id (Int64), ranking_points (float)
    """
    ranking_files = cfg['ranking_files']
    files = [raw_dir / f for f in ranking_files if (raw_dir / f).exists()]

    if not files:
        raise FileNotFoundError(
            f"Aucun fichier de classement trouvé dans {raw_dir} "
            f"(cherché : {ranking_files})"
        )

    dfs = []
    for f in files:
        df = pd.read_csv(
            f,
            header=None,
            names=['ranking_date', 'rank', 'player_id', 'ranking_points'],
            dtype=str
        )
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # Typage explicite
    df_all['ranking_date']   = pd.to_datetime(df_all['ranking_date'], format='%Y%m%d', errors='coerce')
    df_all['rank']           = pd.to_numeric(df_all['rank'],           errors='coerce').astype('Int64')
    df_all['player_id']      = pd.to_numeric(df_all['player_id'],      errors='coerce').astype('Int64')
    df_all['ranking_points'] = pd.to_numeric(df_all['ranking_points'], errors='coerce')

    # Dédoublonnage et tri
    df_all = (df_all
              .dropna(subset=['ranking_date', 'player_id', 'rank'])
              .drop_duplicates()
              .sort_values('ranking_date')
              .reset_index(drop=True))

    print(f"✅ Classements chargés  : {len(df_all):>8,}")
    print(f"   Période              : {df_all['ranking_date'].min().date()} "
          f"→ {df_all['ranking_date'].max().date()}")

    return df_all


# ─────────────────────────────────────────────────────────────────────────────
# NETTOYAGE MATCHS
# ─────────────────────────────────────────────────────────────────────────────

def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage du dataset matchs :
    - Supprime les matchs sans date, sans score, sans joueurs identifiés
    - Supprime les walkovers
    - Normalise la surface
    """
    n = len(df)

    df = df.dropna(subset=['tourney_date'])
    df = df.dropna(subset=['winner_id', 'loser_id'])
    df = df.dropna(subset=['score'])
    df = df[~df['score'].str.upper().str.contains('W/O|WALKOVER', na=False)]

    surface_map = {'Hard': 'Hard', 'Clay': 'Clay', 'Grass': 'Grass', 'Carpet': 'Carpet'}
    df['surface'] = df['surface'].map(surface_map).fillna('Unknown')

    df = df.reset_index(drop=True)
    print(f"✅ Nettoyage            : {n:,} → {len(df):,} matchs (retirés : {n - len(df):,})")

    return df


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Sélectionne les colonnes utiles pour le pipeline ML."""

    core = [
        'tourney_id', 'tourney_name', 'tourney_date', 'tourney_level',
        'surface', 'draw_size', 'best_of', 'round', 'year',
        'winner_id', 'winner_name', 'winner_hand', 'winner_age',
        'winner_rank', 'winner_rank_points',
        'loser_id',  'loser_name',  'loser_hand',  'loser_age',
        'loser_rank',  'loser_rank_points',
        'score', 'minutes',
        'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon',
        'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced',
        'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon',
        'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced',
    ]

    keep = [c for c in core if c in df.columns]
    return df[keep].copy()


# ─────────────────────────────────────────────────────────────────────────────
# SAUVEGARDE & RAPPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_all(df_matches: pd.DataFrame,
             df_players: pd.DataFrame,
             df_rankings: pd.DataFrame,
             processed_dir: Path) -> None:

    df_matches.to_parquet( processed_dir / "matches_consolidated.parquet", index=False)
    df_players.to_parquet( processed_dir / "players.parquet",              index=False)
    df_rankings.to_parquet(processed_dir / "rankings.parquet",             index=False)

    print(f"\n💾 Sauvegardés dans : {processed_dir}")
    print(f"   matches_consolidated.parquet → {len(df_matches):>8,} lignes")
    print(f"   players.parquet              → {len(df_players):>8,} lignes")
    print(f"   rankings.parquet             → {len(df_rankings):>8,} lignes")


def rapport_types(df: pd.DataFrame) -> None:
    """Affiche les types finaux des colonnes clés — vérification rapide."""
    cols = ['winner_id', 'loser_id', 'winner_rank', 'winner_age',
            'w_ace', 'tourney_date', 'year']
    print("\n   Dtypes colonnes clés :")
    for c in cols:
        if c in df.columns:
            print(f"   {c:<25}: {df[c].dtype}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Chargement et nettoyage des données tennis.")
    parser.add_argument(
        '--tour', choices=['atp', 'wta'], default='atp',
        help="Circuit à traiter : atp (défaut) ou wta"
    )
    args = parser.parse_args()

    tour = args.tour
    cfg   = get_tour_config(tour)
    paths = get_paths(tour)
    make_dirs(tour)

    raw_dir       = paths['raw_dir']
    processed_dir = paths['processed_dir']

    print("=" * 55)
    print(f"CHARGEMENT DES DONNÉES {tour.upper()}")
    print("=" * 55)

    print("\n── Matchs ──────────────────────────────────────────")
    df_matches = load_matches(raw_dir, cfg)

    print("\n── Joueurs ─────────────────────────────────────────")
    df_players = load_players(raw_dir, cfg)

    print("\n── Classements ─────────────────────────────────────")
    df_rankings = load_rankings(raw_dir, cfg)

    print("\n── Nettoyage ───────────────────────────────────────")
    df_matches = clean_matches(df_matches)
    df_matches = select_columns(df_matches)

    rapport_types(df_matches)

    print("\n── Sauvegarde ──────────────────────────────────────")
    save_all(df_matches, df_players, df_rankings, processed_dir)

    print("\n✅ Pilier 1 terminé.")
