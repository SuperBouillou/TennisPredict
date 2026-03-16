# src/restructure_data.py

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from config import get_tour_config, get_paths, make_dirs


def restructure_matches(df: pd.DataFrame, random_seed: int = 42) -> pd.DataFrame:
    """
    Transforme le format winner/loser en format neutre player1/player2.

    - Player1 et Player2 sont assignés aléatoirement
    - target = 1 si player1 gagne, 0 si player2 gagne
    - Élimine le data leakage structurel du format original
    """
    rng = np.random.default_rng(random_seed)
    n   = len(df)

    # Tirage aléatoire : True = winner devient player1, False = winner devient player2
    flip = rng.random(n) > 0.5

    # ── Colonnes à remapper ──────────────────────────────────────────────────

    player_cols = {
        'id'           : ('winner_id',           'loser_id'),
        'name'         : ('winner_name',         'loser_name'),
        'hand'         : ('winner_hand',         'loser_hand'),
        'age'          : ('winner_age',          'loser_age'),
        'rank'         : ('winner_rank',         'loser_rank'),
        'rank_points'  : ('winner_rank_points',  'loser_rank_points'),
        'ace'          : ('w_ace',               'l_ace'),
        'df'           : ('w_df',                'l_df'),
        'svpt'         : ('w_svpt',              'l_svpt'),
        '1stIn'        : ('w_1stIn',             'l_1stIn'),
        '1stWon'       : ('w_1stWon',            'l_1stWon'),
        '2ndWon'       : ('w_2ndWon',            'l_2ndWon'),
        'SvGms'        : ('w_SvGms',             'l_SvGms'),
        'bpSaved'      : ('w_bpSaved',           'l_bpSaved'),
        'bpFaced'      : ('w_bpFaced',           'l_bpFaced'),
    }

    # ── Construction du nouveau DataFrame ───────────────────────────────────

    records = {}

    # Colonnes contextuelles — inchangées
    context_cols = [
        'tourney_id', 'tourney_name', 'tourney_date', 'tourney_level',
        'surface', 'draw_size', 'best_of', 'round', 'year',
        'score', 'minutes'
    ]
    for col in context_cols:
        if col in df.columns:
            records[col] = df[col].values

    # Colonnes joueurs — assignment selon flip
    for stat, (w_col, l_col) in player_cols.items():
        if w_col not in df.columns or l_col not in df.columns:
            continue

        w_vals = df[w_col].values
        l_vals = df[l_col].values

        records[f'p1_{stat}'] = np.where(flip, w_vals, l_vals)
        records[f'p2_{stat}'] = np.where(flip, l_vals, w_vals)

    # Target : 1 si player1 a gagné, 0 si player2 a gagné
    records['target'] = flip.astype(int)

    df_out = pd.DataFrame(records)

    # ── Validation ───────────────────────────────────────────────────────────
    p1_wins = df_out['target'].mean()
    print(f"  ✅ Target équilibré : player1 gagne {p1_wins:.1%} du temps (attendu ~50%)")

    return df_out


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule des features dérivées directement des stats brutes du match.
    Ces features seront utilisées pour construire les moyennes glissantes.

    Note : ces stats sont CELLES DU MATCH — elles ne seront PAS utilisées
    comme features ML directes (data leakage). Elles servent à calculer
    les moyennes glissantes sur les matchs PASSÉS.
    """
    eps = 1e-6  # Éviter division par zéro

    for p in ['p1', 'p2']:

        svpt   = df.get(f'{p}_svpt',   pd.Series(np.nan, index=df.index))
        first  = df.get(f'{p}_1stIn',  pd.Series(np.nan, index=df.index))
        fw     = df.get(f'{p}_1stWon', pd.Series(np.nan, index=df.index))
        sw     = df.get(f'{p}_2ndWon', pd.Series(np.nan, index=df.index))
        bpf    = df.get(f'{p}_bpFaced',pd.Series(np.nan, index=df.index))
        bps    = df.get(f'{p}_bpSaved',pd.Series(np.nan, index=df.index))
        ace    = df.get(f'{p}_ace',    pd.Series(np.nan, index=df.index))
        df_col = df.get(f'{p}_df',     pd.Series(np.nan, index=df.index))

        second = svpt - first  # Points de 2ème service

        # % 1er service rentré
        df[f'{p}_1stIn_pct']   = first  / (svpt  + eps)
        # % points gagnés sur 1er service
        df[f'{p}_1stWon_pct']  = fw     / (first + eps)
        # % points gagnés sur 2ème service
        df[f'{p}_2ndWon_pct']  = sw     / (second.clip(lower=0) + eps)
        # % break points sauvés
        df[f'{p}_bpSaved_pct'] = bps    / (bpf   + eps)
        # Ace ratio (aces par point de service)
        df[f'{p}_ace_ratio']   = ace    / (svpt  + eps)
        # Double fautes ratio
        df[f'{p}_df_ratio']    = df_col / (svpt  + eps)

    return df


def filter_for_ml(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtre le dataset pour le ML :
    - Retire Davis Cup (stats absentes, contexte différent)
    - Retire Surface Unknown
    - Garde 1968+ pour l'ELO, mais flag les matchs pré-1991 (pas de stats)
    """
    n_init = len(df)

    # Retirer Davis Cup
    df = df[df['tourney_level'] != 'D'].copy()
    print(f"  Après retrait Davis Cup    : {len(df):>7,} matchs")

    # Retirer surface inconnue
    df = df[df['surface'] != 'Unknown'].copy()
    print(f"  Après retrait Unknown surf : {len(df):>7,} matchs")

    # Flag pre/post 1991 (stats fiables)
    df['has_stats'] = (df['year'] >= 1991).astype(int)
    stats_ok = df['has_stats'].sum()
    print(f"  Matchs avec stats (≥1991)  : {stats_ok:>7,} ({stats_ok/len(df):.1%})")

    print(f"\n  Total supprimés : {n_init - len(df):,}")

    return df.reset_index(drop=True)


def audit_restructured(df: pd.DataFrame) -> None:
    """
    Vérifications rapides sur le dataset restructuré.
    """
    print("\n" + "=" * 55)
    print("AUDIT DATASET RESTRUCTURÉ")
    print("=" * 55)

    print(f"\n  Shape              : {df.shape}")
    print(f"  Période            : {df['tourney_date'].min().date()} "
          f"→ {df['tourney_date'].max().date()}")
    print(f"  Target (p1 wins)   : {df['target'].mean():.3f} (doit être ~0.500)")
    print(f"\n  Colonnes p1        : {[c for c in df.columns if c.startswith('p1_')]}")
    print(f"\n  Colonnes p2        : {[c for c in df.columns if c.startswith('p2_')]}")
    print(f"\n  Colonnes contexte  : {[c for c in df.columns if not c.startswith(('p1_', 'p2_'))]}")

    # Vérification target par surface
    print("\n  Target par surface (doit être ~0.5 partout) :")
    print(df.groupby('surface')['target'].mean().round(3).to_string())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Restructuration winner/loser → p1/p2 neutre.")
    parser.add_argument(
        '--tour', choices=['atp', 'wta'], default='atp',
        help="Circuit à traiter : atp (défaut) ou wta"
    )
    args = parser.parse_args()

    tour = args.tour
    cfg   = get_tour_config(tour)
    paths = get_paths(tour)
    make_dirs(tour)

    processed_dir = paths['processed_dir']

    print("=" * 55)
    print(f"RESTRUCTURATION DU DATASET {tour.upper()}")
    print("=" * 55)

    # Chargement
    df = pd.read_parquet(processed_dir / "matches_consolidated.parquet")
    print(f"\n📂 Chargé : {len(df):,} matchs\n")

    # Restructuration
    print("1. Restructuration winner/loser → player1/player2")
    df = restructure_matches(df)

    # Features dérivées
    print("\n2. Calcul des features dérivées (ratios de stats)")
    df = add_derived_features(df)

    # Filtrage ML
    print("\n3. Filtrage pour le ML")
    df = filter_for_ml(df)

    # Audit
    audit_restructured(df)

    # Sauvegarde
    output = processed_dir / "matches_ml_ready.parquet"
    df.to_parquet(output, index=False)
    print(f"\n💾 Sauvegardé : matches_ml_ready.parquet")
    print(f"   Shape finale : {df.shape}")
