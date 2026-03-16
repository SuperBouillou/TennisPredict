# src/compute_rolling_features.py

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import get_tour_config, get_paths, make_dirs

# Fenêtres glissantes à calculer
WINDOWS = [5, 10, 20]

# Paramètres pour le win rate pondéré par qualité adversaire
RANK_FALLBACK = 300       # Rang conservateur pour adversaires sans classement ATP (seuil ~250)
MIN_QUALITY_PERIODS = 3   # Minimum de matchs précédents pour calculer le quality win rate

# Stats sur lesquelles calculer les moyennes glissantes
ROLLING_STATS = [
    'ace_ratio',    # Aces par point de service
    'df_ratio',     # Double fautes par point de service
    '1stIn_pct',    # % 1er service rentré
    '1stWon_pct',   # % points gagnés sur 1er service
    '2ndWon_pct',   # % points gagnés sur 2ème service
    'bpSaved_pct',  # % break points sauvés
]


def build_player_match_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme le dataset ML (p1/p2) en vue par joueur.
    Chaque match apparaît 2 fois : une fois pour p1, une fois pour p2.
    Permet de calculer l'historique individuel de chaque joueur.

    Précondition : df doit avoir un index 0-based contigu (pd.read_parquet garantit cela).
    """
    stat_cols = ROLLING_STATS + ['rank', 'rank_points', 'age']

    # Vue joueur 1
    p1_cols = {f'p1_{s}': s for s in stat_cols if f'p1_{s}' in df.columns}
    df_p1 = df[['tourney_date', 'tourney_level', 'surface',
                 'p1_id', 'target'] + list(p1_cols.keys())].copy()
    df_p1 = df_p1.rename(columns={'p1_id': 'player_id', 'target': 'won'})
    df_p1 = df_p1.rename(columns=p1_cols)
    # NOUVEAU : rang de l'adversaire (p2 est l'adversaire de p1)
    # .values requis car df_p1 est une copie dont l'index peut différer de df
    df_p1['opponent_rank'] = df['p2_rank'].values

    # Vue joueur 2
    p2_cols = {f'p2_{s}': s for s in stat_cols if f'p2_{s}' in df.columns}
    df_p2 = df[['tourney_date', 'tourney_level', 'surface',
                 'p2_id', 'target'] + list(p2_cols.keys())].copy()
    df_p2 = df_p2.rename(columns={'p2_id': 'player_id'})
    df_p2['won'] = 1 - df_p2['target']
    df_p2 = df_p2.drop(columns=['target'])
    df_p2 = df_p2.rename(columns=p2_cols)
    # NOUVEAU : rang de l'adversaire (p1 est l'adversaire de p2)
    df_p2['opponent_rank'] = df['p1_rank'].values

    # Concaténation et tri chronologique
    df_history = pd.concat([df_p1, df_p2], ignore_index=True)
    df_history = df_history.sort_values(['player_id', 'tourney_date']).reset_index(drop=True)

    print(f"✅ Historique joueur construit : {len(df_history):,} entrées")
    return df_history


def compute_rolling_stats(df_history: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les moyennes glissantes par joueur.
    Utilise .shift(1) pour garantir no-leakage.
    """
    all_results = []
    players     = df_history['player_id'].unique()

    for pid in tqdm(players, desc="Moyennes glissantes"):
        df_pid = df_history[df_history['player_id'] == pid].copy()

        if len(df_pid) < 2:
            continue

        row = {'player_id': df_pid['player_id'].values,
               'tourney_date': df_pid['tourney_date'].values,
               'match_index': df_pid.index.values}

        # Win rate glissant global
        for w in WINDOWS:
            row[f'winrate_{w}'] = (df_pid['won']
                                   .shift(1)
                                   .rolling(w, min_periods=1)
                                   .mean()
                                   .values)

        # Win rate pondéré par qualité adversaire
        # Formule : sum(won * w) / sum(w), avec w = 1/log2(opponent_rank + 2)
        # Résultat dans [0, 1] : vrai win rate pondéré par la qualité des adversaires
        opp_weight = 1.0 / np.log2(
            df_pid['opponent_rank'].fillna(RANK_FALLBACK) + 2
        )
        win_quality = df_pid['won'] * opp_weight

        for w in WINDOWS:
            num = (win_quality
                   .shift(1)
                   .rolling(w, min_periods=MIN_QUALITY_PERIODS)
                   .sum())
            den = (opp_weight
                   .shift(1)
                   .rolling(w, min_periods=MIN_QUALITY_PERIODS)
                   .sum())
            row[f'winrate_quality_{w}'] = (num / den).values

        # Stats de service glissantes
        for stat in ROLLING_STATS:
            if stat not in df_pid.columns:
                continue
            for w in WINDOWS:
                row[f'{stat}_roll{w}'] = (df_pid[stat]
                                          .shift(1)
                                          .rolling(w, min_periods=1)
                                          .mean()
                                          .values)

        # Win rate par surface
        for surface in ['Hard', 'Clay', 'Grass']:
            surf_won = df_pid['won'].where(df_pid['surface'] == surface)
            row[f'winrate_surf_{surface}'] = (surf_won
                                              .shift(1)
                                              .rolling(10, min_periods=1)
                                              .mean()
                                              .values)

        # Streak
        streak_vals = np.zeros(len(df_pid))
        current     = 0
        for i, won in enumerate(df_pid['won'].values):
            streak_vals[i] = current
            if won:
                current = current + 1 if current >= 0 else 1
            else:
                current = current - 1 if current <= 0 else -1
        row['streak'] = streak_vals

        all_results.append(pd.DataFrame(row))

    df_rolled = pd.concat(all_results, ignore_index=True)

    # Fusionner avec df_history via l'index original
    df_history = df_history.reset_index(drop=True)
    df_history.index.name = 'match_index'
    df_rolled = df_rolled.set_index('match_index')

    rolling_cols = [c for c in df_rolled.columns
                    if c not in ['player_id', 'tourney_date']]

    for col in rolling_cols:
        df_history[col] = df_rolled[col]

    n_cols = len(rolling_cols)
    print(f"✅ Features glissantes calculées : {n_cols} colonnes")
    return df_history


def join_rolling_to_ml(df_ml: pd.DataFrame,
                        df_history: pd.DataFrame) -> pd.DataFrame:
    """
    Joint les features glissantes sur le dataset ML.
    Dédoublonne AVANT la jointure pour éviter l'explosion de lignes.
    """
    rolling_cols = [c for c in df_history.columns
                    if any(x in c for x in ['winrate', 'roll', 'streak'])]

    df_hist_sub = df_history[['player_id', 'tourney_date'] + rolling_cols].copy()

    # ── CORRECTION CLÉ : dédoublonnage avant jointure ──────────────────────
    # Si un joueur a joué 2 matchs le même jour, on garde le dernier
    df_hist_sub = (df_hist_sub
                   .sort_values('tourney_date')
                   .drop_duplicates(subset=['player_id', 'tourney_date'], keep='last'))

    print(f"  df_hist_sub après dédoublonnage : {len(df_hist_sub):,} lignes")

    # Jointure p1
    df_p1_roll = df_hist_sub.rename(columns={
        'player_id': 'p1_id',
        **{c: f'p1_{c}' for c in rolling_cols}
    })
    df_ml = df_ml.merge(df_p1_roll, on=['p1_id', 'tourney_date'], how='left')
    print(f"  Après jointure p1 : {len(df_ml):,} lignes")

    # Jointure p2
    df_p2_roll = df_hist_sub.rename(columns={
        'player_id': 'p2_id',
        **{c: f'p2_{c}' for c in rolling_cols}
    })
    df_ml = df_ml.merge(df_p2_roll, on=['p2_id', 'tourney_date'], how='left')
    print(f"  Après jointure p2 : {len(df_ml):,} lignes")

    # Différences p1 - p2
    for w in WINDOWS:
        df_ml[f'winrate_diff_{w}'] = (df_ml[f'p1_winrate_{w}']
                                      - df_ml[f'p2_winrate_{w}'])

    df_ml['streak_diff'] = df_ml['p1_streak'] - df_ml['p2_streak']

    # Différences win rate qualité adversaire p1 - p2
    for w in WINDOWS:
        col_p1 = f'p1_winrate_quality_{w}'
        col_p2 = f'p2_winrate_quality_{w}'
        if col_p1 in df_ml.columns and col_p2 in df_ml.columns:
            df_ml[f'winrate_quality_diff_{w}'] = df_ml[col_p1] - df_ml[col_p2]

    for surface in ['Hard', 'Clay', 'Grass']:
        df_ml[f'winrate_surf_diff_{surface}'] = (
            df_ml[f'p1_winrate_surf_{surface}']
            - df_ml[f'p2_winrate_surf_{surface}']
        )

    return df_ml


def audit_rolling(df_ml: pd.DataFrame) -> None:
    print("\n" + "=" * 55)
    print("AUDIT FEATURES GLISSANTES")
    print("=" * 55)

    roll_cols = [c for c in df_ml.columns if 'winrate' in c or 'roll' in c or 'streak' in c]
    print(f"\n  Nouvelles colonnes : {len(roll_cols)}")

    # Couverture
    print("\n  Couverture (% non-NaN) pour features clés :")
    key_cols = ['p1_winrate_5', 'p1_winrate_10', 'p1_winrate_20',
                'p1_streak', 'p1_winrate_surf_Hard']
    for c in key_cols:
        if c in df_ml.columns:
            pct = df_ml[c].notna().mean()
            print(f"    {c:<30}: {pct:.1%}")

    # Sanity check : winrate moyen doit être ~50%
    for w in WINDOWS:
        col = f'p1_winrate_{w}'
        if col in df_ml.columns:
            mean = df_ml[col].mean()
            print(f"\n  {col} moyenne : {mean:.3f} (attendu ~0.500)")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calcul des features glissantes (win rates, streaks, stats service).")
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
    print(f"CALCUL DES FEATURES GLISSANTES — {tour.upper()}")
    print("=" * 55)

    df_ml = pd.read_parquet(processed_dir / "matches_with_elo.parquet")
    print(f"\n📂 Dataset ML : {len(df_ml):,} matchs, {df_ml.shape[1]} colonnes")

    # Construction historique joueur
    print("\n── Historique joueur ────────────────────────────────")
    df_history = build_player_match_history(df_ml)

    # Calcul des moyennes glissantes
    print("\n── Moyennes glissantes ──────────────────────────────")
    df_history = compute_rolling_stats(df_history)

    # Jointure sur le dataset ML
    print("\n── Jointure sur dataset ML ──────────────────────────")
    df_ml = join_rolling_to_ml(df_ml, df_history)

    # Audit
    audit_rolling(df_ml)

    # Sauvegarde
    df_ml.to_parquet(processed_dir / "matches_with_features.parquet", index=False)
    print(f"\n💾 Sauvegardé : matches_with_features.parquet")
    print(f"   Shape finale : {df_ml.shape}")
