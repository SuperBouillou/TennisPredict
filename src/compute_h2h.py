# src/compute_h2h.py

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import get_tour_config, get_paths, make_dirs


_H2H_HALFLIFE_YEARS = 4.0   # demi-vie exponentielle pour la pondération temporelle
_H2H_LAMBDA = np.log(2) / _H2H_HALFLIFE_YEARS  # λ tel que poids = exp(-λ * années)


def compute_h2h(df_ml: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les stats H2H entre chaque paire de joueurs AVANT chaque match.
    Features produites :
    - h2h_p1_wins          : victoires de p1 contre p2 (historique brut)
    - h2h_total            : total matchs entre les deux
    - h2h_p1_winrate       : % victoires p1 (non pondéré)
    - h2h_surf_p1_winrate  : % victoires p1 sur cette surface
    - h2h_p1_winrate_recent: win rate pondéré par récence [NOUVEAU]
                             (demi-vie 4 ans : un match d'il y a 4 ans
                              pèse moitié moins qu'un match récent)
    """

    df_ml = df_ml.sort_values('tourney_date').reset_index(drop=True)

    p1_id = df_ml['p1_id'].values
    p2_id = df_ml['p2_id'].values

    pair_a = np.minimum(p1_id, p2_id)
    pair_b = np.maximum(p1_id, p2_id)

    df_ml['pair_key'] = [f"{a}_{b}" for a, b in zip(pair_a, pair_b)]
    df_ml['p1_is_a']  = (p1_id == pair_a)

    # Résultats bruts
    h2h_p1_wins    = np.zeros(len(df_ml))
    h2h_total      = np.zeros(len(df_ml))
    h2h_surf_wins  = np.zeros(len(df_ml))
    h2h_surf_total = np.zeros(len(df_ml))
    # Résultats pondérés récence
    h2h_recent_num = np.zeros(len(df_ml))   # somme pondérée des victoires a
    h2h_recent_den = np.zeros(len(df_ml))   # somme des poids

    # {pair_key: [wins_a, total]}
    h2h_global: dict  = {}
    # {(pair_key, surface): [wins_a, total]}
    h2h_surface: dict = {}
    # {pair_key: [(date, a_won)]}  — pour le calcul pondéré
    h2h_history: dict = {}

    dates = df_ml['tourney_date'].values  # numpy datetime64

    for i, row in tqdm(df_ml.iterrows(), total=len(df_ml), desc="Calcul H2H"):

        key      = row['pair_key']
        surface  = row['surface']
        surf_key = (key, surface)
        p1_is_a  = row['p1_is_a']
        p1_won   = row['target'] == 1

        # ── Lire l'historique AVANT ce match ────────────────────────────────
        g = h2h_global.get(key, [0, 0])
        s = h2h_surface.get(surf_key, [0, 0])

        wins_p1_global = g[0] if p1_is_a else (g[1] - g[0])
        wins_p1_surf   = s[0] if p1_is_a else (s[1] - s[0])

        h2h_p1_wins[i]    = wins_p1_global
        h2h_total[i]      = g[1]
        h2h_surf_wins[i]  = wins_p1_surf
        h2h_surf_total[i] = s[1]

        # ── H2H pondéré récence (avant ce match) ───────────────────────────
        if key in h2h_history and h2h_history[key]:
            cur_date = dates[i]
            num = 0.0
            den = 0.0
            for past_date, a_won_past in h2h_history[key]:
                years_ago = (cur_date - past_date).astype('timedelta64[D]').astype(float) / 365.25
                w = np.exp(-_H2H_LAMBDA * max(years_ago, 0.0))
                # Convertir a_won → p1_won
                p1_won_past = a_won_past if p1_is_a else not a_won_past
                num += w * float(p1_won_past)
                den += w
            h2h_recent_num[i] = num
            h2h_recent_den[i] = den

        # ── Mettre à jour après ce match ────────────────────────────────────
        a_won = p1_won if p1_is_a else not p1_won

        h2h_global[key]       = [g[0] + int(a_won), g[1] + 1]
        h2h_surface[surf_key] = [s[0] + int(a_won), s[1] + 1]
        h2h_history.setdefault(key, []).append((dates[i], a_won))

    # ── Assignation ─────────────────────────────────────────────────────────
    df_ml['h2h_p1_wins']   = h2h_p1_wins
    df_ml['h2h_total']     = h2h_total
    df_ml['h2h_p1_winrate'] = np.where(
        h2h_total > 0,
        h2h_p1_wins / h2h_total,
        0.5
    )

    df_ml['h2h_surf_p1_wins']    = h2h_surf_wins
    df_ml['h2h_surf_total']      = h2h_surf_total
    df_ml['h2h_surf_p1_winrate'] = np.where(
        h2h_surf_total > 0,
        h2h_surf_wins / h2h_surf_total,
        0.5
    )

    # H2H pondéré récence : 0.5 quand pas d'historique
    df_ml['h2h_p1_winrate_recent'] = np.where(
        h2h_recent_den > 0,
        h2h_recent_num / h2h_recent_den,
        0.5
    )

    df_ml['h2h_played'] = (h2h_total > 0).astype(int)

    df_ml = df_ml.drop(columns=['pair_key', 'p1_is_a'])

    return df_ml


def audit_h2h(df: pd.DataFrame) -> None:

    print("\n" + "=" * 55)
    print("AUDIT H2H")
    print("=" * 55)

    print(f"\n  Matchs avec H2H existant   : "
          f"{df['h2h_played'].sum():,} "
          f"({df['h2h_played'].mean():.1%})")

    print(f"  h2h_total moyenne          : {df['h2h_total'].mean():.2f}")
    print(f"  h2h_total max              : {df['h2h_total'].max():.0f}")

    print(f"\n  h2h_p1_winrate moyenne     : "
          f"{df['h2h_p1_winrate'].mean():.3f} (attendu ~0.500)")

    # Calibration : est-ce que le H2H prédit mieux que 50% ?
    has_h2h = df[df['h2h_played'] == 1]
    if len(has_h2h) > 0:
        h2h_acc = (
            ((has_h2h['h2h_p1_winrate'] > 0.5) & (has_h2h['target'] == 1)) |
            ((has_h2h['h2h_p1_winrate'] < 0.5) & (has_h2h['target'] == 0))
        ).mean()
        print(f"  Accuracy H2H (si H2H > 0) : {h2h_acc:.1%}")

    # Distribution du nombre de confrontations
    print(f"\n  Distribution h2h_total :")
    bins = [0, 1, 3, 5, 10, 20, 999]
    labels = ['0', '1-2', '3-4', '5-9', '10-19', '20+']
    df['h2h_bin'] = pd.cut(df['h2h_total'], bins=bins, labels=labels, right=False)
    print(df['h2h_bin'].value_counts().sort_index().to_string())
    df = df.drop(columns=['h2h_bin'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calcul des statistiques head-to-head.")
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
    print(f"CALCUL HEAD-TO-HEAD — {tour.upper()}")
    print("=" * 55)

    df = pd.read_parquet(processed_dir / "matches_with_features.parquet")
    print(f"\n📂 Dataset : {len(df):,} matchs, {df.shape[1]} colonnes")

    df = compute_h2h(df)

    audit_h2h(df)

    df.to_parquet(processed_dir / "matches_with_h2h.parquet", index=False)
    print(f"\n💾 Sauvegardé : matches_with_h2h.parquet")
    print(f"   Shape finale : {df.shape}")
