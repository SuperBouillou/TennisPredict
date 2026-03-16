# src/compute_glicko.py
"""
Calcul des ratings Glicko-2 pour le dataset ATP/WTA.

Glicko-2 améliore l'ELO en ajoutant :
  - phi (RD)   : incertitude sur le vrai niveau du joueur
  - sigma      : volatilité (consistance des performances)

Traitement par périodes mensuelles (standard Glicko-2).
Les ratings enregistrés pour chaque match sont ceux du DÉBUT de la période,
ce qui évite tout leakage intra-période.
"""

import argparse
import math
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import get_tour_config, get_paths, make_dirs

# ── Constantes Glicko-2 ──────────────────────────────────────────────────────
MU_INIT    = 1500.0   # Rating initial (même échelle qu'ELO)
PHI_INIT   = 350.0    # Déviation initiale (incertitude max)
SIGMA_INIT = 0.06     # Volatilité initiale
TAU        = 0.5      # Constante système (contrôle la vitesse de changement de σ)
SCALE      = 173.7178 # Facteur de conversion (400 / ln(10))
EPSILON    = 1e-6     # Tolérance convergence algorithme Illinois

SURFACE_LIST = ['Hard', 'Clay', 'Grass', 'Carpet']


# ── Fonctions Glicko-2 ───────────────────────────────────────────────────────

def _g(phi_p: float) -> float:
    """Fonction g — réduit l'impact des opposants avec grande incertitude."""
    return 1.0 / math.sqrt(1.0 + 3.0 * phi_p ** 2 / math.pi ** 2)


def _E(mu_p: float, mu_j: float, phi_j: float) -> float:
    """Score espéré de mu_p contre mu_j (échelle Glicko-2 interne)."""
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu_p - mu_j)))


def _to_g2(mu: float, phi: float):
    """Convertit de l'échelle ELO-like vers l'échelle interne Glicko-2."""
    return (mu - MU_INIT) / SCALE, phi / SCALE


def _from_g2(mu_p: float, phi_p: float):
    """Convertit de l'échelle interne Glicko-2 vers l'échelle ELO-like."""
    return MU_INIT + SCALE * mu_p, SCALE * phi_p


def _update_sigma(sigma: float, phi_p: float, delta: float, v: float) -> float:
    """
    Mise à jour de la volatilité par l'algorithme Illinois (bisection améliorée).
    Résout f(x) = 0 où x = ln(sigma'^2).
    """
    a = math.log(sigma ** 2)

    def f(x: float) -> float:
        ex = math.exp(x)
        d  = phi_p ** 2 + v + ex
        if d == 0:
            return float('inf')
        return ex * (delta ** 2 - d) / (2.0 * d ** 2) - (x - a) / TAU ** 2

    A = a
    if delta ** 2 > phi_p ** 2 + v:
        B = math.log(max(delta ** 2 - phi_p ** 2 - v, 1e-10))
    else:
        k = 1
        while f(a - k * TAU) < 0:
            k += 1
        B = a - k * TAU

    fa, fb = f(A), f(B)

    for _ in range(100):
        if abs(B - A) <= EPSILON:
            break
        C  = A + (A - B) * fa / (fb - fa)
        fc = f(C)
        if fc * fb < 0:
            A, fa = B, fb
        else:
            fa /= 2.0
        B, fb = C, fc

    return math.exp(A / 2.0)


def _update_player(mu: float, phi: float, sigma: float,
                   matches: list) -> tuple:
    """
    Met à jour les ratings d'un joueur après une période.

    matches : liste de (opp_mu, opp_phi, score, weight)
              score = 1.0 si victoire, 0.0 si défaite
    """
    mu_p, phi_p = _to_g2(mu, phi)

    if not matches:
        # Joueur inactif : seule l'incertitude augmente
        phi_star = math.sqrt(phi_p ** 2 + sigma ** 2)
        new_mu, new_phi = _from_g2(mu_p, phi_star)
        return new_mu, new_phi, sigma

    # Accumulation sur tous les matchs de la période
    v_sum     = 0.0
    delta_sum = 0.0

    for opp_mu, opp_phi, score, weight in matches:
        mu_j, phi_j = _to_g2(opp_mu, opp_phi)
        g_j  = _g(phi_j)
        E_j  = _E(mu_p, mu_j, phi_j)
        v_sum     += weight * g_j ** 2 * E_j * (1.0 - E_j)
        delta_sum += weight * g_j * (score - E_j)

    if v_sum == 0:
        phi_star = math.sqrt(phi_p ** 2 + sigma ** 2)
        new_mu, new_phi = _from_g2(mu_p, phi_star)
        return new_mu, new_phi, sigma

    v     = 1.0 / v_sum
    delta = v * delta_sum

    # Nouvelle volatilité
    new_sigma = _update_sigma(sigma, phi_p, delta, v, )

    # Nouvelle déviation
    phi_star  = math.sqrt(phi_p ** 2 + new_sigma ** 2)
    new_phi_p = 1.0 / math.sqrt(1.0 / phi_star ** 2 + 1.0 / v)

    # Nouveau rating
    new_mu_p = mu_p + new_phi_p ** 2 * delta_sum

    new_mu, new_phi = _from_g2(new_mu_p, new_phi_p)
    return new_mu, new_phi, new_sigma


def _get(ratings: dict, pid) -> tuple:
    return ratings.get(pid, (MU_INIT, PHI_INIT, SIGMA_INIT))


def glicko_win_prob(mu1: float, phi1: float, mu2: float, phi2: float) -> float:
    """Probabilité de victoire de mu1 selon Glicko-2 (tient compte de phi)."""
    mu1_p, phi1_p = _to_g2(mu1, phi1)
    mu2_p, phi2_p = _to_g2(mu2, phi2)
    combined_phi  = math.sqrt(phi1_p ** 2 + phi2_p ** 2)
    return 1.0 / (1.0 + math.exp(-_g(combined_phi) * (mu1_p - mu2_p)))


# ── Calcul principal ─────────────────────────────────────────────────────────

def compute_glicko(df: pd.DataFrame, level_weight: dict) -> tuple:
    """
    Calcule les ratings Glicko-2 global et par surface.
    Retourne le df enrichi + les dicts de ratings finaux.
    """
    df = df.copy().sort_values('tourney_date').reset_index(drop=True)
    df['_period'] = df['tourney_date'].dt.to_period('M')

    glicko_global  = {}
    glicko_surface = {s: {} for s in SURFACE_LIST}

    n = len(df)
    w_mu   = np.full(n, np.nan); l_mu   = np.full(n, np.nan)
    w_phi  = np.full(n, np.nan); l_phi  = np.full(n, np.nan)
    w_mu_s = np.full(n, np.nan); l_mu_s = np.full(n, np.nan)
    w_phi_s= np.full(n, np.nan); l_phi_s= np.full(n, np.nan)

    periods = sorted(df['_period'].unique())

    for period in tqdm(periods, desc="Glicko-2"):
        mask      = df['_period'] == period
        idx_list  = df.index[mask].tolist()

        # ── Étape 1 : enregistrer les ratings de début de période ────────────
        for i in idx_list:
            row = df.iloc[i]
            w_id, l_id = row['winner_id'], row['loser_id']
            surface    = row['surface']

            wmu, wphi, _ = _get(glicko_global, w_id)
            lmu, lphi, _ = _get(glicko_global, l_id)
            w_mu[i] = wmu;  l_mu[i] = lmu
            w_phi[i] = wphi; l_phi[i] = lphi

            if surface in glicko_surface:
                wmu_s, wphi_s, _ = _get(glicko_surface[surface], w_id)
                lmu_s, lphi_s, _ = _get(glicko_surface[surface], l_id)
                w_mu_s[i] = wmu_s;  l_mu_s[i] = lmu_s
                w_phi_s[i] = wphi_s; l_phi_s[i] = lphi_s

        # ── Étape 2 : agréger les matchs par joueur pour la période ─────────
        pm_global  = {}
        pm_surface = {s: {} for s in SURFACE_LIST}

        for i in idx_list:
            row    = df.iloc[i]
            w_id   = row['winner_id']
            l_id   = row['loser_id']
            surface = row['surface']
            weight  = level_weight.get(str(row.get('tourney_level', 'A')), 1.0)

            wmu, wphi, _ = _get(glicko_global, w_id)
            lmu, lphi, _ = _get(glicko_global, l_id)

            pm_global.setdefault(w_id, []).append((lmu, lphi, 1.0, weight))
            pm_global.setdefault(l_id, []).append((wmu, wphi, 0.0, weight))

            if surface in glicko_surface:
                wmu_s, wphi_s, _ = _get(glicko_surface[surface], w_id)
                lmu_s, lphi_s, _ = _get(glicko_surface[surface], l_id)
                pm_surface[surface].setdefault(w_id, []).append((lmu_s, lphi_s, 1.0, weight))
                pm_surface[surface].setdefault(l_id, []).append((wmu_s, wphi_s, 0.0, weight))

        # ── Étape 3 : mettre à jour les ratings ─────────────────────────────
        for pid, matches in pm_global.items():
            mu, phi, sigma = _get(glicko_global, pid)
            glicko_global[pid] = _update_player(mu, phi, sigma, matches)

        for surface in SURFACE_LIST:
            for pid, matches in pm_surface[surface].items():
                mu, phi, sigma = _get(glicko_surface[surface], pid)
                glicko_surface[surface][pid] = _update_player(mu, phi, sigma, matches)

    df['winner_glicko']          = w_mu
    df['loser_glicko']           = l_mu
    df['winner_glicko_rd']       = w_phi
    df['loser_glicko_rd']        = l_phi
    df['winner_glicko_surface']  = w_mu_s
    df['loser_glicko_surface']   = l_mu_s
    df['winner_glicko_rd_surface'] = w_phi_s
    df['loser_glicko_rd_surface']  = l_phi_s

    return df, glicko_global, glicko_surface


def join_glicko_to_ml(df_ml: pd.DataFrame, df_glicko: pd.DataFrame) -> pd.DataFrame:
    """
    Joint les features Glicko-2 au dataset ML via match_key.
    Ajoute 8 features : glicko_diff, glicko_rd_diff, glicko_win_prob_p1,
    glicko_surface_diff, glicko_rd_surface_diff, p1/p2 glicko_rd.
    """
    df_ml = df_ml.copy()

    # Reconstruire winner_id / loser_id depuis target
    df_ml['_winner_id'] = np.where(df_ml['target'] == 1, df_ml['p1_id'], df_ml['p2_id'])
    df_ml['_loser_id']  = np.where(df_ml['target'] == 1, df_ml['p2_id'], df_ml['p1_id'])
    df_ml['_match_key'] = (df_ml['tourney_id'].astype(str) + '__' +
                           df_ml['_winner_id'].astype(str)  + '__' +
                           df_ml['_loser_id'].astype(str))

    # Clé identique dans df_glicko
    df_glicko = df_glicko.copy()
    df_glicko['_match_key'] = (df_glicko['tourney_id'].astype(str) + '__' +
                               df_glicko['winner_id'].astype(str)  + '__' +
                               df_glicko['loser_id'].astype(str))

    glicko_cols = ['_match_key',
                   'winner_glicko', 'loser_glicko',
                   'winner_glicko_rd', 'loser_glicko_rd',
                   'winner_glicko_surface', 'loser_glicko_surface',
                   'winner_glicko_rd_surface', 'loser_glicko_rd_surface']

    sub = df_glicko[glicko_cols].drop_duplicates('_match_key')
    df_ml = df_ml.merge(sub, on='_match_key', how='left')

    is_p1_winner = df_ml['target'] == 1

    df_ml['p1_glicko']    = np.where(is_p1_winner, df_ml['winner_glicko'],   df_ml['loser_glicko'])
    df_ml['p2_glicko']    = np.where(is_p1_winner, df_ml['loser_glicko'],    df_ml['winner_glicko'])
    df_ml['p1_glicko_rd'] = np.where(is_p1_winner, df_ml['winner_glicko_rd'],df_ml['loser_glicko_rd'])
    df_ml['p2_glicko_rd'] = np.where(is_p1_winner, df_ml['loser_glicko_rd'], df_ml['winner_glicko_rd'])

    df_ml['p1_glicko_surface']    = np.where(is_p1_winner, df_ml['winner_glicko_surface'],    df_ml['loser_glicko_surface'])
    df_ml['p2_glicko_surface']    = np.where(is_p1_winner, df_ml['loser_glicko_surface'],     df_ml['winner_glicko_surface'])
    df_ml['p1_glicko_rd_surface'] = np.where(is_p1_winner, df_ml['winner_glicko_rd_surface'], df_ml['loser_glicko_rd_surface'])
    df_ml['p2_glicko_rd_surface'] = np.where(is_p1_winner, df_ml['loser_glicko_rd_surface'],  df_ml['winner_glicko_rd_surface'])

    # Features agrégées
    df_ml['glicko_diff']          = df_ml['p1_glicko']         - df_ml['p2_glicko']
    df_ml['glicko_rd_diff']       = df_ml['p2_glicko_rd']      - df_ml['p1_glicko_rd']   # positif = p1 plus certain
    df_ml['glicko_surface_diff']  = df_ml['p1_glicko_surface'] - df_ml['p2_glicko_surface']
    df_ml['glicko_rd_surface_diff'] = df_ml['p2_glicko_rd_surface'] - df_ml['p1_glicko_rd_surface']

    # Probabilité de victoire Glicko-2 (tient compte de phi)
    df_ml['glicko_win_prob_p1'] = df_ml.apply(
        lambda r: glicko_win_prob(r['p1_glicko'], r['p1_glicko_rd'],
                                  r['p2_glicko'], r['p2_glicko_rd'])
        if pd.notna(r['p1_glicko']) and pd.notna(r['p2_glicko']) else np.nan,
        axis=1
    )

    # Nettoyage
    drop_cols = ['_winner_id', '_loser_id', '_match_key',
                 'winner_glicko', 'loser_glicko',
                 'winner_glicko_rd', 'loser_glicko_rd',
                 'winner_glicko_surface', 'loser_glicko_surface',
                 'winner_glicko_rd_surface', 'loser_glicko_rd_surface']
    df_ml = df_ml.drop(columns=drop_cols, errors='ignore')

    return df_ml


def snapshot_glicko(glicko_global: dict, glicko_surface: dict,
                    df_players: pd.DataFrame) -> pd.DataFrame:
    """Snapshot des ratings Glicko-2 finaux par joueur."""
    df_players = df_players.copy()
    df_players['player_id'] = pd.to_numeric(df_players['player_id'], errors='coerce')

    rows = []
    for pid in glicko_global:
        mu, phi, sigma = glicko_global[pid]
        row = {'player_id': pid, 'glicko': mu, 'glicko_rd': phi, 'glicko_sigma': sigma}
        for s in SURFACE_LIST:
            mu_s, phi_s, _ = _get(glicko_surface[s], pid)
            row[f'glicko_{s}']    = mu_s
            row[f'glicko_rd_{s}'] = phi_s
        rows.append(row)

    df_snap = pd.DataFrame(rows)
    df_snap['player_id'] = pd.to_numeric(df_snap['player_id'], errors='coerce')
    df_snap = df_snap.merge(df_players[['player_id', 'full_name']], on='player_id', how='left')

    return df_snap.sort_values('glicko', ascending=False).reset_index(drop=True)


# ── Pipeline ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calcul Glicko-2 par tour")
    parser.add_argument('--tour', default='atp', choices=['atp', 'wta'],
                        help="Tour à traiter : atp ou wta (défaut: atp)")
    args = parser.parse_args()

    tour  = args.tour.lower()
    cfg   = get_tour_config(tour)
    paths = get_paths(tour)
    make_dirs(tour)

    PROCESSED_DIR = paths['processed_dir']

    # Construire LEVEL_WEIGHT depuis la config du tour
    # Les k_factor_map servent de poids relatifs — on les normalise par la valeur de base
    k_map  = cfg['k_factor_map']
    k_base = min(k_map.values())
    LEVEL_WEIGHT = {lvl: k / k_base for lvl, k in k_map.items()}

    print("=" * 55)
    print(f"CALCUL GLICKO-2 {tour.upper()}")
    print("=" * 55)

    # Dataset original winner/loser
    df_raw = pd.read_parquet(PROCESSED_DIR / "matches_consolidated.parquet")
    df_raw = df_raw[df_raw['tourney_level'] != 'D'].copy()
    df_raw = df_raw[df_raw['surface'] != 'Unknown'].copy()
    df_raw = df_raw.sort_values('tourney_date').reset_index(drop=True)
    print(f"\nDataset source : {len(df_raw):,} matchs")

    # Calcul Glicko-2
    df_raw, glicko_global, glicko_surface = compute_glicko(df_raw, LEVEL_WEIGHT)

    # Snapshot ratings finaux
    df_players = pd.read_parquet(PROCESSED_DIR / "players.parquet")
    df_snap    = snapshot_glicko(glicko_global, glicko_surface, df_players)

    print(f"\nTop 10 joueurs (Glicko-2 global) :")
    print(df_snap[['full_name', 'glicko', 'glicko_rd',
                   'glicko_Hard', 'glicko_Clay', 'glicko_Grass']].head(10).to_string(index=False))

    # Jointure sur dataset ML final
    df_ml = pd.read_parquet(PROCESSED_DIR / "matches_features_final.parquet")
    print(f"\nDataset ML     : {len(df_ml):,} matchs")

    df_ml = join_glicko_to_ml(df_ml, df_raw)

    # Audit
    pct = df_ml['glicko_diff'].notna().mean()
    print(f"\nMatchs avec Glicko-2 : {pct:.1%}")
    print(f"glicko_win_prob_p1 moyenne : {df_ml['glicko_win_prob_p1'].mean():.3f} (attendu ~0.500)")

    acc = (
        ((df_ml['glicko_diff'] > 0) & (df_ml['target'] == 1)) |
        ((df_ml['glicko_diff'] < 0) & (df_ml['target'] == 0))
    ).mean()
    print(f"Accuracy Glicko brut : {acc:.1%}")

    elo_acc = (
        ((df_ml['elo_diff'] > 0) & (df_ml['target'] == 1)) |
        ((df_ml['elo_diff'] < 0) & (df_ml['target'] == 0))
    ).mean()
    print(f"Accuracy ELO brut    : {elo_acc:.1%}  (référence)")

    # Sauvegarde
    df_ml.to_parquet(PROCESSED_DIR / "matches_features_final.parquet", index=False)
    df_snap.to_parquet(PROCESSED_DIR / "glicko_ratings_final.parquet", index=False)

    print(f"\nSauvegardes :")
    print(f"  matches_features_final.parquet (+Glicko features)")
    print(f"  glicko_ratings_final.parquet   ({len(df_snap):,} joueurs)")
    print(f"\nFeatures ajoutees : glicko_diff, glicko_rd_diff, glicko_surface_diff,")
    print(f"  glicko_rd_surface_diff, glicko_win_prob_p1, p1/p2_glicko_rd")
    print(f"\nGlicko-2 termine.")
