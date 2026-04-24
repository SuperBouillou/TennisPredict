# update_database.py
# Met à jour la base joueurs avec les données récentes de tennis-data.co.uk
# Usage :
#   python src/update_database.py --tour atp
#   python src/update_database.py --tour wta

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings('ignore')

import argparse
import re
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from datetime import datetime

from config import get_tour_config, get_paths, make_dirs
from espn_client import fetch_recent

# ─────────────────────────────────────────────────────────────────────────────
# TÉLÉCHARGEMENT DONNÉES FRAÎCHES
# ─────────────────────────────────────────────────────────────────────────────

def refresh_source_files(years: list, cfg: dict, odds_dir: Path) -> None:
    """
    Force le re-téléchargement des fichiers tennis-data.co.uk pour les années
    spécifiées (supprime le fichier local avant de télécharger).
    """
    for year in years:
        # Supprimer les fichiers existants pour forcer le refresh
        prefix = cfg['odds_filename'](year).rsplit('.', 1)[0]  # e.g. 'atp_2025' or 'wta_2025'
        for ext in ['xlsx', 'csv', 'xls']:
            old = odds_dir / f"{prefix}.{ext}"
            if old.exists():
                old.unlink()

        downloaded = False
        base_url = cfg['odds_url'](year).rsplit('.', 1)[0]
        for ext in ['xlsx', 'csv', 'xls']:
            url = f"{base_url}.{ext}"
            try:
                r = requests.get(url, timeout=20)
                if r.status_code == 200 and len(r.content) > 1000:
                    dest = odds_dir / f"{prefix}.{ext}"
                    dest.write_bytes(r.content)
                    print(f"  Telecharge {year}.{ext} ({len(r.content)/1024:.0f} KB)")
                    downloaded = True
                    break
            except Exception:
                continue

        if not downloaded:
            print(f"  {year} : source indisponible (donnees locales conservees si existantes)")


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES NOUVELLES DONNÉES
# ─────────────────────────────────────────────────────────────────────────────

def load_new_matches(years: list, cfg: dict, odds_dir: Path) -> pd.DataFrame:
    """
    Charge les matchs depuis les fichiers tennis-data.co.uk.
    Format : Winner, Loser, WRank, LRank, Surface, Date, Tournament...
    """
    dfs = []
    for year in years:
        path = odds_dir / cfg['odds_filename'](year)
        if not path.exists():
            print(f"  ⚠️  {year} non trouvé dans {odds_dir}")
            continue

        df = pd.read_excel(path, engine='openpyxl')
        df['year'] = year
        dfs.append(df)
        print(f"  ✅ {year} : {len(df):,} matchs chargés")

    if not dfs:
        return pd.DataFrame()

    df_all = pd.concat(dfs, ignore_index=True)

    # Nettoyage
    df_all['Date'] = pd.to_datetime(df_all['Date'], errors='coerce')
    df_all = df_all.dropna(subset=['Date', 'Winner', 'Loser'])
    df_all = df_all[df_all['Comment'] == 'Completed'].copy()

    # Colonnes numériques
    for col in ['WRank', 'LRank', 'WPts', 'LPts', 'B365W', 'B365L', 'PSW', 'PSL']:
        if col in df_all.columns:
            df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

    print(f"\n  Total : {len(df_all):,} matchs | "
          f"{df_all['Date'].min().date()} → {df_all['Date'].max().date()}")

    return df_all


def convert_to_sackmann_format(df_new: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Convertit le format tennis-data → format Sackmann pour réutiliser
    nos fonctions de calcul ELO et rolling features.
    """
    # Mapping surface
    surface_map = {'Hard': 'Hard', 'Clay': 'Clay', 'Grass': 'Grass',
                   'Indoor Hard': 'Hard', 'Carpet': 'Carpet'}

    # Mapping niveau tournoi depuis cfg (ATP: 'Series', WTA: 'Series' or 'Tier')
    level_map = cfg['td_level_map']

    df = pd.DataFrame()
    df['tourney_date']     = df_new['Date']
    df['tourney_name']     = df_new['Tournament']
    df['surface']          = df_new['Surface'].map(surface_map).fillna('Hard')
    series_col = 'Series' if 'Series' in df_new.columns else 'Tier'
    df['tourney_level']    = df_new[series_col].map(level_map).fillna('A') if series_col in df_new.columns else 'A'
    df['round']            = df_new['Round']
    df['best_of']          = df_new['Best of'].fillna(3).astype(int)

    # Joueurs — on n'a que les noms, pas les IDs
    # On va créer des IDs synthétiques basés sur le nom
    df['winner_name']      = df_new['Winner'].str.strip()
    df['loser_name']       = df_new['Loser'].str.strip()
    df['winner_rank']      = df_new['WRank']
    df['loser_rank']       = df_new['LRank']
    df['winner_rank_pts']  = df_new.get('WPts', np.nan)
    df['loser_rank_pts']   = df_new.get('LPts', np.nan)

    # Score sets (pour le format)
    for i, (wc, lc) in enumerate([('W1','L1'),('W2','L2'),('W3','L3'),
                                   ('W4','L4'),('W5','L5')], 1):
        if wc in df_new.columns:
            df[f'w{i}'] = pd.to_numeric(df_new[wc], errors='coerce')
            df[f'l{i}'] = pd.to_numeric(df_new[lc], errors='coerce')

    # Cotes
    for col in ['B365W','B365L','PSW','PSL','MaxW','MaxL','AvgW','AvgL']:
        if col in df_new.columns:
            df[col] = df_new[col]

    df['source'] = 'tennis-data'

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# ELO DECAY
# ─────────────────────────────────────────────────────────────────────────────

# Taux de déclin annuel : l'ELO d'un joueur inactif revient vers 1500
# à raison de 10% par an (ex: ELO 2000 → 1950 après 1 an sans match)
ELO_DECAY_PER_YEAR = 0.10

def apply_elo_decay(elo_ratings: dict, elo_surface: dict,
                    active_players: set, reference_date: datetime) -> tuple:
    """
    Applique un decay temporel aux joueurs absents de la période récente.
    Le decay est proportionnel au temps écoulé depuis la fin des données
    de base (fin 2024), estimé à partir de reference_date.
    """
    # Les données Sackmann de base couvrent jusqu'à fin 2024
    base_cutoff = datetime(2024, 12, 31)
    years_elapsed = max(0, (reference_date - base_cutoff).days / 365.25)

    if years_elapsed < 0.1:
        return elo_ratings, elo_surface

    decay_factor = (1 - ELO_DECAY_PER_YEAR) ** years_elapsed
    decayed = 0

    for name in list(elo_ratings.keys()):
        if name in active_players:
            continue  # joueur actif en 2025-2026 : pas de decay
        old_elo = elo_ratings[name]
        elo_ratings[name] = 1500 + (old_elo - 1500) * decay_factor

        if name in elo_surface:
            for surf in elo_surface[name]:
                old_s = elo_surface[name][surf]
                elo_surface[name][surf] = 1500 + (old_s - 1500) * decay_factor

        decayed += 1

    print(f"  Decay {years_elapsed:.1f} an(s) applique a {decayed} joueurs inactifs "
          f"(facteur {decay_factor:.3f})")
    return elo_ratings, elo_surface


# ─────────────────────────────────────────────────────────────────────────────
# MISE À JOUR ELO
# ─────────────────────────────────────────────────────────────────────────────

def elo_expected(ra: float, rb: float) -> float:
    return 1 / (1 + 10 ** ((rb - ra) / 400))


def update_elo(df_new: pd.DataFrame, elo_ratings: dict,
               elo_surface: dict, cfg: dict) -> tuple:
    """
    Met à jour les ELO existants avec les nouveaux matchs.
    elo_ratings  : dict player_name → elo_global
    elo_surface  : dict player_name → {Hard, Clay, Grass, Carpet}
    """
    K_MAP = cfg['k_factor_map']

    df_sorted = df_new.sort_values('tourney_date').reset_index(drop=True)
    updated   = 0

    for _, row in df_sorted.iterrows():
        w = str(row['winner_name']).strip()
        l = str(row['loser_name']).strip()
        surf  = str(row.get('surface', 'Hard'))
        level = str(row.get('tourney_level', 'A'))

        K = K_MAP.get(level, 32)

        # ELO global
        elo_w = elo_ratings.get(w, 1500.0)
        elo_l = elo_ratings.get(l, 1500.0)
        exp_w = elo_expected(elo_w, elo_l)

        elo_ratings[w] = elo_w + K * (1 - exp_w)
        elo_ratings[l] = elo_l + K * (0 - (1 - exp_w))

        # ELO surface
        if w not in elo_surface:
            elo_surface[w] = {'Hard':1500,'Clay':1500,'Grass':1500,'Carpet':1500}
        if l not in elo_surface:
            elo_surface[l] = {'Hard':1500,'Clay':1500,'Grass':1500,'Carpet':1500}

        if surf in elo_surface[w]:
            elo_ws = elo_surface[w][surf]
            elo_ls = elo_surface[l][surf]
            exp_ws = elo_expected(elo_ws, elo_ls)
            elo_surface[w][surf] = elo_ws + K * (1 - exp_ws)
            elo_surface[l][surf] = elo_ls + K * (0 - (1 - exp_ws))

        updated += 1

    print(f"  ✅ ELO mis à jour : {updated:,} matchs traités")
    return elo_ratings, elo_surface


# ─────────────────────────────────────────────────────────────────────────────
# MISE À JOUR ROLLING FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def update_player_profiles(df_new: pd.DataFrame,
                           elo_ratings: dict,
                           elo_surface: dict,
                           name_mapping: dict = None) -> pd.DataFrame:
    """
    Recalcule le profil récent de chaque joueur depuis les nouveaux matchs.
    Retourne un DataFrame avec une ligne par joueur.
    """
    df = df_new.sort_values('tourney_date').copy()
    df['tourney_date'] = pd.to_datetime(df['tourney_date'])

    today = df['tourney_date'].max()

    profiles = {}

    # Pour chaque joueur, calculer ses stats depuis les matchs récents
    all_players = set(df['winner_name'].tolist() + df['loser_name'].tolist())

    for player in all_players:
        # Matchs du joueur (en tant que winner ou loser)
        w_mask = df['winner_name'] == player
        l_mask = df['loser_name']  == player

        # Set score columns available in tennis-data (w1..w5, l1..l5)
        set_cols = [c for c in [f'{p}{i}' for p in ['w','l'] for i in range(1,6)]
                    if c in df.columns]

        df_w = df[w_mask][['tourney_date','surface','tourney_level','tourney_name'] + set_cols].copy()
        df_w['won'] = 1
        df_w['_is_winner'] = True

        df_l = df[l_mask][['tourney_date','surface','tourney_level','tourney_name'] + set_cols].copy()
        df_l['won'] = 0
        df_l['_is_winner'] = False

        df_p = pd.concat([df_w, df_l]).sort_values('tourney_date').reset_index(drop=True)

        # Compute sets per match (vectorized)
        n_sets_arr   = np.zeros(len(df_p), dtype=np.int16)
        sets_won_arr = np.zeros(len(df_p), dtype=np.int16)
        has_tb_arr   = np.zeros(len(df_p), dtype=bool)
        is_win = df_p['_is_winner'].values
        for i in range(1, 6):
            wc, lc = f'w{i}', f'l{i}'
            if wc not in df_p.columns or lc not in df_p.columns:
                break
            wv = pd.to_numeric(df_p[wc], errors='coerce').values
            lv = pd.to_numeric(df_p[lc], errors='coerce').values
            valid = ~(np.isnan(wv) | np.isnan(lv))
            n_sets_arr   += valid.astype(np.int16)
            player_won_set = np.where(is_win, wv > lv, lv > wv)
            sets_won_arr += (valid & player_won_set).astype(np.int16)
            tb = valid & (((wv == 7) & (lv == 6)) | ((wv == 6) & (lv == 7)))
            has_tb_arr   |= tb
        df_p['_n_sets']   = n_sets_arr
        df_p['_sets_won'] = sets_won_arr
        df_p['_has_tb']   = has_tb_arr

        if len(df_p) == 0:
            continue

        last_match = df_p['tourney_date'].max()
        days_since = (today - last_match).days

        # Winrates glissants + forme réelle
        results = df_p['won'].values
        n = len(results)
        wr5  = results[-5:].mean()  if n >= 1 else 0.5
        wr10 = results[-10:].mean() if n >= 1 else 0.5
        wr20 = results[-20:].mean() if n >= 1 else 0.5
        form_last5 = ','.join(['W' if r == 1 else 'L' for r in results[-5:]])

        # Streak courant
        streak = 0
        for r in reversed(results):
            if r == 1:
                if streak >= 0: streak += 1
                else: break
            else:
                if streak <= 0: streak -= 1
                else: break

        # Winrate par surface
        wr_surf = {}
        for surf in ['Hard', 'Clay', 'Grass']:
            surf_results = df_p[df_p['surface'] == surf]['won'].values
            wr_surf[surf] = surf_results.mean() if len(surf_results) > 0 else 0.5

        # Fatigue (matches + sets in last 7/14/21 days)
        profiles.setdefault(player, {})
        for d in [7, 14, 21]:
            cutoff = today - pd.Timedelta(days=d)
            recent = df_p[df_p['tourney_date'] >= cutoff]
            profiles[player][f'matches_{d}d'] = int(len(recent))
        for d in [7, 14]:
            cutoff = today - pd.Timedelta(days=d)
            recent = df_p[df_p['tourney_date'] >= cutoff]
            profiles[player][f'sets_{d}d'] = int(recent['_n_sets'].sum())

        # Sets ratio over last 10 matches
        last10 = df_p.tail(10)
        tot_sets = int(last10['_n_sets'].sum())
        won_sets = int(last10['_sets_won'].sum())
        sets_ratio_10 = won_sets / tot_sets if tot_sets > 0 else 0.5

        # Tiebreak win rate over last 10 matches that contained a tiebreak
        last10_tb = last10[last10['_has_tb']]
        tiebreak_winrate_10 = (last10_tb['won'].mean()
                               if len(last10_tb) >= 2 else 0.5)

        # Tournament win rate at each seen tournament (for predict_today lookup)
        tourney_wr = {}
        for tname, tgrp in df_p.groupby('tourney_name'):
            if not isinstance(tname, str) or not tname.strip():
                continue
            tourney_wr[tname.strip().lower()] = float(tgrp['won'].mean())

        # Rang depuis le dernier match (victoire OU défaite, selon le plus récent)
        # On filtre sur les matchs ayant une donnée de rang (les matchs ESPN n'en ont pas)
        last_w = df[w_mask & df['winner_rank'].notna()].sort_values('tourney_date')
        last_l = df[l_mask & df['loser_rank'].notna()].sort_values('tourney_date')

        rank = np.nan
        rank_pts = np.nan
        w_date = last_w.iloc[-1]['tourney_date'] if len(last_w) > 0 else pd.NaT
        l_date = last_l.iloc[-1]['tourney_date'] if len(last_l) > 0 else pd.NaT
        use_win = (pd.notna(w_date) and (pd.isna(l_date) or w_date >= l_date))
        if use_win:
            rank     = last_w.iloc[-1].get('winner_rank', np.nan)
            rank_pts = last_w.iloc[-1].get('winner_rank_pts', np.nan)
        elif len(last_l) > 0:
            rank     = last_l.iloc[-1].get('loser_rank', np.nan)
            rank_pts = last_l.iloc[-1].get('loser_rank_pts', np.nan)

        # Résoudre le nom Sackmann pour le lookup ELO
        elo_key = name_mapping.get(player, player) if name_mapping else player

        profiles[player].update({
            'player_name'           : player,
            'last_match'            : last_match.date(),
            'days_since'            : days_since,
            'n_matches'             : n,
            'winrate_5'             : round(wr5, 4),
            'winrate_10'            : round(wr10, 4),
            'winrate_20'            : round(wr20, 4),
            'streak'                : streak,
            'winrate_surf_Hard'     : round(wr_surf.get('Hard', 0.5), 4),
            'winrate_surf_Clay'     : round(wr_surf.get('Clay', 0.5), 4),
            'winrate_surf_Grass'    : round(wr_surf.get('Grass', 0.5), 4),
            'rank'                  : rank,
            'rank_points'           : rank_pts,
            'elo'                   : elo_ratings.get(elo_key, 1500),
            'elo_Hard'              : elo_surface.get(elo_key, {}).get('Hard', 1500),
            'elo_Clay'              : elo_surface.get(elo_key, {}).get('Clay', 1500),
            'elo_Grass'             : elo_surface.get(elo_key, {}).get('Grass', 1500),
            'form_last5'            : form_last5,
            # New momentum / fatigue features
            'sets_ratio_10'         : round(sets_ratio_10, 4),
            'tiebreak_winrate_10'   : round(tiebreak_winrate_10, 4),
            'tourney_winrates'      : tourney_wr,   # dict {tourney_lower: wr}
        })

    df_profiles = pd.DataFrame(list(profiles.values()))

    # Clés de recherche
    df_profiles['last_name']  = df_profiles['player_name'].str.split().str[-1].str.lower()
    df_profiles['first_init'] = df_profiles['player_name'].str.split().str[0].str[0].str.lower()
    df_profiles['name_key']   = df_profiles['player_name'].str.lower().str.strip()

    print(f"  ✅ Profils mis à jour : {len(df_profiles):,} joueurs")
    return df_profiles


# ── Mapping noms tennis-data → noms Sackmann ─────────────────────────────

_INITIALS_RE = re.compile(r'^[A-Za-z](\.[A-Za-z])*\.?$')

def _is_initials(s: str) -> bool:
    """Détecte un token d'initiales : J. J.L. T.A. J.M. etc."""
    return bool(_INITIALS_RE.match(s))

def _norm(s: str) -> str:
    """Normalise un nom : minuscules, sans tirets/apostrophes/points."""
    return s.lower().replace('-', ' ').replace("'", '').replace('.', '').strip()

def _norm_nospace(s: str) -> str:
    """Normalise sans espaces (pour O'Connell → oconnell)."""
    return _norm(s).replace(' ', '')


def build_name_mapping(df_new: pd.DataFrame, elo_ratings: dict) -> dict:
    """
    Construit un dict : nom tennis-data ("Sinner J.") → nom Sackmann ("Jannik Sinner").
    Stratégies par ordre de priorité :
      1. Clé exacte normalisée   : (tout-après-prénom, initiale)
      2. Dernier mot             : (dernier-mot-nom, initiale)    ← noms composés
      3. Premier mot du nom      : (premier-mot-nom, initiale)    ← ex. Mpetshi Perricard
      4. Sans espaces/ponctuation: évite les problèmes O'Connell / Oconnell
      5. N'importe quel mot      : (mot-du-nom, initiale) + score de chevauchement
    """
    # ── Construire les indices Sackmann ──────────────────────────────────────
    idx_full    = {}   # (' '.join(parts[1:]) normalisé, fi) → name
    idx_last    = {}   # (dernier_mot, fi) → name
    idx_nospace = {}   # (nom_sans_espaces, fi) → name
    idx_any     = {}   # (chaque_mot_après_premier, fi) → [names]

    for name in elo_ratings:
        parts = name.strip().split()
        if len(parts) < 2:
            continue
        if _is_initials(parts[-1]):
            continue  # déjà format tennis-data, ignorer
        fi        = parts[0][0].lower()
        norm_rest = _norm(' '.join(parts[1:]))
        norm_words = norm_rest.split()

        idx_full.setdefault((norm_rest, fi), name)
        idx_last.setdefault((norm_words[-1], fi), name)
        idx_nospace.setdefault((_norm_nospace(' '.join(parts[1:])), fi), name)
        for w in norm_words:
            if len(w) > 2:
                idx_any.setdefault((w, fi), []).append(name)

    # ── Mapper chaque nom tennis-data ────────────────────────────────────────
    td_names = set(df_new['winner_name'].tolist() + df_new['loser_name'].tolist())
    mapping  = {}
    matched  = 0

    for td_name in td_names:
        parts = td_name.strip().split()
        # Seuls les noms au format tennis-data "Nom Initiale(s)." sont à mapper
        if not (len(parts) >= 2 and _is_initials(parts[-1])):
            mapping[td_name] = td_name
            continue

        fi        = parts[-1][0].lower()
        raw_last  = ' '.join(parts[:-1])
        norm_last = _norm(raw_last)
        norm_words = norm_last.split()

        sackmann = (
            # 1. Clé exacte normalisée
            idx_full.get((norm_last, fi))
            # 2. Dernier mot du nom comme clé
            or idx_last.get((norm_words[-1], fi))
            # 3. Premier mot du nom comme clé (nom composé type "Mpetshi Perricard")
            or idx_last.get((norm_words[0], fi))
            # 4. Sans espaces (O Connell → oconnell)
            or idx_nospace.get((_norm_nospace(raw_last), fi))
        )

        # 5. N'importe quel mot + score de chevauchement
        if not sackmann:
            best_score, best_cand = 0, None
            for w in norm_words:
                if len(w) <= 2:
                    continue
                for cand in idx_any.get((w, fi), []):
                    score = sum(1 for p in norm_words if p in _norm(cand))
                    if score > best_score:
                        best_score, best_cand = score, cand
            if best_score > 0:
                sackmann = best_cand

        if sackmann:
            mapping[td_name] = sackmann
            matched += 1
        else:
            mapping[td_name] = td_name

    print(f"  Mapping noms : {matched}/{len(td_names)} joueurs relies a la base Sackmann")
    return mapping

# ─────────────────────────────────────────────────────────────────────────────
# MULTI-SOURCE DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_last(name: str) -> str:
    """
    Extracts a normalised last name from any source format:
      - tennis-data format  : "Sinner J."           → "sinner"
      - ESPN format         : "Jannik Sinner"        → "sinner"
      - Compound last name  : "Mpetshi Perricard G." → "perricard"
    Returns "" for empty/unrecognisable input.
    Reuses _is_initials() and _norm() already defined in this module.
    """
    if not name or not name.strip():
        return ""
    parts = name.strip().split()
    if len(parts) == 1:
        return _norm(parts[0])

    # tennis-data format: last token is initials (e.g. "J.", "J.L.")
    if _is_initials(parts[-1]):
        non_init = [p for p in parts[:-1] if not _is_initials(p)]
        if not non_init:
            return ""
        return _norm(non_init[-1])

    # ESPN / Sackmann format: last token is the surname
    return _norm(parts[-1])


def _make_dedup_key(tourney_date, winner: str, loser: str):
    """
    Returns a hashable dedup key: (date, frozenset({last_winner, last_loser})).
    Returns None if date is invalid or either name is empty.
    """
    try:
        ts = pd.Timestamp(tourney_date)
        if pd.isna(ts):
            return None
        d = ts.date()
    except Exception:
        return None
    w_last = _extract_last(winner)
    l_last = _extract_last(loser)
    if not w_last or not l_last:
        return None
    if w_last == l_last:
        return None
    return (d, frozenset([w_last, l_last]))


def fetch_espn_complement(df_td: pd.DataFrame, tour: str, days: int = 21) -> pd.DataFrame:
    """
    Fetches ESPN completed matches for the last `days` days and returns
    only matches NOT already present in df_td (deduplicated by date + last names).

    ESPN names ("Jannik Sinner") and tennis-data names ("Sinner J.") are compared
    via _extract_last() so cross-format duplicates are correctly detected.

    Returns an empty DataFrame if ESPN is unavailable or all matches are duplicates.
    Matches where both players share the same last name are skipped by _make_dedup_key
    (returns None) and are conservatively kept (not treated as duplicates).
    """
    print(f"\n── Source 3 : ESPN (derniers {days} jours) ──────────────")
    df_espn = fetch_recent(tour, days=days)
    if df_espn.empty:
        print("  ESPN : aucun match retourné")
        return pd.DataFrame()

    # Build dedup set from tennis-data (or empty set if df_td is empty)
    td_keys: set = set()
    if not df_td.empty:
        for _, row in df_td.iterrows():
            key = _make_dedup_key(row["tourney_date"], row["winner_name"], row["loser_name"])
            if key is not None:
                td_keys.add(key)

    # Keep only ESPN rows absent from tennis-data.
    # Rows where _make_dedup_key returns None (same last name / ambiguous) are kept
    # conservatively to avoid false suppression.
    new_rows = []
    for _, row in df_espn.iterrows():
        key = _make_dedup_key(row["tourney_date"], row["winner_name"], row["loser_name"])
        if key is None or key not in td_keys:
            new_rows.append(row)

    if not new_rows:
        print("  ESPN : 0 nouveau match (tous deja dans tennis-data)")
        return pd.DataFrame()

    df_complement = pd.DataFrame(new_rows).reset_index(drop=True)
    print(f"  ESPN : {len(df_complement)} nouveaux matchs "
          f"({df_complement['tourney_date'].min().date()} "
          f"→ {df_complement['tourney_date'].max().date()})")
    return df_complement


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Mise à jour base joueurs")
    parser.add_argument('--tour', default='atp', choices=['atp', 'wta'],
                        help="Tour à traiter : atp ou wta (défaut: atp)")
    args = parser.parse_args()

    tour  = args.tour.lower()
    cfg   = get_tour_config(tour)
    paths = get_paths(tour)
    make_dirs(tour)

    PROCESSED_DIR = paths['processed_dir']
    ODDS_DIR      = paths['odds_dir']

    print("=" * 55)
    print(f"MISE À JOUR BASE JOUEURS — {tour.upper()} — 2025/2026")
    print("=" * 55)

    # ── Charger les anciens ELO ───────────────────────────────────────────────
    print("\n── Chargement ELO existants ─────────────────────────")
    elo_path = PROCESSED_DIR / "elo_ratings_final.parquet"

    if elo_path.exists():
        df_elo_old = pd.read_parquet(elo_path)
        print(f"  Base ELO : {len(df_elo_old):,} joueurs")

        # Construire les dicts
        elo_ratings = {}
        elo_surface = {}

        for _, row in df_elo_old.iterrows():
            name = str(row['full_name']).strip()
            elo_ratings[name] = float(row.get('elo_global', 1500))
            elo_surface[name] = {
                'Hard'  : float(row.get('elo_Hard',   1500)),
                'Clay'  : float(row.get('elo_Clay',   1500)),
                'Grass' : float(row.get('elo_Grass',  1500)),
                'Carpet': float(row.get('elo_Carpet', 1500)),
            }
    else:
        print("  ⚠️  ELO non trouvé, initialisation à 1500")
        elo_ratings = {}
        elo_surface = {}

    # ── Téléchargement données fraîches ───────────────────────────────────────
    print("\n── Téléchargement données fraîches ──────────────────")
    current_year = datetime.now().year
    refresh_source_files([current_year - 1, current_year], cfg, ODDS_DIR)

    # ── Charger les nouvelles données ─────────────────────────────────────────
    print("\n── Chargement nouvelles données ─────────────────────")
    df_raw = load_new_matches([current_year - 1, current_year], cfg, ODDS_DIR)

    if df_raw.empty:
        print("  tennis-data : aucun fichier disponible — ESPN sera la seule source")
        df_new = pd.DataFrame()
    else:
        df_new = convert_to_sackmann_format(df_raw, cfg)

    n_td = len(df_new)
    if n_td:
        print(f"  tennis-data : {n_td:,} matchs "
              f"({df_new['tourney_date'].min().date()} "
              f"→ {df_new['tourney_date'].max().date()})")
    else:
        print("  tennis-data : aucun match chargé (WTA ou source manquante)")

    # ── Source 3 : ESPN — complément récent ──────────────────────────────────
    # ESPN est une source optionnelle : une erreur inattendue ne doit pas
    # faire perdre les données tennis-data déjà chargées.
    try:
        df_espn = fetch_espn_complement(df_new, tour, days=21)
    except Exception as exc:
        print(f"  ⚠️  ESPN indisponible ({exc}) — tennis-data seul sera utilisé")
        df_espn = pd.DataFrame()
    n_espn  = len(df_espn)

    # Fusionner les deux sources, trier chronologiquement
    if not df_espn.empty:
        df_combined = (
            pd.concat([df_new, df_espn], ignore_index=True)
            .sort_values("tourney_date")
            .reset_index(drop=True)
        )
    else:
        df_combined = df_new.copy()

    n_combined = len(df_combined)
    if df_combined.empty:
        print(f"❌ Aucune donnée trouvée. Vérifier {ODDS_DIR} et la connectivité ESPN")
        return

    print(f"\n── Couverture combinée ──────────────────────────────")
    print(f"  tennis-data : {n_td:,} matchs")
    print(f"  ESPN        : {n_espn:,} matchs nouveaux")
    print(f"  Total       : {n_combined:,} matchs")
    print(f"  Période     : {df_combined['tourney_date'].min().date()} "
          f"→ {df_combined['tourney_date'].max().date()}")

    # ── Mapping noms tennis-data → noms Sackmann ──────────────────────────────
    print("\n── Mapping noms joueurs ─────────────────────────────")
    name_mapping = build_name_mapping(df_combined, elo_ratings)

    # Pour update_elo : noms traduits en noms Sackmann
    df_elo_input = df_combined.copy()
    df_elo_input["winner_name"] = df_elo_input["winner_name"].map(name_mapping)
    df_elo_input["loser_name"]  = df_elo_input["loser_name"].map(name_mapping)

    # ── Mise à jour ELO ───────────────────────────────────────────────────────
    print("\n── Mise à jour ELO ──────────────────────────────────")
    elo_ratings, elo_surface = update_elo(df_elo_input, elo_ratings, elo_surface, cfg)

    # ── ELO decay joueurs inactifs ────────────────────────────────────────────
    print("\n── ELO decay joueurs inactifs ───────────────────────")
    active_players = set(df_elo_input['winner_name'].tolist() +
                         df_elo_input['loser_name'].tolist())
    elo_ratings, elo_surface = apply_elo_decay(
        elo_ratings, elo_surface, active_players, datetime.now()
    )

    # ── Calcul profils joueurs ────────────────────────────────────────────────
    print("\n── Calcul profils joueurs ───────────────────────────")
    # df_elo_input : noms déjà traduits en noms Sackmann via name_mapping.
    # Tous les matchs d'un même joueur partagent ainsi une clé canonique unique,
    # évitant les profils fractionnés entre tennis-data ("Sinner J.") et ESPN ("Jannik Sinner").
    df_profiles = update_player_profiles(df_elo_input, elo_ratings, elo_surface, name_mapping)

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    print("\n── Sauvegarde ───────────────────────────────────────")

    # Profils joueurs mis à jour
    profiles_path = PROCESSED_DIR / "player_profiles_updated.parquet"
    df_profiles.to_parquet(profiles_path, index=False)
    print(f"  ✅ Profils : {profiles_path.name}")

    # ELO mis à jour
    elo_updated = []
    for name, elo_g in elo_ratings.items():
        surf = elo_surface.get(name, {})
        elo_updated.append({
            'full_name'  : name,
            'elo_global' : elo_g,
            'elo_Hard'   : surf.get('Hard',   1500),
            'elo_Clay'   : surf.get('Clay',   1500),
            'elo_Grass'  : surf.get('Grass',  1500),
            'elo_Carpet' : surf.get('Carpet', 1500),
        })
    df_elo_updated = pd.DataFrame(elo_updated)
    elo_updated_path = PROCESSED_DIR / "elo_ratings_updated.parquet"
    df_elo_updated.to_parquet(elo_updated_path, index=False)
    print(f"  ✅ ELO    : {elo_updated_path.name} ({len(df_elo_updated):,} joueurs)")

    # Top 10 ELO global pour validation
    print("\n── Top 10 ELO Global (mis à jour) ───────────────────")
    top10 = df_elo_updated.nlargest(10, 'elo_global')[['full_name','elo_global','elo_Hard','elo_Clay']]
    for _, row in top10.iterrows():
        print(f"  {row['full_name']:<25} ELO={row['elo_global']:.0f} "
              f"Hard={row['elo_Hard']:.0f} Clay={row['elo_Clay']:.0f}")

    # Stats finales
    print(f"\n── Résumé ───────────────────────────────────────────")
    print(f"  Sources            : tennis-data ({n_td}) + ESPN ({n_espn})")
    print(f"  Matchs intégrés    : {n_combined:,}")
    print(f"  Joueurs mis à jour : {len(df_profiles):,}")
    print(f"  Dernière date      : {df_combined['tourney_date'].max().date()}")
    print(f"\n✅ Base joueurs à jour — relancer predict_today.py")
    print(f"\n   Note : modifier predict_today.py pour charger")
    print(f"   'player_profiles_updated.parquet' au lieu de")
    print(f"   'matches_features_final.parquet'")


if __name__ == "__main__":
    main()
