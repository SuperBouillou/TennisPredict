# predict_today.py — v2
# Prédictions ATP/WTA pour les matchs du jour
# Placer dans : tennis_ml/src/predict_today.py
#
# Usage :
#   python src/predict_today.py                        ← matchs du jour (saisie manuelle)
#   python src/predict_today.py --date 2025-06-01      ← date spécifique
#   python src/predict_today.py --save                 ← sauvegarde CSV
#   python src/predict_today.py --odds                 ← saisie cotes Pinnacle
#   python src/predict_today.py --tour wta             ← prédictions WTA

import argparse
import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
import requests
from datetime import date, datetime
from pathlib import Path
from config import get_tour_config, get_paths

ROOT = Path(__file__).resolve().parent.parent

# Module-level mutable globals — set in main() once the tour is known
_PROCESSED_DIR: Path = ROOT / "data" / "processed" / "atp"
_PLAYER_FILE: str = "atp_players.csv"


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT MODÈLE
# ─────────────────────────────────────────────────────────────────────────────

def load_model_artifacts(models_dir: Path):
    print("── Chargement modèle ────────────────────────────────")
    model    = joblib.load(models_dir / "xgb_tuned.pkl")
    imputer  = joblib.load(models_dir / "imputer.pkl")
    features = joblib.load(models_dir / "feature_list.pkl")
    # Préférer platt_pinnacle.pkl (calibré contre Pinnacle no-vig)
    # sinon fallback sur platt_scaler.pkl (calibré contre outcomes)
    platt = None
    for platt_name in ("platt_pinnacle.pkl", "platt_scaler.pkl"):
        p = models_dir / platt_name
        if p.exists():
            platt = joblib.load(p)
            print(f"  ✅ XGBoost tuned + {platt_name} | {len(features)} features")
            break
    if platt is None:
        print(f"  ✅ XGBoost tuned (pas de scaler) | {len(features)} features")

    # Scalers surface-spécifiques (platt_Hard.pkl, platt_Clay.pkl, platt_Grass.pkl)
    platt_surfaces = {}
    for surf in ['Hard', 'Clay', 'Grass']:
        sp = models_dir / f"platt_{surf}.pkl"
        if sp.exists():
            platt_surfaces[surf] = joblib.load(sp)
    if platt_surfaces:
        print(f"  Surface scalers : {list(platt_surfaces.keys())}")

    return model, imputer, features, platt, platt_surfaces


# ─────────────────────────────────────────────────────────────────────────────
# BASE DE DONNÉES JOUEURS
# ─────────────────────────────────────────────────────────────────────────────

def load_player_database(processed_dir: Path):
    print("── Chargement base joueurs ──────────────────────────")

    updated_path = processed_dir / "player_profiles_updated.parquet"
    legacy_path  = processed_dir / "matches_features_final.parquet"

    if updated_path.exists():
        df = pd.read_parquet(updated_path)
        print(f"  ✅ {len(df):,} joueurs | MAJ : {df['last_match'].max()}")
        # Profiles are in "First Last" format (e.g. "Alexander Zverev")
        # last_name = last word, first_init = first letter of first word
        df['name_key']   = df['player_name'].str.lower().str.strip()
        df['last_name']  = df['player_name'].str.split().str[-1].str.lower()
        df['first_init'] = df['player_name'].str.split().str[0].str[0].str.lower()
        return df

    # Fallback : ancienne base
    print("  ⚠️  Profils mis à jour non trouvés → fallback legacy")
    df = pd.read_parquet(legacy_path)
    df['tourney_date'] = pd.to_datetime(df['tourney_date'])
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=2)
    df = df[df['tourney_date'] >= cutoff].copy()

    stat_cols_p1 = [c for c in df.columns if c.startswith('p1_')
                    and c not in ['p1_id','p1_name','p1_rank','p1_rank_points']]
    stat_cols_p2 = [c for c in df.columns if c.startswith('p2_')
                    and c not in ['p2_id','p2_name','p2_rank','p2_rank_points']]

    df_p1 = (df.sort_values('tourney_date')
               .groupby('p1_name')[['p1_name','p1_rank','p1_rank_points'] + stat_cols_p1]
               .last().reset_index(drop=True))
    df_p1.columns = ['player_name','rank','rank_points'] + \
                    [c.replace('p1_','') for c in stat_cols_p1]

    df_p2 = (df.sort_values('tourney_date')
               .groupby('p2_name')[['p2_name','p2_rank','p2_rank_points'] + stat_cols_p2]
               .last().reset_index(drop=True))
    df_p2.columns = ['player_name','rank','rank_points'] + \
                    [c.replace('p2_','') for c in stat_cols_p2]

    df_all = pd.concat([df_p1, df_p2], ignore_index=True)
    df_all = df_all.sort_values('rank').drop_duplicates('player_name', keep='first')
    df_all['name_key']   = df_all['player_name'].str.lower().str.strip()
    df_all['last_name']  = df_all['player_name'].str.split().str[-1].str.lower()
    df_all['first_init'] = df_all['player_name'].str.split().str[0].str[0].str.lower()

    print(f"  ✅ {len(df_all):,} joueurs (legacy)")
    return df_all

def load_elo_ratings(processed_dir: Path):
    updated_path = processed_dir / "elo_ratings_updated.parquet"
    legacy_path  = processed_dir / "elo_ratings_final.parquet"

    path = updated_path if updated_path.exists() else legacy_path
    if not path.exists():
        print("  ⚠️  ELO non trouvé")
        return pd.DataFrame()

    df = pd.read_parquet(path)
    df['last_name']  = df['full_name'].str.split().str[-1].str.lower()
    df['first_init'] = df['full_name'].str.split().str[0].str[0].str.lower()
    print(f"  ✅ ELO : {len(df):,} joueurs ({path.name})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# RÉCUPÉRATION MATCHS DU JOUR — SOURCES FIABLES
# ─────────────────────────────────────────────────────────────────────────────

def fetch_matches(target_date: date, cfg: dict, tour: str = "atp") -> list:
    """
    Recupere automatiquement les matchs du tour (ATP ou WTA) via ESPN.
    Retourne une liste de dicts {p1_name, p2_name, tournament, surface, round, best_of}
    """
    date_str = target_date.strftime("%Y-%m-%d")
    print(f"\n── Recuperation matchs {tour.upper()} — {date_str} via ESPN ─────────")

    matches = try_espn(target_date, tour)
    if matches:
        print(f"  ESPN : {len(matches)} matchs")
        return matches

    print("  Aucun match ESPN → saisie manuelle")
    return []


def try_espn(target_date: date, tour: str) -> list:
    """Recupere les matchs programmes depuis ESPN scoreboard."""
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from espn_client import fetch_scheduled
        raw = fetch_scheduled(tour, target_date)
        # Filtrer les TBD (tirage non fait)
        return [m for m in raw if "TBD" not in m["p1_name"] and "TBD" not in m["p2_name"]]
    except Exception as e:
        print(f"  ESPN erreur : {e}")
        return []


def detect_surface(tournament_name: str) -> str:
    """Détecte la surface depuis le nom du tournoi."""
    name = tournament_name.lower()
    grass_keywords   = ['wimbledon', 'halle', 'queens', "s-hertogenbosch",
                        'eastbourne', 'mallorca', 'stuttgart', 'nottingham']
    clay_keywords    = ['roland garros', 'madrid', 'rome', 'monte-carlo',
                        'barcelona', 'hamburg', 'munich', 'marrakech',
                        'bucharest', 'geneva', 'lyon', 'houston', 'santiago',
                        'buenos aires', 'estoril', 'kitzbuhel', 'umag',
                        'bastad', 'gstaad', 'acapulco']

    if any(k in name for k in grass_keywords):
        return 'Grass'
    if any(k in name for k in clay_keywords):
        return 'Clay'
    return 'Hard'


def prompt_manual_matches() -> list:
    """Saisie manuelle des matchs."""
    print("\n  ── Saisie manuelle des matchs ──────────────────────")
    print("  Format : Djokovic N. vs Sinner J.")
    print("  Laissez vide pour terminer\n")

    tournament = input("  Tournoi [ATP] : ").strip() or "ATP"
    surface    = ""
    while surface not in ['Hard','Clay','Grass']:
        surface = input("  Surface (Hard/Clay/Grass) [Hard] : ").strip() or "Hard"
    best_of    = int(input("  Best of (3 ou 5) [3] : ").strip() or "3")

    matches = []
    i = 1
    while True:
        line = input(f"  Match {i} : ").strip()
        if not line:
            break

        # Séparateurs acceptés : " vs ", " VS ", " v ", "/"
        sep = None
        for s in [' vs ', ' VS ', ' v ', '/']:
            if s in line:
                sep = s
                break

        if sep:
            parts = [p.strip() for p in line.split(sep, 1)]
            if len(parts) == 2 and parts[0] and parts[1]:
                matches.append({
                    'p1_name'   : parts[0],
                    'p2_name'   : parts[1],
                    'tournament': tournament,
                    'surface'   : surface,
                    'round'     : '',
                    'best_of'   : best_of,
                })
                i += 1
            else:
                print("  ⚠️  Format invalide. Ex: Sinner J. vs Alcaraz C.")
        else:
            print("  ⚠️  Format invalide. Ex: Sinner J. vs Alcaraz C.")

    return matches


# ─────────────────────────────────────────────────────────────────────────────
# RECHERCHE JOUEUR
# ─────────────────────────────────────────────────────────────────────────────

def normalize(name: str) -> str:
    return str(name).lower().strip()


def find_player(name: str, df_players: pd.DataFrame, df_elo: pd.DataFrame) -> dict:
    """
    Cherche un joueur par nom. Gère :
      - "Sinner J."      → last=sinner, init=j
      - "Jannik Sinner"  → last=sinner, init=j
      - "sinner"         → last=sinner
    """
    raw = normalize(name)
    parts = raw.replace('.', '').split()

    # Déterminer last_name et first_init selon le format
    if len(parts) == 0:
        return {}

    if len(parts) == 1:
        last_name  = parts[0]
        first_init = ''
    elif len(parts[-1]) <= 2:
        # Format "Nom Init." : dernier token = initiale
        last_name  = ' '.join(parts[:-1])
        first_init = parts[-1][0]
    else:
        # Format "Prénom Nom" : premier token = prénom
        first_init = parts[0][0]
        last_name  = ' '.join(parts[1:])

    # ── Recherche dans df_players ────────────────────────────────────────────
    candidates = df_players[df_players['last_name'] == last_name]
    if len(candidates) == 1:
        return candidates.iloc[0].to_dict()
    elif len(candidates) > 1 and first_init:
        refined = candidates[candidates['first_init'] == first_init]
        if len(refined) > 0:
            return refined.iloc[0].to_dict()
        return candidates.iloc[0].to_dict()

    # Recherche partielle (préfixe)
    candidates = df_players[df_players['last_name'].str.startswith(last_name[:4])]
    if len(candidates) >= 1:
        if first_init:
            refined = candidates[candidates['first_init'] == first_init]
            if len(refined) > 0:
                return refined.iloc[0].to_dict()
        return candidates.iloc[0].to_dict()

    # ── Fallback sur ELO ─────────────────────────────────────────────────────
    if len(df_elo) > 0:
        candidates = df_elo[df_elo['last_name'] == last_name]
        if len(candidates) == 0:
            candidates = df_elo[df_elo['last_name'].str.startswith(last_name[:4])]
        if len(candidates) > 0:
            if first_init:
                refined = candidates[candidates['first_init'] == first_init]
                if len(refined) > 0:
                    row = refined.iloc[0]
                else:
                    row = candidates.iloc[0]
            else:
                row = candidates.iloc[0]
            return {
                'player_name' : row['full_name'],
                'elo'         : row.get('elo_global', 1500),
                'elo_Hard'    : row.get('elo_Hard', 1500),
                'elo_Clay'    : row.get('elo_Clay', 1500),
                'elo_Grass'   : row.get('elo_Grass', 1500),
            }

    print(f"  ⚠️  Introuvable : '{name}' → last='{last_name}' init='{first_init}'")
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCTION DES FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def g(d: dict, key: str, default=np.nan):
    """Getter sécurisé sur un dict joueur."""
    v = d.get(key, default)
    return v if (v is not None and not (isinstance(v, float) and np.isnan(v))) else default


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS FEATURES MANQUANTES
# ─────────────────────────────────────────────────────────────────────────────

_h2h_cache = None
_dob_cache: dict | None = None


def load_dob_lookup() -> dict:
    """
    Charge le fichier players CSV (ATP ou WTA) et retourne un dict
    (last_name_lower, first_init) -> age en années (float) à aujourd'hui.
    Uses module-level _PROCESSED_DIR and _PLAYER_FILE set in main().
    """
    global _dob_cache
    if _dob_cache is not None:
        return _dob_cache

    path = _PROCESSED_DIR / _PLAYER_FILE
    if not path.exists():
        _dob_cache = {}
        return _dob_cache

    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=['dob', 'name_last', 'name_first'])
    df['dob'] = pd.to_numeric(df['dob'], errors='coerce')
    df = df.dropna(subset=['dob'])

    today = pd.Timestamp.today()
    lookup = {}
    for _, row in df.iterrows():
        last  = str(row['name_last']).lower().strip()
        first = str(row['name_first']).strip()
        if not last or not first:
            continue
        first_init = first[0].lower()
        dob_int    = int(row['dob'])
        try:
            dob = pd.Timestamp(str(dob_int))
            age = (today - dob).days / 365.25
            key = (last, first_init)
            # Garder le joueur le plus récent en cas de doublon
            if key not in lookup or age < lookup[key]:
                lookup[key] = age
        except Exception:
            continue

    _dob_cache = lookup
    return _dob_cache


def get_player_age(player: dict) -> float | None:
    """Retourne l'âge du joueur en années, ou None si introuvable."""
    dob_lookup = load_dob_lookup()
    name = player.get('player_name', '')
    if not name:
        return None
    parts = name.replace('.', '').split()
    if not parts:
        return None
    last_name  = parts[0].lower()
    first_init = parts[-1][0].lower() if len(parts) > 1 else ''
    return dob_lookup.get((last_name, first_init))


_H2H_LAMBDA = np.log(2) / 4.0   # exponential decay half-life = 4 years


def load_h2h_data() -> pd.DataFrame:
    """Charge matches_features_final en cache mémoire (colonnes nécessaires seulement)."""
    global _h2h_cache
    if _h2h_cache is None:
        path = _PROCESSED_DIR / "matches_features_final.parquet"
        if path.exists():
            _h2h_cache = pd.read_parquet(
                path, columns=['p1_name', 'p2_name', 'target', 'surface',
                               'tourney_name', 'tourney_date'])
        else:
            _h2h_cache = pd.DataFrame(columns=['p1_name', 'p2_name', 'target', 'surface',
                                               'tourney_name', 'tourney_date'])
    return _h2h_cache


def get_last_name(player_name: str) -> str:
    """Extrait le nom de famille depuis le format tennis-data 'Sinner J.'."""
    return player_name.lower().replace('.', '').split()[0] if player_name else ''


def compute_h2h(p1_name: str, p2_name: str, surface: str) -> dict:
    """
    Calcule les stats H2H à la volée depuis matches_features_final.
    Les noms dans ce fichier sont en format Sackmann (ex: 'Jannik Sinner').
    On cherche par nom de famille (dernier token du nom Sackmann).
    """
    df = load_h2h_data()
    if df.empty:
        return {'h2h_p1_winrate': 0.5, 'h2h_surf_p1_winrate': 0.5,
                'h2h_total': 0, 'h2h_played': 0}

    ln1 = get_last_name(p1_name)
    ln2 = get_last_name(p2_name)

    # Dans matches_features_final, p1_name/p2_name sont en format "Prénom Nom"
    # Le nom de famille est le dernier mot → on cherche si ln1 est dans p1_name ou p2_name
    p1_col = df['p1_name'].str.lower().str.split().str[-1]
    p2_col = df['p2_name'].str.lower().str.split().str[-1]

    mask = ((p1_col == ln1) & (p2_col == ln2)) | ((p1_col == ln2) & (p2_col == ln1))
    h2h  = df[mask]

    if len(h2h) == 0:
        return {'h2h_p1_winrate': 0.5, 'h2h_surf_p1_winrate': 0.5,
                'h2h_total': 0, 'h2h_played': 0, 'h2h_p1_winrate_recent': 0.5}

    # p1 (notre joueur 1) gagne quand : il est en position p1 ET target=1
    #                                   OU il est en position p2 ET target=0
    p1_is_row_p1 = (p1_col[mask] == ln1).values
    wins = ((p1_is_row_p1) & (h2h['target'].values == 1)) | \
           ((~p1_is_row_p1) & (h2h['target'].values == 0))
    total    = len(h2h)
    winrate  = float(wins.sum()) / total

    # H2H sur la surface
    surf_mask  = h2h['surface'] == surface
    h2h_surf   = h2h[surf_mask]
    if len(h2h_surf) > 0:
        p1_is_p1_s = p1_is_row_p1[surf_mask.values]
        wins_s     = ((p1_is_p1_s) & (h2h_surf['target'].values == 1)) | \
                     ((~p1_is_p1_s) & (h2h_surf['target'].values == 0))
        surf_winrate = float(wins_s.sum()) / len(h2h_surf)
    else:
        surf_winrate = winrate

    # H2H pondéré récence (exponential decay, half-life 4 years)
    now = pd.Timestamp.now()
    recent_num, recent_den = 0.0, 0.0
    if 'tourney_date' in h2h.columns:
        dates = pd.to_datetime(h2h['tourney_date'])
        for idx_i, (is_p1, won_t, dt) in enumerate(
                zip(p1_is_row_p1, h2h['target'].values, dates)):
            years_ago = (now - dt).days / 365.25
            w = np.exp(-_H2H_LAMBDA * max(years_ago, 0.0))
            p1_won_past = (is_p1 and won_t == 1) or (not is_p1 and won_t == 0)
            recent_num += w * float(p1_won_past)
            recent_den += w
    recent_wr = recent_num / recent_den if recent_den > 0 else 0.5

    return {
        'h2h_p1_winrate'        : winrate,
        'h2h_surf_p1_winrate'   : surf_winrate,
        'h2h_total'             : total,
        'h2h_played'            : total,
        'h2h_p1_winrate_recent' : recent_wr,
    }


def get_tourney_winrate(player: dict, tournament_name: str) -> float:
    """
    Retourne le win rate historique d'un joueur à ce tournoi.
    Ordre de recherche :
      1. player profile dict → 'tourney_winrates' (dict from update_database.py)
      2. matches_features_final historique (via load_h2h_data cache)
      3. Défaut 0.5
    """
    tname_lower = tournament_name.strip().lower() if tournament_name else ''

    # Source 1 : profil joueur pré-calculé (update_database.py)
    if 'tourney_winrates' in player and isinstance(player['tourney_winrates'], dict):
        twr = player['tourney_winrates']
        if tname_lower in twr:
            return float(twr[tname_lower])
        # Partial match (e.g. "miami" in "miami open")
        for k, v in twr.items():
            if tname_lower in k or k in tname_lower:
                return float(v)

    # Source 2 : historique global
    df = load_h2h_data()
    if df.empty or 'tourney_name' not in df.columns:
        return 0.5
    player_name = player.get('player_name', '')
    if not player_name:
        return 0.5

    p_last = player_name.lower().replace('.', '').split()[0]
    p1_last = df['p1_name'].str.lower().str.split().str[-1]
    p2_last = df['p2_name'].str.lower().str.split().str[-1]
    t_mask  = df['tourney_name'].str.lower().str.contains(tname_lower, na=False)

    p_rows_as_p1 = df[t_mask & (p1_last == p_last)]
    p_rows_as_p2 = df[t_mask & (p2_last == p_last)]

    won_p1 = (p_rows_as_p1['target'] == 1).sum()
    won_p2 = (p_rows_as_p2['target'] == 0).sum()
    total  = len(p_rows_as_p1) + len(p_rows_as_p2)
    if total == 0:
        return 0.5
    return float(won_p1 + won_p2) / total


# Mapping importance tournoi — identique à compute_contextual_features.py
_TOURNEY_LEVEL_MAP = {
    'G': 5, 'F': 4, 'M': 3, 'A': 2, 'O': 1,
}
_ROUND_ORDER = {
    'R128': 1, 'R64': 2, 'R32': 3, 'R16': 4,
    'QF': 5, 'SF': 6, 'F': 7, 'RR': 3, 'BR': 6,
}
_GRAND_SLAMS  = ['australian open', 'roland garros', 'wimbledon', 'us open']
_MASTERS_1000 = ['indian wells', 'miami', 'monte-carlo', 'madrid', 'rome',
                 'canada', 'montreal', 'toronto', 'cincinnati', 'shanghai',
                 'paris', 'bercy', 'atp finals']


def detect_tourney_level(tournament_name: str) -> str:
    name = tournament_name.lower()
    if any(gs in name for gs in _GRAND_SLAMS):
        return 'G'
    if any(m in name for m in _MASTERS_1000):
        return 'M'
    return 'A'


def compute_surface_specialization(elo_surf: float, elo_global: float) -> float:
    """Ratio ELO surface / ELO global — identique à compute_contextual_features.py."""
    if elo_global > 0:
        return elo_surf / elo_global
    return 1.0


def get_elo_global(player: dict, df_elo: pd.DataFrame) -> float:
    """Retourne l'ELO global du joueur (profil ou df_elo)."""
    elo_g = g(player, 'elo', 1500)
    if elo_g != 1500:
        return float(elo_g)
    # Fallback df_elo
    name  = player.get('player_name', '')
    parts = name.lower().replace('.', '').split()
    ln    = parts[0] if parts else ''
    fi    = parts[-1][0] if len(parts) > 1 else ''
    cands = df_elo[df_elo['last_name'] == ln]
    if len(cands) == 0:
        cands = df_elo[df_elo['last_name'].str.startswith(ln[:5])]
    if len(cands) > 1 and fi:
        refined = cands[cands['first_init'] == fi]
        if len(refined) > 0:
            cands = refined
    if len(cands) == 0:
        return 1500.0
    return float(cands.iloc[0].get('elo_global', 1500))


def build_feature_vector(match: dict, p1: dict, p2: dict,
                          features: list, df_elo: pd.DataFrame) -> np.ndarray:
    """Construit le vecteur de features pour le modèle."""
    surface  = match.get('surface', 'Hard')
    best_of5 = 1 if match.get('best_of', 3) == 5 else 0

    # ── ELO surface — profil d'abord, fallback df_elo si 1500 ────────────────
    elo_p1 = g(p1, f'elo_{surface}', g(p1, 'elo', 1500))
    elo_p2 = g(p2, f'elo_{surface}', g(p2, 'elo', 1500))

    def lookup_elo_from_db(player: dict, current_elo: float) -> float:
        if current_elo != 1500 or len(df_elo) == 0:
            return current_elo
        name  = player.get('player_name', '')
        if not name:
            return current_elo
        parts      = name.lower().replace('.', '').split()
        last_name  = parts[0] if parts else ''
        first_init = parts[-1][0] if len(parts) > 1 else ''
        cands = df_elo[df_elo['last_name'] == last_name]
        if len(cands) == 0:
            cands = df_elo[df_elo['last_name'].str.startswith(last_name[:5])]
        if len(cands) > 1 and first_init:
            refined = cands[cands['first_init'] == first_init]
            if len(refined) > 0:
                cands = refined
        if len(cands) == 0:
            return current_elo
        e = cands.iloc[0]
        return float(e.get(f'elo_{surface}', e.get('elo_global', 1500)))

    elo_p1 = lookup_elo_from_db(p1, elo_p1)
    elo_p2 = lookup_elo_from_db(p2, elo_p2)

    elo_diff = float(elo_p1) - float(elo_p2)
    elo_prob = 1 / (1 + 10 ** (-elo_diff / 400))

    # ── ELO global (pour surface specialization) ──────────────────────────────
    elo_global_p1 = get_elo_global(p1, df_elo)
    elo_global_p2 = get_elo_global(p2, df_elo)

    fv = {}

    # ── ELO ─────────────────────────────────────────────────────────────────
    fv['elo_diff']         = elo_diff
    fv['elo_win_prob_p1']  = elo_prob
    fv['elo_surface_diff'] = elo_diff

    # ── Correction 1 : Surface specialization ────────────────────────────────
    spec_p1 = compute_surface_specialization(elo_p1, elo_global_p1)
    spec_p2 = compute_surface_specialization(elo_p2, elo_global_p2)
    fv['p1_surface_specialization'] = spec_p1
    fv['p2_surface_specialization'] = spec_p2
    fv['surface_specialization_diff'] = spec_p1 - spec_p2

    # ── Correction 2 : H2H à la volée ────────────────────────────────────────
    h2h = compute_h2h(
        p1.get('player_name', ''),
        p2.get('player_name', ''),
        surface
    )
    fv['h2h_p1_winrate']          = h2h['h2h_p1_winrate']
    fv['h2h_surf_p1_winrate']     = h2h['h2h_surf_p1_winrate']
    fv['h2h_total']               = h2h['h2h_total']
    fv['h2h_played']              = h2h['h2h_played']
    fv['h2h_p1_winrate_recent']   = h2h['h2h_p1_winrate_recent']

    # ── Correction 3 : Contexte tournoi / round ───────────────────────────────
    tourney_level            = detect_tourney_level(match.get('tournament', ''))
    fv['tourney_importance'] = _TOURNEY_LEVEL_MAP.get(tourney_level, 2)
    round_str                = match.get('round', '')
    fv['round_importance']   = _ROUND_ORDER.get(round_str, 2)

    # Age diff — lookup dans atp_players.csv
    age_p1 = get_player_age(p1)
    age_p2 = get_player_age(p2)
    fv['age_diff'] = (age_p1 - age_p2) if (age_p1 is not None and age_p2 is not None) else np.nan

    # ── Winrates ─────────────────────────────────────────────────────────────
    for w in [5, 10, 20]:
        v1 = g(p1, f'winrate_{w}', 0.5)
        v2 = g(p2, f'winrate_{w}', 0.5)
        fv[f'p1_winrate_{w}']   = float(v1)
        fv[f'p2_winrate_{w}']   = float(v2)
        fv[f'winrate_diff_{w}'] = float(v1) - float(v2)

    # ── Winrate par surface ───────────────────────────────────────────────────
    for surf in ['Hard','Clay','Grass']:
        v1 = g(p1, f'winrate_surf_{surf}', 0.5)
        v2 = g(p2, f'winrate_surf_{surf}', 0.5)
        fv[f'p1_winrate_surf_{surf}']   = float(v1)
        fv[f'p2_winrate_surf_{surf}']   = float(v2)
        fv[f'winrate_surf_diff_{surf}'] = float(v1) - float(v2)

    # ── Streaks ───────────────────────────────────────────────────────────────
    fv['p1_streak']   = g(p1, 'streak', 0)
    fv['p2_streak']   = g(p2, 'streak', 0)
    fv['streak_diff'] = float(fv['p1_streak']) - float(fv['p2_streak'])

    # ── Fatigue ───────────────────────────────────────────────────────────────
    for d in [7, 14, 21]:
        v1 = g(p1, f'matches_{d}d', 2)
        v2 = g(p2, f'matches_{d}d', 2)
        fv[f'p1_matches_{d}d']   = float(v1)
        fv[f'p2_matches_{d}d']   = float(v2)
        fv[f'fatigue_diff_{d}d'] = float(v1) - float(v2)

    # ── Repos ────────────────────────────────────────────────────────────────
    fv['p1_days_since'] = g(p1, 'days_since', 7)
    fv['p2_days_since'] = g(p2, 'days_since', 7)

    # ── Fatigue sets/minutes (new) ────────────────────────────────────────────
    for d in [7, 14]:
        sv1 = g(p1, f'sets_{d}d', np.nan)
        sv2 = g(p2, f'sets_{d}d', np.nan)
        fv[f'p1_sets_{d}d'] = sv1
        fv[f'p2_sets_{d}d'] = sv2
        fv[f'fatigue_sets_diff_{d}d'] = (sv1 - sv2
            if not (np.isnan(sv1) or np.isnan(sv2)) else np.nan)
    # minutes not available in live pipeline → leave as NaN (imputed)
    for suffix in ['p1_minutes_7d', 'p2_minutes_7d', 'fatigue_min_diff_7d',
                   'p1_minutes_14d', 'p2_minutes_14d', 'fatigue_min_diff_14d']:
        fv[suffix] = np.nan

    # ── Momentum (new) ────────────────────────────────────────────────────────
    # Sets ratio (last 10)
    sr1 = g(p1, 'sets_ratio_10', np.nan)
    sr2 = g(p2, 'sets_ratio_10', np.nan)
    fv['p1_sets_ratio_10'] = sr1
    fv['p2_sets_ratio_10'] = sr2
    fv['sets_ratio_10_diff'] = (sr1 - sr2
        if not (np.isnan(sr1) or np.isnan(sr2)) else np.nan)

    # Tiebreak win rate (last 10)
    tb1 = g(p1, 'tiebreak_winrate_10', np.nan)
    tb2 = g(p2, 'tiebreak_winrate_10', np.nan)
    fv['p1_tiebreak_winrate_10'] = tb1
    fv['p2_tiebreak_winrate_10'] = tb2
    fv['tiebreak_winrate_10_diff'] = (tb1 - tb2
        if not (np.isnan(tb1) or np.isnan(tb2)) else np.nan)

    # Tournament win rate (on-the-fly lookup)
    tournament = match.get('tournament', '')
    tw1 = get_tourney_winrate(p1, tournament)
    tw2 = get_tourney_winrate(p2, tournament)
    fv['p1_tourney_winrate']   = tw1
    fv['p2_tourney_winrate']   = tw2
    fv['tourney_winrate_diff'] = tw1 - tw2

    # ── Classement ───────────────────────────────────────────────────────────
    r1 = g(p1, 'rank', 100)
    r2 = g(p2, 'rank', 100)
    fv['rank_diff']        = float(r1) - float(r2)
    fv['rank_ratio']       = float(r1) / max(float(r2), 1)
    fv['rank_points_diff'] = g(p1, 'rank_points', 1000) - g(p2, 'rank_points', 1000)

    # ── Contexte surface + best_of ────────────────────────────────────────────
    fv['is_best_of_5'] = best_of5
    for surf in ['Hard','Clay','Grass','Carpet']:
        fv[f'surface_{surf}'] = 1 if surface == surf else 0

    # ── Assembler selon l'ordre exact des features ────────────────────────────
    return np.array([fv.get(f, np.nan) for f in features], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# PRÉDICTIONS
# ─────────────────────────────────────────────────────────────────────────────

def predict_matches(matches: list, model, imputer, features: list,
                    df_players: pd.DataFrame, df_elo: pd.DataFrame,
                    platt=None, platt_surfaces: dict | None = None) -> pd.DataFrame:

    print(f"\n── Prédictions ({len(matches)} matchs) ─────────────────────")

    rows = []
    for match in matches:
        p1 = find_player(match['p1_name'], df_players, df_elo)
        p2 = find_player(match['p2_name'], df_players, df_elo)

        def _predict_one(pa, pb):
            Xv    = build_feature_vector(match, pa, pb, features, df_elo).reshape(1, -1)
            Ximp  = imputer.transform(Xv)
            p     = float(model.predict_proba(Ximp)[0, 1])
            # Scaler surface-spécifique > scaler global > brut
            surface = match.get('surface', '')
            active_scaler = (platt_surfaces or {}).get(surface) or platt
            if active_scaler is not None:
                if hasattr(active_scaler, 'predict_proba'):
                    p = float(active_scaler.predict_proba([[p]])[0, 1])
                else:
                    p = float(np.clip(active_scaler.predict([[p]])[0], 0.01, 0.99))
            return p

        prob_fwd = _predict_one(p1, p2)          # prob p1 gagne quand p1 est "p1"
        prob_bwd = _predict_one(p2, p1)          # prob p2 gagne quand p2 est "p1"
        prob_p1  = (prob_fwd + (1 - prob_bwd)) / 2   # moyenne des deux sens
        prob_p2  = 1 - prob_p1

        X     = build_feature_vector(match, p1, p2, features, df_elo).reshape(1, -1)

        elo_diff = X[0, features.index('elo_diff')] if 'elo_diff' in features else 0
        elo_prob = float(1 / (1 + 10 ** (-elo_diff / 400)))

        rows.append({
            'tournament': match.get('tournament', ''),
            'surface'   : match.get('surface', ''),
            'round'     : match.get('round', ''),
            'best_of'   : match.get('best_of', 3),
            'p1_name'   : match['p1_name'],
            'p2_name'   : match['p2_name'],
            'prob_p1'   : prob_p1,
            'prob_p2'   : prob_p2,
            'elo_prob'  : elo_prob,
            'predicted' : match['p1_name'] if prob_p1 > 0.5 else match['p2_name'],
            'confidence': max(prob_p1, prob_p2),
            'p1_found'  : bool(p1),
            'p2_found'  : bool(p2),
        })

    df = pd.DataFrame(rows)

    # ── Affichage ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*78}")
    print(f"{'Match':<38} {'ML P1':>7} {'ML P2':>7} {'ELO':>7}  Favori")
    print(f"{'─'*78}")

    for _, r in df.iterrows():
        ok  = '✅' if r['p1_found'] and r['p2_found'] else '⚠️ '
        ms  = f"{r['p1_name']} vs {r['p2_name']}"[:37]
        fav = f"→ {r['predicted']}"[:22]
        print(f"{ok} {ms:<37} {r['prob_p1']:>6.1%} {r['prob_p2']:>6.1%} "
              f"{r['elo_prob']:>6.1%}  {fav}")

    print(f"{'─'*78}")
    n_ok = df['p1_found'].sum() + df['p2_found'].sum()
    print(f"  Joueurs trouvés : {n_ok}/{len(df)*2} | "
          f"Matchs complets : {(df['p1_found'] & df['p2_found']).sum()}/{len(df)}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# VALUE BETS
# ─────────────────────────────────────────────────────────────────────────────

def prompt_odds(df_pred: pd.DataFrame) -> pd.DataFrame:
    """Saisie interactive des cotes Pinnacle."""
    print("\n── Saisie cotes Pinnacle ────────────────────────────")
    print("  (Appuyez Entrée pour passer)\n")

    df = df_pred.copy()
    df['bk_odd_p1'] = np.nan
    df['bk_odd_p2'] = np.nan

    for i, row in df.iterrows():
        print(f"  {row['p1_name']} vs {row['p2_name']} "
              f"  [ML: {row['prob_p1']:.0%} / {row['prob_p2']:.0%}]")
        try:
            o1 = input(f"    Cote {row['p1_name'][:20]} : ").strip()
            o2 = input(f"    Cote {row['p2_name'][:20]} : ").strip()
            if o1: df.at[i, 'bk_odd_p1'] = float(o1.replace(',', '.'))
            if o2: df.at[i, 'bk_odd_p2'] = float(o2.replace(',', '.'))
        except ValueError:
            print("    ⚠️  Cote invalide, ignorée")

    return df


def compute_value_bets(df: pd.DataFrame,
                        min_edge: float = 0.03,
                        min_prob: float = 0.55,
                        min_bk_dir_prob: float = 0.40) -> pd.DataFrame:
    """
    Identifie les value bets en comparant la proba modèle à la proba implicite bookmaker.

    min_bk_dir_prob : probabilité implicite bookmaker minimale pour le côté parié.
        Ne pas signaler de value bet sur un joueur que le marché donne à < 40%
        (cote > ~2.5) — le modèle surévalue systématiquement les gros outsiders
        faute de données d'entraînement suffisantes.
    """
    df = df.copy()
    df['edge_p1']   = np.nan
    df['edge_p2']   = np.nan
    df['ev_p1']     = np.nan
    df['ev_p2']     = np.nan
    df['value_bet'] = ''

    if 'bk_odd_p1' not in df.columns:
        return df

    for i, row in df.iterrows():
        o1, o2 = row.get('bk_odd_p1'), row.get('bk_odd_p2')
        if pd.isna(o1) or pd.isna(o2) or o1 <= 1 or o2 <= 1:
            continue

        margin = 1/o1 + 1/o2 - 1
        imp_p1 = (1/o1) / (1 + margin)
        imp_p2 = (1/o2) / (1 + margin)

        e1 = row['prob_p1'] - imp_p1
        e2 = row['prob_p2'] - imp_p2

        df.at[i, 'edge_p1'] = e1
        df.at[i, 'edge_p2'] = e2
        df.at[i, 'ev_p1']   = row['prob_p1'] * o1 - 1
        df.at[i, 'ev_p2']   = row['prob_p2'] * o2 - 1

        vbs = []
        if e1 >= min_edge and row['prob_p1'] >= min_prob and imp_p1 >= min_bk_dir_prob:
            vbs.append(f"✅ {row['p1_name']} edge={e1:+.1%} EV={df.at[i,'ev_p1']:+.1%}")
        if e2 >= min_edge and row['prob_p2'] >= min_prob and imp_p2 >= min_bk_dir_prob:
            vbs.append(f"✅ {row['p2_name']} edge={e2:+.1%} EV={df.at[i,'ev_p2']:+.1%}")

        df.at[i, 'value_bet'] = ' | '.join(vbs)

    return df


def print_value_bets(df: pd.DataFrame):
    if 'value_bet' not in df.columns:
        return
    vb = df[df['value_bet'].str.len() > 0]
    print(f"\n── Value Bets ({'─'*40}")
    if len(vb) == 0:
        print("  Aucun value bet (cotes non saisies ou edge < 3%)")
        return
    for _, row in vb.iterrows():
        print(f"\n  🎾 {row['p1_name']} vs {row['p2_name']} "
              f"| {row['surface']} | {row['tournament']}")
        print(f"     {row['value_bet']}")
    print(f"\n  Total : {len(vb)} match(s) avec value bet")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', default=str(date.today()))
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--odds', action='store_true')
    parser.add_argument('--tour', default='atp', choices=['atp', 'wta'],
                        help="Tour à prédire : atp (défaut) ou wta")
    args = parser.parse_args()

    target_date = datetime.strptime(args.date, "%Y-%m-%d").date()

    tour  = args.tour.lower()
    cfg   = get_tour_config(tour)
    paths = get_paths(tour)

    global _PROCESSED_DIR, _PLAYER_FILE
    _PROCESSED_DIR = paths['processed_dir']
    _PLAYER_FILE   = cfg['player_file']

    MODELS_DIR = paths['models_dir']
    OUTPUT_DIR = paths['predictions_dir']
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print(f"PRÉDICTIONS {tour.upper()} — {target_date}")
    print("=" * 55)

    model, imputer, features, platt, platt_surfaces = load_model_artifacts(MODELS_DIR)
    df_players = load_player_database(paths['processed_dir'])
    df_elo     = load_elo_ratings(paths['processed_dir'])

    # Récupération matchs
    matches = fetch_matches(target_date, cfg, tour=tour)
    if not matches:
        matches = prompt_manual_matches()
    if not matches:
        print("\n⚠️  Aucun match. Fin.")
        return

    # Prédictions
    df = predict_matches(matches, model, imputer, features, df_players, df_elo,
                         platt=platt, platt_surfaces=platt_surfaces)

    # Cotes + value bets
    if args.odds:
        df = prompt_odds(df)
    df = compute_value_bets(df)
    print_value_bets(df)

    # Sauvegarde
    if args.save:
        path = OUTPUT_DIR / f"predictions_{target_date}.csv"
        df.to_csv(path, index=False)
        print(f"\n✅ Sauvegardé : {path}")

    return df


if __name__ == "__main__":
    main()
