# src/config.py
"""
Configuration centrale ATP / WTA.

Chaque script du pipeline charge ses chemins et paramètres via :
    from config import get_paths, get_tour_config
    cfg = get_tour_config(tour)
    paths = get_paths(tour)
"""

import pandas as pd
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── Split temporel — référence unique pour tout le pipeline ──────────────────
TEMPORAL_SPLIT = {
    'train_end'   : pd.Timestamp('2023-12-31'),
    'valid_start' : pd.Timestamp('2024-01-01'),
    'valid_end'   : pd.Timestamp('2024-12-31'),
    'test_start'  : pd.Timestamp('2025-01-01'),
}

# ── Configuration par tour ────────────────────────────────────────────────────

TOUR_CONFIG = {
    'atp': {
        # Sources Sackmann
        'sackmann_repo'    : 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master',
        'file_prefix'      : 'atp',
        'player_file'      : 'atp_players.csv',
        'ranking_files'    : [
            'atp_rankings_00s.csv', 'atp_rankings_10s.csv',
            'atp_rankings_20s.csv', 'atp_rankings_current.csv',
        ],
        'match_years'      : list(range(1968, 2026)),
        'qual_chall_years' : list(range(1978, 2026)),

        # Cotes tennis-data.co.uk
        'odds_filename'    : lambda year: f'atp_{year}.xlsx',
        'odds_url'         : lambda year: f'http://www.tennis-data.co.uk/{year}/{year}.xlsx',

        # ELO / Glicko — K-factors par niveau de tournoi
        'k_factor_map': {'G': 48, 'M': 40, 'A': 32, 'F': 40, 'O': 32, 'D': 20},

        # Importance tournoi (pour compute_contextual_features)
        'level_importance': {'G': 1.0, 'M': 0.85, 'A': 0.65, 'F': 0.90, 'O': 0.50, 'D': 0.40},

        # Niveau de tournoi → label lisible (pour dashboard/backtest)
        'level_labels': {
            'G': 'Grand Chelem', 'M': 'Masters 1000',
            'A': 'ATP 250/500', 'F': 'Finals', 'O': 'Autre',
        },

        # Noms tennis-data → code Sackmann (pour update_database)
        'td_level_map': {
            'Grand Slam': 'G', 'Masters': 'M', 'Masters 1000': 'M',
            'ATP500': 'A', 'ATP250': 'A', 'ATP Finals': 'F',
            'Next Gen Finals': 'F', 'United Cup': 'O',
        },

        # Filtrage best-of-5 (Grand Chelem ATP uniquement)
        'bo5_levels': {'G'},

        # Sofascore : filtre catégorie
        'sofascore_cat': 'ATP',

        # Noms tournois majeurs (pour predict_today)
        'grand_slams': {
            'Australian Open', 'Roland Garros', 'Wimbledon', 'US Open',
        },
        'top_events': {
            'Indian Wells', 'Miami', 'Monte Carlo', 'Madrid', 'Rome',
            'Canada', 'Cincinnati', 'Shanghai', 'Paris', 'Australian Open',
            'Roland Garros', 'Wimbledon', 'US Open',
        },
    },

    'wta': {
        # Sources Sackmann
        'sackmann_repo'    : 'https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master',
        'file_prefix'      : 'wta',
        'player_file'      : 'wta_players.csv',
        'ranking_files'    : [
            'wta_rankings_00s.csv', 'wta_rankings_10s.csv',
            'wta_rankings_20s.csv', 'wta_rankings_current.csv',
        ],
        'match_years'      : list(range(1968, 2026)),
        'qual_chall_years' : [],   # pas de fichier séparé pour WTA

        # Cotes tennis-data.co.uk
        'odds_filename'    : lambda year: f'wta_{year}.xlsx',
        'odds_url'         : lambda year: f'http://www.tennis-data.co.uk/{year}/{year}W.xlsx',

        # ELO / Glicko — K-factors par niveau de tournoi
        # WTA levels: G=Grand Slam, PM=Premier Mandatory/WTA1000, P=Premier/WTA500,
        #             I=International/WTA250, F=Finals, D=BJK Cup
        'k_factor_map': {'G': 48, 'PM': 40, 'P': 32, 'I': 28, 'F': 40, 'D': 20},

        # Importance tournoi
        'level_importance': {'G': 1.0, 'PM': 0.85, 'P': 0.65, 'I': 0.50, 'F': 0.90, 'D': 0.40},

        # Niveau de tournoi → label lisible
        'level_labels': {
            'G': 'Grand Chelem', 'PM': 'WTA 1000',
            'P': 'WTA 500', 'I': 'WTA 250', 'F': 'Finals', 'D': 'BJK Cup',
        },

        # Noms tennis-data → code Sackmann (pour update_database)
        'td_level_map': {
            'Grand Slam': 'G', 'Premier Mandatory': 'PM', 'WTA 1000': 'PM',
            'Premier 5': 'PM', 'Premier': 'P', 'WTA 500': 'P',
            'International': 'I', 'WTA 250': 'I',
            'WTA Finals': 'F', 'WTA Elite Trophy': 'F',
        },

        # WTA : jamais de best-of-5
        'bo5_levels': set(),

        # Sofascore : filtre catégorie
        'sofascore_cat': 'WTA',

        # Noms tournois majeurs
        'grand_slams': {
            'Australian Open', 'Roland Garros', 'Wimbledon', 'US Open',
        },
        'top_events': {
            'Indian Wells', 'Miami', 'Madrid', 'Rome', 'Canada',
            'Cincinnati', 'Beijing', 'Wuhan', 'Australian Open',
            'Roland Garros', 'Wimbledon', 'US Open',
        },
    },
}


def get_tour_config(tour: str) -> dict:
    """Retourne la configuration d'un tour (atp ou wta)."""
    t = tour.lower()
    if t not in TOUR_CONFIG:
        raise ValueError(f"tour doit être 'atp' ou 'wta', reçu: '{tour}'")
    return TOUR_CONFIG[t]


def get_paths(tour: str) -> dict:
    """Retourne tous les chemins résolus pour un tour donné."""
    t = tour.lower()
    if t not in TOUR_CONFIG:
        raise ValueError(f"tour doit être 'atp' ou 'wta', reçu: '{tour}'")
    return {
        'raw_dir'        : ROOT / 'data' / 'raw'        / t,
        'odds_dir'       : ROOT / 'data' / 'odds'       / t,
        'processed_dir'  : ROOT / 'data' / 'processed'  / t,
        'models_dir'     : ROOT / 'data' / 'models'     / t,
        'predictions_dir': ROOT / 'data' / 'predictions'/ t,
        'root'           : ROOT,
    }


def make_dirs(tour: str) -> None:
    """Crée tous les dossiers nécessaires pour un tour."""
    paths = get_paths(tour)
    for key, path in paths.items():
        if key != 'root':
            path.mkdir(parents=True, exist_ok=True)


def _file_age_hours(path: Path) -> float:
    """Retourne l'âge du fichier en heures, ou inf s'il n'existe pas."""
    if not path.exists():
        return float("inf")
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime).total_seconds() / 3600
