"""ML wrapper — loads artifacts once, exposes predict()."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Feature builder
# ─────────────────────────────────────────────────────────────────────────────

_ROUND_IMP = {
    'R128': 0.1, 'R64': 0.2, 'R32': 0.3, 'R16': 0.4,
    'QF': 0.6, 'SF': 0.8, 'F': 1.0, 'RR': 0.3,
}
_SURFACE_ENC = {'Hard': 0, 'Clay': 1, 'Grass': 2, 'Carpet': 3}
_GS = {'australian open', 'roland garros', 'wimbledon', 'us open'}
_MASTERS = {
    'miami open', 'monte-carlo', 'madrid open', 'rome', 'canadian open',
    'cincinnati', 'shanghai', 'paris masters', 'indian wells',
}


def _tourney_importance(name: str) -> float:
    n = name.lower()
    if any(g in n for g in _GS):
        return 1.0
    if any(m in n for m in _MASTERS):
        return 0.7
    return 0.4


def _get_player(profiles: pd.DataFrame, name: str) -> dict | None:
    key = name.lower().strip()
    rows = profiles[profiles['name_key'] == key]
    if rows.empty:
        last = key.split()[0] if key else ''
        rows = profiles[profiles['name_key'].str.startswith(last)]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def _elo_win_prob(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def _build_features(p1: dict, p2: dict, tournament: str, surface: str,
                    round_: str, best_of: int, feature_list: list[str]) -> np.ndarray:
    """Build feature vector matching feature_list order. Missing → NaN."""
    surf_key = f'elo_{surface.lower()}'
    elo1 = p1.get('elo', 1500.0) or 1500.0
    elo2 = p2.get('elo', 1500.0) or 1500.0
    elo_surf1 = p1.get(surf_key, elo1) or elo1
    elo_surf2 = p2.get(surf_key, elo2) or elo2

    row: dict[str, Any] = {
        'elo_diff':             elo1 - elo2,
        'elo_surface_diff':     elo_surf1 - elo_surf2,
        'elo_win_prob_p1':      _elo_win_prob(elo1, elo2),
        'rank_diff':            (p1.get('rank', 500) or 500) - (p2.get('rank', 500) or 500),
        'rank_ratio':           ((p1.get('rank', 500) or 500) /
                                 max((p2.get('rank', 500) or 500), 1)),
        'rank_points_diff':     (p1.get('rank_points', 0) or 0) - (p2.get('rank_points', 0) or 0),
        'form_win_rate_5_diff': (p1.get('form_win_rate_5', 0.5) or 0.5) -
                                (p2.get('form_win_rate_5', 0.5) or 0.5),
        'form_win_rate_10_diff':(p1.get('form_win_rate_10', 0.5) or 0.5) -
                                (p2.get('form_win_rate_10', 0.5) or 0.5),
        'form_win_rate_20_diff':(p1.get('form_win_rate_20', 0.5) or 0.5) -
                                (p2.get('form_win_rate_20', 0.5) or 0.5),
        'h2h_win_rate_p1':      p1.get('h2h_win_rate', 0.5) or 0.5,
        'surface_enc':          _SURFACE_ENC.get(surface, 0),
        'tourney_importance':   _tourney_importance(tournament),
        'round_importance':     _ROUND_IMP.get(round_, 0.3),
        'best_of_5':            1 if best_of == 5 else 0,
    }

    vec = np.array([row.get(f, np.nan) for f in feature_list], dtype=float)
    return vec.reshape(1, -1)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    artifacts: dict,
    p1_name: str,
    p2_name: str,
    tournament: str,
    surface: str,
    round_: str,
    best_of: int,
    odd_p1: float | None,
    odd_p2: float | None,
    bankroll: float,
) -> dict:
    """Run prediction and return result dict."""
    profiles     = artifacts['profiles']
    model        = artifacts['model']
    imputer      = artifacts['imputer']
    platt        = artifacts['platt']
    feature_list = artifacts['feature_list']

    p1 = _get_player(profiles, p1_name)
    p2 = _get_player(profiles, p2_name)

    p1_found = p1 is not None
    p2_found = p2 is not None

    if not p1_found:
        p1 = {}
    if not p2_found:
        p2 = {}

    elo_only = not p1_found or not p2_found

    elo1 = p1.get('elo', 1500.0) or 1500.0
    elo2 = p2.get('elo', 1500.0) or 1500.0
    elo_prob = _elo_win_prob(elo1, elo2)

    X = _build_features(p1, p2, tournament, surface, round_, best_of, feature_list)
    X_imp = imputer.transform(X)
    raw_prob = model.predict_proba(X_imp)[0, 1]
    cal_prob = float(platt.predict_proba([[raw_prob]])[0, 1])
    cal_prob = max(0.01, min(0.99, cal_prob))

    edge = ev = kelly_frac = kelly_eur = None
    if odd_p1 is not None and odd_p1 > 1.0:
        implied = 1.0 / odd_p1
        edge = round(cal_prob - implied, 4)
        ev   = round(cal_prob * (odd_p1 - 1) - (1 - cal_prob), 4)
        if edge > 0:
            raw_kelly = (cal_prob * odd_p1 - 1) / (odd_p1 - 1)
            kelly_frac = round(min(raw_kelly * 0.25, 0.25), 4)
            kelly_eur  = round(kelly_frac * bankroll, 2)

    return {
        'prob_p1':    round(cal_prob, 4),
        'prob_p2':    round(1 - cal_prob, 4),
        'elo_prob':   round(elo_prob, 4),
        'edge':       edge,
        'ev':         ev,
        'kelly_frac': kelly_frac,
        'kelly_eur':  kelly_eur,
        'p1_found':   p1_found,
        'p2_found':   p2_found,
        'elo_only':   elo_only,
        'confidence': round(cal_prob, 4),
    }
