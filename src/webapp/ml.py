"""ML wrapper — loads artifacts once, exposes predict()."""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_ROUND_IMP = {
    'R128': 0.1, 'R64': 0.2, 'R32': 0.3, 'R16': 0.4,
    'QF': 0.6, 'SF': 0.8, 'F': 1.0, 'RR': 0.3,
}
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


def _has_rank(row: pd.Series) -> bool:
    v = row.get('rank')
    return v is not None and not (isinstance(v, float) and np.isnan(v))


def _name_tokens(name: str) -> frozenset:
    """Tokenize a normalized name into a frozenset — order-independent, hyphen-insensitive."""
    return frozenset(name.lower().strip().replace('-', ' ').split())


def _get_player(profiles, name: str) -> dict | None:
    key = name.lower().strip()
    last = key.split()[-1] if key else ''
    tokens = _name_tokens(name)

    # Fast path: O(1) dict lookup (profiles_dict built at startup)
    if isinstance(profiles, dict):
        if key in profiles:
            return profiles[key]
        # Token-set match: handles "Zheng Qinwen" ↔ "Qinwen Zheng" and hyphenated names
        if tokens:
            for k, v in profiles.items():
                if _name_tokens(k) == tokens:
                    return v
        # Last-name fallback
        if last:
            for k, v in profiles.items():
                if k.split()[-1] == last if k.split() else False:
                    return v
        return None

    # Fallback: DataFrame scan (backward compat)
    valid = profiles['name_key'].notna()
    exact = profiles[valid & (profiles['name_key'] == key)]
    lastname = profiles[valid & profiles['name_key'].str.endswith(last)] if last else pd.DataFrame()

    candidates = pd.concat([exact, lastname]).drop_duplicates()
    if candidates.empty:
        return None

    ranked = candidates[candidates.apply(_has_rank, axis=1)] if 'rank' in candidates.columns else pd.DataFrame()
    best = ranked if not ranked.empty else candidates
    return best.iloc[0].to_dict()


def _elo_win_prob(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def _rank_adjusted_elo(elo: float, rank: float | None) -> float:
    """Blend profile ELO with rank-implied ELO when they diverge (stale ELO).

    Uses a linear rank→ELO mapping: rank 1 ≈ 2100, rank 100 ≈ 1700.
    Blending weight increases with divergence to handle stale WTA ELOs
    (e.g. ex-#1 Pliskova ELO=2078 while currently ranked #41).

    Gap thresholds (ELO points):
      > 400 : 5 % historical ELO + 95 % rank-implied  (extreme staleness)
      > 200 : 20 %  ELO + 80 % rank
      >  50 : 50 %  ELO + 50 % rank
      ≤  50 : keep historical ELO as-is
    """
    if rank is None or (isinstance(rank, float) and np.isnan(rank)):
        return elo
    elo_from_rank = max(1200.0, 2100.0 - 4.0 * float(rank))
    gap = abs(elo - elo_from_rank)
    if gap > 400:
        w_elo = 0.05
    elif gap > 200:
        w_elo = 0.20
    elif gap > 50:
        w_elo = 0.50
    else:
        return elo
    return w_elo * elo + (1.0 - w_elo) * elo_from_rank


def _v(d: dict, key: str, default: float) -> float:
    """Safe numeric get with fallback."""
    v = d.get(key)
    if v is None:
        return default
    try:
        f = float(v)
        return default if np.isnan(f) else f
    except (TypeError, ValueError):
        return default


def _build_features(p1: dict, p2: dict, tournament: str, surface: str,
                    round_: str, best_of: int, feature_list: list[str],
                    h2h: dict | None = None) -> np.ndarray:
    """Build feature vector matching feature_list order. Missing → NaN."""
    # ELO — profile columns: elo, elo_Hard, elo_Clay, elo_Grass
    # Apply rank-based ELO correction to reduce stale-ELO bias (e.g. player who
    # peaked years ago still has high ELO but current rank has dropped significantly).
    r1_raw = p1.get('rank')
    r2_raw = p2.get('rank')
    r1_rank = float(r1_raw) if r1_raw is not None and not (isinstance(r1_raw, float) and np.isnan(r1_raw)) else None
    r2_rank = float(r2_raw) if r2_raw is not None and not (isinstance(r2_raw, float) and np.isnan(r2_raw)) else None

    elo1 = _rank_adjusted_elo(_v(p1, 'elo', 1500.0), r1_rank)
    elo2 = _rank_adjusted_elo(_v(p2, 'elo', 1500.0), r2_rank)
    elo_surf_key = f'elo_{surface}'  # e.g. 'elo_Hard'
    elo_s1 = _rank_adjusted_elo(_v(p1, elo_surf_key, elo1), r1_rank)
    elo_s2 = _rank_adjusted_elo(_v(p2, elo_surf_key, elo2), r2_rank)

    # Win rates — clamped to [0.1, 0.9] and neutralised when sample too small (<3 matches)
    def _wr(d: dict, key: str) -> float:
        recent = _v(d, 'matches_14d', 10.0)  # default 10 = assume enough history
        if recent < 3:
            return 0.5  # too few matches to trust the rolling winrate
        return max(0.1, min(0.9, _v(d, key, 0.5)))

    wr1_5  = _wr(p1, 'winrate_5')
    wr2_5  = _wr(p2, 'winrate_5')
    wr1_10 = _wr(p1, 'winrate_10')
    wr2_10 = _wr(p2, 'winrate_10')
    wr1_20 = _wr(p1, 'winrate_20')
    wr2_20 = _wr(p2, 'winrate_20')

    # Streaks
    st1 = _v(p1, 'streak', 0.0)
    st2 = _v(p2, 'streak', 0.0)

    # Surface win rates — same clamping + small-sample neutralisation
    wrs1_h = _wr(p1, 'winrate_surf_Hard')
    wrs2_h = _wr(p2, 'winrate_surf_Hard')
    wrs1_c = _wr(p1, 'winrate_surf_Clay')
    wrs2_c = _wr(p2, 'winrate_surf_Clay')
    wrs1_g = _wr(p1, 'winrate_surf_Grass')
    wrs2_g = _wr(p2, 'winrate_surf_Grass')

    # Fatigue — profile columns: matches_7d, matches_14d, days_since
    m7_1  = _v(p1, 'matches_7d',  0.0)
    m7_2  = _v(p2, 'matches_7d',  0.0)
    m14_1 = _v(p1, 'matches_14d', 0.0)
    m14_2 = _v(p2, 'matches_14d', 0.0)
    ds1   = _v(p1, 'days_since',  7.0)
    ds2   = _v(p2, 'days_since',  7.0)

    # Rank — use NaN when missing so imputer treats it as neutral (0.5)
    # Do NOT default to 500: a NaN-ranked player appearing as rank=500 would
    # create a huge artificial rank_diff against any known-ranked opponent.
    def _rank(d: dict) -> float:
        v = d.get('rank')
        if v is None:
            return np.nan
        try:
            f = float(v)
            return np.nan if np.isnan(f) else f
        except (TypeError, ValueError):
            return np.nan

    r1  = _rank(p1)
    r2  = _rank(p2)
    rp1 = _v(p1, 'rank_points', 0.0)
    rp2 = _v(p2, 'rank_points', 0.0)

    # Surface one-hot
    surf_hard   = 1.0 if surface == 'Hard'   else 0.0
    surf_clay   = 1.0 if surface == 'Clay'   else 0.0
    surf_grass  = 1.0 if surface == 'Grass'  else 0.0
    surf_carpet = 1.0 if surface == 'Carpet' else 0.0

    row: dict[str, Any] = {
        # ELO
        'elo_diff':          elo1 - elo2,
        'elo_surface_diff':  elo_s1 - elo_s2,
        'elo_win_prob_p1':   _elo_win_prob(elo1, elo2),
        # Rank — NaN when either player rank is unknown → imputer fills neutral
        'rank_diff':         (r1 - r2) if not (np.isnan(r1) or np.isnan(r2)) else np.nan,
        'rank_ratio':        (r1 / max(r2, 1.0)) if not (np.isnan(r1) or np.isnan(r2)) else np.nan,
        'rank_points_diff':  rp1 - rp2,
        # Win rates — individual + diff
        'p1_winrate_5':       wr1_5,
        'p2_winrate_5':       wr2_5,
        'winrate_diff_5':     wr1_5 - wr2_5,
        'p1_winrate_10':      wr1_10,
        'p2_winrate_10':      wr2_10,
        'winrate_diff_10':    wr1_10 - wr2_10,
        'p1_winrate_20':      wr1_20,
        'p2_winrate_20':      wr2_20,
        'winrate_diff_20':    wr1_20 - wr2_20,
        # Streaks
        'p1_streak':          st1,
        'p2_streak':          st2,
        'streak_diff':        st1 - st2,
        # Surface win rates
        'p1_winrate_surf_Hard':    wrs1_h,
        'p2_winrate_surf_Hard':    wrs2_h,
        'winrate_surf_diff_Hard':  wrs1_h - wrs2_h,
        'p1_winrate_surf_Clay':    wrs1_c,
        'p2_winrate_surf_Clay':    wrs2_c,
        'winrate_surf_diff_Clay':  wrs1_c - wrs2_c,
        'p1_winrate_surf_Grass':   wrs1_g,
        'p2_winrate_surf_Grass':   wrs2_g,
        'winrate_surf_diff_Grass': wrs1_g - wrs2_g,
        # H2H — use real stats from lookup when available, neutral default otherwise
        'h2h_p1_winrate':      (h2h['p1_wins'] / h2h['total']) if h2h and h2h.get('total', 0) > 0 else 0.5,
        'h2h_surf_p1_winrate': (h2h['surf_p1_wins'] / h2h['surf_total']) if h2h and h2h.get('surf_total', 0) > 0 else 0.5,
        'h2h_total':           float(h2h['total']) if h2h else 0.0,
        'h2h_played':          float(h2h['total']) if h2h else 0.0,
        # Fatigue
        'p1_matches_7d':    m7_1,
        'p2_matches_7d':    m7_2,
        'fatigue_diff_7d':  m7_1 - m7_2,
        'p1_matches_14d':   m14_1,
        'p2_matches_14d':   m14_2,
        'fatigue_diff_14d': m14_1 - m14_2,
        'p1_days_since':    ds1,
        'p2_days_since':    ds2,
        # Context
        'tourney_importance': _tourney_importance(tournament),
        'round_importance':   _ROUND_IMP.get(round_, 0.3),
        'is_best_of_5':       1.0 if best_of == 5 else 0.0,
        'age_diff':           0.0,  # unknown at prediction time
        # Surface one-hot
        'surface_Hard':   surf_hard,
        'surface_Clay':   surf_clay,
        'surface_Grass':  surf_grass,
        'surface_Carpet': surf_carpet,
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
    kelly_fraction: float = 0.25,
    h2h: dict | None = None,
) -> dict:
    """Run prediction and return result dict."""
    # Use O(1) dict lookup if available, fall back to DataFrame scan
    profiles       = artifacts.get('profiles_dict') or artifacts['profiles']
    model          = artifacts['model']
    imputer        = artifacts['imputer']
    platt          = artifacts['platt']
    feature_list   = artifacts['feature_list']
    ranking_lookup = artifacts.get('ranking_lookup', {})

    p1 = _get_player(profiles, p1_name)
    p2 = _get_player(profiles, p2_name)

    # Override rank from current weekly rankings if available
    if ranking_lookup:
        for p, name in [(p1, p1_name), (p2, p2_name)]:
            if p is not None:
                key = name.lower().strip()
                if key in ranking_lookup:
                    p['rank'] = float(ranking_lookup[key])
                else:
                    # Try last name only
                    last = key.split()[-1]
                    matches = {k: v for k, v in ranking_lookup.items() if k.endswith(last)}
                    if len(matches) == 1:
                        p['rank'] = float(list(matches.values())[0])

    p1_found = p1 is not None
    p2_found = p2 is not None

    if not p1_found:
        p1 = {}
    if not p2_found:
        p2 = {}

    elo_only = not p1_found or not p2_found

    r1_blend = p1.get('rank')
    r2_blend = p2.get('rank')
    r1_rank = float(r1_blend) if r1_blend is not None and not (isinstance(r1_blend, float) and np.isnan(r1_blend)) else None
    r2_rank = float(r2_blend) if r2_blend is not None and not (isinstance(r2_blend, float) and np.isnan(r2_blend)) else None
    elo1 = _rank_adjusted_elo(_v(p1, 'elo', 1500.0), r1_rank)
    elo2 = _rank_adjusted_elo(_v(p2, 'elo', 1500.0), r2_rank)
    elo_prob = _elo_win_prob(elo1, elo2)

    if elo_only:
        # One or both players not in profiles — use ELO probability only.
        cal_prob = max(0.01, min(0.99, elo_prob))
    else:
        X = _build_features(p1, p2, tournament, surface, round_, best_of, feature_list, h2h=h2h)
        X_df = pd.DataFrame(X, columns=feature_list)
        X_imp = imputer.transform(X_df)
        raw_prob = model.predict_proba(X_imp)[0, 1]
        if artifacts.get('platt_pinnacle'):
            # LinearRegression calibrated against Pinnacle no-vig probabilities
            xgb_prob = float(np.clip(platt.predict([[raw_prob]])[0], 0.01, 0.99))
        else:
            xgb_prob = float(platt.predict_proba([[raw_prob]])[0, 1])

        # Dynamic ELO blend: XGBoost was trained on rich historical features.
        # When recent match samples are sparse (ESPN sync only has 14 days),
        # winrates and fatigue features are unreliable → trust ELO more.
        m14_p1 = _v(p1, 'matches_14d', 10.0)
        m14_p2 = _v(p2, 'matches_14d', 10.0)
        min_matches = min(m14_p1, m14_p2)
        if min_matches >= 3:
            elo_w = 0.30   # Good data: 70% XGBoost, 30% ELO
        elif min_matches >= 1:
            elo_w = 0.50   # Medium data: 50/50
        else:
            elo_w = 0.70   # Sparse data: 70% ELO, 30% XGBoost

        cal_prob = elo_w * elo_prob + (1 - elo_w) * xgb_prob
        cal_prob = max(0.01, min(0.99, cal_prob))

    # Market odds guardrail: le modèle ne peut pas s'éloigner de plus de 2× la proba
    # no-vig du marché dans AUCUN sens. Protège contre les ELOs WTA aberrants et les
    # profils manquants qui génèrent des probabilités inversées ou extrêmes.
    # Exemples : Kenin @4.07 (marché ~24%) → max 48% ; Sakkari @1.72 (marché ~58%) → min 29%.
    if odd_p1 is not None and odd_p1 > 1.0 and odd_p2 is not None and odd_p2 > 1.0:
        total_implied = 1.0 / odd_p1 + 1.0 / odd_p2
        novid_p1 = (1.0 / odd_p1) / total_implied   # no-vig market prob for p1
        cal_prob = min(cal_prob, novid_p1 * 2.0)     # model can't exceed 2× market
        cal_prob = max(cal_prob, novid_p1 * 0.5)     # model can't be below ½ market

    prob_p2 = 1 - cal_prob

    # No-vig implied probabilities (remove bookmaker margin before comparing)
    # If both odds available: novid_p1 = (1/odd_p1) / (1/odd_p1 + 1/odd_p2)
    # If only one odd available: fallback to raw 1/odd
    if odd_p1 is not None and odd_p1 > 1.0 and odd_p2 is not None and odd_p2 > 1.0:
        total_implied = 1.0 / odd_p1 + 1.0 / odd_p2
        novid_p1 = (1.0 / odd_p1) / total_implied
        novid_p2 = (1.0 / odd_p2) / total_implied
    else:
        novid_p1 = (1.0 / odd_p1) if odd_p1 and odd_p1 > 1.0 else None
        novid_p2 = (1.0 / odd_p2) if odd_p2 and odd_p2 > 1.0 else None

    def _odds_discount(odd: float) -> float:
        """Variance penalty on Kelly: high odds = high variance = smaller recommended stake.
        Applied on top of the standard quarter-Kelly already in use.
        ≥10: no recommendation (psychologically unsustainable, model unreliable at these odds).
        ≥6 : 1/4 factor → effectively 1/16 Kelly.
        ≥3 : 1/2 factor → effectively 1/8 Kelly.
        <3  : no extra discount (quarter-Kelly as normal).
        """
        if odd >= 10.0: return 0.0
        if odd >= 6.0:  return 0.25
        if odd >= 3.0:  return 0.5
        return 1.0

    # Seuil de direction marché : ne pas signaler de value bet si le bookmaker
    # donne < 40% à ce joueur (cote > ~2.5). Le modèle surévalue les outsiders
    # extrêmes faute de signal Pinnacle suffisant en entraînement.
    MIN_BK_DIR_PROB = 0.40

    edge_p1 = ev_p1 = kelly_frac_p1 = kelly_eur_p1 = None
    if odd_p1 is not None and odd_p1 > 1.0 and novid_p1 is not None:
        edge_p1 = round(cal_prob - novid_p1, 4)
        ev_p1   = round(cal_prob * (odd_p1 - 1) - (1 - cal_prob), 4)
        discount = _odds_discount(odd_p1)
        if edge_p1 > 0 and discount > 0 and novid_p1 >= MIN_BK_DIR_PROB:
            raw_kelly = (cal_prob * odd_p1 - 1) / (odd_p1 - 1)
            if raw_kelly > 0:
                kelly_frac_p1 = round(min(raw_kelly * kelly_fraction * discount, 0.05), 4)
                kelly_eur_p1  = round(kelly_frac_p1 * bankroll, 2)

    edge_p2 = ev_p2 = kelly_frac_p2 = kelly_eur_p2 = None
    if odd_p2 is not None and odd_p2 > 1.0 and novid_p2 is not None:
        edge_p2 = round(prob_p2 - novid_p2, 4)
        ev_p2   = round(prob_p2 * (odd_p2 - 1) - (1 - prob_p2), 4)
        discount2 = _odds_discount(odd_p2)
        if edge_p2 > 0 and discount2 > 0 and novid_p2 >= MIN_BK_DIR_PROB:
            raw_kelly2 = (prob_p2 * odd_p2 - 1) / (odd_p2 - 1)
            if raw_kelly2 > 0:
                kelly_frac_p2 = round(min(raw_kelly2 * kelly_fraction * discount2, 0.05), 4)
                kelly_eur_p2  = round(kelly_frac_p2 * bankroll, 2)

    # Data quality: based on how much recent match data we have for both players.
    # This determines ELO vs XGBoost blend weight AND is used to calibrate badge thresholds.
    # Use 14d for high/medium split, but extend to 21d for medium/low:
    # normal inter-tournament gap (e.g. Monte Carlo → Madrid) is 15-20 days,
    # so players who last played 15-21 days ago are NOT data-poor — use 'medium' not 'low'.
    min_m14 = min(
        _v(p1, 'matches_14d', 0.0) if p1_found else 0.0,
        _v(p2, 'matches_14d', 0.0) if p2_found else 0.0,
    ) if not elo_only else 0.0
    min_m21 = min(
        _v(p1, 'matches_21d', 0.0) if p1_found else 0.0,
        _v(p2, 'matches_21d', 0.0) if p2_found else 0.0,
    ) if not elo_only else 0.0
    if min_m14 >= 3:
        data_quality = 'high'    # 70% XGBoost → edge estimates are reliable
    elif min_m14 >= 1 or min_m21 >= 1:
        data_quality = 'medium'  # moderate reliability (incl. players 15-21d inactive)
    else:
        data_quality = 'low'     # 70% ELO → no recent data, suppress signals

    return {
        'prob_p1':      round(cal_prob, 4),
        'prob_p2':      round(prob_p2, 4),
        'elo_prob':     round(elo_prob, 4),
        'edge_p1':      edge_p1,
        'edge_p2':      edge_p2,
        'ev_p1':        ev_p1,
        'ev_p2':        ev_p2,
        'kelly_frac_p1': kelly_frac_p1,
        'kelly_frac_p2': kelly_frac_p2,
        'kelly_eur_p1': kelly_eur_p1,
        'kelly_eur_p2': kelly_eur_p2,
        # Backward compat
        'edge':         edge_p1,
        'ev':           ev_p1,
        'kelly_frac':   kelly_frac_p1,
        'kelly_eur':    kelly_eur_p1,
        'p1_found':     p1_found,
        'p2_found':     p2_found,
        'elo_only':     elo_only,
        'confidence':   round(cal_prob, 4),
        'data_quality': data_quality,
        'min_matches_14d': int(min_m14),
    }
