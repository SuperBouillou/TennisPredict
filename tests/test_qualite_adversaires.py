# tests/test_qualite_adversaires.py
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

ROOT = Path(__file__).parent.parent


def make_ml_df(n_extra=0):
    """
    Dataset ML minimal (format p1/p2 avec index 0-based) pour les tests.
    n_extra : matchs supplémentaires à ajouter pour le joueur 1 (pour tester min_periods).
    """
    base = {
        'tourney_date': pd.to_datetime([
            '2020-01-06', '2020-01-13', '2020-01-20',
            '2020-01-27', '2020-02-03',
        ]),
        'tourney_level': ['G', 'G', 'M', 'M', 'A'],
        'surface':       ['Hard', 'Hard', 'Clay', 'Clay', 'Grass'],
        'p1_id':   [1, 1, 1, 2, 2],
        'p2_id':   [2, 3, 2, 1, 3],
        'target':  [1, 0, 1, 0, 1],
        'p1_rank': [5.0,  5.0,  5.0,  20.0, 20.0],
        'p2_rank': [20.0, 50.0, 20.0, 5.0,  80.0],
        'p1_rank_points': [3000, 3000, 3000, 500, 500],
        'p2_rank_points': [500,  400,  500,  3000, 200],
        'p1_age': [25.0, 25.0, 25.0, 28.0, 28.0],
        'p2_age': [28.0, 30.0, 28.0, 25.0, 22.0],
    }
    df = pd.DataFrame(base)

    if n_extra > 0:
        extras = pd.DataFrame({
            'tourney_date': pd.date_range('2020-02-10', periods=n_extra, freq='7D'),
            'tourney_level': ['A'] * n_extra,
            'surface': ['Hard'] * n_extra,
            'p1_id':   [1] * n_extra,
            'p2_id':   [4] * n_extra,
            'target':  [1] * n_extra,
            'p1_rank': [5.0] * n_extra,
            'p2_rank': [100.0] * n_extra,
            'p1_rank_points': [3000] * n_extra,
            'p2_rank_points': [300] * n_extra,
            'p1_age': [25.0] * n_extra,
            'p2_age': [26.0] * n_extra,
        })
        df = pd.concat([df, extras], ignore_index=True)

    # Index 0-based propre — précondition requise par build_player_match_history()
    return df.reset_index(drop=True)


# ── Tests build_player_match_history ──────────────────────────────────────────

def test_opponent_rank_column_exists():
    """opponent_rank doit être présente dans df_history."""
    from compute_rolling_features import build_player_match_history
    hist = build_player_match_history(make_ml_df())
    assert 'opponent_rank' in hist.columns


def test_opponent_rank_p1_view():
    """Vue p1 : opponent_rank doit être p2_rank du match."""
    from compute_rolling_features import build_player_match_history
    df = make_ml_df()
    hist = build_player_match_history(df)
    row = hist[(hist['player_id'] == 1) &
               (hist['tourney_date'] == pd.Timestamp('2020-01-06'))].iloc[0]
    assert row['opponent_rank'] == 20.0


def test_opponent_rank_p2_view():
    """Vue p2 : opponent_rank doit être p1_rank du match."""
    from compute_rolling_features import build_player_match_history
    df = make_ml_df()
    hist = build_player_match_history(df)
    row = hist[(hist['player_id'] == 2) &
               (hist['tourney_date'] == pd.Timestamp('2020-01-06'))].iloc[0]
    assert row['opponent_rank'] == 5.0


# ── Tests compute_rolling_stats ────────────────────────────────────────────────

def test_winrate_quality_columns_present():
    """winrate_quality_5/10/20 doivent apparaître dans les stats glissantes."""
    from compute_rolling_features import build_player_match_history, compute_rolling_stats
    hist = build_player_match_history(make_ml_df())
    hist = compute_rolling_stats(hist)
    for w in [5, 10, 20]:
        assert f'winrate_quality_{w}' in hist.columns, f"Colonne manquante : winrate_quality_{w}"


def test_winrate_quality_bounded():
    """winrate_quality doit être dans [0, 1] (win rate pondéré normalisé)."""
    from compute_rolling_features import build_player_match_history, compute_rolling_stats
    hist = build_player_match_history(make_ml_df(n_extra=5))
    hist = compute_rolling_stats(hist)
    col = hist['winrate_quality_10'].dropna()
    assert len(col) > 0, "Aucune valeur non-NaN produite"
    assert (col >= 0.0).all(), f"Valeurs < 0 trouvées : {col[col < 0]}"
    assert (col <= 1.0).all(), f"Valeurs > 1 trouvées : {col[col > 1]}"


def test_winrate_quality_no_leakage():
    """Test anti-leakage : vérifie que .shift(1) est bien appliqué."""
    from compute_rolling_features import build_player_match_history, compute_rolling_stats
    df = make_ml_df(n_extra=1)
    hist = build_player_match_history(df)
    hist = compute_rolling_stats(hist)

    p1 = hist[hist['player_id'] == 1].sort_values('tourney_date').reset_index(drop=True)
    assert pd.isna(p1.iloc[0]['winrate_quality_5']), \
        "Fuite détectée : le 1er match ne devrait avoir aucune valeur quality"
    assert not pd.isna(p1.iloc[3]['winrate_quality_5']), \
        "La 4ème ligne devrait avoir une valeur quality (3 matchs précédents)"


# ── Tests join_rolling_to_ml ──────────────────────────────────────────────────

def test_diff_columns_present():
    """winrate_quality_diff_5/10/20 doivent être dans le dataset ML après jointure."""
    from compute_rolling_features import (build_player_match_history,
                                          compute_rolling_stats, join_rolling_to_ml)
    df = make_ml_df(n_extra=5)
    hist = build_player_match_history(df)
    hist = compute_rolling_stats(hist)
    df_out = join_rolling_to_ml(df, hist)
    for w in [5, 10, 20]:
        assert f'winrate_quality_diff_{w}' in df_out.columns


def test_diff_equals_p1_minus_p2():
    """winrate_quality_diff_10 = p1_winrate_quality_10 - p2_winrate_quality_10."""
    from compute_rolling_features import (build_player_match_history,
                                          compute_rolling_stats, join_rolling_to_ml)
    df = make_ml_df(n_extra=5)
    hist = build_player_match_history(df)
    hist = compute_rolling_stats(hist)
    df_out = join_rolling_to_ml(df, hist)
    valid = df_out.dropna(subset=['p1_winrate_quality_10', 'p2_winrate_quality_10',
                                   'winrate_quality_diff_10'])
    if len(valid) > 0:
        expected = valid['p1_winrate_quality_10'] - valid['p2_winrate_quality_10']
        pd.testing.assert_series_equal(
            valid['winrate_quality_diff_10'].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False
        )


# ── Tests prepare_ml_dataset ──────────────────────────────────────────────────

def test_qualite_adversaires_group_defined():
    """Le groupe 'qualite_adversaires' doit exister dans define_feature_sets()."""
    from prepare_ml_dataset import define_feature_sets
    fs = define_feature_sets()
    assert 'qualite_adversaires' in fs, "Groupe 'qualite_adversaires' absent"


def test_qualite_adversaires_features_count():
    """Le groupe doit contenir exactement 9 features."""
    from prepare_ml_dataset import define_feature_sets
    fs = define_feature_sets()
    n = len(fs['qualite_adversaires'])
    assert n == 9, f"Attendu 9 features, trouvé {n}: {fs['qualite_adversaires']}"
