"""Tests for ml.py — prediction wrapper."""
import pytest
import numpy as np
from unittest.mock import MagicMock
from src.webapp import ml


@pytest.fixture
def mock_artifacts():
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.35, 0.65]])

    imputer = MagicMock()
    imputer.transform.side_effect = lambda x: x

    platt = MagicMock()
    platt.predict_proba.return_value = np.array([[0.33, 0.67]])

    features = ['elo_diff', 'rank_diff', 'form_win_rate_5_diff']

    import pandas as pd
    profiles = pd.DataFrame([{
        'player_name': 'Jannik Sinner',
        'name_key': 'jannik sinner',
        'rank': 1, 'rank_points': 11000,
        'elo': 2200.0, 'elo_hard': 2210.0, 'elo_clay': 2150.0, 'elo_grass': 2100.0,
        'form_win_rate_5': 0.8, 'form_win_rate_10': 0.75, 'form_win_rate_20': 0.72,
        'h2h_win_rate': 0.6,
    }, {
        'player_name': 'Carlos Alcaraz',
        'name_key': 'carlos alcaraz',
        'rank': 2, 'rank_points': 9500,
        'elo': 2180.0, 'elo_hard': 2190.0, 'elo_clay': 2200.0, 'elo_grass': 2160.0,
        'form_win_rate_5': 0.75, 'form_win_rate_10': 0.72, 'form_win_rate_20': 0.70,
        'h2h_win_rate': 0.5,
    }])
    return {
        'model': model, 'imputer': imputer, 'platt': platt,
        'feature_list': features, 'profiles': profiles,
    }


def test_predict_known_players(mock_artifacts):
    result = ml.predict(
        mock_artifacts,
        p1_name='Jannik Sinner', p2_name='Carlos Alcaraz',
        tournament='Australian Open', surface='Hard',
        round_='R64', best_of=3, odd_p1=None, odd_p2=None, bankroll=1000.0,
    )
    assert result['p1_found'] is True
    assert result['p2_found'] is True
    assert 0 < result['prob_p1'] < 1
    assert abs(result['prob_p1'] + result['prob_p2'] - 1.0) < 1e-6
    assert result['edge'] is None
    assert result['kelly_frac'] is None


def test_predict_with_odds(mock_artifacts):
    result = ml.predict(
        mock_artifacts,
        p1_name='Jannik Sinner', p2_name='Carlos Alcaraz',
        tournament='Australian Open', surface='Hard',
        round_='R64', best_of=3, odd_p1=1.50, odd_p2=2.50, bankroll=1000.0,
    )
    assert result['edge'] is not None
    assert result['kelly_frac'] is not None
    assert result['kelly_eur'] is not None


def test_predict_unknown_player_returns_elo_only(mock_artifacts):
    result = ml.predict(
        mock_artifacts,
        p1_name='Unknown Player', p2_name='Carlos Alcaraz',
        tournament='Roland Garros', surface='Clay',
        round_='QF', best_of=5, odd_p1=None, odd_p2=None, bankroll=1000.0,
    )
    assert result['p1_found'] is False
    assert result['elo_only'] is True


def test_kelly_capped_at_max_fraction(mock_artifacts):
    """Kelly fraction should never exceed 0.25."""
    mock_artifacts['platt'].predict_proba.return_value = np.array([[0.01, 0.99]])
    result = ml.predict(
        mock_artifacts,
        p1_name='Jannik Sinner', p2_name='Carlos Alcaraz',
        tournament='Wimbledon', surface='Grass',
        round_='F', best_of=5, odd_p1=10.0, odd_p2=1.02, bankroll=1000.0,
    )
    if result['kelly_frac'] is not None:
        assert result['kelly_frac'] <= 0.25


def test_prob_sums_to_one(mock_artifacts):
    result = ml.predict(
        mock_artifacts,
        p1_name='Jannik Sinner', p2_name='Carlos Alcaraz',
        tournament='US Open', surface='Hard',
        round_='SF', best_of=5, odd_p1=1.80, odd_p2=2.00, bankroll=500.0,
    )
    assert abs(result['prob_p1'] + result['prob_p2'] - 1.0) < 1e-6
