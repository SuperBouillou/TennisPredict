"""Integration tests for FastAPI routes."""
import pytest
import sqlite3
from unittest.mock import MagicMock
import numpy as np
import pandas as pd


@pytest.fixture(scope="module")
def client():
    """TestClient with mocked ML models and in-memory DB."""
    import src.webapp.main as main_module
    from fastapi.testclient import TestClient

    profiles = pd.DataFrame([{
        'player_name': 'Jannik Sinner', 'name_key': 'jannik sinner',
        'rank': 1, 'rank_points': 11000,
        'elo': 2200.0, 'elo_hard': 2210.0, 'elo_clay': 2150.0, 'elo_grass': 2100.0,
        'form_win_rate_5': 0.8, 'form_win_rate_10': 0.75, 'form_win_rate_20': 0.72,
        'h2h_win_rate': 0.6,
    }, {
        'player_name': 'Carlos Alcaraz', 'name_key': 'carlos alcaraz',
        'rank': 2, 'rank_points': 9500,
        'elo': 2180.0, 'elo_hard': 2190.0, 'elo_clay': 2200.0, 'elo_grass': 2160.0,
        'form_win_rate_5': 0.75, 'form_win_rate_10': 0.72, 'form_win_rate_20': 0.70,
        'h2h_win_rate': 0.5,
    }])

    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.35, 0.65]])
    model.feature_importances_ = np.array([0.5, 0.3, 0.2])

    imputer = MagicMock()
    imputer.transform.side_effect = lambda x: x

    platt = MagicMock()
    platt.predict_proba.return_value = np.array([[0.33, 0.67]])

    with TestClient(main_module.app, raise_server_exceptions=False) as c:
        # Set AFTER lifespan runs so it is not overwritten by startup
        main_module.APP_STATE['models'] = {
            'atp': {
                'model': model, 'imputer': imputer, 'platt': platt,
                'feature_list': ['elo_diff', 'rank_diff', 'form_win_rate_5_diff'],
                'profiles': profiles,
                'players': pd.DataFrame(),
            },
            'wta': None,
        }
        conn = sqlite3.connect(':memory:', check_same_thread=False)
        conn.row_factory = sqlite3.Row
        from src.webapp.db import init_db
        init_db(conn)
        main_module.APP_STATE['db'] = conn
        main_module.APP_STATE['sync_status'] = {'atp': 'idle', 'wta': 'idle'}
        yield c


def test_root_redirects(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)


def test_today_returns_200(client):
    r = client.get("/today")
    assert r.status_code == 200
    assert 'text/html' in r.headers['content-type']
    assert '<html' in r.text.lower()


def test_predictions_returns_200(client):
    r = client.get("/predictions")
    assert r.status_code == 200
    assert 'text/html' in r.headers['content-type']


def test_history_returns_200(client):
    r = client.get("/history")
    assert r.status_code == 200
    assert 'text/html' in r.headers['content-type']


def test_stats_returns_200(client):
    r = client.get("/stats")
    assert r.status_code == 200
    assert 'text/html' in r.headers['content-type']


def test_joueurs_returns_200(client):
    r = client.get("/joueurs")
    assert r.status_code == 200


def test_autocomplete_returns_fragment(client):
    r = client.get("/predictions/autocomplete?q=sin&tour=atp&field=p1")
    assert r.status_code == 200
    # Should be a fragment, NOT a full page
    assert '<html' not in r.text.lower()
    assert 'Sinner' in r.text


def test_autocomplete_min_2_chars(client):
    r = client.get("/predictions/autocomplete?q=s&tour=atp&field=p1")
    assert r.status_code == 200
    assert r.text.strip() == ''


def test_today_partial_is_fragment(client):
    r = client.get("/today/matches?tour=atp")
    assert r.status_code == 200
    assert '<html' not in r.text.lower()


def test_sync_status_is_fragment(client):
    r = client.get("/sync/status?tour=atp")
    assert r.status_code == 200
    assert '<html' not in r.text.lower()


def test_stats_equity_json(client):
    r = client.get("/stats/equity?tour=atp&strategy=Kelly")
    assert r.status_code == 200
    data = r.json()
    assert 'labels' in data and 'values' in data


def test_settings_save(client):
    r = client.post("/settings", data={
        "min_edge": "0.04", "min_prob": "0.57", "kelly_fraction": "0.20"
    })
    assert r.status_code == 200
    # The route returns '&#x2705; Seuils sauvegardés.'
    assert '&#x2705;' in r.text or 'sauveg' in r.text


def test_run_prediction_returns_fragment(client):
    r = client.post("/predictions/run", data={
        "tour": "atp",
        "p1_name": "Jannik Sinner",
        "p2_name": "Carlos Alcaraz",
        "tournament": "Australian Open",
        "surface": "Hard",
        "round": "R64",
        "best_of": "3",
    })
    assert r.status_code == 200
    assert '<html' not in r.text.lower()
    assert 'Sinner' in r.text or '%' in r.text


def test_resolve_already_resolved_returns_400(client):
    import src.webapp.main as m
    conn = m.APP_STATE['db']
    from src.webapp.db import add_bet, set_bankroll
    set_bankroll(conn, 'atp', 1000.0)
    bet_id = add_bet(conn, {
        'tour': 'atp', 'tournament': 'T', 'surface': 'Hard', 'round': 'R64',
        'p1_name': 'A', 'p2_name': 'B', 'bet_on': 'p1',
        'prob': 0.6, 'edge': 0.04, 'odd': 2.0, 'stake': 10.0, 'kelly_frac': 0.05,
    })
    r1 = client.post(f"/bets/{bet_id}/resolve", data={"outcome": "won"})
    assert r1.status_code == 200
    r2 = client.post(f"/bets/{bet_id}/resolve", data={"outcome": "lost"})
    assert r2.status_code == 400
