"""Tests for autocomplete endpoint."""
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import numpy as np
import sqlite3


@pytest.fixture(scope="module")
def client_ac():
    import src.webapp.main as main_module

    profiles = pd.DataFrame([
        {'player_name': 'Jannik Sinner',         'name_key': 'jannik sinner',          'rank': 1,   'elo': 2200.0},
        {'player_name': 'Carlos Alcaraz',         'name_key': 'carlos alcaraz',         'rank': 2,   'elo': 2180.0},
        {'player_name': 'Rafael Nadal',           'name_key': 'rafael nadal',           'rank': 300, 'elo': 1900.0},
        {'player_name': 'Tommy Müller',           'name_key': 'tommy müller',           'rank': 50,  'elo': 2050.0},
        {'player_name': 'Roberto Bautista Agut',  'name_key': 'roberto bautista agut',  'rank': 35,  'elo': 2060.0},
    ])

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
                'profiles': profiles, 'players': pd.DataFrame(),
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


def test_ac_basic(client_ac):
    r = client_ac.get("/predictions/autocomplete?q=sin&tour=atp&field=p1")
    assert r.status_code == 200
    assert 'Sinner' in r.text


def test_ac_accent(client_ac):
    # The name_key for 'Tommy Müller' is 'tommy müller' — search with accented ü
    r = client_ac.get("/predictions/autocomplete?q=tommy+m%C3%BC&tour=atp&field=p1")
    assert r.status_code == 200
    assert 'Müller' in r.text


def test_ac_compound_name(client_ac):
    r = client_ac.get("/predictions/autocomplete?q=baut&tour=atp&field=p1")
    assert r.status_code == 200
    assert 'Bautista' in r.text


def test_ac_case_insensitive(client_ac):
    r = client_ac.get("/predictions/autocomplete?q=ALCARAZ&tour=atp&field=p1")
    assert r.status_code == 200
    assert 'Alcaraz' in r.text


def test_ac_min_2_chars_empty(client_ac):
    r = client_ac.get("/predictions/autocomplete?q=a&tour=atp&field=p1")
    assert r.status_code == 200
    assert r.text.strip() == ''
