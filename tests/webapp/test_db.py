import pytest
import sqlite3
from pathlib import Path
from src.webapp import db


@pytest.fixture
def conn(tmp_path):
    c = db.get_connection(tmp_path / "test.db")
    db.init_db(c)
    return c


def _bet(tour='atp'):
    return {
        'tour': tour, 'tournament': 'Australian Open', 'surface': 'Hard',
        'round': 'R64', 'p1_name': 'Sinner J.', 'p2_name': 'Alcaraz C.',
        'bet_on': 'p1', 'prob': 0.67, 'edge': 0.05, 'odd': 1.75,
        'stake': 50.0, 'kelly_frac': 0.08,
    }


def test_init_creates_tables(conn):
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {'bets', 'bankroll', 'settings'} <= tables


def test_get_bankroll_default(conn):
    assert db.get_bankroll(conn, 'atp') == 1000.0


def test_set_and_get_bankroll(conn):
    db.set_bankroll(conn, 'atp', 1250.0)
    assert db.get_bankroll(conn, 'atp') == 1250.0


def test_add_bet_debits_bankroll(conn):
    bet_id = db.add_bet(conn, _bet())
    assert bet_id > 0
    assert db.get_bankroll(conn, 'atp') == pytest.approx(950.0)


def test_resolve_bet_won(conn):
    bet_id = db.add_bet(conn, _bet())
    # bankroll = 950 after debit
    db.resolve_bet(conn, bet_id, 'won')
    # won: bankroll += stake + profit = 950 + 50 + 50*(1.75-1) = 950 + 50 + 37.5 = 1037.5
    assert db.get_bankroll(conn, 'atp') == pytest.approx(1037.5)
    bet = db.get_bet(conn, bet_id)
    assert bet['status'] == 'won'
    assert bet['pnl'] == pytest.approx(37.5)


def test_resolve_bet_lost(conn):
    bet_id = db.add_bet(conn, _bet())
    db.resolve_bet(conn, bet_id, 'lost')
    # bankroll stays at 950 (stake already debited, no change on loss)
    assert db.get_bankroll(conn, 'atp') == pytest.approx(950.0)
    bet = db.get_bet(conn, bet_id)
    assert bet['status'] == 'lost'
    assert bet['pnl'] == pytest.approx(-50.0)


def test_resolve_already_resolved_raises(conn):
    bet_id = db.add_bet(conn, _bet())
    db.resolve_bet(conn, bet_id, 'won')
    with pytest.raises(ValueError, match="already resolved"):
        db.resolve_bet(conn, bet_id, 'lost')


def test_get_setting_default(conn):
    assert db.get_setting(conn, 'min_edge', '0.03') == '0.03'


def test_set_and_get_setting(conn):
    db.set_setting(conn, 'min_edge', '0.05')
    assert db.get_setting(conn, 'min_edge', '0.03') == '0.05'


def test_list_bets_filter_by_tour(conn):
    db.add_bet(conn, _bet('atp'))
    db.set_bankroll(conn, 'wta', 1000.0)
    db.add_bet(conn, {**_bet('wta'), 'tour': 'wta'})
    atp_bets = db.list_bets(conn, tour='atp')
    assert len(atp_bets) == 1
    assert atp_bets[0]['tour'] == 'atp'
