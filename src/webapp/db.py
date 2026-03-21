"""SQLite helpers — bets, bankroll, settings."""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DB = ROOT / "data" / "tennis_predict.db"
_SCHEMA = Path(__file__).parent / "migrations" / "001_init.sql"


def get_connection(path: Path = _DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA.read_text())
    conn.commit()
    conn.execute(
        "INSERT OR IGNORE INTO bankroll (tour, amount, updated_at) VALUES ('global', 1000.0, ?)",
        (_now(),),
    )
    defaults = {'min_edge': '0.03', 'min_prob': '0.55', 'kelly_fraction': '0.25'}
    for k, v in defaults.items():
        conn.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (k, v))
    conn.commit()


# ── Bankroll ──────────────────────────────────────────────────────────────────

def get_bankroll(conn: sqlite3.Connection, tour: str = 'global') -> float:
    row = conn.execute("SELECT amount FROM bankroll WHERE tour = ?", (tour,)).fetchone()
    return row['amount'] if row else 1000.0


def set_bankroll(conn: sqlite3.Connection, tour: str = 'global', amount: float = 1000.0) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO bankroll (tour, amount, updated_at) VALUES (?, ?, ?)",
        (tour, round(amount, 2), _now()),
    )
    conn.commit()


# ── Bets ──────────────────────────────────────────────────────────────────────

def add_bet(conn: sqlite3.Connection, bet: dict) -> int:
    """Insert bet and debit stake from bankroll. Returns new bet id."""
    cur = conn.execute(
        """INSERT INTO bets
           (tour, created_at, tournament, surface, round, p1_name, p2_name,
            bet_on, prob, edge, odd, stake, kelly_frac, status, pnl)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,'pending',0)""",
        (bet['tour'], _now(), bet['tournament'], bet['surface'], bet.get('round'),
         bet['p1_name'], bet['p2_name'], bet['bet_on'], bet['prob'],
         bet.get('edge'), bet['odd'], bet['stake'], bet.get('kelly_frac')),
    )
    current = get_bankroll(conn, bet['tour'])
    set_bankroll(conn, bet['tour'], current - bet['stake'])
    conn.commit()
    return cur.lastrowid


def get_bet(conn: sqlite3.Connection, bet_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM bets WHERE id = ?", (bet_id,)).fetchone()
    return dict(row) if row else None


def resolve_bet(conn: sqlite3.Connection, bet_id: int, outcome: str) -> None:
    """outcome: 'won' | 'lost'"""
    bet = get_bet(conn, bet_id)
    if bet is None:
        raise ValueError(f"Bet {bet_id} not found")
    if bet['status'] != 'pending':
        raise ValueError(f"Bet {bet_id} already resolved (status={bet['status']})")

    if outcome == 'won':
        profit = round(bet['stake'] * (bet['odd'] - 1), 2)
        pnl = profit
        current = get_bankroll(conn, bet['tour'])
        set_bankroll(conn, bet['tour'], current + bet['stake'] + profit)
    else:
        pnl = -bet['stake']

    conn.execute(
        "UPDATE bets SET status=?, pnl=?, resolved_at=? WHERE id=?",
        (outcome, pnl, _now(), bet_id),
    )
    conn.commit()


def delete_bet(conn: sqlite3.Connection, bet_id: int) -> None:
    """Delete a pending bet and refund its stake to bankroll."""
    bet = get_bet(conn, bet_id)
    if bet is None:
        raise ValueError(f"Bet {bet_id} not found")
    if bet['status'] != 'pending':
        raise ValueError(f"Bet {bet_id} is already resolved — cannot delete")
    current = get_bankroll(conn)
    set_bankroll(conn, amount=current + bet['stake'])
    conn.execute("DELETE FROM bets WHERE id = ?", (bet_id,))
    conn.commit()


def clear_bets(conn: sqlite3.Connection, tour: str | None = None) -> int:
    """Delete all bets (optionally filtered by tour). Returns number deleted."""
    if tour:
        cur = conn.execute("DELETE FROM bets WHERE tour = ?", (tour,))
    else:
        cur = conn.execute("DELETE FROM bets")
    conn.commit()
    return cur.rowcount


def list_bets(
    conn: sqlite3.Connection,
    tour: str | None = None,
    status: str | None = None,
    surface: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[dict]:
    clauses, params = [], []
    if tour:
        clauses.append("tour = ?"); params.append(tour)
    if status:
        clauses.append("status = ?"); params.append(status)
    if surface:
        clauses.append("surface = ?"); params.append(surface)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = conn.execute(
        f"SELECT * FROM bets {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
        params + [limit, offset],
    ).fetchall()
    return [dict(r) for r in rows]


# ── Settings ──────────────────────────────────────────────────────────────────

def get_setting(conn: sqlite3.Connection, key: str, default: str = '') -> str:
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row['value'] if row else default


def set_setting(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
    conn.commit()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
