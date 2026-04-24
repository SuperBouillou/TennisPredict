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
    """Insert bet and debit stake from bankroll atomically. Returns new bet id."""
    try:
        cur = conn.execute(
            """INSERT INTO bets
               (tour, created_at, tournament, surface, round, p1_name, p2_name,
                bet_on, prob, edge, odd, stake, kelly_frac, status, pnl)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,'pending',0)""",
            (bet['tour'], _now(), bet['tournament'], bet['surface'], bet.get('round'),
             bet['p1_name'], bet['p2_name'], bet['bet_on'], bet['prob'],
             bet.get('edge'), bet['odd'], bet['stake'], bet.get('kelly_frac')),
        )
        current = get_bankroll(conn)
        conn.execute(
            "INSERT OR REPLACE INTO bankroll (tour, amount, updated_at) VALUES ('global', ?, ?)",
            (round(current - bet['stake'], 2), _now()),
        )
        conn.commit()
        return cur.lastrowid
    except Exception:
        conn.rollback()
        raise


def get_bet(conn: sqlite3.Connection, bet_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM bets WHERE id = ?", (bet_id,)).fetchone()
    return dict(row) if row else None


def resolve_bet(conn: sqlite3.Connection, bet_id: int, outcome: str) -> None:
    """outcome: 'won' | 'lost'"""
    if outcome not in ('won', 'lost'):
        raise ValueError(f"Invalid outcome '{outcome}' — must be 'won' or 'lost'")
    bet = get_bet(conn, bet_id)
    if bet is None:
        raise ValueError(f"Bet {bet_id} not found")
    if bet['status'] != 'pending':
        raise ValueError(f"Bet {bet_id} already resolved (status={bet['status']})")
    try:
        if outcome == 'won':
            profit = round(bet['stake'] * (bet['odd'] - 1), 2)
            pnl = profit
            current = get_bankroll(conn)
            conn.execute(
                "INSERT OR REPLACE INTO bankroll (tour, amount, updated_at) VALUES ('global', ?, ?)",
                (round(current + bet['stake'] + profit, 2), _now()),
            )
        else:
            pnl = -bet['stake']

        conn.execute(
            "UPDATE bets SET status=?, pnl=?, resolved_at=? WHERE id=?",
            (outcome, pnl, _now(), bet_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def delete_bet(conn: sqlite3.Connection, bet_id: int) -> None:
    """Delete a pending bet and refund its stake to bankroll."""
    bet = get_bet(conn, bet_id)
    if bet is None:
        raise ValueError(f"Bet {bet_id} not found")
    if bet['status'] != 'pending':
        raise ValueError(f"Bet {bet_id} is already resolved — cannot delete")
    try:
        current = get_bankroll(conn)
        conn.execute(
            "INSERT OR REPLACE INTO bankroll (tour, amount, updated_at) VALUES ('global', ?, ?)",
            (round(current + bet['stake'], 2), _now()),
        )
        conn.execute("DELETE FROM bets WHERE id = ?", (bet_id,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def delete_resolved_bet(conn: sqlite3.Connection, bet_id: int) -> None:
    """Delete a resolved bet and reverse its P&L impact on bankroll."""
    bet = get_bet(conn, bet_id)
    if bet is None:
        raise ValueError(f"Bet {bet_id} not found")
    if bet['status'] == 'pending':
        raise ValueError(f"Bet {bet_id} is pending — use delete_bet instead")
    try:
        current = get_bankroll(conn)
        if bet['status'] == 'won':
            new_amount = round(current - bet['stake'] - bet['pnl'], 2)
        else:
            new_amount = round(current + bet['stake'], 2)
        conn.execute(
            "INSERT OR REPLACE INTO bankroll (tour, amount, updated_at) VALUES ('global', ?, ?)",
            (new_amount, _now()),
        )
        conn.execute("DELETE FROM bets WHERE id = ?", (bet_id,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def auto_resolve_pending(conn: sqlite3.Connection, tour: str, results: list[dict]) -> int:
    """
    Auto-resolve pending bets using ESPN results.
    results: list of dicts with p1_name (winner), p2_name (loser) from fetch_results().
    Returns number of bets resolved.
    """
    pending = list_bets(conn, tour=tour, status='pending', limit=1000)
    if not pending or not results:
        return 0

    # Build lookup: (winner_lower, loser_lower) present in results
    result_pairs: set[tuple[str, str]] = set()
    for r in results:
        w = r.get('p1_name', '').lower().strip()  # p1_name = winner in fetch_results
        l = r.get('p2_name', '').lower().strip()  # p2_name = loser
        if w and l:
            result_pairs.add((w, l))

    resolved = 0
    for bet in pending:
        bet_on   = bet['bet_on'].lower().strip()
        p1_lower = bet['p1_name'].lower().strip()
        opponent = bet['p2_name'].lower().strip() if p1_lower == bet_on else p1_lower

        if (bet_on, opponent) in result_pairs:
            resolve_bet(conn, bet['id'], 'won')
            resolved += 1
        elif (opponent, bet_on) in result_pairs:
            resolve_bet(conn, bet['id'], 'lost')
            resolved += 1

    return resolved


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
    clauses: list[str] = []
    params: list = []
    if tour:
        clauses.append("tour = ?"); params.append(tour)
    if status:
        clauses.append("status = ?"); params.append(status)
    if surface:
        clauses.append("surface = ?"); params.append(surface)
    base = "SELECT * FROM bets"
    if clauses:
        base += " WHERE " + " AND ".join(clauses)
    base += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    rows = conn.execute(base, params + [limit, offset]).fetchall()
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
