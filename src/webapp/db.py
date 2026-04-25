"""SQLite helpers — bets, bankroll, settings, signal_log."""
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


# ── Signal log (track record automatique) ────────────────────────────────────

def log_signal(conn: sqlite3.Connection, signal: dict) -> int | None:
    """
    Insère un signal VALUE dans signal_log si pas encore présent pour ce match/côté.
    Retourne l'id inséré ou None si doublon.
    """
    existing = conn.execute(
        """SELECT id FROM signal_log
           WHERE tour=? AND p1_name=? AND p2_name=? AND bet_on=?
             AND date(created_at)=date(?)""",
        (signal['tour'], signal['p1_name'], signal['p2_name'],
         signal['bet_on'], _now()),
    ).fetchone()
    if existing:
        return None

    cur = conn.execute(
        """INSERT INTO signal_log
           (created_at, tour, tournament, surface, level, round,
            p1_name, p2_name, bet_on, prob_model, odd_snapshot, edge, stake_units)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,1.0)""",
        (_now(), signal['tour'], signal.get('tournament'), signal.get('surface'),
         signal.get('level'), signal.get('round'),
         signal['p1_name'], signal['p2_name'], signal['bet_on'],
         signal.get('prob_model'), signal.get('odd_snapshot'), signal.get('edge')),
    )
    conn.commit()
    return cur.lastrowid


def resolve_signals(conn: sqlite3.Connection, tour: str, results: list[dict]) -> int:
    """
    Résout les signaux pending à partir des résultats ESPN.
    results: liste de dicts avec p1_name (winner) et p2_name (loser).
    Retourne le nombre de signaux résolus.
    """
    pending = conn.execute(
        "SELECT * FROM signal_log WHERE tour=? AND result='pending'", (tour,)
    ).fetchall()
    if not pending or not results:
        return 0

    result_pairs: set[tuple[str, str]] = set()
    for r in results:
        w = r.get('p1_name', '').lower().strip()
        l = r.get('p2_name', '').lower().strip()
        if w and l:
            result_pairs.add((w, l))

    resolved = 0
    now = _now()
    for sig in pending:
        bet_on   = sig['bet_on'].lower().strip()
        p1_lower = sig['p1_name'].lower().strip()
        p2_lower = sig['p2_name'].lower().strip()
        opponent = p2_lower if bet_on == p1_lower else p1_lower
        odd      = sig['odd_snapshot'] or 2.0

        if (bet_on, opponent) in result_pairs:
            pnl = round(odd - 1, 4)   # gain : (cote - 1) × 1 unité
            conn.execute(
                "UPDATE signal_log SET result='won', pnl_units=?, resolved_at=? WHERE id=?",
                (pnl, now, sig['id']),
            )
            resolved += 1
        elif (opponent, bet_on) in result_pairs:
            conn.execute(
                "UPDATE signal_log SET result='lost', pnl_units=-1.0, resolved_at=? WHERE id=?",
                (now, sig['id']),
            )
            resolved += 1

    if resolved:
        conn.commit()
    return resolved


def get_signal_stats(conn: sqlite3.Connection,
                     tour: str | None = None) -> dict:
    """Retourne les KPIs du track record."""
    clauses = ["result != 'void'"]
    params: list = []
    if tour:
        clauses.append("tour = ?"); params.append(tour)

    base = f"SELECT * FROM signal_log WHERE {' AND '.join(clauses)}"
    rows = conn.execute(base, params).fetchall()

    total    = len(rows)
    pending  = sum(1 for r in rows if r['result'] == 'pending')
    resolved = [r for r in rows if r['result'] in ('won', 'lost')]
    won      = sum(1 for r in resolved if r['result'] == 'won')
    pnl      = sum(r['pnl_units'] or 0 for r in resolved)
    n_res    = len(resolved)
    win_rate = won / n_res if n_res else None
    roi      = pnl / n_res if n_res else None

    return {
        'total': total, 'pending': pending, 'resolved': n_res,
        'won': won, 'lost': n_res - won,
        'win_rate': round(win_rate, 4) if win_rate is not None else None,
        'roi': round(roi, 4) if roi is not None else None,
        'pnl_units': round(pnl, 2),
    }


def list_signals(conn: sqlite3.Connection,
                 tour: str | None = None,
                 limit: int = 50) -> list[dict]:
    """Retourne les derniers signaux, du plus récent au plus ancien."""
    q = "SELECT * FROM signal_log"
    params: list = []
    if tour:
        q += " WHERE tour = ?"; params.append(tour)
    q += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    return [dict(r) for r in conn.execute(q, params).fetchall()]


def get_signal_curve(conn: sqlite3.Connection,
                     tour: str | None = None) -> dict:
    """Retourne les données pour la courbe P&L cumulative."""
    clauses = ["result IN ('won','lost')"]
    params: list = []
    if tour:
        clauses.append("tour = ?"); params.append(tour)

    rows = conn.execute(
        f"SELECT resolved_at, pnl_units FROM signal_log "
        f"WHERE {' AND '.join(clauses)} ORDER BY resolved_at",
        params,
    ).fetchall()

    if not rows:
        return {'labels': [], 'values': []}

    labels, values, cumul = [], [], 0.0
    for r in rows:
        cumul += r['pnl_units'] or 0
        labels.append(r['resolved_at'][:10])
        values.append(round(cumul, 2))

    return {'labels': labels, 'values': values}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
