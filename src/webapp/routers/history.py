"""Router — Bet history."""
from __future__ import annotations

import csv
import io
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Request, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from src.webapp.db import get_bankroll, list_bets, resolve_bet, delete_bet, delete_resolved_bet, clear_bets, auto_resolve_pending
from src.webapp.state import get_state

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")

# Simple in-memory cache for the P&L time series (invalidated on any bet resolution/deletion)
_pnl_cache: dict[str, tuple[dict, float]] = {}  # key → (data, timestamp)
_PNL_TTL = 30  # seconds

# Cooldown for auto-resolve: at most one ESPN fetch per N seconds across all sessions
_AUTO_RESOLVE_COOLDOWN = 300  # 5 minutes
_auto_resolve_last_run: float = 0.0


def _invalidate_pnl_cache() -> None:
    _pnl_cache.clear()


_SRC = str(Path(__file__).resolve().parents[3] / 'src')


def _try_auto_resolve(db) -> int:
    """
    Fetch ESPN results for the last 3 days and auto-resolve pending bets.
    Runs at most once every _AUTO_RESOLVE_COOLDOWN seconds to avoid blocking
    every history page load with 6 synchronous ESPN HTTP requests.
    Returns total number of bets resolved across ATP + WTA.
    """
    global _auto_resolve_last_run
    if time.time() - _auto_resolve_last_run < _AUTO_RESOLVE_COOLDOWN:
        return 0

    if str(_SRC) not in sys.path:
        sys.path.insert(0, _SRC)
    try:
        from espn_client import fetch_results  # noqa: PLC0415
    except Exception:
        return 0

    _auto_resolve_last_run = time.time()
    today = date.today()
    total = 0
    for tour in ('atp', 'wta'):
        all_results: list[dict] = []
        for delta in (1, 2, 3):
            target = today - timedelta(days=delta)
            try:
                all_results.extend(fetch_results(tour, target))
            except Exception:
                pass
        if all_results:
            total += auto_resolve_pending(db, tour, all_results)
    return total


def _badge_from_edge(edge) -> str:
    """Approximate badge tier from stored edge value."""
    try:
        e = float(edge) if edge else 0.0
    except (TypeError, ValueError):
        e = 0.0
    if e >= 0.10:
        return 'value'
    if e >= 0.05:
        return 'edge'
    return 'neutral'


def _compute_stats(bets: list[dict]) -> dict:
    """Compute performance stats from a list of resolved bets."""
    resolved = [b for b in bets if b['status'] != 'pending']
    if not resolved:
        return {"win_rate": None, "roi": None, "streak": 0, "streak_type": None,
                "avg_odds": None, "total_staked": 0, "num_bets": 0, "num_won": 0,
                "badge_stats": {}}

    num_won = sum(1 for b in resolved if b['status'] == 'won')
    total_staked = sum(b['stake'] for b in resolved)
    pnl = sum(b['pnl'] for b in resolved)
    avg_odds = sum(b['odd'] for b in resolved) / len(resolved)

    # Current streak — sort by resolved_at so the last resolved bet is first
    by_resolved = sorted(resolved, key=lambda b: b.get('resolved_at') or '', reverse=True)
    streak, streak_type = 0, None
    for b in by_resolved:
        if streak == 0:
            streak_type = b['status']
            streak = 1
        elif b['status'] == streak_type:
            streak += 1
        else:
            break

    # Badge breakdown (approximate, based on stored edge)
    badge_stats: dict[str, dict] = {
        'value':   {'total': 0, 'won': 0, 'pnl': 0.0},
        'edge':    {'total': 0, 'won': 0, 'pnl': 0.0},
        'neutral': {'total': 0, 'won': 0, 'pnl': 0.0},
    }
    for b in resolved:
        bg = _badge_from_edge(b.get('edge'))
        badge_stats[bg]['total'] += 1
        badge_stats[bg]['pnl'] += b.get('pnl', 0.0)
        if b['status'] == 'won':
            badge_stats[bg]['won'] += 1
    for label, g in badge_stats.items():
        g['win_rate'] = round(g['won'] / g['total'] * 100) if g['total'] else 0
        staked = sum(b['stake'] for b in resolved if _badge_from_edge(b.get('edge')) == label)
        g['roi'] = round(g['pnl'] / staked * 100, 1) if staked > 0 else 0
        g['pnl'] = round(g['pnl'], 2)

    # Drawdown and best winning streak (consecutive P&L)
    by_chrono = sorted(resolved, key=lambda b: b.get('resolved_at') or b.get('created_at') or '')
    max_dd = 0.0; dd_count = 0; best_run = 0.0; run_count = 0
    cur_loss = 0.0; cur_loss_n = 0; cur_win = 0.0; cur_win_n = 0
    for b in by_chrono:
        p = b.get('pnl', 0) or 0
        if b['status'] == 'lost':
            cur_loss += p; cur_loss_n += 1; cur_win = 0; cur_win_n = 0
            if abs(cur_loss) > abs(max_dd): max_dd = cur_loss; dd_count = cur_loss_n
        else:
            cur_win += p; cur_win_n += 1; cur_loss = 0; cur_loss_n = 0
            if cur_win > best_run: best_run = cur_win; run_count = cur_win_n

    return {
        "win_rate": round(num_won / len(resolved) * 100, 1),
        "roi": round(pnl / total_staked * 100, 1) if total_staked > 0 else 0,
        "streak": streak,
        "streak_type": streak_type,
        "avg_odds": round(avg_odds, 2),
        "total_staked": round(total_staked, 2),
        "num_bets": len(resolved),
        "num_won": num_won,
        "badge_stats": badge_stats,
        "drawdown": round(max_dd, 2),
        "drawdown_count": dd_count,
        "best_streak_pnl": round(best_run, 2),
        "best_streak_count": run_count,
    }


@router.get("/history", response_class=HTMLResponse)
async def history_page(
    request: Request,
    tour: str | None = Query(default=None),  # None = all tours
    surface: str | None = None,
    status: str | None = None,
    period: str | None = None,  # "7j" | "mois" | None (all)
):
    db = get_state()['db']

    # Auto-resolve pending bets against ESPN results (last 3 days)
    auto_resolved = _try_auto_resolve(db)

    # Fetch all bets — no pagination, history is small enough to show in full
    all_bets = list_bets(db, tour=tour, surface=surface, status=status, limit=5000)

    # Optional period filter (applied in Python since list is already in memory)
    if period == "7j":
        cutoff = (date.today() - timedelta(days=7)).isoformat()
        all_bets = [b for b in all_bets if (b.get('created_at') or '') >= cutoff]
    elif period == "mois":
        cutoff = date.today().replace(day=1).isoformat()
        all_bets = [b for b in all_bets if (b.get('created_at') or '') >= cutoff]

    pending  = [b for b in all_bets if b['status'] == 'pending']
    resolved = [b for b in all_bets if b['status'] != 'pending']

    pnl_total = sum(b['pnl'] for b in resolved)
    stats = _compute_stats(all_bets)

    return templates.TemplateResponse(request, "history.html", {
        "active": "history",
        "tour": tour,  # None = tous
        "pending": pending, "resolved": resolved,
        "bankroll": get_bankroll(db),
        "pnl_total": round(pnl_total, 2),
        "stats": stats,
        "surface_filter": surface,
        "status_filter": status,
        "period_filter": period,
        "auto_resolved": auto_resolved,
    })


@router.post("/bets/{bet_id}/resolve", response_class=HTMLResponse)
async def resolve(request: Request, bet_id: int, outcome: Literal['won', 'lost'] = Form(...)):
    db = get_state()['db']
    try:
        resolve_bet(db, bet_id, outcome)
    except ValueError as e:
        return HTMLResponse(
            f'<div class="card" style="color:var(--red)">{e}</div>',
            status_code=400,
        )
    _invalidate_pnl_cache()
    # HX-Refresh reloads the full page so stats bar, P&L, bankroll and chart all update
    return HTMLResponse("", headers={"HX-Refresh": "true"})


@router.post("/bets/{bet_id}/delete", response_class=HTMLResponse)
async def delete(request: Request, bet_id: int):
    db = get_state()['db']
    try:
        delete_bet(db, bet_id)
    except ValueError as e:
        return HTMLResponse(
            f'<div class="card" style="color:var(--red)">{e}</div>',
            status_code=400,
        )
    _invalidate_pnl_cache()
    return HTMLResponse("", headers={"HX-Refresh": "true"})


@router.post("/bets/{bet_id}/delete-resolved", response_class=HTMLResponse)
async def delete_resolved(request: Request, bet_id: int):
    db = get_state()['db']
    try:
        delete_resolved_bet(db, bet_id)
    except ValueError as e:
        return HTMLResponse(
            f'<tr><td colspan="7" style="color:var(--red)">{e}</td></tr>',
            status_code=400,
        )
    _invalidate_pnl_cache()
    return HTMLResponse("", headers={"HX-Refresh": "true"})


@router.post("/bets/clear", response_class=HTMLResponse)
async def clear_history(request: Request, tour: str = Form("")):
    db = get_state()['db']
    n = clear_bets(db, tour=tour or None)
    _invalidate_pnl_cache()
    return HTMLResponse(
        f'<div class="card" style="color:var(--muted);text-align:center;padding:20px">'
        f'{n} paris supprimés.</div>'
    )


@router.get("/history/pnl-data")
async def pnl_data(tour: str | None = Query(default=None)):
    """Return cumulative P&L time series for Chart.js equity curve."""
    cache_key = tour or "all"
    cached = _pnl_cache.get(cache_key)
    if cached and (time.time() - cached[1]) < _PNL_TTL:
        return JSONResponse(cached[0])

    db = get_state()['db']
    if tour:
        rows = db.execute(
            "SELECT resolved_at, pnl FROM bets WHERE tour=? AND status!='pending' ORDER BY resolved_at",
            (tour,),
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT resolved_at, pnl FROM bets WHERE status!='pending' ORDER BY resolved_at",
        ).fetchall()
    cumul = 0.0
    points = []
    for r in rows:
        cumul += r["pnl"]
        points.append({"date": r["resolved_at"][:10], "pnl": round(cumul, 2)})
    result = {"points": points}
    _pnl_cache[cache_key] = (result, time.time())
    return JSONResponse(result)


@router.get("/history/export")
async def export_csv(tour: str | None = Query(default=None)):
    db = get_state()['db']
    bets = list_bets(db, tour=tour, limit=10000)
    output = io.StringIO()
    if bets:
        writer = csv.DictWriter(output, fieldnames=bets[0].keys())
        writer.writeheader()
        writer.writerows(bets)
    filename = f"bets_{tour or 'all'}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
