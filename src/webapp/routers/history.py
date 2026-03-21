"""Router — Bet history."""
from __future__ import annotations

import csv
import io
from pathlib import Path

from fastapi import APIRouter, Request, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from src.webapp.db import get_bankroll, list_bets, resolve_bet, delete_bet, clear_bets

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")


def _state():
    from src.webapp.main import APP_STATE
    return APP_STATE


@router.get("/history", response_class=HTMLResponse)
async def history_page(
    request: Request,
    tour: str | None = Query(default=None),  # None = all tours
    surface: str | None = None,
    status: str | None = None,
    page: int = 1,
):
    db = _state()['db']
    offset = (page - 1) * 20
    all_bets = list_bets(db, tour=tour, surface=surface, status=status, limit=20, offset=offset)
    pending  = [b for b in all_bets if b['status'] == 'pending']
    resolved = [b for b in all_bets if b['status'] != 'pending']

    # P&L total (all resolved bets, no limit)
    all_resolved = list_bets(db, tour=tour, status=None, limit=5000)
    pnl_total = sum(b['pnl'] for b in all_resolved if b['status'] != 'pending')

    return templates.TemplateResponse(request, "history.html", {
        "active": "history",
        "tour": tour,  # None = tous
        "pending": pending, "resolved": resolved,
        "bankroll": get_bankroll(db),
        "pnl_total": round(pnl_total, 2),
        "page": page,
        "surface_filter": surface,
        "status_filter": status,
    })


@router.post("/bets/{bet_id}/resolve", response_class=HTMLResponse)
async def resolve(request: Request, bet_id: int, outcome: str = Form(...)):
    db = _state()['db']
    try:
        resolve_bet(db, bet_id, outcome)
    except ValueError as e:
        return HTMLResponse(
            f'<div class="card" style="color:var(--red)">{e}</div>',
            status_code=400,
        )
    bankroll = get_bankroll(db)
    return HTMLResponse(
        f'<div class="card" style="color:var(--green)">✅ Résolu. Bankroll: {bankroll:.0f}€</div>'
    )


@router.post("/bets/{bet_id}/delete", response_class=HTMLResponse)
async def delete(request: Request, bet_id: int):
    db = _state()['db']
    try:
        delete_bet(db, bet_id)
    except ValueError as e:
        return HTMLResponse(
            f'<div class="card" style="color:var(--red)">{e}</div>',
            status_code=400,
        )
    return HTMLResponse("")  # HTMX removes the element


@router.post("/bets/clear", response_class=HTMLResponse)
async def clear_history(request: Request, tour: str = Form("")):
    db = _state()['db']
    n = clear_bets(db, tour=tour or None)
    return HTMLResponse(
        f'<div class="card" style="color:var(--muted);text-align:center;padding:20px">'
        f'{n} paris supprimés.</div>'
    )


@router.get("/history/pnl-data")
async def pnl_data(tour: str | None = Query(default=None)):
    """Return cumulative P&L time series for Chart.js equity curve."""
    db = _state()['db']
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
    return JSONResponse({"points": points})


@router.get("/history/export")
async def export_csv(tour: str | None = Query(default=None)):
    db = _state()['db']
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
