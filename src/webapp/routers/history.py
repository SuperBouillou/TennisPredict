"""Router — Bet history."""
from __future__ import annotations

import csv
import io
from pathlib import Path

from fastapi import APIRouter, Request, Form, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from src.webapp.db import get_bankroll, list_bets, resolve_bet

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")


def _state():
    from src.webapp.main import APP_STATE
    return APP_STATE


@router.get("/history", response_class=HTMLResponse)
async def history_page(
    request: Request,
    tour: str = "atp",
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

    return templates.TemplateResponse("history.html", {
        "request": request, "active": "history",
        "tour": tour, "pending": pending, "resolved": resolved,
        "bankroll_atp": get_bankroll(db, 'atp'),
        "bankroll_wta": get_bankroll(db, 'wta'),
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
    bankroll_atp = get_bankroll(db, 'atp')
    bankroll_wta = get_bankroll(db, 'wta')
    return HTMLResponse(
        f'<div class="card" style="color:var(--green)">✅ Résolu. '
        f'ATP: {bankroll_atp:.0f}€ · WTA: {bankroll_wta:.0f}€</div>'
    )


@router.get("/history/export")
async def export_csv(tour: str = "atp"):
    db = _state()['db']
    bets = list_bets(db, tour=tour, limit=10000)
    output = io.StringIO()
    if bets:
        writer = csv.DictWriter(output, fieldnames=bets[0].keys())
        writer.writeheader()
        writer.writerows(bets)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=bets_{tour}.csv"},
    )
