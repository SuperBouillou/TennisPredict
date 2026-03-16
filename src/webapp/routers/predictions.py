"""Router — Manual predictions."""
from __future__ import annotations

from pathlib import Path
from fastapi import APIRouter, Request, Form, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.webapp import ml as ml_module
from src.webapp.db import get_bankroll, add_bet

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")

_ROUNDS   = ['R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'F', 'RR']
_SURFACES = ['Hard', 'Clay', 'Grass']


def _state():
    from src.webapp.main import APP_STATE
    return APP_STATE


@router.get("/predictions", response_class=HTMLResponse)
async def predictions_page(request: Request, tour: str = "atp"):
    db = _state()['db']
    return templates.TemplateResponse("predictions.html", {
        "request": request, "active": "predictions",
        "tour": tour, "rounds": _ROUNDS, "surfaces": _SURFACES,
        "bankroll": get_bankroll(db, tour),
    })


@router.get("/predictions/autocomplete", response_class=HTMLResponse)
async def autocomplete(
    request: Request,
    q: str = "",
    tour: str = "atp",
    field: str = "p1",
):
    if len(q) < 2:
        return HTMLResponse("")
    artifacts = _state().get('models', {}).get(tour)
    if not artifacts:
        return HTMLResponse("")
    profiles = artifacts['profiles']
    mask = profiles['name_key'].str.contains(q.lower(), na=False, regex=False)
    results = (
        profiles[mask]
        .sort_values('rank', na_position='last')
        .head(8)[['player_name', 'rank']]
        .to_dict('records')
    )

    input_id   = f"{field}_input"
    list_id    = f"{field}-list"
    field_name = f"{field}_name"

    html = ""
    for r in results:
        name = r['player_name']
        rank = r.get('rank') or '—'
        try:
            rank_str = str(int(rank))
        except (ValueError, TypeError):
            rank_str = '—'
        safe_name = name.replace("'", "\\'")
        html += (
            f'<div class="autocomplete-item" '
            f'onclick="selectPlayer(\'{safe_name}\', \'{field_name}\', \'{input_id}\', \'{list_id}\')">'
            f'{name} <span style="color:var(--muted)">#{rank_str}</span></div>'
        )
    return HTMLResponse(html)


@router.post("/predictions/run", response_class=HTMLResponse)
async def run_prediction(
    request: Request,
    tour: str = Form(...),
    p1_name: str = Form(...),
    p2_name: str = Form(...),
    tournament: str = Form(...),
    surface: str = Form(...),
    round_: str = Form(..., alias="round"),
    best_of: int = Form(3),
    odd_p1: float | None = Form(None),
    odd_p2: float | None = Form(None),
):
    artifacts = _state().get('models', {}).get(tour)
    db = _state()['db']
    bankroll = get_bankroll(db, tour)

    if not artifacts:
        return HTMLResponse(
            '<div class="card" style="color:var(--red)">Modèle non disponible pour ce circuit.</div>'
        )

    result = ml_module.predict(
        artifacts,
        p1_name=p1_name, p2_name=p2_name,
        tournament=tournament, surface=surface,
        round_=round_, best_of=best_of,
        odd_p1=odd_p1, odd_p2=odd_p2,
        bankroll=bankroll,
    )
    return templates.TemplateResponse("partials/prediction_result.html", {
        "request": request, "result": result,
        "p1_name": p1_name, "p2_name": p2_name,
        "tour": tour, "tournament": tournament, "surface": surface,
        "round": round_, "best_of": best_of, "odd_p1": odd_p1,
    })


@router.post("/bets", response_class=HTMLResponse)
async def save_bet(
    request: Request,
    tour: str = Form(...),
    p1_name: str = Form(...),
    p2_name: str = Form(...),
    bet_on: str = Form(...),
    tournament: str = Form(...),
    surface: str = Form(...),
    round_: str = Form(..., alias="round"),
    prob: float = Form(...),
    edge: float | None = Form(None),
    odd: float = Form(...),
    stake: float = Form(...),
    kelly_frac: float | None = Form(None),
):
    db = _state()['db']
    add_bet(db, {
        'tour': tour, 'tournament': tournament, 'surface': surface,
        'round': round_, 'p1_name': p1_name, 'p2_name': p2_name,
        'bet_on': bet_on, 'prob': prob, 'edge': edge,
        'odd': odd, 'stake': stake, 'kelly_frac': kelly_frac,
    })
    bankroll = get_bankroll(db, tour)
    return HTMLResponse(
        f'<div class="card" style="color:var(--green)">✅ Pari enregistré. '
        f'Bankroll {tour.upper()}: <strong>{bankroll:.0f}€</strong></div>'
    )
