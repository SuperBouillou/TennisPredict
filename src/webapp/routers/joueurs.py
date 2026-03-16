"""Router — Player profiles."""
from __future__ import annotations

from pathlib import Path
from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.webapp.players import search_players, get_profile

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")


def _state():
    from src.webapp.main import APP_STATE
    return APP_STATE


@router.get("/joueurs", response_class=HTMLResponse)
async def joueurs_page(request: Request, tour: str = "atp"):
    return templates.TemplateResponse("joueurs.html", {
        "request": request, "active": "joueurs", "tour": tour,
    })


@router.get("/joueurs/search", response_class=HTMLResponse)
async def joueurs_search(request: Request, q: str = "", tour: str = "atp"):
    artifacts = _state().get('models', {}).get(tour)
    if not artifacts or len(q) < 2:
        return HTMLResponse("")
    results = search_players(artifacts['profiles'], artifacts['players'], q)
    html = ""
    for p in results:
        name = p['player_name']
        rank = p.get('rank') or '—'
        elo  = int(p.get('elo', 0))
        surf = p.get('surface', '?')
        safe = name.replace("'", "\\'")
        html += (
            f'<div class="card" style="margin-bottom:6px;cursor:pointer" '
            f'onclick="window.location=\'/joueurs/{safe}?tour={tour}\'">'
            f'<strong>{name}</strong> '
            f'<span style="color:var(--muted);font-size:12px">'
            f'#{rank} · ELO {elo} · {surf}</span></div>'
        )
    return HTMLResponse(html or '<div style="color:var(--muted)">Aucun résultat.</div>')


@router.get("/joueurs/{player_name:path}", response_class=HTMLResponse)
async def joueur_profile(request: Request, player_name: str, tour: str = "atp"):
    artifacts = _state().get('models', {}).get(tour)
    if not artifacts:
        return HTMLResponse("Circuit non disponible.", status_code=503)
    profile = get_profile(artifacts['profiles'], artifacts['players'], player_name)
    if not profile:
        return HTMLResponse(
            f'<div class="content"><div class="card" style="color:var(--red)">'
            f'Joueur "{player_name}" non trouvé.</div></div>',
            status_code=404,
        )
    return templates.TemplateResponse("joueurs_profile.html", {
        "request": request, "active": "joueurs", "tour": tour, "p": profile,
    })
