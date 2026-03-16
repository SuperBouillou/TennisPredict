"""Router — Today's matches."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.webapp import ml as ml_module
from src.webapp.db import get_bankroll

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")


def _app_state() -> dict:
    """Lazy import to avoid circular import at module load time."""
    from src.webapp.main import APP_STATE  # noqa: PLC0415
    return APP_STATE


def _get_today_matches(tour: str, match_date: str) -> list[dict]:
    """Fetch matches from ESPN; fall back gracefully."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
        from espn_client import fetch_day
        matches = fetch_day(tour, match_date)
        return matches if matches else []
    except Exception:
        return []


def _enrich_with_predictions(matches: list[dict], tour: str, bankroll: float) -> list[dict]:
    """Add ML prediction to each match if players are known."""
    artifacts = _app_state().get('models', {}).get(tour)
    if not artifacts:
        return matches
    enriched = []
    for m in matches:
        try:
            result = ml_module.predict(
                artifacts,
                p1_name=m.get('p1_name', ''),
                p2_name=m.get('p2_name', ''),
                tournament=m.get('tournament', ''),
                surface=m.get('surface', 'Hard'),
                round_=m.get('round', 'R64'),
                best_of=m.get('best_of', 3),
                odd_p1=m.get('odd_p1'),
                odd_p2=m.get('odd_p2'),
                bankroll=bankroll,
            )
            m.update(result)
        except Exception:
            pass
        edge = m.get('edge') or 0
        if edge >= 0.05:
            m['badge'] = 'value'
        elif edge >= 0.02:
            m['badge'] = 'edge'
        else:
            m['badge'] = 'neutral'
        enriched.append(m)
    return sorted(enriched, key=lambda x: -(x.get('edge') or -99))


@router.get("/today", response_class=HTMLResponse)
async def today_page(request: Request, tour: str = "atp",
                     match_date: str = Query(default=None)):
    if not match_date:
        match_date = date.today().isoformat()
    state = _app_state()
    db = state['db']
    matches = _get_today_matches(tour, match_date)
    matches = _enrich_with_predictions(matches, tour, get_bankroll(db, tour))
    return templates.TemplateResponse("today.html", {
        "request": request, "active": "today",
        "tour": tour, "match_date": match_date,
        "matches": matches, "match_count": len(matches),
        "bankroll_atp": get_bankroll(db, 'atp'),
        "bankroll_wta": get_bankroll(db, 'wta'),
        "sync_status": state.get('sync_status', {}).get(tour, 'idle'),
    })


@router.get("/today/matches", response_class=HTMLResponse)
async def today_matches_partial(request: Request, tour: str = "atp",
                                match_date: str = Query(default=None)):
    """HTMX partial — swap #match-list."""
    if not match_date:
        match_date = date.today().isoformat()
    db = _app_state()['db']
    bankroll = get_bankroll(db, tour)
    matches = _get_today_matches(tour, match_date)
    matches = _enrich_with_predictions(matches, tour, bankroll)
    return templates.TemplateResponse("partials/match_card.html", {
        "request": request, "matches": matches, "tour": tour,
    })
