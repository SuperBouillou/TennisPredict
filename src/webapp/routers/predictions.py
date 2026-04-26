"""Router — Manual predictions."""
from __future__ import annotations

import math
from pathlib import Path
from fastapi import APIRouter, Request, Form, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.webapp import ml as ml_module
from src.webapp.db import get_bankroll, get_setting, add_bet, list_bets
from src.webapp.state import get_state
from src.webapp.utils import safe_get

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")

_ROUNDS   = ['R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'F', 'RR']
_SURFACES = ['Hard', 'Clay', 'Grass']

def _map_round(espn_round: str) -> str:
    """Map ESPN round string → model round key. Passes through if already valid."""
    if espn_round in _ROUNDS:
        return espn_round
    r = espn_round.lower()
    if 'qualifying' in r:
        return 'R64'
    if 'round robin' in r:
        return 'RR'
    if 'quarterfinal' in r:
        return 'QF'
    if 'semifinal' in r:
        return 'SF'
    if r.strip() in ('final', 'the final'):
        return 'F'
    if 'round of 16' in r or 'round 4' in r:
        return 'R16'
    if 'round of 32' in r or 'round 3' in r:
        return 'R32'
    if 'round of 128' in r:
        return 'R128'
    if 'round of 64' in r or 'round 1' in r or 'round 2' in r:
        return 'R64'
    return 'R64'




def _build_context_items(artifacts: dict, p1_name: str, p2_name: str, surface: str) -> list[dict]:
    """Build top-6 match context items for the feature bar display."""
    items: list[dict] = []
    try:
        profiles = artifacts.get('profiles')
        if profiles is None:
            return items
        p1 = ml_module._get_player(profiles, p1_name)
        p2 = ml_module._get_player(profiles, p2_name)
        if not p1 or not p2:
            return items

        elo1 = safe_get(p1, 'elo', 1500.0)
        elo2 = safe_get(p2, 'elo', 1500.0)
        elo_diff = int(elo1 - elo2)
        sign = '+' if elo_diff >= 0 else ''
        items.append({'label': f'ELO diff ({sign}{elo_diff})', 'raw': abs(elo_diff)})

        wr10_1 = safe_get(p1, 'winrate_10', 0.5) * 100
        items.append({'label': 'Forme récente 10M', 'raw': wr10_1})

        surf_elo_key = {'Clay': 'elo_clay', 'Grass': 'elo_grass', 'Hard': 'elo_hard'}.get(surface, 'elo')
        elo_s1 = safe_get(p1, surf_elo_key, elo1)
        elo_s2 = safe_get(p2, surf_elo_key, elo2)
        surf_diff = int(elo_s1 - elo_s2)
        sign_s = '+' if surf_diff >= 0 else ''
        items.append({'label': f'ELO {surface.lower()} ({sign_s}{surf_diff})', 'raw': abs(surf_diff)})

        rank1 = safe_get(p1, 'rank', 200.0)
        rank2 = safe_get(p2, 'rank', 200.0)
        rank_diff = abs(int(rank2 - rank1))
        items.append({'label': f'Classement (#{int(rank1)} vs #{int(rank2)})', 'raw': rank_diff})

        wr_surf_key = f'winrate_surf_{surface}'
        wr_surf1 = safe_get(p1, wr_surf_key, 0.5) * 100
        items.append({'label': f'Win rate {surface}', 'raw': wr_surf1})

        ds1 = safe_get(p1, 'days_since', 7.0)
        freshness = max(0.0, 100.0 - ds1 * 6)
        items.append({'label': f'Repos P1 ({int(ds1)}j)', 'raw': freshness})

    except Exception:
        pass

    if not items:
        return items

    # Normalize to 0-95 scale
    max_raw = max(it['raw'] for it in items) or 1.0
    for it in items:
        it['pct'] = round(it['raw'] / max_raw * 95)
        del it['raw']
    return items


@router.get("/predictions", response_class=HTMLResponse)
async def predictions_page(
    request: Request,
    tour: str = "atp",
    p1_name: str = "",
    p2_name: str = "",
    tournament: str = "",
    surface: str = "Hard",
    round_: str = Query(default="", alias="round"),
    best_of: int = 3,
    odd_p1: float | None = None,
    odd_p2: float | None = None,
):
    db = get_state()['db']
    prefill = {
        "p1_name": p1_name,
        "p2_name": p2_name,
        "tournament": tournament,
        "surface": surface if surface in _SURFACES else "Hard",
        "round": _map_round(round_) if round_ else "R64",
        "best_of": best_of,
        "odd_p1": odd_p1,
        "odd_p2": odd_p2,
    }
    return templates.TemplateResponse(request, "predictions.html", {
        "active": "predictions",
        "tour": tour, "rounds": _ROUNDS, "surfaces": _SURFACES,
        "bankroll": get_bankroll(db),
        "prefill": prefill,
        "auto_run": bool(p1_name and p2_name),
    })


@router.get("/predictions/player-info", response_class=HTMLResponse)
async def player_info(name: str = "", tour: str = "atp"):
    """Return a small HTML fragment with player rank/ELO/form for the setup card."""
    if not name:
        return HTMLResponse("")
    artifacts = get_state().get('models', {}).get(tour)
    if not artifacts:
        return HTMLResponse("")
    try:
        p = ml_module._get_player(artifacts['profiles'], name)
    except Exception:
        p = None
    if not p:
        return HTMLResponse(
            f'<span style="font-size:10px;color:var(--fg-4);font-family:var(--font-mono)">Joueur introuvable</span>'
        )

    rank = p.get('rank')
    elo  = p.get('elo')
    ioc  = (p.get('ioc') or p.get('country') or '').upper()[:3]
    wr10 = safe_get(p, 'winrate_10', 0.5)

    rank_str = f'#{int(rank)}' if rank and not math.isnan(float(rank)) else '—'
    elo_str  = str(int(float(elo))) if elo and not math.isnan(float(elo)) else '—'

    # Form dots: approximate from winrate_5 (last 5 matches)
    wr5 = safe_get(p, 'winrate_5', 0.5)
    wins = round(wr5 * 5)
    dots_html = ''.join(
        f'<i style="display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:3px;background:{"var(--lime)" if i < wins else "var(--loss)"};opacity:{1 if i < wins else 0.4}"></i>'
        for i in range(5)
    )

    html = f'''<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
      <span style="font-family:var(--font-mono);font-size:10px;color:var(--fg-3);letter-spacing:.08em">{ioc} · {rank_str}</span>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-bottom:8px">
      <div><div style="font-family:var(--font-mono);font-size:9px;color:var(--fg-4);letter-spacing:.08em;margin-bottom:2px">ELO</div>
           <div style="font-family:var(--font-mono);font-size:14px;font-weight:600;color:var(--fg)">{elo_str}</div></div>
      <div><div style="font-family:var(--font-mono);font-size:9px;color:var(--fg-4);letter-spacing:.08em;margin-bottom:2px">FORME 10M</div>
           <div style="font-family:var(--font-mono);font-size:14px;font-weight:600;color:{"var(--lime)" if wr10 >= 0.6 else "var(--fg)"}">{round(wr10*100)}%</div></div>
    </div>
    <div>{dots_html}</div>'''
    return HTMLResponse(html)


@router.get("/predictions/autocomplete", response_class=HTMLResponse)
async def autocomplete(
    request: Request,
    q: str = "",
    tour: str = "atp",
    field: str = "p1",
):
    if len(q) < 2:
        return HTMLResponse("")
    artifacts = get_state().get('models', {}).get(tour)
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
    artifacts = get_state().get('models', {}).get(tour)
    db = get_state()['db']
    bankroll = get_bankroll(db)
    kelly_fraction = float(get_setting(db, 'kelly_fraction', '0.25'))

    if not artifacts:
        return HTMLResponse(
            '<div class="card" style="color:var(--red)">Modèle non disponible pour ce circuit.</div>'
        )

    if odd_p1 is not None and odd_p1 < 1.01:
        return HTMLResponse(
            '<div class="card" style="color:var(--red)">Cote P1 invalide — minimum 1.01.</div>',
            status_code=400,
        )
    if odd_p2 is not None and odd_p2 < 1.01:
        return HTMLResponse(
            '<div class="card" style="color:var(--red)">Cote P2 invalide — minimum 1.01.</div>',
            status_code=400,
        )

    result = ml_module.predict(
        artifacts,
        p1_name=p1_name, p2_name=p2_name,
        tournament=tournament, surface=surface,
        round_=round_, best_of=best_of,
        odd_p1=odd_p1, odd_p2=odd_p2,
        bankroll=bankroll,
        kelly_fraction=kelly_fraction,
    )
    context_items = _build_context_items(artifacts, p1_name, p2_name, surface)
    return templates.TemplateResponse(request, "partials/prediction_result.html", {
        "result": result,
        "p1_name": p1_name, "p2_name": p2_name,
        "tour": tour, "tournament": tournament, "surface": surface,
        "round": round_, "best_of": best_of,
        "odd_p1": odd_p1, "odd_p2": odd_p2,
        "context_items": context_items,
    })


@router.post("/bets/quick", response_class=HTMLResponse)
async def quick_bet(
    tour: str = Form(...),
    p1_name: str = Form(...),
    p2_name: str = Form(...),
    bet_on: str = Form(...),
    tournament: str = Form(...),
    surface: str = Form(...),
    round_: str = Form("", alias="round"),
    prob: float = Form(0.5),
    edge: float | None = Form(None),
    odd: float = Form(...),
    stake: float = Form(...),
    kelly_frac: float | None = Form(None),
):
    db = get_state()['db']
    bankroll = get_bankroll(db)
    if stake <= 0 or stake > bankroll:
        return HTMLResponse(
            f'<span style="color:var(--red);font-size:12px">Mise invalide (bankroll: {bankroll:.2f}€)</span>'
        )
    if odd < 1.01:
        return HTMLResponse(
            '<span style="color:var(--red);font-size:12px">Cote invalide (minimum 1.01)</span>',
            status_code=400,
        )
    # Guard against duplicate pending bets on the same match/player
    for b in list_bets(db, tour=tour, status='pending', limit=200):
        if (b['bet_on'].lower().strip() == bet_on.lower().strip()
                and b['p1_name'].lower().strip() == p1_name.lower().strip()
                and b['p2_name'].lower().strip() == p2_name.lower().strip()):
            return HTMLResponse(
                f'<span style="color:var(--orange);font-size:12px">⚠ Pari déjà en attente pour {bet_on}</span>'
            )
    add_bet(db, {
        'tour': tour, 'tournament': tournament, 'surface': surface,
        'round': round_, 'p1_name': p1_name, 'p2_name': p2_name,
        'bet_on': bet_on, 'prob': prob, 'edge': edge,
        'odd': odd, 'stake': round(stake, 2), 'kelly_frac': kelly_frac,
    })
    new_bankroll = get_bankroll(db)
    # OOB swap updates sidebar bankroll without page reload
    # Must match the canonical _bankroll_sidebar_html() format from today.py
    oob = (
        f'<span id="bankroll-global" hx-swap-oob="true" class="bw-amount" '
        f'title="Cliquer pour modifier" '
        f'hx-get="/bankroll/edit?sidebar=1" hx-target="#bankroll-global" hx-swap="outerHTML">'
        f'{new_bankroll:.2f}€</span>'
    )
    return HTMLResponse(
        f'<span style="color:var(--green);font-size:12px">✓ Pari enregistré · Bankroll {new_bankroll:.2f}€</span>'
        + oob
    )


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
    db = get_state()['db']
    bankroll = get_bankroll(db)
    if stake <= 0 or stake > bankroll:
        return HTMLResponse(
            f'<div class="card" style="color:var(--red)">Mise invalide (bankroll: {bankroll:.2f}€)</div>'
        )
    if odd < 1.01:
        return HTMLResponse(
            '<div class="card" style="color:var(--red)">Cote invalide (minimum 1.01)</div>',
            status_code=400,
        )
    add_bet(db, {
        'tour': tour, 'tournament': tournament, 'surface': surface,
        'round': round_, 'p1_name': p1_name, 'p2_name': p2_name,
        'bet_on': bet_on, 'prob': prob, 'edge': edge,
        'odd': odd, 'stake': stake, 'kelly_frac': kelly_frac,
    })
    new_bankroll = get_bankroll(db)
    return HTMLResponse(
        f'<div class="card" style="color:var(--green)">✅ Pari enregistré. '
        f'Bankroll: <strong>{new_bankroll:.0f}€</strong></div>'
    )
