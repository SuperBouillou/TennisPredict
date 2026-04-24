"""Router — Today's matches."""
from __future__ import annotations

import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path

from fastapi import APIRouter, Form, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.webapp import ml as ml_module
from src.webapp.db import get_bankroll, set_bankroll, get_setting, list_bets

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")

_SRC = str(Path(__file__).resolve().parents[3] / 'src')


def _app_state() -> dict:
    """Lazy import to avoid circular import at module load time."""
    from src.webapp.main import APP_STATE  # noqa: PLC0415
    return APP_STATE


def _get_today_matches(tour: str, match_date: str) -> list[dict]:
    """Fetch scheduled matches from ESPN for given tour+date."""
    try:
        if str(_SRC) not in sys.path:
            sys.path.insert(0, _SRC)
        from espn_client import fetch_scheduled
        target = date.fromisoformat(match_date)
        matches = fetch_scheduled(tour, target)
        return matches if matches else []
    except Exception as e:
        print(f"[today] ESPN fetch error ({tour} {match_date}): {e}")
        return []


def _get_results(tour: str, match_date: str) -> list[dict]:
    """Fetch completed match results from ESPN for given tour+date."""
    try:
        if str(_SRC) not in sys.path:
            sys.path.insert(0, _SRC)
        from espn_client import fetch_results
        target = date.fromisoformat(match_date)
        matches = fetch_results(tour, target)
        return matches if matches else []
    except Exception as e:
        print(f"[today] ESPN results fetch error ({tour} {match_date}): {e}")
        return []


def _get_odds(tour: str, match_date: str) -> tuple[dict, str | None]:
    """
    Return (odds_dict, fetched_at_str).
    Fetches odds for today AND tomorrow — l'API retourne tous les matchs à venir.
    Pas de cotes pour les dates passées (hier).
    Uses file cache: data/odds_cache/{tour}/odds_YYYY-MM-DD.json
    """
    today = date.today()
    tomorrow = (today + timedelta(days=1)).isoformat()
    if match_date not in (today.isoformat(), tomorrow):
        return {}, None
    try:
        if str(_SRC) not in sys.path:
            sys.path.insert(0, _SRC)
        from odds_api_client import fetch_odds_today
        result = fetch_odds_today(tour)
        return result.odds, result.fetched_at
    except Exception as e:
        print(f"[today] Odds API error ({tour}): {e}")
        return {}, None


def _delete_odds_cache(tour: str) -> None:
    """Delete today's odds cache file to force a fresh API call."""
    try:
        if str(_SRC) not in sys.path:
            sys.path.insert(0, _SRC)
        from odds_api_client import _cache_path  # noqa: PLC0415
        path = _cache_path(tour, date.today())
        if path.exists():
            path.unlink()
            print(f"[today] Deleted odds cache: {path.name}")
    except Exception as e:
        print(f"[today] Could not delete odds cache: {e}")


def _build_matches(tour: str, match_date: str, bankroll: float, kelly_fraction: float = 0.25) -> tuple[list[dict], str | None]:
    """Full pipeline: ESPN → odds merge → ML enrichment."""
    sys.path.insert(0, _SRC)
    matches = _get_today_matches(tour, match_date)
    odds, fetched_at = _get_odds(tour, match_date)
    if odds:
        from odds_api_client import merge_odds
        matches = merge_odds(matches, odds)
    matches = _enrich_with_predictions(matches, tour, bankroll, kelly_fraction)
    return matches, fetched_at


def _player_stats(profiles, name: str, surface: str) -> dict:
    """Extract display stats from player profile: rank, form dots, surface winrate, days_since."""
    profile = ml_module._get_player(profiles, name)
    if not profile:
        return {}
    rank = profile.get('rank')
    try:
        rank_int = int(rank) if rank is not None and not math.isnan(float(rank)) else None
    except (TypeError, ValueError):
        rank_int = None

    wr5    = profile.get('winrate_5')
    streak = profile.get('streak', 0)
    wr_surf_key = f'winrate_surf_{surface}'
    wr_surf = profile.get(wr_surf_key)
    days   = profile.get('days_since')
    m14    = profile.get('matches_14d', 0)

    surf_pct = round(float(wr_surf) * 100) if (wr_surf is not None and not math.isnan(float(wr_surf))) else None

    # Forme réelle si disponible, sinon reconstruction approximative
    form_str = profile.get('form_last5')
    if form_str and isinstance(form_str, str) and form_str.strip():
        form = [c.strip() for c in form_str.split(',') if c.strip() in ('W', 'L')]
    else:
        form = []
    if not form:
        form = _form_dots(wr5, streak) or []
    # Toujours 5 tuiles — compléter à gauche avec 'U' (unknown) si données insuffisantes
    if 0 < len(form) < 5:
        form = ['U'] * (5 - len(form)) + form

    return {
        'rank':       rank_int,
        'form':       form,
        'surf_pct':   surf_pct,
        'days_since': int(days) if days is not None else None,
        'matches_14d': int(m14) if m14 is not None else 0,
    }


def _resolve_h2h(h2h_lookup: dict, p1_name: str, p2_name: str, surface: str) -> tuple[dict | None, dict | None]:
    """
    Résout le H2H pour (p1, p2) depuis le lookup.
    Retourne (h2h_for_ml, h2h_display) ou (None, None) si pas de données.
    h2h_for_ml = {'total', 'p1_wins', 'surf_total', 'surf_p1_wins'}
    h2h_display = {'total', 'p1_wins', 'p2_wins'}
    """
    k1 = p1_name.lower()
    k2 = p2_name.lower()
    h2h_key = (min(k1, k2), max(k1, k2))
    h2h_raw = h2h_lookup.get(h2h_key)
    if not h2h_raw or h2h_raw['total'] == 0:
        return None, None
    p1_wins = h2h_raw['wins_key0'] if k1 == h2h_key[0] else h2h_raw['total'] - h2h_raw['wins_key0']
    surf_data = h2h_raw.get('by_surface', {}).get(surface, {})
    surf_total = surf_data.get('total', 0)
    surf_wins_key0 = surf_data.get('wins_key0', 0)
    surf_p1_wins = surf_wins_key0 if k1 == h2h_key[0] else surf_total - surf_wins_key0
    h2h_for_ml = {
        'total': h2h_raw['total'], 'p1_wins': p1_wins,
        'surf_total': surf_total, 'surf_p1_wins': surf_p1_wins,
    }
    h2h_display = {
        'total': h2h_raw['total'],
        'p1_wins': p1_wins,
        'p2_wins': h2h_raw['total'] - p1_wins,
    }
    return h2h_for_ml, h2h_display


def _enrich_with_predictions(matches: list[dict], tour: str, bankroll: float, kelly_fraction: float = 0.25) -> list[dict]:
    """Add ML prediction + player display stats to each match."""
    artifacts = _app_state().get('models', {}).get(tour)
    if not artifacts:
        return matches
    # Prefer O(1) dict lookup if built at startup
    profiles = artifacts.get('profiles_dict') or artifacts.get('profiles')
    enriched = []
    h2h_lookup = _app_state().get('h2h', {}).get(tour, {})
    for m in matches:
        surface = m.get('surface', 'Hard')

        # H2H lookup — done first so it can be passed into predict()
        h2h_for_ml, h2h_display = _resolve_h2h(
            h2h_lookup, m.get('p1_name', ''), m.get('p2_name', ''), surface
        )
        if h2h_display:
            m['h2h_total'] = h2h_display['total']
            m['h2h_p1_wins'] = h2h_display['p1_wins']
            m['h2h_p2_wins'] = h2h_display['p2_wins']

        try:
            result = ml_module.predict(
                artifacts,
                p1_name=m.get('p1_name', ''),
                p2_name=m.get('p2_name', ''),
                tournament=m.get('tournament', ''),
                surface=surface,
                round_=m.get('round', 'R64'),
                best_of=m.get('best_of', 3),
                odd_p1=m.get('odd_p1'),
                odd_p2=m.get('odd_p2'),
                bankroll=bankroll,
                kelly_fraction=kelly_fraction,
                h2h=h2h_for_ml,
            )
            m.update(result)
        except Exception:
            pass

        # Player profile stats (rank, form, surface winrate)
        if profiles is not None:
            for prefix, pname in [('p1', m.get('p1_name', '')), ('p2', m.get('p2_name', ''))]:
                stats = _player_stats(profiles, pname, surface)
                for k, v in stats.items():
                    m[f'{prefix}_{k}'] = v

        ep1 = m.get('edge_p1') or -99
        ep2 = m.get('edge_p2') or -99
        best_edge = max(ep1, ep2)
        # Odd of the "value side" — the side we'd actually bet on
        active_odd = m.get('odd_p1') if ep1 >= ep2 else m.get('odd_p2')

        dq = m.get('data_quality', 'low')
        # Thresholds scale with data quality to avoid false signals from stale ELO.
        # low=70% ELO weighting (sparse/stale data): suppress ALL recommendations.
        # Prevents false VALUE BET and EDGE MODÉRÉ when player history is insufficient.
        if dq == 'high':
            value_thr, edge_thr = 0.12, 0.06   # ≥3 matchs récents — signaux fiables
        elif dq == 'medium':
            value_thr, edge_thr = 0.15, 0.09   # 1-2 matchs récents — seuils relevés
        else:                        # low — aucun match récent, ELO pur non fiable
            value_thr, edge_thr = float('inf'), float('inf')

        if best_edge >= value_thr:
            m['badge'] = 'value'
        elif best_edge >= edge_thr:
            m['badge'] = 'edge'
        else:
            m['badge'] = 'neutral'

        # High-odds cap: model reliability drops sharply at high odds.
        # @6: VALUE → EDGE MODÉRÉ (uncertainty too high for a value call)
        # @7: EDGE MODÉRÉ → NEUTRE (variance too high to recommend)
        # @10: anything → NEUTRE (failsafe)
        if active_odd is not None:
            if active_odd >= 10.0 and m['badge'] != 'neutral':
                m['badge'] = 'neutral'
            elif active_odd >= 7.0 and m['badge'] in ('value', 'edge'):
                m['badge'] = 'neutral'
            elif active_odd >= 6.0 and m['badge'] == 'value':
                m['badge'] = 'edge'
        enriched.append(m)
    return sorted(enriched, key=lambda x: -max(x.get('edge_p1') or -99, x.get('edge_p2') or -99))


def _form_dots(winrate_5, streak, n: int = 5) -> list[str] | None:
    """
    Retourne une liste de 'W'/'L' (index 0 = plus ancien, n-1 = plus récent)
    en utilisant winrate_5 et streak pour reconstituer la forme récente.
    Retourne None si pas assez de données.
    """
    if winrate_5 is None:
        return None
    try:
        wr = float(winrate_5)
        st = int(streak or 0)
    except (TypeError, ValueError):
        return None

    n_wins = round(wr * n)
    n_wins = max(0, min(n, n_wins))

    dots = ['L'] * n
    if st > 0:
        # Les `st` derniers matchs sont des victoires
        wins_at_end = min(st, n)
        for i in range(wins_at_end):
            dots[n - 1 - i] = 'W'
        # Remplir le reste avec le reliquat de victoires
        remaining_wins = max(0, n_wins - wins_at_end)
        remaining_positions = [i for i in range(n - wins_at_end) if dots[i] == 'L']
        for i in remaining_positions[:remaining_wins]:
            dots[i] = 'W'
    elif st < 0:
        # Les `|st|` derniers matchs sont des défaites
        losses_at_end = min(-st, n)
        # Les victoires vont dans les premières positions
        for i in range(n_wins):
            dots[i] = 'W'
    else:
        # Pas de streak: victoires en premier
        for i in range(n_wins):
            dots[i] = 'W'

    return dots


def _fmt(amount: float) -> str:
    return f'{amount:.2f}' if amount != int(amount) else str(int(amount))


def _bankroll_sidebar_html(amount: float) -> str:
    """Sidebar widget span (base.html)."""
    return (
        f'<span id="bankroll-global" class="bw-amount" title="Cliquer pour modifier" '
        f'hx-get="/bankroll/edit?sidebar=1" hx-target="#bankroll-global" hx-swap="outerHTML">'
        f'{_fmt(amount)}€</span>'
    )


def _bankroll_card_html(amount: float) -> str:
    """kpi-card div for history.html."""
    return (
        f'<div class="kpi-card" id="bankroll-global" title="Cliquer pour modifier" style="cursor:pointer" '
        f'hx-get="/bankroll/edit?card=1" hx-target="#bankroll-global" hx-swap="outerHTML">'
        f'<div class="kpi-value">{_fmt(amount)}€</div>'
        f'<div class="kpi-label">Bankroll ✏️</div></div>'
    )


@router.get("/bankroll/display", response_class=HTMLResponse)
async def bankroll_display(card: int = 0, sidebar: int = 0):
    db = _app_state()['db']
    amount = get_bankroll(db)
    if sidebar:
        return HTMLResponse(_bankroll_sidebar_html(amount))
    return HTMLResponse(_bankroll_card_html(amount) if card else _bankroll_sidebar_html(amount))


@router.get("/bankroll/edit", response_class=HTMLResponse)
async def bankroll_edit(card: int = 0, sidebar: int = 0):
    db = _app_state()['db']
    amount = get_bankroll(db)
    if sidebar:
        cancel_url = "/bankroll/display?sidebar=1"
        html = (
            f'<span id="bankroll-global" class="bw-amount" style="display:inline-flex;align-items:center;gap:4px">'
            f'<form hx-post="/bankroll/set" hx-target="#bankroll-global" hx-swap="outerHTML" '
            f'style="display:inline-flex;align-items:center;gap:4px;margin:0">'
            f'<input type="hidden" name="sidebar" value="1">'
            f'<input type="number" name="amount" value="{round(amount, 2)}" min="0" step="0.01" '
            f'style="width:80px;padding:2px 6px;font-size:13px;border-radius:4px;'
            f'border:1px solid var(--border);background:var(--surface2);color:var(--text)" autofocus>'
            f'<button type="submit" class="btn btn-sm" '
            f'style="background:var(--green);color:#000;border:none;padding:2px 6px">✓</button>'
            f'<button type="button" class="btn btn-sm" '
            f'style="background:transparent;color:var(--muted);border:1px solid var(--border);padding:2px 6px" '
            f'hx-get="{cancel_url}" hx-target="#bankroll-global" hx-swap="outerHTML">✗</button>'
            f'</form></span>'
        )
        return HTMLResponse(html)
    cancel_url = "/bankroll/display" + ("?card=1" if card else "")
    input_style = (
        f'width:80px;padding:2px 6px;font-size:{"14" if card else "12"}px;border-radius:4px;'
        f'border:1px solid var(--border);background:var(--surface);color:var(--text)'
    )
    form_inner = (
        f'<input type="hidden" name="card" value="{card}">'
        f'<input type="number" name="amount" value="{round(amount, 2)}" min="0" step="0.01" '
        f'style="{input_style}" autofocus>'
        f'<button type="submit" class="btn btn-sm" '
        f'style="background:var(--green);color:#000;border:none;padding:2px 8px">✓</button>'
        f'<button type="button" class="btn btn-sm" '
        f'style="background:var(--surface);color:var(--text);border:1px solid var(--border);padding:2px 8px" '
        f'hx-get="{cancel_url}" hx-target="#bankroll-global" hx-swap="outerHTML">✗</button>'
    )
    form = (
        f'<form hx-post="/bankroll/set" hx-target="#bankroll-global" hx-swap="outerHTML" '
        f'style="display:inline-flex;align-items:center;gap:4px;margin:0">{form_inner}</form>'
    )
    if card:
        html = (
            f'<div class="kpi-card" id="bankroll-global" style="display:flex;flex-direction:column;gap:6px">'
            f'<div class="kpi-label">Bankroll</div>{form}</div>'
        )
    else:
        html = (
            f'<span id="bankroll-global" style="display:inline-flex;align-items:center;gap:4px">'
            f'{form}</span>'
        )
    return HTMLResponse(html)


@router.post("/bankroll/set", response_class=HTMLResponse)
async def bankroll_set(amount: float = Form(...), card: int = Form(0), sidebar: int = Form(0)):
    db = _app_state()['db']
    set_bankroll(db, amount=max(0.0, amount))
    new_amount = get_bankroll(db)
    if sidebar:
        return HTMLResponse(_bankroll_sidebar_html(new_amount))
    html = _bankroll_card_html(new_amount) if card else _bankroll_sidebar_html(new_amount)
    return HTMLResponse(html)


def _get_ranking_updated_at(tour: str) -> str | None:
    """Read ranking meta timestamp from data/raw/{tour}/{tour}_ranking_lookup_meta.json."""
    try:
        if str(_SRC) not in sys.path:
            sys.path.insert(0, _SRC)
        from config import get_paths  # noqa: PLC0415
        paths = get_paths(tour)
        meta_path = Path(paths['raw_dir']) / f"{tour}_ranking_lookup_meta.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text()).get('updated_at')
    except Exception:
        pass
    return None


@router.get("/today", response_class=HTMLResponse)
async def today_page(request: Request, tour: str = "atp",
                     match_date: str = Query(default=None),
                     client_today: str | None = Query(default=None)):
    server_today = date.today()
    server_today_str = server_today.isoformat()

    # Use the client's local date as reference when it differs from server UTC
    # (e.g. user in UTC+2 at midnight already has a different date than the server)
    ref_today_str = server_today_str
    if client_today:
        try:
            ct = date.fromisoformat(client_today)
            if -2 <= (ct - server_today).days <= 2:
                ref_today_str = client_today
        except ValueError:
            pass

    ref_today = date.fromisoformat(ref_today_str)
    yesterday_str = (ref_today - timedelta(days=1)).isoformat()
    tomorrow_str = (ref_today + timedelta(days=1)).isoformat()

    # Restrict navigation to ±1 day from the client's reference date
    if not match_date:
        match_date = ref_today_str
    if match_date < yesterday_str:
        match_date = yesterday_str
    elif match_date > tomorrow_str:
        match_date = tomorrow_str

    # Determine page mode
    if match_date == yesterday_str:
        page_mode = "hier"
    elif match_date == tomorrow_str:
        page_mode = "demain"
    else:
        page_mode = "today"

    state = _app_state()
    db = state['db']
    bankroll = get_bankroll(db)
    kelly_fraction = float(get_setting(db, 'kelly_fraction', '0.25'))

    # Build a lookup of pending bets keyed by player pair so match cards can
    # show "✓ Parié" instead of the bet form when a bet is already in flight.
    pending_bets = list_bets(db, tour=tour, status='pending', limit=500)
    pending_lookup: dict = {}
    for b in pending_bets:
        k1 = f"{b['p1_name']}|{b['p2_name']}"
        k2 = f"{b['p2_name']}|{b['p1_name']}"
        pending_lookup[k1] = b
        pending_lookup[k2] = b

    if page_mode == "hier":
        matches = _get_results(tour, match_date)
        matches = _enrich_with_predictions(matches, tour, bankroll, kelly_fraction)
        odds_fetched_at = None
        ml_total = sum(1 for m in matches if m.get('prob_p1') is not None)
        ml_correct = sum(1 for m in matches if m.get('prob_p1') is not None and m.get('prob_p1', 0) > 0.5)
        ml_summary = {
            'correct': ml_correct, 'total': ml_total,
            'pct': round(ml_correct / ml_total * 100, 1) if ml_total else 0,
        }
    else:
        matches, odds_fetched_at = _build_matches(tour, match_date, bankroll, kelly_fraction)
        ml_summary = None

    # Read ranking freshness timestamp
    ranking_updated_at = _get_ranking_updated_at(tour)

    return templates.TemplateResponse(request, "today.html", {
        "active": "today",
        "tour": tour,
        "match_date": match_date,
        "page_mode": page_mode,
        "today_date": server_today_str,   # raw server date — used by JS to detect drift
        "ref_today": ref_today_str,        # client-adjusted today
        "client_today": client_today or "",
        "yesterday": yesterday_str,
        "tomorrow": tomorrow_str,
        "matches": matches, "match_count": len(matches),
        "bankroll": bankroll,
        "pending_lookup": pending_lookup,
        "sync_status": state.get('sync_status', {}).get(tour, 'idle'),
        "odds_fetched_at": odds_fetched_at,
        "ranking_updated_at": ranking_updated_at,
        "ml_summary": ml_summary,
    })


@router.get("/today/matches", response_class=HTMLResponse)
async def today_matches_partial(request: Request, tour: str = "atp",
                                match_date: str = Query(default=None),
                                client_today: str | None = Query(default=None)):
    """HTMX partial — swap #match-list."""
    server_today = date.today()
    server_today_str = server_today.isoformat()

    ref_today_str = server_today_str
    if client_today:
        try:
            ct = date.fromisoformat(client_today)
            if -2 <= (ct - server_today).days <= 2:
                ref_today_str = client_today
        except ValueError:
            pass

    ref_today = date.fromisoformat(ref_today_str)
    yesterday_str = (ref_today - timedelta(days=1)).isoformat()
    tomorrow_str = (ref_today + timedelta(days=1)).isoformat()

    if not match_date:
        match_date = ref_today_str

    page_mode = "hier" if match_date == yesterday_str else (
        "demain" if match_date > ref_today_str else "today"
    )

    db = _app_state()['db']
    bankroll = get_bankroll(db)
    kelly_fraction = float(get_setting(db, 'kelly_fraction', '0.25'))

    pending_bets = list_bets(db, tour=tour, status='pending', limit=500)
    pending_lookup: dict = {}
    for b in pending_bets:
        pending_lookup[f"{b['p1_name']}|{b['p2_name']}"] = b
        pending_lookup[f"{b['p2_name']}|{b['p1_name']}"] = b

    if page_mode == "hier":
        matches = _get_results(tour, match_date)
        matches = _enrich_with_predictions(matches, tour, bankroll, kelly_fraction)
    else:
        matches, _ = _build_matches(tour, match_date, bankroll, kelly_fraction)

    return templates.TemplateResponse(request, "partials/match_card.html", {
        "matches": matches, "tour": tour, "page_mode": page_mode,
        "pending_lookup": pending_lookup,
    })


@router.post("/today/refresh-odds", response_class=HTMLResponse)
async def refresh_odds(request: Request, tour: str = "atp"):
    """Force re-fetch odds from API (deletes today's cache), then returns refreshed match list."""
    _delete_odds_cache(tour)
    db = _app_state()['db']
    bankroll = get_bankroll(db)
    kelly_fraction = float(get_setting(db, 'kelly_fraction', '0.25'))
    match_date = date.today().isoformat()
    matches, fetched_at = _build_matches(tour, match_date, bankroll, kelly_fraction)

    pending_bets = list_bets(db, tour=tour, status='pending', limit=500)
    pending_lookup: dict = {}
    for b in pending_bets:
        pending_lookup[f"{b['p1_name']}|{b['p2_name']}"] = b
        pending_lookup[f"{b['p2_name']}|{b['p1_name']}"] = b

    # Return the partial + updated odds badge via OOB swap
    time_str = fetched_at[11:16] if fetched_at else "—"
    n_with_odds = sum(1 for m in matches if m.get('odd_p1') is not None)
    oob = (
        f'<span id="odds-status" hx-swap-oob="true" style="color:var(--green)">'
        f'Cotes Pinnacle·{n_with_odds} matchs·{time_str}</span>'
    )
    cards_html = templates.TemplateResponse(request, "partials/match_card.html", {
        "matches": matches, "tour": tour, "pending_lookup": pending_lookup,
    })
    body = cards_html.body.decode()
    return HTMLResponse(body + oob)


@router.get("/today/match-detail", response_class=HTMLResponse)
async def match_detail(
    request: Request,
    tour: str = "atp",
    p1_name: str = "",
    p2_name: str = "",
    tournament: str = "",
    surface: str = "Hard",
    round: str = "R64",
    best_of: int = 3,
    odd_p1: float | None = None,
    odd_p2: float | None = None,
):
    """HTMX modal — detailed match prediction card."""
    state = _app_state()
    artifacts = state.get('models', {}).get(tour)

    # H2H lookup — done before predict() so it feeds into ML features
    h2h_lookup = state.get('h2h', {}).get(tour, {})
    h2h_for_ml, h2h = _resolve_h2h(h2h_lookup, p1_name, p2_name, surface)

    result: dict = {}
    if artifacts:
        db = state['db']
        bankroll = get_bankroll(db)
        kelly_fraction = float(get_setting(db, 'kelly_fraction', '0.25'))
        try:
            result = ml_module.predict(
                artifacts,
                p1_name=p1_name, p2_name=p2_name,
                tournament=tournament, surface=surface,
                round_=round, best_of=best_of,
                odd_p1=odd_p1, odd_p2=odd_p2,
                bankroll=bankroll,
                kelly_fraction=kelly_fraction,
                h2h=h2h_for_ml,
            )
        except Exception:
            pass

    return templates.TemplateResponse(request, "partials/match_detail_modal.html", {
        "tour": tour,
        "p1_name": p1_name, "p2_name": p2_name,
        "tournament": tournament, "surface": surface,
        "round": round, "best_of": best_of,
        "odd_p1": odd_p1, "odd_p2": odd_p2,
        "result": result,
        "h2h": h2h,
    })
