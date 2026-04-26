"""Router — Background sync."""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import HTMLResponse

from src.webapp.state import get_state

router = APIRouter()

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _reload_profiles(tour: str) -> None:
    """Reload player profiles and ranking lookup into APP_STATE after a sync."""
    import json
    from src.config import get_paths
    state = get_state()
    if state['models'].get(tour) is None:
        return
    paths = get_paths(tour)

    profiles_path = paths['processed_dir'] / 'player_profiles_updated.parquet'
    if not profiles_path.exists():
        profiles_path = paths['processed_dir'] / 'matches_features_final.parquet'
    if profiles_path.exists():
        profiles = pd.read_parquet(profiles_path)
        if 'name_key' not in profiles.columns:
            profiles['name_key'] = profiles['player_name'].str.lower().str.strip()
        state['models'][tour]['profiles'] = profiles
        print(f"[sync] {tour.upper()} profiles reloaded ({len(profiles)} rows)")

    ranking_path = paths['processed_dir'] / 'ranking_lookup.json'
    if ranking_path.exists():
        state['models'][tour]['ranking_lookup'] = json.loads(ranking_path.read_text())
        print(f"[sync] {tour.upper()} ranking_lookup reloaded")


async def _run_sync(tour: str, force: bool = False) -> None:
    state = get_state()
    state['sync_status'][tour] = 'running'
    try:
        import fetch_live_data as fld
        await asyncio.to_thread(fld.run_update, tour, force)
        _reload_profiles(tour)
        state['sync_status'][f'{tour}_last'] = datetime.now().strftime('%H:%M')
    except Exception as e:
        print(f"[sync] {tour}: {e}")
    finally:
        state['sync_status'][tour] = 'idle'


@router.post("/sync", response_class=HTMLResponse)
async def trigger_sync(background_tasks: BackgroundTasks, tour: str = "atp"):
    status = get_state().get('sync_status', {}).get(tour, 'idle')
    if status == 'running':
        return HTMLResponse(
            f'<span style="color:var(--orange)">Sync {tour.upper()} déjà en cours…</span>'
        )
    background_tasks.add_task(_run_sync, tour)
    return HTMLResponse(
        f'<span style="color:var(--blue)">Sync {tour.upper()} démarrée…</span>'
    )


@router.get("/sync/status", response_class=HTMLResponse)
async def sync_status():
    ss = get_state().get('sync_status', {})
    atp_running = ss.get('atp') == 'running'
    wta_running = ss.get('wta') == 'running'
    running = atp_running or wta_running

    if running:
        labels = ' + '.join(t.upper() for t in ('atp', 'wta') if ss.get(t) == 'running')
        return HTMLResponse(
            f'<span id="sync-status" '
            f'hx-get="/sync/status" hx-trigger="every 3s" hx-swap="outerHTML" '
            f'style="color:var(--orange);font-size:12px">⟳ Mise à jour {labels}…</span>'
        )

    atp_last = ss.get('atp_last', '')
    wta_last = ss.get('wta_last', '')
    last = atp_last or wta_last
    label = f'Données à jour ({last})' if last else 'Données à jour'
    return HTMLResponse(
        f'<span id="sync-status" '
        f'hx-get="/sync/status" hx-trigger="every 60s" hx-swap="outerHTML" '
        f'style="color:var(--muted);font-size:12px">✓ {label}</span>'
    )
