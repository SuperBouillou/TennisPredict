"""Router — Background sync."""
from __future__ import annotations

import asyncio
from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import HTMLResponse

router = APIRouter()


def _state():
    from src.webapp.main import APP_STATE
    return APP_STATE


async def _run_sync(tour: str) -> None:
    _state()['sync_status'][tour] = 'running'
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
        import importlib
        fld = importlib.import_module('fetch_live_data')
        await asyncio.to_thread(fld.run_update, tour, False)
    except Exception as e:
        print(f"[sync] {tour}: {e}")
    finally:
        _state()['sync_status'][tour] = 'idle'


@router.post("/sync", response_class=HTMLResponse)
async def trigger_sync(background_tasks: BackgroundTasks, tour: str = "atp"):
    status = _state().get('sync_status', {}).get(tour, 'idle')
    if status == 'running':
        return HTMLResponse(
            f'<span style="color:var(--orange)">Sync {tour.upper()} déjà en cours…</span>'
        )
    background_tasks.add_task(_run_sync, tour)
    return HTMLResponse(
        f'<span style="color:var(--blue)">Sync {tour.upper()} démarrée…</span>'
    )


@router.get("/sync/status", response_class=HTMLResponse)
async def sync_status(tour: str = "atp"):
    status = _state().get('sync_status', {}).get(tour, 'idle')
    if status == 'running':
        return HTMLResponse(
            f'<span hx-get="/sync/status?tour={tour}" hx-trigger="every 3s" '
            f'hx-swap="outerHTML" style="color:var(--orange)">⟳ Sync en cours…</span>'
        )
    return HTMLResponse(f'<span style="color:var(--muted)">idle</span>')
