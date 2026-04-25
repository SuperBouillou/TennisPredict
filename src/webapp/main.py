"""FastAPI entry point."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.webapp.db import get_connection, init_db
from src.config import get_paths, _file_age_hours

ROOT = Path(__file__).resolve().parent.parent.parent


def _build_h2h_lookup(processed_dir: Path) -> dict:
    """Build H2H dict keyed on sorted (name1, name2) pair from matches_features_final.parquet.
    Stores overall H2H + per-surface H2H so ml.py can use h2h_surf_p1_winrate correctly.
    """
    path = processed_dir / "matches_features_final.parquet"
    if not path.exists():
        return {}
    try:
        df = pd.read_parquet(
            path,
            columns=["p1_name", "p2_name", "surface", "h2h_p1_wins", "h2h_total",
                     "h2h_surf_p1_wins", "h2h_surf_total", "tourney_date"],
        )
        df = df.dropna(subset=["p1_name", "p2_name"]).sort_values("tourney_date")
        result: dict = {}
        for row in df.itertuples(index=False):
            k1 = str(row.p1_name).lower()
            k2 = str(row.p2_name).lower()
            key = (min(k1, k2), max(k1, k2))
            total = int(row.h2h_total or 0)
            p1w = int(row.h2h_p1_wins or 0)
            wins_key0 = p1w if k1 == key[0] else total - p1w
            if key not in result:
                result[key] = {"total": total, "wins_key0": wins_key0, "by_surface": {}}
            else:
                result[key]["total"] = total
                result[key]["wins_key0"] = wins_key0
            # Surface-specific H2H
            surf = str(row.surface) if row.surface else None
            if surf in ("Hard", "Clay", "Grass"):
                s_total = int(row.h2h_surf_total or 0)
                s_p1w = int(row.h2h_surf_p1_wins or 0)
                s_wins_key0 = s_p1w if k1 == key[0] else s_total - s_p1w
                result[key]["by_surface"][surf] = {"total": s_total, "wins_key0": s_wins_key0}
        print(f"[startup] H2H lookup: {len(result)} pairs")
        return result
    except Exception as e:
        print(f"[WARN] H2H lookup build failed: {e}")
        return {}


# Shared state — populated once at startup, read-only during requests
APP_STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models + open DB connection at startup."""
    db_path = ROOT / "data" / "tennis_predict.db"
    conn = get_connection(db_path)
    init_db(conn)
    APP_STATE['db'] = conn

    models = {}
    for tour in ('atp', 'wta'):
        paths = get_paths(tour)
        mdir = paths['models_dir']
        pdir = paths['processed_dir']
        try:
            profiles_path = pdir / 'player_profiles_updated.parquet'
            if not profiles_path.exists():
                profiles_path = pdir / 'matches_features_final.parquet'

            profiles = pd.read_parquet(profiles_path)
            if 'name_key' not in profiles.columns:
                profiles['name_key'] = profiles['player_name'].str.lower().str.strip()

            players_path = pdir / 'players.parquet'
            players = pd.read_parquet(players_path) if players_path.exists() else pd.DataFrame()

            ranking_path = pdir / 'ranking_lookup.json'
            ranking_lookup: dict = {}
            if ranking_path.exists():
                import json
                ranking_lookup = json.loads(ranking_path.read_text())

            # Prefer Pinnacle-calibrated scaler (LinearRegression) if available
            platt_path = mdir / 'platt_pinnacle.pkl'
            if not platt_path.exists():
                platt_path = mdir / 'platt_scaler.pkl'

            # Build O(1) profiles lookup dict (avoids full DataFrame scan per request)
            profiles_dict: dict = {}
            for _, row in profiles.iterrows():
                k = str(row.get('name_key', '')).lower().strip()
                if k:
                    profiles_dict[k] = row.to_dict()

            # Load per-surface scalers if available
            platt_surfaces = {}
            for surf in ['Hard', 'Clay', 'Grass']:
                sp = mdir / f'platt_{surf}.pkl'
                if sp.exists():
                    platt_surfaces[surf] = joblib.load(sp)
            if platt_surfaces:
                print(f"  [{tour.upper()}] Surface scalers: {list(platt_surfaces.keys())}")

            models[tour] = {
                'model':           joblib.load(mdir / 'xgb_tuned.pkl'),
                'imputer':         joblib.load(mdir / 'imputer.pkl'),
                'platt':           joblib.load(platt_path),
                'platt_pinnacle':  platt_path.stem == 'platt_pinnacle',
                'platt_surfaces':  platt_surfaces,
                'feature_list':    joblib.load(mdir / 'feature_list.pkl'),
                'profiles':        profiles,
                'profiles_dict':   profiles_dict,
                'players':         players,
                'ranking_lookup':  ranking_lookup,
            }
        except FileNotFoundError as e:
            print(f"[WARN] {tour.upper()} artifacts missing: {e}")
            models[tour] = None

    APP_STATE['models'] = models
    APP_STATE['sync_status'] = {'atp': 'idle', 'wta': 'idle'}

    # Pre-load backtest DataFrames once (files only change when backtest scripts are re-run)
    _BACKTEST_FILES = [
        'backtest_strat_Kelly_1_4_cap2%.parquet',
        'backtest_strat_Flat_10\u20ac.parquet',
        'backtest_strat_Pct_2%.parquet',
        'backtest_real_Pinnacle.parquet',
        'backtest_real_Bet365.parquet',
        'backtest_real_Best.parquet',
        'backtest_real_Avg.parquet',
        'backtest_all_candidates.parquet',
        'backtest_kelly.parquet',
    ]
    backtest: dict = {}
    for _tour in ('atp', 'wta'):
        _mdir = get_paths(_tour)['models_dir']
        _cache: dict = {}
        for _fname in _BACKTEST_FILES:
            _fp = _mdir / _fname
            if _fp.exists():
                try:
                    _cache[_fname] = pd.read_parquet(_fp)
                except Exception as _e:
                    print(f"[startup] backtest cache skip {_fname}: {_e}")
        backtest[_tour] = _cache
        print(f"[startup] backtest cache {_tour.upper()}: {len(_cache)} fichiers")
    APP_STATE['backtest'] = backtest

    # Build H2H lookup per tour
    h2h: dict = {}
    for tour in ('atp', 'wta'):
        paths = get_paths(tour)
        h2h[tour] = _build_h2h_lookup(paths['processed_dir'])
    APP_STATE['h2h'] = h2h

    # Auto-sync on startup if profiles are stale (> 4h)
    from src.webapp.routers.sync import _run_sync
    for tour in ('atp', 'wta'):
        paths = get_paths(tour)
        profiles_path = paths['processed_dir'] / 'player_profiles_updated.parquet'
        age_h = _file_age_hours(profiles_path)
        if age_h > 4:
            print(f"[startup] {tour.upper()} profiles {age_h:.0f}h old — launching background sync")
            asyncio.create_task(_run_sync(tour))
        else:
            print(f"[startup] {tour.upper()} profiles fresh ({age_h:.1f}h) — skip sync")

    # Periodic cron: re-sync every 6h if profiles are stale
    async def _cron_sync():
        while True:
            await asyncio.sleep(6 * 3600)
            for t in ('atp', 'wta'):
                if APP_STATE.get('sync_status', {}).get(t) == 'running':
                    continue
                p = get_paths(t)['processed_dir'] / 'player_profiles_updated.parquet'
                if _file_age_hours(p) > 6:
                    print(f"[cron] {t.upper()} profiles stale — auto-sync")
                    asyncio.create_task(_run_sync(t))

    cron_task = asyncio.create_task(_cron_sync())

    yield
    cron_task.cancel()
    conn.close()


app = FastAPI(title="TennisPredict", lifespan=lifespan)

# Auth middleware + rate limiter
from slowapi import _rate_limit_exceeded_handler  # noqa: E402
from slowapi.errors import RateLimitExceeded  # noqa: E402
from src.webapp.limiter import limiter  # noqa: E402
from src.webapp.middleware import AuthMiddleware  # noqa: E402
from src.webapp.routers.auth import router as auth_router  # noqa: E402

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(AuthMiddleware)
app.include_router(auth_router)

_HERE = Path(__file__).parent
app.mount("/static", StaticFiles(directory=_HERE / "static"), name="static")
templates = Jinja2Templates(directory=_HERE / "templates")

# Routers
from src.webapp.routers import today, predictions, history, joueurs, stats, sync  # noqa: E402

app.include_router(today.router)
app.include_router(predictions.router)
app.include_router(history.router)
app.include_router(joueurs.router)
app.include_router(stats.router)
app.include_router(sync.router)


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse(request, "landing.html", {})
