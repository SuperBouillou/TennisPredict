"""FastAPI entry point."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.webapp.db import get_connection, init_db
from src.config import get_paths

ROOT = Path(__file__).resolve().parent.parent.parent

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

            models[tour] = {
                'model':        joblib.load(mdir / 'xgb_tuned.pkl'),
                'imputer':      joblib.load(mdir / 'imputer.pkl'),
                'platt':        joblib.load(mdir / 'platt_scaler.pkl'),
                'feature_list': joblib.load(mdir / 'feature_list.pkl'),
                'profiles':     profiles,
                'players':      players,
            }
        except FileNotFoundError as e:
            print(f"[WARN] {tour.upper()} artifacts missing: {e}")
            models[tour] = None

    APP_STATE['models'] = models
    APP_STATE['sync_status'] = {'atp': 'idle', 'wta': 'idle'}
    yield
    conn.close()


app = FastAPI(title="TennisPredict", lifespan=lifespan)

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
async def root():
    return RedirectResponse("/today")
