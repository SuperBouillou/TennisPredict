"""Router — Backtest stats and configuration."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from src.webapp.db import get_bankroll, get_setting, set_setting
from src.config import get_paths

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")


def _state():
    from src.webapp.main import APP_STATE
    return APP_STATE


def _load_equity(tour: str, strategy: str) -> dict:
    paths = get_paths(tour)
    strat_map = {
        'Kelly':   'backtest_strat_Kelly_1_4_cap2%.parquet',
        'Flat':    'backtest_strat_Flat_10\u20ac.parquet',
        'Percent': 'backtest_strat_Pct_2%.parquet',
    }
    fname = strat_map.get(strategy, 'backtest_kelly.parquet')
    fpath = paths['models_dir'] / fname
    if not fpath.exists():
        return {'labels': [], 'values': []}
    df = pd.read_parquet(fpath)
    # Find date and bankroll columns
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    bank_col = next(
        (c for c in df.columns if 'bankroll' in c.lower() or 'cumul' in c.lower()),
        None
    )
    if not date_col or not bank_col:
        # Fallback: try cumsum of pnl + 1000
        if 'pnl' in df.columns and date_col:
            df = df.sort_values(date_col).dropna(subset=[date_col])
            values = (1000 + df['pnl'].cumsum()).round(2).tolist()
            labels = df[date_col].astype(str).tolist()
            return {'labels': labels, 'values': values}
        return {'labels': [], 'values': []}
    df = df.sort_values(date_col).dropna(subset=[date_col, bank_col])
    return {
        'labels': df[date_col].astype(str).tolist(),
        'values': df[bank_col].round(2).tolist(),
    }


def _load_roi_bookmakers(tour: str) -> dict:
    paths = get_paths(tour)
    bookmakers = ['Bet365', 'Pinnacle', 'Best', 'Avg']
    roi_list = []
    for bk in bookmakers:
        fpath = paths['models_dir'] / f'backtest_real_{bk}.parquet'
        if not fpath.exists():
            roi_list.append(None)
            continue
        df = pd.read_parquet(fpath)
        if 'roi' in df.columns:
            roi_list.append(round(float(df['roi'].iloc[-1]), 4))
        elif 'pnl' in df.columns and 'stake' in df.columns:
            total_stake = df['stake'].sum()
            roi = float(df['pnl'].sum() / total_stake) if total_stake > 0 else 0.0
            roi_list.append(round(roi, 4))
        else:
            roi_list.append(None)
    return {'bookmakers': bookmakers, 'roi': roi_list}


def _load_feature_importance(tour: str) -> dict:
    artifacts = _state().get('models', {}).get(tour)
    if not artifacts or not artifacts.get('model'):
        return {'features': [], 'values': [], 'groups': []}
    model    = artifacts['model']
    features = artifacts['feature_list']
    try:
        importances = model.feature_importances_
    except AttributeError:
        return {'features': [], 'values': [], 'groups': []}

    # Color by feature group
    def _group(f):
        if 'elo' in f:    return 'elo'
        if 'form' in f or 'streak' in f: return 'forme'
        if 'h2h' in f:   return 'h2h'
        if 'serve' in f or 'ace' in f or '1st' in f or '2nd' in f or 'bp' in f: return 'stats'
        return 'other'

    _group_colors = {
        'elo': '#3b82f6', 'forme': '#22c55e',
        'h2h': '#f97316', 'stats': '#a855f7', 'other': '#64748b',
    }

    pairs = sorted(zip(features, importances), key=lambda x: -x[1])[:15]
    return {
        'features': [p[0] for p in pairs],
        'values':   [round(float(p[1]), 4) for p in pairs],
        'groups':   [_group_colors[_group(p[0])] for p in pairs],
    }


def _kpis(tour: str) -> dict:
    """Compute summary KPIs from backtest_real_Pinnacle if available."""
    paths = get_paths(tour)
    fpath = paths['models_dir'] / 'backtest_real_Pinnacle.parquet'
    if not fpath.exists():
        fpath = paths['models_dir'] / 'backtest_kelly.parquet'
    if not fpath.exists():
        return {}
    df = pd.read_parquet(fpath)
    n_bets = len(df)
    if 'pnl' not in df.columns:
        return {'n_bets': n_bets}
    total_stake = df.get('stake', pd.Series([1]*n_bets)).sum()
    roi  = round(float(df['pnl'].sum() / total_stake) if total_stake > 0 else 0, 4)
    won  = int((df['pnl'] > 0).sum())
    wr   = round(won / n_bets, 4) if n_bets > 0 else 0
    return {'n_bets': n_bets, 'roi': roi, 'win_rate': wr, 'pnl': round(float(df['pnl'].sum()), 2)}


@router.get("/stats", response_class=HTMLResponse)
async def stats_page(request: Request, tour: str = "atp"):
    db = _state()['db']
    settings = {
        'min_edge':       get_setting(db, 'min_edge', '0.03'),
        'min_prob':       get_setting(db, 'min_prob', '0.55'),
        'kelly_fraction': get_setting(db, 'kelly_fraction', '0.25'),
    }
    kpis     = _kpis(tour)
    roi_bk   = _load_roi_bookmakers(tour)
    features = _load_feature_importance(tour)
    return templates.TemplateResponse("stats.html", {
        "request": request, "active": "stats", "tour": tour,
        "settings": settings, "kpis": kpis,
        "roi_bk": roi_bk, "features": features,
        "bankroll": get_bankroll(db, tour),
    })


@router.get("/stats/equity")
async def equity_data(tour: str = "atp", strategy: str = "Kelly"):
    return JSONResponse(_load_equity(tour, strategy))


@router.get("/stats/roi-bookmakers")
async def roi_bookmakers_data(tour: str = "atp"):
    return JSONResponse(_load_roi_bookmakers(tour))


@router.get("/stats/features")
async def features_data(tour: str = "atp"):
    return JSONResponse(_load_feature_importance(tour))


@router.post("/settings", response_class=HTMLResponse)
async def save_settings(
    min_edge: str = Form(...),
    min_prob: str = Form(...),
    kelly_fraction: str = Form(...),
):
    db = _state()['db']
    set_setting(db, 'min_edge', min_edge)
    set_setting(db, 'min_prob', min_prob)
    set_setting(db, 'kelly_fraction', kelly_fraction)
    return HTMLResponse('<div style="color:var(--green);padding:8px">&#x2705; Seuils sauvegardés.</div>')
