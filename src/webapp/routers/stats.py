"""Router — Backtest stats and configuration."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
from fastapi import APIRouter, Request, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from src.webapp.db import get_bankroll, get_setting, set_setting, get_signal_stats, list_signals, get_signal_curve
from src.webapp.state import get_state

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")


def _load_equity(tour: str, strategy: str) -> dict:
    strat_map = {
        'Kelly':   'backtest_strat_Kelly_1_4_cap2%.parquet',
        'Flat':    'backtest_strat_Flat_10\u20ac.parquet',
        'Percent': 'backtest_strat_Pct_2%.parquet',
    }
    fname = strat_map.get(strategy, 'backtest_kelly.parquet')
    df = get_state().get('backtest', {}).get(tour, {}).get(fname)
    if df is None:
        return {'labels': [], 'values': []}
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
    bookmakers = ['Bet365', 'Pinnacle', 'Best', 'Avg']
    roi_list = []
    bt_cache = get_state().get('backtest', {}).get(tour, {})
    for bk in bookmakers:
        df = bt_cache.get(f'backtest_real_{bk}.parquet')
        if df is None:
            roi_list.append(None)
            continue
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
    artifacts = get_state().get('models', {}).get(tour)
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
    """Compute summary KPIs — same source as the default (Kelly) equity curve."""
    bt_cache = get_state().get('backtest', {}).get(tour, {})
    # Priority: Kelly strategy file (matches equity curve default) → Pinnacle real → Kelly raw
    df = bt_cache.get('backtest_strat_Kelly_1_4_cap2%.parquet')
    if df is None:
        df = bt_cache.get('backtest_real_Pinnacle.parquet')
    if df is None:
        df = bt_cache.get('backtest_kelly.parquet')
    if df is None:
        return {}
    n_bets = len(df)
    if 'pnl' not in df.columns:
        return {'n_bets': n_bets}
    stake_col = df['stake'] if 'stake' in df.columns else pd.Series([1.0] * n_bets)
    total_stake = stake_col.sum()
    roi  = round(float(df['pnl'].sum() / total_stake) if total_stake > 0 else 0, 4)
    won  = int((df['pnl'] > 0).sum())
    wr   = round(won / n_bets, 4) if n_bets > 0 else 0
    # Final bankroll from equity curve (strategy simulation starts at 1000€)
    bk_col = next((c for c in df.columns if 'bankroll' in c.lower()), None)
    final_bankroll = round(float(df[bk_col].iloc[-1]), 2) if bk_col else None
    return {
        'n_bets': n_bets, 'roi': roi, 'win_rate': wr,
        'pnl': round(float(df['pnl'].sum()), 2),
        'final_bankroll': final_bankroll,
    }


@router.get("/stats", response_class=HTMLResponse)
async def stats_page(request: Request, tour: str = "atp"):
    db = get_state()['db']
    settings = {
        'min_edge':       get_setting(db, 'min_edge', '0.03'),
        'min_prob':       get_setting(db, 'min_prob', '0.55'),
        'kelly_fraction': get_setting(db, 'kelly_fraction', '0.25'),
    }
    kpis     = _kpis(tour)
    roi_bk   = _load_roi_bookmakers(tour)
    features = _load_feature_importance(tour)
    return templates.TemplateResponse(request, "stats.html", {
        "active": "stats", "tour": tour,
        "settings": settings, "kpis": kpis,
        "roi_bk": roi_bk, "features": features,
        "bankroll": get_bankroll(db),
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


@router.get("/stats/perf-by-edge")
async def perf_by_edge(tour: str = "atp"):
    """Win rate + ROI bucketed by edge size (from backtest_real_Pinnacle)."""
    bt_cache = get_state().get('backtest', {}).get(tour, {})
    df = bt_cache.get('backtest_real_Pinnacle.parquet')
    if df is None:
        df = bt_cache.get('backtest_all_candidates.parquet')
    if df is None:
        return JSONResponse({'buckets': [], 'win_rates': [], 'rois': [], 'counts': []})
    prob_col = 'our_prob' if 'our_prob' in df.columns else 'prob'
    if 'edge' not in df.columns or 'won' not in df.columns:
        return JSONResponse({'buckets': [], 'win_rates': [], 'rois': [], 'counts': []})

    bins = [0.0, 0.05, 0.10, 0.15, 0.20, 1.0]
    labels = ['<5%', '5–10%', '10–15%', '15–20%', '>20%']
    df['edge_bucket'] = pd.cut(df['edge'], bins=bins, labels=labels, right=False)
    grp = df.groupby('edge_bucket', observed=True)

    win_rates, rois, counts = [], [], []
    for lbl in labels:
        g = grp.get_group(lbl) if lbl in grp.groups else pd.DataFrame()
        if len(g) == 0:
            win_rates.append(None)
            rois.append(None)
            counts.append(0)
            continue
        wr = round(float(g['won'].mean()), 4)
        if 'stake' in g.columns and g['stake'].sum() > 0:
            roi = round(float(g['pnl'].sum() / g['stake'].sum()), 4)
        elif 'pnl' in g.columns:
            roi = round(float(g['pnl'].mean()), 4)
        else:
            roi = None
        win_rates.append(wr)
        rois.append(roi)
        counts.append(len(g))

    return JSONResponse({'buckets': labels, 'win_rates': win_rates, 'rois': rois, 'counts': counts})


@router.get("/stats/perf-by-surface")
async def perf_by_surface(tour: str = "atp"):
    """Win rate + ROI broken down by surface (from backtest_real_Pinnacle)."""
    bt_cache = get_state().get('backtest', {}).get(tour, {})
    df = bt_cache.get('backtest_real_Pinnacle.parquet')
    if df is None:
        df = bt_cache.get('backtest_all_candidates.parquet')
    if df is None:
        return JSONResponse({'surfaces': [], 'win_rates': [], 'rois': [], 'counts': []})
    if 'surface' not in df.columns or 'won' not in df.columns:
        return JSONResponse({'surfaces': [], 'win_rates': [], 'rois': [], 'counts': []})

    surfaces = ['Hard', 'Clay', 'Grass']
    win_rates, rois, counts = [], [], []
    for surf in surfaces:
        g = df[df['surface'] == surf]
        if len(g) == 0:
            win_rates.append(None)
            rois.append(None)
            counts.append(0)
            continue
        wr = round(float(g['won'].mean()), 4)
        if 'stake' in g.columns and g['stake'].sum() > 0:
            roi = round(float(g['pnl'].sum() / g['stake'].sum()), 4)
        else:
            roi = None
        win_rates.append(wr)
        rois.append(roi)
        counts.append(len(g))

    return JSONResponse({'surfaces': surfaces, 'win_rates': win_rates, 'rois': rois, 'counts': counts})


@router.get("/stats/live-perf")
async def live_perf(tour: str | None = None):
    """Real-bet performance stats from the bets SQLite table."""
    db = get_state()['db']
    q = "SELECT tour, odd, stake, pnl, status FROM bets WHERE status != 'pending'"
    rows = db.execute(q).fetchall()
    if not rows:
        return JSONResponse({'n': 0})

    import pandas as pd
    df = pd.DataFrame([dict(r) for r in rows])
    if tour:
        df = df[df['tour'] == tour]
    if df.empty:
        return JSONResponse({'n': 0})

    total_stake = df['stake'].sum()
    total_pnl   = df['pnl'].sum()
    n_won       = int((df['pnl'] > 0).sum())
    n           = len(df)
    roi         = round(float(total_pnl / total_stake), 4) if total_stake > 0 else 0.0
    win_rate    = round(n_won / n, 4) if n > 0 else 0.0

    # Per-tour breakdown
    by_tour = {}
    for t in ('atp', 'wta'):
        g = df[df['tour'] == t]
        if g.empty:
            continue
        gs = g['stake'].sum()
        by_tour[t] = {
            'n': len(g),
            'won': int((g['pnl'] > 0).sum()),
            'roi': round(float(g['pnl'].sum() / gs), 4) if gs > 0 else 0.0,
            'pnl': round(float(g['pnl'].sum()), 2),
        }

    return JSONResponse({
        'n': n, 'won': n_won,
        'roi': roi, 'win_rate': win_rate,
        'pnl': round(float(total_pnl), 2),
        'by_tour': by_tour,
    })


@router.get("/stats/monthly")
async def monthly_breakdown(tour: str | None = Query(default=None)):
    """Monthly P&L breakdown: stacked wins/losses bar + ROI line."""
    db = get_state()['db']
    base_q = """
        SELECT substr(resolved_at, 1, 7) AS month,
               COUNT(*) AS n,
               SUM(CASE WHEN status = 'won' THEN 1 ELSE 0 END) AS won,
               SUM(pnl) AS pnl,
               SUM(stake) AS staked
        FROM bets
        WHERE status != 'pending' AND resolved_at IS NOT NULL
    """
    rows = db.execute(
        base_q + (" AND tour = ?" if tour else "") + " GROUP BY month ORDER BY month",
        (tour,) if tour else ()
    ).fetchall()
    if not rows:
        return JSONResponse({"months": [], "won": [], "lost": [], "roi": [], "pnl": []})
    months, won, lost, roi, pnl = [], [], [], [], []
    for r in rows:
        months.append(r["month"])
        won.append(r["won"])
        lost.append(r["n"] - r["won"])
        roi.append(round(float(r["pnl"]) / float(r["staked"]) * 100, 1) if r["staked"] else 0.0)
        pnl.append(round(float(r["pnl"]), 2))
    return JSONResponse({"months": months, "won": won, "lost": lost, "roi": roi, "pnl": pnl})


@router.get("/stats/calibration")
async def calibration_curve(tour: str | None = Query(default=None)):
    """Predicted probability vs actual win rate, bucketed in 5% intervals."""
    db = get_state()['db']
    rows = db.execute(
        "SELECT prob, status FROM bets WHERE status != 'pending' AND prob IS NOT NULL"
        + (" AND tour = ?" if tour else ""),
        (tour,) if tour else ()
    ).fetchall()
    if not rows:
        return JSONResponse({"buckets": [], "predicted": [], "actual": [], "counts": []})

    # 5% buckets: [0.50-0.55, 0.55-0.60, ..., 0.95-1.00]
    edges = [i / 100 for i in range(50, 101, 5)]
    data = [{"wins": 0, "total": 0} for _ in range(len(edges) - 1)]
    for r in rows:
        prob = float(r["prob"])
        if prob < 0.50 or prob >= 1.0:
            continue
        idx = min(int((prob - 0.50) / 0.05), len(data) - 1)
        data[idx]["total"] += 1
        if r["status"] == "won":
            data[idx]["wins"] += 1

    buckets, predicted, actual, counts = [], [], [], []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if data[i]["total"] == 0:
            continue
        buckets.append(f"{int(lo * 100)}-{int(hi * 100)}%")
        predicted.append(round((lo + hi) / 2 * 100, 1))
        actual.append(round(data[i]["wins"] / data[i]["total"] * 100, 1))
        counts.append(data[i]["total"])
    return JSONResponse({"buckets": buckets, "predicted": predicted, "actual": actual, "counts": counts})


@router.get("/stats/signals")
async def signal_stats(tour: str | None = Query(default=None)):
    """KPIs du track record automatique."""
    db = get_state()['db']
    return JSONResponse(get_signal_stats(db, tour=tour))


@router.get("/stats/signals/curve")
async def signal_curve(tour: str | None = Query(default=None)):
    """Courbe P&L cumulative (en unités) pour le track record."""
    db = get_state()['db']
    return JSONResponse(get_signal_curve(db, tour=tour))


@router.get("/stats/signals/recent")
async def signal_recent(tour: str | None = Query(default=None),
                        limit: int = Query(default=50)):
    """Derniers signaux VALUE pour le tableau du track record."""
    db = get_state()['db']
    signals = list_signals(db, tour=tour, limit=limit)
    return JSONResponse(signals)


@router.post("/settings", response_class=HTMLResponse)
async def save_settings(
    min_edge: str = Form(...),
    min_prob: str = Form(...),
    kelly_fraction: str = Form(...),
):
    db = get_state()['db']
    set_setting(db, 'min_edge', min_edge)
    set_setting(db, 'min_prob', min_prob)
    set_setting(db, 'kelly_fraction', kelly_fraction)
    return HTMLResponse('<div style="color:var(--green);padding:8px">&#x2705; Seuils sauvegardés.</div>')
