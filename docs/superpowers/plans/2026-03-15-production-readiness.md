# Production Readiness — Threshold Optimizer + CLV + Dashboard Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the system production-ready for long-term profitability by adding optimal threshold selection, CLV tracking, and integrating both into the dashboard so every prediction shows a clear GO/PASS signal.

**Architecture:** Three focused changes: (1) `backtest_real.py` adds CLV metric and saves an unfiltered candidates parquet; (2) `optimize_thresholds.py` grid-searches min_edge × min_prob on that parquet and saves the optimal thresholds + profitable tournament levels to JSON; (3) `dashboard.py` loads the JSON to replace hardcoded thresholds, shows CLV alongside edge/EV, and populates the Stats tab with backtest ROI analysis.

**Tech Stack:** pandas, numpy, json, streamlit, matplotlib — all already installed. No new dependencies.

---

## Chunk 1: backtest_real.py — CLV + raw candidates

### Task 1: Add CLV to run_backtest() and backtest_metrics()

**Files:**
- Modify: `src/backtest_real.py` (lines ~370–434)

The CLV (Closing Line Value) measures whether our model probability beats the no-vig market probability. `clv = our_prob / bk_imp - 1`. Positive CLV means the model is sharper than Pinnacle on that bet — this is the gold standard for long-term edge verification.

- [ ] **Step 1: Add `clv` to the history dict in run_backtest()**

In `src/backtest_real.py`, find the `history.append({` block (around line 370). Find this exact text:
```python
                'bookmaker'         : odds_col_w.replace('W',''),
            })
```

Replace with:
```python
                'bookmaker'         : odds_col_w.replace('W',''),
                'clv'               : our_prob / bk_imp - 1 if bk_imp > 0 else 0.0,
            })
```

- [ ] **Step 2: Print avg_clv in backtest_metrics()**

In `backtest_metrics()`, find:
```python
    print(f"    EV moyen       : {df_hist['ev'].mean():+.1%}")
```

Replace with:
```python
    print(f"    EV moyen       : {df_hist['ev'].mean():+.1%}")
    if 'clv' in df_hist.columns:
        print(f"    CLV moyen      : {df_hist['clv'].mean():+.1%}")
```

- [ ] **Step 3: Add avg_clv to backtest_metrics() return dict**

Find the return statement in `backtest_metrics()`:
```python
        'avg_edge': df_hist['edge'].mean(),
        'avg_odd': df_hist['odd'].mean(),
    }
```

Replace with:
```python
        'avg_edge': df_hist['edge'].mean(),
        'avg_odd': df_hist['odd'].mean(),
        'avg_clv': df_hist['clv'].mean() if 'clv' in df_hist.columns else 0.0,
    }
```

- [ ] **Step 4: Verify syntax**

```bash
cd E:/Claude/botbet/tennis/tennis_ml
venv/Scripts/python -c "import ast; ast.parse(open('src/backtest_real.py').read()); print('OK')"
```

Expected: `OK`

---

### Task 2: Save unfiltered candidates parquet in backtest_real.py main

**Files:**
- Modify: `src/backtest_real.py` (around line 583)

This parquet (min_edge=0, min_prob=0.50) gives `optimize_thresholds.py` the full dataset to grid-search over — including bets that would be filtered out by the current thresholds.

- [ ] **Step 1: Add raw candidates save after the join step**

In `src/backtest_real.py` main, find:
```python
    # ── Backtests par bookmaker (Flat 10€) ───────────────────────────────────
    print("\n── Backtests par bookmaker (Flat 10€) ───────────────")
```

Insert BEFORE that block:
```python
    # ── Candidats bruts pour optimize_thresholds.py ──────────────────────────
    print("\n── Sauvegarde candidats bruts ───────────────────────")
    hist_all = run_backtest(
        df_joined,
        odds_col_w='PSW', odds_col_l='PSL',
        min_edge=0.0, min_prob=0.50, min_odd=1.10,
        bankroll_init=BANKROLL, strategy='flat', flat_stake=10.0,
    )
    if len(hist_all) > 0:
        hist_all.to_parquet(MODELS_DIR / "backtest_all_candidates.parquet", index=False)
        print(f"  {len(hist_all):,} candidats sauvegardés → backtrack_all_candidates.parquet")
        print(f"  CLV moyen  : {hist_all['clv'].mean():+.1%}")
        print(f"  Edge moyen : {hist_all['edge'].mean():+.1%}")

```

- [ ] **Step 2: Verify syntax**

```bash
cd E:/Claude/botbet/tennis/tennis_ml
venv/Scripts/python -c "import ast; ast.parse(open('src/backtest_real.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Re-run backtest ATP to verify CLV appears and parquet is created**

```bash
cd E:/Claude/botbet/tennis/tennis_ml
venv/Scripts/python src/backtest_real.py --tour atp 2>&1 | head -80
```

Expected output includes:
- `CLV moyen` printed in each bookmaker result
- `X,XXX candidats sauvegardés → backtest_all_candidates.parquet`

- [ ] **Step 4: Verify parquet columns**

```bash
venv/Scripts/python -c "
import pandas as pd
df = pd.read_parquet('data/models/atp/backtest_all_candidates.parquet')
print('Colonnes:', list(df.columns))
print('N bets:', len(df))
print('CLV moyen:', df['clv'].mean())
print('Edge moyen:', df['edge'].mean())
"
```

Expected: columns include `clv`, `edge`, `our_prob`, `odd`, `won`, `pnl`, `level`, `surface`

- [ ] **Step 5: Re-run backtest WTA**

```bash
venv/Scripts/python src/backtest_real.py --tour wta 2>&1 | tail -20
```

---

## Chunk 2: optimize_thresholds.py — Grid Search

### Task 3: Create src/optimize_thresholds.py

**Files:**
- Create: `src/optimize_thresholds.py`

- [ ] **Step 1: Create the file**

```python
# src/optimize_thresholds.py
"""
Optimisation des seuils de pari (min_edge × min_prob) sur données Pinnacle.

Usage:
    python src/optimize_thresholds.py --tour atp
    python src/optimize_thresholds.py --tour wta --min-bets 20

Prérequis : avoir lancé backtest_real.py pour générer backtest_all_candidates.parquet

Output:
    data/models/{tour}/optimal_thresholds.json
"""

import sys
import json
import argparse
import warnings
import itertools
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from pathlib import Path

from config import get_paths, make_dirs


def compute_threshold_grid(df: pd.DataFrame,
                            min_edge_range: list,
                            min_prob_range: list,
                            min_bets: int = 30) -> pd.DataFrame:
    """
    Grid search (min_edge × min_prob) → tableau ROI / N_bets / Sharpe.

    df doit avoir les colonnes : edge, our_prob, pnl, stake, won
    """
    rows = []
    for min_edge, min_prob in itertools.product(min_edge_range, min_prob_range):
        sub = df[(df['edge'] >= min_edge) & (df['our_prob'] >= min_prob)]
        n = len(sub)
        if n < min_bets:
            roi = np.nan
            sharpe = np.nan
        else:
            roi = sub['pnl'].sum() / sub['stake'].sum()
            sharpe = (sub['pnl'].mean() / sub['pnl'].std() * np.sqrt(252)
                      if sub['pnl'].std() > 0 else 0.0)
        rows.append({
            'min_edge': round(min_edge, 3),
            'min_prob': round(min_prob, 3),
            'n_bets'  : n,
            'roi'     : roi,
            'sharpe'  : sharpe,
        })
    return pd.DataFrame(rows)


def analyse_by_group(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """ROI / n_bets / win_rate groupé par une colonne (level, surface)."""
    if col not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby(col)
        .apply(lambda g: pd.Series({
            'n_bets'  : len(g),
            'win_rate': round(g['won'].mean(), 3),
            'roi'     : round(g['pnl'].sum() / g['stake'].sum(), 3),
            'pnl'     : round(g['pnl'].sum(), 1),
            'avg_odd' : round(g['odd'].mean(), 2),
            'avg_clv' : round(g['clv'].mean(), 3) if 'clv' in g.columns else np.nan,
        }))
        .sort_values('roi', ascending=False)
    )


def main():
    parser = argparse.ArgumentParser(description="Optimisation seuils de pari")
    parser.add_argument('--tour',     default='atp', choices=['atp', 'wta'])
    parser.add_argument('--min-bets', type=int, default=30,
                        help="Minimum de paris requis pour valider un combo (défaut: 30)")
    args = parser.parse_args()

    tour  = args.tour.lower()
    paths = get_paths(tour)
    MODELS_DIR = paths['models_dir']

    print("=" * 55)
    print(f"THRESHOLD OPTIMIZER — {tour.upper()}")
    print("=" * 55)

    # ── Chargement des candidats bruts ───────────────────────────────────────
    raw_path = MODELS_DIR / "backtest_all_candidates.parquet"
    if not raw_path.exists():
        print(f"\n  ERREUR : {raw_path} non trouvé.")
        print("  Lance d'abord : python src/backtest_real.py")
        sys.exit(1)

    df = pd.read_parquet(raw_path)
    print(f"\n  Candidats chargés : {len(df):,} paris")
    print(f"  Période           : {df['date'].min()} → {df['date'].max()}")
    print(f"  Edge moyen        : {df['edge'].mean():+.1%}")
    if 'clv' in df.columns:
        print(f"  CLV moyen         : {df['clv'].mean():+.1%}")

    # ── Grid search ──────────────────────────────────────────────────────────
    min_edge_range = [i / 100 for i in range(0, 12)]          # 0% → 11%
    min_prob_range = [0.50 + i * 0.01 for i in range(17)]     # 0.50 → 0.66

    n_combos = len(min_edge_range) * len(min_prob_range)
    print(f"\n  Grid search : {len(min_edge_range)} × {len(min_prob_range)} = {n_combos} combos")
    print(f"  Minimum bets requis : {args.min_bets}")

    df_grid = compute_threshold_grid(df, min_edge_range, min_prob_range, args.min_bets)

    valid = df_grid.dropna(subset=['roi'])
    if len(valid) == 0:
        print("\n  Aucun combo valide (trop peu de paris partout).")
        print("  Diminue --min-bets ou relance backtest_real.py avec des données supplémentaires.")
        sys.exit(1)

    best_roi    = valid.loc[valid['roi'].idxmax()].to_dict()
    best_sharpe = valid.loc[valid['sharpe'].idxmax()].to_dict()

    # ── Résultats ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("RÉSULTATS")
    print("=" * 55)

    print(f"\n  Meilleur ROI :")
    print(f"    min_edge : {best_roi['min_edge']:.0%}")
    print(f"    min_prob : {best_roi['min_prob']:.2f}")
    print(f"    n_bets   : {int(best_roi['n_bets'])}")
    print(f"    ROI      : {best_roi['roi']:+.2%}")
    print(f"    Sharpe   : {best_roi['sharpe']:.2f}")

    print(f"\n  Meilleur Sharpe :")
    print(f"    min_edge : {best_sharpe['min_edge']:.0%}")
    print(f"    min_prob : {best_sharpe['min_prob']:.2f}")
    print(f"    n_bets   : {int(best_sharpe['n_bets'])}")
    print(f"    ROI      : {best_sharpe['roi']:+.2%}")
    print(f"    Sharpe   : {best_sharpe['sharpe']:.2f}")

    # ── Analyse par niveau et surface au threshold optimal ────────────────────
    sub_opt = df[
        (df['edge'] >= best_roi['min_edge']) &
        (df['our_prob'] >= best_roi['min_prob'])
    ]

    print(f"\n  Paris au threshold optimal : {len(sub_opt):,}")

    print("\n  ROI par niveau de tournoi :")
    by_level = analyse_by_group(sub_opt, 'level')
    if not by_level.empty:
        print(by_level[['n_bets', 'win_rate', 'roi', 'pnl', 'avg_odd']].to_string())
        profitable_levels = by_level[by_level['roi'] > 0].index.tolist()
    else:
        profitable_levels = []

    print("\n  ROI par surface :")
    by_surface = analyse_by_group(sub_opt, 'surface')
    if not by_surface.empty:
        print(by_surface[['n_bets', 'win_rate', 'roi', 'pnl', 'avg_odd']].to_string())
        profitable_surfaces = by_surface[by_surface['roi'] > 0].index.tolist()
    else:
        profitable_surfaces = []

    print("\n  Top 15 combos par ROI :")
    print(valid.sort_values('roi', ascending=False).head(15).to_string(index=False))

    # ── Sauvegarde JSON ───────────────────────────────────────────────────────
    out = MODELS_DIR / "optimal_thresholds.json"
    result = {
        'tour'       : tour,
        'n_candidates': len(df),
        'best_roi'   : {
            'min_edge'  : float(best_roi['min_edge']),
            'min_prob'  : float(best_roi['min_prob']),
            'n_bets'    : int(best_roi['n_bets']),
            'roi'       : float(best_roi['roi']),
            'sharpe'    : float(best_roi['sharpe']),
        },
        'best_sharpe': {
            'min_edge'  : float(best_sharpe['min_edge']),
            'min_prob'  : float(best_sharpe['min_prob']),
            'n_bets'    : int(best_sharpe['n_bets']),
            'roi'       : float(best_sharpe['roi']),
            'sharpe'    : float(best_sharpe['sharpe']),
        },
        'profitable_levels'  : profitable_levels,
        'profitable_surfaces': profitable_surfaces,
    }
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n  Thresholds sauvegardés → {out}")
    print(f"\nOptimisation terminée.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

```bash
cd E:/Claude/botbet/tennis/tennis_ml
venv/Scripts/python -c "import ast; ast.parse(open('src/optimize_thresholds.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Run optimizer ATP**

```bash
venv/Scripts/python src/optimize_thresholds.py --tour atp
```

Expected output:
- Grid search table printed
- `Thresholds sauvegardés → data/models/atp/optimal_thresholds.json`
- ROI by level and surface tables printed

- [ ] **Step 4: Verify JSON created**

```bash
venv/Scripts/python -c "
import json
with open('data/models/atp/optimal_thresholds.json') as f:
    d = json.load(f)
print('best_roi:', d['best_roi'])
print('profitable_levels:', d['profitable_levels'])
"
```

Expected: JSON with keys `best_roi`, `best_sharpe`, `profitable_levels`, `profitable_surfaces`

- [ ] **Step 5: Run optimizer WTA**

```bash
venv/Scripts/python src/optimize_thresholds.py --tour wta
```

---

## Chunk 3: Dashboard — Production Integration

### Task 4: Load optimal thresholds in dashboard + replace hardcoded is_value

**Files:**
- Modify: `src/dashboard.py`

The dashboard currently uses `is_value = edge >= 0.03 and prob >= 0.35` (hardcoded). This task replaces that with the optimized thresholds from JSON.

- [ ] **Step 1: Add load_optimal_thresholds() function**

In `src/dashboard.py`, after the `load_tournament_names()` function (around line 200), add:

```python
@st.cache_data
def load_optimal_thresholds(tour: str) -> dict:
    """Charge les thresholds optimisés depuis JSON (generé par optimize_thresholds.py)."""
    p = ROOT / "data" / "models" / tour / "optimal_thresholds.json"
    if p.exists():
        return json.loads(p.read_text())
    # Valeurs par défaut si JSON absent
    return {
        "best_roi": {"min_edge": 0.03, "min_prob": 0.55},
        "profitable_levels": [],
        "profitable_surfaces": [],
    }
```

- [ ] **Step 2: Load thresholds in render_predictions_tab()**

In `render_predictions_tab()`, after the `try: model, imputer...` block (around line 1102–1106), find:

```python
    player_names    = load_player_names(tour)
```

Insert BEFORE that line:
```python
    thresholds = load_optimal_thresholds(tour)
    opt_edge   = thresholds.get("best_roi", {}).get("min_edge", 0.03)
    opt_prob   = thresholds.get("best_roi", {}).get("min_prob", 0.55)

```

- [ ] **Step 3: Replace hardcoded is_value threshold**

Find (line ~1234):
```python
                is_value = edge >= 0.03 and prob >= 0.35
```

Replace with:
```python
                is_value = edge >= opt_edge and prob >= opt_prob
```

- [ ] **Step 4: Add CLV display in the Value Bets section**

In `render_predictions_tab()`, after:
```python
        edge1, ev1, imp1 = compute_edge(prob_p1, odd1, odd2)
        edge2, ev2, imp2 = compute_edge(prob_p2, odd2, odd1)
```

Add:
```python
        clv1 = prob_p1 / imp1 - 1 if imp1 > 0 else 0.0
        clv2 = prob_p2 / imp2 - 1 if imp2 > 0 else 0.0
```

Then in the loop `for i, (name, prob, odds, edge, ev, imp) in enumerate([...])`, update the tuple to include clv:

Find:
```python
        for i, (name, prob, odds, edge, ev, imp) in enumerate([
            (r["p1_name"], prob_p1, odd1, edge1, ev1, imp1),
            (r["p2_name"], prob_p2, odd2, edge2, ev2, imp2),
        ]):
```

Replace with:
```python
        for i, (name, prob, odds, edge, ev, imp, clv) in enumerate([
            (r["p1_name"], prob_p1, odd1, edge1, ev1, imp1, clv1),
            (r["p2_name"], prob_p2, odd2, edge2, ev2, imp2, clv2),
        ]):
```

Then after `value_card(name, odds, imp, prob, edge, ev, is_value)`, add CLV + GO/PASS display:

```python
                # CLV + GO/PASS badge
                clv_color  = "#3DFFA0" if clv >= 0 else "#FF4D6D"
                go_bg      = "rgba(61,255,160,0.10)" if is_value else "rgba(255,77,109,0.08)"
                go_border  = "rgba(61,255,160,0.30)" if is_value else "rgba(255,77,109,0.20)"
                go_text    = "GO" if is_value else "PASS"
                go_color   = "#3DFFA0" if is_value else "#FF4D6D"
                threshold_tip = f"Seuil optimal : edge ≥ {opt_edge:.0%} · prob ≥ {opt_prob:.0%}"
                st.html(f"""
                <div style="display:flex; gap:8px; align-items:center; margin-top:6px; margin-bottom:4px;">
                  <div style="flex:1; background:{go_bg}; border:1px solid {go_border};
                              border-radius:8px; padding:8px 14px; display:flex;
                              justify-content:space-between; align-items:center;">
                    <span style="font-family:'DM Mono',monospace; font-size:0.6rem;
                                 letter-spacing:1.5px; text-transform:uppercase; color:#4E6A90;">
                      {go_text}
                    </span>
                    <span style="font-family:'Syne',sans-serif; font-size:1.1rem;
                                 font-weight:800; color:{go_color};">{go_text}</span>
                  </div>
                  <div style="background:rgba(10,13,26,0.7); border:1px solid rgba(80,110,180,0.12);
                              border-radius:8px; padding:8px 14px; text-align:center;">
                    <div style="font-family:'DM Mono',monospace; font-size:0.52rem;
                                letter-spacing:1.5px; text-transform:uppercase; color:#4E6A90;">CLV</div>
                    <div style="font-family:'DM Mono',monospace; font-size:1rem;
                                color:{clv_color}; font-weight:600;">{clv:+.1%}</div>
                  </div>
                </div>
                <div style="font-family:'DM Mono',monospace; font-size:0.6rem;
                            color:#2A3D5A; margin-bottom:6px;">{threshold_tip}</div>
                """)
```

- [ ] **Step 5: Verify syntax**

```bash
cd E:/Claude/botbet/tennis/tennis_ml
venv/Scripts/python -c "import ast; ast.parse(open('src/dashboard.py').read()); print('OK')"
```

Expected: `OK`

---

### Task 5: Add CLV to Today tab popover + Analyse section in Stats tab

**Files:**
- Modify: `src/dashboard.py`

**Part A — CLV in Today tab popover:**

- [ ] **Step 1: Add CLV row to the edge/EV table in render_today_tab()**

In `render_today_tab()`, find the HTML table (around line 1415):
```python
                    st.html(f"""
                    <div style="background:#0f1628;border-radius:10px;padding:10px 12px;margin:6px 0;font-size:0.8rem">
                      <div style="color:#94a3b8;margin-bottom:6px">Marge bookmaker : <b style="color:#f1f5f9">{margin:+.1f}%</b></div>
                      <table style="width:100%;border-collapse:collapse">
                        <tr style="color:#64748b;font-size:0.72rem">
                          <td></td><td style="text-align:center">Modèle</td>
                          <td style="text-align:center">Implicite</td>
                          <td style="text-align:center">Edge</td>
                          <td style="text-align:center">EV</td>
                        </tr>
```

Before computing the table, add CLV computation:
```python
                    # CLV computation for today tab
                    clv1_today = prob_p1 / imp1 - 1 if imp1 > 0 else 0.0
                    clv2_today = prob_p2 / imp2 - 1 if imp2 > 0 else 0.0
```

Then update the table header and rows to add CLV column:

Find the `<tr style="color:#64748b;...">` header row and the two player rows. Replace the entire `st.html(...)` table block with:

```python
                    st.html(f"""
                    <div style="background:#0f1628;border-radius:10px;padding:10px 12px;margin:6px 0;font-size:0.8rem">
                      <div style="color:#94a3b8;margin-bottom:6px">Marge bookmaker : <b style="color:#f1f5f9">{margin:+.1f}%</b></div>
                      <table style="width:100%;border-collapse:collapse">
                        <tr style="color:#64748b;font-size:0.72rem">
                          <td></td><td style="text-align:center">Modèle</td>
                          <td style="text-align:center">Implicite</td>
                          <td style="text-align:center">Edge</td>
                          <td style="text-align:center">EV</td>
                          <td style="text-align:center">CLV</td>
                        </tr>
                        <tr>
                          <td style="color:#cbd5e1;padding:2px 0">{m['p1_name'].split()[0]}</td>
                          <td style="text-align:center;color:#f1f5f9">{prob_p1:.0%}</td>
                          <td style="text-align:center;color:#94a3b8">{imp1:.0%}</td>
                          <td style="text-align:center;color:{_ev_color(edge1)};font-weight:600">{edge1:+.1%}</td>
                          <td style="text-align:center;color:{_ev_color(ev1)};font-weight:600">{ev1:+.1%}</td>
                          <td style="text-align:center;color:{_ev_color(clv1_today)};font-weight:600">{clv1_today:+.1%}</td>
                        </tr>
                        <tr>
                          <td style="color:#cbd5e1;padding:2px 0">{m['p2_name'].split()[0]}</td>
                          <td style="text-align:center;color:#f1f5f9">{prob_p2:.0%}</td>
                          <td style="text-align:center;color:#94a3b8">{imp2:.0%}</td>
                          <td style="text-align:center;color:{_ev_color(edge2)};font-weight:600">{edge2:+.1%}</td>
                          <td style="text-align:center;color:{_ev_color(ev2)};font-weight:600">{ev2:+.1%}</td>
                          <td style="text-align:center;color:{_ev_color(clv2_today)};font-weight:600">{clv2_today:+.1%}</td>
                        </tr>
                      </table>
                    </div>
                    """)
```

**Part B — Analyse section in Stats tab:**

- [ ] **Step 2: Add backtest analysis section to tab_stats**

In `src/dashboard.py`, find the end of the `with tab_stats:` block (the last section, around line 1760+). Add after the last `st.divider()` in that block:

```python
        # ── Analyse Backtest ─────────────────────────────────────────────────
        st.divider()
        section_header("Analyse Backtest (Pinnacle)")

        # Sélection du circuit pour l'analyse
        analyse_tour = st.radio("Circuit analyse", ["ATP", "WTA"],
                                horizontal=True, key="stats_analyse_tour")
        bt_tour = analyse_tour.lower()

        # Charger thresholds et parquet backtest
        bt_thresholds  = load_optimal_thresholds(bt_tour)
        bt_opt         = bt_thresholds.get("best_roi", {})
        bt_parquet     = ROOT / "data" / "models" / bt_tour / "backtest_real_Pinnacle.parquet"

        if bt_parquet.exists():
            df_bt = pd.read_parquet(bt_parquet)

            # KPI thresholds
            col_ta, col_tb, col_tc = st.columns(3)
            with col_ta:
                st.metric("Threshold Edge", f"{bt_opt.get('min_edge', 0.03):.0%}")
            with col_tb:
                st.metric("Threshold Prob", f"{bt_opt.get('min_prob', 0.55):.2f}")
            with col_tc:
                roi_opt = bt_opt.get('roi', 0)
                st.metric("ROI optimal", f"{roi_opt:+.1%}")

            col_lv, col_sv = st.columns(2)

            with col_lv:
                section_header("ROI par niveau")
                if 'level' in df_bt.columns:
                    by_lv = (df_bt.groupby('level')
                             .apply(lambda g: pd.Series({
                                 'Paris': len(g),
                                 'Win%': f"{g['won'].mean():.0%}",
                                 'ROI': f"{g['pnl'].sum()/g['stake'].sum():+.1%}",
                                 'P&L': f"{g['pnl'].sum():+.0f}€",
                             }))
                             .sort_values('ROI', ascending=False)
                             .reset_index())
                    st.dataframe(by_lv, hide_index=True, use_container_width=True)
                    profs = bt_thresholds.get("profitable_levels", [])
                    if profs:
                        st.caption(f"Niveaux rentables : **{', '.join(profs)}**")

            with col_sv:
                section_header("ROI par surface")
                if 'surface' in df_bt.columns:
                    by_sv = (df_bt.groupby('surface')
                             .apply(lambda g: pd.Series({
                                 'Paris': len(g),
                                 'Win%': f"{g['won'].mean():.0%}",
                                 'ROI': f"{g['pnl'].sum()/g['stake'].sum():+.1%}",
                                 'P&L': f"{g['pnl'].sum():+.0f}€",
                             }))
                             .sort_values('ROI', ascending=False)
                             .reset_index())
                    st.dataframe(by_sv, hide_index=True, use_container_width=True)

            # CLV distribution si disponible
            if 'clv' in df_bt.columns:
                section_header("Distribution CLV (Pinnacle)")
                avg_clv = df_bt['clv'].mean()
                pct_pos = (df_bt['clv'] > 0).mean()
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    clv_c = "#3DFFA0" if avg_clv >= 0 else "#FF4D6D"
                    st.metric("CLV moyen", f"{avg_clv:+.1%}")
                with col_c2:
                    st.metric("Bets CLV positif", f"{pct_pos:.0%}")
                st.caption("CLV > 0 : notre modèle bat Pinnacle no-vig sur ce pari — signal d'edge réel.")
        else:
            st.info(f"Lance backtest_real.py --tour {bt_tour} pour générer les données.")
```

- [ ] **Step 3: Verify syntax**

```bash
cd E:/Claude/botbet/tennis/tennis_ml
venv/Scripts/python -c "import ast; ast.parse(open('src/dashboard.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Start dashboard and verify it loads without error**

```bash
cd E:/Claude/botbet/tennis/tennis_ml
venv/Scripts/python -c "
import subprocess, sys
result = subprocess.run(
    [sys.executable, '-m', 'py_compile', 'src/dashboard.py'],
    capture_output=True, text=True
)
print('STDERR:', result.stderr or 'none')
print('Return:', result.returncode)
"
```

Expected: `Return: 0` (no errors)

---

## Résumé des fichiers et pipeline

| Fichier | Action | Rôle |
|---|---|---|
| `src/backtest_real.py` | Modifier | CLV dans history + save loose parquet |
| `src/optimize_thresholds.py` | Créer | Grid search → optimal_thresholds.json |
| `src/dashboard.py` | Modifier | Thresholds + CLV + Analyse tab |
| `data/models/{tour}/backtest_all_candidates.parquet` | Généré | Candidats bruts (min_edge=0) |
| `data/models/{tour}/optimal_thresholds.json` | Généré | Seuils optimaux + niveaux rentables |

## Pipeline production complet (à lancer dans l'ordre)

```bash
# 1. Backtest avec CLV + save candidats bruts
python src/backtest_real.py --tour atp
python src/backtest_real.py --tour wta  # si WTA odds disponibles

# 2. Optimiser les seuils
python src/optimize_thresholds.py --tour atp
python src/optimize_thresholds.py --tour wta

# 3. Dashboard (thresholds chargés automatiquement)
streamlit run src/dashboard.py
```
