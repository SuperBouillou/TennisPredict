"""
Dashboard ATP / WTA — Prédictions + Bankroll + Historique des paris
Lancer : streamlit run src/dashboard.py
"""

import json
import glob
import warnings
warnings.filterwarnings('ignore')

from odds_api_client import fetch_odds_today, merge_odds

import numpy as np
import pandas as pd
import streamlit as st
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG BANKROLL (commune ATP + WTA)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CFG = {
    "initial_bankroll": 1000.0,
    "current_bankroll": 1000.0,
    "stake_method": "flat",
    "stake_value": 10.0,
}

def cfg_path() -> Path:
    return ROOT / "data" / "bankroll.json"

def bets_path() -> Path:
    return ROOT / "data" / "bets.csv"

def load_config() -> dict:
    p = cfg_path()
    if p.exists():
        return json.loads(p.read_text())
    return DEFAULT_CFG.copy()

def save_config(cfg: dict):
    cfg_path().write_text(json.dumps(cfg, indent=2))

# ─────────────────────────────────────────────────────────────────────────────
# HISTORIQUE BETS (commun ATP + WTA — colonne 'tour')
# ─────────────────────────────────────────────────────────────────────────────

BETS_COLS = [
    "date", "tour", "tournament", "surface", "p1_name", "p2_name",
    "bet_on", "odds", "stake", "prob_model", "edge",
    "result", "pnl", "bankroll_after",
]

@st.cache_data
def load_bets() -> pd.DataFrame:
    p = bets_path()
    if p.exists():
        df = pd.read_csv(p)
        for col in BETS_COLS:
            if col not in df.columns:
                df[col] = np.nan
        return df
    dfs = []
    for t in ["atp", "wta"]:
        old = ROOT / "data" / f"bets_{t}.csv"
        if old.exists():
            df_t = pd.read_csv(old)
            df_t["tour"] = t
            dfs.append(df_t)
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        for col in BETS_COLS:
            if col not in df.columns:
                df[col] = np.nan
        save_bets(df)
        return df
    return pd.DataFrame(columns=BETS_COLS)

def save_bets(df: pd.DataFrame):
    df.to_csv(bets_path(), index=False)

def _recalculate_bankroll(df: pd.DataFrame) -> None:
    """Recalculate current_bankroll from scratch: initial + resolved PnL - pending stakes."""
    cfg = load_config()
    bk  = cfg["initial_bankroll"]
    resolved = df[df["result"].isin(["won", "lost"])]
    if not resolved.empty:
        bk += float(resolved["pnl"].dropna().sum())
    pending = df[df["result"] == "pending"]
    if not pending.empty:
        bk -= float(pending["stake"].dropna().sum())
    cfg["current_bankroll"] = round(bk, 2)
    save_config(cfg)

def add_bet(bet: dict) -> pd.DataFrame:
    df  = load_bets()
    new = pd.DataFrame([bet])
    df  = pd.concat([df, new], ignore_index=True)
    save_bets(df)
    # Deduct stake immediately for pending bets
    cfg = load_config()
    cfg["current_bankroll"] = round(cfg["current_bankroll"] - float(bet["stake"]), 2)
    save_config(cfg)
    load_bets.clear()
    return df

def update_bet_result(idx: int, result: str):
    df  = load_bets()
    cfg = load_config()
    row = df.iloc[idx]
    stake = float(row["stake"])
    odds  = float(row["odds"])
    # Display PnL: profit or total loss
    pnl = round(stake * (odds - 1) if result == "won" else -stake, 2)
    # Stake was already deducted at bet recording — on win return full payout
    if result == "won":
        cfg["current_bankroll"] = round(cfg["current_bankroll"] + stake * odds, 2)
    # On loss: stake already gone, no further adjustment
    df.at[idx, "result"]         = result
    df.at[idx, "pnl"]            = pnl
    df.at[idx, "bankroll_after"] = cfg["current_bankroll"]
    save_bets(df)
    save_config(cfg)
    load_bets.clear()

def delete_bet(idx: int):
    df = load_bets()
    df = df.drop(index=idx).reset_index(drop=True)
    save_bets(df)
    _recalculate_bankroll(df)
    load_bets.clear()

def edit_bet(idx: int, new_stake: float, new_odds: float):
    df = load_bets()
    old_stake = float(df.at[idx, "stake"])
    df.at[idx, "stake"] = round(new_stake, 2)
    df.at[idx, "odds"]  = new_odds
    save_bets(df)
    # Adjust bankroll if editing a pending bet (stake reservation changed)
    if str(df.at[idx, "result"]) == "pending":
        cfg = load_config()
        cfg["current_bankroll"] = round(cfg["current_bankroll"] + old_stake - new_stake, 2)
        save_config(cfg)
    load_bets.clear()

# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT MODÈLE
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(tour: str):
    import joblib
    models_dir = ROOT / "data" / "models" / tour
    model    = joblib.load(models_dir / "xgb_tuned.pkl")
    imputer  = joblib.load(models_dir / "imputer.pkl")
    features = joblib.load(models_dir / "feature_list.pkl")
    platt    = joblib.load(models_dir / "platt_scaler.pkl")
    return model, imputer, features, platt

@st.cache_resource
def load_player_data(tour: str):
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from predict_today import load_player_database, load_elo_ratings
    processed_dir = ROOT / "data" / "processed" / tour
    df_players = load_player_database(processed_dir)
    df_elo     = load_elo_ratings(processed_dir)
    return df_players, df_elo

@st.cache_data
def load_player_names(tour: str) -> list:
    processed_dir = ROOT / "data" / "processed" / tour
    path = processed_dir / "player_profiles_updated.parquet"
    if not path.exists():
        path2 = processed_dir / "matches_features_final.parquet"
        if path2.exists():
            df = pd.read_parquet(path2, columns=["p1_name"])
            return sorted(df["p1_name"].dropna().unique().tolist())
        return []
    df = pd.read_parquet(path, columns=["player_name"])
    return sorted(df["player_name"].dropna().unique().tolist())

@st.cache_data
def load_tournament_names(tour: str) -> list:
    # Parquet en priorité (100x plus rapide qu'openpyxl)
    processed_dir = ROOT / "data" / "processed" / tour
    path = processed_dir / "matches_features_final.parquet"
    if path.exists():
        df = pd.read_parquet(path, columns=["tourney_name"])
        return sorted(df["tourney_name"].dropna().unique().tolist())
    # Fallback : lire les xlsx tennis-data si le parquet est absent
    tourneys = set()
    odds_dir = ROOT / "data" / "odds" / tour
    for f in glob.glob(str(odds_dir / f"{tour}_202*.xlsx")):
        try:
            d = pd.read_excel(f, engine="openpyxl", usecols=["Tournament"])
            tourneys.update(d["Tournament"].dropna().unique().tolist())
        except Exception:
            pass
    return sorted(tourneys)

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

# ─────────────────────────────────────────────────────────────────────────────
# PRÉDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_match(p1_name, p2_name, tournament, surface, best_of, round_name,
                  model, imputer, features, df_players, df_elo, platt=None) -> dict:
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from predict_today import find_player, build_feature_vector

    p1    = find_player(p1_name, df_players, df_elo)
    p2    = find_player(p2_name, df_players, df_elo)
    match = {"surface": surface, "best_of": best_of,
             "tournament": tournament, "round": round_name}

    def _pred(pa, pb):
        Xv   = build_feature_vector(match, pa, pb, features, df_elo).reshape(1, -1)
        Ximp = imputer.transform(Xv)
        p    = float(model.predict_proba(Ximp)[0, 1])
        return float(platt.predict_proba([[p]])[0, 1]) if platt else p

    prob_fwd = _pred(p1, p2)
    prob_bwd = _pred(p2, p1)
    prob_p1  = (prob_fwd + (1 - prob_bwd)) / 2

    return {
        "p1_name": p1_name, "p2_name": p2_name,
        "prob_p1": prob_p1, "prob_p2": 1 - prob_p1,
        "p1_found": bool(p1), "p2_found": bool(p2),
    }

def compute_edge(prob_model: float, odds: float, odds_other: float) -> tuple:
    if odds <= 1 or odds_other <= 1:
        return 0, 0, 0
    margin   = 1/odds + 1/odds_other - 1
    imp_prob = (1/odds) / (1 + margin)
    return prob_model - imp_prob, prob_model * odds - 1, imp_prob

def kelly_stake(prob: float, odds: float, bankroll: float, fraction: float = 1.0) -> float:
    b = odds - 1
    return max(0, (b * prob - (1 - prob)) / b * fraction * bankroll)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Tennis Predictor", page_icon="🎾", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM CSS
# ─────────────────────────────────────────────────────────────────────────────

st.html("""
<style>
/* ── FONTS ──────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,400&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

/* ── ROOT VARS ───────────────────────────────────────────────────────────── */
:root {
  --bg:        #07090F;
  --bg-s:      #0C1020;
  --bg-card:   #0F1628;
  --bg-input:  #0A0D1A;
  --border:    rgba(80, 110, 180, 0.12);
  --border-h:  rgba(61, 255, 160, 0.28);
  --text:      #D8E8FF;
  --text-2:    #4E6A90;
  --text-3:    #2A3D5A;
  --green:     #3DFFA0;
  --green-d:   rgba(61,255,160,0.12);
  --red:       #FF4D6D;
  --red-d:     rgba(255,77,109,0.10);
  --blue:      #4D9EFF;
  --gold:      #FFB347;
}

/* ── BASE ────────────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif !important;
  color: var(--text) !important;
}

.stApp {
  background-color: var(--bg) !important;
  background-image:
    radial-gradient(ellipse 70% 45% at 15% 5%,  rgba(61,255,160,0.035) 0%, transparent 65%),
    radial-gradient(ellipse 55% 40% at 85% 95%, rgba(77,158,255,0.035) 0%, transparent 65%),
    radial-gradient(circle at 1px 1px, rgba(80,110,180,0.055) 1px, transparent 0);
  background-size: auto, auto, 30px 30px;
}

/* ── SIDEBAR ─────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: var(--bg-s) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div { padding-top: 16px !important; }

/* ── TOP-LEVEL TABS ──────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  gap: 2px !important;
  border-bottom: 1px solid var(--border) !important;
  padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
  font-family: 'Syne', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.6px !important;
  color: var(--text-2) !important;
  background: transparent !important;
  border: none !important;
  border-radius: 8px 8px 0 0 !important;
  padding: 10px 22px !important;
  transition: color 0.2s, background 0.2s !important;
}
.stTabs [aria-selected="true"] {
  color: var(--green) !important;
  background: rgba(61,255,160,0.05) !important;
}
.stTabs [data-baseweb="tab-highlight"] {
  background: var(--green) !important;
  height: 2px !important;
}
.stTabs [data-baseweb="tab-border"] { background: var(--border) !important; }

/* ── METRICS ─────────────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 16px 18px !important;
  transition: border-color 0.2s, transform 0.15s !important;
}
[data-testid="metric-container"]:hover {
  border-color: var(--border-h) !important;
  transform: translateY(-1px) !important;
}
[data-testid="stMetricLabel"] > div {
  font-family: 'DM Mono', monospace !important;
  font-size: 0.6rem !important;
  letter-spacing: 1.8px !important;
  text-transform: uppercase !important;
  color: var(--text-2) !important;
  opacity: 1 !important;
}
[data-testid="stMetricValue"] > div {
  font-family: 'DM Mono', monospace !important;
  font-size: 1.55rem !important;
  font-weight: 400 !important;
  color: var(--text) !important;
  line-height: 1.15 !important;
}
[data-testid="stMetricDelta"] > div {
  font-family: 'DM Mono', monospace !important;
  font-size: 0.7rem !important;
}

/* ── BUTTONS ─────────────────────────────────────────────────────────────── */
.stButton > button {
  font-family: 'Syne', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.5px !important;
  border-radius: 8px !important;
  border: 1px solid var(--border) !important;
  background: rgba(255,255,255,0.02) !important;
  color: var(--text-2) !important;
  transition: all 0.2s !important;
  padding: 6px 14px !important;
}
.stButton > button:hover {
  border-color: var(--border-h) !important;
  background: rgba(61,255,160,0.05) !important;
  color: var(--green) !important;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, rgba(61,255,160,0.12), rgba(77,158,255,0.08)) !important;
  border: 1px solid rgba(61,255,160,0.30) !important;
  color: var(--green) !important;
}
.stButton > button[kind="primary"]:hover {
  background: linear-gradient(135deg, rgba(61,255,160,0.22), rgba(77,158,255,0.14)) !important;
  border-color: rgba(61,255,160,0.55) !important;
  color: var(--green) !important;
  box-shadow: 0 4px 24px rgba(61,255,160,0.12) !important;
  transform: translateY(-1px) !important;
}

/* ── FORM SUBMIT ─────────────────────────────────────────────────────────── */
.stFormSubmitButton > button {
  width: 100% !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  font-size: 0.82rem !important;
  letter-spacing: 2px !important;
  text-transform: uppercase !important;
  border-radius: 10px !important;
  padding: 13px 20px !important;
  background: linear-gradient(135deg, rgba(61,255,160,0.14), rgba(77,158,255,0.09)) !important;
  border: 1px solid rgba(61,255,160,0.35) !important;
  color: var(--green) !important;
  transition: all 0.25s !important;
}
.stFormSubmitButton > button:hover {
  background: linear-gradient(135deg, rgba(61,255,160,0.24), rgba(77,158,255,0.16)) !important;
  border-color: rgba(61,255,160,0.60) !important;
  box-shadow: 0 6px 32px rgba(61,255,160,0.16) !important;
  transform: translateY(-1px) !important;
}

/* ── INPUTS ──────────────────────────────────────────────────────────────── */
.stSelectbox label, .stNumberInput label, .stRadio label > div {
  font-family: 'DM Mono', monospace !important;
  font-size: 0.65rem !important;
  letter-spacing: 1.2px !important;
  text-transform: uppercase !important;
  color: var(--text-2) !important;
}
.stSelectbox > div > div {
  background: var(--bg-input) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.88rem !important;
}
.stSelectbox > div > div:focus-within {
  border-color: rgba(61,255,160,0.35) !important;
  box-shadow: 0 0 0 3px rgba(61,255,160,0.06) !important;
}
.stNumberInput > div > div > input {
  background: var(--bg-input) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 0.92rem !important;
}
.stNumberInput > div > div > input:focus {
  border-color: rgba(61,255,160,0.35) !important;
  box-shadow: 0 0 0 3px rgba(61,255,160,0.06) !important;
}
div[data-baseweb="input"] {
  background: var(--bg-input) !important;
  border-radius: 8px !important;
}
div[data-baseweb="select"] > div {
  background: var(--bg-input) !important;
}

/* ── RADIO ───────────────────────────────────────────────────────────────── */
.stRadio > div { gap: 6px !important; }
.stRadio label {
  font-family: 'DM Mono', monospace !important;
  font-size: 0.8rem !important;
  color: var(--text-2) !important;
}

/* ── EXPANDER ────────────────────────────────────────────────────────────── */
details > summary {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  font-family: 'Syne', sans-serif !important;
  font-size: 0.8rem !important;
  color: var(--text-2) !important;
  padding: 10px 14px !important;
  transition: color 0.2s !important;
}
details > summary:hover { color: var(--text) !important; }
details[open] > summary { border-radius: 8px 8px 0 0 !important; border-bottom: none !important; }
details > div {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-top: none !important;
  border-radius: 0 0 8px 8px !important;
  padding: 12px !important;
}

/* ── DATAFRAME ───────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
  border-radius: 12px !important;
  overflow: hidden !important;
  border: 1px solid var(--border) !important;
}

/* ── CONTAINERS ──────────────────────────────────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  transition: border-color 0.2s !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
  border-color: rgba(80,110,180,0.22) !important;
}

/* ── FORM CONTAINER ──────────────────────────────────────────────────────── */
[data-testid="stForm"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  padding: 6px !important;
}

/* ── ALERTS ──────────────────────────────────────────────────────────────── */
.stSuccess > div {
  background: rgba(61,255,160,0.07) !important;
  border: 1px solid rgba(61,255,160,0.25) !important;
  border-radius: 8px !important;
  color: var(--green) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 0.82rem !important;
}
.stWarning > div {
  background: rgba(255,179,71,0.07) !important;
  border: 1px solid rgba(255,179,71,0.25) !important;
  border-radius: 8px !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 0.82rem !important;
}
.stError > div {
  background: rgba(255,77,109,0.07) !important;
  border: 1px solid rgba(255,77,109,0.25) !important;
  border-radius: 8px !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 0.82rem !important;
}

/* ── SPINNER ─────────────────────────────────────────────────────────────── */
.stSpinner > div { border-top-color: var(--green) !important; }

/* ── DIVIDER ─────────────────────────────────────────────────────────────── */
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 20px 0 !important; }

/* ── CAPTION ─────────────────────────────────────────────────────────────── */
.stCaption, .stCaption p {
  font-family: 'DM Mono', monospace !important;
  font-size: 0.62rem !important;
  color: var(--text-3) !important;
  letter-spacing: 0.3px !important;
}

/* ── SECTION HEADER ──────────────────────────────────────────────────────── */
h4 {
  font-family: 'Syne', sans-serif !important;
  font-size: 0.9rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.5px !important;
  color: var(--text) !important;
  margin: 20px 0 12px !important;
}

/* ── LINE CHART ──────────────────────────────────────────────────────────── */
[data-testid="stArrowVegaLiteChart"] { border-radius: 12px !important; overflow: hidden !important; }

/* ── SCROLLBAR ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--text-3); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-2); }

/* ── ANIMATIONS ──────────────────────────────────────────────────────────── */
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse-ring {
  0%, 100% { box-shadow: 0 0 0 0 rgba(61,255,160,0.0),  inset 0 0 0 0 rgba(61,255,160,0.0); }
  50%       { box-shadow: 0 0 0 4px rgba(61,255,160,0.10), inset 0 0 20px rgba(61,255,160,0.03); }
}
@keyframes slide-in-right {
  from { opacity: 0; transform: translateX(20px); }
  to   { opacity: 1; transform: translateX(0); }
}
@keyframes grow-bar {
  from { width: 0% !important; }
}
@keyframes tick-pulse {
  0%, 100% { opacity: 0.3; }
  50%       { opacity: 0.7; }
}

.fade-up   { animation: fadeInUp 0.38s cubic-bezier(0.34,1.1,0.64,1) both; }
.fade-up-2 { animation: fadeInUp 0.38s cubic-bezier(0.34,1.1,0.64,1) 0.08s both; }
.fade-up-3 { animation: fadeInUp 0.38s cubic-bezier(0.34,1.1,0.64,1) 0.16s both; }
.slide-r   { animation: slide-in-right 0.36s cubic-bezier(0.34,1.1,0.64,1) both; }

.value-glow { animation: pulse-ring 2.5s ease-in-out infinite; }

/* ── VEGALTECHART THEMING ────────────────────────────────────────────────── */
[data-testid="stArrowVegaLiteChart"] canvas {
  border-radius: 10px !important;
}

/* ── SIDEBAR LINKS ───────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] a {
  color: var(--blue) !important;
  text-decoration: none !important;
}

/* ── PROGRESS BAR ────────────────────────────────────────────────────────── */
.stProgress > div > div > div > div {
  background: linear-gradient(90deg, var(--green), var(--blue)) !important;
}

/* ── TOOLTIP-LIKE CAPTION ────────────────────────────────────────────────── */
abbr[title] {
  cursor: help;
  text-decoration: underline dotted rgba(80,110,180,0.4);
}

/* ── FORM FIELD GROUPS ───────────────────────────────────────────────────── */
.field-group {
  background: rgba(10,13,26,0.5);
  border: 1px solid rgba(80,110,180,0.10);
  border-radius: 12px;
  padding: 16px;
  margin-bottom: 12px;
}

/* ── MATCH RESULT HEADER ─────────────────────────────────────────────────── */
.match-header {
  background: linear-gradient(135deg, rgba(15,22,40,0.95), rgba(12,16,32,0.95));
  border: 1px solid rgba(80,110,180,0.15);
  border-radius: 16px;
  overflow: hidden;
  position: relative;
}
.match-header::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(61,255,160,0.4), transparent);
}

/* ── KPI CARD ────────────────────────────────────────────────────────────── */
.kpi-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 18px 20px;
  transition: border-color 0.2s, transform 0.15s;
}
.kpi-card:hover {
  border-color: var(--border-h);
  transform: translateY(-1px);
}
</style>
""")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS VISUELS
# ─────────────────────────────────────────────────────────────────────────────

def get_initials(name: str) -> str:
    parts = name.strip().split()
    if len(parts) >= 2:
        return (parts[0][0] + parts[-1][0]).upper()
    return name[:2].upper() if name else "??"


def prob_bar(p1: float, p2: float, name1: str, name2: str):
    i1 = get_initials(name1)
    i2 = get_initials(name2)
    pbar_g = int(p1 * 100)
    pbar_r = int(p2 * 100)

    # Confidence label
    diff = abs(p1 - p2)
    if diff < 0.08:
        conf_label = "MATCH SERRÉ"
        conf_color = "#FFB347"
    elif diff < 0.20:
        conf_label = "LÉGER AVANTAGE"
        conf_color = "#4D9EFF"
    elif diff < 0.35:
        conf_label = "AVANTAGE CLAIR"
        conf_color = "#3DFFA0"
    else:
        conf_label = "TRÈS FAVORI"
        conf_color = "#3DFFA0"

    st.html(f"""
    <div class="fade-up" style="margin: 28px 0 20px; font-family: 'DM Sans', sans-serif;">

      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:18px;">

        <div style="display:flex; align-items:center; gap:12px;">
          <div style="width:48px; height:48px; border-radius:50%; flex-shrink:0;
                      background:linear-gradient(135deg,#3DFFA0,#4D9EFF);
                      display:flex; align-items:center; justify-content:center;
                      font-family:'DM Mono',monospace; font-size:0.72rem; font-weight:500;
                      color:#07090F; letter-spacing:0.5px;
                      box-shadow:0 0 20px rgba(61,255,160,0.30), 0 0 40px rgba(61,255,160,0.08);">
            {i1}
          </div>
          <div>
            <div style="font-family:'Syne',sans-serif; font-size:0.9rem; font-weight:700;
                        color:#D8E8FF; margin-bottom:3px; max-width:180px;
                        overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{name1}</div>
            <div style="font-family:'DM Mono',monospace; font-size:1.7rem; font-weight:400;
                        color:#3DFFA0; line-height:1; letter-spacing:-0.5px;">{p1:.1%}</div>
          </div>
        </div>

        <div style="display:flex; flex-direction:column; align-items:center; gap:4px;">
          <div style="font-family:'DM Mono',monospace; font-size:0.55rem; letter-spacing:2.5px;
                      color:#2A3D5A; text-transform:uppercase;">vs</div>
          <div style="font-family:'DM Mono',monospace; font-size:0.52rem; letter-spacing:1.5px;
                      color:{conf_color}; text-transform:uppercase; white-space:nowrap;
                      background:rgba(80,110,180,0.06); border:1px solid rgba(80,110,180,0.12);
                      border-radius:20px; padding:2px 8px;">{conf_label}</div>
        </div>

        <div style="display:flex; align-items:center; gap:12px; flex-direction:row-reverse;">
          <div style="width:48px; height:48px; border-radius:50%; flex-shrink:0;
                      background:linear-gradient(135deg,#FF4D6D,#FF8C42);
                      display:flex; align-items:center; justify-content:center;
                      font-family:'DM Mono',monospace; font-size:0.72rem; font-weight:500;
                      color:#fff; letter-spacing:0.5px;
                      box-shadow:0 0 20px rgba(255,77,109,0.30), 0 0 40px rgba(255,77,109,0.08);">
            {i2}
          </div>
          <div style="text-align:right;">
            <div style="font-family:'Syne',sans-serif; font-size:0.9rem; font-weight:700;
                        color:#D8E8FF; margin-bottom:3px; max-width:180px;
                        overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{name2}</div>
            <div style="font-family:'DM Mono',monospace; font-size:1.7rem; font-weight:400;
                        color:#FF4D6D; line-height:1; letter-spacing:-0.5px;">{p2:.1%}</div>
          </div>
        </div>

      </div>

      <!-- Bar -->
      <div style="position:relative; height:8px; border-radius:8px; overflow:hidden;
                  background:rgba(80,110,180,0.08); margin-bottom:10px;">
        <div style="position:absolute; left:0; width:{pbar_g}%; height:100%;
                    background:linear-gradient(90deg,#3DFFA0,#4D9EFF);
                    border-radius:8px 0 0 8px;
                    animation: grow-bar 0.8s cubic-bezier(0.34,1.1,0.64,1) both;"></div>
        <div style="position:absolute; right:0; width:{pbar_r}%; height:100%;
                    background:linear-gradient(270deg,#FF4D6D,#FF8C42);
                    border-radius:0 8px 8px 0;
                    animation: grow-bar 0.8s cubic-bezier(0.34,1.1,0.64,1) 0.1s both;"></div>
        <!-- Center tick -->
        <div style="position:absolute; left:50%; transform:translateX(-50%);
                    width:2px; height:100%; background:#07090F; opacity:0.8;"></div>
        <!-- Quarter ticks -->
        <div style="position:absolute; left:25%; width:1px; height:100%;
                    background:rgba(80,110,180,0.25);"></div>
        <div style="position:absolute; left:75%; width:1px; height:100%;
                    background:rgba(80,110,180,0.25);"></div>
      </div>

      <!-- Tick labels -->
      <div style="position:relative; display:flex; justify-content:space-between;
                  font-family:'DM Mono',monospace; font-size:0.52rem; color:#2A3D5A;
                  letter-spacing:0.5px; padding:0 2px;">
        <span>0%</span>
        <span style="position:absolute; left:25%; transform:translateX(-50%);">25%</span>
        <span style="position:absolute; left:50%; transform:translateX(-50%);">50%</span>
        <span style="position:absolute; left:75%; transform:translateX(-50%);">75%</span>
        <span>100%</span>
      </div>

    </div>
    """)


def value_card(name: str, odds: float, imp: float, prob: float,
               edge: float, ev: float, is_value: bool):
    if is_value:
        accent = "#3DFFA0"
        bg     = "rgba(61,255,160,0.04)"
        border = "rgba(61,255,160,0.22)"
        glow   = "rgba(61,255,160,0.40)"
        badge_bg = "rgba(61,255,160,0.14)"
        badge_c  = "#3DFFA0"
        badge_border = "rgba(61,255,160,0.30)"
        badge_text = "● VALUE"
        stat_border = "rgba(61,255,160,0.12)"
    else:
        accent = "#FF4D6D"
        bg     = "rgba(255,77,109,0.03)"
        border = "rgba(255,77,109,0.12)"
        glow   = "transparent"
        badge_bg = "rgba(255,77,109,0.10)"
        badge_c  = "rgba(255,77,109,0.75)"
        badge_border = "rgba(255,77,109,0.18)"
        badge_text = "○ NO VALUE"
        stat_border = "rgba(80,110,180,0.10)"

    anim_class = "value-glow fade-up" if is_value else "fade-up-2"
    st.html(f"""
    <div class="{anim_class}" style="position:relative; background:{bg}; border:1px solid {border};
                border-radius:16px; padding:20px; overflow:hidden; height:100%;">
      <div style="position:absolute; top:0; left:0; right:0; height:1px;
                  background:linear-gradient(90deg,transparent,{glow},transparent);"></div>

      <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:16px;">
        <div style="font-family:'Syne',sans-serif; font-size:1rem; font-weight:700;
                    color:#D8E8FF; max-width:58%; overflow:hidden; text-overflow:ellipsis;
                    white-space:nowrap;">{name}</div>
        <div style="font-family:'DM Mono',monospace; font-size:0.58rem; font-weight:500;
                    letter-spacing:1.5px; padding:4px 10px; border-radius:20px;
                    background:{badge_bg}; color:{badge_c}; border:1px solid {badge_border};
                    white-space:nowrap; flex-shrink:0;">{badge_text}</div>
      </div>

      <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
        <div style="background:rgba(10,13,26,0.70); border:1px solid rgba(80,110,180,0.10);
                    border-radius:10px; padding:12px;">
          <div style="font-family:'DM Mono',monospace; font-size:0.55rem; letter-spacing:1.5px;
                      color:#4E6A90; margin-bottom:5px; text-transform:uppercase;">Cote</div>
          <div style="font-family:'DM Mono',monospace; font-size:1.2rem; font-weight:400;
                      color:#D8E8FF;">{odds}</div>
        </div>
        <div style="background:rgba(10,13,26,0.70); border:1px solid rgba(80,110,180,0.10);
                    border-radius:10px; padding:12px;">
          <div style="font-family:'DM Mono',monospace; font-size:0.55rem; letter-spacing:1.5px;
                      color:#4E6A90; margin-bottom:5px; text-transform:uppercase;">Prob Implicite</div>
          <div style="font-family:'DM Mono',monospace; font-size:1.2rem; font-weight:400;
                      color:#D8E8FF;">{imp:.1%}</div>
        </div>
        <div style="background:rgba(10,13,26,0.70); border:1px solid {stat_border};
                    border-radius:10px; padding:12px;">
          <div style="font-family:'DM Mono',monospace; font-size:0.55rem; letter-spacing:1.5px;
                      color:#4E6A90; margin-bottom:5px; text-transform:uppercase;">Prob Modèle</div>
          <div style="font-family:'DM Mono',monospace; font-size:1.2rem; font-weight:400;
                      color:{accent};">{prob:.1%}</div>
        </div>
        <div style="background:rgba(10,13,26,0.70); border:1px solid {stat_border};
                    border-radius:10px; padding:12px;">
          <div style="font-family:'DM Mono',monospace; font-size:0.55rem; letter-spacing:1.5px;
                      color:#4E6A90; margin-bottom:5px; text-transform:uppercase;">EV attendu</div>
          <div style="font-family:'DM Mono',monospace; font-size:1.2rem; font-weight:400;
                      color:{accent};">{ev:+.1%}</div>
        </div>
      </div>

      <!-- Edge meter -->
      <div style="margin-top:14px; padding-top:14px;
                  border-top:1px solid rgba(80,110,180,0.08);">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
          <div style="font-family:'DM Mono',monospace; font-size:0.55rem; letter-spacing:1.5px;
                      color:#4E6A90; text-transform:uppercase;">Edge vs bookmaker</div>
          <div style="font-family:'DM Mono',monospace; font-size:0.78rem; font-weight:500;
                      color:{accent};">{edge:+.2%}</div>
        </div>
        <div style="height:4px; border-radius:3px; background:rgba(80,110,180,0.10); overflow:hidden;">
          <div style="height:100%; width:{min(abs(edge)*500,100):.1f}%;
                      background:linear-gradient(90deg,{accent},{accent}88);
                      border-radius:3px; opacity:{0.9 if is_value else 0.45};
                      animation: grow-bar 0.9s cubic-bezier(0.34,1.1,0.64,1) 0.2s both;"></div>
        </div>
      </div>
    </div>
    """)


def surface_pill(surface: str) -> str:
    palette = {
        "Hard":  ("#4D9EFF", "rgba(77,158,255,0.10)", "rgba(77,158,255,0.22)"),
        "Clay":  ("#E07040", "rgba(224,112,64,0.10)",  "rgba(224,112,64,0.22)"),
        "Grass": ("#3DFFA0", "rgba(61,255,160,0.10)",  "rgba(61,255,160,0.22)"),
    }
    c, bg, border = palette.get(surface, ("#888", "rgba(136,136,136,0.10)", "rgba(136,136,136,0.22)"))
    return (f'<span style="background:{bg}; color:{c}; border:1px solid {border}; '
            f'border-radius:20px; padding:3px 10px; font-size:0.62rem; '
            f'font-family:\'DM Mono\',monospace; letter-spacing:0.8px; font-weight:500;">{surface}</span>')


def tour_badge(tour: str) -> str:
    t = tour.upper()
    if t == "ATP":
        return ('<span style="background:rgba(77,158,255,0.10); color:#4D9EFF; '
                'border:1px solid rgba(77,158,255,0.22); border-radius:4px; '
                'padding:2px 7px; font-size:0.58rem; font-family:\'DM Mono\',monospace; '
                'letter-spacing:1px; font-weight:500;">ATP</span>')
    return ('<span style="background:rgba(255,179,71,0.10); color:#FFB347; '
            'border:1px solid rgba(255,179,71,0.22); border-radius:4px; '
            'padding:2px 7px; font-size:0.58rem; font-family:\'DM Mono\',monospace; '
            'letter-spacing:1px; font-weight:500;">WTA</span>')


def section_header(text: str, sub: str = ""):
    sub_html = f'<div style="font-family:\'DM Mono\',monospace; font-size:0.62rem; letter-spacing:1px; color:#2A3D5A; margin-top:2px;">{sub}</div>' if sub else ""
    st.html(f"""
    <div style="display:flex; align-items:baseline; gap:10px; margin:24px 0 14px;">
      <div style="font-family:'Syne',sans-serif; font-size:0.92rem; font-weight:700;
                  letter-spacing:0.3px; color:#D8E8FF;">{text}</div>
      <div style="flex:1; height:1px; background:linear-gradient(90deg,rgba(80,110,180,0.2),transparent);
                  margin-bottom:2px;"></div>
      {sub_html}
    </div>
    """)


def kpi_row(items: list):
    """items = list of (label, value, delta, value_color)"""
    cols_html = ""
    for label, value, delta, color in items:
        delta_html = ""
        if delta:
            d_color = "#3DFFA0" if not str(delta).startswith("-") else "#FF4D6D"
            delta_html = f'<div style="font-family:\'DM Mono\',monospace;font-size:0.7rem;color:{d_color};margin-top:5px;">{delta}</div>'
        cols_html += f"""
        <div class="kpi-card fade-up" style="flex:1;">
          <div style="font-family:'DM Mono',monospace;font-size:0.58rem;letter-spacing:2px;
                      text-transform:uppercase;color:#4E6A90;margin-bottom:8px;">{label}</div>
          <div style="font-family:'DM Mono',monospace;font-size:1.55rem;font-weight:400;
                      color:{color};line-height:1.1;">{value}</div>
          {delta_html}
        </div>"""
    st.html(f"""
    <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:8px;">
      {cols_html}
    </div>
    """)


def win_rate_donut(win_rate: float, n_won: int, n_total: int) -> str:
    color = "#3DFFA0" if win_rate >= 0.5 else "#FFB347"
    pct = win_rate * 100
    return f"""
    <div style="display:flex;flex-direction:column;align-items:center;padding:20px 0 8px;">
      <div style="position:relative;width:136px;height:136px;">
        <div style="width:136px;height:136px;border-radius:50%;
                    background:conic-gradient({color} {pct:.1f}%,rgba(30,45,74,0.7) {pct:.1f}%);
                    display:flex;align-items:center;justify-content:center;">
          <div style="width:108px;height:108px;border-radius:50%;background:#0C1020;
                      display:flex;flex-direction:column;align-items:center;justify-content:center;gap:2px;">
            <div style="font-family:'DM Mono',monospace;font-size:1.65rem;font-weight:400;
                        color:{color};line-height:1;">{pct:.0f}<span style="font-size:0.9rem;color:#4E6A90;">%</span></div>
            <div style="font-family:'DM Mono',monospace;font-size:0.5rem;letter-spacing:2px;
                        text-transform:uppercase;color:#2A3D5A;">Win rate</div>
          </div>
        </div>
      </div>
      <div style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#4E6A90;margin-top:8px;">
        {n_won} / {n_total} paris gagnés
      </div>
    </div>
    """


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.html("""
<div style="display:flex; align-items:center; gap:14px; margin-bottom:6px; padding-bottom:4px;">
  <div style="width:42px; height:42px; border-radius:12px; flex-shrink:0;
              background:linear-gradient(135deg,rgba(61,255,160,0.20),rgba(77,158,255,0.15));
              border:1px solid rgba(61,255,160,0.25);
              display:flex; align-items:center; justify-content:center; font-size:1.3rem;">🎾</div>
  <div>
    <div style="font-family:'Syne',sans-serif; font-size:1.5rem; font-weight:800;
                letter-spacing:-0.5px; color:#D8E8FF; line-height:1;">Tennis Predictor</div>
    <div style="font-family:'DM Mono',monospace; font-size:0.6rem; color:#2A3D5A;
                letter-spacing:2px; text-transform:uppercase; margin-top:3px;">
      XGBoost · ELO · Platt Calibration · ATP &amp; WTA
    </div>
  </div>
</div>
""")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    cfg = load_config()

    ratio     = cfg["current_bankroll"] / max(cfg["initial_bankroll"], 1)
    gauge_deg = min(ratio, 2.0) * 180   # 0-360, 1.0 ratio = 180 deg = half circle
    pnl_total = cfg["current_bankroll"] - cfg["initial_bankroll"]
    pnl_color = "#3DFFA0" if pnl_total >= 0 else "#FF4D6D"
    pnl_sign  = "+" if pnl_total >= 0 else ""
    gauge_color = "#3DFFA0" if ratio >= 1 else "#FF4D6D"
    gauge_track = "rgba(30,45,74,0.6)"

    gauge_pct = min(ratio / 2.0, 1.0) * 100  # 0→100% maps 0→2x bankroll

    st.html(f"""
    <div style="display:flex;flex-direction:column;align-items:center;padding:20px 0 12px;">
      <div style="width:136px;height:136px;border-radius:50%;margin-bottom:14px;
                  background:conic-gradient({gauge_color} {gauge_pct:.1f}%,rgba(30,45,74,0.55) {gauge_pct:.1f}%);
                  display:flex;align-items:center;justify-content:center;">
        <div style="width:108px;height:108px;border-radius:50%;background:#0C1020;
                    display:flex;flex-direction:column;align-items:center;justify-content:center;gap:2px;">
          <div style="font-family:'DM Mono',monospace;font-size:1.3rem;font-weight:400;
                      color:#D8E8FF;line-height:1;">{cfg['current_bankroll']:.0f}<span style="font-size:0.7rem;color:#4E6A90;margin-left:1px;">€</span></div>
          <div style="font-family:'DM Mono',monospace;font-size:0.52rem;letter-spacing:2px;
                      text-transform:uppercase;color:#2A3D5A;">Bankroll</div>
        </div>
      </div>
      <div style="font-family:'DM Mono',monospace;font-size:0.88rem;color:{pnl_color};
                  display:flex;align-items:center;gap:4px;">
        <span style="font-size:0.6rem;">{'▲' if pnl_total >= 0 else '▼'}</span>
        {pnl_sign}{abs(pnl_total):.2f} €
      </div>
      <div style="font-family:'DM Mono',monospace;font-size:0.58rem;color:#2A3D5A;
                  letter-spacing:0.5px;margin-top:3px;">depuis le départ</div>
    </div>
    <div style="height:1px;background:linear-gradient(90deg,transparent,rgba(80,110,180,0.18),transparent);
                margin:8px 0 12px;"></div>
    """)

    # Last-5 results strip
    _df_sb = load_bets()
    _resolved_sb = _df_sb[_df_sb["result"].isin(["won","lost"])].tail(5)
    if not _resolved_sb.empty:
        dots = ""
        for _, _r in _resolved_sb.iterrows():
            c = "#3DFFA0" if _r["result"] == "won" else "#FF4D6D"
            dots += f'<div style="width:10px;height:10px;border-radius:50%;background:{c};flex-shrink:0;"></div>'
        # Pad remaining
        for _ in range(5 - len(_resolved_sb)):
            dots += '<div style="width:10px;height:10px;border-radius:50%;background:rgba(80,110,180,0.15);flex-shrink:0;"></div>'
        st.html(f"""
        <div style="display:flex;flex-direction:column;align-items:center;gap:8px;margin-bottom:14px;">
          <div style="font-family:'DM Mono',monospace;font-size:0.52rem;letter-spacing:2px;
                      text-transform:uppercase;color:#2A3D5A;">Derniers résultats</div>
          <div style="display:flex;gap:6px;align-items:center;">
            {dots}
          </div>
        </div>
        <div style="height:1px;background:linear-gradient(90deg,transparent,rgba(80,110,180,0.15),transparent);
                    margin-bottom:16px;"></div>
        """)

    with st.expander("⚙️ Réglages"):
        init_bk = st.number_input("Bankroll initiale (€)", value=cfg["initial_bankroll"],
                                   min_value=1.0, step=50.0)
        method  = st.selectbox("Méthode de mise", ["flat", "percent", "kelly"],
                                index=["flat","percent","kelly"].index(cfg["stake_method"]))
        label   = {"flat":"Mise fixe (€)","percent":"% bankroll","kelly":"Fraction Kelly"}[method]
        value   = st.number_input(label, value=cfg["stake_value"], min_value=0.01, step=1.0)
        if st.button("Sauvegarder", use_container_width=True):
            cfg.update({"initial_bankroll": init_bk, "current_bankroll": init_bk,
                        "stake_method": method, "stake_value": value})
            save_config(cfg)
            st.success("Sauvegardé !")
            st.rerun()

    st.html("""
    <div style="font-family:'DM Mono',monospace; font-size:0.58rem; letter-spacing:2px;
                text-transform:uppercase; color:#2A3D5A; margin:18px 0 8px; padding-left:2px;">
      Mise à jour données (ESPN + tennis-data)
    </div>
    """)

    def run_live_sync(tour: str):
        import subprocess, sys
        with st.spinner(f"Sync live {tour.upper()}..."):
            result = subprocess.run(
                [sys.executable, str(ROOT / "src" / "fetch_live_data.py"),
                 "--tour", tour, "--force"],
                capture_output=True, text=True, encoding="utf-8", cwd=str(ROOT),
            )
        if result.returncode == 0:
            st.cache_resource.clear()
            st.cache_data.clear()
            st.success(f"{tour.upper()} synced !")
            st.rerun()
        else:
            st.error(f"Erreur sync {tour.upper()}")
            st.code(result.stderr[-2000:] if result.stderr else result.stdout[-2000:])

    col_ls_atp, col_ls_wta = st.columns(2)
    if col_ls_atp.button("⚡ ATP", key="live_atp", use_container_width=True, type="primary"):
        run_live_sync("atp")
    if col_ls_wta.button("⚡ WTA", key="live_wta", use_container_width=True, type="primary"):
        run_live_sync("wta")

    for tour_upd in ["atp", "wta"]:
        profiles_path = ROOT / "data" / "processed" / tour_upd / "player_profiles_updated.parquet"
        if profiles_path.exists():
            mtime = datetime.fromtimestamp(profiles_path.stat().st_mtime)
            st.caption(f"{tour_upd.upper()} — {mtime.strftime('%d/%m %H:%M')}")

# ─────────────────────────────────────────────────────────────────────────────
# TABS PRINCIPAUX
# ─────────────────────────────────────────────────────────────────────────────
tab_today, tab_pred_atp, tab_pred_wta, tab_history, tab_stats = st.tabs([
    "🎾  Matchs du jour", "📊  ATP", "📊  WTA", "📋  Historique", "📈  Stats",
])


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — Onglet Prédictions
# ─────────────────────────────────────────────────────────────────────────────

def render_predictions_tab(tour: str, tour_label: str, cfg: dict):
    with st.spinner("Chargement modèle..."):
        try:
            model, imputer, features, platt = load_model(tour)
            df_players, df_elo              = load_player_data(tour)
        except Exception as e:
            st.error(f"Erreur chargement modèle {tour_label} : {e}")
            return

    thresholds = load_optimal_thresholds(tour)
    opt_edge   = thresholds.get("best_roi", {}).get("min_edge", 0.03)
    opt_prob   = thresholds.get("best_roi", {}).get("min_prob", 0.55)

    player_names    = load_player_names(tour)
    tourney_names   = load_tournament_names(tour)
    best_of_options = [3, 5] if tour == "atp" else [3]
    pred_key        = f"pred_{tour}"

    with st.form(f"match_form_{tour}"):
        col1, col2 = st.columns(2)
        with col1:
            p1_name    = st.selectbox("Joueur 1", [""] + player_names,
                                      index=0, placeholder="Tapez pour filtrer…")
            tournament = st.selectbox("Tournoi", [""] + tourney_names,
                                      index=0, placeholder="Tapez pour filtrer…")
        with col2:
            p2_name = st.selectbox("Joueur 2", [""] + player_names,
                                   index=0, placeholder="Tapez pour filtrer…")
            surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"])

        col3, col4, col5, col6 = st.columns(4)
        with col3:
            odd1 = st.number_input("Cote J1", min_value=1.01, value=1.50, step=0.05)
        with col4:
            odd2 = st.number_input("Cote J2", min_value=1.01, value=2.50, step=0.05)
        with col5:
            best_of = st.radio("Best of", best_of_options, horizontal=True)
        with col6:
            round_name = st.selectbox("Tour", ["R128","R64","R32","R16","QF","SF","F"], index=4)

        st.html("<div style='padding:0 0 8px;'></div>")
        submitted = st.form_submit_button("⚡  Analyser le match", type="primary",
                                          use_container_width=True)

    if submitted:
        if not p1_name or not p2_name:
            st.warning("Sélectionne les deux joueurs.")
        else:
            try:
                with st.spinner("Analyse en cours..."):
                    result = predict_match(p1_name, p2_name, tournament, surface,
                                           best_of, round_name, model, imputer,
                                           features, df_players, df_elo, platt)
                st.session_state[pred_key] = {**result, "tournament": tournament,
                    "surface": surface, "best_of": best_of, "odd1": odd1, "odd2": odd2,
                    "round_name": round_name}
            except Exception as e:
                import traceback
                st.error(f"Erreur : {e}")
                st.code(traceback.format_exc())

    if pred_key in st.session_state:
        r          = st.session_state[pred_key]
        prob_p1    = r["prob_p1"]
        prob_p2    = r["prob_p2"]
        odd1       = r["odd1"]
        odd2       = r["odd2"]
        round_name = r.get("round_name", "")

        for key, msg in [("p1_found", r["p1_name"]), ("p2_found", r["p2_name"])]:
            if not r[key]:
                st.warning(f"Joueur non trouvé dans la base : {msg}")

        favori    = r["p1_name"] if prob_p1 >= prob_p2 else r["p2_name"]
        surf_html = surface_pill(r["surface"])

        st.html(f"""
        <div class="fade-up" style="position:relative; overflow:hidden;
                    background:linear-gradient(135deg,rgba(15,22,40,0.95),rgba(10,14,28,0.98));
                    border:1px solid rgba(80,110,180,0.18); border-radius:18px;
                    padding:24px 26px; margin-top:8px;">

          <!-- Top glow line -->
          <div style="position:absolute; top:0; left:0; right:0; height:1px;
                      background:linear-gradient(90deg,transparent,rgba(61,255,160,0.45),rgba(77,158,255,0.3),transparent);"></div>

          <!-- Corner accent -->
          <div style="position:absolute; top:0; right:0; width:120px; height:120px;
                      background:radial-gradient(ellipse at top right,rgba(61,255,160,0.05),transparent 70%);
                      pointer-events:none;"></div>

          <!-- Meta tags -->
          <div style="display:flex; align-items:center; gap:8px; margin-bottom:14px; flex-wrap:wrap;">
            {tour_badge(tour)}
            {surf_html}
            <span style="font-family:'DM Mono',monospace; font-size:0.6rem; letter-spacing:1px;
                         color:#2A3D5A; background:rgba(80,110,180,0.07);
                         border:1px solid rgba(80,110,180,0.12); border-radius:4px;
                         padding:2px 7px;">{round_name}</span>
            <span style="font-family:'DM Mono',monospace; font-size:0.62rem; letter-spacing:0.5px;
                         color:#2A3D5A;">{r['tournament']}</span>
          </div>

          <!-- Match title -->
          <div style="font-family:'Syne',sans-serif; font-size:1.25rem; font-weight:800;
                      color:#D8E8FF; margin-bottom:10px; letter-spacing:-0.3px;">
            {r['p1_name']}
            <span style="color:#2A3D5A; font-weight:400; font-size:0.95rem; margin:0 10px;">vs</span>
            {r['p2_name']}
          </div>

          <!-- Favori -->
          <div style="display:inline-flex; align-items:center; gap:6px;
                      background:rgba(61,255,160,0.06); border:1px solid rgba(61,255,160,0.15);
                      border-radius:20px; padding:4px 12px;">
            <span style="font-family:'DM Mono',monospace; font-size:0.58rem; letter-spacing:1px;
                         text-transform:uppercase; color:#4E6A90;">Favori modèle</span>
            <span style="font-family:'Syne',sans-serif; font-size:0.72rem; font-weight:700;
                         color:#3DFFA0;">{favori}</span>
          </div>

        </div>
        """)

        prob_bar(prob_p1, prob_p2, r["p1_name"], r["p2_name"])

        edge1, ev1, imp1 = compute_edge(prob_p1, odd1, odd2)
        edge2, ev2, imp2 = compute_edge(prob_p2, odd2, odd1)

        clv1 = prob_p1 / imp1 - 1 if imp1 > 0 else 0.0
        clv2 = prob_p2 / imp2 - 1 if imp2 > 0 else 0.0

        section_header("Value Bets")
        cols = st.columns(2)

        current_cfg = load_config()

        for i, (name, prob, odds, edge, ev, imp, clv) in enumerate([
            (r["p1_name"], prob_p1, odd1, edge1, ev1, imp1, clv1),
            (r["p2_name"], prob_p2, odd2, edge2, ev2, imp2, clv2),
        ]):
            with cols[i]:
                is_value = edge >= opt_edge and prob >= opt_prob
                value_card(name, odds, imp, prob, edge, ev, is_value)

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
                      SIGNAL
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

                if is_value:
                    st.html("<div style='height:10px;'></div>")
                    bk = current_cfg["current_bankroll"]
                    if current_cfg["stake_method"] == "flat":
                        stake = current_cfg["stake_value"]
                    elif current_cfg["stake_method"] == "percent":
                        stake = bk * current_cfg["stake_value"] / 100
                    else:
                        stake = min(
                            kelly_stake(prob, odds, bk, current_cfg["stake_value"]),
                            bk * 0.05,
                        )
                    stake = round(stake, 2)

                    method_label = {"flat": "Mise fixe", "percent": "% Bankroll",
                                   "kelly": "Kelly fraction"}.get(current_cfg["stake_method"], "")
                    st.html(f"""
                    <div class="slide-r" style="position:relative; overflow:hidden;
                                background:rgba(61,255,160,0.05);
                                border:1px solid rgba(61,255,160,0.22);
                                border-radius:12px; padding:16px 20px; margin-bottom:10px;">
                      <div style="position:absolute; top:0; left:0; right:0; height:1px;
                                  background:linear-gradient(90deg,transparent,rgba(61,255,160,0.4),transparent);"></div>
                      <div style="display:flex; align-items:center; justify-content:space-between;">
                        <div>
                          <div style="font-family:'DM Mono',monospace; font-size:0.52rem; letter-spacing:2px;
                                      color:#4E6A90; text-transform:uppercase; margin-bottom:6px;">
                            Mise suggérée · {method_label}
                          </div>
                          <div style="font-family:'DM Mono',monospace; font-size:1.75rem; font-weight:400;
                                      color:#3DFFA0; line-height:1; letter-spacing:-0.5px;">
                            {stake}<span style="font-size:0.9rem; color:#4E6A90; margin-left:3px;">€</span>
                          </div>
                        </div>
                        <div style="width:40px; height:40px; border-radius:50%;
                                    background:rgba(61,255,160,0.10); border:1px solid rgba(61,255,160,0.20);
                                    display:flex; align-items:center; justify-content:center;
                                    font-size:1.1rem; color:#3DFFA0;">→</div>
                      </div>
                    </div>
                    """)

                    if st.button(f"Enregistrer", key=f"bet_{tour}_{i}",
                                 use_container_width=True, type="primary"):
                        add_bet({
                            "date": str(date.today()), "tour": tour,
                            "tournament": r["tournament"], "surface": r["surface"],
                            "p1_name": r["p1_name"], "p2_name": r["p2_name"],
                            "bet_on": name, "odds": odds, "stake": stake,
                            "prob_model": round(prob, 4), "edge": round(edge, 4),
                            "result": "pending", "pnl": np.nan, "bankroll_after": np.nan,
                        })
                        st.success("Pari enregistré !")
                        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 0 — MATCHS DU JOUR
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)  # 5 min — évite un appel ESPN à chaque interaction
def _fetch_today_matches(tour: str) -> list:
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from espn_client import fetch_scheduled
    from datetime import date as _date
    raw = fetch_scheduled(tour, _date.today())
    return [m for m in raw if "TBD" not in m["p1_name"] and "TBD" not in m["p2_name"]]


def _pill(text: str, color: str) -> str:
    return (f'<span style="background:{color};color:#fff;padding:2px 8px;'
            f'border-radius:9px;font-size:0.72rem;font-weight:600">{text}</span>')


SURF_COLORS = {"Hard": "#2563eb", "Clay": "#c2410c", "Grass": "#15803d", "Carpet": "#7e22ce"}


def render_today_tab():
    tour_sel = st.radio("Circuit", ["ATP", "WTA"], horizontal=True, key="today_tour")
    tour     = tour_sel.lower()

    col_refresh, col_date = st.columns([1, 3])
    with col_date:
        st.caption(f"Prédictions automatiques — ESPN live")

    with st.spinner("Chargement matchs ESPN..."):
        matches = _fetch_today_matches(tour)

    if not matches:
        st.info("Aucun match programme aujourd'hui pour ce circuit.")
        return

    odds_result = fetch_odds_today(tour)
    matches     = merge_odds(matches, odds_result.odds)
    thresholds  = load_optimal_thresholds(tour)
    opt_edge    = thresholds.get("best_roi", {}).get("min_edge", 0.03)
    opt_prob    = thresholds.get("best_roi", {}).get("min_prob", 0.55)

    try:
        model, imputer, features, platt = load_model(tour)
        df_players, df_elo = load_player_data(tour)
    except Exception as e:
        st.error(f"Modele {tour.upper()} non disponible : {e}")
        return

    cfg_today = load_config()
    bankroll  = cfg_today.get("current_bankroll", 1000.0)

    st.caption(f"{len(matches)} match(s) programme(s) — bankroll : **{bankroll:.0f} €**")

    # ── Status bar Odds API ───────────────────────────────────────────────────
    n_enriched = sum(1 for m in matches if m.get("odd_p1") is not None)
    if odds_result.fetched_at:
        fetched_time = datetime.fromisoformat(odds_result.fetched_at).strftime("%H:%M")
        _sb1, _sb2 = st.columns([5, 1])
        with _sb1:
            st.html(
                f'<span style="font-family:\'DM Mono\',monospace;font-size:0.72rem;color:#3DFFA0;">'
                f'● Cotes Odds API</span>'
                f'<span style="font-size:0.72rem;color:#64748b;"> — {n_enriched} match'
                f'{"s" if n_enriched != 1 else ""} enrichi'
                f'{"s" if n_enriched != 1 else ""} · mis à jour {fetched_time}</span>'
            )
        with _sb2:
            if st.button("↻ Rafraîchir", key=f"refresh_odds_{tour}",
                         use_container_width=True):
                _cache_file = (ROOT / "data" / "odds_cache" / tour
                               / f"odds_{date.today().isoformat()}.json")
                if _cache_file.exists():
                    _cache_file.unlink()
                st.rerun()

    for i, m in enumerate(matches):
        try:
            result = predict_match(
                m["p1_name"], m["p2_name"], m["tournament"],
                m["surface"], m["best_of"], m["round"],
                model, imputer, features, df_players, df_elo, platt
            )
        except Exception:
            continue

        prob_p1 = result["prob_p1"]
        prob_p2 = result["prob_p2"]
        fav     = m["p1_name"] if prob_p1 >= 0.5 else m["p2_name"]
        fav_prob = max(prob_p1, prob_p2)
        surf_col = SURF_COLORS.get(m["surface"], "#374151")

        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 2, 1])
            with c1:
                time_str = f'<span style="color:#94a3b8;font-size:0.75rem">{m["time"]}</span> ' if m.get("time") else ""
                st.html(
                    f'{time_str}<b>{m["p1_name"]}</b> vs <b>{m["p2_name"]}</b><br>'
                    f'<span style="color:#94a3b8;font-size:0.75rem">{m["tournament"]} · {m["round"]}</span>&nbsp;'
                    f'{_pill(m["surface"], surf_col)}'
                )
            with c2:
                bar_w = int(prob_p1 * 100)
                st.html(
                    f'<div style="display:flex;gap:6px;align-items:center;margin-top:4px">'
                    f'<span style="font-size:0.8rem;min-width:36px">{prob_p1:.0%}</span>'
                    f'<div style="flex:1;background:#374151;border-radius:4px;height:8px">'
                    f'<div style="width:{bar_w}%;background:#3b82f6;height:8px;border-radius:4px"></div>'
                    f'</div>'
                    f'<span style="font-size:0.8rem;min-width:36px;text-align:right">{prob_p2:.0%}</span>'
                    f'</div>'
                    f'<div style="font-size:0.73rem;color:#94a3b8;margin-top:2px">'
                    f'Favori : <b style="color:#f1f5f9">{fav}</b> ({fav_prob:.0%})'
                    f'</div>'
                )

            # ── GO/PASS badge (auto-computed from Odds API) ───────────────────
            _op1 = m.get("odd_p1")
            _op2 = m.get("odd_p2")
            if _op1 is not None and _op2 is not None:
                _e1, _, _i1 = compute_edge(prob_p1, _op1, _op2)
                _e2, _, _i2 = compute_edge(prob_p2, _op2, _op1)
                _go_p1 = _e1 >= opt_edge and prob_p1 >= opt_prob
                _go_p2 = _e2 >= opt_edge and prob_p2 >= opt_prob
                if _go_p1 and (not _go_p2 or _e1 >= _e2):
                    _badge = (f"GO · {m['p1_name'].split()[0]} @ {_op1:.2f}"
                              f" · edge {_e1:+.1%}")
                    _bcol, _bbg, _bbd = "#3DFFA0", "rgba(61,255,160,0.07)", "rgba(61,255,160,0.25)"
                elif _go_p2:
                    _badge = (f"GO · {m['p2_name'].split()[0]} @ {_op2:.2f}"
                              f" · edge {_e2:+.1%}")
                    _bcol, _bbg, _bbd = "#3DFFA0", "rgba(61,255,160,0.07)", "rgba(61,255,160,0.25)"
                else:
                    _badge = "PASS"
                    _bcol, _bbg, _bbd = "#64748b", "rgba(30,40,60,0.4)", "rgba(80,100,140,0.15)"
                st.html(
                    f'<div style="margin-top:4px;padding:5px 14px;'
                    f'background:{_bbg};border:1px solid {_bbd};'
                    f'border-radius:6px;font-family:\'DM Mono\',monospace;'
                    f'font-size:0.7rem;color:{_bcol};font-weight:700;'
                    f'letter-spacing:0.5px">{_badge}</div>'
                )

            with c3:
                with st.popover("+ Pari", use_container_width=True):
                    st.markdown(
                        f"**{m['p1_name']}** vs **{m['p2_name']}**  \n"
                        f"<span style='color:#94a3b8;font-size:0.78rem'>"
                        f"{m['tournament']} · {m['round']}</span>",
                        unsafe_allow_html=True,
                    )
                    st.divider()

                    # Cotes des deux joueurs pour calcul de marge
                    ca, cb = st.columns(2)
                    with ca:
                        o1 = st.number_input(
                            f"Cote {m['p1_name'].split()[0]}",
                            min_value=1.01,
                            value=float(m.get("odd_p1") or 2.0),
                            step=0.05,
                            key=f"today_o1_{tour}_{i}",
                        )
                    with cb:
                        o2 = st.number_input(
                            f"Cote {m['p2_name'].split()[0]}",
                            min_value=1.01,
                            value=float(m.get("odd_p2") or 2.0),
                            step=0.05,
                            key=f"today_o2_{tour}_{i}",
                        )

                    # Analyse edge / EV en temps réel
                    edge1, ev1, imp1 = compute_edge(prob_p1, o1, o2)
                    edge2, ev2, imp2 = compute_edge(prob_p2, o2, o1)
                    margin = round((1/o1 + 1/o2 - 1) * 100, 1)

                    def _ev_color(ev):
                        return "#22c55e" if ev > 0 else "#ef4444"

                    clv1_today = prob_p1 / imp1 - 1 if imp1 > 0 else 0.0
                    clv2_today = prob_p2 / imp2 - 1 if imp2 > 0 else 0.0
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

                    st.divider()
                    bet_player = st.radio(
                        "Pari sur", [m["p1_name"], m["p2_name"]],
                        key=f"today_bplayer_{tour}_{i}",
                    )

                    # Sélection des valeurs selon le joueur choisi
                    is_p1    = (bet_player == m["p1_name"])
                    bet_odds = o1 if is_p1 else o2
                    bet_prob = prob_p1 if is_p1 else prob_p2
                    bet_edge = edge1 if is_p1 else edge2
                    bet_ev   = ev1 if is_p1 else ev2

                    # Mise Kelly 1/4 avec cap à 5% du bankroll
                    KELLY_FRACTION  = 0.25
                    MAX_KELLY_PCT   = 0.05  # jamais plus de 5% du bankroll par pari
                    kelly_raw       = kelly_stake(bet_prob, bet_odds, bankroll, KELLY_FRACTION)
                    kelly_capped    = min(kelly_raw, bankroll * MAX_KELLY_PCT)
                    kelly_default   = max(0.0, round(kelly_capped, 1))

                    if kelly_default == 0:
                        st.warning("EV négatif — Kelly recommande de ne pas parier.")
                    elif kelly_raw > kelly_capped:
                        st.info(f"Kelly brut ({kelly_raw:.1f}€) plafonné à 5% bankroll ({kelly_capped:.1f}€)")

                    bet_stake = st.number_input(
                        f"Mise (€)  — Kelly ¼ cap 5% : {kelly_default:.1f} €",
                        min_value=0.0, value=kelly_default,
                        step=1.0, key=f"today_stake_{tour}_{i}",
                    )

                    if st.button("Enregistrer", key=f"today_add_{tour}_{i}", type="primary"):
                        if bet_stake <= 0:
                            st.error("Mise invalide (EV négatif — Kelly = 0).")
                        else:
                            new_bet = {
                                "date":           str(date.today()),
                                "tour":           tour.upper(),
                                "tournament":     m["tournament"],
                                "surface":        m["surface"],
                                "p1_name":        m["p1_name"],
                                "p2_name":        m["p2_name"],
                                "bet_on":         bet_player,
                                "odds":           bet_odds,
                                "stake":          bet_stake,
                                "prob_model":     round(bet_prob, 4),
                                "edge":           round(bet_edge, 4),
                                "result":         "pending",
                                "pnl":            None,
                                "bankroll_after": None,
                            }
                            add_bet(new_bet)
                            st.success("Pari enregistre !")
                        st.rerun()


with tab_today:
    render_today_tab()

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — ATP
# ═════════════════════════════════════════════════════════════════════════════
with tab_pred_atp:
    render_predictions_tab("atp", "ATP", cfg)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — WTA
# ═════════════════════════════════════════════════════════════════════════════
with tab_pred_wta:
    render_predictions_tab("wta", "WTA", cfg)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — HISTORIQUE
# ═════════════════════════════════════════════════════════════════════════════
with tab_history:
    df_bets = load_bets()

    if df_bets.empty:
        st.html("""
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center;
                    padding:80px 20px; color:#2A3D5A;">
          <div style="font-size:2.5rem; margin-bottom:16px; opacity:0.4;">📋</div>
          <div style="font-family:'Syne',sans-serif; font-size:1rem; font-weight:600; color:#2A3D5A;">
            Aucun pari enregistré
          </div>
          <div style="font-family:'DM Mono',monospace; font-size:0.7rem; color:#1E2D3F; margin-top:6px;">
            Lance une prédiction et enregistre ton premier pari
          </div>
        </div>
        """)
    else:
        pending = df_bets[df_bets["result"] == "pending"]

        if not pending.empty:
            section_header(f"⏳ En attente", f"{len(pending)} pari{'s' if len(pending) > 1 else ''}")

            for idx, row in pending.iterrows():
                edit_key = f"editing_{idx}"
                surf_h   = surface_pill(str(row.get("surface", "")))
                t_badge  = tour_badge(str(row.get("tour", "")))

                with st.container(border=True):
                    c1, c2 = st.columns([5, 3])
                    with c1:
                        st.html(f"""
                        <div style="padding:2px 0;">
                          <div style="font-family:'Syne',sans-serif; font-size:0.96rem; font-weight:700;
                                      color:#D8E8FF; margin-bottom:4px; display:flex; align-items:center; gap:8px;">
                            {row['bet_on']}
                            {t_badge}
                          </div>
                          <div style="font-family:'DM Sans',sans-serif; font-size:0.82rem;
                                      color:#4E6A90; margin-bottom:8px;">
                            {row['p1_name']} <span style="color:#2A3D5A;">vs</span> {row['p2_name']}
                          </div>
                          <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
                            <span style="font-family:'DM Mono',monospace; font-size:0.68rem;
                                         color:#2A3D5A;">{row['tournament']}</span>
                            <span style="color:#2A3D5A; font-size:0.6rem;">·</span>
                            {surf_h}
                            <span style="color:#2A3D5A; font-size:0.6rem;">·</span>
                            <span style="font-family:'DM Mono',monospace; font-size:0.68rem;
                                         color:#2A3D5A;">{row['date']}</span>
                          </div>
                        </div>
                        """)
                    with c2:
                        payout = round(float(row["stake"]) * float(row["odds"]), 2)
                        edge_v = float(row["edge"]) if pd.notna(row.get("edge", np.nan)) else 0
                        edge_c = "#3DFFA0" if edge_v >= 0 else "#FF4D6D"
                        st.html(f"""
                        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; margin-bottom:10px;">
                          <div style="background:rgba(10,13,26,0.8); border:1px solid rgba(80,110,180,0.10);
                                      border-radius:8px; padding:10px 12px; text-align:center;">
                            <div style="font-family:'DM Mono',monospace; font-size:0.52rem; letter-spacing:1.5px;
                                        color:#4E6A90; text-transform:uppercase; margin-bottom:4px;">Cote</div>
                            <div style="font-family:'DM Mono',monospace; font-size:1.05rem; color:#D8E8FF;">{row['odds']}</div>
                          </div>
                          <div style="background:rgba(10,13,26,0.8); border:1px solid rgba(80,110,180,0.10);
                                      border-radius:8px; padding:10px 12px; text-align:center;">
                            <div style="font-family:'DM Mono',monospace; font-size:0.52rem; letter-spacing:1.5px;
                                        color:#4E6A90; text-transform:uppercase; margin-bottom:4px;">Mise</div>
                            <div style="font-family:'DM Mono',monospace; font-size:1.05rem; color:#D8E8FF;">{row['stake']:.0f}€</div>
                          </div>
                          <div style="background:rgba(10,13,26,0.8); border:1px solid rgba(80,110,180,0.10);
                                      border-radius:8px; padding:10px 12px; text-align:center;">
                            <div style="font-family:'DM Mono',monospace; font-size:0.52rem; letter-spacing:1.5px;
                                        color:#4E6A90; text-transform:uppercase; margin-bottom:4px;">Edge</div>
                            <div style="font-family:'DM Mono',monospace; font-size:1.05rem; color:{edge_c};">{edge_v:+.1%}</div>
                          </div>
                        </div>
                        <div style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#2A3D5A;
                                    text-align:center; margin-bottom:8px;">
                          Gain potentiel · <span style="color:#3DFFA0;">{payout:.2f} €</span>
                        </div>
                        """)
                        col_w, col_l, col_e, col_d = st.columns(4)
                        if col_w.button("✅", key=f"won_{idx}", help="Gagné"):
                            update_bet_result(idx, "won"); st.rerun()
                        if col_l.button("❌", key=f"lost_{idx}", help="Perdu"):
                            update_bet_result(idx, "lost"); st.rerun()
                        if col_e.button("✏️", key=f"edit_{idx}", help="Modifier"):
                            st.session_state[edit_key] = not st.session_state.get(edit_key, False)
                            st.rerun()
                        if col_d.button("🗑️", key=f"del_{idx}", help="Supprimer"):
                            delete_bet(idx); st.rerun()

                    if st.session_state.get(edit_key, False):
                        with st.form(key=f"form_edit_{idx}"):
                            ec1, ec2, ec3 = st.columns([2, 2, 1])
                            new_stake = ec1.number_input("Mise (€)", value=float(row["stake"]),
                                                         min_value=0.01, step=1.0)
                            new_odds  = ec2.number_input("Cote", value=float(row["odds"]),
                                                         min_value=1.01, step=0.05)
                            if ec3.form_submit_button("OK"):
                                edit_bet(idx, new_stake, new_odds)
                                st.session_state.pop(edit_key, None)
                                st.rerun()

            st.divider()

        filter_tour = st.radio("Afficher", ["Tous", "ATP", "WTA"],
                               horizontal=True, label_visibility="collapsed")
        if filter_tour != "Tous":
            display_df = df_bets[df_bets["tour"].str.upper() == filter_tour].copy()
        else:
            display_df = df_bets.copy()

        section_header(f"Tous les paris", f"{len(display_df)} enregistré{'s' if len(display_df) > 1 else ''}")

        display = display_df.copy()
        display["Résultat"] = display["result"].map({
            "won": "✅ Gagné", "lost": "❌ Perdu", "pending": "⏳ En cours"
        })
        display["P&L"]    = display["pnl"].apply(lambda x: f"{x:+.2f} €" if pd.notna(x) else "—")
        display["Prob"]   = display["prob_model"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
        display["Edge"]   = display["edge"].apply(lambda x: f"{x:+.1%}" if pd.notna(x) else "—")
        display["Circuit"] = display["tour"].str.upper()

        st.dataframe(
            display[["date","Circuit","tournament","surface","p1_name","p2_name",
                      "bet_on","odds","stake","Prob","Edge","Résultat","P&L","bankroll_after"
                      ]].rename(columns={
                "date": "Date", "Circuit": "Circuit", "tournament": "Tournoi",
                "surface": "Surface", "p1_name": "J1", "p2_name": "J2",
                "bet_on": "Pari sur", "odds": "Cote", "stake": "Mise",
                "bankroll_after": "Bankroll",
            }),
            use_container_width=True, hide_index=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — STATS
# ═════════════════════════════════════════════════════════════════════════════
with tab_stats:
    df_bets  = load_bets()
    resolved = df_bets[df_bets["result"].isin(["won", "lost"])]

    if resolved.empty:
        st.html("""
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center;
                    padding:80px 20px;">
          <div style="width:64px; height:64px; border-radius:16px; margin-bottom:20px;
                      background:rgba(80,110,180,0.06); border:1px solid rgba(80,110,180,0.12);
                      display:flex; align-items:center; justify-content:center; font-size:1.8rem;
                      opacity:0.5;">📈</div>
          <div style="font-family:'Syne',sans-serif; font-size:1rem; font-weight:700; color:#2A3D5A;">
            Pas encore de résultats
          </div>
          <div style="font-family:'DM Mono',monospace; font-size:0.68rem; color:#1E2D3F; margin-top:6px;
                      letter-spacing:0.5px;">
            Résous tes premiers paris pour voir les statistiques apparaître ici
          </div>
        </div>
        """)
    else:
        filter_stats = st.radio("Circuit", ["Tous", "ATP", "WTA"],
                                horizontal=True, label_visibility="collapsed")
        if filter_stats != "Tous":
            resolved = resolved[resolved["tour"].str.upper() == filter_stats]

        n_bets       = len(resolved)
        n_won        = int((resolved["result"] == "won").sum())
        win_rate     = n_won / n_bets if n_bets else 0
        total_pnl    = resolved["pnl"].sum()
        total_staked = resolved["stake"].sum()
        roi          = total_pnl / total_staked if total_staked > 0 else 0
        avg_odds     = resolved["odds"].mean()
        avg_edge     = resolved["edge"].mean()
        pnl_color_v  = "#3DFFA0" if total_pnl >= 0 else "#FF4D6D"
        roi_color_v  = "#3DFFA0" if roi >= 0 else "#FF4D6D"

        # ── Hero row: donut + KPI blocks ──────────────────────────────────
        section_header("Performance globale")

        col_donut, col_kpis = st.columns([1, 2.5])

        with col_donut:
            st.html(win_rate_donut(win_rate, n_won, n_bets))

        with col_kpis:
            st.html("<div style='height:8px;'></div>")
            kpi_row([
                ("P&L total",    f"{total_pnl:+.2f} €",   f"{total_pnl:+.2f}", pnl_color_v),
                ("ROI",          f"{roi:+.1%}",            f"sur {total_staked:.0f} € misés", roi_color_v),
            ])
            kpi_row([
                ("Cote moyenne", f"{avg_odds:.2f}",        "", "#D8E8FF"),
                ("Edge moyen",   f"{avg_edge:+.1%}",       "vs bookmaker", "#3DFFA0" if avg_edge >= 0 else "#FF4D6D"),
            ])
            kpi_row([
                ("Paris joués",  str(n_bets),              "", "#D8E8FF"),
                ("En attente",   str(len(df_bets[df_bets["result"] == "pending"])), "", "#4E6A90"),
            ])

        st.divider()

        # ── Bankroll chart ─────────────────────────────────────────────────
        cfg_stats = load_config()
        bk_data   = df_bets[df_bets["bankroll_after"].notna()][["date","bankroll_after"]].copy()
        if not bk_data.empty:
            bk_data["date"] = pd.to_datetime(bk_data["date"])
            start = pd.DataFrame([{
                "date": bk_data["date"].min() - pd.Timedelta(days=1),
                "bankroll_after": cfg_stats["initial_bankroll"],
            }])
            bk_data = pd.concat([start, bk_data]).set_index("date")
            section_header("Évolution de la bankroll")
            st.line_chart(bk_data, y="bankroll_after", color="#3DFFA0", use_container_width=True)

        # ── Surface analysis + calibration ────────────────────────────────
        col_surf, col_model = st.columns(2)

        with col_surf:
            section_header("P&L par surface")
            pnl_surf = resolved.groupby("surface")["pnl"].agg(["sum","count"]).reset_index()
            pnl_surf.columns = ["Surface", "P&L (€)", "Nb paris"]
            pnl_surf["P&L (€)"] = pnl_surf["P&L (€)"].round(2)

            # Styled surface cards
            for _, row in pnl_surf.iterrows():
                surf_h  = surface_pill(str(row["Surface"]))
                pnl_c   = "#3DFFA0" if row["P&L (€)"] >= 0 else "#FF4D6D"
                pnl_sgn = "+" if row["P&L (€)"] >= 0 else ""
                st.html(f"""
                <div style="display:flex; align-items:center; justify-content:space-between;
                            background:#0F1628; border:1px solid rgba(80,110,180,0.12);
                            border-radius:10px; padding:12px 16px; margin-bottom:8px;">
                  <div style="display:flex; align-items:center; gap:10px;">
                    {surf_h}
                    <span style="font-family:'DM Mono',monospace; font-size:0.7rem;
                                 color:#4E6A90;">{int(row['Nb paris'])} paris</span>
                  </div>
                  <div style="font-family:'DM Mono',monospace; font-size:1rem; font-weight:400;
                              color:{pnl_c};">{pnl_sgn}{row['P&L (€)']:.2f} €</div>
                </div>
                """)

        with col_model:
            section_header("Calibration modèle")
            res = resolved.copy()
            res["prob_model"] = pd.to_numeric(res["prob_model"], errors="coerce")
            res = res.dropna(subset=["prob_model"])

            if len(res) >= 3:
                res["correct"] = (
                    ((res["prob_model"] >= 0.5) & (res["result"] == "won")) |
                    ((res["prob_model"] <  0.5) & (res["result"] == "lost"))
                )
                accuracy = res["correct"].mean()
                acc_color = "#3DFFA0" if accuracy >= 0.55 else "#FFB347"

                st.html(f"""
                <div style="background:rgba(61,255,160,0.04); border:1px solid rgba(61,255,160,0.15);
                            border-radius:12px; padding:16px 20px; margin-bottom:14px;
                            display:flex; align-items:center; justify-content:space-between;">
                  <div>
                    <div style="font-family:'DM Mono',monospace; font-size:0.58rem; letter-spacing:2px;
                                text-transform:uppercase; color:#4E6A90; margin-bottom:6px;">Précision globale</div>
                    <div style="font-family:'DM Mono',monospace; font-size:1.8rem; font-weight:400;
                                color:{acc_color}; line-height:1;">{accuracy:.1%}</div>
                  </div>
                  <div style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#2A3D5A;
                              text-align:right; line-height:1.6;">
                    % de paris où le<br>modèle prédit correct
                  </div>
                </div>
                """)

                bins   = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01]
                labels = ["30-40%","40-50%","50-60%","60-70%","70-80%","80%+"]
                res["prob_bin"] = pd.cut(res["prob_model"], bins=bins, labels=labels, right=False)
                calib = (
                    res.groupby("prob_bin", observed=True)
                    .agg(N=("result","count"),
                         Prob=("prob_model","mean"),
                         Reel=("result", lambda x: (x=="won").mean()))
                    .reset_index()
                    .rename(columns={"prob_bin":"Tranche"})
                )
                calib = calib[calib["N"] > 0]

                for _, row in calib.iterrows():
                    bar_w  = int(row["Reel"] * 100)
                    c_bar  = "#3DFFA0" if row["Reel"] >= row["Prob"] else "#FFB347"
                    st.html(f"""
                    <div style="margin-bottom:8px;">
                      <div style="display:flex; justify-content:space-between; margin-bottom:4px;
                                  font-family:'DM Mono',monospace; font-size:0.62rem;">
                        <span style="color:#4E6A90;">{row['Tranche']}</span>
                        <span style="color:{c_bar};">{row['Reel']:.0%} réel
                          <span style="color:#2A3D5A;"> / {row['Prob']:.0%} modèle</span>
                          <span style="color:#2A3D5A;"> · {int(row['N'])}p</span>
                        </span>
                      </div>
                      <div style="height:4px; border-radius:3px; background:rgba(80,110,180,0.10); overflow:hidden;">
                        <div style="width:{bar_w}%; height:100%; background:{c_bar}; border-radius:3px; opacity:0.75;"></div>
                      </div>
                    </div>
                    """)
            else:
                st.info("Minimum 3 paris résolus requis.")

    # ── Analyse Backtest ─────────────────────────────────────────────────
    st.divider()
    section_header("Analyse Backtest (Pinnacle)")

    analyse_tour = st.radio("Circuit analyse", ["ATP", "WTA"],
                            horizontal=True, key="stats_analyse_tour")
    bt_tour = analyse_tour.lower()

    bt_thresholds = load_optimal_thresholds(bt_tour)
    bt_opt        = bt_thresholds.get("best_roi", {})
    bt_parquet    = ROOT / "data" / "models" / bt_tour / "backtest_all_candidates.parquet"

    if bt_parquet.exists():
        df_bt = pd.read_parquet(bt_parquet)

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
                         }), include_groups=False)
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
                         }), include_groups=False)
                         .sort_values('ROI', ascending=False)
                         .reset_index())
                st.dataframe(by_sv, hide_index=True, use_container_width=True)

        if 'clv' in df_bt.columns:
            section_header("Distribution CLV (Pinnacle)")
            avg_clv = df_bt['clv'].mean()
            pct_pos = (df_bt['clv'] > 0).mean()
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.metric("CLV moyen", f"{avg_clv:+.1%}")
            with col_c2:
                st.metric("Bets CLV positif", f"{pct_pos:.0%}")
            st.caption("CLV > 0 : notre modèle bat Pinnacle no-vig sur ce pari — signal d'edge réel.")
    else:
        st.info(f"Lance backtest_real.py --tour {bt_tour} pour générer les données.")
