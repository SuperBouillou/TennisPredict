# src/evaluate_2025.py
"""
Evaluation out-of-sample 2025 du modèle XGBoost ATP.

Utilise les données tennis-data.co.uk 2025 (matchs complétés + cotes réelles)
avec des features construites à partir du snapshot fin-2024 uniquement :
  - ELO final = elo_ratings_final.parquet (calculé jusqu'au 31/12/2024)
  - Profils joueurs = dernier match connu dans matches_features_final.parquet
  - H2H = toutes les données historiques jusqu'à fin 2024

Garantit l'absence de look-ahead bias : toutes les prédictions sont faites
avec des informations strictement antérieures à chaque match 2025.

Usage:
    python src/evaluate_2025.py
    python src/evaluate_2025.py --tour atp
"""

import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import get_tour_config, get_paths
from predict_today import (
    build_feature_vector,
    detect_tourney_level, _TOURNEY_LEVEL_MAP, _ROUND_ORDER,
)
import predict_today as pt


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DONNÉES DE BASE (snapshot fin-2024)
# ─────────────────────────────────────────────────────────────────────────────

def load_end_2024_profiles(processed_dir: Path) -> pd.DataFrame:
    """
    Construit les profils joueurs à partir de matches_features_final.parquet
    en ne gardant que les matchs <= 31/12/2024.
    Pour chaque joueur : dernier état connu des features.
    """
    path = processed_dir / "matches_features_final.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Introuvable : {path}")

    stat_keys = ['winrate_5', 'winrate_10', 'winrate_20',
                 'winrate_surf_Hard', 'winrate_surf_Clay', 'winrate_surf_Grass',
                 'streak', 'matches_7d', 'matches_14d', 'days_since',
                 'rank', 'rank_points',
                 'winrate_quality_5', 'winrate_quality_10', 'winrate_quality_20']

    df = pd.read_parquet(path)
    df['tourney_date'] = pd.to_datetime(df['tourney_date'])
    df = df[df['tourney_date'] <= '2024-12-31'].copy()
    df = df.sort_values('tourney_date')

    frames = []
    for side in ('p1', 'p2'):
        name_col = f'{side}_name'
        if name_col not in df.columns:
            continue
        stat_cols = [f'{side}_{k}' for k in stat_keys if f'{side}_{k}' in df.columns]
        sub = df[['tourney_date', name_col] + stat_cols].copy()
        # Rename to neutral keys
        rename = {name_col: 'player_name', 'tourney_date': 'last_seen'}
        for c in stat_cols:
            rename[c] = c.replace(f'{side}_', '')
        sub = sub.rename(columns=rename)
        frames.append(sub)

    df_all = pd.concat(frames, ignore_index=True)
    df_all = (df_all.sort_values('last_seen')
                    .drop_duplicates('player_name', keep='last')
                    .reset_index(drop=True))

    df_all['name_key']   = df_all['player_name'].str.lower().str.strip()
    df_all['last_name']  = df_all['player_name'].str.split().str[-1].str.lower()
    df_all['first_init'] = df_all['player_name'].str.split().str[0].str[0].str.lower()

    print(f"  Profils fin-2024 : {len(df_all):,} joueurs | "
          f"dernier match: {df_all['last_seen'].max().date()}")
    return df_all


def load_end_2024_elo(processed_dir: Path) -> pd.DataFrame:
    """Charge l'ELO final (fin-2024) — avant tout update ESPN/live."""
    path = processed_dir / "elo_ratings_final.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Introuvable : {path}")
    df = pd.read_parquet(path)
    df['last_name']  = df['full_name'].str.split().str[-1].str.lower()
    df['first_init'] = df['full_name'].str.split().str[0].str[0].str.lower()
    print(f"  ELO fin-2024  : {len(df):,} joueurs")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT MATCHS 2025
# ─────────────────────────────────────────────────────────────────────────────

_TD_SURFACE_MAP = {
    'Hard': 'Hard', 'Clay': 'Clay', 'Grass': 'Grass',
    'Indoor Hard': 'Hard', 'Carpet': 'Carpet',
}
_TD_ROUND_MAP = {
    '1st Round': 'R64', '2nd Round': 'R32', '3rd Round': 'R16',
    '4th Round': 'R32', 'Quarterfinals': 'QF', 'Semifinals': 'SF',
    'The Final': 'F', 'Round Robin': 'RR',
}


def load_2025_matches(odds_dir: Path, odds_filename, cutoff: date = None) -> pd.DataFrame:
    """Charge les matchs 2025 complétés depuis tennis-data.co.uk."""
    path = odds_dir / odds_filename(2025)
    if not path.exists():
        raise FileNotFoundError(f"Introuvable : {path}")

    df = pd.read_excel(path)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date', 'Winner', 'Loser'])
    df = df[df['Comment'] == 'Completed'].copy()

    # Filtrer par cutoff (ex: aujourd'hui pour éviter les matchs futurs)
    if cutoff:
        df = df[df['Date'] <= pd.Timestamp(cutoff)]

    # Normaliser
    df['surface'] = df['Surface'].map(_TD_SURFACE_MAP).fillna('Hard')
    df['round']   = df['Round'].map(_TD_ROUND_MAP).fillna('R32')

    series_col = 'Series' if 'Series' in df.columns else 'Tier'
    df['tourney_level'] = df[series_col] if series_col in df.columns else 'A'

    for col in ['B365W','B365L','PSW','PSL','MaxW','MaxL','AvgW','AvgL',
                'WRank','LRank','WPts','LPts']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['winner_name'] = df['Winner'].str.strip()
    df['loser_name']  = df['Loser'].str.strip()

    print(f"  Matchs 2025 : {len(df):,} | "
          f"{df['Date'].min().date()} -> {df['Date'].max().date()}")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# INDEX H2H PRÉ-CALCULÉ (batch, une seule lecture)
# ─────────────────────────────────────────────────────────────────────────────

def build_h2h_index(processed_dir: Path) -> dict:
    """
    Construit un index H2H depuis matches_features_final.parquet (<= 2024).
    Clé : frozenset({last_name_1, last_name_2})
    Valeur : liste de (p1_last, p2_last, target, surface)
    """
    path = processed_dir / "matches_features_final.parquet"
    df = pd.read_parquet(path, columns=['tourney_date', 'p1_name', 'p2_name', 'target', 'surface'])
    df = df[pd.to_datetime(df['tourney_date']) <= '2024-12-31']

    # Extraire last_name (dernier token)
    df['ln1'] = df['p1_name'].str.lower().str.split().str[-1]
    df['ln2'] = df['p2_name'].str.lower().str.split().str[-1]

    index: dict = {}
    for _, row in df[['ln1','ln2','target','surface']].iterrows():
        key = frozenset([row['ln1'], row['ln2']])
        if key not in index:
            index[key] = []
        index[key].append((row['ln1'], row['ln2'], int(row['target']), row['surface']))

    print(f"  Index H2H : {len(index):,} paires de joueurs")
    return index


def lookup_h2h(p1_last: str, p2_last: str, surface: str, h2h_index: dict) -> dict:
    """Lookup H2H depuis l'index pré-calculé (O(1) au lieu de O(n))."""
    key = frozenset([p1_last, p2_last])
    matches = h2h_index.get(key, [])

    if not matches:
        return {'h2h_p1_winrate': 0.5, 'h2h_surf_p1_winrate': 0.5,
                'h2h_total': 0, 'h2h_played': 0}

    wins = sum(
        1 for ln1, ln2, tgt, _ in matches
        if (ln1 == p1_last and tgt == 1) or (ln1 == p2_last and tgt == 0)
    )
    total    = len(matches)
    winrate  = wins / total

    surf_matches = [(ln1, ln2, tgt) for ln1, ln2, tgt, s in matches if s == surface]
    if surf_matches:
        surf_wins = sum(
            1 for ln1, ln2, tgt in surf_matches
            if (ln1 == p1_last and tgt == 1) or (ln1 == p2_last and tgt == 0)
        )
        surf_winrate = surf_wins / len(surf_matches)
    else:
        surf_winrate = winrate

    return {
        'h2h_p1_winrate':     winrate,
        'h2h_surf_p1_winrate': surf_winrate,
        'h2h_total':           total,
        'h2h_played':          total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCTION FEATURES PAR MATCH
# ─────────────────────────────────────────────────────────────────────────────

def td_name_to_dict(name: str, profiles: pd.DataFrame, df_elo: pd.DataFrame) -> dict:
    """
    Cherche un joueur tennis-data ('Sinner J.') dans les profils fin-2024.
    tennis-data format: "Sinner J." → last_name=sinner, first_init=j
    """
    name = str(name).strip()
    parts = name.replace('.', '').split()
    if not parts:
        return {}

    if len(parts) >= 2 and len(parts[-1]) <= 2:
        # Format "Nom Init." tennis-data
        last_name  = ' '.join(parts[:-1]).lower()
        first_init = parts[-1][0].lower()
    elif len(parts) == 2:
        # Format "Prénom Nom" ou "Nom Prenom"
        last_name  = parts[-1].lower()
        first_init = parts[0][0].lower()
    else:
        last_name  = parts[-1].lower()
        first_init = parts[0][0].lower() if parts else ''

    # Chercher dans profils (format Sackmann "Novak Djokovic" → last=djokovic)
    cands = profiles[profiles['last_name'] == last_name]
    if len(cands) == 0:
        cands = profiles[profiles['last_name'].str.startswith(last_name[:5])]

    if len(cands) > 1 and first_init:
        refined = cands[cands['first_init'] == first_init]
        if len(refined) > 0:
            cands = refined

    if len(cands) > 0:
        return cands.iloc[0].to_dict()

    # Fallback: chercher dans ELO
    cands_elo = df_elo[df_elo['last_name'] == last_name]
    if len(cands_elo) == 0:
        cands_elo = df_elo[df_elo['last_name'].str.startswith(last_name[:5])]
    if len(cands_elo) > 1 and first_init:
        r = cands_elo[cands_elo['first_init'] == first_init]
        if len(r) > 0:
            cands_elo = r
    if len(cands_elo) > 0:
        row = cands_elo.iloc[0]
        return {
            'player_name': row['full_name'],
            'elo':         row.get('elo_global', 1500),
            'elo_Hard':    row.get('elo_Hard', 1500),
            'elo_Clay':    row.get('elo_Clay', 1500),
            'elo_Grass':   row.get('elo_Grass', 1500),
            'last_name':   last_name,
            'first_init':  first_init,
        }

    return {'player_name': name, 'last_name': last_name, 'first_init': first_init}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def kelly_fraction(prob: float, odd: float, fraction: float = 0.25) -> float:
    b = odd - 1
    if b <= 0:
        return 0.0
    kelly = (b * prob - (1 - prob)) / b
    return max(0.0, kelly * fraction)


def run_betting_metrics(df: pd.DataFrame, label: str,
                        bankroll_init: float = 1000.0,
                        min_edge: float = 0.03,
                        min_prob: float = 0.55,
                        flat_stake: float = 10.0,
                        kelly_frac: float = 0.25,
                        max_kelly_pct: float = 0.05) -> None:
    """Calcule les métriques de paris flat et Kelly capé."""
    df = df.dropna(subset=['PSW', 'PSL']).copy()
    df = df[df['PSW'] > 1.3].copy()
    print(f"\n  [{label}] Matchs avec cotes Pinnacle : {len(df):,}")

    for strategy in ['flat', 'kelly']:
        bankroll = bankroll_init
        history  = []

        for _, row in df.iterrows():
            odd_w = float(row['PSW'])  # cote du vainqueur réel
            odd_l = float(row['PSL'])  # cote du perdant réel

            if odd_w <= 1 or odd_l <= 1:
                continue
            margin = 1/odd_w + 1/odd_l - 1
            imp_w  = (1/odd_w) / (1 + margin)
            imp_l  = (1/odd_l) / (1 + margin)

            # Côté gagnant = p1 (winner_name = vainqueur réel)
            # Notre modèle prédit prob_winner
            prob_w = float(row['prob_winner'])
            prob_l = float(row['prob_loser'])

            for our_prob, bk_imp, odd, won_val in [
                (prob_w, imp_w, odd_w, True),
                (prob_l, imp_l, odd_l, False),
            ]:
                edge = our_prob - bk_imp
                if edge < min_edge or our_prob < min_prob or odd < 1.3:
                    continue

                if strategy == 'flat':
                    stake = flat_stake
                else:
                    stake = bankroll * kelly_fraction(our_prob, odd, kelly_frac)
                    stake = min(stake, bankroll * max_kelly_pct)

                stake = min(max(stake, 0), bankroll)
                if stake <= 0:
                    continue

                pnl      = stake * (odd - 1) if won_val else -stake
                bankroll = bankroll + pnl

                history.append({
                    'pnl': pnl, 'stake': stake, 'won': int(won_val),
                    'bankroll': bankroll, 'edge': edge, 'odd': odd,
                    'our_prob': our_prob,
                })

        if not history:
            print(f"    {strategy}: aucun pari")
            continue

        dh = pd.DataFrame(history)
        n = len(dh)
        wr = dh['won'].mean()
        roi = dh['pnl'].sum() / dh['stake'].sum()
        final_bk = dh['bankroll'].iloc[-1]
        running_max = dh['bankroll'].cummax()
        max_dd = ((dh['bankroll'] - running_max) / running_max).min()
        sharpe = (dh['pnl'].mean() / dh['pnl'].std() * (252**0.5)
                  if dh['pnl'].std() > 0 else 0)

        lbl = f"Flat {flat_stake}€" if strategy == 'flat' else f"Kelly {kelly_frac} (cap {max_kelly_pct:.0%})"
        print(f"    {lbl:30s} | n={n:4d} | WR={wr:.1%} | ROI={roi:+.1%} | "
              f"P&L={dh['pnl'].sum():+.0f}€ | MaxDD={max_dd:.1%} | Sharpe={sharpe:.2f}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Evaluation out-of-sample 2025")
    parser.add_argument('--tour', default='atp', choices=['atp', 'wta'])
    parser.add_argument('--cutoff', default=None,
                        help="Date max (YYYY-MM-DD). Defaut: aujourd'hui")
    args = parser.parse_args()

    tour    = args.tour.lower()
    cfg     = get_tour_config(tour)
    paths   = get_paths(tour)
    cutoff  = date.fromisoformat(args.cutoff) if args.cutoff else date.today()

    PROCESSED_DIR = paths['processed_dir']
    MODELS_DIR    = paths['models_dir']
    ODDS_DIR      = paths['odds_dir']

    # Pointer predict_today vers les bons fichiers
    pt._PROCESSED_DIR = PROCESSED_DIR
    pt._PLAYER_FILE   = cfg['player_file']

    print("=" * 60)
    print(f"EVALUATION OUT-OF-SAMPLE 2025 — {tour.upper()}")
    print(f"Cutoff : {cutoff}")
    print("=" * 60)

    # ── 1. Modèle ─────────────────────────────────────────────────────────────
    print("\n-- Chargement modele --")
    model    = joblib.load(MODELS_DIR / "xgb_tuned.pkl")
    imputer  = joblib.load(MODELS_DIR / "imputer.pkl")
    features = joblib.load(MODELS_DIR / "feature_list.pkl")
    platt    = joblib.load(MODELS_DIR / "platt_scaler.pkl")
    print(f"  XGBoost tuned + Platt | {len(features)} features")

    # ── 2. Profils fin-2024 ───────────────────────────────────────────────────
    print("\n-- Profils joueurs fin-2024 --")
    profiles_2024 = load_end_2024_profiles(PROCESSED_DIR)
    elo_2024      = load_end_2024_elo(PROCESSED_DIR)

    print("\n-- Index H2H fin-2024 --")
    h2h_index = build_h2h_index(PROCESSED_DIR)

    # Patcher predict_today pour utiliser elo_ratings_final (pas le live)
    pt._PROCESSED_DIR = PROCESSED_DIR

    # ── 3. Matchs 2025 ────────────────────────────────────────────────────────
    print("\n-- Matchs 2025 --")
    if tour == 'wta':
        print("  WTA : tennis-data n'a pas de donnees WTA — arret.")
        sys.exit(0)

    df_2025 = load_2025_matches(ODDS_DIR, cfg['odds_filename'], cutoff=cutoff)

    # ── 4. Prédictions ────────────────────────────────────────────────────────
    print("\n-- Predictions --")
    results = []
    n_found  = 0
    n_miss   = 0

    for _, row in df_2025.iterrows():
        tournament = str(row.get('Tournament', ''))
        surface    = row['surface']
        round_str  = row['round']
        best_of    = int(row.get('Best of', 3))
        winner_td  = row['winner_name']
        loser_td   = row['loser_name']

        p_winner = td_name_to_dict(winner_td, profiles_2024, elo_2024)
        p_loser  = td_name_to_dict(loser_td,  profiles_2024, elo_2024)

        if not p_winner or not p_loser:
            n_miss += 1
            continue

        # Injecter le H2H pré-calculé dans predict_today (contourne compute_h2h lent)
        w_last = p_winner.get('last_name', winner_td.lower().split()[-1])
        l_last = p_loser.get('last_name',  loser_td.lower().split()[-1])
        h2h_stats = lookup_h2h(w_last, l_last, surface, h2h_index)

        # Patch temporaire du cache H2H de predict_today
        import predict_today as _pt
        _pt._h2h_cache = pd.DataFrame()  # force compute_h2h à retourner 0.5 (fallback)
        # On va patcher directement les features via fv dict trick ci-dessous

        match = {
            'tournament': tournament,
            'surface':    surface,
            'round':      round_str,
            'best_of':    best_of,
            'p1_name':    winner_td,
            'p2_name':    loser_td,
        }

        # Construire feature vector forward (winner=p1)
        Xv = build_feature_vector(match, p_winner, p_loser, features, elo_2024)
        # Injecter H2H calculé manuellement
        for feat_name, val in [
            ('h2h_p1_winrate',      h2h_stats['h2h_p1_winrate']),
            ('h2h_surf_p1_winrate', h2h_stats['h2h_surf_p1_winrate']),
            ('h2h_total',           h2h_stats['h2h_total']),
            ('h2h_played',          h2h_stats['h2h_played']),
        ]:
            if feat_name in features:
                Xv[features.index(feat_name)] = val

        Ximp = imputer.transform(Xv.reshape(1, -1))
        raw  = float(model.predict_proba(Ximp)[0, 1])
        prob_winner = float(platt.predict_proba([[raw]])[0, 1])

        # Feature vector backward (loser=p1) — H2H inversé
        h2h_inv = lookup_h2h(l_last, w_last, surface, h2h_index)
        match2  = {**match, 'p1_name': loser_td, 'p2_name': winner_td}
        Xv2     = build_feature_vector(match2, p_loser, p_winner, features, elo_2024)
        for feat_name, val in [
            ('h2h_p1_winrate',      h2h_inv['h2h_p1_winrate']),
            ('h2h_surf_p1_winrate', h2h_inv['h2h_surf_p1_winrate']),
            ('h2h_total',           h2h_inv['h2h_total']),
            ('h2h_played',          h2h_inv['h2h_played']),
        ]:
            if feat_name in features:
                Xv2[features.index(feat_name)] = val

        Ximp2  = imputer.transform(Xv2.reshape(1, -1))
        raw2   = float(model.predict_proba(Ximp2)[0, 1])
        prob_winner_bwd = float(platt.predict_proba([[raw2]])[0, 1])

        prob_winner_final = (prob_winner + (1 - prob_winner_bwd)) / 2

        n_found += 1
        results.append({
            'date':         row['Date'],
            'tournament':   tournament,
            'surface':      surface,
            'round':        round_str,
            'best_of':      best_of,
            'winner_name':  winner_td,
            'loser_name':   loser_td,
            'prob_winner':  prob_winner_final,
            'prob_loser':   1 - prob_winner_final,
            'correct':      prob_winner_final > 0.5,
            'PSW':          row.get('PSW', np.nan),
            'PSL':          row.get('PSL', np.nan),
            'B365W':        row.get('B365W', np.nan),
            'B365L':        row.get('B365L', np.nan),
            'MaxW':         row.get('MaxW', np.nan),
            'MaxL':         row.get('MaxL', np.nan),
        })

    print(f"  Joueurs trouvés : {n_found:,} matchs | non trouvés : {n_miss:,}")

    df_res = pd.DataFrame(results)

    if df_res.empty:
        print("  Aucun résultat — vérifier les données.")
        sys.exit(1)

    # ── 5. Métriques classification ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("METRIQUES CLASSIFICATION 2025")
    print("=" * 60)

    from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

    y_true = np.ones(len(df_res))           # winner_name a toujours target=1
    y_prob = df_res['prob_winner'].values
    y_pred = (y_prob > 0.5).astype(int)

    acc   = accuracy_score(y_true, y_pred)
    ll    = log_loss(y_true, y_prob, labels=[0, 1])
    brier = brier_score_loss(y_true, y_prob)

    print(f"  Matchs évalués : {len(df_res):,}")
    print(f"  Accuracy       : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Log-loss       : {ll:.4f}")
    print(f"  Brier score    : {brier:.4f}")

    # Par surface
    print("\n  Accuracy par surface :")
    for surf, grp in df_res.groupby('surface'):
        a = (grp['prob_winner'] > 0.5).mean()
        print(f"    {surf:8s}: {a:.3f} ({len(grp):4d} matchs)")

    # ── 6. Métriques paris ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("METRIQUES PARIS 2025 (Pinnacle)")
    print("=" * 60)
    run_betting_metrics(df_res, "2025 OOS",
                        bankroll_init=1000.0, min_edge=0.03, min_prob=0.55,
                        flat_stake=10.0, kelly_frac=0.25, max_kelly_pct=0.05)

    # ── 7. Sauvegarde ─────────────────────────────────────────────────────────
    out = MODELS_DIR / "evaluate_2025_results.parquet"
    df_res.to_parquet(out, index=False)
    print(f"\n  Resultats sauvegardes : {out.name}")
