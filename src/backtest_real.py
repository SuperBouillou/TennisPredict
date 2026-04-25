# src/backtest_real.py

import sys
import argparse
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config import get_tour_config, get_paths, make_dirs


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT ET NETTOYAGE DES COTES
# ─────────────────────────────────────────────────────────────────────────────

def load_real_odds(years: list, odds_dir: Path, odds_filename) -> pd.DataFrame:
    """Charge et consolide les fichiers de cotes tennis-data.co.uk"""
    dfs = []

    for year in years:
        filename = odds_filename(year)
        path = odds_dir / filename
        if not path.exists():
            print(f"  {year} manquant ({filename})")
            continue

        df = pd.read_excel(path, engine='openpyxl')
        df['year'] = year
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # Nettoyage
    df_all['Date'] = pd.to_datetime(df_all['Date'], errors='coerce')
    df_all = df_all.dropna(subset=['Date', 'Winner', 'Loser'])

    # Garder uniquement les matchs complétés
    df_all = df_all[df_all['Comment'] == 'Completed'].copy()

    # Nettoyage des noms — format "Nom P." dans les cotes
    # On va normaliser pour la jointure
    df_all['winner_clean'] = df_all['Winner'].str.strip()
    df_all['loser_clean']  = df_all['Loser'].str.strip()

    # Cotes numériques
    for col in ['B365W','B365L','PSW','PSL','MaxW','MaxL','AvgW','AvgL']:
        df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

    print(f"Cotes chargées : {len(df_all):,} matchs ({years[0]}-{years[-1]})")
    print(f"   Colonnes       : {list(df_all.columns)}")
    print(f"   Période        : {df_all['Date'].min().date()} → {df_all['Date'].max().date()}")

    # Taux de couverture Pinnacle
    ps_cov = df_all['PSW'].notna().mean()
    b365_cov = df_all['B365W'].notna().mean()
    print(f"   Couverture Pinnacle : {ps_cov:.1%} | Bet365 : {b365_cov:.1%}")

    return df_all


def normalize_player_name(name: str) -> str:
    """
    Normalise les noms de joueurs pour la jointure.
    tennis-data : "Djokovic N."
    Sackmann    : "Novak Djokovic"
    → On extrait le prénom en initial + nom complet
    """
    if pd.isna(name):
        return ''
    name = str(name).strip()

    # Format "Nom P." → extraire nom de famille
    parts = name.split()
    if len(parts) >= 2:
        # Dernier token = initiale du prénom (ex: "N.")
        # Premier(s) token(s) = nom de famille
        last_name = ' '.join(parts[:-1]) if parts[-1].endswith('.') else name
        return last_name.lower()
    return name.lower()


def build_name_lookup(df_sackmann: pd.DataFrame) -> dict:
    """
    Construit un dictionnaire nom_de_famille → player_id + full_name
    depuis le dataset Sackmann.
    """
    lookup = {}

    for _, row in df_sackmann[['winner_id','winner_name']].drop_duplicates().iterrows():
        full = str(row['winner_name']).strip()
        parts = full.split()
        if len(parts) >= 2:
            # Clé = "Nom Initial" ex: "djokovic n"
            last  = parts[-1].lower()
            first = parts[0][0].lower()
            key1  = f"{last} {first}"
            key2  = last
            lookup[key1] = (row['winner_id'], full)
            lookup[key2] = (row['winner_id'], full)

    return lookup


# ─────────────────────────────────────────────────────────────────────────────
# JOINTURE COTES ↔ PRÉDICTIONS
# ─────────────────────────────────────────────────────────────────────────────

import unicodedata as _ud

# Noms de famille composés connus (construits dynamiquement depuis les odds)
_COMPOUND_LASTNAMES = set()


def _strip_accents(s: str) -> str:
    return ''.join(c for c in _ud.normalize('NFD', s) if _ud.category(c) != 'Mn')


def normalize_name_for_join(name: str) -> str:
    """
    v2 : Normalisation robuste pour la jointure entre les deux sources.

    tennis-data : "Djokovic N." / "Huesler M.A." / "Bautista Agut R." / "Auger-Aliassime F."
    Sackmann    : "Novak Djokovic" / "Marc Andrea Huesler" / "Roberto Bautista Agut"

    Stratégie :
      1. Strip accents
      2. Supprimer tirets et apostrophes
      3. tennis-data (dernier token contient un point) → last_name + première initiale
      4. Sackmann 2 mots → last_name + première initiale
      5. Sackmann 3+ mots → détection noms composés via _COMPOUND_LASTNAMES
    """
    if pd.isna(name) or name == '':
        return ''
    name = _strip_accents(str(name).strip())
    name = name.replace('-', ' ').replace("'", '')
    parts = name.split()
    if not parts:
        return ''

    # Format tennis-data : dernier token contient un point → "Nom P." ou "Nom M.A."
    if len(parts) >= 2 and '.' in parts[-1] and parts[-1].endswith('.'):
        last_name  = ' '.join(parts[:-1]).lower()
        first_init = parts[-1][0].lower()
        return f"{last_name}_{first_init}"

    # Format Sackmann 2 mots : "Novak Djokovic"
    if len(parts) == 2:
        return f"{parts[1].lower()}_{parts[0][0].lower()}"

    # Format Sackmann 3+ mots : détecter nom composé
    if len(parts) >= 3:
        first_init = parts[0][0].lower()
        last3 = ' '.join(parts[-3:]).lower() if len(parts) >= 4 else None
        last2 = ' '.join(parts[-2:]).lower()
        if last3 and last3 in _COMPOUND_LASTNAMES:
            return f"{last3}_{first_init}"
        if last2 in _COMPOUND_LASTNAMES:
            return f"{last2}_{first_init}"
        # Fallback : dernier mot seul
        return f"{parts[-1].lower()}_{first_init}"

    return parts[0].lower()


def build_compound_lastnames(df_odds: pd.DataFrame) -> set:
    """Construit le set des noms de famille composés depuis tennis-data."""
    lastnames = set()
    for col in ['winner_clean', 'loser_clean']:
        for name in df_odds[col].dropna():
            name = _strip_accents(str(name)).replace('-', ' ').replace("'", '')
            parts = name.split()
            # Format tennis-data 3+ tokens avec point final → nom composé de 2 mots
            if len(parts) >= 3 and '.' in parts[-1] and parts[-1].endswith('.'):
                ln2 = ' '.join(parts[-3:-1]).lower()
                lastnames.add(ln2)
                if len(parts) >= 4:
                    ln3 = ' '.join(parts[-4:-1]).lower()
                    lastnames.add(ln3)
    return lastnames


def join_odds_to_predictions(df_pred: pd.DataFrame,
                              df_odds: pd.DataFrame,
                              compound_lastnames: set = None) -> pd.DataFrame:

    odds_cols = ['B365W','B365L','PSW','PSL','MaxW','MaxL','AvgW','AvgL']

    # Injecter les noms composés dans le module pour normalize_name_for_join
    if compound_lastnames is not None:
        global _COMPOUND_LASTNAMES
        _COMPOUND_LASTNAMES = compound_lastnames

    df_pred = df_pred.copy()
    df_odds = df_odds.copy()

    df_pred['tourney_date'] = pd.to_datetime(df_pred['tourney_date'])
    df_odds['Date']         = pd.to_datetime(df_odds['Date'])

    df_pred['key_p1'] = df_pred['p1_name'].apply(normalize_name_for_join)
    df_pred['key_p2'] = df_pred['p2_name'].apply(normalize_name_for_join)
    df_odds['key_w']  = df_odds['winner_clean'].apply(normalize_name_for_join)
    df_odds['key_l']  = df_odds['loser_clean'].apply(normalize_name_for_join)

    # Initialiser les colonnes résultat
    for col in odds_cols:
        df_pred[col] = np.nan
    df_pred['p1_is_winner_odds'] = np.nan

    # ── Construire un lookup dict pour la jointure ────────────────────────────
    # Clé : (date, name_a, name_b) → row d'odds
    # On indexe les deux sens (winner/loser et loser/winner)

    lookup = {}  # (date, key1, key2) → dict de cotes + p1_is_winner

    for _, row in df_odds.iterrows():
        date  = row['Date'].date()
        kw    = row['key_w']
        kl    = row['key_l']
        cotes = {c: row[c] for c in odds_cols if c in row.index}

        # Sens normal : p1=winner, p2=loser
        lookup[(date, kw, kl)] = {**cotes, 'p1_is_winner_odds': True}

        # Sens inversé : p1=loser, p2=winner → inverser les cotes W↔L
        cotes_inv = {}
        for bk in ['B365','PS','Max','Avg']:
            w, l = f'{bk}W', f'{bk}L'
            if w in cotes and l in cotes:
                cotes_inv[w] = cotes[l]
                cotes_inv[l] = cotes[w]
        lookup[(date, kl, kw)] = {**cotes_inv, 'p1_is_winner_odds': False}

    print(f"\n  Lookup construit : {len(lookup):,} entrées")
    print(f"  Tentatives de jointure (±14 jours) :")

    joined_count = 0

    for idx, row in df_pred.iterrows():
        if pd.notna(df_pred.at[idx, 'PSW']):
            continue

        date  = row['tourney_date'].date()
        kp1   = row['key_p1']
        kp2   = row['key_p2']

        # Tenter ±14 jours (Sackmann stocke la date de début du tournoi)
        found = False
        for delta in range(-14, 15):
            test_date = date + pd.Timedelta(days=delta).to_pytimedelta()

            if (test_date, kp1, kp2) in lookup:
                entry = lookup[(test_date, kp1, kp2)]
                for col in odds_cols:
                    if col in entry:
                        df_pred.at[idx, col] = entry[col]
                df_pred.at[idx, 'p1_is_winner_odds'] = entry['p1_is_winner_odds']
                joined_count += 1
                found = True
                break

    # ── Rapport ──────────────────────────────────────────────────────────────
    final_rate = df_pred['PSW'].notna().mean()
    joined     = df_pred[df_pred['PSW'].notna()]
    unjoined   = df_pred[df_pred['PSW'].isna()]

    print(f"\n  Taux de jointure final   : {final_rate:.1%} "
          f"({df_pred['PSW'].notna().sum():,}/{len(df_pred):,})")
    print(f"  Target=1 joints    : {joined['target'].mean():.3f} ← doit être ~0.500")
    print(f"  Target=1 non joints: {unjoined['target'].mean():.3f}")

    if len(joined) > 0:
        coherent = (
            ((joined['p1_is_winner_odds'] == True)  & (joined['target'] == 1)) |
            ((joined['p1_is_winner_odds'] == False) & (joined['target'] == 0))
        ).mean()
        print(f"  Cohérence odds/target      : {coherent:.1%} ← doit être ~100%")

    return df_pred

# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST AVEC COTES RÉELLES
# ─────────────────────────────────────────────────────────────────────────────

def kelly_fraction(prob: float, odd: float, fraction: float = 0.25) -> float:
    b = odd - 1
    if b <= 0:
        return 0.0
    q     = 1 - prob
    kelly = (b * prob - q) / b
    return max(0.0, kelly * fraction)


def run_backtest(df: pd.DataFrame,
                 odds_col_w: str = 'PSW',
                 odds_col_l: str = 'PSL',
                 min_edge: float  = 0.03,
                 min_prob: float  = 0.55,
                 min_odd: float   = 1.30,
                 min_bk_dir_prob: float = 0.40,
                 bankroll_init: float = 1000.0,
                 strategy: str    = 'flat',
                 flat_stake: float= 10.0,
                 kelly_frac: float= 0.25,
                 max_kelly_pct: float = 0.05) -> tuple:
    """
    Backtest avec cotes réelles.

    odds_col_w      : colonne cote du vainqueur  (PSW, B365W, MaxW, AvgW)
    odds_col_l      : colonne cote du perdant    (PSL, B365L, MaxL, AvgL)
    min_bk_dir_prob : probabilité implicite bookmaker minimale pour le côté parié.
                      Bloque les paris où le marché donne < 35% à ce joueur —
                      évite de miser sur des outsiders extrêmes que le modèle
                      surévalue par manque de données d'entraînement.
                      Ex: cote > 2.6 → bk_imp ~0.36 → autorisé
                          cote > 2.85 → bk_imp ~0.32 → bloqué
    """
    # Garder uniquement les matchs avec cotes disponibles
    df = df.dropna(subset=[odds_col_w, odds_col_l]).copy()
    df = df[df[odds_col_w] > min_odd].copy()
    df = df.sort_values('tourney_date').reset_index(drop=True)

    print(f"\n  [{odds_col_w}] Matchs avec cotes : {len(df):,}")

    bankroll = bankroll_init
    history  = []

    for _, row in df.iterrows():

        # Après lookup, PSW/PSL sont DÉJÀ orientés P1/P2 — pas d'inversion
        odd_p1 = float(row[odds_col_w])
        odd_p2 = float(row[odds_col_l])

        if odd_p1 <= 1 or odd_p2 <= 1:
            continue
        margin    = 1/odd_p1 + 1/odd_p2 - 1
        bk_imp_p1 = (1/odd_p1) / (1 + margin)
        bk_imp_p2 = (1/odd_p2) / (1 + margin)

        p1_prob = float(row['p1_prob'])
        p2_prob = 1 - p1_prob
        target  = int(row['target'])

        for side, our_prob, odd, bk_imp, won_val in [
            ('p1', p1_prob, odd_p1, bk_imp_p1, target == 1),
            ('p2', p2_prob, odd_p2, bk_imp_p2, target == 0),
        ]:
            edge = our_prob - bk_imp
            ev   = our_prob * odd - 1

            # Filtre principal : edge, probabilité modèle, cote minimale
            if edge < min_edge or our_prob < min_prob or odd < min_odd:
                continue

            # Filtre direction marché : ne pas parier contre le consensus fort
            # Si le marché donne < min_bk_dir_prob à ce joueur, le modèle
            # crée un "edge" artificiel (outsider × proba modèle surévaluée).
            if bk_imp < min_bk_dir_prob:
                continue

            if strategy == 'flat':
                stake = flat_stake
            elif strategy == 'kelly':
                raw_kelly = bankroll * kelly_fraction(our_prob, odd, kelly_frac)
                stake = min(raw_kelly, bankroll * max_kelly_pct)
            elif strategy == 'percent':
                stake = bankroll * 0.02

            stake = min(max(stake, 0), bankroll)
            if stake <= 0:
                continue

            pnl      = stake * (odd - 1) if won_val else -stake
            bankroll = bankroll + pnl

            history.append({
                'date'              : row['tourney_date'],
                'surface'           : row['surface'],
                'level'             : row['tourney_level'],
                'p1_name'           : row['p1_name'],
                'p2_name'           : row['p2_name'],
                'bet_on'            : side,
                'p1_is_winner_odds' : row.get('p1_is_winner_odds', np.nan),
                'our_prob'          : our_prob,
                'bk_imp_prob'       : bk_imp,
                'edge'              : edge,
                'ev'                : ev,
                'odd'               : odd,
                'stake'             : stake,
                'won'               : int(won_val),
                'pnl'               : pnl,
                'bankroll'          : bankroll,
                'bookmaker'         : odds_col_w.replace('W',''),
                'clv'               : our_prob / bk_imp - 1 if bk_imp > 0 else 0.0,
            })

    df_hist = pd.DataFrame(history)
    return df_hist


def backtest_metrics(df_hist: pd.DataFrame,
                     bankroll_init: float,
                     label: str) -> dict:

    if len(df_hist) == 0:
        print(f"  [{label}] Aucun pari")
        return {}

    n            = len(df_hist)
    wins         = df_hist['won'].sum()
    total_staked = df_hist['stake'].sum()
    total_pnl    = df_hist['pnl'].sum()
    roi          = total_pnl / total_staked
    final_bk     = df_hist['bankroll'].iloc[-1]

    running_max  = df_hist['bankroll'].cummax()
    drawdown     = (df_hist['bankroll'] - running_max) / running_max
    max_dd       = drawdown.min()

    sharpe       = (df_hist['pnl'].mean() / df_hist['pnl'].std() * np.sqrt(252)
                    if df_hist['pnl'].std() > 0 else 0)

    print(f"\n  [{label}]")
    print(f"    Paris          : {n:,}")
    print(f"    Win rate       : {wins/n:.1%}")
    print(f"    ROI            : {roi:+.2%}")
    print(f"    P&L            : {total_pnl:+.1f}€")
    print(f"    Bankroll finale: {final_bk:.1f}€")
    print(f"    Max drawdown   : {max_dd:.1%}")
    print(f"    Sharpe         : {sharpe:.2f}")
    print(f"    Edge moyen     : {df_hist['edge'].mean():+.1%}")
    print(f"    Cote moyenne   : {df_hist['odd'].mean():.2f}")
    print(f"    EV moyen       : {df_hist['ev'].mean():+.1%}")
    if 'clv' in df_hist.columns:
        print(f"    CLV moyen      : {df_hist['clv'].mean():+.1%}")

    return {
        'label': label, 'n_bets': n, 'win_rate': wins/n,
        'roi': roi, 'pnl': total_pnl, 'final_bk': final_bk,
        'max_dd': max_dd, 'sharpe': sharpe,
        'avg_edge': df_hist['edge'].mean(),
        'avg_odd': df_hist['odd'].mean(),
        'avg_clv': df_hist['clv'].mean() if 'clv' in df_hist.columns else 0.0,
    }


def plot_real_backtest(results: dict, bankroll_init: float,
                       level_labels: dict, models_dir: Path) -> None:

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Backtest REEL 2023-2024 — Cotes Pinnacle', fontsize=14)

    colors = {'Pinnacle': 'steelblue', 'Bet365': 'orange',
              'Best': 'green', 'Avg': 'red'}

    # 1. Évolution bankroll
    ax = axes[0, 0]
    ax.axhline(y=bankroll_init, color='black', linestyle='--', alpha=0.5)
    for label, df_h in results.items():
        if len(df_h) > 0:
            color = colors.get(label.split('_')[0], 'gray')
            ax.plot(df_h['date'], df_h['bankroll'],
                    label=f"{label} ({len(df_h)} paris)", color=color)
    ax.set_title('Evolution du bankroll — Flat betting')
    ax.set_ylabel('Bankroll (€)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Distribution des edges
    ax = axes[0, 1]
    first = list(results.values())[0]
    if len(first) > 0:
        ax.hist(first['edge'], bins=25, color='steelblue', alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='--')
        ax.axvline(x=first['edge'].mean(), color='red',
                   linestyle='--', label=f"Moyen: {first['edge'].mean():.1%}")
        ax.set_title('Distribution des edges réels')
        ax.set_xlabel('Edge (notre prob - prob implicite Pinnacle)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 3. P&L cumulatif par surface
    ax = axes[1, 0]
    first = list(results.values())[0]
    if len(first) > 0:
        for surface, color in [('Hard','steelblue'),('Clay','orange'),('Grass','green')]:
            s = first[first['surface'] == surface]
            if len(s) > 0:
                ax.plot(range(len(s)), s['pnl'].cumsum(),
                        label=f"{surface} ({len(s)})", color=color)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_title('P&L cumulatif par surface')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 4. ROI par niveau de tournoi — utilise level_labels depuis la config
    ax = axes[1, 1]
    first = list(results.values())[0]
    if len(first) > 0 and 'level' in first.columns:
        first = first.copy()
        first['level_name'] = first['level'].map(level_labels).fillna('Autre')
        roi_by_level = first.groupby('level_name').apply(
            lambda x: x['pnl'].sum() / x['stake'].sum()
        )
        roi_by_level.plot(kind='bar', ax=ax, color='steelblue', alpha=0.7)
        ax.axhline(y=0, color='red', linestyle='--')
        ax.set_title('ROI par niveau de tournoi')
        ax.set_ylabel('ROI')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(models_dir / "backtest_real.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nGraphique sauvegardé : backtest_real.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Backtest réel avec cotes tennis-data par tour")
    parser.add_argument('--tour', default='atp', choices=['atp', 'wta'],
                        help="Tour à traiter : atp ou wta (défaut: atp)")
    args = parser.parse_args()

    tour  = args.tour.lower()
    cfg   = get_tour_config(tour)
    paths = get_paths(tour)
    make_dirs(tour)

    MODELS_DIR = paths['models_dir']
    ODDS_DIR   = paths['odds_dir']

    BANKROLL   = 1000.0
    TEST_YEARS = [2025]

    print("=" * 55)
    print(f"BACKTEST REEL — COTES TENNIS-DATA.CO.UK — {tour.upper()}")
    print("=" * 55)

    # ── Chargement prédictions modèle ────────────────────────────────────────
    splits   = joblib.load(MODELS_DIR / "splits.pkl")
    features = joblib.load(MODELS_DIR / "feature_list.pkl")
    imputer  = joblib.load(MODELS_DIR / "imputer.pkl")
    model    = joblib.load(MODELS_DIR / "xgb_tuned.pkl")

    # Charger le scaler de calibration — préférer platt_pinnacle.pkl (calibré sur
    # Pinnacle no-vig, évite la surconfiance vs marché) sinon fallback platt_scaler.pkl
    platt = None
    for platt_name in ("platt_pinnacle.pkl", "platt_scaler.pkl"):
        p = MODELS_DIR / platt_name
        if p.exists():
            platt = joblib.load(p)
            print(f"  Calibration scaler : {platt_name}")
            break

    # Utilise valid si test est vide (split train≤2022 / valid=2023-2024 / test≥2025)
    if len(splits['X_test']) > 0:
        X_test = splits['X_test']
        y_test = splits['y_test']
        meta   = splits['meta_test']
    else:
        print("  Test set vide — utilisation du valid set (2023-2024)")
        X_test = splits['X_valid']
        y_test = splits['y_valid']
        meta   = splits['meta_valid']

    X_imp    = imputer.transform(X_test)
    raw_prob = model.predict_proba(X_imp)[:, 1]

    # Charger les scalers surface-spécifiques (platt_Hard.pkl, etc.)
    surface_scalers = {}
    for surf in ['Hard', 'Clay', 'Grass']:
        sp = MODELS_DIR / f"platt_{surf}.pkl"
        if sp.exists():
            surface_scalers[surf] = joblib.load(sp)
    if surface_scalers:
        print(f"  Scalers surface : {list(surface_scalers.keys())}")

    # Appliquer la calibration par surface si disponible,
    # fallback sur le scaler global (LinearRegression ou LogisticRegression)
    def _apply_scaler(scaler, probs):
        if scaler is None:
            return probs
        if hasattr(scaler, 'predict_proba'):
            return scaler.predict_proba(probs.reshape(-1, 1))[:, 1]
        return np.clip(scaler.predict(probs.reshape(-1, 1)), 0.01, 0.99)

    p1_prob = np.full(len(raw_prob), np.nan)
    surf_col = meta['surface'].values if 'surface' in meta.columns else np.array([''] * len(meta))
    for surf, scaler in surface_scalers.items():
        mask = surf_col == surf
        if mask.any():
            p1_prob[mask] = _apply_scaler(scaler, raw_prob[mask])
    # Fallback global pour les surfaces sans scaler ou surfaces rares (Carpet…)
    remaining = np.isnan(p1_prob)
    if remaining.any():
        p1_prob[remaining] = _apply_scaler(platt, raw_prob[remaining])

    calib = "Platt" if platt else "brut"
    print(f"  Modèle : xgb_tuned | Calibration : {calib}")

    df_pred = meta.copy().reset_index(drop=True)
    df_pred['p1_prob'] = p1_prob
    df_pred['target']  = y_test.values

    print(f"\nPrédictions : {len(df_pred):,} matchs")

    # ── Chargement cotes réelles ─────────────────────────────────────────────
    print("\n── Chargement cotes réelles ─────────────────────────")
    df_odds = load_real_odds(TEST_YEARS, ODDS_DIR, cfg['odds_filename'])

    # Initialiser le dictionnaire des noms composés (ex: "Bautista Agut", "Auger Aliassime")
    compound_lastnames = build_compound_lastnames(df_odds)

    print("\n── Diagnostic noms ──────────────────────────────────")
    print("  tennis-data 2023 (5 premiers) :")
    for name in df_odds['winner_clean'].head(5):
        print(f"    '{name}' → '{normalize_name_for_join(name)}'")
    print("  Sackmann 2023 (5 premiers) :")
    for name in df_pred['p1_name'].head(5):
        print(f"    '{name}' → '{normalize_name_for_join(name)}'")

    # ── Jointure ─────────────────────────────────────────────────────────────
    print("\n── Jointure prédictions ↔ cotes ─────────────────────")
    df_joined = join_odds_to_predictions(df_pred, df_odds, compound_lastnames)


    # Ajoute ce bloc AVANT run_backtest dans le main :

    print("\n── Diagnostic jointure ─────────────────────────────")
    joined = df_joined[df_joined['PSW'].notna()].copy()

    # Séparation selon le sens de la jointure
    upset_join  = joined[joined['p1_is_winner_odds'] == False]  # p1 = bookmaker's loser
    normal_join = joined[joined['p1_is_winner_odds'] == True]   # p1 = bookmaker's winner

    print(f"  Jointure normale (p1=favori bk) : {len(normal_join):,} matchs "
          f"| target=1 : {normal_join['target'].mean():.3f}")
    print(f"  Jointure inversée (p1=outsider) : {len(upset_join):,} matchs "
          f"| target=1 : {upset_join['target'].mean():.3f}")

    # Distribution des cotes PSW
    print(f"\n  Cote PSW moyenne - jointure normale  : {normal_join['PSW'].mean():.2f}")
    print(f"  Cote PSW moyenne - jointure inversée : {upset_join['PSW'].mean():.2f}")

    # Accuracy modèle selon le sens
    p1_probs = joined['p1_prob'].values
    targets  = joined['target'].values
    preds    = (p1_probs > 0.5).astype(int)

    acc_normal = (preds[joined['p1_is_winner_odds'] == True]  ==
                  targets[joined['p1_is_winner_odds'] == True]).mean()
    acc_upset  = (preds[joined['p1_is_winner_odds'] == False] ==
                  targets[joined['p1_is_winner_odds'] == False]).mean()

    print(f"\n  Accuracy modèle - jointure normale  : {acc_normal:.3f}")
    print(f"  Accuracy modèle - jointure inversée : {acc_upset:.3f}")

    # Calibration rapide
    print(f"\n  Calibration sur sous-ensemble joint :")
    bins = np.arange(0.5, 1.01, 0.05)
    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]
        mask = (p1_probs >= lo) & (p1_probs < hi)
        if mask.sum() >= 10:
            actual_wr = targets[mask].mean()
            print(f"    prob [{lo:.2f}-{hi:.2f}] → {mask.sum():4d} paris, "
                  f"win rate réel : {actual_wr:.3f} "
                  f"{'ok' if abs(actual_wr - (lo+hi)/2) < 0.07 else 'warn'}")

    # Value bets selon sens de jointure
    print(f"\n── Value bets par sens de jointure ──────────────────")
    for name, df_sub in [("Normale (p1=favori)", normal_join),
                         ("Inversée (p1=outsider)", upset_join)]:
        vb_mask = (df_sub['p1_prob'] > 0.55) & \
                  ((df_sub['p1_prob'] - 1/df_sub['PSW']) > 0.03)
        vb = df_sub[vb_mask]
        if len(vb) > 0:
            wr = (vb['target'] == 1).mean()
            avg_odd = vb['PSW'].mean()
            print(f"  {name}: {len(vb):,} VB | win rate : {wr:.3f} | cote moy : {avg_odd:.2f}")


    # ── Candidats bruts pour optimize_thresholds.py ──────────────────────────
    # Sans filtre min_bk_dir_prob pour garder tous les candidats potentiels
    print("\n── Sauvegarde candidats bruts ───────────────────────")
    hist_all = run_backtest(
        df_joined,
        odds_col_w='PSW', odds_col_l='PSL',
        min_edge=0.0, min_prob=0.50, min_odd=1.10,
        min_bk_dir_prob=0.0,
        bankroll_init=BANKROLL, strategy='flat', flat_stake=10.0,
    )
    if len(hist_all) > 0:
        hist_all.to_parquet(MODELS_DIR / "backtest_all_candidates.parquet", index=False)
        print(f"  {len(hist_all):,} candidats sauvegardés → backtest_all_candidates.parquet")
        print(f"  CLV moyen  : {hist_all['clv'].mean():+.1%}")
        print(f"  Edge moyen : {hist_all['edge'].mean():+.1%}")

    # ── Backtests par bookmaker (Flat 10€) ───────────────────────────────────
    print("\n── Backtests par bookmaker (Flat 10€) ───────────────")

    all_results  = {}
    all_metrics  = []

    for bk_w, bk_l, label in [
        ('PSW',   'PSL',   'Pinnacle'),
        ('B365W', 'B365L', 'Bet365'),
        ('MaxW',  'MaxL',  'Best'),
        ('AvgW',  'AvgL',  'Avg'),
    ]:
        hist = run_backtest(
            df_joined,
            odds_col_w    = bk_w,
            odds_col_l    = bk_l,
            min_edge      = 0.03,
            min_prob      = 0.55,
            min_odd       = 1.30,
            bankroll_init = BANKROLL,
            strategy      = 'flat',
            flat_stake    = 10.0,
        )
        all_results[label] = hist
        m = backtest_metrics(hist, BANKROLL, label)
        if m:
            all_metrics.append(m)

    # ── Backtests Pinnacle — comparaison stratégies ───────────────────────────
    print("\n" + "=" * 55)
    print("COMPARAISON STRATEGIES — Pinnacle")
    print("=" * 55)

    strategy_results = {}
    strategy_metrics = []

    for strategy, label, extra in [
        ('flat',    'Flat_10€',         {'flat_stake': 10.0}),
        ('percent', 'Pct_2%',           {}),
        ('kelly',   'Kelly_1/4_cap5%',  {'kelly_frac': 0.25,  'max_kelly_pct': 0.05}),
        ('kelly',   'Kelly_1/4_cap2%',  {'kelly_frac': 0.25,  'max_kelly_pct': 0.02}),
        ('kelly',   'Kelly_1/8_cap5%',  {'kelly_frac': 0.125, 'max_kelly_pct': 0.05}),
    ]:
        hist = run_backtest(
            df_joined,
            odds_col_w    = 'PSW',
            odds_col_l    = 'PSL',
            min_edge      = 0.03,
            min_prob      = 0.55,
            min_odd       = 1.30,
            bankroll_init = BANKROLL,
            strategy      = strategy,
            **extra,
        )
        strategy_results[label] = hist
        m = backtest_metrics(hist, BANKROLL, label)
        if m:
            strategy_metrics.append(m)

    print("\n── Tableau comparatif stratégies ────────────────────")
    if strategy_metrics:
        df_strat = pd.DataFrame(strategy_metrics)
        print(df_strat[['label','n_bets','win_rate','roi',
                         'pnl','max_dd','sharpe','avg_edge']].to_string(index=False))

    for label, hist in strategy_results.items():
        if len(hist) > 0:
            hist.to_parquet(MODELS_DIR / f"backtest_strat_{label.replace('/','_')}.parquet",
                            index=False)

    # ── Tableau comparatif ────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("TABLEAU COMPARATIF — BOOKMAKERS")
    print("=" * 55)
    if all_metrics:
        df_m = pd.DataFrame(all_metrics)
        print(df_m[['label','n_bets','win_rate','roi',
                     'max_dd','sharpe','avg_edge']].to_string(index=False))

    plot_real_backtest(all_results, BANKROLL,
                       level_labels=cfg['level_labels'],
                       models_dir=MODELS_DIR)

    # ── Diagnostic breakdown des paris (Pinnacle) ─────────────────────────────
    print("\n" + "=" * 55)
    print(f"BREAKDOWN PARIS — Pinnacle (orientation × côté)")
    print("=" * 55)
    ps_hist = all_results.get('Pinnacle', pd.DataFrame())
    if len(ps_hist) > 0 and 'p1_is_winner_odds' in ps_hist.columns:
        ps_hist['orientation'] = ps_hist['p1_is_winner_odds'].map(
            {True: 'Normal (p1=winner)', False: 'Inversé (p1=loser)'}
        ).fillna('?')
        breakdown = ps_hist.groupby(['orientation', 'bet_on']).agg(
            n_bets   = ('won', 'count'),
            win_rate = ('won', 'mean'),
            roi      = ('pnl', lambda x: x.sum() / (ps_hist.loc[x.index, 'stake'].sum())),
            avg_odd  = ('odd', 'mean'),
        ).round(3)
        print(breakdown.to_string())
        print("""
  Interprétation attendue :
    Normal + p1  → p1 est le vainqueur réel, model correct → win rate ~élevé
    Normal + p2  → p2 est le perdant réel, model faux → win rate ~0%
    Inversé + p1 → p1 est le perdant réel, model faux → win rate ~0%
    Inversé + p2 → p2 est le vainqueur réel, model correct → win rate ~élevé
""")

    # ── Breakdown ROI par niveau de tournoi (Pinnacle) — via level_labels ────
    print("\n" + "=" * 55)
    print("BREAKDOWN ROI — PAR NIVEAU DE TOURNOI (Pinnacle)")
    print("=" * 55)
    ps_hist = all_results.get('Pinnacle', pd.DataFrame())
    if len(ps_hist) > 0 and 'level' in ps_hist.columns:
        ps_hist = ps_hist.copy()
        ps_hist['level_name'] = ps_hist['level'].map(cfg['level_labels']).fillna('Autre')
        by_level = ps_hist.groupby('level_name').apply(lambda g: pd.Series({
            'n_bets'  : len(g),
            'win_rate': g['won'].mean(),
            'roi'     : g['pnl'].sum() / g['stake'].sum(),
            'pnl'     : g['pnl'].sum(),
            'avg_odd' : g['odd'].mean(),
            'avg_edge': g['edge'].mean(),
        })).round(3).sort_values('roi', ascending=False)
        print(by_level[['n_bets','win_rate','roi','pnl','avg_odd','avg_edge']].to_string())

    # ── Breakdown ROI par surface (Pinnacle) ─────────────────────────────────
    print("\n" + "=" * 55)
    print("BREAKDOWN ROI — PAR SURFACE (Pinnacle)")
    print("=" * 55)
    if len(ps_hist) > 0 and 'surface' in ps_hist.columns:
        by_surface = ps_hist.groupby('surface').apply(lambda g: pd.Series({
            'n_bets'  : len(g),
            'win_rate': g['won'].mean(),
            'roi'     : g['pnl'].sum() / g['stake'].sum(),
            'pnl'     : g['pnl'].sum(),
            'avg_odd' : g['odd'].mean(),
        })).round(3).sort_values('roi', ascending=False)
        print(by_surface[['n_bets','win_rate','roi','pnl','avg_odd']].to_string())

    # Sauvegarde
    for label, hist in all_results.items():
        if len(hist) > 0:
            hist.to_parquet(MODELS_DIR / f"backtest_real_{label}.parquet", index=False)

    print(f"\nBacktest réel terminé.")
