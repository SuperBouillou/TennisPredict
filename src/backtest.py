# src/backtest.py

import argparse
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def load_test_data(models_dir: Path, processed_dir: Path) -> pd.DataFrame:
    splits   = joblib.load(models_dir / "splits.pkl")
    features = joblib.load(models_dir / "feature_list.pkl")
    imputer  = joblib.load(models_dir / "imputer.pkl")
    model    = joblib.load(models_dir / "xgb_tuned.pkl")

    platt_path = models_dir / "platt_scaler.pkl"
    platt = joblib.load(platt_path) if platt_path.exists() else None

    X_test = splits['X_test']
    y_test = splits['y_test']
    meta   = splits['meta_test']

    X_imp      = imputer.transform(X_test)
    raw_prob   = model.predict_proba(X_imp)[:, 1]
    p1_prob    = platt.predict_proba(raw_prob.reshape(-1, 1))[:, 1] if platt else raw_prob

    df = meta.copy().reset_index(drop=True)
    df['p1_prob']   = p1_prob
    df['p2_prob']   = 1 - p1_prob
    df['target']    = y_test.values
    df['predicted'] = (p1_prob >= 0.5).astype(int)
    df['correct']   = (df['predicted'] == df['target']).astype(int)

    calib = "Platt" if platt else "brut"
    print(f"✅ Test chargé : {len(df):,} matchs | "
          f"Accuracy : {df['correct'].mean():.1%} | Calibration : {calib}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION COTES BOOKMAKER (basée sur ELO — plus réaliste que les rangs)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_bookmaker_odds(df: pd.DataFrame, processed_dir: Path) -> pd.DataFrame:
    """
    Simule des cotes bookmaker basées sur l'ELO.
    Pour un backtest réel → utiliser backtest_real.py avec tennis-data.co.uk.
    """
    MARGIN = 0.06

    df_feat = pd.read_parquet(processed_dir / "matches_features_final.parquet")
    df_feat = df_feat[df_feat['year'] >= 2023][
        ['tourney_date', 'p1_id', 'p2_id', 'elo_win_prob_p1']
    ].copy()

    df = df.merge(df_feat, on=['tourney_date', 'p1_id', 'p2_id'], how='left')

    mask_missing = df['elo_win_prob_p1'].isna()
    if mask_missing.sum() > 0:
        r1 = df.loc[mask_missing, 'p1_rank'].fillna(200)
        r2 = df.loc[mask_missing, 'p2_rank'].fillna(200)
        df.loc[mask_missing, 'elo_win_prob_p1'] = (1/r1) / (1/r1 + 1/r2)

    true_p1 = df['elo_win_prob_p1'].clip(0.01, 0.99)
    true_p2 = 1 - true_p1

    bk_p1 = true_p1 * (1 + MARGIN)
    bk_p2 = true_p2 * (1 + MARGIN)

    df['bk_odd_p1']  = (1 / bk_p1).round(2)
    df['bk_odd_p2']  = (1 / bk_p2).round(2)
    df['bk_prob_p1'] = true_p1
    df['bk_prob_p2'] = true_p2

    print(f"  Cotes simulées (ELO + marge {MARGIN:.0%})")
    print(f"  Cote p1 moyenne : {df['bk_odd_p1'].mean():.2f}")
    print(f"  Cote p2 moyenne : {df['bk_odd_p2'].mean():.2f}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# DÉTECTION DE VALUE
# ─────────────────────────────────────────────────────────────────────────────

def compute_value_bets(df: pd.DataFrame,
                       min_edge: float = 0.05,
                       min_prob: float = 0.55) -> pd.DataFrame:
    df['edge_p1'] = df['p1_prob'] - df['bk_prob_p1']
    df['edge_p2'] = df['p2_prob'] - df['bk_prob_p2']

    df['ev_p1'] = df['p1_prob'] * df['bk_odd_p1'] - 1
    df['ev_p2'] = df['p2_prob'] * df['bk_odd_p2'] - 1

    df['bet_p1'] = ((df['edge_p1'] >= min_edge) & (df['p1_prob'] >= min_prob)).astype(int)
    df['bet_p2'] = ((df['edge_p2'] >= min_edge) & (df['p2_prob'] >= min_prob)).astype(int)

    mask_both = (df['bet_p1'] == 1) & (df['bet_p2'] == 1)
    df.loc[mask_both & (df['edge_p1'] >= df['edge_p2']), 'bet_p2'] = 0
    df.loc[mask_both & (df['edge_p2'] >  df['edge_p1']), 'bet_p1'] = 0

    n_bets = df['bet_p1'].sum() + df['bet_p2'].sum()
    print(f"  Value bets détectés : {n_bets:,} / {len(df):,} matchs "
          f"({n_bets/len(df):.1%})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STRATÉGIES DE MISE
# ─────────────────────────────────────────────────────────────────────────────

def kelly_fraction(prob: float, odd: float, fraction: float = 0.25) -> float:
    b = odd - 1
    if b <= 0:
        return 0
    kelly = (b * prob - (1 - prob)) / b
    return max(0, kelly * fraction)


def simulate_betting(df: pd.DataFrame,
                     bankroll_init: float = 1000.0,
                     strategy: str = 'flat',
                     flat_stake: float = 10.0,
                     kelly_frac: float = 0.25) -> pd.DataFrame:
    df = df.sort_values('tourney_date').reset_index(drop=True)
    bankroll = bankroll_init
    history  = []

    for _, row in df.iterrows():
        for side in ['p1', 'p2']:
            if row[f'bet_{side}'] != 1:
                continue

            prob = row[f'{side}_prob']
            odd  = row[f'bk_odd_{side}']
            won  = (row['target'] == 1) if side == 'p1' else (row['target'] == 0)

            if strategy == 'flat':
                stake = flat_stake
            elif strategy == 'kelly':
                stake = bankroll * kelly_fraction(prob, odd, kelly_frac)
            elif strategy == 'percent':
                stake = bankroll * 0.02
            else:
                stake = flat_stake

            stake    = min(stake, bankroll)
            pnl      = stake * (odd - 1) if won else -stake
            bankroll = bankroll + pnl

            history.append({
                'date'    : row['tourney_date'],
                'tourney' : row['tourney_name'],
                'surface' : row['surface'],
                'level'   : row['tourney_level'],
                'p1_name' : row['p1_name'],
                'p2_name' : row['p2_name'],
                'bet_on'  : side,
                'prob'    : prob,
                'odd'     : odd,
                'edge'    : row[f'edge_{side}'],
                'ev'      : row[f'ev_{side}'],
                'stake'   : stake,
                'won'     : int(won),
                'pnl'     : pnl,
                'bankroll': bankroll,
            })

    df_hist = pd.DataFrame(history)
    if len(df_hist) == 0:
        print("  ⚠️  Aucun pari simulé")
    return df_hist


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRIQUES BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

def backtest_metrics(df_hist: pd.DataFrame,
                     bankroll_init: float,
                     strategy_name: str) -> dict:
    if len(df_hist) == 0:
        return {}

    total_bets   = len(df_hist)
    wins         = df_hist['won'].sum()
    win_rate     = wins / total_bets
    total_pnl    = df_hist['pnl'].sum()
    roi          = total_pnl / df_hist['stake'].sum()
    final_bk     = df_hist['bankroll'].iloc[-1]
    total_return = (final_bk - bankroll_init) / bankroll_init

    running_max  = df_hist['bankroll'].cummax()
    drawdown     = (df_hist['bankroll'] - running_max) / running_max
    max_drawdown = drawdown.min()

    pnl_series = df_hist['pnl']
    sharpe     = (pnl_series.mean() / pnl_series.std() * np.sqrt(252)
                  if pnl_series.std() > 0 else 0)

    metrics = {
        'strategy'      : strategy_name,
        'total_bets'    : total_bets,
        'win_rate'      : win_rate,
        'total_pnl'     : total_pnl,
        'roi'           : roi,
        'final_bankroll': final_bk,
        'total_return'  : total_return,
        'max_drawdown'  : max_drawdown,
        'sharpe'        : sharpe,
        'avg_edge'      : df_hist['edge'].mean(),
        'avg_odd'       : df_hist['odd'].mean(),
    }

    print(f"\n  [{strategy_name}]")
    print(f"    Nombre de paris    : {total_bets:,}")
    print(f"    Win rate           : {win_rate:.1%}")
    print(f"    ROI                : {roi:+.1%}")
    print(f"    P&L total          : {total_pnl:+.1f}€")
    print(f"    Bankroll finale    : {final_bk:.1f}€")
    print(f"    Rendement total    : {total_return:+.1%}")
    print(f"    Max drawdown       : {max_drawdown:.1%}")
    print(f"    Sharpe             : {sharpe:.2f}")
    print(f"    Edge moyen         : {metrics['avg_edge']:+.1%}")
    print(f"    Cote moyenne       : {metrics['avg_odd']:.2f}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_backtest(strategies: dict, bankroll_init: float, models_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Backtest — Analyse complète', fontsize=14)
    colors = ['steelblue', 'green', 'orange']

    ax = axes[0, 0]
    for (name, df_h), color in zip(strategies.items(), colors):
        if len(df_h) > 0:
            ax.plot(df_h['date'], df_h['bankroll'], label=name, color=color)
    ax.axhline(y=bankroll_init, color='black', linestyle='--', alpha=0.5)
    ax.set_title('Évolution du bankroll')
    ax.set_ylabel('Bankroll (€)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    first_hist = list(strategies.values())[0]
    if len(first_hist) > 0:
        ax.hist(first_hist['edge'], bins=30, color='steelblue', alpha=0.7)
        ax.axvline(x=first_hist['edge'].mean(), color='red', linestyle='--',
                   label=f"Edge moyen: {first_hist['edge'].mean():.1%}")
        ax.set_title('Distribution des edges')
        ax.set_xlabel('Edge (notre prob - prob bookmaker)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if len(first_hist) > 0:
        for surface, color in [('Hard','steelblue'),('Clay','orange'),('Grass','green')]:
            surf_data = first_hist[first_hist['surface'] == surface]
            if len(surf_data) > 0:
                pnl_cum = surf_data['pnl'].cumsum()
                ax.plot(range(len(pnl_cum)), pnl_cum,
                        label=f"{surface} ({len(surf_data)} paris)", color=color)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_title('P&L cumulatif par surface (Flat)')
        ax.set_xlabel('Nombre de paris')
        ax.set_ylabel('P&L (€)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if len(first_hist) > 0:
        first_hist['odd_bin'] = pd.cut(first_hist['odd'],
                                       bins=[1, 1.5, 2, 2.5, 3, 5, 100],
                                       labels=['1-1.5','1.5-2','2-2.5','2.5-3','3-5','5+'])
        wr_by_odd = first_hist.groupby('odd_bin', observed=True)['won'].mean()
        wr_by_odd.plot(kind='bar', ax=ax, color='steelblue', alpha=0.7)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        ax.set_title('Win rate par tranche de cote')
        ax.set_ylabel('Win rate')
        ax.set_xlabel('Cote bookmaker')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = models_dir / "backtest_results.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n📊 Graphique sauvegardé : {out}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest simulé (cotes ELO)")
    parser.add_argument('--tour', choices=['atp', 'wta'], default='atp',
                        help="Circuit : atp ou wta (défaut: atp)")
    parser.add_argument('--bankroll', type=float, default=1000.0)
    args = parser.parse_args()

    MODELS_DIR    = ROOT / "data" / "models"    / args.tour
    PROCESSED_DIR = ROOT / "data" / "processed" / args.tour

    BANKROLL = args.bankroll

    print("=" * 55)
    print(f"BACKTEST {args.tour.upper()} — Cotes simulées (ELO)")
    print("=" * 55)

    df = load_test_data(MODELS_DIR, PROCESSED_DIR)

    print("\n── Simulation cotes bookmaker ───────────────────────")
    df = simulate_bookmaker_odds(df, PROCESSED_DIR)

    print("\n── Détection value bets ─────────────────────────────")
    df = compute_value_bets(df, min_edge=0.03, min_prob=0.55)

    print("\n── Simulation des stratégies ────────────────────────")
    strategies  = {}
    all_metrics = []

    for strategy in ['flat', 'kelly', 'percent']:
        print(f"\n  Stratégie : {strategy}")
        hist = simulate_betting(df, bankroll_init=BANKROLL, strategy=strategy,
                                flat_stake=10.0, kelly_frac=0.25)
        strategies[strategy] = hist
        metrics = backtest_metrics(hist, BANKROLL, strategy)
        all_metrics.append(metrics)

    print("\n" + "=" * 55)
    print("TABLEAU COMPARATIF STRATÉGIES")
    print("=" * 55)
    df_metrics = pd.DataFrame(all_metrics)
    print(df_metrics[['strategy','total_bets','win_rate',
                       'roi','total_return','max_drawdown','sharpe']]
          .to_string(index=False))

    print("\n── Top 10 meilleurs paris ───────────────────────────")
    flat_hist = strategies['flat']
    if len(flat_hist) > 0:
        top10 = (flat_hist.sort_values('pnl', ascending=False)
                 .head(10)[['date','p1_name','p2_name','bet_on',
                             'prob','odd','edge','pnl','won']])
        print(top10.to_string(index=False))

    plot_backtest(strategies, BANKROLL, MODELS_DIR)

    for name, hist in strategies.items():
        if len(hist) > 0:
            hist.to_parquet(MODELS_DIR / f"backtest_{name}.parquet", index=False)

    print(f"\n✅ Backtest {args.tour.upper()} terminé.")
