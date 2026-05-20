"""
analyze_profitable_clusters.py
-------------------------------
Error analysis : trouver des sous-segments rentables dans les 1208 paris filtres
du backtest 2025-2026 (ATP) / 2024-2026 (WTA).

APPROCHE
- Hypothesis-driven (anti curve-fitting) : chaque segment est ancre dans une
  logique metier (fatigue, rebond ELO, serveur dominant, etc.)
- Stats rigoureuses :
  * Wilson CI sur win rate
  * Bootstrap CI sur ROI (2000 iter)
  * Test binomial WR vs prob implicite marche
  * Correction Bonferroni sur N hypotheses
- Min 50 paris par cluster pour figurer dans le report
- Classement par ROI desc, marquage "VALIDATED" (N>=100, ROI>=5%, lower CI>0, p<corr)
  et "candidate" (N>=50, ROI>=5%, lower CI>-2%)

USAGE
  python src/analyze_profitable_clusters.py --tour atp
  python src/analyze_profitable_clusters.py --tour wta
  python src/analyze_profitable_clusters.py --tour both --csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import get_paths

MIN_N_REPORT = 50          # min paris pour figurer dans la sortie
MIN_N_VALIDATED = 100      # min paris pour etre "validated"
ROI_THRESHOLD = 0.05       # ROI cible pour "pocket"


# =============================================================================
# STATS HELPERS
# =============================================================================

def wilson_ci(wins: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    z = stats.norm.ppf(1 - alpha / 2)
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_roi_ci(pnl: np.ndarray, stake: np.ndarray,
                     n_boot: int = 2000, alpha: float = 0.05,
                     rng: np.random.Generator | None = None) -> tuple[float, float]:
    if len(pnl) == 0:
        return 0.0, 0.0
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(pnl)
    rois = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s_sum = stake[idx].sum()
        rois[i] = pnl[idx].sum() / s_sum if s_sum > 0 else 0.0
    return float(np.percentile(rois, 100 * alpha / 2)), float(np.percentile(rois, 100 * (1 - alpha / 2)))


def segment_stats(df: pd.DataFrame, name: str, rationale: str,
                  alpha_corrected: float) -> dict | None:
    n = len(df)
    if n < MIN_N_REPORT:
        return None

    wins = int(df['won'].sum())
    wr = wins / n
    wr_lo, wr_hi = wilson_ci(wins, n)

    pnl_arr = df['pnl'].values.astype(float)
    stake_arr = df['stake'].values.astype(float)
    pnl = pnl_arr.sum()
    stake = stake_arr.sum()
    roi = pnl / stake if stake > 0 else 0.0
    roi_lo, roi_hi = bootstrap_roi_ci(pnl_arr, stake_arr)

    bk_imp_mean = float(df['bk_imp_prob'].mean())
    if 0 < bk_imp_mean < 1:
        try:
            p_value = float(stats.binomtest(wins, n, bk_imp_mean, alternative='greater').pvalue)
        except Exception:
            p_value = 1.0
    else:
        p_value = 1.0

    pocket_validated = (n >= MIN_N_VALIDATED and roi >= ROI_THRESHOLD
                        and roi_lo > 0 and p_value < alpha_corrected)
    pocket_candidate = (n >= MIN_N_REPORT and roi >= ROI_THRESHOLD and roi_lo > -0.02)

    return {
        'segment': name,
        'rationale': rationale,
        'n_bets': n,
        'win_rate': wr,
        'wr_ci_lo': wr_lo, 'wr_ci_hi': wr_hi,
        'bk_imp_mean': bk_imp_mean,
        'wr_vs_market': wr - bk_imp_mean,
        'edge_mean': float(df['edge'].mean()),
        'odd_mean': float(df['odd'].mean()),
        'roi': roi,
        'roi_ci_lo': roi_lo, 'roi_ci_hi': roi_hi,
        'pnl': pnl,
        'p_value': p_value,
        'pocket_validated': pocket_validated,
        'pocket_candidate': pocket_candidate,
    }


# =============================================================================
# CHARGEMENT — backtest filtre + jointure features
# =============================================================================

def load_bets_with_features(tour: str) -> pd.DataFrame:
    paths = get_paths(tour)
    bets = pd.read_parquet(paths['models_dir'] / 'backtest_real_Pinnacle.parquet')
    bets['date'] = pd.to_datetime(bets['date'])

    feats = pd.read_parquet(paths['processed_dir'] / 'matches_features_final.parquet')
    feats['tourney_date'] = pd.to_datetime(feats['tourney_date'])

    # Limiter aux annees du backtest pour vitesse
    min_y, max_y = bets['date'].dt.year.min(), bets['date'].dt.year.max()
    feats = feats[feats['year'].between(min_y, max_y)]

    feat_cols = [c for c in [
        'tourney_date', 'p1_name', 'p2_name',
        'elo_diff', 'elo_surface_diff', 'elo_win_prob_p1',
        'rank_diff',
        'p1_winrate_5', 'p2_winrate_5',
        'p1_winrate_10', 'p2_winrate_10',
        'p1_streak', 'p2_streak',
        'p1_days_since', 'p2_days_since',
        'p1_matches_7d', 'p2_matches_7d',
        'p1_sets_7d', 'p2_sets_7d',
        'p1_winrate_surf_Hard', 'p2_winrate_surf_Hard',
        'p1_winrate_surf_Clay', 'p2_winrate_surf_Clay',
        'p1_winrate_surf_Grass', 'p2_winrate_surf_Grass',
        'h2h_total', 'h2h_p1_winrate',
        'p1_tourney_winrate', 'p2_tourney_winrate',
        'p1_1stIn_pct_roll10', 'p2_1stIn_pct_roll10',
        'p1_1stWon_pct_roll10', 'p2_1stWon_pct_roll10',
        'p1_sets_ratio_10', 'p2_sets_ratio_10',
        'is_best_of_5', 'round_importance',
    ] if c in feats.columns]

    # Cle de jointure : (lowercase names, semaine YYYY-WW) pour tolerance dates
    feats['_k'] = (feats['p1_name'].str.lower() + '|' + feats['p2_name'].str.lower() +
                   '|' + feats['tourney_date'].dt.strftime('%Y-%W'))
    bets['_k'] = (bets['p1_name'].str.lower() + '|' + bets['p2_name'].str.lower() +
                  '|' + bets['date'].dt.strftime('%Y-%W'))
    feats_sub = feats[feat_cols + ['_k']].drop_duplicates(subset='_k')

    merged = bets.merge(feats_sub, on='_k', how='left', suffixes=('', '_f'))
    n_with_feat = merged['elo_diff'].notna().sum() if 'elo_diff' in merged.columns else 0
    print(f"  {tour.upper()}: {len(merged):,} paris, {n_with_feat:,} avec features ({n_with_feat/max(len(merged),1):.0%})")

    # Features "side-relative" (du POV du cote sur lequel on parie)
    is_p1 = (merged['bet_on'] == 'p1').values
    def sided(p1col, p2col):
        if p1col not in merged.columns or p2col not in merged.columns:
            return None
        return np.where(is_p1, merged[p1col].values, merged[p2col].values)

    def oppos(p1col, p2col):
        if p1col not in merged.columns or p2col not in merged.columns:
            return None
        return np.where(is_p1, merged[p2col].values, merged[p1col].values)

    for label, p1c, p2c in [
        ('side_streak', 'p1_streak', 'p2_streak'),
        ('opp_streak', 'p1_streak', 'p2_streak'),
        ('side_days', 'p1_days_since', 'p2_days_since'),
        ('opp_days', 'p1_days_since', 'p2_days_since'),
        ('side_sets7', 'p1_sets_7d', 'p2_sets_7d'),
        ('opp_sets7', 'p1_sets_7d', 'p2_sets_7d'),
        ('side_wr5', 'p1_winrate_5', 'p2_winrate_5'),
        ('opp_wr5', 'p1_winrate_5', 'p2_winrate_5'),
        ('side_1stIn', 'p1_1stIn_pct_roll10', 'p2_1stIn_pct_roll10'),
        ('opp_1stIn', 'p1_1stIn_pct_roll10', 'p2_1stIn_pct_roll10'),
        ('side_1stWon', 'p1_1stWon_pct_roll10', 'p2_1stWon_pct_roll10'),
        ('opp_1stWon', 'p1_1stWon_pct_roll10', 'p2_1stWon_pct_roll10'),
        ('side_tw', 'p1_tourney_winrate', 'p2_tourney_winrate'),
        ('opp_tw', 'p1_tourney_winrate', 'p2_tourney_winrate'),
        ('side_setratio', 'p1_sets_ratio_10', 'p2_sets_ratio_10'),
        ('opp_setratio', 'p1_sets_ratio_10', 'p2_sets_ratio_10'),
    ]:
        if label.startswith('side_'):
            v = sided(p1c, p2c)
        else:
            v = oppos(p1c, p2c)
        if v is not None:
            merged[label] = v

    # ELO du cote relativement a l'adversaire (signe ajuste)
    if 'elo_diff' in merged.columns:
        merged['side_elo_advantage'] = np.where(is_p1, merged['elo_diff'].values,
                                                -merged['elo_diff'].values)

    # H2H winrate du cote
    if 'h2h_p1_winrate' in merged.columns:
        merged['side_h2h_wr'] = np.where(is_p1, merged['h2h_p1_winrate'].values,
                                          1 - merged['h2h_p1_winrate'].values)

    return merged


# =============================================================================
# HYPOTHESES METIER — chacune avec son rationale
# =============================================================================

def build_hypotheses(df: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
    out: list[tuple[str, str, pd.Series]] = []

    # === H1. Fatigue asymetrique (sets joues sur 7j) ===
    if 'side_sets7' in df.columns and 'opp_sets7' in df.columns:
        gap_sets = pd.Series(df['opp_sets7'].fillna(0).values - df['side_sets7'].fillna(0).values,
                             index=df.index)
        out.append((
            "H1a_fatigue_adv_sets_diff>=6",
            "Notre joueur a joue 6+ sets de moins que l'adv sur 7j",
            gap_sets >= 6,
        ))
        out.append((
            "H1b_fatigue_disadv_sets_diff<=-6",
            "Notre joueur a 6+ sets de PLUS — desavantage fatigue",
            gap_sets <= -6,
        ))

    # === H1bis. Asymetrie days_since ===
    if 'side_days' in df.columns and 'opp_days' in df.columns:
        out.append((
            "H1c_freshness_active_vs_rusty",
            "Notre joueur <=4j depuis dernier match, adv >=14j de repos",
            (df['side_days'] <= 4) & (df['opp_days'] >= 14),
        ))
        out.append((
            "H1d_rust_inverse",
            "Notre joueur >=14j de repos, adv <=4j actif",
            (df['side_days'] >= 14) & (df['opp_days'] <= 4),
        ))

    # === H2. Syndrome du rebond — gros ELO en sortie de defaites ===
    if 'side_elo_advantage' in df.columns and 'side_streak' in df.columns and 'opp_streak' in df.columns:
        out.append((
            "H2a_rebond_elo+_streak-",
            "Notre joueur +100 ELO, sort de >=2 defaites, adv en hot streak >=2",
            (df['side_elo_advantage'] >= 100) & (df['side_streak'] <= -2) & (df['opp_streak'] >= 2),
        ))
        out.append((
            "H2b_rebond_elo+_streak-_soft",
            "Notre joueur +50 ELO, sort de >=1 defaite",
            (df['side_elo_advantage'] >= 50) & (df['side_streak'] <= -1),
        ))

    # === H3. Avantage au service — 1stIn% ===
    if 'side_1stIn' in df.columns:
        gap_1stIn = pd.Series(df['side_1stIn'].fillna(0).values - df['opp_1stIn'].fillna(0).values,
                              index=df.index)
        out.append((
            "H3a_server_dom_1stIn_+5pp",
            "Notre joueur +5pp de 1stIn% sur l'adv",
            gap_1stIn >= 0.05,
        ))
        if 'side_1stWon' in df.columns and 'opp_1stWon' in df.columns:
            gap_1stWon = pd.Series(df['side_1stWon'].fillna(0).values - df['opp_1stWon'].fillna(0).values,
                                   index=df.index)
            out.append((
                "H3b_server_dom_combo",
                "Avantage combine : +3pp 1stIn ET +3pp 1stWon sur l'adv",
                (gap_1stIn >= 0.03) & (gap_1stWon >= 0.03),
            ))

    # === H4. Familiarite tournoi (post leak-fix : tourney_winrate sur tour en cours legitime) ===
    if 'side_tw' in df.columns and 'opp_tw' in df.columns:
        out.append((
            "H4_familiarite_tournoi",
            "Notre joueur winrate=1.0 sur >=1 match prior du tournoi, adv sans match prior",
            (df['side_tw'] == 1.0) & df['opp_tw'].isna(),
        ))

    # === H5. H2H ===
    if 'side_h2h_wr' in df.columns and 'h2h_total' in df.columns:
        out.append((
            "H5_h2h_domination",
            "Au moins 4 matchs H2H et notre joueur gagne >=75% historiquement",
            (df['h2h_total'].fillna(0) >= 4) & (df['side_h2h_wr'] >= 0.75),
        ))

    # === H6. Specialiste surface ===
    is_p1 = (df['bet_on'] == 'p1').values
    for s in ['Hard', 'Clay', 'Grass']:
        p1c, p2c = f'p1_winrate_surf_{s}', f'p2_winrate_surf_{s}'
        if p1c not in df.columns:
            continue
        side_wr = np.where(is_p1, df[p1c].values, df[p2c].values)
        opp_wr = np.where(is_p1, df[p2c].values, df[p1c].values)
        out.append((
            f"H6_{s.lower()}_specialist",
            f"Surface {s} : notre joueur winrate_surf>=70%, adv<=50%",
            (df['surface'] == s) & (pd.Series(side_wr, index=df.index) >= 0.70) &
            (pd.Series(opp_wr, index=df.index) <= 0.50),
        ))

    # === H7. Hot streak ===
    if 'side_streak' in df.columns:
        out.append((
            "H7_side_hot_streak>=5",
            "Notre joueur sur >=5 victoires consecutives",
            df['side_streak'] >= 5,
        ))

    # === H8. Niveau tournoi ===
    out.append(("H8a_grand_chelem", "Grand Chelem uniquement", df['level'] == 'G'))
    out.append(("H8b_masters_1000", "Masters 1000 uniquement", df['level'] == 'M'))
    out.append(("H8c_atp_wta_250_500", "ATP 250/500 (level A)", df['level'] == 'A'))

    # === H9. Best-of-5 (ATP GS) ===
    if 'is_best_of_5' in df.columns:
        out.append(("H9_best_of_5", "Matches Best-of-5 (Grand Slam ATP)", df['is_best_of_5'] == 1.0))

    # === H10. Cotes / niches edge ===
    out.append(("H10a_cote_close_1.5_1.8", "Cote dans [1.5, 1.8] (favoris solides)",
                df['odd'].between(1.5, 1.8)))
    out.append(("H10b_cote_2.0+", "Cote >= 2.0 (legers outsiders)", df['odd'] >= 2.0))
    out.append(("H10c_edge_>=10pp", "Edge >= 10pp", df['edge'] >= 0.10))
    out.append(("H10d_high_conf_low_odd", "our_prob>=0.65 ET cote<=1.8",
                (df['our_prob'] >= 0.65) & (df['odd'] <= 1.8)))

    # === H11. CLV positif ===
    if 'clv' in df.columns:
        out.append(("H11a_clv_positive", "CLV > 0 (notre prob > prob marche)", df['clv'] > 0))
        out.append(("H11b_clv_>=10pct", "CLV >= +10%", df['clv'] >= 0.10))

    # === H12. Round (precoce vs tardif) ===
    if 'round_importance' in df.columns:
        out.append(("H12a_early_rounds", "Rounds precoces (round_importance<=0.4)",
                    df['round_importance'] <= 0.4))
        out.append(("H12b_late_rounds", "Rounds tardifs QF/SF/F (round_importance>=0.6)",
                    df['round_importance'] >= 0.6))

    # === H13. Hot streak ET avantage ELO ===
    if 'side_streak' in df.columns and 'side_elo_advantage' in df.columns:
        out.append((
            "H13_elo+_hot_streak_combo",
            "Notre joueur +50 ELO ET hot streak >=3",
            (df['side_elo_advantage'] >= 50) & (df['side_streak'] >= 3),
        ))

    return out


# =============================================================================
# MAIN
# =============================================================================

def analyse_tour(tour: str) -> pd.DataFrame:
    print(f"\n{'='*70}\nANALYSE POCHETS RENTABILITE — {tour.upper()}\n{'='*70}")

    df = load_bets_with_features(tour)
    print(f"  {len(df):,} paris filtres (test 2024-2026)")

    hypotheses = build_hypotheses(df)
    n_tests = len(hypotheses)
    alpha = 0.05
    alpha_corrected = alpha / max(n_tests, 1)
    print(f"  {n_tests} hypotheses testees -> seuil Bonferroni: p<{alpha_corrected:.4f}\n")

    # Baseline = tous les paris
    base = segment_stats(df, "BASELINE_all_bets", "Reference (tous les paris filtres)",
                         alpha_corrected)
    results = [base] if base else []

    for name, rationale, mask in hypotheses:
        sub = df[mask.fillna(False)]
        res = segment_stats(sub, name, rationale, alpha_corrected)
        if res:
            results.append(res)

    df_res = pd.DataFrame(results).sort_values('roi', ascending=False)

    # Affichage
    print("="*120)
    print(f"SEGMENTS — tri par ROI desc (N>={MIN_N_REPORT})")
    print("="*120)
    for _, r in df_res.iterrows():
        flag = " [VALIDATED]" if r['pocket_validated'] else (" [candidate]" if r['pocket_candidate'] else "")
        print(f"\n{r['segment']}{flag}")
        print(f"  > {r['rationale']}")
        print(f"  N={int(r['n_bets']):>4d} | WR={r['win_rate']:.1%} CI[{r['wr_ci_lo']:.1%},{r['wr_ci_hi']:.1%}] "
              f"vs marche {r['bk_imp_mean']:.1%} (Δ={r['wr_vs_market']:+.1%}) | "
              f"ROI={r['roi']:+.1%} CI[{r['roi_ci_lo']:+.1%},{r['roi_ci_hi']:+.1%}] | "
              f"edge={r['edge_mean']:+.1%} | cote={r['odd_mean']:.2f} | p={r['p_value']:.4f}")

    # Synthese
    n_validated = int(df_res['pocket_validated'].sum())
    n_candidate = int(df_res['pocket_candidate'].sum() - n_validated)
    print(f"\n{'='*70}")
    print(f"SYNTHESE {tour.upper()} : {n_validated} VALIDATED + {n_candidate} candidates")
    print('='*70)
    if n_validated:
        v = df_res[df_res['pocket_validated']]
        print("\nVALIDATED (N>=100, ROI>=5%, ROI_lo>0, p<Bonferroni) :")
        print(v[['segment','n_bets','win_rate','roi','roi_ci_lo','roi_ci_hi','p_value']].to_string(index=False))
    if n_candidate:
        c = df_res[df_res['pocket_candidate'] & ~df_res['pocket_validated']]
        print("\nCANDIDATES (N>=50, ROI>=5%, ROI_lo>-2%, besoin de plus de data) :")
        print(c[['segment','n_bets','win_rate','roi','roi_ci_lo','roi_ci_hi','p_value']].to_string(index=False))
    if not n_validated and not n_candidate:
        print("\nAucun cluster ne satisfait les criteres apres correction Bonferroni.")

    df_res['tour'] = tour
    return df_res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--tour', choices=['atp','wta','both'], default='both')
    parser.add_argument('--csv', action='store_true')
    args = parser.parse_args()

    if args.tour == 'both':
        df_atp = analyse_tour('atp')
        df_wta = analyse_tour('wta')
        if args.csv:
            from config import get_paths
            for d, t in [(df_atp,'atp'),(df_wta,'wta')]:
                p = get_paths(t)['models_dir'] / 'profitable_clusters_report.csv'
                d.to_csv(p, index=False)
                print(f"  CSV: {p}")
    else:
        d = analyse_tour(args.tour)
        if args.csv:
            p = get_paths(args.tour)['models_dir'] / 'profitable_clusters_report.csv'
            d.to_csv(p, index=False)
            print(f"  CSV: {p}")


if __name__ == "__main__":
    main()
