# CLAUDE.md

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately — don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons. md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes — don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests — then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items  
2. **Verify Plan**: Check in before starting implementation  
3. **Track Progress**: Mark items complete as you go  
4. **Explain Changes**: High-level summary at each step  
5. **Document Results**: Add review section to `tasks/todo. md`  
6. **Capture Lessons**: Update `tasks/lessons. md` after corrections  

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ATP tennis match outcome prediction system using ML (XGBoost). Predicts match winners and assesses betting value against bookmaker odds.

**Python environment:** `venv/` (activate before running scripts)
```bash
source venv/Scripts/activate   # Windows Git Bash
# or
venv\Scripts\activate          # Windows CMD
```

## Common Commands

```bash
# Full pipeline (run in order):
python src/download_data.py          # Download raw ATP data from Sackmann repo
python src/download_odds.py          # Download odds from tennis-data.co.uk (2010-2026)
python src/load_data.py              # Parse + clean → data/processed/matches_consolidated.parquet
python src/restructure_data.py       # winner/loser → neutral p1/p2 format (eliminates leakage)
python src/compute_elo.py            # ELO ratings → data/processed/matches_with_elo.parquet
python src/compute_rolling_features.py   # Rolling win rates, streaks, fatigue
python src/compute_h2h.py            # Head-to-head stats
python src/compute_contextual_features.py  # Tournament/round importance, surface encoding
python src/prepare_ml_dataset.py     # Final feature selection + temporal train/valid/test split
python src/train_baseline.py         # Logistic regression baseline
python src/train_xgboost.py          # XGBoost default + tuned + Platt calibration
python src/backtest.py               # Simulated backtest (flat/Kelly/% stake)
python src/backtest_real.py          # Real odds backtest (Bet365/Pinnacle/Avg/Max)
python src/update_database.py        # Update player profiles with recent matches (2025-2026)

# Daily prediction
python src/predict_today.py                    # Interactive match entry
python src/predict_today.py --date 2025-06-01  # Specific date
python src/predict_today.py --save             # Save CSV output
python src/predict_today.py --odds             # Enter Pinnacle odds for EV calculation

# Diagnostic / leakage checks
python src/audit_data.py
python src/check_leakage.py
python src/search_leakage.py
python src/check_join_bias.py
python src/check_sources.py
```

## Architecture

### Data Flow

```
Raw CSVs (data/raw/)          ← Jeff Sackmann ATP repo (1968–2025)
    + Odds XLSXs (data/odds/) ← tennis-data.co.uk (2010–2026)
         ↓ load_data.py
matches_consolidated.parquet  (winner/loser format)
         ↓ restructure_data.py
matches_ml_ready.parquet      (neutral p1/p2, target=1 if p1 wins)
         ↓ compute_elo / compute_rolling_features / compute_h2h / compute_contextual_features
matches_features_final.parquet  (~55 MB, all features)
         ↓ prepare_ml_dataset.py
splits.pkl + feature_list.pkl (train ≤2021, valid 2022, test ≥2023)
         ↓ train_baseline / train_xgboost
xgb_default.pkl, xgb_tuned.pkl, imputer.pkl, platt_scaler.pkl
         ↓ predict_today / backtest
Predictions / betting signals
```

### Key Design Decisions

**Structural leakage prevention:** Raw data uses winner/loser columns — knowing who is the "winner" leaks the target. `restructure_data.py` randomly assigns players to `p1`/`p2` roles before feature computation. All subsequent features are computed in this neutral format.

**Temporal split (strict):** Train ≤2021, Valid = 2022, Test ≥2023. Never use random shuffling for time-series match data.

**Two data sources:**
- **Sackmann repo** (GitHub): Historical match results 1968–2025, used for feature computation and training
- **tennis-data.co.uk** (XLSX): Real bookmaker odds (Bet365, Pinnacle, Avg, Max) 2010–2026, used for backtesting and `update_database.py`

**Player database updates:** `update_database.py` converts tennis-data format → Sackmann format, then recomputes ELO and rolling features, outputting `player_profiles_updated.parquet`. `predict_today.py` loads this updated profile first, falling back to `matches_features_final.parquet`.

### Feature Groups (defined in `prepare_ml_dataset.py`)

| Group | Key features |
|---|---|
| `elo` | `elo_diff`, `elo_surface_diff`, `elo_win_prob_p1`, surface specialization |
| `ranking` | `rank_diff`, `rank_ratio`, `rank_points_diff` |
| `forme` | Rolling win rates (5/10/20 matches), streak |
| `surface_forme` | Surface-specific win rates (Hard/Clay/Grass) |
| `h2h` | Overall and surface-specific head-to-head win rate |
| `fatigue` | Matches in last 7/14 days, days since last match |
| `contexte` | Tournament importance, round importance, best-of-5, age diff, surface encoding |
| `stats_service` | 1stIn%, 1stWon%, 2ndWon%, bpSaved%, ace/df ratios (rolling 10) — only post-1991 |

### Model Artifacts (data/models/)

- `splits.pkl` — train/valid/test DataFrames
- `feature_list.pkl` — ordered feature names
- `xgb_default.pkl`, `xgb_tuned.pkl` — raw XGBoost models (need imputer separately)
- `imputer.pkl` — SimpleImputer (constant=0.5) fitted on train set
- `platt_scaler.pkl` — LogisticRegression fitted on valid set probabilities
- `lr_full.pkl`, `lr_elo_only.pkl` — baseline Logistic Regression models

### ELO System

Computed in `compute_elo.py`. K-factor varies by tournament level: Grand Slams (48), Masters/Finals (40), others (32). Both overall ELO and surface-specific ELO (Hard/Clay/Grass/Carpet) are maintained. Uses `match_key` (`tourney_id__winner_id__loser_id`) to avoid cartesian join issues.
