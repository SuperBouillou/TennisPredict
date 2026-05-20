#!/bin/bash
# weekly_refresh.sh — refresh complet base d'entraînement depuis Sackmann
# Lance par cron lundi 06:30 UTC (apres la maj hebdomadaire Sackmann du dimanche soir)
#
# Effet : regenere matches_features_final.parquet pour ATP + WTA.
# Ne TOUCHE PAS aux modeles (laisse au monthly_retrain.sh).
# Ne redemarre PAS le service (training data only, pas de hot-reload necessaire).
set -euo pipefail

APP=/app/tennisml
LOG=/var/log/tennisml_weekly.log

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S UTC')] $1" | tee -a "$LOG"; }

log "=== WEEKLY REFRESH START ==="

cd "$APP"
source venv/bin/activate

for TOUR in atp wta; do
    log "--- ${TOUR^^} : Sackmann download ---"
    if ! python3 src/download_data.py --tour $TOUR >> "$LOG" 2>&1; then
        log "${TOUR^^} download FAILED — skipping rest of pipeline"
        continue
    fi

    log "--- ${TOUR^^} : load_data ---"
    python3 src/load_data.py --tour $TOUR >> "$LOG" 2>&1 || { log "${TOUR^^} load_data FAILED"; continue; }

    log "--- ${TOUR^^} : restructure_data ---"
    python3 src/restructure_data.py --tour $TOUR >> "$LOG" 2>&1 || { log "${TOUR^^} restructure FAILED"; continue; }

    log "--- ${TOUR^^} : compute_elo ---"
    python3 src/compute_elo.py --tour $TOUR >> "$LOG" 2>&1 || { log "${TOUR^^} compute_elo FAILED"; continue; }

    log "--- ${TOUR^^} : compute_rolling_features ---"
    python3 src/compute_rolling_features.py --tour $TOUR >> "$LOG" 2>&1 || { log "${TOUR^^} rolling FAILED"; continue; }

    log "--- ${TOUR^^} : compute_h2h ---"
    python3 src/compute_h2h.py --tour $TOUR >> "$LOG" 2>&1 || { log "${TOUR^^} h2h FAILED"; continue; }

    log "--- ${TOUR^^} : compute_contextual_features ---"
    python3 src/compute_contextual_features.py --tour $TOUR >> "$LOG" 2>&1 || { log "${TOUR^^} contextual FAILED"; continue; }

    log "--- ${TOUR^^} : compute_glicko ---"
    python3 src/compute_glicko.py --tour $TOUR >> "$LOG" 2>&1 || { log "${TOUR^^} glicko FAILED"; continue; }

    log "--- ${TOUR^^} : add_pinnacle_feature ---"
    python3 src/add_pinnacle_feature.py --tour $TOUR >> "$LOG" 2>&1 || { log "${TOUR^^} pinnacle_feat FAILED"; continue; }

    log "--- ${TOUR^^} : prepare_ml_dataset ---"
    python3 src/prepare_ml_dataset.py --tour $TOUR >> "$LOG" 2>&1 || { log "${TOUR^^} prepare FAILED"; continue; }

    log "${TOUR^^} pipeline OK"
done

log "=== WEEKLY REFRESH END ==="
