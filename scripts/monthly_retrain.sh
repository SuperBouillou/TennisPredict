#!/bin/bash
# monthly_retrain.sh — retrain XGBoost + recalibrate Platt pour ATP + WTA
# Lance par cron le 1er du mois a 07:00 UTC (apres le weekly_refresh du lundi)
#
# Backup automatique des modeles avant retrain.
# Restart du service en fin de run.
set -euo pipefail

APP=/app/tennisml
LOG=/var/log/tennisml_monthly.log
TS=$(date +%Y%m%d_%H%M%S)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S UTC')] $1" | tee -a "$LOG"; }

log "=== MONTHLY RETRAIN START (tag=$TS) ==="

cd "$APP"
source venv/bin/activate

# Backup
mkdir -p data/models/backup_$TS
for TOUR in atp wta; do
    if [ -d "data/models/$TOUR" ]; then
        cp -r "data/models/$TOUR" "data/models/backup_$TS/${TOUR}_pre_retrain"
        log "Backup ${TOUR^^} -> data/models/backup_$TS/${TOUR}_pre_retrain"
    fi
done

# ATP : Platt calibre contre Pinnacle no-vig (design par defaut)
log "--- ATP : train_xgboost --optuna ---"
if ! python3 src/train_xgboost.py --tour atp --optuna >> "$LOG" 2>&1; then
    log "ATP train FAILED — abandon ATP"
else
    log "--- ATP : recalibrate_platt (Pinnacle) ---"
    python3 src/recalibrate_platt.py --tour atp >> "$LOG" 2>&1 || log "ATP recalibrate FAILED"
fi

# WTA : Platt calibre contre outcomes reels (le modele WTA est meilleur ainsi)
log "--- WTA : train_xgboost --optuna ---"
if ! python3 src/train_xgboost.py --tour wta --optuna >> "$LOG" 2>&1; then
    log "WTA train FAILED — abandon WTA"
else
    log "--- WTA : recalibrate_platt --use-outcomes ---"
    python3 src/recalibrate_platt.py --tour wta --use-outcomes >> "$LOG" 2>&1 || log "WTA recalibrate FAILED"
    # WTA --use-outcomes ne genere PAS les scalers per-surface — supprimer les anciens
    rm -f data/models/wta/platt_Hard.pkl data/models/wta/platt_Clay.pkl data/models/wta/platt_Grass.pkl
    log "WTA per-surface scalers purges (calibration globale outcomes)"
fi

log "--- Restart webapp ---"
if systemctl restart tennisml; then
    log "Webapp restarted"
else
    log "Restart FAILED"
fi

# Cleanup vieux backups (>90j)
find data/models -maxdepth 1 -type d -name 'backup_*' -mtime +90 -exec rm -rf {} \; 2>/dev/null || true

log "=== MONTHLY RETRAIN END ==="
