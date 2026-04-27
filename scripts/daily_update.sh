#!/bin/bash
# daily_update.sh - mise a jour quotidienne des profils joueurs ATP/WTA
# Lance par cron a 5h30 UTC chaque jour
set -euo pipefail

APP=/app/tennisml
LOG=/var/log/tennisml_daily.log

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S UTC')] $1" | tee -a "$LOG"; }

log "=== DAILY UPDATE START ==="

cd "$APP"
source venv/bin/activate

log "--- Resolve signals ATP ---"
if python3 src/resolve_signals.py --tour atp --days 3 >> "$LOG" 2>&1; then
    log "Signals ATP resolved"
else
    log "Signals ATP resolve FAILED (exit $?)"
fi

log "--- Resolve signals WTA ---"
if python3 src/resolve_signals.py --tour wta --days 3 >> "$LOG" 2>&1; then
    log "Signals WTA resolved"
else
    log "Signals WTA resolve FAILED (exit $?)"
fi

log "--- ATP update ---"
if python3 src/update_database.py --tour atp >> "$LOG" 2>&1; then
    log "ATP OK"
else
    log "ATP FAILED (exit $?)"
fi

log "--- WTA update ---"
if python3 src/update_database.py --tour wta >> "$LOG" 2>&1; then
    log "WTA OK"
else
    log "WTA FAILED (exit $?)"
fi

log "--- Restart webapp ---"
if systemctl restart tennisml; then
    log "Webapp restarted"
else
    log "Restart FAILED"
fi

log "=== DAILY UPDATE END ==="
