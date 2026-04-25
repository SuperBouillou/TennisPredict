# src/resolve_signals.py
# Résolution automatique des signaux VALUE en attente (signal_log.result='pending')
# Appelé chaque nuit par le cron daily_update.sh après update_database.py
#
# Logique : pour chaque signal pending, cherche dans les résultats ESPN
# des dernières 48h si le match a été joué et qui a gagné.
#
# Usage: python3 src/resolve_signals.py [--tour atp|wta] [--days 2]

import sys, argparse, logging
from datetime import date, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from webapp.db import get_connection, resolve_signals


def main():
    parser = argparse.ArgumentParser(description="Résolution des signaux VALUE en attente")
    parser.add_argument('--tour', default='atp', choices=['atp', 'wta'])
    parser.add_argument('--days', type=int, default=3,
                        help='Nombre de jours passés à scruter pour les résultats (défaut: 3)')
    args = parser.parse_args()

    db_path = ROOT / "data" / "tennis_predict.db"
    conn = get_connection(db_path)

    # Vérifier combien de signaux sont en attente
    pending = conn.execute(
        "SELECT COUNT(*) as n FROM signal_log WHERE tour=? AND result='pending'",
        (args.tour,)
    ).fetchone()
    n_pending = pending['n'] if pending else 0

    if n_pending == 0:
        log.info("Aucun signal %s en attente.", args.tour.upper())
        return

    log.info("%d signaux %s en attente — recherche des résultats...", n_pending, args.tour.upper())

    # Récupérer les résultats ESPN des derniers N jours
    try:
        from espn_client import fetch_results
    except ImportError:
        log.warning("fetch_results non disponible dans espn_client — tentative fetch_scheduled")
        fetch_results = None

    all_results: list[dict] = []

    if fetch_results:
        for delta in range(1, args.days + 1):
            d = date.today() - timedelta(days=delta)
            try:
                results = fetch_results(args.tour, d)
                if results:
                    all_results.extend(results)
                    log.info("  %s : %d résultats", d, len(results))
            except Exception as exc:
                log.warning("  %s : erreur ESPN — %s", d, exc)
    else:
        # Fallback : tenter via fetch_scheduled (retourne parfois les matchs terminés)
        from espn_client import fetch_scheduled
        for delta in range(1, args.days + 1):
            d = date.today() - timedelta(days=delta)
            try:
                matches = fetch_scheduled(args.tour, d)
                # fetch_scheduled peut retourner des matchs avec winner info
                completed = [m for m in matches if m.get('winner') or m.get('status') == 'final']
                all_results.extend(completed)
            except Exception as exc:
                log.warning("  %s : erreur ESPN — %s", d, exc)

    if not all_results:
        log.warning("Aucun résultat ESPN récupéré.")
        return

    log.info("Total résultats récupérés : %d", len(all_results))

    # Résoudre
    resolved = resolve_signals(conn, args.tour, all_results)
    log.info("Signaux résolus : %d / %d en attente", resolved, n_pending)

    conn.close()


if __name__ == "__main__":
    main()
