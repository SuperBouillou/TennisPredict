# src/scheduled_sync.py
"""
Wrapper pour le planificateur Windows.
Lance fetch_live_data.py pour ATP (et WTA si disponible) chaque matin.
Log les resultats dans logs/sync.log.

Enregistrement de la tache planifiee (a lancer une seule fois en admin) :
    python src/scheduled_sync.py --register

Execution manuelle :
    python src/scheduled_sync.py
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR  = ROOT / "logs"
LOG_PATH = LOG_DIR / "sync.log"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

sys.path.insert(0, str(ROOT / "src"))


def run_sync():
    """Lance fetch_live_data pour ATP (+ WTA si parquet present)."""
    from fetch_live_data import run_update

    for tour in ["atp", "wta"]:
        try:
            logging.info(f"Demarrage sync {tour.upper()}")
            run_update(tour=tour, force=False)
            logging.info(f"Sync {tour.upper()} OK")
        except Exception as e:
            logging.error(f"Sync {tour.upper()} ECHEC: {e}", exc_info=True)
            print(f"[ERREUR] {tour.upper()}: {e}")


def register_task():
    """Enregistre la tache planifiee Windows (necessite droits admin)."""
    import subprocess

    python_exe = sys.executable
    script     = str(ROOT / "src" / "scheduled_sync.py")

    # Tache quotidienne a 07h00 et 19h00
    tasks = [
        ("TennisML_Sync_Morning", "07:00"),
        ("TennisML_Sync_Evening", "19:00"),
    ]

    for task_name, start_time in tasks:
        cmd = [
            "schtasks", "/Create",
            "/TN", task_name,
            "/TR", f'"{python_exe}" "{script}"',
            "/SC", "DAILY",
            "/ST", start_time,
            "/F",
            "/RL", "HIGHEST",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  Tache creee : {task_name} a {start_time}")
        else:
            print(f"  Erreur creation {task_name}: {result.stderr.strip()}")
            print(f"  (Lancer en tant qu'administrateur si necessaire)")

    print("\n  Pour verifier :")
    print("    schtasks /Query /TN TennisML_Sync_Morning /FO LIST")
    print("\n  Pour lancer manuellement :")
    print("    schtasks /Run /TN TennisML_Sync_Morning")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tennis ML data sync scheduler")
    parser.add_argument("--register", action="store_true",
                        help="Enregistre les taches planifiees Windows (admin requis)")
    args = parser.parse_args()

    if args.register:
        register_task()
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Demarrage sync...")
        run_sync()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Sync termine. Log: {LOG_PATH}")
