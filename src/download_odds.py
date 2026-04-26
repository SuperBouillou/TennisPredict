# src/download_odds.py

import argparse
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

# Ajouter src/ au path pour l'import de config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import get_tour_config, get_paths, make_dirs

YEARS = list(range(2010, 2027))


def download_odds_file(year: int, cfg: dict, odds_dir: Path) -> bool:
    """
    Télécharge les cotes pour une année donnée en utilisant la config du tour.
    """
    local_name = cfg['odds_filename'](year)
    dest = odds_dir / local_name

    if dest.exists():
        return True

    url = cfg['odds_url'](year)
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            dest.write_bytes(r.content)
            print(f"  ✅ {local_name} ({len(r.content) / 1024:.0f} KB)")
            return True
    except Exception:
        pass

    print(f"  ❌ {year} — non trouvé ({url})")
    return False


def inspect_odds_file(year: int, cfg: dict, odds_dir: Path) -> None:
    """Inspecte la structure d'un fichier de cotes."""
    import pandas as pd

    local_name = cfg['odds_filename'](year)
    path = odds_dir / local_name
    if not path.exists():
        print(f"  Fichier {local_name} introuvable.")
        return

    ext = path.suffix.lstrip('.')
    if ext in ['xlsx', 'xls']:
        from backtest_real import _read_excel_auto
        df = _read_excel_auto(path)
        df = df.head(3)
    else:
        df = pd.read_csv(path, nrows=3, encoding='latin-1')

    print(f"\n  Colonnes {local_name} :")
    print(f"  {list(df.columns)}")
    print(f"\n  Aperçu :")
    print(df.to_string())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Télécharge les cotes historiques depuis tennis-data.co.uk."
    )
    parser.add_argument(
        '--tour',
        choices=['atp', 'wta'],
        default='atp',
        help="Tour à télécharger : atp (défaut) ou wta",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tour = args.tour

    cfg = get_tour_config(tour)
    paths = get_paths(tour)
    make_dirs(tour)

    odds_dir = paths['odds_dir']

    print("=" * 55)
    print(f"TÉLÉCHARGEMENT COTES HISTORIQUES {tour.upper()} — tennis-data.co.uk")
    print("=" * 55)
    print(f"Destination : {odds_dir}\n")

    success = 0
    for year in tqdm(YEARS, desc="Téléchargement"):
        ok = download_odds_file(year, cfg, odds_dir)
        success += ok
        time.sleep(0.5)

    print(f"\n✅ {success}/{len(YEARS)} fichiers récupérés")

    # Inspecter 2023 pour voir la structure
    print("\n── Structure du fichier 2023 ────────────────────────")
    inspect_odds_file(2023, cfg, odds_dir)
