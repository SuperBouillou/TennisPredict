# src/download_data.py

import argparse
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import requests
from tqdm import tqdm

# Ajouter src/ au path pour l'import de config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import get_tour_config, get_paths, make_dirs


def download_file(url: str, dest: Path, label: str) -> bool:
    if dest.exists():
        return True
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            dest.write_bytes(response.content)
            return True
        else:
            print(f"  ❌ {label} — HTTP {response.status_code}")
            return False
    except requests.exceptions.Timeout:
        print(f"  ⏱️  {label} — Timeout")
        return False
    except requests.exceptions.ConnectionError:
        print(f"  🔌 {label} — Erreur connexion")
        return False


def download_main_matches(years: list, base_url: str, prefix: str, raw_dir: Path) -> None:
    print("\n" + "=" * 55)
    print(f"1/4 — MATCHS {prefix.upper()} TOUR PRINCIPAL")
    print("=" * 55)
    success, skipped, failed = 0, 0, 0
    for year in tqdm(years, desc=f"{prefix.upper()} Tour"):
        filename = f"{prefix}_matches_{year}.csv"
        dest = raw_dir / filename
        if dest.exists():
            skipped += 1
            continue
        ok = download_file(f"{base_url}/{filename}", dest, filename)
        success += ok
        failed += not ok
        time.sleep(0.1)
    print(f"  ✅ Téléchargés : {success} | ⏭️  Déjà présents : {skipped} | ❌ Échecs : {failed}")


def download_qual_chall(years: list, base_url: str, prefix: str, raw_dir: Path) -> None:
    if not years:
        print("\n" + "=" * 55)
        print("2/4 — QUALIFICATIONS + CHALLENGERS (ignoré pour ce tour)")
        print("=" * 55)
        return
    print("\n" + "=" * 55)
    print("2/4 — QUALIFICATIONS + CHALLENGERS")
    print("=" * 55)
    success, skipped, failed = 0, 0, 0
    for year in tqdm(years, desc="Qual+Chall"):
        filename = f"{prefix}_matches_qual_chall_{year}.csv"
        dest = raw_dir / filename
        if dest.exists():
            skipped += 1
            continue
        ok = download_file(f"{base_url}/{filename}", dest, filename)
        success += ok
        failed += not ok
        time.sleep(0.1)
    print(f"  ✅ Téléchargés : {success} | ⏭️  Déjà présents : {skipped} | ❌ Échecs : {failed}")


def download_rankings(ranking_files: list, base_url: str, raw_dir: Path) -> None:
    print("\n" + "=" * 55)
    print("3/4 — CLASSEMENTS")
    print("=" * 55)
    for filename in ranking_files:
        dest = raw_dir / filename
        ok = download_file(f"{base_url}/{filename}", dest, filename)
        print(f"  {'✅' if ok else '❌'} {filename}")


def download_players(player_file: str, base_url: str, raw_dir: Path) -> None:
    print("\n" + "=" * 55)
    print("4/4 — PROFILS JOUEURS")
    print("=" * 55)
    dest = raw_dir / player_file
    ok = download_file(f"{base_url}/{player_file}", dest, player_file)
    print(f"  {'✅' if ok else '❌'} {player_file}")


def rapport_final(prefix: str, raw_dir: Path) -> None:
    print("\n" + "=" * 55)
    print("RAPPORT FINAL")
    print("=" * 55)
    main_files = sorted(raw_dir.glob(f"{prefix}_matches_[0-9]*.csv"))
    qual_files = sorted(raw_dir.glob(f"{prefix}_matches_qual_chall_*.csv"))
    rank_files = sorted(raw_dir.glob(f"{prefix}_rankings_*.csv"))
    player_file = raw_dir / f"{prefix}_players.csv"
    print(f"  🎾 Matchs {prefix.upper()} Tour      : {len(main_files)} fichiers")
    print(f"  🎾 Qual + Challengers   : {len(qual_files)} fichiers")
    print(f"  📊 Classements          : {len(rank_files)} fichiers")
    print(f"  👤 Profils joueurs      : {'✅' if player_file.exists() else '❌'}")
    if main_files:
        years = [int(f.stem.split('_')[-1]) for f in main_files]
        print(f"\n  📅 Période couverte : {min(years)} → {max(years)}")
    total_mb = sum(f.stat().st_size for f in raw_dir.iterdir()) / 1_048_576
    print(f"  💾 Taille totale        : {total_mb:.1f} MB")
    print(f"\n  📁 Destination          : {raw_dir.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Télécharge les données historiques de matchs depuis le repo Sackmann."
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

    raw_dir = paths['raw_dir']
    base_url = cfg['sackmann_repo']
    prefix = cfg['file_prefix']

    print(f"🎾 Démarrage du téléchargement — Données {tour.upper()} complètes")
    print(f"   Destination : {raw_dir.resolve()}")

    download_main_matches(cfg['match_years'], base_url, prefix, raw_dir)
    download_qual_chall(cfg['qual_chall_years'], base_url, prefix, raw_dir)
    download_rankings(cfg['ranking_files'], base_url, raw_dir)
    download_players(cfg['player_file'], base_url, raw_dir)
    rapport_final(prefix, raw_dir)

    print("\n✅ Téléchargement terminé.")
