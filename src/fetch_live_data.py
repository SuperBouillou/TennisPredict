# src/fetch_live_data.py
"""
Mise a jour des donnees joueurs depuis ESPN (quasi temps-reel)
et tennis-data.co.uk (hebdomadaire, fallback ATP).

Sources par tour :
  ATP : ESPN scoreboard (14 derniers jours) + tennis-data.co.uk XLSX
  WTA : ESPN scoreboard uniquement (tennis-data n'a pas WTA)

Delai ESPN : 0-24h (live/J+1)
Delai tennis-data : 0-7 jours

Usage :
    python src/fetch_live_data.py --tour atp
    python src/fetch_live_data.py --tour wta
    python src/fetch_live_data.py --tour atp --force
"""

import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

# Force UTF-8 pour les prints avec emojis/fleches (Windows cp1252 par defaut)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
from pathlib import Path
from datetime import datetime, date

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import get_tour_config, get_paths, make_dirs


def _file_age_hours(path: Path) -> float:
    if not path.exists():
        return float("inf")
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime).total_seconds() / 3600


def _load_elo(processed_dir: Path):
    """Charge l'ELO depuis le snapshot le plus recent."""
    elo_upd   = processed_dir / "elo_ratings_updated.parquet"
    elo_final = processed_dir / "elo_ratings_final.parquet"
    elo_src   = elo_upd if elo_upd.exists() else elo_final

    if not elo_src.exists():
        print("  ERREUR: elo_ratings_final.parquet introuvable. Lancer compute_elo.py.")
        return None, None, elo_upd

    df = pd.read_parquet(elo_src)
    ratings = dict(zip(df["full_name"], df["elo_global"]))
    surface = {
        name: {
            "Hard":   float(row.get("elo_Hard",   1500)),
            "Clay":   float(row.get("elo_Clay",   1500)),
            "Grass":  float(row.get("elo_Grass",  1500)),
            "Carpet": float(row.get("elo_Carpet", 1500)),
        }
        for name, row in df.set_index("full_name").iterrows()
    }
    print(f"  ELO: {len(ratings):,} joueurs (source: {elo_src.name})")
    return ratings, surface, elo_upd


def _save_outputs(df_profiles, elo_ratings, elo_surface, processed_dir, elo_out_path):
    """Sauvegarde profils + snapshot ELO."""
    p = processed_dir / "player_profiles_updated.parquet"
    df_profiles.to_parquet(p, index=False)
    print(f"\n  -> {p.name} ({len(df_profiles)} joueurs)")

    all_names = set(elo_ratings) | set(elo_surface)
    recs = []
    for name in all_names:
        surf = elo_surface.get(name, {})
        recs.append({
            "full_name":  name,
            "elo_global": elo_ratings.get(name, 1500.0),
            "elo_Hard":   surf.get("Hard",   1500.0),
            "elo_Clay":   surf.get("Clay",   1500.0),
            "elo_Grass":  surf.get("Grass",  1500.0),
            "elo_Carpet": surf.get("Carpet", 1500.0),
        })
    pd.DataFrame(recs).to_parquet(elo_out_path, index=False)
    print(f"  -> {elo_out_path.name}")
    print(f"  Done: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


def _fetch_espn(tour: str, days: int = 14) -> pd.DataFrame:
    """Fetche les matchs ESPN des `days` derniers jours."""
    from espn_client import fetch_recent
    print(f"\n  ESPN {tour.upper()}: fetch des {days} derniers jours...")
    df = fetch_recent(tour, days=days)
    if df.empty:
        print("  ESPN: aucun match recupere.")
    else:
        print(f"  ESPN: {len(df)} matchs ({df['tourney_date'].min().date()} -> {df['tourney_date'].max().date()})")
    return df


def _fetch_tennis_data(tour: str, force: bool, cfg: dict, paths: dict) -> pd.DataFrame:
    """
    Telecharge et charge les matchs tennis-data.co.uk.
    ATP uniquement (tennis-data n'a pas WTA).
    """
    if tour == "wta":
        return pd.DataFrame()

    from update_database import refresh_source_files, load_new_matches, convert_to_sackmann_format

    odds_dir       = paths["odds_dir"]
    current_year   = date.today().year
    years_to_fetch = [current_year - 1, current_year]

    print(f"\n  tennis-data.co.uk: refresh {years_to_fetch}...")
    for year in years_to_fetch:
        xlsx_path = odds_dir / cfg["odds_filename"](year)
        age_h = _file_age_hours(xlsx_path)
        if not force and age_h < 6:
            print(f"  {year}: fichier recent ({age_h:.1f}h) - skip")
            continue
        print(f"  {year}: telechargement...")
        refresh_source_files([year], cfg, odds_dir)

    df_raw = load_new_matches(years_to_fetch, cfg, odds_dir)
    if df_raw.empty:
        return pd.DataFrame()

    df_td = convert_to_sackmann_format(df_raw, cfg)
    df_td["source"] = "tennis-data"
    print(f"  tennis-data: {len(df_td)} matchs convertis")
    return df_td


def _merge_sources(df_espn: pd.DataFrame, df_td: pd.DataFrame, espn_days: int = 14) -> pd.DataFrame:
    """
    Fusionne ESPN (recent) + tennis-data (historique).

    Strategie :
    - tennis-data fournit l'historique complet (annees N-1 et N)
    - ESPN couvre les `espn_days` derniers jours avec J+1 de delai
    - Pour la periode commune : ESPN prend priorite (plus frais)
    - Deduplication par (tourney_date, winner_name, loser_name)
    """
    if df_espn.empty and df_td.empty:
        return pd.DataFrame()
    if df_espn.empty:
        return df_td
    if df_td.empty:
        return df_espn

    # Retirer de tennis-data les matchs de la periode ESPN (evite doublons)
    espn_cutoff = pd.Timestamp(date.today()) - pd.Timedelta(days=espn_days + 1)
    df_td_old = df_td[df_td["tourney_date"] < espn_cutoff].copy()

    df_merged = pd.concat([df_td_old, df_espn], ignore_index=True)
    before = len(df_merged)
    df_merged = df_merged.drop_duplicates(subset=["tourney_date", "winner_name", "loser_name"])
    df_merged = df_merged.sort_values("tourney_date").reset_index(drop=True)

    if before != len(df_merged):
        print(f"  Deduplication: {before - len(df_merged)} doublons retires")
    print(f"  Total fusionne: {len(df_merged)} matchs")
    return df_merged


def run_update(tour: str, force: bool = False, espn_days: int = 14) -> None:
    cfg   = get_tour_config(tour)
    paths = get_paths(tour)
    make_dirs(tour)

    processed_dir = paths["processed_dir"]

    print("\n" + "=" * 55)
    print(f"LIVE DATA UPDATE - {tour.upper()}")
    print("=" * 55)

    # 1. Fetch ESPN (ATP + WTA)
    df_espn = _fetch_espn(tour, days=espn_days)

    # 2. Fetch tennis-data (ATP uniquement)
    df_td = _fetch_tennis_data(tour, force, cfg, paths)

    # 3. Fusionner
    df_new = _merge_sources(df_espn, df_td, espn_days=espn_days)
    if df_new.empty:
        print("\n  Aucun match disponible - arret.")
        return

    print(f"\n  Matchs totaux: {len(df_new)}")

    # 4. ELO
    elo_ratings, elo_surface, elo_updated_path = _load_elo(processed_dir)
    if elo_ratings is None:
        return

    from update_database import build_name_mapping, update_elo, apply_elo_decay

    name_map = build_name_mapping(df_new, elo_ratings)
    df_mapped = df_new.copy()
    df_mapped["winner_name"] = df_mapped["winner_name"].map(lambda n: name_map.get(n, n))
    df_mapped["loser_name"]  = df_mapped["loser_name"].map(lambda n: name_map.get(n, n))

    print(f"\n  ELO update ({len(df_mapped)} matchs)...")
    elo_ratings, elo_surface = update_elo(df_mapped, elo_ratings, elo_surface, cfg)

    active = set(df_mapped["winner_name"].tolist() + df_mapped["loser_name"].tolist())
    elo_ratings, elo_surface = apply_elo_decay(
        elo_ratings, elo_surface, active, datetime.today()
    )

    # 5. Profils joueurs
    from update_database import update_player_profiles

    print("\n  Calcul profils joueurs...")
    df_profiles = update_player_profiles(df_mapped, elo_ratings, elo_surface, name_map)

    _save_outputs(df_profiles, elo_ratings, elo_surface, processed_dir, elo_updated_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Refresh ESPN + tennis-data & recalcule profils joueurs"
    )
    parser.add_argument("--tour",  default="atp", choices=["atp", "wta"])
    parser.add_argument("--force", action="store_true",
                        help="Re-telecharge tennis-data meme si fichier recent")
    parser.add_argument("--days",  type=int, default=14,
                        help="Jours en arriere pour ESPN (defaut: 14)")
    args = parser.parse_args()
    run_update(tour=args.tour, force=args.force, espn_days=args.days)
