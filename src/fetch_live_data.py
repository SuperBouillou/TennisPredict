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

from config import get_tour_config, get_paths, make_dirs, _file_age_hours


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


# Colonnes lues depuis matches_features_final.parquet (subset pour performance)
_HIST_COLS = [
    'tourney_date', 'p1_name', 'p2_name', 'target',
    'p1_winrate_5', 'p1_winrate_10', 'p1_winrate_20',
    'p1_winrate_surf_Hard', 'p1_winrate_surf_Clay', 'p1_winrate_surf_Grass',
    'p1_streak', 'p1_rank', 'p1_rank_points',
    'p2_winrate_5', 'p2_winrate_10', 'p2_winrate_20',
    'p2_winrate_surf_Hard', 'p2_winrate_surf_Clay', 'p2_winrate_surf_Grass',
    'p2_streak', 'p2_rank', 'p2_rank_points',
]


def _extract_historical_profiles(processed_dir: Path) -> pd.DataFrame:
    """
    Extrait le dernier état connu de chaque joueur depuis matches_features_final.parquet.
    Donne des winrates fiables calculés sur tout l'historique (vs ESPN 14 jours seulement).
    Retourne un DataFrame : une ligne par joueur.
    """
    hist_path = processed_dir / 'matches_features_final.parquet'
    if not hist_path.exists():
        print("  [profiles] matches_features_final.parquet introuvable - skip historique")
        return pd.DataFrame()

    # Charger seulement les colonnes nécessaires (173 cols → 20 cols, beaucoup plus rapide)
    df = pd.read_parquet(hist_path, columns=_HIST_COLS)

    rename = {
        'p1_winrate_5': 'winrate_5', 'p1_winrate_10': 'winrate_10', 'p1_winrate_20': 'winrate_20',
        'p1_winrate_surf_Hard': 'winrate_surf_Hard', 'p1_winrate_surf_Clay': 'winrate_surf_Clay',
        'p1_winrate_surf_Grass': 'winrate_surf_Grass',
        'p1_streak': 'streak', 'p1_rank': 'rank', 'p1_rank_points': 'rank_points',
    }

    p1 = df[['tourney_date', 'p1_name', 'target'] + list(rename.keys())].copy()
    p1 = p1.rename(columns={'p1_name': 'player_name', 'target': 'won', **rename})

    rename2 = {k.replace('p1_', 'p2_'): v for k, v in rename.items()}
    p2 = df[['tourney_date', 'p2_name', 'target'] + list(rename2.keys())].copy()
    p2 = p2.rename(columns={'p2_name': 'player_name', **rename2})
    p2['won'] = 1 - p2['target']  # p2 won when target=0
    p2 = p2.drop(columns=['target'])

    combined = pd.concat([p1, p2], ignore_index=True)
    combined = combined.dropna(subset=['player_name'])
    combined = combined.sort_values('tourney_date')

    # Dernier match par joueur = état le plus récent
    last = combined.groupby('player_name').last().reset_index()
    last = last.rename(columns={'tourney_date': 'last_match_hist'})
    last['name_key'] = last['player_name'].str.lower().str.strip()

    # Forme réelle : derniers 5 résultats par joueur
    def _form5(series):
        return ','.join(['W' if v == 1 else 'L' for v in series.tolist()[-5:]])

    form_df = combined.groupby('player_name')['won'].apply(_form5).reset_index()
    form_df.columns = ['player_name', 'form_last5']
    last = last.merge(form_df, on='player_name', how='left')

    print(f"  [profiles] Historique: {len(last):,} joueurs (base depuis matches_features_final)")
    return last


def _merge_profiles(hist: pd.DataFrame, recent: pd.DataFrame,
                    elo_ratings: dict, elo_surface: dict) -> pd.DataFrame:
    """
    Fusionne les profils historiques (winrates fiables, 5 670 joueurs) avec
    les profils récents ESPN/tennis-data (fraîcheur, fatigue).

    Règle:
    - Si >= 5 matchs récents (matches_14d) : utiliser winrates récents
    - Sinon : conserver winrates historiques, mettre à jour fatigue/streak uniquement
    - ELO toujours depuis elo_ratings/elo_surface (déjà mis à jour)
    - Tous les joueurs historiques sont inclus (pas seulement ceux des 21 derniers jours)
    """
    today = pd.Timestamp('today').normalize()

    if hist.empty:
        return recent

    # Déduplication sur name_key (garder la ligne avec last_match la plus récente)
    if not recent.empty and 'name_key' in recent.columns:
        recent = recent.sort_values('last_match', ascending=False).drop_duplicates('name_key')
        recent_idx = recent.set_index('name_key')
    else:
        recent_idx = pd.DataFrame()

    winrate_fields = ['winrate_5', 'winrate_10', 'winrate_20',
                      'winrate_surf_Hard', 'winrate_surf_Clay', 'winrate_surf_Grass', 'streak']
    fatigue_fields = ['matches_7d', 'matches_14d', 'matches_21d', 'days_since', 'last_match', 'n_matches']

    merged_rows = []
    hist_keys = set(hist['name_key'])

    for _, h_row in hist.iterrows():
        nk       = h_row['name_key']
        player   = h_row['player_name']
        elo      = elo_ratings.get(player, 1500.0)
        elo_h    = elo_surface.get(player, {}).get('Hard',  1500.0)
        elo_c    = elo_surface.get(player, {}).get('Clay',  1500.0)
        elo_g    = elo_surface.get(player, {}).get('Grass', 1500.0)

        if not recent_idx.empty and nk in recent_idx.index:
            r = recent_idx.loc[nk]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            n_recent = float(r.get('matches_14d', 0) or 0)

            if n_recent >= 5:
                row = r.to_dict()
            else:
                # Sparse recent data: historical winrates + recent fatigue
                row = h_row.to_dict()
                for field in fatigue_fields + ['streak']:
                    if field in r and pd.notna(r[field]):
                        row[field] = r[field]
            # Recent form always takes priority when player has any recent matches
            r_form = r.get('form_last5')
            if r_form and isinstance(r_form, str) and r_form.strip():
                row['form_last5'] = r_form
        else:
            # Pas joué récemment: historique + recalcul days_since
            row = h_row.to_dict()
            last_hist = row.get('last_match_hist')
            if last_hist is not None:
                last_ts = pd.Timestamp(last_hist)
                row['last_match']   = last_ts.date()
                row['days_since']   = (today - last_ts).days
            row['matches_7d']   = 0
            row['matches_14d']  = 0
            row['matches_21d']  = 0
            row['n_matches']    = 0

        row.update({'player_name': player, 'name_key': nk,
                    'elo': elo, 'elo_Hard': elo_h, 'elo_Clay': elo_c, 'elo_Grass': elo_g})
        merged_rows.append(row)

    # Joueurs dans recent mais absents de l'historique (nouveaux pros)
    if not recent_idx.empty:
        for nk, r in recent_idx.iterrows():
            if nk not in hist_keys:
                row = r.to_dict()
                player = row.get('player_name', '')
                row.update({
                    'elo':      elo_ratings.get(player, 1500.0),
                    'elo_Hard': elo_surface.get(player, {}).get('Hard',  1500.0),
                    'elo_Clay': elo_surface.get(player, {}).get('Clay',  1500.0),
                    'elo_Grass':elo_surface.get(player, {}).get('Grass', 1500.0),
                })
                merged_rows.append(row)

    df = pd.DataFrame(merged_rows)
    # Nettoyer : enlever les lignes sans player_name valide
    df = df.dropna(subset=['player_name'])
    df = df[df['player_name'].astype(str).str.strip() != '']

    if 'name_key'   not in df.columns or df['name_key'].isna().any():
        df['name_key']   = df['player_name'].str.lower().str.strip()
    if 'last_name'  not in df.columns: df['last_name']  = df['player_name'].str.split().str[-1].str.lower()
    if 'first_init' not in df.columns: df['first_init'] = df['player_name'].str.split().str[0].str[0].str.lower()

    print(f"  [profiles] Fusionnes: {len(df):,} joueurs total")
    return df


def _inject_rankings(df_profiles: pd.DataFrame, ranking_lookup: dict) -> pd.DataFrame:
    """
    Injecte les classements depuis ranking_lookup dans les profils.
    N'écrase que les rangs manquants (NaN/0) — conserve les rangs des matchs récents.
    """
    if not ranking_lookup or df_profiles.empty:
        return df_profiles

    import numpy as np

    def get_rank(name_key) -> float:
        if not isinstance(name_key, str) or not name_key:
            return np.nan
        if name_key in ranking_lookup:
            return float(ranking_lookup[name_key])
        last = name_key.split()[-1]
        if not last:
            return np.nan
        matches = {k: v for k, v in ranking_lookup.items() if k.endswith(last)}
        if len(matches) == 1:
            return float(list(matches.values())[0])
        return np.nan

    lookup_ranks = df_profiles['name_key'].apply(get_rank)
    # ranking_lookup (dec 2024) est toujours plus récent que l'historique Sackmann
    # → priorité ranking_lookup, fallback sur le rang historique
    df_profiles['rank'] = lookup_ranks.where(lookup_ranks.notna(), df_profiles['rank'])

    filled = lookup_ranks.notna().sum()
    print(f"  [profiles] Rankings injectes: {filled:,} joueurs")
    return df_profiles


_RANKING_URLS = {
    "atp": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_rankings_current.csv",
    "wta": "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_rankings_current.csv",
}
_PLAYERS_URLS = {
    "atp": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_players.csv",
    "wta": "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_players.csv",
}

_ESPN_RANKINGS_URL = "https://sports.core.api.espn.com/v2/sports/tennis/leagues/{tour}/rankings"


def _fetch_espn_rankings(tour: str) -> dict:
    """
    Fetch live rankings from ESPN API for ATP or WTA.
    Returns {full_name_lower: rank} or {} on failure.
    ESPN has up-to-date rankings vs Sackmann which may lag months behind.
    """
    import requests as _req
    try:
        index_url = _ESPN_RANKINGS_URL.format(tour=tour)
        resp = _req.get(index_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200 or not resp.json().get("items"):
            return {}
        latest_ref = resp.json()["items"][0]["$ref"]
        data = _req.get(latest_ref, timeout=10, headers={"User-Agent": "Mozilla/5.0"}).json()
        ranks = data.get("ranks", [])
        lookup = {}
        for entry in ranks:
            rank = entry.get("current")
            ref = entry.get("athlete", {}).get("$ref")
            if not rank or not ref:
                continue
            athlete = _req.get(ref, timeout=8, headers={"User-Agent": "Mozilla/5.0"}).json()
            name = (athlete.get("displayName") or athlete.get("fullName") or "").lower().strip()
            if name:
                lookup[name] = int(rank)
        print(f"  Rankings ESPN: {len(lookup)} joueurs {tour.upper()} (live)")
        return lookup
    except Exception as e:
        print(f"  Rankings ESPN: erreur ({e}) — fallback Sackmann")
        return {}


def build_ranking_lookup(tour: str, raw_dir: Path, force: bool = False) -> dict:
    """
    Downloads (if stale) and returns a dict {full_name_lower: rank}.
    Uses atp/wta_rankings_current.csv joined with atp/wta_players.csv.
    """
    import requests as _req
    import io

    rankings_path = raw_dir / f"{tour}_rankings_current.csv"
    players_path  = raw_dir / f"{tour}_players.csv"

    # Download if stale (> 4h) or missing
    for path, url in [(rankings_path, _RANKING_URLS[tour]),
                      (players_path,  _PLAYERS_URLS[tour])]:
        age_h = _file_age_hours(path)
        if force or age_h > 4:
            try:
                resp = _req.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
                if resp.status_code == 200:
                    path.write_bytes(resp.content)
                    print(f"  Rankings: {path.name} téléchargé ({len(resp.content)//1024} KB)")
            except Exception as e:
                print(f"  Rankings: erreur téléchargement {path.name}: {e}")

    if not rankings_path.exists() or not players_path.exists():
        return {}

    try:
        ranks = pd.read_csv(rankings_path, low_memory=False,
                            dtype={"rank": float, "player": str, "points": float})
        ranks = ranks.rename(columns={"ranking_date": "date", "player": "player_id"})
        ranks["date"] = pd.to_numeric(ranks["date"], errors="coerce")
        ranks = ranks.dropna(subset=["date", "rank", "player_id"])
        ranks = ranks.sort_values("date").groupby("player_id").last().reset_index()

        players = pd.read_csv(players_path, low_memory=False, dtype=str)
        players["full_name"] = (players["name_first"].fillna("") + " " + players["name_last"].fillna("")).str.strip()
        players["player_id"] = players["player_id"].astype(str)
        ranks["player_id"] = ranks["player_id"].astype(str)

        merged = ranks.merge(players[["player_id","full_name"]], on="player_id", how="left")
        merged = merged.dropna(subset=["full_name"])
        merged = merged[merged["full_name"].str.strip() != ""]

        lookup = dict(zip(merged["full_name"].str.lower().str.strip(),
                          merged["rank"].astype(int)))
        print(f"  Rankings: {len(lookup)} joueurs {tour.upper()} chargés (Sackmann)")

        # Sackmann rankings lag months behind for both ATP and WTA — override with ESPN live data
        espn_lookup = _fetch_espn_rankings(tour)
        if espn_lookup:
            lookup.update(espn_lookup)   # ESPN takes precedence for known players

        # Save timestamp so webapp can display ranking freshness
        import json as _json
        from datetime import datetime as _dt
        meta_path = raw_dir / f"{tour}_ranking_lookup_meta.json"
        meta_path.write_text(_json.dumps({"updated_at": _dt.now().isoformat(timespec="seconds")}))
        return lookup
    except Exception as e:
        print(f"  Rankings: erreur lecture: {e}")
        return {}


def run_update(tour: str, force: bool = False, espn_days: int = 30) -> None:
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
    # 5a. Profils récents depuis ESPN/tennis-data (fatigue, streaks, winrates frais)
    recent_profiles = update_player_profiles(df_mapped, elo_ratings, elo_surface, name_map)

    # 5b. Base historique (winrates fiables calculés sur tout l'historique Sackmann)
    hist_profiles = _extract_historical_profiles(processed_dir)

    # 5c. Fusion: winrates historiques pour les joueurs sans assez de données récentes
    df_profiles = _merge_profiles(hist_profiles, recent_profiles, elo_ratings, elo_surface)

    # 6. Rankings actuels (Sackmann GitHub) + injection dans les profils
    raw_dir = paths["raw_dir"]
    ranking_lookup = build_ranking_lookup(tour, raw_dir, force=force)
    df_profiles = _inject_rankings(df_profiles, ranking_lookup)

    _save_outputs(df_profiles, elo_ratings, elo_surface, processed_dir, elo_updated_path)

    if ranking_lookup:
        import json
        out = processed_dir / "ranking_lookup.json"
        out.write_text(json.dumps(ranking_lookup, ensure_ascii=False))
        print(f"  -> {out.name} ({len(ranking_lookup)} joueurs)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Refresh ESPN + tennis-data & recalcule profils joueurs"
    )
    parser.add_argument("--tour",  default="atp", choices=["atp", "wta"])
    parser.add_argument("--force", action="store_true",
                        help="Re-telecharge tennis-data meme si fichier recent")
    parser.add_argument("--days",  type=int, default=30,
                        help="Jours en arriere pour ESPN (defaut: 30)")
    args = parser.parse_args()
    run_update(tour=args.tour, force=args.force, espn_days=args.days)
