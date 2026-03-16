# src/espn_client.py
"""
Client ESPN pour les resultats tennis ATP/WTA en quasi-temps-reel.
Endpoint public, sans authentification, noms + gagnant inline.

URL: https://site.api.espn.com/apis/site/v2/sports/tennis/{tour}/scoreboard?dates=YYYYMMDD

Usage :
    from espn_client import fetch_recent
    df = fetch_recent("atp", days=14)
    df = fetch_recent("wta", days=7)
"""

import re
import time
import requests
import pandas as pd
from datetime import date, timedelta

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/tennis/{tour}/scoreboard"

# ─────────────────────────────────────────────────────────────────────────────
# Surface + niveau tournoi depuis le nom
# ─────────────────────────────────────────────────────────────────────────────

_CLAY = re.compile(
    r"roland[- ]garros|monte[- ]carlo|madrid|internazionali|rome|hamburg|"
    r"barcelona|buenos[- ]aires|rio|lyon|geneva|munich|acapulco|marrakech|"
    r"estoril|bucharest|bastad|umag|kitzbuhel|gstaad|houston|bogota|cordoba|"
    r"istanbul|casablanca|marbella|portugal|parma|cagliari|palermo",
    re.I,
)
_GRASS = re.compile(
    r"wimbledon|halle|queens|queen\'?s|eastbourne|mallorca|newport|"
    r"birmingham|nottingham|hertogenbosch|rosmalen",
    re.I,
)
_GRAND_SLAMS = re.compile(
    r"australian open|roland[- ]garros|wimbledon|us open", re.I
)
_MASTERS = re.compile(
    r"bnp paribas open|miami open|monte[- ]carlo|madrid open|"
    r"internazionali|cincinnati|canadian|montreal|toronto|"
    r"shanghai|rolex paris|paris masters|nitto atp|china open",
    re.I,
)
_FINALS = re.compile(r"atp finals|wta finals|barclays atp|nitto atp finals", re.I)


def _surface(name: str) -> str:
    if _CLAY.search(name):
        return "Clay"
    if _GRASS.search(name):
        return "Grass"
    return "Hard"


def _level(name: str) -> str:
    if _GRAND_SLAMS.search(name):
        return "G"
    if _MASTERS.search(name):
        return "M"
    if _FINALS.search(name):
        return "F"
    return "A"


def _best_of(level: str, round_str: str) -> int:
    """Grand Slam SF/F = 5 sets pour ATP, 3 pour WTA."""
    if level == "G" and re.search(r"\b(sf|semifinal|final)\b", round_str, re.I):
        return 5
    return 3


def _parse_round(comp: dict) -> str:
    """Extrait le nom du round depuis la competition ESPN."""
    notes = comp.get("notes", [])
    if notes and isinstance(notes, list):
        headline = notes[0].get("headline", "")
        if headline:
            return headline
    # Fallback: type.shortDetail ou type.description
    status = comp.get("status", {})
    detail = status.get("type", {}).get("shortDetail", "")
    return detail


# ─────────────────────────────────────────────────────────────────────────────
# Fetch ESPN
# ─────────────────────────────────────────────────────────────────────────────

_HEADERS = {
    "Accept": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
}

# Slugs des groupings ESPN a conserver par tour (singles uniquement)
_SINGLES_SLUG = {
    "atp": "mens-singles",
    "wta": "womens-singles",
}


def _parse_competitions(event: dict, tour: str) -> list[dict]:
    """
    Extrait les competitions singles du bon tour depuis un event ESPN.
    Gere les tournois mixtes (ex: Indian Wells) via le slug du grouping.
    """
    target_slug = _SINGLES_SLUG[tour]
    matches = []

    tourney_name = event.get("name", "Unknown")
    surface = _surface(tourney_name)
    level   = _level(tourney_name)

    for grouping in event.get("groupings", []):
        grp_slug = grouping.get("grouping", {}).get("slug", "")
        if grp_slug != target_slug:
            continue

        for comp in grouping.get("competitions", []):
            if not comp.get("status", {}).get("type", {}).get("completed", False):
                continue

            competitors = comp.get("competitors", [])
            if len(competitors) != 2:
                continue

            winner_c = next((c for c in competitors if c.get("winner")), None)
            loser_c  = next((c for c in competitors if not c.get("winner")), None)
            if winner_c is None or loser_c is None:
                continue

            winner_name = winner_c.get("athlete", {}).get("displayName", "").strip()
            loser_name  = loser_c.get("athlete",  {}).get("displayName", "").strip()
            if not winner_name or not loser_name:
                continue

            round_str = comp.get("round", {}).get("displayName", "")
            # Utiliser la date reelle du match (startDate), pas la date de fetch
            start_date_raw = comp.get("startDate", "")
            try:
                match_date = pd.Timestamp(start_date_raw).tz_convert(None).normalize()
            except Exception:
                match_date = None  # sera filtre dans fetch_recent

            matches.append({
                "tourney_date":    match_date,
                "tourney_name":    tourney_name,
                "surface":         surface,
                "tourney_level":   level,
                "round":           round_str,
                "best_of":         _best_of(level, round_str),
                "winner_name":     winner_name,
                "loser_name":      loser_name,
                "winner_rank":     None,
                "loser_rank":      None,
                "winner_rank_pts": None,
                "loser_rank_pts":  None,
                "source":          "espn",
            })

    return matches


def _fetch_day(tour: str, date_str: str, session: requests.Session) -> list[dict]:
    """
    Fetche le scoreboard ESPN pour un tour+date (date_str = 'YYYYMMDD').
    Retourne tous les matchs completes du tournoi en cours ce jour-la.
    La date du match est extraite de startDate (pas de la date de fetch).
    """
    url = SCOREBOARD_URL.format(tour=tour)
    try:
        r = session.get(url, params={"dates": date_str}, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"    ESPN {tour} {date_str}: erreur {e}")
        return []

    matches = []
    for event in data.get("events", []):
        matches.extend(_parse_competitions(event, tour))

    return matches


def fetch_recent(tour: str, days: int = 14) -> pd.DataFrame:
    """
    Fetche les matchs ESPN des `days` derniers jours pour ATP ou WTA.

    ESPN retourne tous les matchs completes du tournoi en cours pour une date donnee.
    On effectue 1 appel par semaine + aujourd'hui pour couvrir la periode sans doublons.

    Args:
        tour: 'atp' ou 'wta'
        days: fenetre en jours (defaut: 14)

    Returns:
        DataFrame au format Sackmann-compatible, deduplique, filtre sur la periode.
    """
    today = date.today()
    cutoff = pd.Timestamp(today) - pd.Timedelta(days=days)

    session = requests.Session()
    session.headers.update(_HEADERS)

    # Dates a fetcher : J-days, J-days+7, J-days+14... + aujourd'hui
    # (ESPN retourne tout le tournoi pour chaque date → deduplication suffisante)
    fetch_dates = []
    d = today - timedelta(days=days)
    while d <= today:
        fetch_dates.append(d)
        d += timedelta(days=7)
    if today not in fetch_dates:
        fetch_dates.append(today)

    all_matches: list[dict] = []
    for d in fetch_dates:
        date_str = d.strftime("%Y%m%d")
        day_matches = _fetch_day(tour, date_str, session)
        print(f"    ESPN {tour.upper()} fetch@{d}: {len(day_matches)} matchs (raw)")
        all_matches.extend(day_matches)
        time.sleep(0.1)

    if not all_matches:
        return pd.DataFrame()

    df = pd.DataFrame(all_matches)
    df = df.dropna(subset=["tourney_date"])
    df = df[df["tourney_date"] >= cutoff]
    df = df.drop_duplicates(subset=["tourney_date", "winner_name", "loser_name"])
    df = df.sort_values("tourney_date").reset_index(drop=True)

    if not df.empty:
        print(f"    ESPN {tour.upper()}: {len(df)} matchs uniques "
              f"({df['tourney_date'].min().date()} -> {df['tourney_date'].max().date()})")
    return df


def fetch_scheduled(tour: str, target_date: date | None = None) -> list[dict]:
    """
    Retourne les matchs programmes (pas encore joues) pour un tour et une date.

    Args:
        tour: 'atp' ou 'wta'
        target_date: date cible (defaut: aujourd'hui)

    Returns:
        Liste de dicts {p1_name, p2_name, tournament, surface, round, best_of}
        compatibles avec predict_today.predict_matches().
    """
    if target_date is None:
        target_date = date.today()

    target_slug = _SINGLES_SLUG[tour]
    date_str = target_date.strftime("%Y%m%d")
    url = SCOREBOARD_URL.format(tour=tour)

    session = requests.Session()
    session.headers.update(_HEADERS)

    try:
        r = session.get(url, params={"dates": date_str}, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  ESPN scheduled {tour} {date_str}: erreur {e}")
        return []

    matches = []
    for event in data.get("events", []):
        tourney_name = event.get("name", "Unknown")
        surface = _surface(tourney_name)
        level   = _level(tourney_name)

        for grouping in event.get("groupings", []):
            grp_slug = grouping.get("grouping", {}).get("slug", "")
            if grp_slug != target_slug:
                continue

            for comp in grouping.get("competitions", []):
                state = comp.get("status", {}).get("type", {}).get("state", "")
                # Garder uniquement pre (programme) et in (en cours)
                if state == "post":
                    continue

                competitors = comp.get("competitors", [])
                if len(competitors) != 2:
                    continue

                p1_c = competitors[0]
                p2_c = competitors[1]
                p1_name = p1_c.get("athlete", {}).get("displayName", "").strip()
                p2_name = p2_c.get("athlete", {}).get("displayName", "").strip()
                if not p1_name or not p2_name:
                    continue

                round_str = comp.get("round", {}).get("displayName", "")

                # Heure estimee du match
                start_raw = comp.get("startDate", "")
                try:
                    start_time = pd.Timestamp(start_raw).tz_convert("Europe/Paris")
                    time_str = start_time.strftime("%H:%M")
                except Exception:
                    time_str = ""

                matches.append({
                    "p1_name":    p1_name,
                    "p2_name":    p2_name,
                    "tournament": tourney_name,
                    "surface":    surface,
                    "level":      level,
                    "round":      round_str,
                    "best_of":    _best_of(level, round_str),
                    "time":       time_str,
                    "state":      state,
                })

    return matches
