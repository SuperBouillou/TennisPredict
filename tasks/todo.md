# Sofascore Live Data Feed — Implementation Plan

> **Goal:** Replace/supplement the weekly tennis-data.co.uk source with near real-time match results from Sofascore (unofficial API), so `player_profiles_updated.parquet` reflects matches from the last few hours, not the last week.

**Architecture:** Two new modules — `sofascore_client.py` (pure HTTP/parsing) and `fetch_sofascore.py` (orchestration). `fetch_sofascore.py` is a drop-in supplement to `update_database.py`: it fetches the last N days of matches from Sofascore, converts them to the internal Sackmann format, then calls the same incremental ELO + profile update logic already used by `update_database.py`. A scheduled Windows task runs it each morning.

**Tech Stack:** `requests`, `pandas`, `joblib`, `python-dateutil`; Windows Task Scheduler via `schtasks`

---

## Task 1 — Sofascore API Client (`src/sofascore_client.py`)

**Files:**
- Create: `src/sofascore_client.py`

**What it does:** Pure HTTP layer. No business logic. Fetches raw JSON from Sofascore's unofficial API and returns parsed Python dicts.

**Key Sofascore endpoints:**
```
GET https://api.sofascore.com/api/v1/sport/tennis/scheduled-events/{YYYY-MM-DD}
→ {"events": [...]}   # all tennis events for that date

GET https://api.sofascore.com/api/v1/unique-tournament/{tournament_id}
→ {"uniqueTournament": {"groundType": "hard", ...}}   # surface info
```

**Match object structure (Sofascore):**
```json
{
  "id": 12345,
  "status": {"type": "finished"},
  "winnerCode": 1,
  "homeTeam": {"id": 111, "name": "Sinner Jannik"},
  "awayTeam": {"id": 222, "name": "Djokovic Novak"},
  "homeScore": {"current": 2},
  "awayScore": {"current": 1},
  "tournament": {
    "id": 9999,
    "name": "Australian Open",
    "category": {"name": "ATP", "slug": "atp"},
    "uniqueTournament": {"id": 888, "name": "Australian Open"}
  },
  "roundInfo": {"round": 7, "name": "Final"},
  "startTimestamp": 1738123456
}
```

- [ ] **1.1 — Create `sofascore_client.py` with `fetch_events(date_str)`**

```python
# src/sofascore_client.py
import time
import requests
from datetime import date, timedelta
from typing import Optional

BASE_URL = "https://api.sofascore.com/api/v1"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://www.sofascore.com",
}

SURFACE_CACHE: dict[int, str] = {}

KNOWN_SURFACES: dict[int, str] = {
    # uniqueTournament IDs for major ATP/WTA events
    # Australian Open
    2230: "Hard",
    # Roland Garros
    2316: "Clay",
    # Wimbledon
    2317: "Grass",
    # US Open
    2318: "Hard",
    # Indian Wells
    2239: "Hard",
    # Miami
    2240: "Hard",
    # Monte Carlo
    2243: "Clay",
    # Madrid
    2244: "Clay",
    # Rome
    2245: "Clay",
    # Canadian Open
    2246: "Hard",
    # Cincinnati
    2247: "Hard",
    # Shanghai
    2251: "Hard",
    # Paris
    2254: "Hard",
}


def _get(url: str, retries: int = 3) -> Optional[dict]:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(2 ** attempt)
        except requests.RequestException:
            time.sleep(1)
    return None


def fetch_surface(unique_tournament_id: int) -> str:
    """Return surface string ('Hard','Clay','Grass') for a tournament, with caching."""
    if unique_tournament_id in KNOWN_SURFACES:
        return KNOWN_SURFACES[unique_tournament_id]
    if unique_tournament_id in SURFACE_CACHE:
        return SURFACE_CACHE[unique_tournament_id]

    data = _get(f"{BASE_URL}/unique-tournament/{unique_tournament_id}")
    surface = "Hard"  # default
    if data:
        ground = (data.get("uniqueTournament") or {}).get("groundType", "")
        surface = {
            "hard":    "Hard",
            "clay":    "Clay",
            "grass":   "Grass",
            "indoor":  "Hard",
            "carpet":  "Carpet",
        }.get(ground.lower(), "Hard")

    SURFACE_CACHE[unique_tournament_id] = surface
    time.sleep(0.3)  # be polite
    return surface


def fetch_events(date_str: str, tour_filter: str = "atp") -> list[dict]:
    """
    Fetch finished tennis events for `date_str` (YYYY-MM-DD).
    `tour_filter`: 'atp' | 'wta' | 'both'
    Returns list of raw event dicts.
    """
    data = _get(f"{BASE_URL}/sport/tennis/scheduled-events/{date_str}")
    if not data:
        return []

    tours = {"atp", "wta"} if tour_filter == "both" else {tour_filter}
    results = []
    for ev in data.get("events", []):
        if ev.get("status", {}).get("type") != "finished":
            continue
        cat = ev.get("tournament", {}).get("category", {}).get("slug", "").lower()
        if not any(t in cat for t in tours):
            continue
        results.append(ev)

    return results


def fetch_date_range(start: date, end: date, tour_filter: str = "atp") -> list[dict]:
    """Fetch all finished events between start and end (inclusive)."""
    events = []
    current = start
    while current <= end:
        day_events = fetch_events(current.strftime("%Y-%m-%d"), tour_filter)
        events.extend(day_events)
        current += timedelta(days=1)
        time.sleep(0.5)
    return events
```

- [ ] **1.2 — Quick smoke test (run in terminal)**

```bash
cd E:/Claude/botbet/tennis/tennis_ml
source venv/Scripts/activate
PYTHONIOENCODING=utf-8 python -c "
import sys; sys.path.insert(0, 'src')
from sofascore_client import fetch_events
from datetime import date, timedelta
yesterday = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')
evts = fetch_events(yesterday, 'atp')
print(f'Found {len(evts)} finished ATP events on {yesterday}')
if evts:
    e = evts[0]
    print(f'  Sample: {e[\"homeTeam\"][\"name\"]} vs {e[\"awayTeam\"][\"name\"]}')
    print(f'  Tournament: {e[\"tournament\"][\"name\"]}')
    print(f'  WinnerCode: {e.get(\"winnerCode\")}')
"
```

Expected: `Found N finished ATP events` with at least 1 sample during active tournament weeks. During off-weeks, 0 is normal.

---

## Task 2 — Match Converter (`src/fetch_sofascore.py`, part 1)

**Files:**
- Create: `src/fetch_sofascore.py`

Converts Sofascore event dicts → internal Sackmann-compatible format. Then maps player names to internal IDs.

**Internal match format required by `update_database.py` logic:**
```
tourney_id, tourney_name, tourney_date, tourney_level, surface,
draw_size, best_of, round, year,
winner_id, winner_name, loser_id, loser_name,
winner_rank, loser_rank, winner_rank_points, loser_rank_points,
winner_age, loser_age, score, minutes
```

**Round name mapping (Sofascore → Sackmann):**
```python
ROUND_MAP = {
    "Final":              "F",
    "Semifinal":          "SF",
    "Quarterfinal":       "QF",
    "Round of 16":        "R16",
    "Round of 32":        "R32",
    "Round of 64":        "R64",
    "Round of 128":       "R128",
    "1st Round":          "R128",
    "2nd Round":          "R64",
    "3rd Round":          "R32",
    "4th Round":          "R16",
}
```

**Tournament level mapping (Sofascore category → Sackmann level):**
Sofascore `uniqueTournament.primaryColorHex` or `category.name` gives us: "ATP", "WTA", "ATP Challenger", etc.
Use tournament name heuristics:
```python
LEVEL_MAP_ATP = {
    "Grand Slam": "G",
    "Masters": "M",      # ATP 1000
    "ATP 500": "A",
    "ATP 250": "A",
    "Finals": "F",
    "United Cup": "O",
    "Davis Cup": "D",
}
```

- [ ] **2.1 — Write `convert_events_to_matches()`**

```python
# Append to src/fetch_sofascore.py

import sys
import json
import time
import joblib
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import get_tour_config, get_paths
from sofascore_client import fetch_date_range, fetch_surface


ROUND_MAP = {
    "Final": "F", "Semifinal": "SF", "Quarterfinal": "QF",
    "Round of 16": "R16", "Round of 32": "R32",
    "Round of 64": "R64", "Round of 128": "R128",
    "1st Round": "R128", "2nd Round": "R64", "3rd Round": "R32",
    "4th Round": "R16",
}

LEVEL_KEYWORDS_ATP = [
    (["grand slam", "australian open", "roland garros", "wimbledon", "us open"], "G"),
    (["masters 1000", "masters", "atp 1000"], "M"),
    (["atp 500"], "A"),
    (["atp 250"], "A"),
    (["finals", "nitto", "year-end"], "F"),
    (["united cup", "laver cup"], "O"),
    (["davis cup"], "D"),
]

LEVEL_KEYWORDS_WTA = [
    (["grand slam", "australian open", "roland garros", "wimbledon", "us open"], "G"),
    (["wta 1000", "premier mandatory", "premier 5"], "PM"),
    (["wta 500", "premier"], "P"),
    (["wta 250", "international"], "I"),
    (["finals", "year-end"], "F"),
    (["billie jean king", "fed cup"], "D"),
]


def _infer_level(tourney_name: str, tour: str) -> str:
    name_low = tourney_name.lower()
    keywords = LEVEL_KEYWORDS_ATP if tour == "atp" else LEVEL_KEYWORDS_WTA
    for kws, level in keywords:
        if any(k in name_low for k in kws):
            return level
    return "A" if tour == "atp" else "I"   # default: regular tour event


def _sofascore_name_to_normalized(name: str) -> str:
    """'Sinner Jannik' → 'Jannik Sinner' (Sofascore uses 'Last First' order)."""
    parts = name.strip().split()
    if len(parts) >= 2:
        # Sofascore convention: Last First (but not always — check for known patterns)
        return " ".join(parts)   # return as-is; matching will handle it
    return name


def convert_events_to_matches(events: list[dict], tour: str) -> pd.DataFrame:
    """Convert raw Sofascore event dicts to internal match format."""
    rows = []
    for ev in events:
        try:
            winner_code = ev.get("winnerCode")  # 1=home, 2=away
            if winner_code not in (1, 2):
                continue

            home = ev["homeTeam"]
            away = ev["awayTeam"]
            winner = home if winner_code == 1 else away
            loser  = away if winner_code == 1 else home

            tourney      = ev.get("tournament", {})
            unique_t     = tourney.get("uniqueTournament", {})
            unique_t_id  = unique_t.get("id", 0)
            tourney_name = unique_t.get("name", tourney.get("name", "Unknown"))
            ts           = ev.get("startTimestamp", 0)
            match_date   = pd.Timestamp(ts, unit="s") if ts else pd.NaT
            round_name   = ROUND_MAP.get(
                ev.get("roundInfo", {}).get("name", ""), "R128"
            )
            surface      = fetch_surface(unique_t_id) if unique_t_id else "Hard"
            level        = _infer_level(tourney_name, tour)
            best_of      = 5 if (level == "G" and tour == "atp") else 3
            tourney_id   = f"scs_{unique_t_id}_{match_date.year if pd.notna(match_date) else 0}"

            rows.append({
                "tourney_id":           tourney_id,
                "tourney_name":         tourney_name,
                "tourney_date":         match_date,
                "tourney_level":        level,
                "surface":              surface,
                "draw_size":            64,
                "best_of":              best_of,
                "round":                round_name,
                "year":                 match_date.year if pd.notna(match_date) else 0,
                "winner_id":            f"scs_{winner['id']}",
                "winner_name":          winner["name"],
                "loser_id":             f"scs_{loser['id']}",
                "loser_name":           loser["name"],
                "winner_rank":          np.nan,
                "loser_rank":           np.nan,
                "winner_rank_points":   np.nan,
                "loser_rank_points":    np.nan,
                "winner_age":           np.nan,
                "loser_age":            np.nan,
                "score":                "",
                "minutes":              np.nan,
                "source":               "sofascore",
                "sofascore_id":         ev["id"],
            })
        except (KeyError, TypeError):
            continue

    return pd.DataFrame(rows)
```

- [ ] **2.2 — Test converter on live data**

```bash
PYTHONIOENCODING=utf-8 python -c "
import sys; sys.path.insert(0, 'src')
from datetime import date, timedelta
from sofascore_client import fetch_date_range
from fetch_sofascore import convert_events_to_matches
start = date.today() - timedelta(days=7)
events = fetch_date_range(start, date.today(), 'atp')
print(f'Raw events: {len(events)}')
df = convert_events_to_matches(events, 'atp')
print(f'Converted: {len(df)} rows')
if not df.empty:
    print(df[['tourney_name','surface','round','winner_name','loser_name']].head(5).to_string())
"
```

Expected: table with tournament names, surfaces, rounds, player names.

---

## Task 3 — Player Name Mapping

**Files:**
- Modify: `src/fetch_sofascore.py` (add `build_scs_name_map()` and `resolve_player_ids()`)
- Create: `data/cache/sofascore_player_map_{tour}.json` (auto-generated)

Sofascore uses names like `"Sinner Jannik"` (Last First) or `"Jannik Sinner"` (First Last). Internal pipeline uses Sackmann names like `"Jannik Sinner"`. We need to:
1. Build a mapping `sofascore_name → internal_player_id` using fuzzy name matching
2. Cache it so we don't re-match on every run

The fuzzy matching logic is already in `update_database.py::build_name_mapping()`. We'll import and adapt it.

- [ ] **3.1 — Write `resolve_player_names()` in `fetch_sofascore.py`**

```python
# Append to src/fetch_sofascore.py

import unicodedata
import re


def _normalize(name: str) -> str:
    """Lowercase, strip accents, remove punctuation."""
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[^a-z ]", "", name.lower()).strip()
    return name


def _name_variants(name: str) -> set[str]:
    """Generate candidate variants: original, reversed, no-spaces."""
    parts = _normalize(name).split()
    variants = {
        _normalize(name),
        " ".join(reversed(parts)),
        "".join(parts),
    }
    if len(parts) >= 2:
        variants.add(f"{parts[-1]} {' '.join(parts[:-1])}")  # Last First
        variants.add(f"{parts[0]} {' '.join(parts[1:])}")    # First Last
    return variants


def build_scs_name_map(
    scs_names: list[str],
    known_players: pd.DataFrame,          # columns: player_name (or p1_name)
    cache_path: Path,
) -> dict[str, str]:
    """
    Map Sofascore player names → internal player names.
    Uses fuzzy matching and caches result to JSON.
    Returns: {scs_name → internal_name}
    """
    # Load existing cache
    mapping: dict[str, str] = {}
    if cache_path.exists():
        mapping = json.loads(cache_path.read_text())

    name_col = "player_name" if "player_name" in known_players.columns else "p1_name"
    internal_names = known_players[name_col].dropna().unique().tolist()
    internal_variants: dict[str, str] = {}   # variant → internal_name
    for iname in internal_names:
        for v in _name_variants(iname):
            internal_variants[v] = iname

    new_mappings = 0
    for scs_name in scs_names:
        if scs_name in mapping:
            continue
        for v in _name_variants(scs_name):
            if v in internal_variants:
                mapping[scs_name] = internal_variants[v]
                new_mappings += 1
                break

    if new_mappings:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False))

    return mapping


def resolve_player_ids(
    df_matches: pd.DataFrame,
    known_players: pd.DataFrame,
    cache_path: Path,
) -> pd.DataFrame:
    """Replace Sofascore names/IDs with internal player names."""
    all_names = list(set(df_matches["winner_name"].tolist() + df_matches["loser_name"].tolist()))
    name_map  = build_scs_name_map(all_names, known_players, cache_path)

    df = df_matches.copy()
    df["winner_name"] = df["winner_name"].map(lambda n: name_map.get(n, n))
    df["loser_name"]  = df["loser_name"].map(lambda n: name_map.get(n, n))
    return df, name_map
```

- [ ] **3.2 — Test name resolution**

```bash
PYTHONIOENCODING=utf-8 python -c "
import sys, pandas as pd
sys.path.insert(0, 'src')
from pathlib import Path
from datetime import date, timedelta
from sofascore_client import fetch_date_range
from fetch_sofascore import convert_events_to_matches, resolve_player_ids

ROOT = Path('.')
events = fetch_date_range(date.today() - timedelta(days=7), date.today(), 'atp')
df = convert_events_to_matches(events, 'atp')

# Load known players from existing parquet
pq = ROOT / 'data/processed/atp/player_profiles_updated.parquet'
if not pq.exists():
    pq = ROOT / 'data/processed/atp/matches_features_final.parquet'
known = pd.read_parquet(pq, columns=['player_name'] if 'player_name' in pd.read_parquet(pq, columns=[]).columns else ['p1_name'])

cache_path = ROOT / 'data/cache/sofascore_player_map_atp.json'
df_resolved, name_map = resolve_player_ids(df, known, cache_path)

matched = df_resolved['winner_name'].isin(known.iloc[:,0]) | df_resolved['loser_name'].isin(known.iloc[:,0])
print(f'Total matches: {len(df_resolved)}')
print(f'At least one player matched: {matched.sum()} ({matched.mean():.0%})')
print(f'Name map size: {len(name_map)}')
"
```

Expected: ≥ 70% of matches have at least one matched player. Lower means tournament has many new players (qualifiers) which is normal.

---

## Task 4 — Incremental Profile Update (Main Orchestration)

**Files:**
- Modify: `src/fetch_sofascore.py` (add `run_update()` main function)

This is the core: take the converted+resolved matches, and call `update_database.py`'s logic to update ELO + player profiles. We import directly from `update_database.py`.

**Key functions we'll import from `update_database.py`:**
- `update_elo(new_matches_df, elo_dict, k_map)` → updated elo_dict
- `apply_elo_decay(elo_dict, cutoff_date)` → updated elo_dict
- `update_player_profiles(new_matches_df, all_history_df, elo_dict)` → profiles_df

- [ ] **4.1 — Write `run_update()` orchestration function**

```python
# Append to src/fetch_sofascore.py — main orchestration

def run_update(tour: str, days_back: int = 14, dry_run: bool = False) -> None:
    """
    Fetch last `days_back` days from Sofascore and update player profiles.
    If dry_run=True, print stats but don't write any files.
    """
    cfg   = get_tour_config(tour)
    paths = get_paths(tour)
    processed_dir = paths["processed_dir"]
    models_dir    = paths["models_dir"]

    print(f"\n{'='*55}")
    print(f"SOFASCORE UPDATE — {tour.upper()}")
    print(f"{'='*55}")

    # ── 1. Fetch from Sofascore ───────────────────────────────
    end   = date.today()
    start = end - timedelta(days=days_back)
    print(f"\n  Fetching {tour.upper()} events {start} → {end}...")

    events = fetch_date_range(start, end, tour_filter=tour)
    print(f"  Found {len(events)} finished events")

    if not events:
        print("  Nothing to update.")
        return

    # ── 2. Convert to internal format ─────────────────────────
    df_new = convert_events_to_matches(events, tour)
    print(f"  Converted: {len(df_new)} matches")

    # ── 3. Resolve player names ───────────────────────────────
    profiles_path = processed_dir / "player_profiles_updated.parquet"
    fallback_path = processed_dir / "matches_features_final.parquet"
    if profiles_path.exists():
        known = pd.read_parquet(profiles_path)
    elif fallback_path.exists():
        known = pd.read_parquet(fallback_path, columns=["p1_name"]).rename(columns={"p1_name": "player_name"})
    else:
        print("  ERROR: No player database found. Run full pipeline first.")
        return

    cache_path = ROOT / "data" / "cache" / f"sofascore_player_map_{tour}.json"
    df_new, name_map = resolve_player_ids(df_new, known, cache_path)
    print(f"  Name map: {len(name_map)} entries")

    # ── 4. Import and call update_database logic ──────────────
    from update_database import (
        update_elo,
        apply_elo_decay,
        update_player_profiles,
    )

    elo_path = processed_dir / "elo_ratings_updated.parquet"
    if not elo_path.exists():
        elo_path = processed_dir / "elo_ratings_final.parquet"

    if not elo_path.exists():
        print("  ERROR: No ELO snapshot found. Run compute_elo.py first.")
        return

    elo_df = pd.read_parquet(elo_path)

    # Rebuild elo dicts from parquet snapshot
    elo_global  = dict(zip(elo_df["full_name"], elo_df["elo_global"]))
    elo_surface = {
        name: {
            "Hard":   row.get("elo_Hard",   1500),
            "Clay":   row.get("elo_Clay",   1500),
            "Grass":  row.get("elo_Grass",  1500),
            "Carpet": row.get("elo_Carpet", 1500),
        }
        for name, row in elo_df.set_index("full_name").iterrows()
    }

    print(f"\n  Applying incremental ELO update ({len(df_new)} matches)...")
    elo_global, elo_surface = update_elo(
        df_new, elo_global, elo_surface, cfg["k_factor_map"]
    )
    elo_global, elo_surface = apply_elo_decay(elo_global, elo_surface, date.today())

    # ── 5. Load full history for rolling features ─────────────
    hist_path = processed_dir / "matches_features_final.parquet"
    print(f"\n  Loading history for rolling features...")
    df_hist = pd.read_parquet(hist_path)

    print(f"\n  Updating player profiles...")
    df_profiles = update_player_profiles(
        df_new, df_hist, elo_global, elo_surface, tour
    )
    print(f"  Profiles updated: {len(df_profiles)} players")

    if dry_run:
        print("\n  [DRY RUN] — no files written")
        print(df_profiles.head(5).to_string())
        return

    # ── 6. Save outputs ───────────────────────────────────────
    df_profiles.to_parquet(processed_dir / "player_profiles_updated.parquet", index=False)

    # Save updated ELO snapshot
    elo_records = [
        {
            "full_name":  name,
            "elo_global": elo_global.get(name, 1500),
            "elo_Hard":   elo_surface.get(name, {}).get("Hard",   1500),
            "elo_Clay":   elo_surface.get(name, {}).get("Clay",   1500),
            "elo_Grass":  elo_surface.get(name, {}).get("Grass",  1500),
            "elo_Carpet": elo_surface.get(name, {}).get("Carpet", 1500),
        }
        for name in set(elo_global) | set(elo_surface)
    ]
    pd.DataFrame(elo_records).to_parquet(
        processed_dir / "elo_ratings_updated.parquet", index=False
    )

    # Append new matches to consolidated parquet (for future full retraining)
    consolidated_path = processed_dir / "matches_consolidated.parquet"
    if consolidated_path.exists():
        df_existing = pd.read_parquet(consolidated_path)
        # Deduplicate by sofascore_id to avoid double-inserting
        existing_ids = set(df_existing.get("sofascore_id", pd.Series(dtype=object)).dropna())
        df_truly_new = df_new[~df_new["sofascore_id"].isin(existing_ids)]
        if not df_truly_new.empty:
            df_combined = pd.concat([df_existing, df_truly_new], ignore_index=True)
            df_combined.to_parquet(consolidated_path, index=False)
            print(f"\n  Appended {len(df_truly_new)} new matches to consolidated parquet")

    print(f"\n  Saved → {processed_dir}/player_profiles_updated.parquet")
    print(f"  Done.")
```

- [ ] **4.2 — Add `__main__` entry point**

```python
# Append to src/fetch_sofascore.py

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch & integrate Sofascore results")
    parser.add_argument("--tour",     default="atp", choices=["atp", "wta"])
    parser.add_argument("--days",     default=14,    type=int,
                        help="Number of days back to fetch (default: 14)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Fetch and convert without writing files")
    args = parser.parse_args()

    run_update(tour=args.tour, days_back=args.days, dry_run=args.dry_run)
```

- [ ] **4.3 — End-to-end test with `--dry-run`**

```bash
cd E:/Claude/botbet/tennis/tennis_ml
source venv/Scripts/activate
PYTHONIOENCODING=utf-8 python src/fetch_sofascore.py --tour atp --days 14 --dry-run
```

Expected output:
```
=======================================================
SOFASCORE UPDATE — ATP
=======================================================
  Fetching ATP events 2026-03-01 → 2026-03-15...
  Found N finished events
  Converted: N matches
  Name map: N entries
  Applying incremental ELO update (N matches)...
  Updating player profiles...
  Profiles updated: N players
  [DRY RUN] — no files written
  [table of sample profiles]
```

- [ ] **4.4 — Full run (writes files)**

```bash
PYTHONIOENCODING=utf-8 python src/fetch_sofascore.py --tour atp --days 14
PYTHONIOENCODING=utf-8 python src/fetch_sofascore.py --tour wta --days 14
```

Verify:
```bash
PYTHONIOENCODING=utf-8 python -c "
import pandas as pd
df = pd.read_parquet('data/processed/atp/player_profiles_updated.parquet')
print(f'Profiles: {len(df)} players')
print(df.sort_values('elo', ascending=False).head(5)[['player_name','elo','winrate_10','last_match']].to_string())
"
```

---

## Task 5 — Dashboard Integration

**Files:**
- Modify: `src/dashboard.py` (add Sofascore sync button in sidebar)

Add a "Sync Sofascore" section to the existing "Mise à jour données" sidebar area. After sync, clear cache and rerun.

- [ ] **5.1 — Add sync function and buttons in sidebar**

In `src/dashboard.py`, find the `run_update()` function definition (around line 1023) and add a new `run_sofascore_sync()` function. Then add the button below the existing ATP/WTA update buttons.

Find this block:
```python
    col_atp_btn, col_wta_btn = st.columns(2)
    if col_atp_btn.button("🔄 ATP", use_container_width=True, type="primary"):
        run_update("atp")
    if col_wta_btn.button("🔄 WTA", use_container_width=True, type="primary"):
        run_update("wta")
```

Add after it:
```python
    st.html("""
    <div style="font-family:'DM Mono',monospace; font-size:0.58rem; letter-spacing:2px;
                text-transform:uppercase; color:#2A3D5A; margin:14px 0 8px; padding-left:2px;">
      Sofascore Live
    </div>
    """)

    def run_sofascore_sync(tour: str):
        import subprocess, sys
        with st.spinner(f"Sync Sofascore {tour.upper()}..."):
            result = subprocess.run(
                [sys.executable, str(ROOT / "src" / "fetch_sofascore.py"),
                 "--tour", tour, "--days", "14"],
                capture_output=True, text=True, encoding="utf-8", cwd=str(ROOT),
            )
        if result.returncode == 0:
            st.cache_resource.clear()
            st.cache_data.clear()
            st.success(f"Sofascore {tour.upper()} synced !")
            st.rerun()
        else:
            st.error(f"Erreur Sofascore {tour.upper()}")
            st.code(result.stderr[-2000:] if result.stderr else result.stdout[-2000:])

    col_s_atp, col_s_wta = st.columns(2)
    if col_s_atp.button("⚡ ATP", key="scs_atp", use_container_width=True):
        run_sofascore_sync("atp")
    if col_s_wta.button("⚡ WTA", key="scs_wta", use_container_width=True):
        run_sofascore_sync("wta")
```

- [ ] **5.2 — Verify dashboard loads without error**

```bash
PYTHONIOENCODING=utf-8 python -c "
import ast
ast.parse(open('src/dashboard.py', encoding='utf-8').read())
print('Syntax OK')
"
```

---

## Task 6 — Scheduled Task (Windows)

**Files:**
- Create: `src/scheduled_sync.py` (wrapper script that syncs both ATP and WTA)

- [ ] **6.1 — Create `src/scheduled_sync.py`**

```python
# src/scheduled_sync.py
"""
Wrapper appelé par le planificateur Windows chaque matin.
Synchonise ATP + WTA depuis Sofascore et log le résultat.
"""
import sys
import logging
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "logs" / "sofascore_sync.log"
LOG_PATH.parent.mkdir(exist_ok=True)

logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

sys.path.insert(0, str(ROOT / "src"))

from fetch_sofascore import run_update

for tour in ["atp", "wta"]:
    try:
        logging.info(f"Starting {tour.upper()} sync")
        run_update(tour=tour, days_back=14)
        logging.info(f"Done {tour.upper()} sync")
    except Exception as e:
        logging.error(f"{tour.upper()} sync failed: {e}", exc_info=True)
```

- [ ] **6.2 — Register Windows scheduled task**

```bash
# Run once to create the task (Windows Task Scheduler via schtasks)
VENV_PYTHON=$(cygpath -w "$(which python)")
SCRIPT=$(cygpath -w "E:/Claude/botbet/tennis/tennis_ml/src/scheduled_sync.py")

schtasks /Create \
  /TN "TennisML_SofascoreSync" \
  /TR "\"$VENV_PYTHON\" \"$SCRIPT\"" \
  /SC DAILY \
  /ST 07:00 \
  /F \
  /RL HIGHEST
```

- [ ] **6.3 — Verify task is registered**

```bash
schtasks /Query /TN "TennisML_SofascoreSync" /FO LIST
```

Expected: task listed with `Next Run Time: 07:00:00` tomorrow.

- [ ] **6.4 — Test manual trigger**

```bash
schtasks /Run /TN "TennisML_SofascoreSync"
# Wait 30 seconds then check log:
tail -30 E:/Claude/botbet/tennis/tennis_ml/logs/sofascore_sync.log
```

---

## Potential Issues & Mitigations

| Issue | Mitigation |
|---|---|
| Sofascore API changes/breaks | Graceful error handling + fallback to last known profiles |
| Player name not matched | Log unmatched names; keep existing profile for that player |
| Sofascore uses "Last First" order | `_name_variants()` tries both orderings |
| Rate limiting (HTTP 429) | Exponential backoff in `_get()`, 0.5s delay between dates |
| `update_database.py` internal API changes | Pin function signatures with explicit imports; document in code |
| No matches during off-weeks | Script exits cleanly with "Nothing to update" |
| Double-insert on re-run | Deduplicate by `sofascore_id` before appending to consolidated parquet |

---

## Definition of Done

- [ ] `fetch_sofascore.py --tour atp --dry-run` runs without error and shows real match data
- [ ] `fetch_sofascore.py --tour atp` writes updated `player_profiles_updated.parquet`
- [ ] At least 70% of active ATP top-100 players have `last_match` within 7 days
- [ ] Dashboard sync buttons work and trigger a `st.rerun()`
- [ ] Scheduled task registered and runs successfully via manual trigger
- [ ] Log file written to `logs/sofascore_sync.log`
