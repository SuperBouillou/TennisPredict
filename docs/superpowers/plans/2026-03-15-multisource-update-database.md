# Multi-Source Update Database Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `update_database.py` to layer three data sources (Sackmann historical base → tennis-data.co.uk → ESPN recent) so player profiles are always as fresh as possible, covering the gap between the last tennis-data file (~1–3 weeks ago) and today.

**Architecture:** Source 1 (Sackmann ELO base) is already loaded. Source 2 (tennis-data.co.uk XLSX) is the existing pipeline — no change. Source 3 (ESPN `fetch_recent`) is appended after Source 2 by deduplicating on `(date, frozenset({last_name_winner, last_name_loser}))` to handle the different name formats (tennis-data: "Sinner J.", ESPN: "Jannik Sinner"). The merged dataset feeds the existing ELO update and profile computation unchanged.

**Tech Stack:** Python 3.10+, pandas, `espn_client.fetch_recent()` (already in repo), `update_database.py`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/update_database.py` | Modify | Add `_extract_last`, `_make_dedup_key`, `fetch_espn_complement`; wire into `main()` |
| `tests/test_update_database_multisource.py` | Create | Unit tests for dedup utilities and ESPN complement logic |

---

## Chunk 1: Deduplication utilities + ESPN complement function

### Task 1: Deduplication helper functions

**Files:**
- Modify: `src/update_database.py` (add helpers after `build_name_mapping`)
- Create: `tests/test_update_database_multisource.py`

- [ ] **Step 1: Write the failing tests for `_extract_last` and `_make_dedup_key`**

Create `tests/test_update_database_multisource.py`:

```python
"""Tests for multi-source deduplication utilities in update_database."""
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from update_database import _extract_last, _make_dedup_key


class TestExtractLast:
    """_extract_last normalises player names from any source to a comparable last name."""

    def test_tennis_data_format_simple(self):
        """'Sinner J.' → 'sinner'"""
        assert _extract_last("Sinner J.") == "sinner"

    def test_tennis_data_format_compound(self):
        """'Mpetshi Perricard G.' → 'perricard'"""
        assert _extract_last("Mpetshi Perricard G.") == "perricard"

    def test_tennis_data_format_double_initial(self):
        """'Djokovic N.' → 'djokovic'"""
        assert _extract_last("Djokovic N.") == "djokovic"

    def test_espn_format_simple(self):
        """'Jannik Sinner' → 'sinner'"""
        assert _extract_last("Jannik Sinner") == "sinner"

    def test_espn_format_compound_last(self):
        """'Giovanni Mpetshi Perricard' → 'perricard'"""
        assert _extract_last("Giovanni Mpetshi Perricard") == "perricard"

    def test_espn_format_accented(self):
        """'Novak Djokovic' → 'djokovic'"""
        assert _extract_last("Novak Djokovic") == "djokovic"

    def test_empty_string(self):
        assert _extract_last("") == ""

    def test_single_word(self):
        """Single-token name returns itself normalised."""
        assert _extract_last("Sinner") == "sinner"


class TestMakeDedupKey:
    """_make_dedup_key returns (date, frozenset({last_w, last_l})) or None."""

    def test_basic_key(self):
        d = pd.Timestamp("2026-03-10")
        key = _make_dedup_key(d, "Sinner J.", "Djokovic N.")
        assert key == (d.date(), frozenset(["sinner", "djokovic"]))

    def test_cross_format_same_key(self):
        """tennis-data name and ESPN name produce the same dedup key."""
        d = pd.Timestamp("2026-03-10")
        key_td  = _make_dedup_key(d, "Sinner J.", "Djokovic N.")
        key_espn = _make_dedup_key(d, "Jannik Sinner", "Novak Djokovic")
        assert key_td == key_espn

    def test_order_independent(self):
        """Swapping winner/loser gives the same key (frozenset)."""
        d = pd.Timestamp("2026-03-10")
        k1 = _make_dedup_key(d, "Sinner J.", "Djokovic N.")
        k2 = _make_dedup_key(d, "Djokovic N.", "Sinner J.")
        assert k1 == k2

    def test_bad_date_returns_none(self):
        key = _make_dedup_key(None, "Sinner J.", "Djokovic N.")
        assert key is None

    def test_empty_names_return_none(self):
        d = pd.Timestamp("2026-03-10")
        assert _make_dedup_key(d, "", "Djokovic N.") is None
        assert _make_dedup_key(d, "Sinner J.", "") is None
```

- [ ] **Step 2: Run tests to verify they fail (functions not yet defined)**

```bash
cd E:\Claude\botbet\tennis\tennis_ml
venv\Scripts\activate && python -m pytest tests/test_update_database_multisource.py -v 2>&1 | head -40
```

Expected: `ImportError` or `AttributeError` — `_extract_last` not found.

- [ ] **Step 3: Implement `_extract_last` and `_make_dedup_key` in `update_database.py`**

Add after the `build_name_mapping` function (around line 466), before the `main()` block:

```python
# ─────────────────────────────────────────────────────────────────────────────
# MULTI-SOURCE DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_last(name: str) -> str:
    """
    Extracts a normalised last name from any source format:
      - tennis-data format  : "Sinner J."          → "sinner"
      - ESPN format         : "Jannik Sinner"       → "sinner"
      - Compound last name  : "Mpetshi Perricard G." → "perricard"
    Returns "" for empty/unrecognisable input.
    """
    if not name or not name.strip():
        return ""
    parts = name.strip().split()
    if len(parts) == 1:
        return _norm(parts[0])

    # tennis-data format: last token is initials (e.g. "J.", "J.L.")
    if _is_initials(parts[-1]):
        non_init = [p for p in parts[:-1] if not _is_initials(p)]
        candidates = non_init if non_init else parts[:-1]
        return _norm(candidates[-1])

    # ESPN / Sackmann format: last token is the surname
    return _norm(parts[-1])


def _make_dedup_key(tourney_date, winner: str, loser: str):
    """
    Returns a hashable dedup key: (date, frozenset({last_winner, last_loser})).
    Returns None if date is invalid or either name is empty.
    """
    try:
        d = pd.Timestamp(tourney_date).date()
    except Exception:
        return None
    w_last = _extract_last(winner)
    l_last = _extract_last(loser)
    if not w_last or not l_last:
        return None
    return (d, frozenset([w_last, l_last]))
```

- [ ] **Step 4: Run tests — all should pass**

```bash
python -m pytest tests/test_update_database_multisource.py::TestExtractLast tests/test_update_database_multisource.py::TestMakeDedupKey -v
```

Expected: All 13 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/update_database.py tests/test_update_database_multisource.py
git commit -m "feat: add _extract_last and _make_dedup_key dedup utilities for multi-source pipeline"
```

---

### Task 2: `fetch_espn_complement` function

**Files:**
- Modify: `src/update_database.py` (add `fetch_espn_complement` after dedup helpers)
- Modify: `tests/test_update_database_multisource.py` (add tests for complement logic)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_update_database_multisource.py`:

```python
class TestFetchEspnComplement:
    """fetch_espn_complement returns only ESPN matches absent from the tennis-data DataFrame."""

    def _make_td_df(self, rows):
        """Build a tennis-data-format DataFrame."""
        return pd.DataFrame(rows)

    def _make_espn_df(self, rows):
        """Build an ESPN-format DataFrame (already Sackmann-compatible)."""
        return pd.DataFrame(rows)

    def test_returns_only_new_matches(self, monkeypatch):
        """ESPN match not in tennis-data → returned."""
        td_df = self._make_td_df([{
            "tourney_date": pd.Timestamp("2026-03-10"),
            "winner_name": "Sinner J.",
            "loser_name": "Djokovic N.",
        }])
        espn_df = self._make_espn_df([
            {
                "tourney_date": pd.Timestamp("2026-03-10"),
                "winner_name": "Jannik Sinner",   # same match — already in td
                "loser_name": "Novak Djokovic",
                "surface": "Hard", "tourney_level": "M",
                "tourney_name": "Indian Wells", "round": "QF",
                "best_of": 3, "winner_rank": None,
                "loser_rank": None, "winner_rank_pts": None,
                "loser_rank_pts": None, "source": "espn",
            },
            {
                "tourney_date": pd.Timestamp("2026-03-12"),
                "winner_name": "Carlos Alcaraz",   # NEW match
                "loser_name": "Daniil Medvedev",
                "surface": "Hard", "tourney_level": "M",
                "tourney_name": "Indian Wells", "round": "SF",
                "best_of": 3, "winner_rank": None,
                "loser_rank": None, "winner_rank_pts": None,
                "loser_rank_pts": None, "source": "espn",
            },
        ])

        import update_database as udb
        monkeypatch.setattr("update_database.fetch_recent", lambda tour, days: espn_df)

        result = udb.fetch_espn_complement(td_df, "atp", days=14)
        assert len(result) == 1
        assert result.iloc[0]["winner_name"] == "Carlos Alcaraz"

    def test_empty_espn_returns_empty(self, monkeypatch):
        """If ESPN returns no data → empty DataFrame."""
        td_df = self._make_td_df([{
            "tourney_date": pd.Timestamp("2026-03-10"),
            "winner_name": "Sinner J.", "loser_name": "Djokovic N.",
        }])
        import update_database as udb
        monkeypatch.setattr("update_database.fetch_recent", lambda tour, days: pd.DataFrame())
        result = udb.fetch_espn_complement(td_df, "atp", days=14)
        assert result.empty

    def test_empty_td_returns_all_espn(self, monkeypatch):
        """If tennis-data is empty → all ESPN matches are new."""
        espn_df = self._make_espn_df([{
            "tourney_date": pd.Timestamp("2026-03-14"),
            "winner_name": "Carlos Alcaraz", "loser_name": "Daniil Medvedev",
            "surface": "Hard", "tourney_level": "M", "tourney_name": "Indian Wells",
            "round": "Final", "best_of": 3, "winner_rank": None, "loser_rank": None,
            "winner_rank_pts": None, "loser_rank_pts": None, "source": "espn",
        }])
        import update_database as udb
        monkeypatch.setattr("update_database.fetch_recent", lambda tour, days: espn_df)
        result = udb.fetch_espn_complement(pd.DataFrame(), "atp", days=14)
        assert len(result) == 1

    def test_source_column_is_espn(self, monkeypatch):
        """Returned rows have source='espn'."""
        espn_df = pd.DataFrame([{
            "tourney_date": pd.Timestamp("2026-03-14"),
            "winner_name": "Carlos Alcaraz", "loser_name": "Daniil Medvedev",
            "surface": "Hard", "tourney_level": "M", "tourney_name": "Indian Wells",
            "round": "Final", "best_of": 3, "winner_rank": None, "loser_rank": None,
            "winner_rank_pts": None, "loser_rank_pts": None, "source": "espn",
        }])
        import update_database as udb
        monkeypatch.setattr("update_database.fetch_recent", lambda tour, days: espn_df)
        result = udb.fetch_espn_complement(pd.DataFrame(), "atp", days=14)
        assert result.iloc[0]["source"] == "espn"
```

- [ ] **Step 2: Run tests — confirm failure**

```bash
python -m pytest tests/test_update_database_multisource.py::TestFetchEspnComplement -v 2>&1 | head -20
```

Expected: `AttributeError: module 'update_database' has no attribute 'fetch_espn_complement'`

- [ ] **Step 3: Implement `fetch_espn_complement` in `update_database.py`**

Add the following import at the top of `update_database.py` (after existing imports):

```python
from espn_client import fetch_recent
```

Then add `fetch_espn_complement` right after `_make_dedup_key`:

```python
def fetch_espn_complement(df_td: pd.DataFrame, tour: str, days: int = 21) -> pd.DataFrame:
    """
    Fetches ESPN completed matches for the last `days` days and returns
    only matches NOT already present in df_td (deduplicated by date + last names).

    ESPN names ("Jannik Sinner") and tennis-data names ("Sinner J.") are compared
    via _extract_last() so cross-format duplicates are correctly detected.

    Returns an empty DataFrame if ESPN is unavailable or all matches are duplicates.
    """
    print(f"\n── Source 3 : ESPN (derniers {days} jours) ──────────────")
    df_espn = fetch_recent(tour, days=days)
    if df_espn.empty:
        print("  ESPN : aucun match retourné")
        return pd.DataFrame()

    # Build dedup set from tennis-data (or empty set if df_td is empty)
    td_keys: set = set()
    if not df_td.empty:
        for _, row in df_td.iterrows():
            key = _make_dedup_key(row["tourney_date"], row["winner_name"], row["loser_name"])
            if key:
                td_keys.add(key)

    # Keep only ESPN rows absent from tennis-data
    new_rows = []
    for _, row in df_espn.iterrows():
        key = _make_dedup_key(row["tourney_date"], row["winner_name"], row["loser_name"])
        if key and key not in td_keys:
            new_rows.append(row)

    if not new_rows:
        print("  ESPN : 0 nouveau match (tous deja dans tennis-data)")
        return pd.DataFrame()

    df_complement = pd.DataFrame(new_rows).reset_index(drop=True)
    print(f"  ESPN : {len(df_complement)} nouveaux matchs "
          f"({df_complement['tourney_date'].min().date()} "
          f"→ {df_complement['tourney_date'].max().date()})")
    return df_complement
```

- [ ] **Step 4: Run tests — all should pass**

```bash
python -m pytest tests/test_update_database_multisource.py -v
```

Expected: All 17 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/update_database.py tests/test_update_database_multisource.py
git commit -m "feat: add fetch_espn_complement for ESPN-based data gap filling"
```

---

## Chunk 2: Wire ESPN complement into `main()` and update coverage reporting

### Task 3: Integrate multi-source pipeline in `main()`

**Files:**
- Modify: `src/update_database.py` (main function only, lines ~522–600)

- [ ] **Step 1: Locate the integration point in `main()`**

Find the two blocks to replace, starting around line 524:

```python
    # OLD BLOCK 1 — guard (to replace):
    if df_raw.empty:
        print(f"❌ Aucune donnée trouvée. Vérifier {ODDS_DIR}")
        return

    # OLD BLOCK 2 — convert + print (also to replace):
    df_new = convert_to_sackmann_format(df_raw, cfg)
    print(f"  Matchs convertis : {len(df_new):,}")
    print(f"  Période          : {df_new['tourney_date'].min().date()} "
          f"→ {df_new['tourney_date'].max().date()}")
```

- [ ] **Step 2: Replace BOTH blocks (guard + convert/print) and everything that follows in `main()` until the end**

Replace from the `if df_raw.empty:` guard all the way through the final `print` statements. The complete new block covers the WTA case (empty `df_raw`) by relaxing the guard before calling ESPN:

```python
    if df_raw.empty:
        print("  tennis-data : aucun fichier disponible — ESPN sera la seule source")
        df_new = pd.DataFrame()
    else:
        df_new = convert_to_sackmann_format(df_raw, cfg)

    n_td = len(df_new)
    if n_td:
        print(f"  tennis-data : {n_td:,} matchs "
              f"({df_new['tourney_date'].min().date()} "
              f"→ {df_new['tourney_date'].max().date()})")
    else:
        print("  tennis-data : aucun match chargé (WTA ou source manquante)")

    # ── Source 3 : ESPN — complément récent ──────────────────────────────────
    df_espn = fetch_espn_complement(df_new, tour, days=21)
    n_espn  = len(df_espn)

    # Fusionner les deux sources, trier chronologiquement
    if not df_espn.empty:
        df_combined = (
            pd.concat([df_new, df_espn], ignore_index=True)
            .sort_values("tourney_date")
            .reset_index(drop=True)
        )
    else:
        df_combined = df_new.copy()

    n_combined = len(df_combined)
    if df_combined.empty:
        print(f"❌ Aucune donnée trouvée. Vérifier {ODDS_DIR} et la connectivité ESPN")
        return

    print(f"\n── Couverture combinée ──────────────────────────────")
    print(f"  tennis-data : {n_td:,} matchs")
    print(f"  ESPN        : {n_espn:,} matchs nouveaux")
    print(f"  Total       : {n_combined:,} matchs")
    print(f"  Période     : {df_combined['tourney_date'].min().date()} "
          f"→ {df_combined['tourney_date'].max().date()}")

    # ── Mapping noms tennis-data → noms Sackmann ──────────────────────────────
    print("\n── Mapping noms joueurs ─────────────────────────────")
    name_mapping = build_name_mapping(df_combined, elo_ratings)

    # Pour update_elo : noms traduits en noms Sackmann
    df_elo_input = df_combined.copy()
    df_elo_input["winner_name"] = df_elo_input["winner_name"].map(name_mapping)
    df_elo_input["loser_name"]  = df_elo_input["loser_name"].map(name_mapping)

    # ── Mise à jour ELO ───────────────────────────────────────────────────────
    print("\n── Mise à jour ELO ──────────────────────────────────")
    elo_ratings, elo_surface = update_elo(df_elo_input, elo_ratings, elo_surface, cfg)

    # ── ELO decay joueurs inactifs ────────────────────────────────────────────
    print("\n── ELO decay joueurs inactifs ───────────────────────")
    active_players = set(df_elo_input["winner_name"].tolist() +
                         df_elo_input["loser_name"].tolist())
    elo_ratings, elo_surface = apply_elo_decay(
        elo_ratings, elo_surface, active_players, datetime.now()
    )

    # ── Calcul profils joueurs ────────────────────────────────────────────────
    print("\n── Calcul profils joueurs ───────────────────────────")
    # df_combined garde les noms originaux par source
    # name_mapping permet de récupérer le bon ELO dans elo_ratings (clés Sackmann)
    df_profiles = update_player_profiles(df_combined, elo_ratings, elo_surface, name_mapping)

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    print("\n── Sauvegarde ───────────────────────────────────────")

    profiles_path = PROCESSED_DIR / "player_profiles_updated.parquet"
    df_profiles.to_parquet(profiles_path, index=False)
    print(f"  ✅ Profils : {profiles_path.name}")

    elo_updated = []
    for name, elo_g in elo_ratings.items():
        surf = elo_surface.get(name, {})
        elo_updated.append({
            "full_name"  : name,
            "elo_global" : elo_g,
            "elo_Hard"   : surf.get("Hard",   1500),
            "elo_Clay"   : surf.get("Clay",   1500),
            "elo_Grass"  : surf.get("Grass",  1500),
            "elo_Carpet" : surf.get("Carpet", 1500),
        })
    df_elo_updated = pd.DataFrame(elo_updated)
    elo_updated_path = PROCESSED_DIR / "elo_ratings_updated.parquet"
    df_elo_updated.to_parquet(elo_updated_path, index=False)
    print(f"  ✅ ELO    : {elo_updated_path.name} ({len(df_elo_updated):,} joueurs)")

    print("\n── Top 10 ELO Global (mis à jour) ───────────────────")
    top10 = df_elo_updated.nlargest(10, "elo_global")[
        ["full_name", "elo_global", "elo_Hard", "elo_Clay"]
    ]
    for _, row in top10.iterrows():
        print(f"  {row['full_name']:<25} ELO={row['elo_global']:.0f} "
              f"Hard={row['elo_Hard']:.0f} Clay={row['elo_Clay']:.0f}")

    print(f"\n── Résumé ───────────────────────────────────────────")
    print(f"  Sources            : tennis-data ({n_td}) + ESPN ({n_espn})")
    print(f"  Matchs intégrés    : {n_combined:,}")
    print(f"  Joueurs mis à jour : {len(df_profiles):,}")
    print(f"  Dernière date      : {df_combined['tourney_date'].max().date()}")
    print(f"\n✅ Base joueurs à jour — relancer predict_today.py")
```

- [ ] **Step 3: Verify the full script runs without crash (smoke test)**

```bash
cd E:\Claude\botbet\tennis\tennis_ml
venv\Scripts\activate && python src/update_database.py --tour atp 2>&1 | tail -30
```

Expected output includes:
```
── Source 3 : ESPN (derniers 21 jours) ──────────────
    ESPN ATP fetch@...: N matchs (raw)
  ESPN : X nouveaux matchs (2026-03-01 → 2026-03-15)
── Couverture combinée ──────────────────────────────
  tennis-data : 3035 matchs
  ESPN        : X matchs nouveaux
  Total       : 30XX matchs
  Période     : 2025-01-01 → 2026-03-15
...
✅ Base joueurs à jour — relancer predict_today.py
```

- [ ] **Step 4: Run unit tests to confirm nothing broke**

```bash
python -m pytest tests/test_update_database_multisource.py -v
```

Expected: All 17 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/update_database.py
git commit -m "feat: wire ESPN complement source into update_database main() pipeline"
```

---

### Task 4: Verify WTA also benefits from ESPN

**Files:**
- Modify: none (just verification)

- [ ] **Step 1: Run for WTA tour**

```bash
python src/update_database.py --tour wta 2>&1 | tail -25
```

Expected:
- tennis-data source is empty (WTA XLSX doesn't exist on tennis-data.co.uk)
- ESPN provides recent WTA matches
- Output: `ESPN : X nouveaux matchs` (WTA women's singles)
- Script completes without error

- [ ] **Step 2: Confirm output parquet is updated**

```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/wta/player_profiles_updated.parquet')
print(f'WTA profiles: {len(df)} players')
print(f'Last match  : {df[\"last_match\"].max()}')
"
```

Expected: prints a recent date (within last 14–21 days).

- [ ] **Step 3: Commit (only if WTA parquet was not previously generated)**

```bash
git add -A
git commit -m "verify: WTA multi-source pipeline working via ESPN"
```

---

## Coverage Summary

After implementation, `update_database.py --tour atp` will:
1. Load ELO base (Sackmann historical)
2. Download + parse tennis-data XLSX → covers up to ~1–3 weeks ago
3. Fetch ESPN last 21 days → fills the remaining gap to today
4. Deduplicate by `(date, frozenset({last_name_w, last_name_l}))`
5. Feed merged dataset into existing ELO update + profile computation
6. Report per-source match counts in summary

For WTA: step 2 produces an empty DataFrame; ESPN becomes the sole recent source.
