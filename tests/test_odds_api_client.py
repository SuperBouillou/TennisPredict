# tests/test_odds_api_client.py
"""
Unit tests for src/odds_api_client.py
Run: cd E:/Claude/botbet/tennis/tennis_ml && venv/Scripts/pytest tests/test_odds_api_client.py -v
"""
import json
import sys
from pathlib import Path
from datetime import date

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from odds_api_client import (
    normalize_player_name,
    merge_odds,
    _extract_pinnacle_odds,
    _extract_avg_odds,
    _load_cache,
    _save_cache,
    OddsResult,
)


# ── normalize_player_name ─────────────────────────────────────────────────────

def test_normalize_strips_accents():
    assert normalize_player_name("Björn Borg") == "bjorn borg"

def test_normalize_lowercases():
    assert normalize_player_name("JANNIK SINNER") == "jannik sinner"

def test_normalize_strips_whitespace():
    assert normalize_player_name("  Carlos Alcaraz  ") == "carlos alcaraz"

def test_normalize_mixed():
    assert normalize_player_name("Novak Đoković") == "novak dokovic"


# ── merge_odds ────────────────────────────────────────────────────────────────

def test_merge_forward_match():
    matches = [{"p1_name": "Jannik Sinner", "p2_name": "Carlos Alcaraz"}]
    odds    = {"jannik sinner vs carlos alcaraz": (1.85, 2.05)}
    result  = merge_odds(matches, odds)
    assert result[0]["odd_p1"] == 1.85
    assert result[0]["odd_p2"] == 2.05

def test_merge_reversed_match():
    """API may return home/away in reversed order — odds must be swapped."""
    matches = [{"p1_name": "Jannik Sinner", "p2_name": "Carlos Alcaraz"}]
    odds    = {"carlos alcaraz vs jannik sinner": (2.05, 1.85)}
    result  = merge_odds(matches, odds)
    assert result[0]["odd_p1"] == 1.85   # swapped
    assert result[0]["odd_p2"] == 2.05   # swapped

def test_merge_no_match_returns_none():
    matches = [{"p1_name": "Unknown Player", "p2_name": "Also Unknown"}]
    odds    = {"jannik sinner vs carlos alcaraz": (1.85, 2.05)}
    result  = merge_odds(matches, odds)
    assert result[0]["odd_p1"] is None
    assert result[0]["odd_p2"] is None

def test_merge_does_not_mutate_input():
    matches  = [{"p1_name": "Jannik Sinner", "p2_name": "Carlos Alcaraz"}]
    original = dict(matches[0])
    merge_odds(matches, {})
    assert matches[0] == original

def test_merge_returns_new_list():
    matches = [{"p1_name": "A", "p2_name": "B"}]
    result  = merge_odds(matches, {})
    assert result is not matches

def test_merge_accent_normalization():
    """ESPN and API may format accented names differently."""
    matches = [{"p1_name": "Björn Borg", "p2_name": "Rafael Nadal"}]
    odds    = {"bjorn borg vs rafael nadal": (1.50, 2.60)}
    result  = merge_odds(matches, odds)
    assert result[0]["odd_p1"] == 1.50


# ── _extract_pinnacle_odds ────────────────────────────────────────────────────

def test_extract_pinnacle_present():
    bookmakers = [
        {"key": "bet365",  "markets": [{"key": "h2h", "outcomes": [{"price": 2.0}, {"price": 1.8}]}]},
        {"key": "pinnacle","markets": [{"key": "h2h", "outcomes": [{"price": 1.85},{"price": 2.05}]}]},
    ]
    assert _extract_pinnacle_odds(bookmakers) == (1.85, 2.05)

def test_extract_pinnacle_absent():
    bookmakers = [
        {"key": "bet365", "markets": [{"key": "h2h", "outcomes": [{"price": 2.0}, {"price": 1.8}]}]},
    ]
    assert _extract_pinnacle_odds(bookmakers) is None

def test_extract_pinnacle_empty():
    assert _extract_pinnacle_odds([]) is None


# ── _extract_avg_odds ─────────────────────────────────────────────────────────

def test_extract_avg_two_bookmakers():
    bookmakers = [
        {"key": "bk1", "markets": [{"key": "h2h", "outcomes": [{"price": 2.0}, {"price": 2.0}]}]},
        {"key": "bk2", "markets": [{"key": "h2h", "outcomes": [{"price": 1.8}, {"price": 2.2}]}]},
    ]
    result = _extract_avg_odds(bookmakers)
    assert result is not None
    assert abs(result[0] - 1.9) < 0.001
    assert abs(result[1] - 2.1) < 0.001

def test_extract_avg_empty():
    assert _extract_avg_odds([]) is None


# ── cache roundtrip ───────────────────────────────────────────────────────────

def test_cache_roundtrip(tmp_path, monkeypatch):
    """Write then read returns identical data."""
    import odds_api_client
    monkeypatch.setattr(odds_api_client, "ROOT", tmp_path)
    today  = date(2026, 3, 15)
    result = OddsResult(
        odds={"jannik sinner vs carlos alcaraz": (1.85, 2.05)},
        fetched_at="2026-03-15T09:32:00",
    )
    _save_cache("atp", today, result)
    loaded = _load_cache("atp", today)
    assert loaded is not None
    assert loaded.fetched_at == "2026-03-15T09:32:00"
    assert loaded.odds["jannik sinner vs carlos alcaraz"] == (1.85, 2.05)

def test_cache_missing_returns_none(tmp_path, monkeypatch):
    import odds_api_client
    monkeypatch.setattr(odds_api_client, "ROOT", tmp_path)
    assert _load_cache("atp", date(2026, 1, 1)) is None

def test_cache_corrupted_returns_none(tmp_path, monkeypatch):
    import odds_api_client
    monkeypatch.setattr(odds_api_client, "ROOT", tmp_path)
    today     = date(2026, 3, 15)
    cache_dir = tmp_path / "data" / "odds_cache" / "atp"
    cache_dir.mkdir(parents=True)
    (cache_dir / f"odds_{today.isoformat()}.json").write_text("NOT_VALID_JSON")
    assert _load_cache("atp", today) is None
