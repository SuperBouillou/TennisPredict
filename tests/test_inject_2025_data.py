# tests/test_inject_2025_data.py
"""Tests for inject_2025_data.py — no network required."""
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestSyntheticId:
    """_synthetic_id returns a stable int in [1_000_001, 2_900_000]."""

    def test_returns_int_in_range(self):
        from inject_2025_data import _synthetic_id
        sid = _synthetic_id("Random Player")
        assert isinstance(sid, int)
        assert 1_000_001 <= sid <= 2_900_000

    def test_deterministic(self):
        from inject_2025_data import _synthetic_id
        assert _synthetic_id("Sinner J.") == _synthetic_id("Sinner J.")

    def test_different_names_different_ids(self):
        from inject_2025_data import _synthetic_id
        assert _synthetic_id("Alpha") != _synthetic_id("Beta")

    def test_case_insensitive(self):
        """'sinner j.' and 'SINNER J.' should produce the same ID."""
        from inject_2025_data import _synthetic_id
        assert _synthetic_id("sinner j.") == _synthetic_id("SINNER J.")


class TestBuildPlayerNameToId:
    """_build_player_name_to_id parses atp_players.csv correctly."""

    def _make_csv(self, tmp_path) -> Path:
        csv_path = tmp_path / "atp_players.csv"
        csv_path.write_text(
            "player_id,name_first,name_last,hand,dob,ioc,height,wikidata_id\n"
            "207989,Jannik,Sinner,R,20011116.0,ITA,188.0,Q123\n"
            "104925,Novak,Djokovic,R,19870522.0,SRB,188.0,Q456\n"
            "invalid,,name,R,,,,\n"
        )
        return csv_path

    def test_returns_full_name_to_id(self, tmp_path):
        from inject_2025_data import _build_player_name_to_id
        csv_path = self._make_csv(tmp_path)
        mapping = _build_player_name_to_id(csv_path)
        assert mapping["Jannik Sinner"] == 207989
        assert mapping["Novak Djokovic"] == 104925

    def test_skips_invalid_id_rows(self, tmp_path):
        from inject_2025_data import _build_player_name_to_id
        csv_path = self._make_csv(tmp_path)
        mapping = _build_player_name_to_id(csv_path)
        assert all(isinstance(v, int) for v in mapping.values())

    def test_returns_dict(self, tmp_path):
        from inject_2025_data import _build_player_name_to_id
        csv_path = self._make_csv(tmp_path)
        result = _build_player_name_to_id(csv_path)
        assert isinstance(result, dict)
