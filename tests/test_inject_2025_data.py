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


class TestConvert2025ToConsolidated:
    """convert_2025_to_consolidated maps XLSX rows to consolidated schema."""

    def _make_raw_df(self):
        """Minimal 2-row tennis-data XLSX-style DataFrame."""
        return pd.DataFrame({
            'Date'       : [pd.Timestamp('2025-01-06'), pd.Timestamp('2025-01-07')],
            'Tournament' : ['Brisbane',                  'Brisbane'],
            'Series'     : ['ATP250',                    'ATP250'],
            'Surface'    : ['Hard',                      'Hard'],
            'Round'      : ['1st Round',                 '2nd Round'],
            'Best of'    : [3,                           3],
            'Winner'     : ['Vukic A.',                  'Michelsen A.'],
            'Loser'      : ['Djokovic N.',               'Sinner J.'],
            'WRank'      : [77.0,                        18.0],
            'LRank'      : [7.0,                         1.0],
            'WPts'       : [800.0,                       3400.0],
            'LPts'       : [9960.0,                      10900.0],
            'Comment'    : ['Completed',                 'Completed'],
            'year'       : [2025,                        2025],
        })

    def _make_name_to_id(self):
        return {
            "Novak Djokovic": 104925,
            "Jannik Sinner": 207989,
        }

    def _make_name_mapping(self):
        """tennis-data name → Sackmann name."""
        return {
            "Djokovic N.": "Novak Djokovic",
            "Sinner J.": "Jannik Sinner",
            "Vukic A.": "Vukic A.",
            "Michelsen A.": "Michelsen A.",
        }

    def test_required_columns_present(self):
        from inject_2025_data import convert_2025_to_consolidated
        df_raw = self._make_raw_df()
        result = convert_2025_to_consolidated(
            df_raw, self._make_name_to_id(), self._make_name_mapping(), 'atp'
        )
        required = [
            'tourney_id', 'tourney_name', 'tourney_date', 'tourney_level',
            'surface', 'draw_size', 'best_of', 'round', 'year',
            'winner_id', 'winner_name', 'winner_rank', 'winner_rank_points',
            'loser_id', 'loser_name', 'loser_rank', 'loser_rank_points',
        ]
        for col in required:
            assert col in result.columns, f"Missing column: {col}"

    def test_round_mapping(self):
        from inject_2025_data import convert_2025_to_consolidated
        df_raw = self._make_raw_df()
        result = convert_2025_to_consolidated(
            df_raw, self._make_name_to_id(), self._make_name_mapping(), 'atp'
        )
        assert result.iloc[0]['round'] == 'R64'   # '1st Round'
        assert result.iloc[1]['round'] == 'R32'   # '2nd Round'

    def test_level_mapping(self):
        from inject_2025_data import convert_2025_to_consolidated
        df_raw = self._make_raw_df()
        result = convert_2025_to_consolidated(
            df_raw, self._make_name_to_id(), self._make_name_mapping(), 'atp'
        )
        assert (result['tourney_level'] == 'A').all()  # ATP250 → A

    def test_known_player_gets_sackmann_id(self):
        from inject_2025_data import convert_2025_to_consolidated
        df_raw = self._make_raw_df()
        result = convert_2025_to_consolidated(
            df_raw, self._make_name_to_id(), self._make_name_mapping(), 'atp'
        )
        assert int(result.iloc[0]['loser_id']) == 104925   # Djokovic
        assert int(result.iloc[1]['loser_id']) == 207989   # Sinner

    def test_unknown_player_gets_synthetic_id_in_range(self):
        from inject_2025_data import convert_2025_to_consolidated
        df_raw = self._make_raw_df()
        result = convert_2025_to_consolidated(
            df_raw, self._make_name_to_id(), self._make_name_mapping(), 'atp'
        )
        vukic_id = int(result.iloc[0]['winner_id'])
        assert 1_000_001 <= vukic_id <= 2_900_000

    def test_tourney_id_format(self):
        """tourney_id must match pattern YYYY-NNNN."""
        from inject_2025_data import convert_2025_to_consolidated
        df_raw = self._make_raw_df()
        result = convert_2025_to_consolidated(
            df_raw, self._make_name_to_id(), self._make_name_mapping(), 'atp'
        )
        for tid in result['tourney_id']:
            parts = tid.split('-')
            assert parts[0] == '2025'
            assert parts[1].isdigit()

    def test_year_column_is_2025(self):
        from inject_2025_data import convert_2025_to_consolidated
        df_raw = self._make_raw_df()
        result = convert_2025_to_consolidated(
            df_raw, self._make_name_to_id(), self._make_name_mapping(), 'atp'
        )
        assert (result['year'] == 2025).all()

    def test_best_of_dtype_is_Int64(self):
        from inject_2025_data import convert_2025_to_consolidated
        df_raw = self._make_raw_df()
        result = convert_2025_to_consolidated(
            df_raw, self._make_name_to_id(), self._make_name_mapping(), 'atp'
        )
        assert str(result['best_of'].dtype) == 'Int64'

    def test_stat_columns_are_nan(self):
        """Service stats (w_ace etc.) should be NaN — not available from tennis-data."""
        from inject_2025_data import convert_2025_to_consolidated
        df_raw = self._make_raw_df()
        result = convert_2025_to_consolidated(
            df_raw, self._make_name_to_id(), self._make_name_mapping(), 'atp'
        )
        stat_cols = ['w_ace', 'w_df', 'l_ace', 'l_df']
        for col in stat_cols:
            if col in result.columns:
                assert result[col].isna().all(), f"{col} should be all NaN"


class TestIdempotency:
    """Running inject_2025_data when 2025 data is already present must be a no-op."""

    def test_already_present_returns_true(self, tmp_path):
        """
        _check_already_injected returns True when year==2025 rows exist.
        """
        from inject_2025_data import _check_already_injected

        df = pd.DataFrame({'year': [2024, 2025], 'tourney_name': ['X', 'Y']})
        p = tmp_path / "matches_consolidated.parquet"
        df.to_parquet(p, index=False)

        assert _check_already_injected(p) is True

    def test_not_present_returns_false(self, tmp_path):
        from inject_2025_data import _check_already_injected

        df = pd.DataFrame({'year': [2023, 2024], 'tourney_name': ['X', 'Y']})
        p = tmp_path / "matches_consolidated.parquet"
        df.to_parquet(p, index=False)

        assert _check_already_injected(p) is False
