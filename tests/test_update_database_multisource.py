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

    def test_all_tokens_are_initials_returns_empty(self):
        """All tokens are initials-like → return ''."""
        assert _extract_last("J. A.") == ""


class TestMakeDedupKey:
    """_make_dedup_key returns (date, frozenset({last_w, last_l})) or None."""

    def test_basic_key(self):
        d = pd.Timestamp("2026-03-10")
        key = _make_dedup_key(d, "Sinner J.", "Djokovic N.")
        assert key == (d.date(), frozenset(["sinner", "djokovic"]))

    def test_cross_format_same_key(self):
        """tennis-data name and ESPN name produce the same dedup key."""
        d = pd.Timestamp("2026-03-10")
        key_td   = _make_dedup_key(d, "Sinner J.", "Djokovic N.")
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

    def test_same_last_name_returns_none(self):
        """Two players with identical last names → return None (ambiguous)."""
        d = pd.Timestamp("2026-03-10")
        assert _make_dedup_key(d, "Andy Murray", "Jamie Murray") is None


class TestFetchEspnComplement:
    """fetch_espn_complement returns only ESPN matches absent from the tennis-data DataFrame."""

    def _espn_row(self, date_str, winner, loser, source="espn"):
        return {
            "tourney_date": pd.Timestamp(date_str),
            "winner_name": winner,
            "loser_name": loser,
            "surface": "Hard", "tourney_level": "M",
            "tourney_name": "Indian Wells", "round": "QF",
            "best_of": 3, "winner_rank": None, "loser_rank": None,
            "winner_rank_pts": None, "loser_rank_pts": None,
            "source": source,
        }

    def test_returns_only_new_matches(self, monkeypatch):
        """ESPN match already in td → dropped; new ESPN match → kept."""
        td_df = pd.DataFrame([{
            "tourney_date": pd.Timestamp("2026-03-10"),
            "winner_name": "Sinner J.",
            "loser_name": "Djokovic N.",
        }])
        espn_df = pd.DataFrame([
            self._espn_row("2026-03-10", "Jannik Sinner", "Novak Djokovic"),  # duplicate
            self._espn_row("2026-03-12", "Carlos Alcaraz", "Daniil Medvedev"),  # new
        ])
        import update_database as udb
        monkeypatch.setattr(udb, "fetch_recent", lambda tour, days: espn_df)
        result = udb.fetch_espn_complement(td_df, "atp", days=14)
        assert len(result) == 1
        assert result.iloc[0]["winner_name"] == "Carlos Alcaraz"

    def test_empty_espn_returns_empty(self, monkeypatch):
        """ESPN returns nothing → empty DataFrame."""
        td_df = pd.DataFrame([{
            "tourney_date": pd.Timestamp("2026-03-10"),
            "winner_name": "Sinner J.", "loser_name": "Djokovic N.",
        }])
        import update_database as udb
        monkeypatch.setattr(udb, "fetch_recent", lambda tour, days: pd.DataFrame())
        result = udb.fetch_espn_complement(td_df, "atp", days=14)
        assert result.empty

    def test_empty_td_returns_all_espn(self, monkeypatch):
        """tennis-data empty → all ESPN matches are new."""
        espn_df = pd.DataFrame([self._espn_row("2026-03-14", "Carlos Alcaraz", "Daniil Medvedev")])
        import update_database as udb
        monkeypatch.setattr(udb, "fetch_recent", lambda tour, days: espn_df)
        result = udb.fetch_espn_complement(pd.DataFrame(), "atp", days=14)
        assert len(result) == 1

    def test_source_column_is_espn(self, monkeypatch):
        """Returned rows preserve source='espn'."""
        espn_df = pd.DataFrame([self._espn_row("2026-03-14", "Carlos Alcaraz", "Daniil Medvedev")])
        import update_database as udb
        monkeypatch.setattr(udb, "fetch_recent", lambda tour, days: espn_df)
        result = udb.fetch_espn_complement(pd.DataFrame(), "atp", days=14)
        assert result.iloc[0]["source"] == "espn"

    def test_all_duplicates_returns_empty(self, monkeypatch):
        """All ESPN matches already in td → empty DataFrame."""
        td_df = pd.DataFrame([{
            "tourney_date": pd.Timestamp("2026-03-10"),
            "winner_name": "Sinner J.", "loser_name": "Djokovic N.",
        }])
        espn_df = pd.DataFrame([
            self._espn_row("2026-03-10", "Jannik Sinner", "Novak Djokovic"),
        ])
        import update_database as udb
        monkeypatch.setattr(udb, "fetch_recent", lambda tour, days: espn_df)
        result = udb.fetch_espn_complement(td_df, "atp", days=14)
        assert result.empty
