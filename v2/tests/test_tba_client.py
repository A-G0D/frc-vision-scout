"""Unit tests for frcscout.integrations.tba_client.

All tests monkeypatch the HTTP layer — no real network calls.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure src/ is on the path regardless of how the test runner is invoked
ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from frcscout.integrations.tba_client import (  # noqa: E402
    TBAAuthError,
    TBACacheError,
    TBAClient,
    BreakdownComponents,
    AllianceBreakdown,
    _flatten_breakdown,
    _probe,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _minimal_match(key: str = "2026test_qm1") -> dict:
    """Minimal TBA match dict — no score_breakdown."""
    return {
        "key": key,
        "comp_level": "qm",
        "match_number": 1,
        "blue_score": 120,
        "red_score": 98,
        "score_breakdown": None,
    }


def _full_match(key: str = "2026test_qm2") -> dict:
    """TBA match dict with a realistic (TENTATIVE) score_breakdown."""
    return {
        "key": key,
        "comp_level": "qm",
        "match_number": 2,
        "blue_score": 155,
        "red_score": 112,
        "score_breakdown": {
            "blue": {
                "totalPoints": 155,
                "foulPoints": 10,
                "minorFoulCount": 2,
                "majorFoulCount": 0,
                "adjustPoints": 0,
                "autoTowerPoints": 15,
                "endGameTowerPoints": 30,
                "totalTowerPoints": 45,
                "hubScore": {
                    "autoPoints": 20,
                    "autoCount": 20,
                    "teleopPoints": 80,
                    "teleopCount": 80,
                    "totalPoints": 100,
                    "totalCount": 100,
                },
            },
            "red": {
                "totalPoints": 112,
                "foulPoints": 5,
                "minorFoulCount": 1,
                "majorFoulCount": 0,
                "adjustPoints": 0,
                "autoTowerPoints": 0,
                "endGameTowerPoints": 20,
                "totalTowerPoints": 20,
                "hubScore": {
                    "autoPoints": 15,
                    "autoCount": 15,
                    "teleopPoints": 72,
                    "teleopCount": 72,
                    "totalPoints": 87,
                    "totalCount": 87,
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# Offline mode — no TBA_AUTH_KEY
# ---------------------------------------------------------------------------

class TestOfflineMode:
    def test_no_key_no_cache_raises_auth_error(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TBA_AUTH_KEY", raising=False)
        client = TBAClient(cache_dir=tmp_path)
        try:
            client.get_match("2026test_qm1")
            assert False, "expected TBAAuthError"
        except TBAAuthError as exc:
            assert "TBA_AUTH_KEY" in str(exc)

    def test_explicit_offline_flag_blocks_live(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TBA_AUTH_KEY", "dummy-key")
        client = TBAClient(cache_dir=tmp_path, offline=True)
        try:
            client.get_match("2026test_qm1")
            assert False, "expected TBAAuthError"
        except TBAAuthError as exc:
            assert "offline mode" in str(exc).lower() or "Offline mode" in str(exc)

    def test_import_never_crashes_without_key(self, monkeypatch):
        """Importing the module must not raise even when TBA_AUTH_KEY is absent."""
        monkeypatch.delenv("TBA_AUTH_KEY", raising=False)
        # Constructing a client with no key must not raise at import/construction time;
        # it only raises when a live call is actually made.
        try:
            _ = TBAClient(cache_dir="/tmp/tba_test_offline")
        except TBAAuthError:
            assert False, "TBAClient() must not raise TBAAuthError at construction time"

    def test_cache_hit_works_offline(self, tmp_path, monkeypatch):
        """If a cached response exists, offline mode must return it without any key."""
        monkeypatch.delenv("TBA_AUTH_KEY", raising=False)
        client = TBAClient(cache_dir=tmp_path)
        # Pre-seed the cache
        cache_file = client._cache_path("/match/2026test_qm1")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        match_data = _minimal_match()
        with open(cache_file, "w") as fh:
            json.dump(match_data, fh)

        result = client.get_match("2026test_qm1")
        assert result["key"] == "2026test_qm1"
        assert result["blue_score"] == 120


# ---------------------------------------------------------------------------
# Live fetch — monkeypatched HTTP
# ---------------------------------------------------------------------------

class TestLiveFetch:
    def _make_client_with_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TBA_AUTH_KEY", "test-key-abc")
        return TBAClient(cache_dir=tmp_path)

    def test_get_match_caches_response(self, tmp_path, monkeypatch):
        client = self._make_client_with_key(tmp_path, monkeypatch)
        match_data = _minimal_match()

        with patch.object(client, "_live_fetch", return_value=match_data) as mock_fetch:
            result = client.get_match("2026test_qm1")
            assert result["key"] == "2026test_qm1"
            mock_fetch.assert_called_once_with("/match/2026test_qm1")

        # Second call must use cache — live_fetch should NOT be called again
        with patch.object(client, "_live_fetch", side_effect=AssertionError("should be cached")):
            result2 = client.get_match("2026test_qm1")
            assert result2["key"] == "2026test_qm1"

    def test_get_event_matches_returns_list(self, tmp_path, monkeypatch):
        client = self._make_client_with_key(tmp_path, monkeypatch)
        matches = [_minimal_match("2026test_qm1"), _minimal_match("2026test_qm2")]

        with patch.object(client, "_live_fetch", return_value=matches):
            result = client.get_event_matches("2026test")
        assert isinstance(result, list)
        assert len(result) == 2

    def test_cache_path_is_deterministic(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TBA_AUTH_KEY", raising=False)
        client = TBAClient(cache_dir=tmp_path)
        p1 = client._cache_path("/match/2026test_qm1")
        p2 = client._cache_path("/match/2026test_qm1")
        assert p1 == p2

    def test_cache_path_differs_for_different_routes(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TBA_AUTH_KEY", raising=False)
        client = TBAClient(cache_dir=tmp_path)
        p1 = client._cache_path("/match/2026test_qm1")
        p2 = client._cache_path("/event/2026test/matches")
        assert p1 != p2


# ---------------------------------------------------------------------------
# extract_breakdown_components — defensive parsing
# ---------------------------------------------------------------------------

class TestExtractBreakdown:
    def test_no_breakdown_returns_none_components(self, tmp_path, monkeypatch):
        """When score_breakdown is absent, all components are None but no exception."""
        monkeypatch.delenv("TBA_AUTH_KEY", raising=False)
        client = TBAClient(cache_dir=tmp_path)
        match = _minimal_match()
        bc = client.extract_breakdown_components(match)

        assert isinstance(bc, BreakdownComponents)
        assert bc.breakdown_available is False
        assert bc.blue.total_score == 120   # comes from match.blue_score
        assert bc.red.total_score == 98
        assert bc.blue.fuel_total_points is None
        assert bc.red.tower_total_points is None

    def test_full_breakdown_extracts_fuel_and_tower(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TBA_AUTH_KEY", raising=False)
        client = TBAClient(cache_dir=tmp_path)
        match = _full_match()
        bc = client.extract_breakdown_components(match)

        assert bc.breakdown_available is True

        # Blue alliance
        b = bc.blue
        assert b.total_score == 155
        assert b.fuel_auto_points == 20
        assert b.fuel_teleop_points == 80
        assert b.fuel_total_points == 100
        assert b.fuel_auto_count == 20
        assert b.tower_auto_points == 15
        assert b.tower_endgame_points == 30
        assert b.tower_total_points == 45
        assert b.foul_points == 10
        assert b.minor_foul_count == 2
        assert b.major_foul_count == 0

        # Red alliance
        r = bc.red
        assert r.total_score == 112
        assert r.fuel_total_count == 87

    def test_keys_found_records_actual_key_names(self, tmp_path, monkeypatch):
        """keys_found should map component labels to the real TBA key strings found."""
        monkeypatch.delenv("TBA_AUTH_KEY", raising=False)
        client = TBAClient(cache_dir=tmp_path)
        match = _full_match()
        bc = client.extract_breakdown_components(match)

        b = bc.blue
        # totalPoints should be found (it's in the fixture)
        assert "total_score" in b.keys_found
        assert "foul_points" in b.keys_found
        # The found key for foulPoints should be "foulPoints"
        assert b.keys_found["foul_points"] == "foulPoints"

    def test_unknown_keys_end_up_in_keys_missing(self, tmp_path, monkeypatch):
        """Components with no matching candidate must appear in keys_missing."""
        monkeypatch.delenv("TBA_AUTH_KEY", raising=False)
        client = TBAClient(cache_dir=tmp_path)
        # Intentionally empty breakdown — all components should be missing
        match = {
            "key": "2026test_qm99",
            "blue_score": 50,
            "red_score": 40,
            "score_breakdown": {"blue": {}, "red": {}},
        }
        bc = client.extract_breakdown_components(match)
        # "total_score" is special — satisfied from match.blue_score, not keys_missing
        assert bc.blue.total_score == 50
        # Other components should be missing (no keys in the empty dicts)
        assert "fuel_total_points" in bc.blue.keys_missing
        assert "tower_total_points" in bc.blue.keys_missing

    def test_raw_breakdown_preserved(self, tmp_path, monkeypatch):
        """raw_breakdown must be populated so future key-discovery is possible."""
        monkeypatch.delenv("TBA_AUTH_KEY", raising=False)
        client = TBAClient(cache_dir=tmp_path)
        match = _full_match()
        bc = client.extract_breakdown_components(match)
        assert isinstance(bc.blue.raw_breakdown, dict)
        # raw_breakdown should contain at least the top-level keys
        assert "foulPoints" in bc.blue.raw_breakdown

    def test_schema_note_present(self, tmp_path, monkeypatch):
        """BreakdownComponents must carry a schema-provenance note. The 2026 field
        names are now VERIFIED against real event data (data/tba/2026ctwat_matches.json,
        see metrics/SCHEMA_VERIFIED.md), so the note references that verification."""
        monkeypatch.delenv("TBA_AUTH_KEY", raising=False)
        client = TBAClient(cache_dir=tmp_path)
        bc = client.extract_breakdown_components(_minimal_match())
        assert "verified" in bc.schema_note.lower()
        assert "SCHEMA_VERIFIED.md" in bc.schema_note

    def test_missing_alliance_sub_dict_is_graceful(self, tmp_path, monkeypatch):
        """If score_breakdown has no 'blue' sub-key, return all-None without raising."""
        monkeypatch.delenv("TBA_AUTH_KEY", raising=False)
        client = TBAClient(cache_dir=tmp_path)
        match = {
            "key": "2026test_qm3",
            "blue_score": 60,
            "red_score": 55,
            "score_breakdown": {"red": {"totalPoints": 55, "foulPoints": 5}},
        }
        bc = client.extract_breakdown_components(match)
        # Blue has no sub-dict — all components should be None
        assert bc.blue.fuel_total_points is None
        assert bc.blue.total_score == 60   # still from match root
        # Red has a sub-dict — foul_points should be found
        assert bc.red.foul_points == 5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TestFlattenBreakdown:
    def test_nested_hub_score_is_dotted(self):
        d = {"hubScore": {"autoPoints": 20, "teleopPoints": 80}, "foulPoints": 5}
        flat = _flatten_breakdown(d)
        assert flat["hubScore.autoPoints"] == 20
        assert flat["hubScore.teleopPoints"] == 80
        assert flat["foulPoints"] == 5

    def test_already_flat_is_unchanged(self):
        d = {"totalPoints": 100, "foulPoints": 10}
        flat = _flatten_breakdown(d)
        assert flat["totalPoints"] == 100
        assert flat["foulPoints"] == 10

    def test_empty_dict(self):
        assert _flatten_breakdown({}) == {}


class TestProbe:
    def test_returns_first_matching_candidate(self):
        flat = {"foulPoints": 7, "foul_points": 3}
        val, key = _probe(flat, ["foulPoints", "foul_points"])
        assert val == 7
        assert key == "foulPoints"

    def test_falls_back_to_second_candidate(self):
        flat = {"foul_points": 3}
        val, key = _probe(flat, ["foulPoints", "foul_points"])
        assert val == 3
        assert key == "foul_points"

    def test_returns_none_none_when_no_match(self):
        flat = {"unrelated": 99}
        val, key = _probe(flat, ["foulPoints", "foul_points"])
        assert val is None
        assert key is None
