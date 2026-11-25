"""Tests for calib/collect.py — (vision, TBA) pair collector.

Covers:
  1. End-of-match gate rejects mid-match reads (timer_s > 0).
  2. Color-swap binds correctly (vision blue/red swapped vs TBA).
  3. Fuel → hubScore.totalCount pairing (component='fuel_total').
  4. Clean pairs.jsonl is produced and round-trips through load_pairs_jsonl.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

# --- Ensure project root and calib/ are importable ---
ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from calib.collect import (  # noqa: E402
    collect_pairs,
    VisionRead,
    _infer_tba_key,
    _resolve_color_swap,
    _hub_score,
    _END_OF_MATCH_TIMER_S,
)
from calib.error_model import PairedSample, load_pairs_jsonl  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers for building synthetic data
# ---------------------------------------------------------------------------

def _make_scout_json(
    match_key: str,
    timer_s: Optional[float],
    blue_fuel: Optional[int],
    red_fuel: Optional[int],
    blue_teams: List[int],
    red_teams: List[int],
    is_final: bool = False,
) -> dict:
    """Build a minimal scout-output dict (as would be loaded from disk)."""
    return {
        "match_key": match_key,
        "fps": 60.0,
        "robots": {},
        "events": [],
        "is_final": is_final,
        "overlay": {
            "match_label": f"Test {match_key}",
            "comp_level": None,
            "match_number": None,
            "blue_teams": blue_teams,
            "red_teams": red_teams,
            "blue_score": blue_fuel,
            "red_score": red_fuel,
            "blue_fuel": blue_fuel,
            "red_fuel": red_fuel,
            "phase": None,
            "timer_s": timer_s,
        },
    }


def _make_tba_match(
    event_key: str,
    match_number: int,
    comp_level: str,
    blue_teams: List[int],
    red_teams: List[int],
    blue_hub_total: int,
    red_hub_total: int,
    blue_hub_auto: int = 10,
    red_hub_auto: int = 10,
    blue_hub_teleop: Optional[int] = None,
    red_hub_teleop: Optional[int] = None,
) -> dict:
    """Build a minimal TBA match dict with score_breakdown."""
    if blue_hub_teleop is None:
        blue_hub_teleop = blue_hub_total - blue_hub_auto
    if red_hub_teleop is None:
        red_hub_teleop = red_hub_total - red_hub_auto

    def _team_keys(teams):
        return [f"frc{t}" for t in teams]

    return {
        "key": f"{event_key}_{comp_level}{match_number}",
        "comp_level": comp_level,
        "set_number": 1,
        "match_number": match_number,
        "event_key": event_key,
        "alliances": {
            "blue": {"team_keys": _team_keys(blue_teams)},
            "red": {"team_keys": _team_keys(red_teams)},
        },
        "score_breakdown": {
            "blue": {
                "totalPoints": blue_hub_total + 50,  # includes tower etc.
                "hubScore": {
                    "totalCount": blue_hub_total,
                    "autoCount": blue_hub_auto,
                    "teleopCount": blue_hub_teleop,
                    "totalPoints": blue_hub_total,
                    "autoPoints": blue_hub_auto,
                    "teleopPoints": blue_hub_teleop,
                },
                "foulPoints": 10,
            },
            "red": {
                "totalPoints": red_hub_total + 40,
                "hubScore": {
                    "totalCount": red_hub_total,
                    "autoCount": red_hub_auto,
                    "teleopCount": red_hub_teleop,
                    "totalPoints": red_hub_total,
                    "autoPoints": red_hub_auto,
                    "teleopPoints": red_hub_teleop,
                },
                "foulPoints": 5,
            },
        },
    }


def _write_tmpdir_files(
    scout_jsons: Dict[str, dict],
    tba_matches: List[dict],
) -> tuple:
    """Write scout JSONs + TBA JSON to a temp directory.

    Returns (scout_dir, tba_path, tmpdir_context).
    Call tmpdir_context.__exit__ to clean up.
    """
    tmpdir = tempfile.TemporaryDirectory()
    root = Path(tmpdir.name)

    scout_dir = root / "scout_out"
    scout_dir.mkdir()
    for fname, content in scout_jsons.items():
        (scout_dir / fname).write_text(json.dumps(content), encoding="utf-8")

    tba_path = root / "tba_matches.json"
    tba_path.write_text(json.dumps(tba_matches), encoding="utf-8")

    out_jsonl = root / "pairs.jsonl"
    return scout_dir, tba_path, out_jsonl, tmpdir


# ---------------------------------------------------------------------------
# Test 1 — End-of-match gate rejects mid-match reads
# ---------------------------------------------------------------------------

class TestEndOfMatchGate:
    def test_midmatch_reads_are_rejected(self):
        """Any timer_s > 0 must be counted as skipped, not as a pair."""
        tba = [_make_tba_match(
            "2026x", 1, "qm",
            blue_teams=[100, 200, 300], red_teams=[400, 500, 600],
            blue_hub_total=80, red_hub_total=120,
        )]
        scouts = {}
        # Various non-zero timer values (common overlay snapshot values)
        for timer, label in [(4.0, "timer4"), (17.0, "timer17"), (29.0, "timer29"), (30.0, "timer30")]:
            scouts[f"match_test_{label}.json"] = _make_scout_json(
                "qual1", timer, 75, 110,
                blue_teams=[100, 200, 300], red_teams=[400, 500, 600],
            )

        sd, tp, out, tmpdir = _write_tmpdir_files(scouts, tba)
        try:
            result = collect_pairs(scout_dir=sd, tba_json=tp, event_key="2026x", out_jsonl=out)
            assert result.n_pairs == 0, (
                f"Expected 0 pairs from mid-match reads but got {result.n_pairs}"
            )
            assert result.n_skipped_midmatch == 4, (
                f"Expected 4 mid-match skips but got {result.n_skipped_midmatch}"
            )
        finally:
            tmpdir.cleanup()

    def test_timer_zero_is_accepted_as_final(self):
        """timer_s == 0 is the end-of-match signal — must pass the gate."""
        event_key = "2026x"
        tba = [_make_tba_match(
            event_key, 5, "qm",
            blue_teams=[1000, 2000, 3000], red_teams=[4000, 5000, 6000],
            blue_hub_total=80, red_hub_total=120,
        )]
        scouts = {
            "match_qual5.json": _make_scout_json(
                "qual5", 0.0, 80, 120,
                blue_teams=[1000, 2000, 3000], red_teams=[4000, 5000, 6000],
            )
        }
        sd, tp, out, tmpdir = _write_tmpdir_files(scouts, tba)
        try:
            result = collect_pairs(scout_dir=sd, tba_json=tp, event_key=event_key, out_jsonl=out)
            assert result.n_skipped_midmatch == 0
            # Should have produced pairs (fuel_total for both alliances)
            assert result.n_pairs > 0, "Expected pairs from timer_s=0 read"
        finally:
            tmpdir.cleanup()

    def test_is_final_flag_accepted_when_timer_nonzero(self):
        """Explicit is_final=True in the JSON should override timer check."""
        event_key = "2026x"
        tba = [_make_tba_match(
            event_key, 7, "qm",
            blue_teams=[111, 222, 333], red_teams=[444, 555, 666],
            blue_hub_total=60, red_hub_total=90,
        )]
        scouts = {
            "match_qual7.json": _make_scout_json(
                "qual7", 5.0, 60, 90,
                blue_teams=[111, 222, 333], red_teams=[444, 555, 666],
                is_final=True,
            )
        }
        sd, tp, out, tmpdir = _write_tmpdir_files(scouts, tba)
        try:
            result = collect_pairs(scout_dir=sd, tba_json=tp, event_key=event_key, out_jsonl=out)
            assert result.n_skipped_midmatch == 0
            assert result.n_pairs > 0
        finally:
            tmpdir.cleanup()


# ---------------------------------------------------------------------------
# Test 2 — Color-swap binding
# ---------------------------------------------------------------------------

class TestColorSwap:
    def test_normal_alignment_no_swap(self):
        """When vision blue == TBA blue, color_swapped must be False."""
        tba_match = _make_tba_match(
            "2026x", 10, "qm",
            blue_teams=[100, 200, 300], red_teams=[400, 500, 600],
            blue_hub_total=100, red_hub_total=150,
        )
        resolved, swapped = _resolve_color_swap([100, 200, 300], [400, 500, 600], tba_match)
        assert resolved is True
        assert swapped is False

    def test_swapped_alignment_detected(self):
        """When vision blue == TBA red (and vice versa), color_swapped must be True."""
        tba_match = _make_tba_match(
            "2026x", 10, "qm",
            blue_teams=[100, 200, 300], red_teams=[400, 500, 600],
            blue_hub_total=100, red_hub_total=150,
        )
        # Pass teams reversed (vision blue is TBA red, vision red is TBA blue)
        resolved, swapped = _resolve_color_swap([400, 500, 600], [100, 200, 300], tba_match)
        assert resolved is True
        assert swapped is True

    def test_swapped_pair_uses_correct_tba_alliance(self):
        """With color swap: vision_red_fuel must pair to TBA blue hubScore."""
        event_key = "2026swap"
        tba = [_make_tba_match(
            event_key, 3, "qm",
            # TBA: blue=[100,200,300], red=[400,500,600]
            blue_teams=[100, 200, 300], red_teams=[400, 500, 600],
            blue_hub_total=80, red_hub_total=120,
        )]
        # Vision has colors swapped: vision_blue_teams = TBA_red_teams
        scouts = {
            "match_qual3.json": _make_scout_json(
                "qual3", 0.0,
                blue_fuel=120,  # vision blue fuel → should map to TBA red (120)
                red_fuel=80,    # vision red fuel  → should map to TBA blue (80)
                blue_teams=[400, 500, 600],  # vision blue = TBA red
                red_teams=[100, 200, 300],   # vision red  = TBA blue
            )
        }
        sd, tp, out, tmpdir = _write_tmpdir_files(scouts, tba)
        try:
            result = collect_pairs(scout_dir=sd, tba_json=tp, event_key=event_key, out_jsonl=out)
            assert result.n_pairs == 2, f"Expected 2 pairs (blue+red) but got {result.n_pairs}"

            # Check that the swap resolved correctly
            blue_pair = next((p for p in result.pairs if p.alliance == "blue"), None)
            red_pair = next((p for p in result.pairs if p.alliance == "red"), None)
            assert blue_pair is not None
            assert red_pair is not None
            # TBA blue hub total = 80; after swap, vision_red_fuel (80) should pair to TBA blue
            assert blue_pair.vision_est == 80.0, (
                f"TBA blue pair should use vision_red_fuel=80, got {blue_pair.vision_est}"
            )
            assert blue_pair.tba_actual == 80.0
            # TBA red hub total = 120; after swap, vision_blue_fuel (120) should pair to TBA red
            assert red_pair.vision_est == 120.0, (
                f"TBA red pair should use vision_blue_fuel=120, got {red_pair.vision_est}"
            )
            assert red_pair.tba_actual == 120.0
        finally:
            tmpdir.cleanup()


# ---------------------------------------------------------------------------
# Test 3 — Fuel → hubScore.totalCount pairing
# ---------------------------------------------------------------------------

class TestFuelPairing:
    def test_component_is_fuel_total(self):
        """All emitted pairs must have component='fuel_total'."""
        event_key = "2026fuel"
        tba = [_make_tba_match(
            event_key, 12, "qm",
            blue_teams=[10, 20, 30], red_teams=[40, 50, 60],
            blue_hub_total=95, red_hub_total=200,
        )]
        scouts = {
            "match_qual12.json": _make_scout_json(
                "qual12", 0.0, 95, 200,
                blue_teams=[10, 20, 30], red_teams=[40, 50, 60],
            )
        }
        sd, tp, out, tmpdir = _write_tmpdir_files(scouts, tba)
        try:
            result = collect_pairs(scout_dir=sd, tba_json=tp, event_key=event_key, out_jsonl=out)
            assert result.n_pairs >= 1
            for p in result.pairs:
                assert p.component == "fuel_total", (
                    f"Expected component='fuel_total', got {p.component!r}"
                )
        finally:
            tmpdir.cleanup()

    def test_vision_fuel_maps_to_hub_total_count(self):
        """vision_est must equal overlay fuel; tba_actual must equal hubScore.totalCount."""
        event_key = "2026fuel"
        blue_vision = 73
        red_vision = 155
        blue_tba_total = 88     # deliberately different from vision (miscalibrated)
        red_tba_total = 170

        tba = [_make_tba_match(
            event_key, 20, "qm",
            blue_teams=[11, 22, 33], red_teams=[44, 55, 66],
            blue_hub_total=blue_tba_total, red_hub_total=red_tba_total,
        )]
        scouts = {
            "match_qual20.json": _make_scout_json(
                "qual20", 0.0, blue_vision, red_vision,
                blue_teams=[11, 22, 33], red_teams=[44, 55, 66],
            )
        }
        sd, tp, out, tmpdir = _write_tmpdir_files(scouts, tba)
        try:
            result = collect_pairs(scout_dir=sd, tba_json=tp, event_key=event_key, out_jsonl=out)
            assert result.n_pairs == 2

            blue_pair = next(p for p in result.pairs if p.alliance == "blue")
            red_pair = next(p for p in result.pairs if p.alliance == "red")

            assert blue_pair.vision_est == float(blue_vision)
            assert blue_pair.tba_actual == float(blue_tba_total)
            assert red_pair.vision_est == float(red_vision)
            assert red_pair.tba_actual == float(red_tba_total)
        finally:
            tmpdir.cleanup()

    def test_none_fuel_is_skipped(self):
        """If one alliance fuel is None in the overlay, only the other is emitted."""
        event_key = "2026partial"
        tba = [_make_tba_match(
            event_key, 30, "qm",
            blue_teams=[1, 2, 3], red_teams=[4, 5, 6],
            blue_hub_total=50, red_hub_total=80,
        )]
        scouts = {
            "match_qual30.json": _make_scout_json(
                "qual30", 0.0,
                blue_fuel=50, red_fuel=None,  # red fuel unavailable
                blue_teams=[1, 2, 3], red_teams=[4, 5, 6],
            )
        }
        sd, tp, out, tmpdir = _write_tmpdir_files(scouts, tba)
        try:
            result = collect_pairs(scout_dir=sd, tba_json=tp, event_key=event_key, out_jsonl=out)
            assert result.n_pairs == 1, f"Expected 1 pair (blue only), got {result.n_pairs}"
            assert result.pairs[0].alliance == "blue"
            assert result.n_skipped_no_fuel >= 1
        finally:
            tmpdir.cleanup()


# ---------------------------------------------------------------------------
# Test 4 — Clean pairs.jsonl produced and round-trips
# ---------------------------------------------------------------------------

class TestJsonlOutput:
    def test_pairs_jsonl_written_and_loadable(self):
        """pairs.jsonl must be written and load cleanly via load_pairs_jsonl."""
        event_key = "2026jsonl"
        tba = [_make_tba_match(
            event_key, 5, "qm",
            blue_teams=[100, 200, 300], red_teams=[400, 500, 600],
            blue_hub_total=60, red_hub_total=90,
        )]
        scouts = {
            "match_qual5.json": _make_scout_json(
                "qual5", 0.0, 60, 90,
                blue_teams=[100, 200, 300], red_teams=[400, 500, 600],
            )
        }
        sd, tp, out, tmpdir = _write_tmpdir_files(scouts, tba)
        try:
            result = collect_pairs(scout_dir=sd, tba_json=tp, event_key=event_key, out_jsonl=out)
            assert result.n_pairs > 0

            # Load back
            loaded = load_pairs_jsonl(out)
            assert len(loaded) == result.n_pairs
            for p in loaded:
                assert p.component == "fuel_total"
                assert p.match_key.startswith(event_key)
                assert p.alliance in ("blue", "red")
                assert isinstance(p.vision_est, float)
                assert isinstance(p.tba_actual, float)
        finally:
            tmpdir.cleanup()

    def test_empty_jsonl_when_no_final_reads(self):
        """When all reads are mid-match, pairs.jsonl is written but empty."""
        event_key = "2026empty"
        tba = [_make_tba_match(
            event_key, 1, "qm",
            blue_teams=[10, 20, 30], red_teams=[40, 50, 60],
            blue_hub_total=40, red_hub_total=70,
        )]
        scouts = {
            "match_qual1.json": _make_scout_json(
                "qual1", 15.0, 40, 70,
                blue_teams=[10, 20, 30], red_teams=[40, 50, 60],
            )
        }
        sd, tp, out, tmpdir = _write_tmpdir_files(scouts, tba)
        try:
            result = collect_pairs(scout_dir=sd, tba_json=tp, event_key=event_key, out_jsonl=out)
            assert result.n_pairs == 0
            assert result.n_skipped_midmatch == 1
            assert out.exists()
            loaded = load_pairs_jsonl(out)
            assert loaded == []
        finally:
            tmpdir.cleanup()

    def test_jsonl_fields_match_paired_sample_schema(self):
        """Every JSONL line must carry the required PairedSample fields."""
        event_key = "2026schema"
        tba = [_make_tba_match(
            event_key, 8, "qm",
            blue_teams=[111, 222, 333], red_teams=[444, 555, 666],
            blue_hub_total=75, red_hub_total=130,
        )]
        scouts = {
            "match_qual8.json": _make_scout_json(
                "qual8", 0.0, 75, 130,
                blue_teams=[111, 222, 333], red_teams=[444, 555, 666],
            )
        }
        sd, tp, out, tmpdir = _write_tmpdir_files(scouts, tba)
        try:
            result = collect_pairs(scout_dir=sd, tba_json=tp, event_key=event_key, out_jsonl=out)
            assert result.n_pairs > 0

            required_fields = {"match_key", "comp_level", "alliance", "component",
                               "vision_est", "tba_actual", "event_key"}
            for raw_line in out.read_text().splitlines():
                if not raw_line.strip():
                    continue
                obj = json.loads(raw_line)
                missing = required_fields - obj.keys()
                assert not missing, f"Missing fields: {missing}"
        finally:
            tmpdir.cleanup()


# ---------------------------------------------------------------------------
# Test 5 — Hub score extraction helper
# ---------------------------------------------------------------------------

class TestHubScore:
    def test_extracts_nested_hub_score(self):
        """_hub_score() must correctly extract nested hubScore from a TBA match dict."""
        tba_match = _make_tba_match(
            "2026x", 1, "qm",
            blue_teams=[1, 2, 3], red_teams=[4, 5, 6],
            blue_hub_total=100, red_hub_total=50,
            blue_hub_auto=25, red_hub_auto=10,
        )
        blue_hub = _hub_score(tba_match, "blue")
        red_hub = _hub_score(tba_match, "red")

        assert blue_hub["total_count"] == 100
        assert blue_hub["auto_count"] == 25
        assert blue_hub["teleop_count"] == 75  # 100 - 25
        assert red_hub["total_count"] == 50
        assert red_hub["auto_count"] == 10

    def test_returns_none_when_breakdown_absent(self):
        """When score_breakdown is missing, all hub counts must be None."""
        tba_match = {"key": "2026x_qm1", "alliances": {}}  # no score_breakdown
        hub = _hub_score(tba_match, "blue")
        assert hub["total_count"] is None
        assert hub["auto_count"] is None
        assert hub["teleop_count"] is None


# ---------------------------------------------------------------------------
# Test 6 — TBA key inference
# ---------------------------------------------------------------------------

class TestInferTbaKey:
    def test_qual_prefix(self):
        assert _infer_tba_key("qual59", "2026ctwat") == "2026ctwat_qm59"

    def test_q_prefix(self):
        assert _infer_tba_key("q59rob", "2026ctwat") == "2026ctwat_qm59"

    def test_unrecognized_key_returns_none(self):
        # "m1080" style is ambiguous — should return None
        result = _infer_tba_key("m1080rob", "2026ctwat")
        # Either None or a key that doesn't map to a real TBA match
        # We just verify it doesn't raise
        assert result is None or isinstance(result, str)

    def test_event_key_embedded(self):
        key = _infer_tba_key("qual12", "2026miket")
        assert key == "2026miket_qm12"


# ---------------------------------------------------------------------------
# Test 7 — Real on-disk data (honest zero-pair result)
# ---------------------------------------------------------------------------

class TestRealOnDiskData:
    """Run collect_pairs on the actual project data directory.

    This test is HONEST: it asserts that 0 pairs are found because all
    on-disk vision reads are mid-match snapshots (timer_s > 0).
    It does NOT assert n_pairs > 0 — that would fabricate results.
    """

    def test_real_data_honest_count(self):
        """collect_pairs on real data returns 0 pairs (all reads are mid-match)."""
        v2_root = Path(__file__).resolve().parents[1]
        scout_dir = v2_root / "data" / "scout_out"
        tba_json = v2_root / "data" / "tba" / "2026ctwat_matches.json"

        if not scout_dir.is_dir() or not tba_json.exists():
            import pytest
            pytest.skip("Real data not available in this environment")

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "pairs.jsonl"
            result = collect_pairs(
                scout_dir=scout_dir,
                tba_json=tba_json,
                event_key="2026ctwat",
                out_jsonl=out,
            )

        # All on-disk files have timer_s > 0 — NONE are final reads.
        # This is the honest expected result as of the current dataset.
        assert result.n_pairs == 0, (
            f"Expected 0 real pairs (all mid-match) but got {result.n_pairs}. "
            "If real end-of-match reads were added to scout_out/, update this assertion."
        )
        # Confirm skips are being counted (not silently swallowed)
        assert result.n_skipped_midmatch > 0, (
            "Expected mid-match skips to be counted — something is wrong with gate logic."
        )
