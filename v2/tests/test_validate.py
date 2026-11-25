"""Tests for src/frcscout/eval/validate.py.

All tests are synthetic (no network, no GPU, no files on disk except temp JSON).
Run with:
    PYTHONPATH=src python -m pytest tests/test_validate.py -q
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from frcscout.eval.validate import validate, format_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_tba(tmp_path: Path, matches: list[dict]) -> str:
    """Write a TBA-format JSON file and return its path as a string."""
    p = tmp_path / "tba.json"
    p.write_text(json.dumps({"matches": matches}))
    return str(p)


def _qm_tba(n: int, blue_score: int, red_score: int,
            blue_fuel: int = 0, red_fuel: int = 0) -> dict:
    """Build a minimal TBA qual-match row."""
    return {
        "comp_level": "qm",
        "match_number": n,
        "blue_teams": [1000, 1001, 1002],
        "red_teams": [2000, 2001, 2002],
        "blue_score": blue_score,
        "red_score": red_score,
        "blue_fuel": blue_fuel,
        "red_fuel": red_fuel,
    }


def _computed_match(key: str, blue_est: float, red_est: float,
                    blue_fuel: float = 0.0, red_fuel: float = 0.0) -> dict:
    """Build a computed per-match entry."""
    return {
        "key": key,
        "blue_score_est": blue_est,
        "red_score_est": red_est,
        "blue_total_fuel": blue_fuel,
        "red_total_fuel": red_fuel,
    }


# ---------------------------------------------------------------------------
# Perfect-prediction tests
# ---------------------------------------------------------------------------

class TestPerfectPredictions:
    """When estimates exactly match TBA, all errors should be zero."""

    def test_zero_score_error(self, tmp_path):
        tba = [_qm_tba(1, 100, 80), _qm_tba(2, 90, 110)]
        tba_path = _write_tba(tmp_path, tba)

        computed = {
            "matches": [
                _computed_match("qm1", 100.0, 80.0),
                _computed_match("qm2", 90.0, 110.0),
            ]
        }
        report = validate(computed, tba_path)

        agg = report["aggregates"]
        assert agg["score_mae"] == pytest.approx(0.0), (
            f"score_mae should be 0 for perfect estimates, got {agg['score_mae']}"
        )
        assert agg["score_pct_err"] == pytest.approx(0.0), (
            f"score_pct_err should be 0 for perfect estimates, got {agg['score_pct_err']}"
        )

    def test_winner_accuracy_one(self, tmp_path):
        tba = [_qm_tba(1, 100, 80), _qm_tba(2, 90, 110), _qm_tba(3, 60, 60)]
        tba_path = _write_tba(tmp_path, tba)

        computed = {
            "matches": [
                _computed_match("qm1", 100.0, 80.0),   # blue wins
                _computed_match("qm2", 90.0, 110.0),   # red wins
                _computed_match("qm3", 60.0, 60.0),    # tie
            ]
        }
        report = validate(computed, tba_path)
        assert report["aggregates"]["winner_accuracy"] == pytest.approx(1.0), (
            f"winner_accuracy should be 1.0, got {report['aggregates']['winner_accuracy']}"
        )

    def test_per_match_winner_correct(self, tmp_path):
        tba = [_qm_tba(1, 100, 80)]
        tba_path = _write_tba(tmp_path, tba)

        computed = {"matches": [_computed_match("qm1", 100.0, 80.0)]}
        report = validate(computed, tba_path)

        assert len(report["per_match"]) == 1
        row = report["per_match"][0]
        assert row["winner_correct"] is True
        assert row["est_winner"] == "blue"
        assert row["true_winner"] == "blue"

    def test_fuel_zero_error(self, tmp_path):
        tba = [_qm_tba(1, 100, 80, blue_fuel=50, red_fuel=40)]
        tba_path = _write_tba(tmp_path, tba)

        computed = {"matches": [_computed_match("qm1", 100.0, 80.0,
                                                blue_fuel=50.0, red_fuel=40.0)]}
        report = validate(computed, tba_path)

        agg = report["aggregates"]
        assert agg["fuel_pct_err"] == pytest.approx(0.0), (
            f"fuel_pct_err should be 0 for perfect fuel estimates, got {agg['fuel_pct_err']}"
        )


# ---------------------------------------------------------------------------
# Known-error tests
# ---------------------------------------------------------------------------

class TestKnownErrors:
    """Estimates off by a specific amount should produce the exact expected error."""

    def test_score_error_25_percent(self, tmp_path):
        # Blue true = 100, estimated = 125 → |%err| = 25%
        tba = [_qm_tba(1, 100, 80)]
        tba_path = _write_tba(tmp_path, tba)

        computed = {"matches": [_computed_match("qm1", 125.0, 80.0)]}
        report = validate(computed, tba_path)

        row = report["per_match"][0]
        assert row["blue_score_err"] == pytest.approx(25.0), (
            f"Blue score absolute error should be 25, got {row['blue_score_err']}"
        )
        assert row["blue_score_pct_err"] == pytest.approx(25.0), (
            f"Blue score %err should be 25.0, got {row['blue_score_pct_err']}"
        )
        # Red perfect → 0 error; blue off → aggregate > 0
        assert report["aggregates"]["score_mae"] > 0

    def test_score_mae_computed_correctly(self, tmp_path):
        # Match 1: blue off by 10 (true=100, est=110), red perfect (true=80, est=80)
        # Match 2: blue off by 20 (true=200, est=220), red perfect (true=100, est=100)
        # Per-alliance errors: 10, 0, 20, 0 → MAE = (10+0+20+0)/4 = 7.5
        tba = [_qm_tba(1, 100, 80), _qm_tba(2, 200, 100)]
        tba_path = _write_tba(tmp_path, tba)

        computed = {
            "matches": [
                _computed_match("qm1", 110.0, 80.0),
                _computed_match("qm2", 220.0, 100.0),
            ]
        }
        report = validate(computed, tba_path)

        assert report["aggregates"]["score_mae"] == pytest.approx(7.5), (
            f"Expected score_mae=7.5, got {report['aggregates']['score_mae']}"
        )

    def test_winner_wrong_flips_accuracy(self, tmp_path):
        # True: blue=100, red=80 → blue wins
        # Estimate: blue=70, red=90 → red wins (wrong)
        tba = [_qm_tba(1, 100, 80)]
        tba_path = _write_tba(tmp_path, tba)

        computed = {"matches": [_computed_match("qm1", 70.0, 90.0)]}
        report = validate(computed, tba_path)

        assert report["per_match"][0]["winner_correct"] is False
        assert report["aggregates"]["winner_accuracy"] == pytest.approx(0.0)

    def test_fuel_pct_error_50_percent(self, tmp_path):
        # True blue_fuel=100, estimated=150 → 50% over
        tba = [_qm_tba(1, 100, 80, blue_fuel=100, red_fuel=80)]
        tba_path = _write_tba(tmp_path, tba)

        computed = {"matches": [_computed_match("qm1", 100.0, 80.0,
                                                blue_fuel=150.0, red_fuel=80.0)]}
        report = validate(computed, tba_path)

        # Blue total fuel pct err = (150-100)/100*100 = +50%
        row = report["per_match"][0]
        assert row["blue_total_fuel_pct_err"] == pytest.approx(50.0)

    def test_unmatched_key_skips_gracefully(self, tmp_path):
        """A computed match whose key doesn't exist in TBA should not crash."""
        tba = [_qm_tba(1, 100, 80)]
        tba_path = _write_tba(tmp_path, tba)

        computed = {
            "matches": [
                _computed_match("qm1", 100.0, 80.0),
                _computed_match("qm99", 50.0, 50.0),  # not in TBA
            ]
        }
        report = validate(computed, tba_path)

        assert len(report["per_match"]) == 2
        row_99 = next(r for r in report["per_match"] if r["key"] == "qm99")
        assert row_99["tba_found"] is False
        assert row_99["blue_score_err"] is None


# ---------------------------------------------------------------------------
# Overlay count MAE tests
# ---------------------------------------------------------------------------

class TestOverlayMae:

    def test_perfect_overlay_count_zero_mae(self, tmp_path):
        tba = [_qm_tba(1, 100, 80)]
        tba_path = _write_tba(tmp_path, tba)
        computed = {"matches": [_computed_match("qm1", 100.0, 80.0)]}

        overlay = {
            "cv_counts":      [3, 5, 2, 7, 1],
            "overlay_counts": [3, 5, 2, 7, 1],
        }
        report = validate(computed, tba_path, overlay_truth=overlay)

        assert report["overlay"] is not None
        assert report["overlay"]["count_mae"] == pytest.approx(0.0)
        assert report["overlay"]["n"] == 5

    def test_overlay_mae_correct_value(self, tmp_path):
        """MAE should equal mean(abs(cv - overlay))."""
        tba = [_qm_tba(1, 100, 80)]
        tba_path = _write_tba(tmp_path, tba)
        computed = {"matches": [_computed_match("qm1", 100.0, 80.0)]}

        # Differences: |4-3|=1, |5-5|=0, |4-2|=2 → MAE = 3/3 = 1.0
        overlay = {
            "cv_counts":      [4, 5, 4],
            "overlay_counts": [3, 5, 2],
        }
        report = validate(computed, tba_path, overlay_truth=overlay)

        assert report["overlay"]["count_mae"] == pytest.approx(1.0)
        assert report["overlay"]["n"] == 3

    def test_no_overlay_truth_produces_none(self, tmp_path):
        tba = [_qm_tba(1, 100, 80)]
        tba_path = _write_tba(tmp_path, tba)
        computed = {"matches": [_computed_match("qm1", 100.0, 80.0)]}

        report = validate(computed, tba_path, overlay_truth=None)
        assert report["overlay"] is None

    def test_overlay_appears_in_validated_when_given(self, tmp_path):
        tba = [_qm_tba(1, 100, 80)]
        tba_path = _write_tba(tmp_path, tba)
        computed = {"matches": [_computed_match("qm1", 100.0, 80.0)]}

        overlay = {"cv_counts": [3, 5], "overlay_counts": [3, 5]}
        report = validate(computed, tba_path, overlay_truth=overlay)

        assert "overlay_ball_count" in report["validated_metrics"]

    def test_overlay_not_in_validated_when_absent(self, tmp_path):
        tba = [_qm_tba(1, 100, 80)]
        tba_path = _write_tba(tmp_path, tba)
        computed = {"matches": [_computed_match("qm1", 100.0, 80.0)]}

        report = validate(computed, tba_path, overlay_truth=None)
        assert "overlay_ball_count" not in report["validated_metrics"]


# ---------------------------------------------------------------------------
# Unvalidated-metrics honesty tests
# ---------------------------------------------------------------------------

class TestUnvalidatedMetrics:

    def test_unvalidated_not_empty(self, tmp_path):
        """There must always be some unvalidated metrics — we cannot validate
        things TBA doesn't record."""
        tba = [_qm_tba(1, 100, 80)]
        tba_path = _write_tba(tmp_path, tba)
        computed = {"matches": [_computed_match("qm1", 100.0, 80.0)]}

        report = validate(computed, tba_path)
        assert len(report["unvalidated_metrics"]) > 0, (
            "There should always be unvalidated metrics — TBA doesn't record everything"
        )

    def test_each_unvalidated_has_reason(self, tmp_path):
        """Every unvalidated entry must have both 'metric' and 'reason' fields."""
        tba = [_qm_tba(1, 100, 80)]
        tba_path = _write_tba(tmp_path, tba)
        computed = {"matches": [_computed_match("qm1", 100.0, 80.0)]}

        report = validate(computed, tba_path)
        for entry in report["unvalidated_metrics"]:
            assert "metric" in entry, f"Missing 'metric' key in {entry}"
            assert "reason" in entry, f"Missing 'reason' key in {entry}"
            assert len(entry["reason"]) > 10, (
                f"Reason too short to be meaningful: {entry['reason']}"
            )

    def test_known_unvalidated_metrics_present(self, tmp_path):
        """Specific metrics known to be unvalidatable must appear in the list."""
        tba = [_qm_tba(1, 100, 80)]
        tba_path = _write_tba(tmp_path, tba)
        computed = {"matches": [_computed_match("qm1", 100.0, 80.0)]}

        report = validate(computed, tba_path)
        names = {e["metric"] for e in report["unvalidated_metrics"]}

        for expected in (
            "shot_accuracy",
            "cycle_time",
            "sourcing_mix",
            "intake_rate",
            "fuel_into_inactive_hub",
        ):
            assert expected in names, (
                f"'{expected}' should be in unvalidated_metrics but was not found. "
                f"Got: {sorted(names)}"
            )

    def test_unvalidated_never_in_validated(self, tmp_path):
        """No metric should appear in BOTH validated and unvalidated lists."""
        tba = [_qm_tba(1, 100, 80)]
        tba_path = _write_tba(tmp_path, tba)
        computed = {"matches": [_computed_match("qm1", 100.0, 80.0)]}

        overlay = {"cv_counts": [3, 5], "overlay_counts": [3, 5]}
        report = validate(computed, tba_path, overlay_truth=overlay)

        validated_set = set(report["validated_metrics"])
        unvalidated_set = {e["metric"] for e in report["unvalidated_metrics"]}

        overlap = validated_set & unvalidated_set
        assert not overlap, (
            f"These metrics appear in BOTH validated and unvalidated: {overlap}"
        )

    def test_shot_accuracy_never_validated(self, tmp_path):
        """shot_accuracy must never appear in validated_metrics — TBA cannot check it."""
        tba = [_qm_tba(1, 100, 80)]
        tba_path = _write_tba(tmp_path, tba)
        # Even if the computed dict contains a shot_accuracy field, it cannot be validated
        computed = {
            "matches": [
                {
                    "key": "qm1",
                    "blue_score_est": 100.0,
                    "red_score_est": 80.0,
                    "shot_accuracy": 0.92,  # pipeline estimate — no TBA truth
                }
            ]
        }
        report = validate(computed, tba_path)

        assert "shot_accuracy" not in report["validated_metrics"], (
            "shot_accuracy must never appear in validated_metrics"
        )
        # But it should be in unvalidated
        unvalidated_names = {e["metric"] for e in report["unvalidated_metrics"]}
        assert "shot_accuracy" in unvalidated_names


# ---------------------------------------------------------------------------
# Headline / structure tests
# ---------------------------------------------------------------------------

class TestReportStructure:

    def test_report_has_all_top_level_keys(self, tmp_path):
        tba = [_qm_tba(1, 100, 80)]
        tba_path = _write_tba(tmp_path, tba)
        computed = {"matches": [_computed_match("qm1", 100.0, 80.0)]}

        report = validate(computed, tba_path)

        for key in (
            "per_match", "aggregates", "overlay",
            "validated_metrics", "unvalidated_metrics", "headline",
        ):
            assert key in report, f"Missing top-level key: '{key}'"

    def test_headline_mentions_both_counts(self, tmp_path):
        tba = [_qm_tba(1, 100, 80)]
        tba_path = _write_tba(tmp_path, tba)
        computed = {"matches": [_computed_match("qm1", 100.0, 80.0)]}

        report = validate(computed, tba_path)
        hl = report["headline"].lower()

        # Headline must communicate the validated vs unvalidated split
        assert "truth-check" in hl or "validated" in hl, (
            f"Headline should mention what was validated: {report['headline']}"
        )
        assert "unvalidated" in hl, (
            f"Headline should mention unvalidated count: {report['headline']}"
        )

    def test_aggregates_has_required_keys(self, tmp_path):
        tba = [_qm_tba(1, 100, 80)]
        tba_path = _write_tba(tmp_path, tba)
        computed = {"matches": [_computed_match("qm1", 100.0, 80.0)]}

        report = validate(computed, tba_path)
        agg = report["aggregates"]

        for key in ("score_mae", "score_pct_err", "winner_accuracy", "fuel_pct_err"):
            assert key in agg, f"Missing aggregate key: '{key}'"

    def test_empty_computed_matches_no_crash(self, tmp_path):
        tba = [_qm_tba(1, 100, 80)]
        tba_path = _write_tba(tmp_path, tba)
        computed = {"matches": []}

        report = validate(computed, tba_path)
        assert report["aggregates"]["score_mae"] is None
        assert report["aggregates"]["winner_accuracy"] is None


# ---------------------------------------------------------------------------
# format_report smoke tests
# ---------------------------------------------------------------------------

class TestFormatReport:

    def test_format_report_is_string(self, tmp_path):
        tba = [_qm_tba(1, 100, 80)]
        tba_path = _write_tba(tmp_path, tba)
        computed = {"matches": [_computed_match("qm1", 100.0, 80.0)]}

        report = validate(computed, tba_path)
        text = format_report(report)

        assert isinstance(text, str)
        assert len(text) > 100

    def test_format_report_contains_unvalidated_header(self, tmp_path):
        tba = [_qm_tba(1, 100, 80)]
        tba_path = _write_tba(tmp_path, tba)
        computed = {"matches": [_computed_match("qm1", 100.0, 80.0)]}

        report = validate(computed, tba_path)
        text = format_report(report)

        assert "UNVALIDATED" in text, "Formatted report must call out unvalidated metrics"
        assert "VALIDATED" in text, "Formatted report must call out validated metrics"

    def test_format_report_contains_match_key(self, tmp_path):
        tba = [_qm_tba(1, 100, 80)]
        tba_path = _write_tba(tmp_path, tba)
        computed = {"matches": [_computed_match("qm1", 100.0, 80.0)]}

        report = validate(computed, tba_path)
        text = format_report(report)

        assert "qm1" in text

    def test_format_report_does_not_crash_with_nones(self, tmp_path):
        """format_report must not raise when aggregates contain None values."""
        tba = [_qm_tba(1, 100, 80)]
        tba_path = _write_tba(tmp_path, tba)
        computed = {"matches": []}  # no matches → None aggregates

        report = validate(computed, tba_path)
        text = format_report(report)
        assert isinstance(text, str)

    def test_format_report_overlay_section(self, tmp_path):
        tba = [_qm_tba(1, 100, 80)]
        tba_path = _write_tba(tmp_path, tba)
        computed = {"matches": [_computed_match("qm1", 100.0, 80.0)]}
        overlay = {"cv_counts": [3, 5, 2], "overlay_counts": [3, 5, 2]}

        report = validate(computed, tba_path, overlay_truth=overlay)
        text = format_report(report)

        assert "OVERLAY" in text
        # Perfect counts → MAE = 0.000
        assert "0.000" in text


# ---------------------------------------------------------------------------
# Multi-match aggregate correctness
# ---------------------------------------------------------------------------

class TestMultiMatchAggregates:

    def test_winner_accuracy_partial(self, tmp_path):
        """2/3 correct winner predictions → accuracy = 0.667."""
        tba = [
            _qm_tba(1, 100, 80),   # blue wins (true)
            _qm_tba(2, 90, 110),   # red wins (true)
            _qm_tba(3, 60, 70),    # red wins (true)
        ]
        tba_path = _write_tba(tmp_path, tba)

        computed = {
            "matches": [
                _computed_match("qm1", 100.0, 80.0),   # blue → correct
                _computed_match("qm2", 90.0, 110.0),   # red  → correct
                _computed_match("qm3", 70.0, 60.0),    # blue → WRONG (true=red)
            ]
        }
        report = validate(computed, tba_path)

        assert report["aggregates"]["winner_accuracy"] == pytest.approx(2 / 3, abs=1e-6)

    def test_n_matches_compared_correct(self, tmp_path):
        tba = [_qm_tba(i, 100, 80) for i in range(1, 6)]
        tba_path = _write_tba(tmp_path, tba)

        computed = {
            "matches": [_computed_match(f"qm{i}", 100.0, 80.0) for i in range(1, 6)]
        }
        report = validate(computed, tba_path)

        assert report["aggregates"]["n_matches_compared"] == 5
        assert report["aggregates"]["n_matches_tba_found"] == 5
