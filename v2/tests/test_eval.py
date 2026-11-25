"""Tests for the eval layer: metrics.py and tba_verify.py.

Run with:
    PYTHONPATH=src python -m pytest tests/test_eval.py tests/test_tba_validity.py -q
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from frcscout.eval.metrics import ball_map50, count_mae, int8_recall_ratio
from frcscout.eval.tba_verify import verify, _is_impossible


# ===========================================================================
# Helpers
# ===========================================================================

def _write_label(directory: Path, name: str, lines: list[str]):
    """Write a YOLO label file to *directory/name*."""
    (directory / name).write_text("\n".join(lines) + "\n")


# ===========================================================================
# ball_map50 tests
# ===========================================================================

class TestBallMap50:
    """Tests for the AP@IoU0.5 computation."""

    def test_identical_pred_gt_gives_1(self, tmp_path):
        """Perfect predictions should give mAP50 == 1.0."""
        pred_dir = tmp_path / "preds"
        gt_dir = tmp_path / "gt"
        pred_dir.mkdir()
        gt_dir.mkdir()

        # Two label files, each with two ball detections.
        # Format: cls cx cy w h [conf]
        labels_a = [
            "0 0.1 0.1 0.1 0.1 0.99",  # ball at top-left
            "0 0.5 0.5 0.1 0.1 0.95",  # ball at centre
        ]
        labels_b = [
            "0 0.8 0.8 0.15 0.15 0.90",
        ]
        for name, lines in [("a.txt", labels_a), ("b.txt", labels_b)]:
            _write_label(pred_dir, name, lines)
            # GT has no confidence column — identical boxes
            gt_lines = [l.rsplit(" ", 1)[0] for l in lines]  # strip conf
            _write_label(gt_dir, name, gt_lines)

        ap = ball_map50(str(pred_dir), str(gt_dir))
        assert ap == pytest.approx(1.0, abs=1e-6), f"Expected 1.0 but got {ap}"

    def test_disjoint_pred_gt_gives_near_zero(self, tmp_path):
        """Predictions that share no overlap with GT boxes should give ~0 AP."""
        pred_dir = tmp_path / "preds"
        gt_dir = tmp_path / "gt"
        pred_dir.mkdir()
        gt_dir.mkdir()

        # Pred: centre of image
        _write_label(pred_dir, "a.txt", ["0 0.5 0.5 0.1 0.1 0.99"])
        # GT: far corner, no overlap at all (|0.95-0.5| >> (0.1+0.1)/2)
        _write_label(gt_dir, "a.txt", ["0 0.05 0.05 0.05 0.05"])

        ap = ball_map50(str(pred_dir), str(gt_dir))
        assert ap < 0.1, f"Expected near-zero AP but got {ap}"

    def test_empty_pred_dir_gives_zero(self, tmp_path):
        """No predictions -> AP = 0."""
        pred_dir = tmp_path / "preds"
        gt_dir = tmp_path / "gt"
        pred_dir.mkdir()
        gt_dir.mkdir()
        # GT has boxes but no pred files exist
        _write_label(gt_dir, "a.txt", ["0 0.5 0.5 0.1 0.1"])

        ap = ball_map50(str(pred_dir), str(gt_dir))
        assert ap == 0.0

    def test_no_gt_no_pred_vacuous(self, tmp_path):
        """Both dirs empty -> no meaningful AP; vacuously 1.0 (nothing to detect)."""
        pred_dir = tmp_path / "preds"
        gt_dir = tmp_path / "gt"
        pred_dir.mkdir()
        gt_dir.mkdir()
        # Pred file exists but is empty; GT file doesn't exist.
        _write_label(pred_dir, "a.txt", [])

        ap = ball_map50(str(pred_dir), str(gt_dir))
        # 0 GT total, 0 predictions that matched -> vacuously 1.0
        assert ap == 1.0

    def test_partial_overlap(self, tmp_path):
        """When more GT boxes exist than correct predictions, AP < 1.0.

        We have 2 GT boxes and only 1 correct prediction — so recall never
        exceeds 0.5, meaning the 11-point AP is at most 0.5.
        """
        pred_dir = tmp_path / "preds"
        gt_dir = tmp_path / "gt"
        pred_dir.mkdir()
        gt_dir.mkdir()

        # One prediction matches first GT box; the second GT box has no prediction.
        _write_label(pred_dir, "a.txt", [
            "0 0.1 0.1 0.1 0.1 0.99",  # matches GT box 1
        ])
        _write_label(gt_dir, "a.txt", [
            "0 0.1 0.1 0.1 0.1",   # GT box 1 — will be matched
            "0 0.8 0.8 0.1 0.1",   # GT box 2 — no prediction
        ])

        ap = ball_map50(str(pred_dir), str(gt_dir))
        # With max recall = 0.5, AP is strictly < 1.0 and > 0
        assert 0.0 < ap < 1.0, f"Expected AP in (0, 1) but got {ap}"


# ===========================================================================
# tba_verify tests
# ===========================================================================

def _make_tba_json(tmp_path: Path, matches: list[dict]) -> str:
    p = tmp_path / "tba.json"
    p.write_text(json.dumps({"matches": matches}))
    return str(p)


def _make_detected_json(tmp_path: Path, matches: list[dict]) -> str:
    p = tmp_path / "detected.json"
    p.write_text(json.dumps(matches))
    return str(p)


class TestTbaVerify:
    """Tests for tba_verify.verify()."""

    def _make_valid_qm(self, match_number: int, swap_colors: bool = False):
        b = [1000 + match_number * 3, 1001 + match_number * 3, 1002 + match_number * 3]
        r = [2000 + match_number * 3, 2001 + match_number * 3, 2002 + match_number * 3]
        if swap_colors:
            b, r = r, b
        return {
            "comp_level": "qm",
            "match_number": match_number,
            "blue_teams": b,
            "red_teams": r,
            "blue_score": 50 + match_number,
            "red_score": 40 + match_number,
        }

    def test_perfect_match_accuracy_is_1(self, tmp_path):
        """When detected == TBA exactly, accuracy should be 1.0."""
        matches = [self._make_valid_qm(i) for i in range(1, 6)]
        tba_path = _make_tba_json(tmp_path, matches)
        det_path = _make_detected_json(tmp_path, matches)

        result = verify(det_path, tba_path)

        assert result["overall_accuracy"] == pytest.approx(1.0)
        assert result["n_detected"] == 5
        assert result["n_tba"] == 5
        assert result["structural_error_count"] == 0
        assert len(result["structural_errors"]) == 0

    def test_color_swap_counts_as_correct(self, tmp_path):
        """Color-swapped teams+scores should still count as correct."""
        tba_matches = [self._make_valid_qm(1)]
        # Detected with blue/red swapped
        det_match = self._make_valid_qm(1, swap_colors=True)
        # Also swap scores
        det_match["blue_score"], det_match["red_score"] = (
            det_match["red_score"], det_match["blue_score"]
        )

        tba_path = _make_tba_json(tmp_path, tba_matches)
        det_path = _make_detected_json(tmp_path, [det_match])

        result = verify(det_path, tba_path)
        assert result["overall_accuracy"] == pytest.approx(1.0)

    def test_structural_error_in_playoff_does_not_inflate_accuracy(self, tmp_path):
        """A broken playoff row must appear in structural_errors and NOT be counted correct."""
        # One valid qual match
        valid_qm = self._make_valid_qm(1)

        # One impossible playoff row: a team is on both alliances (the v1 bug)
        broken_sf = {
            "comp_level": "sf",
            "match_label": "sf1m1",
            "match_number": None,      # playoff rows sometimes have null match_number
            "blue_teams": [254, 1114, 3],
            "red_teams": [3, 4909, 4572],  # team 3 is on both alliances!
            "blue_score": 100,
            "red_score": 80,
        }

        # TBA has the correct sf1m1 entry
        tba_sf_correct = {
            "comp_level": "sf",
            "match_label": "sf1m1",
            "match_number": 1,
            "set_number": 1,
            "blue_teams": [254, 1114, 456],
            "red_teams": [3, 4909, 4572],
            "blue_score": 100,
            "red_score": 80,
        }
        tba_qm = self._make_valid_qm(1)

        tba_path = _make_tba_json(tmp_path, [tba_qm, tba_sf_correct])
        det_path = _make_detected_json(tmp_path, [valid_qm, broken_sf])

        result = verify(det_path, tba_path)

        # The broken row MUST appear in structural_errors
        assert result["structural_error_count"] >= 1
        assert len(result["structural_errors"]) >= 1
        broken_errors = [
            e for e in result["structural_errors"]
            if set(e.get("blue_teams", [])) & set(e.get("red_teams", []))
        ]
        assert len(broken_errors) == 1, "Broken playoff row not flagged"

        # Overall accuracy should not be 1.0: the broken row fails the team check
        # (Only the valid_qm row should be fully correct)
        assert result["overall_accuracy"] is not None
        assert result["overall_accuracy"] < 1.0, (
            f"Accuracy should be < 1.0 because broken playoff row must not count as correct, "
            f"got {result['overall_accuracy']}"
        )

    def test_per_level_accuracy_reported(self, tmp_path):
        """Per-level dict should be populated with qm and sf keys."""
        qm1 = self._make_valid_qm(1)
        qm2 = self._make_valid_qm(2)
        sf1 = {
            "comp_level": "sf",
            "match_label": "sf1m1",
            "match_number": 1,
            "set_number": 1,
            "blue_teams": [111, 222, 333],
            "red_teams": [444, 555, 666],
            "blue_score": 70,
            "red_score": 60,
        }
        all_matches = [qm1, qm2, sf1]
        tba_path = _make_tba_json(tmp_path, all_matches)
        det_path = _make_detected_json(tmp_path, all_matches)

        result = verify(det_path, tba_path)

        assert "qm" in result["per_level"]
        assert "sf" in result["per_level"]
        assert result["per_level"]["qm"]["accuracy"] == pytest.approx(1.0)
        assert result["per_level"]["sf"]["accuracy"] == pytest.approx(1.0)

    def test_denominator_reflects_detected_count(self, tmp_path):
        """n_detected should equal how many rows the CV system produced."""
        matches = [self._make_valid_qm(i) for i in range(1, 4)]
        tba_path = _make_tba_json(tmp_path, matches)
        # Only report 2 of the 3 matches (simulating partial detection)
        det_path = _make_detected_json(tmp_path, matches[:2])

        result = verify(det_path, tba_path)

        assert result["n_detected"] == 2
        assert result["n_tba"] == 3
        assert len(result["unmatched_tba"]) == 1  # one TBA row has no detected counterpart

    def test_zero_detected(self, tmp_path):
        """Empty detected file should not crash; overall_accuracy = None."""
        tba_path = _make_tba_json(tmp_path, [self._make_valid_qm(1)])
        det_path = _make_detected_json(tmp_path, [])

        result = verify(det_path, tba_path)

        assert result["n_detected"] == 0
        assert result["overall_accuracy"] is None


# ===========================================================================
# count_mae and int8_recall_ratio (smoke tests)
# ===========================================================================

def test_count_mae_perfect():
    assert count_mae([3, 5, 2], [3, 5, 2]) == pytest.approx(0.0)


def test_count_mae_off_by_one():
    assert count_mae([3, 5, 2], [4, 5, 2]) == pytest.approx(1 / 3)


def test_int8_recall_ratio_normal():
    assert int8_recall_ratio(0.95, 1.0) == pytest.approx(0.95)


def test_int8_recall_ratio_zero_fp32():
    assert int8_recall_ratio(0.5, 0.0) == pytest.approx(0.0)
