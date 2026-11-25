"""Tests for calib/error_model.py — TBA-trained error correction."""
import sys
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from calib.error_model import (  # noqa: E402
    ErrorCorrectionModel, PairedSample, CalibrationArtifact, _fit_ols,
    _identity_calib, MIN_MATCHES_TOTAL,
)


def _make_pairs(n, a_true, b_true, comp="total_score", noise=()):
    """tba = a_true*vision + b_true (+ optional per-index noise)."""
    out = []
    for i in range(n):
        vis = 50.0 + 3.0 * i
        nz = noise[i] if i < len(noise) else 0.0
        out.append(PairedSample(
            match_key=f"2026x_qm{i+1:02d}", comp_level="qm", alliance="blue",
            component=comp, vision_est=vis, tba_actual=a_true * vis + b_true + nz))
    return out


def test_ols_recovers_known_affine():
    a, b = _fit_ols(_make_pairs(12, 1.5, 10.0))
    assert abs(a - 1.5) < 1e-6
    assert abs(b - 10.0) < 1e-6


def test_ols_degenerate_x_is_bias_only():
    # all vision identical -> slope 1.0, bias = mean residual
    pairs = [PairedSample("m", "qm", "blue", "total_score", 100.0, 150.0) for _ in range(5)]
    a, b = _fit_ols(pairs)
    assert a == 1.0 and abs(b - 50.0) < 1e-9


def test_calibration_reduces_holdout_mae():
    """The user's scenario: vision underestimates ~33% (tba = 1.5*vision)."""
    model = ErrorCorrectionModel(mode="ols", event_key="2026x")
    art = model.fit_from_samples(_make_pairs(20, 1.5, 0.0, noise=[(-1) ** i * 2 for i in range(20)]))
    rep = art.eval_reports["total_score"]
    assert not rep.skipped, rep.skip_reason
    assert rep.mae_cal < rep.mae_raw          # calibration helps
    assert rep.mae_reduction_pct > 50.0       # big systematic bias -> big reduction
    assert rep.is_useful


def test_apply_keeps_raw_and_emits_ci():
    model = ErrorCorrectionModel(mode="ols", event_key="2026x")
    art = model.fit_from_samples(_make_pairs(16, 1.4, 5.0))
    out = art.apply({"total_score": 100.0})
    assert out["total_score_raw"] == 100.0
    assert abs(out["total_score_cal"] - (1.4 * 100.0 + 5.0)) < 1.0
    assert out["total_score_cal_ci_lo"] <= out["total_score_cal"] <= out["total_score_cal_ci_hi"]
    assert out["cal_event_key"] == "2026x"


def test_insufficient_data_returns_identity():
    model = ErrorCorrectionModel(mode="ols")
    art = model.fit_from_samples(_make_pairs(MIN_MATCHES_TOTAL - 1, 1.5, 0.0))
    assert all(c.is_identity for c in art.components.values())
    assert art.warnings
    # identity passes vision through unchanged
    out = art.apply({"total_score": 123.0})
    assert out["total_score_cal"] == 123.0


def test_artifact_roundtrip(tmp_path):
    model = ErrorCorrectionModel(mode="ols", event_key="2026x", version="v2")
    art = model.fit_from_samples(_make_pairs(16, 1.3, 2.0))
    p = tmp_path / "art.json"
    art.save(p)
    art2 = CalibrationArtifact.load(p)
    assert art2.version == "v2"
    assert abs(art2.components["total_score"].a - art.components["total_score"].a) < 1e-9
    assert art2.apply({"total_score": 80.0})["total_score_cal"] == art.apply({"total_score": 80.0})["total_score_cal"]


def test_unknown_component_passes_through():
    model = ErrorCorrectionModel(mode="ols", event_key="2026x")
    art = model.fit_from_samples(_make_pairs(16, 1.5, 0.0))
    out = art.apply({"some_unmodeled_metric": 42.0})
    assert out["some_unmodeled_metric_cal"] == 42.0
    assert out["some_unmodeled_metric_identity"] is True
