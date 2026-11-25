"""Field-gate tests (R1 keystone) — pure synthetic homographies, no model / cv2 / GPU.

The gate's job: given the recovered image->field homography (CameraPose.H), decide which
image points / detections lie ON THE FIELD, by BACK-PROJECTION (map the point through H to
field meters and test the field bounds) — robust to the tag-cluster extrapolation problem
that makes forward polygon projection overshoot.

We build a known image->field homography directly (FieldModel reads bounds from the real
WPILib tag map: ~16.518 x 8.043 m), then assert:
  * contains_image_pt accepts on-field points and rejects off-field (crowd) points;
  * an invalid / missing pose makes contains_image_pt return False;
  * gate_detections keeps on-field robots and drops a crowd box (floor-contact + bounds);
  * the main-pane y boundary rejects a box whose floor-contact is in the lower pane;
  * gate_detections is a NO-OP when the pose is invalid (honest degradation, no blanket
    reject), and gates a parallel alliance list in lock-step.
Detection objects are built directly; nothing reads a frame buffer.
"""
from __future__ import annotations

import numpy as np

from frcscout.geometry.field import FieldModel
from frcscout.io.types import CameraPose, Detection

FIELD_CFG = "configs/field_2026_rebuilt.yaml"


def _affine_pose(a=0.02, b=0.012, tx=-4.0, ty=-2.4, valid=True):
    """A simple image(px)->field(m) homography (no perspective). With the defaults:
        image x 200 -> field 0.0,   image x ~1025 -> field ~16.5
        image y 200 -> field 0.0,   image y ~870  -> field ~8.0
    so points well outside that image box back-project off the field rectangle."""
    H = [[a, 0.0, tx], [0.0, b, ty], [0.0, 0.0, 1.0]]
    return CameraPose(H=H, valid=valid, n_tags=4, reproj_err_px=0.5)


def _fm():
    return FieldModel(FIELD_CFG)


def _box(cx, y2, w=60.0, h=80.0):
    """Detection whose floor-contact point is (cx, y2): a box of size w x h with bottom
    edge at y2 and horizontal centre at cx."""
    x1, x2 = cx - w / 2.0, cx + w / 2.0
    y1 = y2 - h
    return Detection(cls="robot", xyxy=(x1, y1, x2, y2), conf=1.0, frame_idx=0)


# ───────────────────────────── field bounds source ───────────────────────────
def test_field_bounds_from_tagmap():
    """FieldModel prefers the real WPILib tag-map field bounds (~16.518 x 8.043) over the
    YAML placeholders (16.46 x 8.23)."""
    fm = _fm()
    assert abs(fm.length_m - 16.518) < 0.05, fm.length_m
    assert abs(fm.width_m - 8.043) < 0.05, fm.width_m


# ───────────────────────────── contains_image_pt ─────────────────────────────
def test_contains_on_field_point():
    fm, pose = _fm(), _affine_pose()
    # image (600, 500) -> field (~8.0, ~3.6): squarely on the field
    assert fm.contains_image_pt((600.0, 500.0), pose) is True


def test_rejects_off_field_crowd_point():
    fm, pose = _fm(), _affine_pose()
    # image (1500, 500) -> field x ~ 0.02*1500-4 = 26 m  >> 16.5 -> off field (crowd right)
    assert fm.contains_image_pt((1500.0, 500.0), pose) is False
    # image (50, 500) -> field x ~ -3 m  < 0 -> off field (crowd left)
    assert fm.contains_image_pt((50.0, 500.0), pose) is False
    # image (600, 60) -> field y ~ 0.012*60-2.4 = -1.68 < 0 -> off field (behind back wall)
    assert fm.contains_image_pt((600.0, 60.0), pose) is False


def test_tolerance_band_brings_just_outside_in():
    fm, pose = _fm(), _affine_pose()
    # a point ~10px past the right edge: x=1030 -> field ~16.6 (just over 16.518).
    just_out = (1030.0, 500.0)
    assert fm.contains_image_pt(just_out, pose, tol_px=0.0) is False
    # with a generous px tolerance the equivalent meter margin pulls it inside.
    assert fm.contains_image_pt(just_out, pose, tol_px=60.0) is True


def test_invalid_or_missing_pose_returns_false():
    fm = _fm()
    assert fm.contains_image_pt((600.0, 500.0), None) is False
    assert fm.contains_image_pt((600.0, 500.0), _affine_pose(valid=False)) is False


# ───────────────────────────── gate_detections ───────────────────────────────
def test_gate_keeps_on_field_drops_crowd():
    """Three on-field robots + one crowd box (right of the field): the crowd box is
    rejected, the three field robots kept."""
    fm, pose = _fm(), _affine_pose()
    y_main = 1080 * 0.52  # ~561 px main-pane boundary; all our floor-contacts are above it
    dets = [
        _box(300.0, 450.0),   # field x ~ 2.0   -> on field
        _box(600.0, 450.0),   # field x ~ 8.0   -> on field
        _box(900.0, 450.0),   # field x ~ 14.0  -> on field
        _box(1500.0, 450.0),  # field x ~ 26.0  -> CROWD (off field, right)
    ]
    kept = fm.gate_detections(dets, pose, y_main)
    assert len(kept) == 3
    assert all(d.xyxy[2] < 1400 for d in kept), "the crowd box leaked through the gate"


def test_gate_main_pane_boundary_rejects_lower_pane():
    """A box whose floor-contact is BELOW the main-pane boundary is rejected even if it
    would back-project on-field (it belongs to the opposite-angle bottom pane)."""
    fm, pose = _fm(), _affine_pose()
    y_main = 1080 * 0.52  # ~561
    on_field_top = _box(600.0, 450.0)        # floor-contact y2=450 < 561 -> in main pane
    on_field_bottom = _box(600.0, 700.0)     # y2=700 > 561 -> lower pane -> rejected
    kept = fm.gate_detections([on_field_top, on_field_bottom], pose, y_main)
    assert len(kept) == 1
    assert kept[0].xyxy[3] == 450.0


def test_gate_noop_without_valid_pose():
    """No valid pose -> the gate cannot localize, so it keeps everything (honest
    degradation), never a blanket reject."""
    fm = _fm()
    dets = [_box(300.0, 450.0), _box(1500.0, 450.0)]
    kept = fm.gate_detections(dets, _affine_pose(valid=False), 561.0)
    assert len(kept) == 2
    kept2 = fm.gate_detections(dets, None, 561.0)
    assert len(kept2) == 2


def test_gate_alliances_in_lockstep():
    """When a parallel alliance list is passed, the same keep-mask is applied to it."""
    fm, pose = _fm(), _affine_pose()
    dets = [_box(300.0, 450.0), _box(1500.0, 450.0), _box(900.0, 450.0)]
    allys = ["red", "blue", "red"]
    kept, kept_allys = fm.gate_detections(dets, pose, 561.0, alliances=allys)
    assert len(kept) == len(kept_allys) == 2
    # the crowd box (index 1, "blue") is dropped together with its alliance label
    assert kept_allys == ["red", "red"]


def test_gate_mask_counts_rejects():
    """gate_mask returns a per-detection keep list parallel to the input (for reject
    counting)."""
    fm, pose = _fm(), _affine_pose()
    dets = [_box(300.0, 450.0), _box(1500.0, 450.0), _box(50.0, 450.0)]
    mask = fm.gate_mask(dets, pose, 561.0)
    assert mask == [True, False, False]
    assert sum(1 for m in mask if not m) == 2  # two crowd rejects
