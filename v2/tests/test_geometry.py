"""Geometry-layer tests: homography round-trip, field masking/tiling, ballistic arc fit.

Dependency imports (cv2 / scipy / pupil_apriltags) are guarded so this file still PARSES
and collects when a dep is missing — the affected tests are skipped instead of erroring.
"""
from __future__ import annotations

import numpy as np
import pytest

# ---- guarded optional deps --------------------------------------------------- #
try:
    import cv2  # noqa: F401
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

try:
    import scipy  # noqa: F401
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

try:
    import pupil_apriltags  # noqa: F401
    HAVE_APRIL = True
except Exception:
    HAVE_APRIL = False

from frcscout.geometry import ballistics
from frcscout.geometry.field import FieldModel
from frcscout.io.types import CameraPose

FIELD_CFG = "configs/field_2026_rebuilt.yaml"


# ----------------------------------------------------------------------------- #
# Homography round-trip
# ----------------------------------------------------------------------------- #
def _apply_h(H, pts):
    """Apply 3x3 H to an N x 2 array -> N x 2."""
    pts = np.asarray(pts, dtype=np.float64)
    hom = np.hstack([pts, np.ones((pts.shape[0], 1))]) @ H.T
    return hom[:, :2] / hom[:, 2:3]


@pytest.mark.skipif(not HAVE_CV2, reason="cv2 not installed")
def test_homography_roundtrip():
    """Pick a known image->field homography, project a set of field points back into the
    image, recover H from the (image, field) correspondences, and assert the recovered
    field coordinates match the originals within tolerance."""
    # A plausible image(px) -> field(m) homography: scale + slight perspective + translate.
    H_true = np.array(
        [
            [0.012, -0.0011, -3.10],
            [0.0009, 0.0135, -1.95],
            [3.0e-5, 1.2e-5, 1.0],
        ],
        dtype=np.float64,
    )

    # Known field-frame points (meters), spread across a ~16x8 field.
    field_pts = np.array(
        [[1.0, 1.0], [15.0, 1.0], [15.0, 7.0], [1.0, 7.0],
         [8.0, 4.0], [4.0, 2.0], [12.0, 6.0], [6.0, 5.5]],
        dtype=np.float64,
    )

    # Forward: field -> image using H_true^-1 (since H_true maps image->field).
    H_inv = np.linalg.inv(H_true)
    img_pts = _apply_h(H_inv, field_pts)

    # Recover image->field homography from the correspondences.
    H_rec, mask = cv2.findHomography(img_pts, field_pts, method=cv2.RANSAC,
                                     ransacReprojThreshold=2.0)
    assert H_rec is not None

    # Apply recovered H to the image points -> should match the original field points.
    field_rec = _apply_h(H_rec, img_pts)
    err = np.linalg.norm(field_rec - field_pts, axis=1)
    assert np.max(err) < 1e-3, f"max field round-trip error {np.max(err):.2e} m"


@pytest.mark.skipif(not HAVE_CV2, reason="cv2 not installed")
def test_field_model_image_to_field_inverse():
    """FieldModel.image_to_field and field_to_image must be exact inverses on the plane."""
    H_true = np.array(
        [[0.012, -0.0011, -3.10], [0.0009, 0.0135, -1.95], [3.0e-5, 1.2e-5, 1.0]],
        dtype=np.float64,
    )
    pose = CameraPose(H=H_true.tolist(), valid=True, n_tags=4, reproj_err_px=0.3)
    fm = FieldModel(FIELD_CFG)
    px = (640.0, 380.0)
    xy = fm.image_to_field(px, pose)
    back = fm.field_to_image(xy, pose)
    assert abs(back[0] - px[0]) < 1e-6 and abs(back[1] - px[1]) < 1e-6


@pytest.mark.skipif(not HAVE_CV2, reason="cv2 needed for fillPoly")
def test_field_mask_and_tiling():
    """field_mask rasterizes the field rectangle into image space; tiling_polygons covers
    it with nx*ny sub-polygons. Sanity-check shapes, coverage, and caching."""
    H_true = np.array(
        [[0.012, -0.0011, -3.10], [0.0009, 0.0135, -1.95], [3.0e-5, 1.2e-5, 1.0]],
        dtype=np.float64,
    )
    pose = CameraPose(H=H_true.tolist(), valid=True, n_tags=4, reproj_err_px=0.3)
    fm = FieldModel(FIELD_CFG)

    hw = (720, 1280)
    mask = fm.field_mask(hw, pose)
    assert mask is not None
    assert mask.shape == hw
    assert mask.max() == 255          # some field pixels are inside the frame
    assert 0 < int((mask > 0).sum()) < hw[0] * hw[1]

    # Caching: a second call with the same pose returns the identical array object.
    mask2 = fm.field_mask(hw, pose)
    assert mask2 is mask

    # Invalid pose -> falls back to cached mask (still the same object here).
    bad = CameraPose(H=H_true.tolist(), valid=False)
    assert fm.field_mask(hw, bad) is mask

    tiles = fm.tiling_polygons(pose, nx=4, ny=2)
    assert len(tiles) == 8
    for poly in tiles:
        assert poly.shape == (4, 2)


def test_field_mask_none_without_pose():
    fm = FieldModel(FIELD_CFG)
    assert fm.field_mask((720, 1280), None) is None
    assert fm.tiling_polygons(None) == []


def test_clamp_field_xy_bounds_and_passthrough():
    """A6.3: clamp_field_xy keeps reported field XY inside [-tol, L+tol] x [-tol, W+tol],
    leaves in-bounds (center-field) points EXACTLY untouched, and passes None through."""
    fm = FieldModel(FIELD_CFG)
    L, W, tol = fm.length_m, fm.width_m, 0.5

    # None passes through unchanged (callers keep their own no-pose fallback).
    assert fm.clamp_field_xy(None) is None

    # Center-field point well inside bounds -> identical (zero delta), the key invariant.
    center = (L / 2.0, W / 2.0)
    out = fm.clamp_field_xy(center)
    assert out == center

    # A point already within the widened tol band is also untouched.
    near = (L + 0.4, W / 2.0)
    assert fm.clamp_field_xy(near) == near

    # Homography-extrapolated off-field points clamp to the widened rectangle edge.
    assert fm.clamp_field_xy((18.0, 9.5)) == (L + tol, W + tol)
    assert fm.clamp_field_xy((-3.0, -2.0)) == (-tol, -tol)

    # Whatever the input, the result is always within the bar's bounds.
    for p in [(100.0, 100.0), (-100.0, 4.0), (8.0, -50.0), (16.5, 8.0)]:
        cx, cy = fm.clamp_field_xy(p)
        assert -tol <= cx <= L + tol and -tol <= cy <= W + tol


def test_in_hub_geometry():
    """Pure field-meter geometry — no deps."""
    fm = FieldModel(FIELD_CFG)
    blue = fm.hub_for("blue")
    cx, cy = blue["center_m"]
    assert fm.in_hub((cx, cy), "blue") is True
    far = (cx + blue["radius_m"] + 1.0, cy)
    assert fm.in_hub(far, "blue") is False
    # tolerance brings a just-outside point in
    edge = (cx + blue["radius_m"] + 0.05, cy)
    assert fm.in_hub(edge, "blue", tol_m=0.2) is True


# ----------------------------------------------------------------------------- #
# Ballistics
# ----------------------------------------------------------------------------- #
def _make_P():
    """A simple but full-rank camera projection matrix P = K [R | t].
    Camera placed downfield looking back toward the field, slight tilt."""
    # Intrinsics (1280x720-ish).
    K = np.array([[900.0, 0.0, 640.0],
                  [0.0, 900.0, 360.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    # Rotation: look roughly along -X of field with a downward pitch. Build from a
    # camera at (8, -6, 3) looking at field center (8, 4, 1).
    cam = np.array([8.0, -6.0, 3.0])
    target = np.array([8.0, 4.0, 1.0])
    fwd = target - cam
    fwd /= np.linalg.norm(fwd)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    # Camera frame: x=right, y=down, z=fwd (OpenCV convention).
    R = np.stack([right, down, fwd], axis=0)   # world->camera rotation
    t = -R @ cam
    Rt = np.hstack([R, t.reshape(3, 1)])
    return K @ Rt


@pytest.mark.skipif(not HAVE_SCIPY, reason="scipy not installed")
def test_ballistics_recovers_known_parabola():
    """Build P, generate the projected pixels of a KNOWN parabola, fit, and assert the
    recovered launch height + velocity are close to ground truth."""
    P = _make_P()

    launch_xy = (8.0, 4.0)
    z0_true = 0.7
    v_true = (1.5, 4.0, 6.0)   # vx, vy, vz

    times = np.linspace(0.0, 1.0, 12)
    theta_true = np.array([v_true[0], v_true[1], v_true[2], z0_true])
    pts3d = ballistics._traj_3d(theta_true, times, launch_xy)
    pixels = ballistics._project(P, pts3d)
    assert np.all(np.isfinite(pixels))

    fit = ballistics.fit_arc(
        observed_px=[tuple(p) for p in pixels],
        times_s=list(times),
        launch_xy_m=launch_xy,
        P=P,
        launch_z_guess_m=0.6,
        rim_height_m=2.1,
    )
    assert fit is not None
    # Reprojection of a noise-free synthetic arc should be sub-pixel.
    assert fit.reproj_err_px < 0.5, fit.reproj_err_px

    vx, vy, vz = fit.velocity_mps
    assert abs(vx - v_true[0]) < 0.3
    assert abs(vy - v_true[1]) < 0.3
    assert abs(vz - v_true[2]) < 0.3
    assert abs(fit.launch_xyz_m[2] - z0_true) < 0.1

    # Apex: z0 + vz^2/(2g)
    apex_expected = z0_true + v_true[2] ** 2 / (2 * ballistics.G)
    assert abs(fit.apex_height_m - apex_expected) < 0.2

    # A clean, well-sampled fit should be high-confidence.
    assert fit.confidence > 0.6, fit.confidence
    # Impact at rim height should be a sensible field point.
    assert fit.impact_xy_m is not None


@pytest.mark.skipif(not HAVE_SCIPY, reason="scipy not installed")
def test_ballistics_too_few_points():
    P = _make_P()
    assert ballistics.fit_arc([(0, 0), (1, 1), (2, 2)], [0, 1, 2], (8.0, 4.0), P) is None
    assert ballistics.fit_arc([(0, 0)] * 5, [0, 1, 2, 3, 4], (8.0, 4.0), None) is None


def test_solve_impact_ground():
    """Pure-physics impact solver: a ball launched straight up returns to z0's ground."""
    # vz=5, z0=1, target ground=0. Descending root: t = (vz + sqrt(vz^2+2 g z0))/g
    xy = ballistics._solve_impact(vz=5.0, z0=1.0, target_z=0.0, vx=2.0, vy=0.0,
                                  launch_xy_m=(0.0, 0.0), t_obs_max=1.0)
    assert xy is not None
    g = ballistics.G
    t_expected = (5.0 + np.sqrt(5.0 ** 2 + 2 * g * 1.0)) / g
    assert abs(xy[0] - 2.0 * t_expected) < 1e-6
    assert abs(xy[1]) < 1e-9


def test_solve_impact_never_reaches_rim():
    """A weak lob that never reaches rim height returns None."""
    assert ballistics._solve_impact(vz=0.5, z0=0.6, target_z=2.1, vx=1.0, vy=0.0,
                                    launch_xy_m=(0.0, 0.0), t_obs_max=1.0) is None
