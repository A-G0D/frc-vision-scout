"""Tests for PoseEstimator wired to the real WPILib AprilTag field map.

All tests that require cv2 or numpy are guarded with importorskip so the file
still collects cleanly in environments where those deps are absent.
"""
from __future__ import annotations

import os

import pytest

# --------------------------------------------------------------------------- #
# Guard optional heavy deps
# --------------------------------------------------------------------------- #
np     = pytest.importorskip("numpy")
cv2    = pytest.importorskip("cv2")

# pupil_apriltags is needed for the real detector; skip gracefully if absent.
try:
    import pupil_apriltags  # noqa: F401
    HAVE_APRIL = True
except Exception:
    HAVE_APRIL = False

from frcscout.geometry.tagmap import load_tag_map, tag_corners_3d
from frcscout.geometry.apriltag import PoseEstimator

_MAP_PATH   = "configs/apriltags_2026.json"
_FIELD_CFG  = "configs/field_2026_rebuilt.yaml"

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_field():
    """Return a minimal FieldModel-like stub that satisfies PoseEstimator.__init__."""
    from frcscout.geometry.field import FieldModel
    return FieldModel(_FIELD_CFG)


def _project_pts(K: "np.ndarray", rvec: "np.ndarray", tvec: "np.ndarray",
                 pts3d: "np.ndarray") -> "np.ndarray":
    """Project Nx3 3D points through K, rvec, tvec -> Nx2 image pixels (cv2.projectPoints)."""
    pts_proj, _ = cv2.projectPoints(
        pts3d.reshape(-1, 1, 3),
        rvec, tvec,
        K, None,
    )
    return pts_proj.reshape(-1, 2)


def _synthetic_K(w: int = 1280, h: int = 720) -> "np.ndarray":
    """A plausible broadcast camera intrinsic matrix."""
    f = 1200.0
    return np.array([[f, 0.0, w / 2.0],
                     [0.0, f, h / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


# --------------------------------------------------------------------------- #
# 1. Tag-map loading
# --------------------------------------------------------------------------- #

def test_pose_estimator_loads_32_tag_map():
    """PoseEstimator.tag_count must equal 32 when the JSON map is present."""
    if not os.path.exists(_MAP_PATH):
        pytest.skip("apriltags_2026.json not present")
    field = _make_field()
    est = PoseEstimator(field)
    assert est.tag_count == 32, (
        f"Expected 32 tags from the real WPILib map, got {est.tag_count}"
    )


def test_pose_estimator_map_dict_populated():
    """_tag_corners_map must have 32 entries with shape (4, 3) each."""
    if not os.path.exists(_MAP_PATH):
        pytest.skip("apriltags_2026.json not present")
    field = _make_field()
    est = PoseEstimator(field)
    assert len(est._tag_corners_map) == 32
    for tid, corners in est._tag_corners_map.items():
        assert isinstance(tid, int)
        assert corners.shape == (4, 3), f"tag {tid}: expected (4,3), got {corners.shape}"


def test_pose_estimator_missing_map_fallback():
    """PoseEstimator must not crash when the tag-map JSON is absent; tag_count==0."""
    field = _make_field()
    # Temporarily monkey-patch the default path so the file won't be found.
    import frcscout.geometry.apriltag as _mod
    orig = _mod._DEFAULT_TAG_MAP_PATH
    _mod._DEFAULT_TAG_MAP_PATH = "configs/__does_not_exist__.json"
    try:
        est = PoseEstimator(field)
        assert est.tag_count == 0
        assert not est._tagmap_loaded
        # _tag_corners_field (YAML fallback) should still be populated.
        assert len(est._tag_corners_field) > 0
    finally:
        _mod._DEFAULT_TAG_MAP_PATH = orig


# --------------------------------------------------------------------------- #
# 2. Synthetic PnP round-trip
# --------------------------------------------------------------------------- #

def _build_synthetic_detections(K, rvec, tvec, tag_ids, tag_map):
    """Project the 3D corners of each tag through the given camera to produce synthetic
    2D 'detected' corners, mimicking pupil_apriltags output."""
    detections = []
    for tid in tag_ids:
        if tid not in tag_map:
            continue
        corners3d = np.array(tag_corners_3d(tag_map[tid], size_m=0.1651), dtype=np.float64)
        corners2d = _project_pts(K, rvec, tvec, corners3d)  # shape (4, 2)
        detections.append((tid, corners2d))
    return detections


class _FakeDet:
    """Minimal stand-in for a pupil_apriltags Detection object."""
    def __init__(self, tag_id: int, corners: "np.ndarray"):
        self.tag_id = tag_id
        self.corners = corners


@pytest.mark.skipif(not HAVE_APRIL, reason="pupil_apriltags not installed")
@pytest.mark.skipif(not os.path.exists(_MAP_PATH), reason="apriltags_2026.json not present")
def test_pnp_roundtrip_with_real_map():
    """Given synthetic tag detections (projected from a known camera pose), PoseEstimator
    should return CameraPose(valid=True) with low reproj_err_px."""
    tag_map = load_tag_map(_MAP_PATH)["tags"]

    K = _synthetic_K(1280, 720)
    # Camera placed ~8 m above field, looking down at a slight angle.
    # rvec/tvec chosen so several mid-field tags project into the frame.
    rvec_true = np.array([1.2, 0.0, 0.0], dtype=np.float64)    # ~69 deg tilt around x
    tvec_true = np.array([-8.0, 0.5, 12.0], dtype=np.float64)  # camera pos in cam frame

    # Use tags 1–6 (all present in the 2026 map); keep whichever project to valid pixels.
    test_ids = [1, 2, 3, 4, 5, 6, 7, 8]
    dets_raw = _build_synthetic_detections(K, rvec_true, tvec_true, test_ids, tag_map)

    # Only keep detections whose corners fall inside the 1280x720 frame.
    dets_in_frame = [
        (tid, c2d)
        for (tid, c2d) in dets_raw
        if np.all(c2d[:, 0] >= 0) and np.all(c2d[:, 0] < 1280)
        and np.all(c2d[:, 1] >= 0) and np.all(c2d[:, 1] < 720)
    ]

    if len(dets_in_frame) < 2:
        pytest.skip("Not enough synthetic tags project into frame for this pose")

    field = _make_field()
    est = PoseEstimator(field)

    # Inject the synthetic detections by patching the detector.
    fake_detections = [_FakeDet(tid, c2d) for (tid, c2d) in dets_in_frame]

    # Provide exact K so PnP gets the right intrinsics.
    # Wrap estimate() to inject fake detections without a real video frame.
    import frcscout.geometry.apriltag as _mod

    class _FakeDetector:
        def detect(self, gray):
            return fake_detections

    est._detector = _FakeDetector()

    # Make a gray frame of the right size.
    gray_frame = np.zeros((720, 1280), dtype=np.uint8)
    pose = est.estimate(gray_frame, frame_idx=0, K=K)

    assert pose is not None, "estimate() returned None; not enough tags?"
    assert pose.valid, (
        f"Expected valid=True, got valid={pose.valid}, "
        f"n_tags={pose.n_tags}, reproj_err_px={pose.reproj_err_px:.3f}"
    )
    assert pose.reproj_err_px < 2.0, (
        f"Reprojection error too large: {pose.reproj_err_px:.3f} px (threshold 2.0 px)"
    )
    assert pose.rvec is not None, "rvec should be populated after PnP"
    assert pose.tvec is not None, "tvec should be populated after PnP"
    assert pose.H is not None
    assert len(pose.H) == 3 and len(pose.H[0]) == 3


# --------------------------------------------------------------------------- #
# 3. Missing-map graceful fallback (no cv2 / pupil needed beyond what importorskip gives)
# --------------------------------------------------------------------------- #

def test_missing_map_valid_false_no_crash():
    """With no JSON map and no actual video (None frame), estimate() returns None gracefully."""
    import frcscout.geometry.apriltag as _mod
    orig = _mod._DEFAULT_TAG_MAP_PATH
    _mod._DEFAULT_TAG_MAP_PATH = "configs/__missing__.json"
    try:
        field = _make_field()
        est = PoseEstimator(field)
        # No detector available (no pupil_apriltags), so _ensure() would fail.
        # But with frame=None, estimate() returns None immediately — no crash.
        result = est.estimate(None, frame_idx=0)
        assert result is None
    finally:
        _mod._DEFAULT_TAG_MAP_PATH = orig
