"""Tests for the real WPILib AprilTag map loader (geolocation calibration anchor)."""
from __future__ import annotations

import os

from frcscout.geometry.tagmap import load_tag_map, tag_corners_3d, _quat_to_R

_JSON = "configs/apriltags_2026.json"


def test_loads_32_tags():
    if not os.path.exists(_JSON):
        import pytest; pytest.skip("tag map not downloaded")
    m = load_tag_map(_JSON)
    assert len(m["tags"]) == 32
    assert 15.5 < m["field"]["length"] < 17.0
    assert 7.0 < m["field"]["width"] < 9.0
    one = m["tags"][1]
    assert len(one["xyz"]) == 3 and len(one["quat"]) == 4


def test_quat_identity_is_identity():
    R = _quat_to_R(1, 0, 0, 0)
    assert abs(R[0][0] - 1) < 1e-9 and abs(R[1][1] - 1) < 1e-9 and abs(R[2][2] - 1) < 1e-9
    assert abs(R[0][1]) < 1e-9 and abs(R[1][0]) < 1e-9


def test_corners_identity_rotation():
    tag = {"xyz": (1.0, 2.0, 0.5), "quat": (1, 0, 0, 0)}  # no rotation
    c = tag_corners_3d(tag, size_m=0.2)
    assert len(c) == 4
    # bottom-left corner = center + (-0.1,-0.1,0)
    assert abs(c[0][0] - 0.9) < 1e-6 and abs(c[0][1] - 1.9) < 1e-6 and abs(c[0][2] - 0.5) < 1e-6
    # all corners share the tag's z under identity rotation
    assert all(abs(p[2] - 0.5) < 1e-6 for p in c)


def test_corners_quat_normalized():
    # a non-unit quaternion should still produce a valid (normalized) rotation
    tag = {"xyz": (0, 0, 0), "quat": (2, 0, 0, 0)}
    c = tag_corners_3d(tag, size_m=1.0)
    assert abs(c[0][0] - (-0.5)) < 1e-6
