"""Tests for refine/hub_regions.py — HUB image-localization.

All pure-logic (point-in-polygon, fraction scaling, Mode A, mode selection) runs with
NO heavy deps. Mode B uses a tiny fake field model that mimics FieldModel's interface
(image_to_field / field_to_image / in_hub / hub_for), so these tests never need
numpy/cv2/scipy or a real homography.
"""
from __future__ import annotations

import math

import pytest

from frcscout.io.config import Config
from frcscout.io.types import CameraPose
from frcscout.refine.hub_regions import (
    HubRegions,
    point_in_polygon,
    scale_fraction_polygon,
)

FIELD_CFG = "configs/field_2026_rebuilt.yaml"

# A unit square polygon, used across the pure-geometry tests.
SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


# ----------------------------------------------------------------------------- #
# Pure point-in-polygon
# ----------------------------------------------------------------------------- #
def test_point_in_polygon_inside_outside():
    assert point_in_polygon((5.0, 5.0), SQUARE) is True
    assert point_in_polygon((50.0, 5.0), SQUARE) is False
    assert point_in_polygon((-1.0, 5.0), SQUARE) is False
    assert point_in_polygon((5.0, -0.01), SQUARE) is False


def test_point_in_polygon_boundary_is_inside():
    # Points exactly on an edge / vertex count as inside.
    assert point_in_polygon((0.0, 5.0), SQUARE) is True      # left edge
    assert point_in_polygon((10.0, 10.0), SQUARE) is True    # corner vertex
    assert point_in_polygon((5.0, 0.0), SQUARE) is True      # bottom edge


def test_point_in_polygon_tolerance_band():
    # Just outside the right edge by 0.5px: out with no tol, in with tol >= 0.5.
    p = (10.5, 5.0)
    assert point_in_polygon(p, SQUARE, tol=0.0) is False
    assert point_in_polygon(p, SQUARE, tol=0.5) is True
    assert point_in_polygon(p, SQUARE, tol=1.0) is True
    # Negative tol is clamped to 0 (no erosion) — a truly-outside point stays out.
    assert point_in_polygon(p, SQUARE, tol=-5.0) is False


def test_point_in_polygon_degenerate():
    assert point_in_polygon((0.0, 0.0), []) is False
    assert point_in_polygon((0.0, 0.0), [(0.0, 0.0), (1.0, 1.0)]) is False  # 2 pts, no area


def test_point_in_polygon_concave():
    # An arrow / chevron 'V' shape: the notch between the prongs is OUTSIDE.
    chevron = [(0.0, 0.0), (10.0, 8.0), (20.0, 0.0), (10.0, 20.0)]
    assert point_in_polygon((10.0, 15.0), chevron) is True    # in the upper body
    assert point_in_polygon((10.0, 2.0), chevron) is False    # inside the bottom notch
    assert point_in_polygon((1.0, 1.0), chevron) is True      # left prong


def test_point_in_polygon_winding_independent():
    cw = list(reversed(SQUARE))
    assert point_in_polygon((5.0, 5.0), cw) is True
    assert point_in_polygon((50.0, 5.0), cw) is False


def test_scale_fraction_polygon():
    frac = [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0)]
    px = scale_fraction_polygon(frac, 1280, 720)
    assert px == [(0.0, 0.0), (640.0, 0.0), (640.0, 720.0)]


# ----------------------------------------------------------------------------- #
# Mode A — config-driven image polygons
# ----------------------------------------------------------------------------- #
def _cfg_two_quadrants() -> dict:
    """Synthetic regions: blue = left-half band, red = right-half band of the frame."""
    return {
        "hub_image_regions": {
            "blue": [[0.10, 0.20], [0.40, 0.20], [0.40, 0.80], [0.10, 0.80]],
            "red":  [[0.60, 0.20], [0.90, 0.20], [0.90, 0.80], [0.60, 0.80]],
        }
    }


def test_mode_a_in_hub_from_dict():
    hr = HubRegions(_cfg_two_quadrants(), frame_wh=(1000, 1000))
    # Centre of the blue band: x in [100,400], y in [200,800].
    assert hr.in_hub((250.0, 500.0), "blue") is True
    assert hr.in_hub((250.0, 500.0), "red") is False
    # Centre of the red band.
    assert hr.in_hub((750.0, 500.0), "red") is True
    assert hr.in_hub((750.0, 500.0), "blue") is False
    # The gap in the middle belongs to neither.
    assert hr.in_hub((500.0, 500.0), "blue") is False
    assert hr.in_hub((500.0, 500.0), "red") is False


def test_mode_a_resolution_independent():
    cfg = _cfg_two_quadrants()
    # Same fractional point (0.25, 0.5) is inside blue at any resolution.
    for wh in [(640, 360), (1280, 720), (1920, 1080)]:
        hr = HubRegions(cfg, frame_wh=wh)
        px = (0.25 * wh[0], 0.5 * wh[1])
        assert hr.in_hub(px, "blue") is True, wh


def test_mode_a_per_call_frame_wh_overrides():
    hr = HubRegions(_cfg_two_quadrants())  # no instance frame size
    # Without a frame size, Mode A cannot scale -> always False.
    assert hr.in_hub((250.0, 500.0), "blue") is False
    # Supplying it per-call works.
    assert hr.in_hub((250.0, 500.0), "blue", frame_wh=(1000, 1000)) is True


def test_mode_a_tolerance():
    hr = HubRegions(_cfg_two_quadrants(), frame_wh=(1000, 1000))
    # x=410 is 10px right of the blue band's right edge (x in [100,400]).
    assert hr.in_hub((410.0, 500.0), "blue") is False
    assert hr.in_hub((410.0, 500.0), "blue", tol_px=12.0) is True


def test_mode_a_missing_alliance_polygon():
    hr = HubRegions({"hub_image_regions": {"blue": [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]]}},
                    frame_wh=(1000, 1000))
    assert hr.has_mode_a("blue") is True
    assert hr.has_mode_a("red") is False
    assert hr.in_hub((250.0, 250.0), "red") is False  # no red polygon -> False


def test_pixel_polygon_and_none_paths():
    hr = HubRegions(_cfg_two_quadrants(), frame_wh=(1000, 1000))
    poly = hr.pixel_polygon("blue")
    assert poly == [(100.0, 200.0), (400.0, 200.0), (400.0, 800.0), (100.0, 800.0)]
    # No frame size anywhere -> None.
    assert HubRegions(_cfg_two_quadrants()).pixel_polygon("blue") is None
    # Unknown alliance -> None.
    assert hr.pixel_polygon("green") is None


def test_polygons_for_drawing_mode_a():
    hr = HubRegions(_cfg_two_quadrants(), frame_wh=(1000, 1000))
    polys = hr.polygons()  # no pose -> Mode A, both alliances
    assert set(polys.keys()) == {"blue", "red"}
    assert polys["blue"][0] == (100.0, 200.0)
    # Restrict to one alliance.
    only_red = hr.polygons(alliance="red")
    assert set(only_red.keys()) == {"red"}


def test_empty_config_has_no_regions():
    hr = HubRegions({})
    assert hr.has_mode_a() is False
    assert hr.in_hub((1.0, 1.0), "blue", frame_wh=(100, 100)) is False
    assert hr.polygons(frame_wh=(100, 100)) == {}


# ----------------------------------------------------------------------------- #
# Loading from the real field config
# ----------------------------------------------------------------------------- #
def test_from_field_config_loads_polygons():
    hr = HubRegions.from_field_config(FIELD_CFG, frame_wh=(1280, 720))
    # The shipped config defines both HUB image regions.
    assert hr.has_mode_a("blue") is True
    assert hr.has_mode_a("red") is True
    blue = hr.pixel_polygon("blue")
    assert blue is not None and len(blue) >= 3
    # Polygon vertices land inside the frame bounds.
    for x, y in blue:
        assert 0.0 <= x <= 1280.0 and 0.0 <= y <= 720.0


def test_config_object_accepted():
    cfg = Config(_cfg_two_quadrants())
    hr = HubRegions(cfg, frame_wh=(1000, 1000))
    assert hr.in_hub((250.0, 500.0), "blue") is True


# ----------------------------------------------------------------------------- #
# Mode B — homography (fake field model, no numpy/cv2 needed)
# ----------------------------------------------------------------------------- #
class _FakeField:
    """Mimics FieldModel just enough for hub_regions Mode B.

    image->field is an affine px->meters map (scale ``s`` + offset). HUBs are circles
    in field meters, exactly like the real config's ``hubs`` block.
    """

    def __init__(self, s=0.01, ox=0.0, oy=0.0):
        self.s, self.ox, self.oy = s, ox, oy
        self.hubs = {
            "blue": {"center_m": [3.20, 4.10], "radius_m": 0.80},
            "red": {"center_m": [13.26, 4.10], "radius_m": 0.80},
        }

    def hub_for(self, alliance):
        return self.hubs.get(alliance, {})

    def image_to_field(self, px, pose):
        return (px[0] * self.s + self.ox, px[1] * self.s + self.oy)

    def field_to_image(self, xy, pose):
        return ((xy[0] - self.ox) / self.s, (xy[1] - self.oy) / self.s)

    def in_hub(self, field_xy, alliance, tol_m=0.0):
        hub = self.hubs.get(alliance)
        if not hub:
            return False
        cx, cy = hub["center_m"]
        r = hub["radius_m"] + tol_m
        return (field_xy[0] - cx) ** 2 + (field_xy[1] - cy) ** 2 <= r * r


def _valid_pose():
    # H is unused by _FakeField but must be a 3x3 so a real FieldModel wouldn't choke.
    return CameraPose(H=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], valid=True, n_tags=4)


def test_mode_selection_a_vs_b():
    fake = _FakeField()
    # With a field model but no/invalid pose -> Mode A.
    hr = HubRegions(_cfg_two_quadrants(), frame_wh=(1000, 1000), field_model=fake)
    assert hr.mode_for(None) == "A"
    assert hr.mode_for(CameraPose(H=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], valid=False)) == "A"
    # With a valid pose -> Mode B.
    assert hr.mode_for(_valid_pose()) == "B"
    # No field model at all -> always A.
    hr_noa = HubRegions(_cfg_two_quadrants(), frame_wh=(1000, 1000))
    assert hr_noa.mode_for(_valid_pose()) == "A"


def test_mode_b_in_hub_uses_field_circle():
    fake = _FakeField(s=0.01)  # 1px = 0.01m
    hr = HubRegions(_cfg_two_quadrants(), frame_wh=(1000, 1000), field_model=fake)
    pose = _valid_pose()
    # Blue hub centre (3.20, 4.10)m -> pixel (320, 410).
    assert hr.in_hub((320.0, 410.0), "blue", pose=pose) is True
    # 0.80m radius = 80px; a point ~100px away in x is outside.
    assert hr.in_hub((320.0 + 100.0, 410.0), "blue", pose=pose) is False
    # Just outside (90px ~ 0.90m) brought in by tol_m.
    assert hr.in_hub((320.0 + 90.0, 410.0), "blue", pose=pose) is False
    assert hr.in_hub((320.0 + 90.0, 410.0), "blue", pose=pose, tol_m=0.2) is True
    # Red hub centre (13.26, 4.10)m -> pixel (1326, 410).
    assert hr.in_hub((1326.0, 410.0), "red", pose=pose) is True
    assert hr.in_hub((1326.0, 410.0), "blue", pose=pose) is False


def test_mode_b_overrides_mode_a_disagreement():
    """A pixel inside the Mode-A blue band but OUTSIDE the Mode-B field circle must
    follow Mode B when a valid pose is present (B is the authoritative path)."""
    fake = _FakeField(s=0.01)
    hr = HubRegions(_cfg_two_quadrants(), frame_wh=(1000, 1000), field_model=fake)
    # (250,500): inside Mode-A blue band, but field (2.5,5.0)m is >0.8m from (3.2,4.1).
    assert hr.in_hub((250.0, 500.0), "blue") is True                      # Mode A
    assert hr.in_hub((250.0, 500.0), "blue", pose=_valid_pose()) is False  # Mode B


def test_mode_b_polygons_projects_circle():
    fake = _FakeField(s=0.01)
    hr = HubRegions({}, field_model=fake)  # no Mode-A polygons at all
    polys = hr.polygons(pose=_valid_pose(), circle_samples=16)
    assert set(polys.keys()) == {"blue", "red"}
    blue = polys["blue"]
    assert len(blue) == 16
    # Every projected vertex is ~80px (=0.80m / 0.01) from the projected centre (320,410).
    for x, y in blue:
        d = math.hypot(x - 320.0, y - 410.0)
        assert abs(d - 80.0) < 1e-6


def test_mode_b_polygons_falls_back_to_mode_a_without_pose():
    fake = _FakeField()
    hr = HubRegions(_cfg_two_quadrants(), frame_wh=(1000, 1000), field_model=fake)
    # No pose -> the drawing path uses Mode-A scaled polygons, not the circle.
    polys = hr.polygons(pose=None)
    assert polys["blue"][0] == (100.0, 200.0)
