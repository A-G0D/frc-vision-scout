"""HUB localization for scored-detection — *where in the IMAGE is the HUB?*

This is the image-space counterpart to ``geometry/field.py``'s ``in_hub`` (which
answers the question in FIELD METERS). The shot/score logic needs to know whether a
ball's *pixel* position lands inside the scoring HUB. There are two ways to answer,
and this module exposes both behind one ``in_hub(px, alliance) -> bool`` call:

Mode A — config-driven approximate HUB image polygons
    For the STATIC broadcast camera the HUBs sit in a roughly fixed part of the frame.
    We store an approximate polygon per alliance as vertices in FRACTIONS of (W, H)
    (the same resolution-independent convention as the overlay-OCR ROIs), scale them
    to the live frame size, and run a plain point-in-polygon test. No homography
    needed — this is the always-available fallback, and the only option before a
    camera pose is recovered. It is *approximate*: a hard pan/zoom invalidates it.

Mode B — field homography (preferred when a pose exists)
    Given a valid :class:`~frcscout.io.types.CameraPose`, map the pixel to the ground
    plane in field meters (``FieldModel.image_to_field``) and defer to the exact
    field-coords HUB test (``FieldModel.in_hub`` — a circle of ``radius_m`` about the
    HUB centre). This survives any camera pan/zoom because the decision happens in
    field coordinates, not pixels.

Selection
    A single :class:`HubRegions` instance prefers Mode B per call when it holds a
    field model AND is given a valid pose; otherwise it falls back to Mode A. So the
    SAME ``in_hub(px, alliance)`` call upgrades automatically the moment a pose lands,
    with no caller change. ``mode_for(pose)`` reports which path a call would take.

Drawing
    ``polygons(...)`` returns HUB outlines in PIXELS for overlay rendering. In Mode A
    it scales the config fractions to (W, H). In Mode B it samples the field circle
    and projects each sample to pixels via ``FieldModel.field_to_image`` — i.e. the
    HUB drawn exactly where the homography places it.

Purity
    The point-in-polygon test and fraction scaling are pure python (no numpy), so the
    geometry is unit-testable on a bare interpreter. numpy is imported guarded and is
    only touched on the Mode-B projection path (which itself needs a pose + field
    model, exercised by the geometry-layer tests, not required to import this file).
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

from frcscout.io.config import Config
from frcscout.io.types import Alliance, CameraPose

# --- guarded heavy import --------------------------------------------------
# The module must PARSE and its pure-logic tests must run without numpy. numpy is
# only needed on the Mode-B circle-projection helper; everything else is pure python.
try:  # pragma: no cover - exercised both with and without numpy installed
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None  # type: ignore


Point = Tuple[float, float]
Polygon = List[Point]            # a closed polygon as an ordered list of (x, y) vertices
ALLIANCES: Tuple[str, str] = ("blue", "red")


# ===========================================================================
# Pure geometry: point-in-polygon (no numpy)
# ===========================================================================

def point_in_polygon(px: Point, poly: Sequence[Point], tol: float = 0.0) -> bool:
    """Even-odd ray-casting point-in-polygon test (pure python).

    Returns True if ``px`` is strictly inside ``poly`` OR within ``tol`` of its
    boundary. The polygon is treated as closed (the last vertex links to the first);
    it need not be convex. Winding direction does not matter.

    The boundary itself is treated as INSIDE (``<=`` comparisons + the explicit
    ``tol`` band), so a point sitting exactly on an edge or vertex counts as in —
    the behaviour scoring logic wants for a ball resting on the HUB lip.

    Args:
        px: the (x, y) query point.
        poly: >=3 vertices as (x, y). Fewer than 3 -> always False (no area).
        tol: extra distance band around the boundary counted as inside. Negative
            values are clamped to 0 (use a shrunk polygon if you need erosion).
    """
    n = len(poly)
    if n < 3:
        return False
    tol = max(0.0, float(tol))
    x, y = float(px[0]), float(px[1])

    # On/near boundary -> inside (covers edges, vertices, and the tol band).
    for i in range(n):
        if _dist_point_to_segment(x, y, poly[i], poly[(i + 1) % n]) <= tol:
            return True

    # Even-odd crossing number. A horizontal ray to +x; count edge crossings.
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        # Does edge (j->i) straddle the horizontal line y? Half-open in y to avoid
        # double-counting a vertex that lies exactly on the ray.
        if (yi > y) != (yj > y):
            # x of the intersection of edge with the horizontal line at y.
            x_cross = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def _dist_point_to_segment(px: float, py: float, a: Point, b: Point) -> float:
    """Euclidean distance from point (px,py) to segment a-b (pure python)."""
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    dx, dy = bx - ax, by - ay
    seg_len2 = dx * dx + dy * dy
    if seg_len2 <= 0.0:  # degenerate segment -> distance to the point
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len2
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def scale_fraction_polygon(frac_poly: Sequence[Point], w: int, h: int) -> Polygon:
    """Scale a polygon given in (W,H) fractions to absolute pixels for size (w,h)."""
    return [(float(fx) * float(w), float(fy) * float(h)) for fx, fy in frac_poly]


# ===========================================================================
# HubRegions — the public localizer
# ===========================================================================

class HubRegions:
    """Localize the scoring HUB in image space, Mode A (polygons) or Mode B (pose).

    Construct from a field config (or a raw mapping). The frame size ``(w, h)`` is
    needed to turn the fractional Mode-A polygons into pixels; pass it at construction
    or per call. A ``field_model`` (anything exposing ``image_to_field(px, pose)``,
    ``field_to_image(xy, pose)`` and ``in_hub(field_xy, alliance, tol_m)`` — i.e. the
    real ``FieldModel``) enables Mode B; without one, only Mode A is available.
    """

    def __init__(
        self,
        config: Config | dict | None = None,
        frame_wh: Optional[Tuple[int, int]] = None,
        field_model: object | None = None,
        key: str = "hub_image_regions",
    ):
        cfg = config if config is not None else {}
        # Accept a Config (dotted get) or a plain dict.
        if isinstance(cfg, Config):
            regions = cfg.get(key, {}) or {}
        elif isinstance(cfg, dict):
            regions = cfg.get(key, {}) or {}
        else:  # pragma: no cover - defensive
            regions = {}
        # Normalize: {alliance: [[fx,fy], ...]} -> {alliance: [(fx,fy), ...]}.
        self._frac: dict[str, Polygon] = {}
        for ally, verts in regions.items():
            self._frac[str(ally)] = [(float(v[0]), float(v[1])) for v in verts]
        self.frame_wh = frame_wh
        self.field_model = field_model

    # ---- config / loading helpers ---------------------------------------
    @classmethod
    def from_field_config(
        cls,
        path: str,
        frame_wh: Optional[Tuple[int, int]] = None,
        field_model: object | None = None,
    ) -> "HubRegions":
        """Build from a field-config YAML path (reads its ``hub_image_regions`` block)."""
        return cls(Config.load(path), frame_wh=frame_wh, field_model=field_model)

    def has_mode_a(self, alliance: Optional[str] = None) -> bool:
        """True if a Mode-A polygon is configured (for ``alliance`` if given, else any)."""
        if alliance is None:
            return any(len(p) >= 3 for p in self._frac.values())
        return len(self._frac.get(alliance, [])) >= 3

    def mode_for(self, pose: Optional[CameraPose]) -> str:
        """Report which path :meth:`in_hub` would take for ``pose``: ``"B"`` if a field
        model + a valid pose are present, else ``"A"`` (its availability is separate —
        see :meth:`has_mode_a`)."""
        if self.field_model is not None and _pose_valid(pose):
            return "B"
        return "A"

    # ---- the core query --------------------------------------------------
    def in_hub(
        self,
        px: Point,
        alliance: Alliance,
        pose: Optional[CameraPose] = None,
        frame_wh: Optional[Tuple[int, int]] = None,
        tol_px: float = 0.0,
        tol_m: float = 0.0,
    ) -> bool:
        """Is image point ``px`` inside ``alliance``'s HUB?

        Mode B (field model + valid ``pose``): unproject ``px`` to field meters and use
        the exact field-coords circle test, widened by ``tol_m``. Mode A otherwise:
        scale the configured fractional polygon to pixels and run point-in-polygon,
        widened by ``tol_px``. Returns False if the chosen mode has nothing to test
        against (no polygon configured / projection failed).

        ``px`` is always interpreted in pixels of the CURRENT frame; ``frame_wh``
        (this call, else the instance default) only matters for Mode A scaling.
        """
        if self.mode_for(pose) == "B":
            return self._in_hub_mode_b(px, alliance, pose, tol_m)  # type: ignore[arg-type]
        return self._in_hub_mode_a(px, alliance, frame_wh, tol_px)

    def _in_hub_mode_a(
        self,
        px: Point,
        alliance: str,
        frame_wh: Optional[Tuple[int, int]],
        tol_px: float,
    ) -> bool:
        poly = self.pixel_polygon(alliance, frame_wh)
        if poly is None or len(poly) < 3:
            return False
        return point_in_polygon(px, poly, tol=tol_px)

    def _in_hub_mode_b(
        self,
        px: Point,
        alliance: str,
        pose: CameraPose,
        tol_m: float,
    ) -> bool:
        fm = self.field_model
        try:
            field_xy = fm.image_to_field((float(px[0]), float(px[1])), pose)  # type: ignore[union-attr]
        except Exception:  # pragma: no cover - bad/singular pose
            return False
        return bool(fm.in_hub(field_xy, alliance, tol_m=tol_m))  # type: ignore[union-attr]

    # ---- drawing ---------------------------------------------------------
    def pixel_polygon(
        self,
        alliance: str,
        frame_wh: Optional[Tuple[int, int]] = None,
    ) -> Optional[Polygon]:
        """The Mode-A HUB polygon for ``alliance`` in PIXELS, or None if unavailable.

        Needs a frame size (this call, else the instance default). This is the polygon
        that Mode A tests against and the natural thing to draw when no pose exists.
        """
        frac = self._frac.get(alliance)
        if not frac or len(frac) < 3:
            return None
        wh = frame_wh if frame_wh is not None else self.frame_wh
        if wh is None:
            return None
        w, h = int(wh[0]), int(wh[1])
        return scale_fraction_polygon(frac, w, h)

    def polygons(
        self,
        alliance: Optional[str] = None,
        pose: Optional[CameraPose] = None,
        frame_wh: Optional[Tuple[int, int]] = None,
        circle_samples: int = 24,
    ) -> dict[str, Polygon]:
        """HUB outlines in PIXELS for drawing, keyed by alliance.

        With a field model + valid ``pose`` (Mode B), each HUB circle is sampled into a
        ``circle_samples``-gon and projected to pixels via ``field_to_image`` — the HUB
        drawn exactly where the homography puts it. Otherwise (Mode A) the configured
        fractional polygon is scaled to pixels. Alliances with nothing to draw are
        omitted. ``alliance`` restricts the result to that one alliance.
        """
        want: Iterable[str] = (alliance,) if alliance is not None else ALLIANCES
        out: dict[str, Polygon] = {}
        use_b = self.field_model is not None and _pose_valid(pose)
        for ally in want:
            poly: Optional[Polygon] = None
            if use_b:
                poly = self._hub_circle_px(ally, pose, circle_samples)  # type: ignore[arg-type]
            if poly is None:  # Mode A (also a fallback if the projection failed)
                poly = self.pixel_polygon(ally, frame_wh)
            if poly is not None and len(poly) >= 3:
                out[ally] = poly
        return out

    def _hub_circle_px(
        self,
        alliance: str,
        pose: CameraPose,
        samples: int,
    ) -> Optional[Polygon]:
        """Sample alliance's field-coords HUB circle and project each point to pixels.

        Returns None if the field model lacks the HUB, the projection backend (numpy)
        is missing, or the homography is singular — callers fall back to Mode A.
        """
        fm = self.field_model
        if fm is None or _np is None:
            return None
        hub = None
        hub_for = getattr(fm, "hub_for", None)
        if callable(hub_for):
            hub = hub_for(alliance)
        if not hub or "center_m" not in hub or "radius_m" not in hub:
            return None
        cx, cy = float(hub["center_m"][0]), float(hub["center_m"][1])
        r = float(hub["radius_m"])
        n = max(3, int(samples))
        poly: Polygon = []
        for k in range(n):
            ang = 2.0 * math.pi * k / n
            fx, fy = cx + r * math.cos(ang), cy + r * math.sin(ang)
            try:
                ipx = fm.field_to_image((fx, fy), pose)  # type: ignore[union-attr]
            except Exception:  # pragma: no cover - singular H
                return None
            poly.append((float(ipx[0]), float(ipx[1])))
        return poly


# ===========================================================================
# Small helpers
# ===========================================================================

def _pose_valid(pose: Optional[CameraPose]) -> bool:
    """True iff ``pose`` is present and flagged valid (mirrors field.py's guard)."""
    return pose is not None and bool(getattr(pose, "valid", False))


__all__ = [
    "HubRegions",
    "point_in_polygon",
    "scale_fraction_polygon",
]
