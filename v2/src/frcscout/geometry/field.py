"""Field model + image<->field transforms.

Holds the field/tag/HUB geometry (configs/field_2026_rebuilt.yaml) and applies the
recovered homography to put tracks into field coordinates. Ground-plane objects map
exactly; airborne balls are deferred to ballistics.py (a homography would wrongly place
a high ball far downfield — the z!=0 parallax).
"""
from __future__ import annotations

import math

import numpy as np

from frcscout.io.config import Config
from frcscout.io.types import CameraPose, Track


class _PoseLike:
    """Minimal duck-typed stand-in exposing only ``.H`` so ``image_to_field`` can be
    reused with a bare numpy homography (used by the gate's local px->m scale probe)."""

    __slots__ = ("H",)

    def __init__(self, H):
        self.H = H


class FieldModel:
    def __init__(self, field_config_path: str):
        self.cfg = Config.load(field_config_path)
        self.tag_family = self.cfg.get("tag_family", "tag36h11")
        self.tag_size_m = self.cfg.get("tag_size_m", 0.1651)
        self.hubs = self.cfg.get("hubs", {})
        self.towers = self.cfg.get("towers", {})
        self.tags = self.cfg.get("apriltags", {})
        # Field bounds (meters). The YAML carries placeholders (16.46 x 8.23); the real
        # WPILib AprilTag map JSON ships the authoritative length/width in its `field`
        # block (16.518 x 8.043). Prefer the JSON map when it is loadable so the field
        # gate tests against true bounds; fall back to the YAML config otherwise.
        self.length_m = float(self.cfg.get("field.length_m", 16.46))
        self.width_m = float(self.cfg.get("field.width_m", 8.23))
        self._load_field_bounds_from_tagmap()
        self._field_poly_px = None          # cached image-space field polygon (from homography)
        self._field_mask = None             # cached binary mask
        self._mask_hw: tuple[int, int] | None = None
        self._mask_pose_id: int | None = None
        # back-projection (field-gate) cache, keyed by id(pose)
        self._gate_pose_id: int | None = None
        self._gate_H: np.ndarray | None = None

    def _load_field_bounds_from_tagmap(self) -> None:
        """Override length/width with the official WPILib map's `field` block when present
        (authoritative real meters). Best-effort: any failure leaves the YAML values."""
        map_path = self.cfg.get("apriltag_map_path") or "configs/apriltags_2026.json"
        try:
            import os
            from frcscout.geometry.tagmap import load_tag_map
            if os.path.exists(map_path):
                fb = load_tag_map(map_path).get("field", {}) or {}
                L, Wd = float(fb.get("length", 0.0)), float(fb.get("width", 0.0))
                if L > 0.0 and Wd > 0.0:
                    self.length_m, self.width_m = L, Wd
        except Exception:
            pass  # keep YAML placeholders

    # ----- transforms ------------------------------------------------------- #
    def image_to_field(self, px: tuple[float, float], pose: CameraPose) -> tuple[float, float]:
        """Map an image point on the ground plane to field meters via the homography."""
        H = np.asarray(pose.H, dtype=np.float64)
        v = H @ np.array([px[0], px[1], 1.0])
        return float(v[0] / v[2]), float(v[1] / v[2])

    def field_to_image(self, xy: tuple[float, float], pose: CameraPose) -> tuple[float, float]:
        """Inverse map: a ground-plane field point (meters) -> image px via H^-1."""
        H = np.asarray(pose.H, dtype=np.float64)
        Hinv = np.linalg.inv(H)
        v = Hinv @ np.array([xy[0], xy[1], 1.0])
        return float(v[0] / v[2]), float(v[1] / v[2])

    def clamp_field_xy(self, xy: tuple[float, float] | None,
                       tol_m: float = 0.5) -> tuple[float, float] | None:
        """Clamp a projected field XY (meters) into the field rectangle widened by ``tol_m``.

        WHY: the homography is fit on a CLUSTER of recovered tags; an edge robot whose
        floor-contact point sits past that tag hull back-projects via the same H to absurd
        coordinates (~18 m on a 16.5 m field — pure extrapolation, not a real position).
        A pure min/max clamp to [-tol, length+tol] x [-tol, width+tol] guarantees no
        reported field_xy_m ever lands off-field, while leaving every point already inside
        those bounds EXACTLY untouched (center-field robots see zero delta). Returns None
        unchanged so callers can keep their own no-pose fallback."""
        if xy is None:
            return None
        x, y = float(xy[0]), float(xy[1])
        x = min(max(x, -tol_m), self.length_m + tol_m)
        y = min(max(y, -tol_m), self.width_m + tol_m)
        return (x, y)

    def project_tracks(self, tracks: list[Track], pose: CameraPose, airborne: bool = False):
        """Fill track.field_xy_m. For airborne balls the homography is only the GROUND
        intersection of the ray — ballistics.py refines the true 3D arc, so we flag those
        tracks (is_flight) rather than trusting the planar fix."""
        if pose is None or not pose.valid:
            return
        for t in tracks:
            t.field_xy_m = self.clamp_field_xy(self.image_to_field(t.centroid_px, pose))
            if airborne:
                # The planar fix is the ray's ground intersection only; mark for ballistics.
                t.is_flight = True

    # ----- field-region masking (image space) ------------------------------ #
    def _field_corners_m(self) -> np.ndarray:
        """The four field-rectangle corners in field meters (ground plane), CCW."""
        return np.array(
            [
                [0.0, 0.0],
                [self.length_m, 0.0],
                [self.length_m, self.width_m],
                [0.0, self.width_m],
            ],
            dtype=np.float64,
        )

    def field_polygon_px(self, pose: CameraPose) -> np.ndarray | None:
        """The field rectangle projected into image space as a 4x2 px polygon (via H^-1)."""
        if pose is None or not getattr(pose, "valid", False):
            return None
        H = np.asarray(pose.H, dtype=np.float64)
        try:
            Hinv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return None
        corners_m = self._field_corners_m()
        ones = np.ones((corners_m.shape[0], 1), dtype=np.float64)
        hom = np.hstack([corners_m, ones]) @ Hinv.T
        poly = hom[:, :2] / hom[:, 2:3]
        return poly

    def field_mask(self, hw: tuple[int, int], pose: CameraPose | None = None):
        """Binary (uint8 0/255) mask of the field region in image space, for the
        FlightDetector / tiling. Derived by rasterizing the field rectangle's image-space
        polygon (inverse homography of the field corners). Cached per (hw, pose). Returns
        None until a valid pose is known."""
        if pose is None or not getattr(pose, "valid", False):
            return self._field_mask  # may be a previously-cached mask, else None

        pose_id = id(pose)
        if (self._field_mask is not None and self._mask_hw == hw
                and self._mask_pose_id == pose_id):
            return self._field_mask

        poly = self.field_polygon_px(pose)
        if poly is None:
            return None

        import cv2
        h, w = int(hw[0]), int(hw[1])
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)

        self._field_poly_px = poly
        self._field_mask = mask
        self._mask_hw = hw
        self._mask_pose_id = pose_id
        return mask

    def tiling_polygons(self, pose: CameraPose, nx: int = 4, ny: int = 2) -> list[np.ndarray]:
        """Partition the field into nx*ny ground-plane sub-rectangles and project each into
        image space. The ball detector runs on these tiles (covering only the carpet, not
        the stands) to keep the 5Hz pass cheap. Each polygon is a 4x2 px array (CW/CCW as
        the homography yields). Returns [] if pose invalid."""
        if pose is None or not getattr(pose, "valid", False):
            return []
        H = np.asarray(pose.H, dtype=np.float64)
        try:
            Hinv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return []
        nx, ny = max(1, int(nx)), max(1, int(ny))
        dx = self.length_m / nx
        dy = self.width_m / ny
        tiles: list[np.ndarray] = []
        for ix in range(nx):
            for iy in range(ny):
                x0, x1 = ix * dx, (ix + 1) * dx
                y0, y1 = iy * dy, (iy + 1) * dy
                rect_m = np.array(
                    [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64
                )
                ones = np.ones((4, 1), dtype=np.float64)
                hom = np.hstack([rect_m, ones]) @ Hinv.T
                tiles.append(hom[:, :2] / hom[:, 2:3])
        return tiles

    # ----- scoring geometry (image-independent: all tests in field meters) -- #
    def hub_for(self, alliance: str) -> dict:
        return self.hubs.get(alliance, {})

    def in_hub(self, field_xy: tuple[float, float], alliance: str, tol_m: float = 0.0) -> bool:
        hub = self.hubs.get(alliance)
        if not hub:
            return False
        cx, cy = hub["center_m"]
        r = hub["radius_m"] + tol_m
        return (field_xy[0] - cx) ** 2 + (field_xy[1] - cy) ** 2 <= r * r

    # ----- field GATE (back-projection; the keystone) ----------------------- #
    # WHY back-projection: forward-projecting the full field rectangle via inv(H)
    # OVERSHOOTS the frame, because the recovered tags cluster in a sub-region and the
    # homography extrapolates wildly outside that cluster (the tech lead's verified note).
    # So instead of asking "is the pixel inside the projected field polygon?", we map the
    # image point THROUGH H (image -> field meters, the direction H is defined for) and
    # test the resulting field XY against the known field bounds. This only ever evaluates
    # H near where it was actually fit (on real on-field points), so it is robust.
    def _gate_homography(self, pose: CameraPose) -> np.ndarray | None:
        """Cached np.float64 H (image->field) for the gate. None if pose invalid."""
        if pose is None or not getattr(pose, "valid", False):
            return None
        pid = id(pose)
        if self._gate_pose_id == pid and self._gate_H is not None:
            return self._gate_H
        H = np.asarray(pose.H, dtype=np.float64)
        self._gate_pose_id = pid
        self._gate_H = H
        return H

    def _px_to_m_scale(self, px: tuple[float, float], H: np.ndarray) -> float:
        """Approximate local pixels->meters scale at image point ``px`` (meters of field
        motion per pixel of image motion), from the homography Jacobian via a 1px probe.
        Used to convert a ``tol_px`` tolerance into a field-meter margin for the bounds
        test. Returns a small positive number; clamps a degenerate/zero scale."""
        f0 = self.image_to_field((px[0], px[1]), _PoseLike(H))
        fx = self.image_to_field((px[0] + 1.0, px[1]), _PoseLike(H))
        fy = self.image_to_field((px[0], px[1] + 1.0), _PoseLike(H))
        sx = math.hypot(fx[0] - f0[0], fx[1] - f0[1])
        sy = math.hypot(fy[0] - f0[0], fy[1] - f0[1])
        s = max(sx, sy)
        return s if s > 1e-9 else 0.0

    def contains_image_pt(self, px_xy: tuple[float, float], pose: CameraPose,
                          tol_px: float = 24.0) -> bool:
        """Does image point ``px_xy`` lie on the FIELD (the carpet rectangle)?

        Implemented by BACK-PROJECTION: map the point through ``pose.H`` (image->field
        meters) and test the field XY against [0, length_m] x [0, width_m], widened by a
        meter tolerance derived from ``tol_px`` (the local px->m scale at this point). This
        sidesteps the tag-cluster extrapolation problem of forward polygon projection.

        Returns False when the pose is missing/invalid (no field frame to test against) or
        the back-projected point is behind the camera (degenerate w<=0)."""
        H = self._gate_homography(pose)
        if H is None:
            return False
        x, y, w = (H @ np.array([px_xy[0], px_xy[1], 1.0]))
        if abs(w) < 1e-12:
            return False
        fx, fy = x / w, y / w
        # behind-camera / wildly-extrapolated points come back with the wrong sign on w;
        # the bounds test below also rejects them, but guard explicitly for clarity.
        tol_m = max(0.0, float(tol_px)) * self._px_to_m_scale(px_xy, H)
        return bool(-tol_m <= fx <= self.length_m + tol_m
                    and -tol_m <= fy <= self.width_m + tol_m)

    def gate_detections(self, dets, pose: CameraPose, y_main_px: float,
                        tol_px: float = 24.0, alliances=None):
        """Keep only detections whose FLOOR-CONTACT point (cx, y2) is in the MAIN pane
        (y2 < ``y_main_px``) AND on the field (``contains_image_pt``). This is the helper
        run_scout/MatchRunner uses to strip crowd / opposite-pane / person boxes BEFORE
        the 6-robot prior ever sees them (a person on the carpet edge back-projects far
        outside the field rectangle and is rejected).

        Parameters
        ----------
        dets : sequence of Detection | objects with ``.xyxy`` (x1,y1,x2,y2 in px).
        pose : CameraPose. If missing/invalid the gate CANNOT localize, so it is a NO-OP
               (returns the inputs unchanged) — we never silently drop everything when we
               simply lack a pose this frame.
        y_main_px : the main-pane lower y boundary in PIXELS (e.g. 0.52 * frame_height).
        tol_px : field-edge tolerance forwarded to ``contains_image_pt``.
        alliances : optional parallel sequence; when given, the SAME keep-mask is applied to
               it and a (kept_dets, kept_alliances) tuple is returned (so MatchRunner can
               gate its parallel alliance list in lock-step).

        Returns the kept detections (or (kept_dets, kept_alliances) when ``alliances`` is
        given). Always returns a ``keep`` boolean list via ``gate_mask`` for callers that
        want the count of rejects.
        """
        dets = list(dets)
        keep = self.gate_mask(dets, pose, y_main_px, tol_px=tol_px)
        kept = [d for d, k in zip(dets, keep) if k]
        if alliances is not None:
            allys = list(alliances)
            if len(allys) < len(dets):
                allys += [None] * (len(dets) - len(allys))
            kept_allys = [a for a, k in zip(allys, keep) if k]
            return kept, kept_allys
        return kept

    def gate_mask(self, dets, pose: CameraPose, y_main_px: float,
                  tol_px: float = 24.0) -> list[bool]:
        """Per-detection keep/reject booleans (parallel to ``dets``) for the field gate.

        A detection is KEPT iff its floor-contact point (cx, y2) is above the main-pane
        boundary AND back-projects onto the field. With no valid pose the gate cannot
        judge, so EVERY detection is kept (no-op) — honest degradation, never a blanket
        reject for a missing pose."""
        dets = list(dets)
        if pose is None or not getattr(pose, "valid", False):
            return [True] * len(dets)
        keep: list[bool] = []
        for d in dets:
            x1, _y1, x2, y2 = d.xyxy
            cx = (x1 + x2) / 2.0
            in_pane = y2 < float(y_main_px)
            keep.append(bool(in_pane and self.contains_image_pt((cx, y2), pose, tol_px=tol_px)))
        return keep
