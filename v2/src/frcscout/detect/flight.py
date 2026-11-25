"""Ball-flight detector — the ONLY ball signal that runs at 60Hz, on the CPU.

Airborne fuel balls are few, high-contrast, and fast against the static carpet. That
makes them cheap to find with classical motion methods (no neural net needed at 60Hz):
    - background subtraction (MOG2) restricted to the field region -> moving blobs
    - AND that motion mask with the FUEL-COLOR mask (vivid yellow) so only moving
      *fuel-colored* blobs survive (kills moving people / banners / pan streaks)
    - keep blobs that are ~ball-sized AND round (circularity)
    - optionally restrict to the ARC region (above the ground-field band in the main
      pane) so a fast ground ball isn't double-counted as a shot
    - hand off to the Kalman FlightTracker for trajectory + 3D arc fitting

The camera on the 2026 broadcast is effectively STABLE (AprilTag pose ~0.7px), so MOG2
is valid: the residual foreground is genuine object motion, not global pan. The dominant
false positives are therefore moving *people* (refs, drive team) and waving banners — all
killed by intersecting with the fuel-color mask, since fuel is a uniquely vivid yellow.

Speed gating (in field m/s) is done AFTER homography by the tracker, not in px here, so it
stays camera-invariant. This module only proposes candidate moving fuel blobs.
"""
from __future__ import annotations

import numpy as np

from frcscout.io.types import Detection

_KERNEL = np.ones((3, 3), np.uint8)

# ----- Fuel-color (vivid yellow) HSV gate ---------------------------------- #
# OpenCV HSV ranges: H in [0,179], S/V in [0,255]. The 2026 fuel ball is a saturated
# yellow-orange foam ball. The hue band covers yellow→yellow-orange; a saturation/value
# floor rejects the desaturated tan carpet, beige skin, and the dim yellow of distant
# banners. Tunable via cfg (fuel_h_lo/hi, fuel_sat_min, fuel_val_min).
#
# Calibrated on REAL qual59 AIRBORNE/moving-ball pixels (measured H~15-21, S~166-236,
# V~114-148): the low hue edge is dropped to 16 (fast balls read as low as H15-17) and the
# value floor to 95 (motion blur + shadow on a thrown ball can dim V below 110) so airborne
# fuel is reliably caught; sat_min stays high (real fuel S>=166; the pale crowd S~77) so
# moving PEOPLE/banners are still rejected. The arc_region + circularity + the Kalman
# tracker's sustained-speed confirmation provide the rest of the precision.
_FUEL_H_LO = 16
_FUEL_H_HI = 40
_FUEL_SAT_MIN = 110
_FUEL_VAL_MIN = 95


class FlightDetector:
    def __init__(self, cfg: dict, field):
        import cv2
        self.cfg = cfg or {}
        self.field = field
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=120,
            varThreshold=24,
            detectShadows=False,
        )
        # Ball-size gates in pixels (area of enclosing circle).
        # min_area: single pixel clusters are noise; max_area: prevents large ROI
        # hits (robots, shadows) from polluting the flight list.
        self.min_area = self.cfg.get("min_area", 4)
        self.max_area = self.cfg.get("max_area", 600)
        # Pose-independent gates for MOVING (broadcast) cameras, where field_mask is
        # unavailable and MOG2 lights up on global motion (pans, crowd, overlay). All
        # default to OFF so the static-camera deploy + existing tests are unchanged;
        # deploy.yaml enables them.
        self.min_circularity = float(self.cfg.get("min_circularity", 0.0))  # 0 = off
        self.max_candidates = int(self.cfg.get("max_candidates", 0))         # 0 = unlimited
        # Minimum enclosing-circle radius (px) — floors out sub-ball motion speckle that can
        # still pass the area+circularity gates (e.g. a 3px glint on a moving edge). 0 = off
        # (keeps the static-cam deploy + existing synthetic tests unchanged). A real airborne
        # fuel ball on this 1080p broadcast is r~5-14 px, so a floor of ~2-3 px is safe.
        self.min_radius_px = float(self.cfg.get("min_radius_px_blob", 0.0))  # 0 = off
        self.roi_frac = self.cfg.get("roi_frac", None)  # [x0,y0,x1,y1] frac; only if no field_mask

        # ----- Fuel-color gate (the key broadcast-FP killer) --------------- #
        # When enabled, the MOG2 foreground is AND-ed with a vivid-yellow HSV mask so
        # only moving FUEL-colored blobs survive. Default ON for the broadcast deploy;
        # set fuel_color_gate: false to disable (e.g. a synthetic/greyscale test feed).
        self.fuel_color_gate = bool(self.cfg.get("fuel_color_gate", True))
        self.fuel_h_lo = int(self.cfg.get("fuel_h_lo", _FUEL_H_LO))
        self.fuel_h_hi = int(self.cfg.get("fuel_h_hi", _FUEL_H_HI))
        self.fuel_sat_min = int(self.cfg.get("fuel_sat_min", _FUEL_SAT_MIN))
        self.fuel_val_min = int(self.cfg.get("fuel_val_min", _FUEL_VAL_MIN))
        # Dilate the color mask a hair so a fast ball whose motion-mask edge sits just
        # outside its color core still intersects (motion blur shifts the two slightly).
        self.fuel_dilate = int(self.cfg.get("fuel_dilate", 2))

        # ----- Arc-region restriction (optional) --------------------------- #
        # A shot ARCS above the carpet. arc_region = [x0,y0,x1,y1] frac of the FRAME
        # restricts candidates to the band above the ground-field within the main pane,
        # so a fast ground ball (rolling fuel) is not double-counted as a flight/shot.
        # Default None = no restriction (a thrown ball can be anywhere in the main pane).
        self.arc_region = self.cfg.get("arc_region", None)

    # ----- fuel-color mask -------------------------------------------------- #
    def _fuel_mask(self, frame) -> "np.ndarray | None":
        """uint8 0/255 mask of vivid-yellow (fuel) pixels in a BGR frame, or None if the
        frame is not 3-channel BGR (e.g. a greyscale synthetic test frame)."""
        import cv2
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            return None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h = hsv[..., 0]
        s = hsv[..., 1]
        v = hsv[..., 2]
        m = (
            (h >= self.fuel_h_lo) & (h <= self.fuel_h_hi)
            & (s >= self.fuel_sat_min) & (v >= self.fuel_val_min)
        )
        mask = (m.astype(np.uint8)) * 255
        if self.fuel_dilate > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * self.fuel_dilate + 1, 2 * self.fuel_dilate + 1),
            )
            mask = cv2.dilate(mask, k)
        return mask

    def detect(self, frame, frame_idx: int) -> list[Detection]:
        import cv2

        # Background subtraction — yields a binary foreground mask.
        fg = self.bg.apply(frame)

        # Restrict to the field region.  field_mask() may return None if the
        # homography has not been solved yet (cold start) — in that case we
        # proceed with the raw foreground mask rather than dropping the frame.
        try:
            mask = self.field.field_mask(frame.shape[:2])
        except Exception:
            mask = None

        if mask is not None:
            # Ensure mask is 8-bit single-channel before bitwise_and.
            if mask.dtype != np.uint8:
                mask = (mask > 0).astype(np.uint8) * 255
            fg = cv2.bitwise_and(fg, mask)
        elif self.roi_frac is not None:
            # No homography yet: fall back to a fixed image ROI (exclude the overlay/
            # scoreboard bar and crowd margins) so we don't track global-motion noise on
            # a panning broadcast camera.
            h_px, w_px = fg.shape[:2]
            x0, y0, x1, y1 = self.roi_frac
            roi = np.zeros(fg.shape, dtype=np.uint8)
            roi[int(y0 * h_px):int(y1 * h_px), int(x0 * w_px):int(x1 * w_px)] = 255
            fg = cv2.bitwise_and(fg, roi)

        # Arc-region restriction: only keep motion in the band above the ground-field
        # (where a shot arcs), so a fast ground ball isn't double-counted as a shot.
        if self.arc_region is not None:
            h_px, w_px = fg.shape[:2]
            x0, y0, x1, y1 = self.arc_region
            arc = np.zeros(fg.shape, dtype=np.uint8)
            arc[int(y0 * h_px):int(y1 * h_px), int(x0 * w_px):int(x1 * w_px)] = 255
            fg = cv2.bitwise_and(fg, arc)

        # FUEL-COLOR gate: AND the motion mask with the vivid-yellow mask so only moving
        # FUEL-colored blobs survive. This is what kills moving people / banners / pan
        # streaks on the broadcast — a moving ref is not fuel-yellow, so it drops out.
        if self.fuel_color_gate:
            fmask = self._fuel_mask(frame)
            if fmask is not None:
                fg = cv2.bitwise_and(fg, fmask)

        # Morphological opening to remove single-pixel noise speckle.
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, _KERNEL)

        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Score each ball-sized blob; optionally gate by circularity and cap the count so
        # a noisy (moving-camera) frame can't emit hundreds of bogus "flight" candidates.
        import math
        scored: list[tuple[float, Detection]] = []
        for c in cnts:
            a = cv2.contourArea(c)
            if not (self.min_area <= a <= self.max_area):
                continue
            circ = 1.0
            if self.min_circularity > 0.0:
                peri = cv2.arcLength(c, True)
                if peri <= 0.0:
                    continue
                circ = 4.0 * math.pi * a / (peri * peri)   # 1.0 == perfect circle
                if circ < self.min_circularity:
                    continue
            (cx, cy), r = cv2.minEnclosingCircle(c)
            if self.min_radius_px > 0.0 and r < self.min_radius_px:
                continue
            scored.append((circ * a, Detection(
                cls="ball",
                xyxy=(cx - r, cy - r, cx + r, cy + r),
                conf=1.0,
                frame_idx=frame_idx,
            )))

        # Keep the most ball-like candidates (circularity*area) when capped.
        if self.max_candidates and len(scored) > self.max_candidates:
            scored.sort(key=lambda s: -s[0])
            scored = scored[:self.max_candidates]
        return [d for _, d in scored]
