"""Classical color-prior fuel-ball detector — a CPU companion to the neural net.

The deploy net sees only 2-45 balls per frame on this stacked-broadcast footage (it was
trained on tighter crops and under-fires on dense piles and tiny carpet balls). Fuel is a
*very* specific colour — vivid yellow — and that colour prior is a strong, cheap signal the
net throws away. This module is the user's "amplify until the balls are white with clear
outlines, then detect on THAT" idea, done with the precision gates that keep it from
flooding on the crowd.

Pipeline (all classical, no GPU, ~a few ms/frame on CPU):

    1. CLAHE-amplify luma (reuse refine.preprocess.apply_clahe) so dim far-pane balls and
       shadowed pile balls lift toward saturation.
    2. Build a FUEL MASK in two colour spaces AND-ed together — this is the crowd-rejection
       core. Calibrated on qual59 f500/f1000:
          * HSV: hue in the yellow band (~15-45 of 0-179), V high, S above a floor.
          * Lab b-channel HIGH (yellow-blue axis). This is the killer discriminator:
            real fuel has Lab b p90 ~ 200 (median ~140); the blue/teal crowd has b ~ 102
            and the gray carpet b ~ 76 — both fail a b>=150 gate cleanly.
    3. connectedComponentsWithStats on the cleaned mask -> candidate blobs.
    4. CIRCULARITY gate (the 4*pi*area/peri^2 pattern from detect/flight.py): fuel balls are
       round; the yellow jacket / banners / pile-edge smears are not.
    5. SIZE gate. With a CameraPose, project the blob's pixel radius to FIELD METERS and
       reject anything projecting larger than a fat single ball (~0.35 m; a real fuel ball
       is ~0.13 m radius). Without a pose, fall back to a pixel-radius gate. Dense piles are
       handled by an UPPER area cap that emits per-ball-sized blobs, not the whole pile as
       one giant box.
    6. ROI gate: a per-pane fractional rectangle (default [0.04,0.13,0.96,0.93]) that strips
       the overlay/scoreboard bar and the worst of the crowd margins WITHOUT clipping the
       field. We deliberately do NOT restrict to the field polygon: airborne balls live
       *above* the carpet and would be discarded by a field-polygon mask.

Output: ``list[Detection(cls="ball", ...)]`` — the same frozen dataclass the net emits, so
the results drop straight into ``refine.detect_fuse.fuse([net_dets, blob_dets])``.

cv2 is imported lazily (mirrors detect/flight.py and refine/preprocess.py) so the module
PARSES in a bare env; the pure-logic helpers (config, gates, ROI geometry, radius
projection) are unit-testable with numpy only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from frcscout.io.types import CameraPose, Detection

# Default per-pane ROI as fractions [x0, y0, x1, y1] of frame size. Trims the overlay bar
# (top) and the lowest crowd strip, keeping both broadcast panes. NOT the field polygon.
_DEFAULT_ROI_FRAC = (0.04, 0.13, 0.96, 0.93)

# A regulation fuel ball is ~0.13 m radius. We reject blobs whose field-projected radius
# exceeds this (a smear / robot / jacket spanning several balls).
_BALL_RADIUS_M = 0.13


def _cv2():
    try:
        import cv2
        return cv2
    except Exception as exc:  # pragma: no cover - cv2 present in CI image
        raise RuntimeError(
            "frcscout.detect.ball_blob requires OpenCV (cv2) for image operations; "
            "the pure-logic helpers (BallBlobConfig, circularity, radius projection, "
            "roi_rect_px) work without it."
        ) from exc


@dataclass(frozen=True)
class BallBlobConfig:
    """Calibrated thresholds for the fuel-ball colour prior + geometric gates.

    Colour (calibrated on data/footage/qual59.mp4 frames 500 & 1000):
        hsv_h_lo / hsv_h_hi : OpenCV hue band for vivid yellow (0-179). Fuel median ~30.
        hsv_s_min           : saturation floor (kills washed-out gray/white highlights).
        hsv_v_min           : value floor (kills dark yellow-ish shadow).
        lab_b_min           : Lab b-channel floor — THE crowd/carpet rejector. Fuel b is
                              high (>=150 at the 90th pct); crowd ~102, carpet ~76.
    Geometry:
        clahe_clip/grid     : amplification before masking.
        min_circularity     : 4*pi*area/peri^2 floor (1.0 = perfect circle).
        min_area_px         : drop single-pixel speckle.
        max_area_px         : pixel upper cap when NO pose (keeps a single ball, not a pile).
        max_radius_m        : field-projected radius cap when a pose IS given.
        min_radius_px       : minimum enclosing-circle radius (px) — distant balls floor.
        morph_kernel        : opening/closing kernel edge (px) to clean the mask.
        max_candidates      : hard cap on emitted detections (0 = unlimited); keeps the
                              roundest/biggest when a pathological frame over-fires.
    """
    clahe_clip: float = 2.5
    clahe_grid: int = 8
    hsv_h_lo: int = 15
    hsv_h_hi: int = 45
    hsv_s_min: int = 60
    hsv_v_min: int = 90
    lab_b_min: int = 150
    min_circularity: float = 0.55
    min_area_px: float = 9.0
    max_area_px: float = 4000.0
    max_radius_m: float = 0.35
    min_radius_px: float = 1.5
    morph_kernel: int = 3
    max_candidates: int = 400
    # ---- Dense-pile AREA estimate (Problem 1) ---------------------------------------- #
    # A merged, occluded pile is ONE big yellow blob that fails the per-ball circularity/
    # size gate, so it is never boxed and the count collapses (~11 vs ~100 real). Instead
    # of pretending to box every occluded ball, we estimate the pile's ball COUNT from its
    # MASK AREA: count ~= component_area / single_ball_area. This is an honest area estimate,
    # never a fake precise integer.
    #   pile_area_mult     : a yellow component is treated as a PILE (area-estimated, not
    #                        boxed) once its area exceeds this many single-ball areas. Below
    #                        it, the component is a single/near-single ball handled by box.
    #   pile_fill_frac     : pixels in a pile are not 100% ball (gaps, shadow between balls,
    #                        carpet bleed). The visible yellow area is scaled by this packing
    #                        factor before dividing by single-ball area, so we don't over-count
    #                        a loosely-packed spread. 1.0 == assume the whole component is ball.
    #   pile_close_kernel  : morphological CLOSE (px) applied to the field-gated mask before
    #                        component-finding, so balls separated by a thin dark seam merge
    #                        into one pile component (we WANT the pile merged for area est).
    #   single_ball_area_px: explicit override single-ball pixel area. 0 => derive it (Hough
    #                        on the pile, else isolated-median, else the calibrated default).
    #   single_ball_radius_px_default: calibrated fallback ball-FACE radius (px) for this
    #                        ~fixed 1080p broadcast. Measured on qual59 f500/f800 with Hough:
    #                        the packed ball faces are r~9-10 px (area ~285 px). Used only when
    #                        neither a Hough measurement nor a robust isolated sample is
    #                        available. Synthesizes single_ball_area_px = pi*r^2.
    #   pile_min_area_px   : a field-gated component smaller than this is ignored entirely
    #                        (mask speckle); it is neither boxed here nor pile-counted.
    #   pile_hough_min_circles: minimum Hough ball-faces needed to trust their median radius as
    #                        the scale (the radius is stable even when the COUNT is partial).
    #   isolated_min_sample: minimum isolated detections needed before their median box area is
    #                        trusted as the scale. Below this, isolated balls are too few / too
    #                        biased-small (only distant balls appear isolated) to be reliable.
    pile_area_mult: float = 3.0
    pile_fill_frac: float = 0.80
    pile_close_kernel: int = 5
    single_ball_area_px: float = 0.0
    single_ball_radius_px_default: float = 9.5
    pile_min_area_px: float = 60.0
    pile_hough_min_circles: int = 6
    isolated_min_sample: int = 8

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> "BallBlobConfig":
        """Build from a plain dict (e.g. a deploy.yaml slice), ignoring unknown keys."""
        cfg = cfg or {}
        d = cls()
        return cls(
            clahe_clip=float(cfg.get("clahe_clip", d.clahe_clip)),
            clahe_grid=int(cfg.get("clahe_grid", d.clahe_grid)),
            hsv_h_lo=int(cfg.get("hsv_h_lo", d.hsv_h_lo)),
            hsv_h_hi=int(cfg.get("hsv_h_hi", d.hsv_h_hi)),
            hsv_s_min=int(cfg.get("hsv_s_min", d.hsv_s_min)),
            hsv_v_min=int(cfg.get("hsv_v_min", d.hsv_v_min)),
            lab_b_min=int(cfg.get("lab_b_min", d.lab_b_min)),
            min_circularity=float(cfg.get("min_circularity", d.min_circularity)),
            min_area_px=float(cfg.get("min_area_px", d.min_area_px)),
            max_area_px=float(cfg.get("max_area_px", d.max_area_px)),
            max_radius_m=float(cfg.get("max_radius_m", d.max_radius_m)),
            min_radius_px=float(cfg.get("min_radius_px", d.min_radius_px)),
            morph_kernel=int(cfg.get("morph_kernel", d.morph_kernel)),
            max_candidates=int(cfg.get("max_candidates", d.max_candidates)),
            pile_area_mult=float(cfg.get("pile_area_mult", d.pile_area_mult)),
            pile_fill_frac=float(cfg.get("pile_fill_frac", d.pile_fill_frac)),
            pile_close_kernel=int(cfg.get("pile_close_kernel", d.pile_close_kernel)),
            single_ball_area_px=float(cfg.get("single_ball_area_px", d.single_ball_area_px)),
            single_ball_radius_px_default=float(
                cfg.get("single_ball_radius_px_default", d.single_ball_radius_px_default)),
            pile_min_area_px=float(cfg.get("pile_min_area_px", d.pile_min_area_px)),
            pile_hough_min_circles=int(
                cfg.get("pile_hough_min_circles", d.pile_hough_min_circles)),
            isolated_min_sample=int(cfg.get("isolated_min_sample", d.isolated_min_sample)),
        )


@dataclass(frozen=True)
class PileComponent:
    """One merged dense-pile component (a yellow mask blob too big/occluded to box per-ball).

    Coordinates are FULL-FRAME image pixels.
        xyxy        : (x1,y1,x2,y2) bounding box of the component.
        centroid_px : (cx,cy) area centroid.
        area_px     : raw yellow-mask pixel area of the component.
        est_count   : honest ball-count estimate for THIS component
                      (= area_px * pile_fill_frac / single_ball_area_px, >=1).
        contour     : Nx2 int array of the component's outer contour (for rendering the
                      pile outline). May be empty if cv2 produced no contour.
    """
    xyxy: tuple[float, float, float, float]
    centroid_px: tuple[float, float]
    area_px: float
    est_count: int
    contour: np.ndarray


@dataclass(frozen=True)
class PileEstimate:
    """Result of the dense-pile area estimate (Problem 1).

    HONEST contract: ``count_est`` is an AREA estimate, not a precise box count. Render it
    flagged as an estimate (the helper :meth:`label` gives the "~N (area est.)" string).

        count_est        : total estimated balls across all pile components (sum of their
                           est_count). 0 when no pile is present.
        components       : list[PileComponent] — the merged pile region(s).
        single_ball_area_px : the per-ball pixel area used for the division (derived from the
                           isolated detections when possible; else the configured fallback).
        single_ball_source : "isolated" | "config" | "default" — provenance of the area, so
                           the caller/log can see whether the estimate is frame-derived
                           (scale-correct) or a fallback.
        isolated_count   : number of individually-boxed isolated balls the caller passed in
                           (echoed so ``total_est`` is self-contained).

    ``total_est`` = isolated_count + count_est is the headline fuel number; render it as
    ``"~{total_est} (area est.)"`` whenever count_est > 0 (i.e. a pile contributed), and as a
    plain integer when count_est == 0 (every ball was boxed).
    """
    count_est: int
    components: list
    single_ball_area_px: float
    single_ball_source: str
    isolated_count: int = 0

    @property
    def total_est(self) -> int:
        """Isolated (boxed) balls + pile area-estimate."""
        return int(self.isolated_count) + int(self.count_est)

    @property
    def is_estimate(self) -> bool:
        """True when a pile contributed (=> render the total as an estimate, not exact)."""
        return self.count_est > 0

    def label(self, total: Optional[int] = None) -> str:
        """Human/overlay label for the fuel total. Honest: flagged "(area est.)" when a pile
        contributed, plain integer otherwise. Pass ``total`` to label a caller-computed total
        (e.g. fused with the net); defaults to :attr:`total_est`."""
        t = self.total_est if total is None else int(total)
        return f"~{t} (area est.)" if self.is_estimate else f"{t}"


# ---------------------------------------------------------------------------
# Pure-logic helpers (numpy only; no cv2) — unit-testable in a bare env
# ---------------------------------------------------------------------------

def circularity(area: float, perimeter: float) -> float:
    """4*pi*area / perimeter^2 (1.0 == perfect circle). Degenerate -> 0.0.

    Mirrors the circularity math in detect/flight.py so the two classical detectors agree
    on what "round" means.
    """
    if perimeter <= 0.0 or area <= 0.0:
        return 0.0
    return float(4.0 * math.pi * area / (perimeter * perimeter))


def roi_rect_px(hw: tuple[int, int],
                roi_frac: Optional[tuple[float, float, float, float]]) -> tuple[int, int, int, int]:
    """Resolve a fractional ROI [x0,y0,x1,y1] to a clamped pixel rectangle (x0,y0,x1,y1).

    None -> the full frame. Fractions are clamped to [0,1] and ordered; the result is a
    valid, non-empty, in-bounds rectangle.
    """
    h = max(1, int(hw[0]))
    w = max(1, int(hw[1]))
    if roi_frac is None:
        return 0, 0, w, h
    x0f, y0f, x1f, y1f = roi_frac
    x0f, x1f = sorted((min(max(float(x0f), 0.0), 1.0), min(max(float(x1f), 0.0), 1.0)))
    y0f, y1f = sorted((min(max(float(y0f), 0.0), 1.0), min(max(float(y1f), 0.0), 1.0)))
    x0 = int(math.floor(x0f * w))
    y0 = int(math.floor(y0f * h))
    x1 = max(x0 + 1, min(int(math.ceil(x1f * w)), w))
    y1 = max(y0 + 1, min(int(math.ceil(y1f * h)), h))
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    return x0, y0, x1, y1


def point_in_roi(cx: float, cy: float, rect: tuple[int, int, int, int]) -> bool:
    """True if (cx, cy) lies inside the (x0,y0,x1,y1) pixel rectangle."""
    x0, y0, x1, y1 = rect
    return x0 <= cx <= x1 and y0 <= cy <= y1


def projected_radius_m(cx: float, cy: float, r_px: float, pose: CameraPose) -> Optional[float]:
    """Approximate a pixel-radius blob's FIELD radius in meters via the ground homography.

    We map the blob centre and a point one radius to its right (cx+r, cy) to field meters
    and take the planar distance. This is exact for a ball resting on the carpet and a
    reasonable over-estimate for an airborne ball (it sits closer to the camera than its
    ground intersection, so its projected radius reads LARGER — biasing toward rejection,
    never toward a false accept of a too-big blob). Returns None if the pose is unusable.
    """
    if pose is None or not getattr(pose, "valid", False):
        return None
    H = np.asarray(pose.H, dtype=np.float64)
    if H.shape != (3, 3):
        return None

    def to_field(px: float, py: float) -> Optional[tuple[float, float]]:
        v = H @ np.array([px, py, 1.0], dtype=np.float64)
        if abs(v[2]) < 1e-12:
            return None
        return float(v[0] / v[2]), float(v[1] / v[2])

    c = to_field(cx, cy)
    e = to_field(cx + max(r_px, 1e-3), cy)
    if c is None or e is None:
        return None
    return math.hypot(e[0] - c[0], e[1] - c[1])


def median_box_area(dets: "list[Detection]") -> Optional[float]:
    """Median bounding-box pixel area over a list of Detections, or None if empty.

    This is the scale-aware single-ball area source for the pile estimate: the isolated
    fuel balls boxed THIS frame sit at the same camera distance as the pile, so their
    median box area is the right per-ball area to divide the pile mask by — no homography
    needed and automatically correct as the camera zooms.
    """
    areas = []
    for d in dets or []:
        x1, y1, x2, y2 = d.xyxy
        a = (float(x2) - float(x1)) * (float(y2) - float(y1))
        if a > 0.0:
            areas.append(a)
    if not areas:
        return None
    return float(np.median(np.asarray(areas, dtype=np.float64)))


def estimate_component_count(area_px: float, single_ball_area_px: float,
                             fill_frac: float = 1.0) -> int:
    """Estimate how many balls a merged pile component of ``area_px`` holds.

    count = round(area_px * fill_frac / single_ball_area_px), floored at 1 (a component big
    enough to be called a pile holds at least one ball). ``fill_frac`` (<=1) accounts for the
    seams/shadow between packed balls so a loosely-spread blob isn't over-counted. Degenerate
    single-ball area -> 0 (cannot estimate).
    """
    if single_ball_area_px <= 0.0 or area_px <= 0.0:
        return 0
    n = (area_px * max(0.0, float(fill_frac))) / single_ball_area_px
    return max(1, int(round(n)))


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class BallBlobDetector:
    """Stateful wrapper (holds the config + a reusable morph kernel). Construct once.

    ``detect(frame, frame_idx, pose=None, roi_frac=None)`` returns ``list[Detection]``.
    The module-level :func:`detect` is a convenience that builds a default detector.
    """

    def __init__(self, cfg: Optional[dict | BallBlobConfig] = None):
        if isinstance(cfg, BallBlobConfig):
            self.cfg = cfg
        else:
            self.cfg = BallBlobConfig.from_dict(cfg)

    # -- the colour prior (calibrated) -------------------------------------- #
    def fuel_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """uint8 0/255 mask of vivid-yellow fuel pixels. CLAHE-amplified, HSV AND Lab gated."""
        from frcscout.refine.preprocess import apply_clahe
        cv2 = _cv2()
        c = self.cfg
        amp = apply_clahe(frame_bgr, clip=c.clahe_clip, grid=c.clahe_grid)

        hsv = cv2.cvtColor(amp, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(amp, cv2.COLOR_BGR2LAB)

        hsv_lo = np.array([c.hsv_h_lo, c.hsv_s_min, c.hsv_v_min], dtype=np.uint8)
        hsv_hi = np.array([c.hsv_h_hi, 255, 255], dtype=np.uint8)
        m_hsv = cv2.inRange(hsv, hsv_lo, hsv_hi)
        # Lab b high == strongly yellow. This is the crowd/carpet rejector.
        m_lab = cv2.inRange(lab[:, :, 2], int(c.lab_b_min), 255)

        mask = cv2.bitwise_and(m_hsv, m_lab)

        k = max(1, int(c.morph_kernel))
        kern = np.ones((k, k), np.uint8)
        # Open to kill speckle, close to merge a single ball's specular highlight back in.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
        return mask

    def detect(
        self,
        frame: np.ndarray,
        frame_idx: int,
        pose: Optional[CameraPose] = None,
        roi_frac: Optional[tuple[float, float, float, float]] = _DEFAULT_ROI_FRAC,
    ) -> list[Detection]:
        """Detect fuel balls. Returns ``list[Detection(cls="ball", ...)]``.

        Args:
            frame:     H×W×3 uint8 BGR.
            frame_idx: stamped onto each Detection.
            pose:      optional CameraPose; when valid+supplied, blobs are size-gated by
                       field-projected radius (rejects > max_radius_m). Else a px gate.
            roi_frac:  [x0,y0,x1,y1] frac rectangle to keep (default trims overlay/crowd).
                       Pass None to search the full frame.
        """
        cv2 = _cv2()
        c = self.cfg
        h, w = frame.shape[:2]
        rect = roi_rect_px((h, w), roi_frac)

        mask = self.fuel_mask(frame)
        # Zero everything outside the ROI rectangle (cheaper than masking the whole frame).
        roi_mask = np.zeros_like(mask)
        x0, y0, x1, y1 = rect
        roi_mask[y0:y1, x0:x1] = 255
        mask = cv2.bitwise_and(mask, roi_mask)

        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)

        use_pose = pose is not None and getattr(pose, "valid", False)
        scored: list[tuple[float, Detection]] = []
        for i in range(1, n):  # 0 is background
            area = float(stats[i, cv2.CC_STAT_AREA])
            if area < c.min_area_px:
                continue
            bx = int(stats[i, cv2.CC_STAT_LEFT])
            by = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            cx, cy = float(cents[i, 0]), float(cents[i, 1])

            # Circularity from this component's contour (peri needs the actual boundary).
            comp = (labels[by:by + bh, bx:bx + bw] == i).astype(np.uint8)
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            peri = cv2.arcLength(cnt, True)
            circ = circularity(cv2.contourArea(cnt), peri)
            if circ < c.min_circularity:
                continue

            (ecx, ecy), r_px = cv2.minEnclosingCircle(cnt)
            r_px = float(r_px)
            ecx += bx
            ecy += by
            if r_px < c.min_radius_px:
                continue

            # Size gate: field meters if pose, else pixel area.
            if use_pose:
                r_m = projected_radius_m(ecx, ecy, r_px, pose)
                if r_m is not None and r_m > c.max_radius_m:
                    continue
            else:
                if area > c.max_area_px:
                    continue

            x1b, y1b = ecx - r_px, ecy - r_px
            x2b, y2b = ecx + r_px, ecy + r_px
            # Confidence: circularity is the most informative single cue here.
            scored.append((circ * math.sqrt(area), Detection(
                cls="ball",
                xyxy=(x1b, y1b, x2b, y2b),
                conf=float(min(1.0, max(0.05, circ))),
                frame_idx=frame_idx,
            )))

        if c.max_candidates and len(scored) > c.max_candidates:
            scored.sort(key=lambda s: -s[0])
            scored = scored[:c.max_candidates]
        return [d for _, d in scored]

    # -- dense-pile AREA estimate (Problem 1) ------------------------------- #
    def _hough_ball_radius(self, region_bgr: np.ndarray) -> Optional[float]:
        """Median ball-FACE radius (px) found by Hough circles in a pile-region BGR crop.

        The packed ball FACES are still circular even though the colour MASK merges them into
        one blob, so Hough recovers the per-ball radius directly from the image — a
        scale-correct, frame-derived measurement (it self-adjusts when the camera zooms).
        Hough's COUNT is partial/conservative, so we use only the radius MEDIAN, not the
        count. Returns None if fewer than ``pile_hough_min_circles`` circles are found (median
        not yet stable). Calibrated radius window 5–18 px on this 1080p shot (faces ~9-10 px).
        """
        cv2 = _cv2()
        if region_bgr is None or region_bgr.size == 0:
            return None
        gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
            param1=120, param2=14, minRadius=5, maxRadius=18,
        )
        if circles is None:
            return None
        rs = np.asarray(circles[0][:, 2], dtype=np.float64)
        if len(rs) < max(1, int(self.cfg.pile_hough_min_circles)):
            return None
        return float(np.median(rs))

    def _resolve_single_ball_area(
        self,
        isolated_dets: "list[Detection]",
        biggest_region_bgr: Optional[np.ndarray],
    ) -> tuple[float, str]:
        """Pick the per-ball pixel area to divide the pile mask by, and its provenance.

        Priority (most → least scale-correct — chosen from real qual59 measurements):
          1. "hough":  median Hough ball-FACE radius measured INSIDE the largest pile region
                       this frame. This is the most reliable scale on a dense pile, because the
                       individual ball faces are visible & circular even when the colour mask
                       has merged them — and it self-adjusts to camera zoom. (Validated: r~9-10
                       px on f500/f800, stable frame-to-frame.)
          2. "config": the explicit cfg.single_ball_area_px override, if > 0.
          3. "isolated": median bounding-box area of the isolated balls boxed THIS frame — used
                       only when there is a ROBUST sample (>= isolated_min_sample). With few
                       isolated balls the sample is biased small (only DISTANT balls appear
                       isolated; near balls clump into the pile) and badly over-counts, so we
                       do NOT trust a tiny isolated sample.
          4. "default": area of a disc of the calibrated cfg.single_ball_radius_px_default.
        """
        r_hough = self._hough_ball_radius(biggest_region_bgr) if biggest_region_bgr is not None else None
        if r_hough is not None:
            return float(math.pi * r_hough * r_hough), "hough"
        if self.cfg.single_ball_area_px > 0.0:
            return float(self.cfg.single_ball_area_px), "config"
        med = median_box_area(isolated_dets)
        if med is not None and len(isolated_dets) >= int(self.cfg.isolated_min_sample):
            return med, "isolated"
        r = max(1.0, float(self.cfg.single_ball_radius_px_default))
        return float(math.pi * r * r), "default"

    def estimate_pile(
        self,
        frame: np.ndarray,
        frame_idx: int,
        isolated_dets: Optional["list[Detection]"] = None,
        pose: Optional[CameraPose] = None,
        field_poly_frac: Optional[list[tuple[float, float]]] = None,
        field_poly_px: Optional[np.ndarray] = None,
    ) -> PileEstimate:
        """HONEST area-based ball count for the dense, occluded pile (Problem 1).

        Builds the vivid-yellow fuel mask (same colour prior as :meth:`detect`), restricts it
        to the FIELD-REGION POLYGON (so the crowd/lower-pane can never contribute — keeps
        crowd FPs ~0), CLOSE-merges balls separated by thin dark seams into single pile
        components, and for every component LARGE enough to be a merged pile (area >
        ``pile_area_mult`` single-ball areas) estimates count = area * fill / single_ball_area.

        It does NOT box individual balls — call :meth:`detect` for the isolated balls and pass
        them here as ``isolated_dets``. The per-ball pixel area used for the division is
        derived from THIS frame (scale-aware): primarily a Hough measurement of the ball-face
        radius INSIDE the pile, with the isolated-detection median and a calibrated default as
        fallbacks (see :meth:`_resolve_single_ball_area`). Total fuel = ``estimate.total_est``
        = len(isolated_dets) + count_est.

        Args:
            frame:           H×W×3 uint8 BGR.
            frame_idx:       informational (not stamped onto anything; piles aren't tracked).
            isolated_dets:   the isolated-ball Detections from :meth:`detect` (used both to
                             derive the single-ball area and as the isolated_count). The pile
                             components are independent of these — a ball boxed individually
                             may also fall inside a small component, but a component only
                             becomes a PILE (counted) when it is several balls big, so the
                             double-count risk is limited to the pile/spread boundary.
            pose:            optional CameraPose (reserved for a future metric fallback).
            field_poly_frac: field polygon as [(xf,yf), ...] in image FRACTIONS — the same
                             representation run_scout/deploy use. Rasterized to a mask.
            field_poly_px:   alternatively, a prebuilt Nx1x2 / Nx2 int pixel polygon (e.g.
                             run_scout's FIELD_POLY). Takes precedence over field_poly_frac.
                             If BOTH are None, the whole frame is searched (not recommended on
                             broadcast footage — gate to the field to stay crowd-safe).

        Returns:
            PileEstimate. ``count_est == 0`` and empty ``components`` when no pile is present.
        """
        cv2 = _cv2()
        c = self.cfg
        isolated_dets = list(isolated_dets or [])
        h, w = frame.shape[:2]

        mask = self.fuel_mask(frame)

        # Gate to the field polygon (crowd-safety). Build a 0/255 polygon mask.
        poly_mask = self._field_poly_mask((h, w), field_poly_frac, field_poly_px)
        if poly_mask is not None:
            mask = cv2.bitwise_and(mask, poly_mask)

        # CLOSE so balls separated by a thin dark seam merge into one pile component (we WANT
        # the pile merged here — the opposite of per-ball boxing).
        kc = max(1, int(c.pile_close_kernel))
        if kc > 1:
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)

        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # Find the LARGEST yellow component (the pile, if any) — its image region gives the
        # best Hough scale measurement (most ball faces to median over).
        biggest_i, biggest_area = -1, 0.0
        for i in range(1, n):
            a = float(stats[i, cv2.CC_STAT_AREA])
            if a >= c.pile_min_area_px and a > biggest_area:
                biggest_area, biggest_i = a, i
        biggest_region = None
        if biggest_i >= 0:
            bx = int(stats[biggest_i, cv2.CC_STAT_LEFT])
            by = int(stats[biggest_i, cv2.CC_STAT_TOP])
            bw = int(stats[biggest_i, cv2.CC_STAT_WIDTH])
            bh = int(stats[biggest_i, cv2.CC_STAT_HEIGHT])
            biggest_region = frame[by:by + bh, bx:bx + bw]

        single_area, source = self._resolve_single_ball_area(isolated_dets, biggest_region)
        pile_threshold = max(c.pile_min_area_px, c.pile_area_mult * single_area)

        components: list[PileComponent] = []
        total = 0
        for i in range(1, n):
            area = float(stats[i, cv2.CC_STAT_AREA])
            if area < c.pile_min_area_px:
                continue
            if area < pile_threshold:
                # Not a pile — a single/near-single ball, already handled by detect()/boxing.
                continue
            bx = int(stats[i, cv2.CC_STAT_LEFT])
            by = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            cx, cy = float(cents[i, 0]), float(cents[i, 1])
            est = estimate_component_count(area, single_area, c.pile_fill_frac)
            comp = (labels[by:by + bh, bx:bx + bw] == i).astype(np.uint8)
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2)
                cnt = cnt + np.array([bx, by], dtype=cnt.dtype)  # to full-frame coords
            else:
                cnt = np.empty((0, 2), dtype=np.int32)
            components.append(PileComponent(
                xyxy=(float(bx), float(by), float(bx + bw), float(by + bh)),
                centroid_px=(cx, cy),
                area_px=area,
                est_count=est,
                contour=cnt,
            ))
            total += est

        return PileEstimate(
            count_est=int(total),
            components=components,
            single_ball_area_px=float(single_area),
            single_ball_source=source,
            isolated_count=len(isolated_dets),
        )

    def detect_with_pile(
        self,
        frame: np.ndarray,
        frame_idx: int,
        pose: Optional[CameraPose] = None,
        roi_frac: Optional[tuple[float, float, float, float]] = _DEFAULT_ROI_FRAC,
        field_poly_frac: Optional[list[tuple[float, float]]] = None,
        field_poly_px: Optional[np.ndarray] = None,
    ) -> tuple[list[Detection], PileEstimate]:
        """One-shot convenience: isolated-ball boxes AND the dense-pile area estimate.

        Equivalent to calling :meth:`detect` then :meth:`estimate_pile` with the detect()
        output as ``isolated_dets`` — but the caller gets both in a single call. The fuel
        mask is built twice (once per stage) for clarity; on this footage that is ~1-2 ms and
        keeps the two gates independent.

        Returns ``(isolated_dets, pile_estimate)``. The headline fuel number is
        ``pile_estimate.total_est`` (= len(isolated_dets) + pile_estimate.count_est), rendered
        via ``pile_estimate.label()``.
        """
        isolated = self.detect(frame, frame_idx, pose=pose, roi_frac=roi_frac)
        pile = self.estimate_pile(
            frame, frame_idx, isolated_dets=isolated, pose=pose,
            field_poly_frac=field_poly_frac, field_poly_px=field_poly_px,
        )
        return isolated, pile

    # -- field-polygon rasterizer (cv2) ------------------------------------- #
    def _field_poly_mask(
        self,
        hw: tuple[int, int],
        field_poly_frac: Optional[list[tuple[float, float]]],
        field_poly_px: Optional[np.ndarray],
    ) -> "np.ndarray | None":
        """0/255 mask of the field polygon, or None to mean 'no gate' (full frame).

        ``field_poly_px`` (a prebuilt pixel polygon, Nx2 or Nx1x2) takes precedence; else
        ``field_poly_frac`` (image-fraction points) is scaled to pixels. Returns None when
        neither is supplied.
        """
        cv2 = _cv2()
        h, w = int(hw[0]), int(hw[1])
        pts: Optional[np.ndarray] = None
        if field_poly_px is not None:
            pts = np.asarray(field_poly_px).reshape(-1, 2).astype(np.int32)
        elif field_poly_frac is not None:
            pts = np.array([(int(round(xf * w)), int(round(yf * h)))
                            for xf, yf in field_poly_frac], dtype=np.int32)
        if pts is None or len(pts) < 3:
            return None
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(m, [pts.reshape(-1, 1, 2)], 255)
        return m


# Module-level convenience matching the requested API signature exactly.
_DEFAULT_DETECTOR: Optional[BallBlobDetector] = None


def detect(
    frame: np.ndarray,
    frame_idx: int,
    pose: Optional[CameraPose] = None,
    roi_frac: Optional[tuple[float, float, float, float]] = _DEFAULT_ROI_FRAC,
) -> list[Detection]:
    """Detect fuel balls with a default-configured detector.

    Stateless from the caller's view; reuses a process-global detector so the CLAHE
    objects/kernels aren't rebuilt every call. For custom thresholds, instantiate
    ``BallBlobDetector(cfg_dict)`` and call its ``.detect``.
    """
    global _DEFAULT_DETECTOR
    if _DEFAULT_DETECTOR is None:
        _DEFAULT_DETECTOR = BallBlobDetector()
    return _DEFAULT_DETECTOR.detect(frame, frame_idx, pose=pose, roi_frac=roi_frac)


def detect_with_pile(
    frame: np.ndarray,
    frame_idx: int,
    pose: Optional[CameraPose] = None,
    roi_frac: Optional[tuple[float, float, float, float]] = _DEFAULT_ROI_FRAC,
    field_poly_frac: Optional[list[tuple[float, float]]] = None,
    field_poly_px: Optional[np.ndarray] = None,
) -> tuple[list[Detection], PileEstimate]:
    """Module-level :meth:`BallBlobDetector.detect_with_pile` on a process-global detector."""
    global _DEFAULT_DETECTOR
    if _DEFAULT_DETECTOR is None:
        _DEFAULT_DETECTOR = BallBlobDetector()
    return _DEFAULT_DETECTOR.detect_with_pile(
        frame, frame_idx, pose=pose, roi_frac=roi_frac,
        field_poly_frac=field_poly_frac, field_poly_px=field_poly_px,
    )
