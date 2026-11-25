"""Classical bumper-blob robot proposer — a recall-lifting *second* detector.

The NN (detect/robot_net.py) is the primary robot finder, but a tiny single-class
net misses robots under motion blur, partial occlusion at the field edge, or odd
poses. Those misses are expensive: a dropped robot breaks tracking identity and
event attribution for the whole alliance.

This module is the cheap classical complement. FRC robots wear *saturated* red or
blue bumper bands — by far the strongest color signal on the field. We threshold
those bands in HSV, glue each band into one blob (morphological close), pull the
connected components, and keep blobs whose size and shape look like a bumper region.
Each surviving blob becomes a robot *proposal*; the fusion layer later reconciles
these with the NN boxes (NN wins on overlap; classical-only boxes recover misses).

Design contract (matches the rest of the inference side):
  * Pure geometry/gating logic uses only Python floats + numpy and is unit-testable
    with NO cv2 (see the module-level helpers and their tests).
  * cv2 is imported lazily inside the methods that need it, so this file *parses*
    and the helpers *run* in a bare environment without OpenCV.
  * Gates are expressed as a FRACTION of frame area and as an aspect ratio, never as
    hardcoded pixel counts, so the detector is resolution-invariant (720p vs 1080p).
  * HSV bumper thresholds mirror robot_net.bumper_alliance so a proposal that is
    later re-voted for alliance lands on the same color.

propose(frame, field_mask=None) -> list[{"xyxy", "alliance", "score"}]
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from frcscout.io.types import Alliance

# --- HSV bumper thresholds (BGR frames -> cv2 HSV, H in [0,180]) ------------- #
# Kept in lockstep with detect/robot_net.bumper_alliance so a recovered box gets
# the same red/blue verdict when it is re-scored downstream.
_SAT_MIN = 90        # saturation floor: reject washed-out carpet / lighting
_VAL_MIN = 60        # value floor: reject deep shadow
_RED_LO_HI = 10      # red wraps the hue circle: H <= 10 ...
_RED_HI_LO = 170     # ... OR H >= 170
_BLUE_LO = 100       # blue band
_BLUE_HI = 130

# --- default size / shape gates (resolution-invariant) ---------------------- #
# A robot bumper region in a broadcast wide shot is a meaningful chunk of the
# frame but never tiny and never the whole image. Areas are fractions of H*W.
_DEF_MIN_AREA_FRAC = 0.0008   # ~ 0.08% of frame  (e.g. ~1660 px @ 1080p)
_DEF_MAX_AREA_FRAC = 0.15     # > 15% of frame is a banner / wall / graphic, not one robot
_DEF_MIN_ASPECT = 0.35        # w/h: bumpers are wide-ish; allow tall partial views
_DEF_MAX_ASPECT = 6.0         # extremely wide = a sponsor wall stripe, not a robot
_DEF_MIN_FILL = 0.35          # blob must fill this fraction of its bbox (reject thin Ls)
_DEF_MIN_BLOB_COLOR_FRAC = 0.18  # of the bbox area, after re-measuring inside the box


# =========================================================================== #
# Pure-logic helpers (NO cv2) — these carry the gating policy and are unit
# testable in a bare Python + numpy environment.
# =========================================================================== #

def _area_frac(xyxy: tuple[float, float, float, float], frame_hw: tuple[int, int]) -> float:
    """Bounding-box area as a fraction of the whole frame area."""
    x1, y1, x2, y2 = xyxy
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    fh, fw = frame_hw
    denom = float(fh) * float(fw)
    if denom <= 0:
        return 0.0
    return (w * h) / denom


def _aspect(xyxy: tuple[float, float, float, float]) -> float:
    """Width / height of the box. Returns inf for a zero-height box."""
    x1, y1, x2, y2 = xyxy
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    if h <= 0:
        return float("inf")
    return w / h


def _passes_gates(
    xyxy: tuple[float, float, float, float],
    frame_hw: tuple[int, int],
    *,
    min_area_frac: float = _DEF_MIN_AREA_FRAC,
    max_area_frac: float = _DEF_MAX_AREA_FRAC,
    min_aspect: float = _DEF_MIN_ASPECT,
    max_aspect: float = _DEF_MAX_ASPECT,
) -> bool:
    """True iff the box passes both the size (fraction-of-frame) and aspect gates.

    Pure function — the heart of the recall/precision trade. Tested directly.
    """
    af = _area_frac(xyxy, frame_hw)
    if not (min_area_frac <= af <= max_area_frac):
        return False
    asp = _aspect(xyxy)
    if not (min_aspect <= asp <= max_aspect):
        return False
    return True


def _box_score(fill: float, color_frac: float) -> float:
    """Confidence proxy for a classical proposal in [0, 1].

    `fill`       = blob_pixels / bbox_area  (how solid the blob is)
    `color_frac` = colored_pixels_in_bbox / bbox_area  (how bumper-colored the box is)

    Both should be high for a real bumper region. We combine them as the geometric
    mean so a box that is large but mostly background (low color_frac) is penalised
    even if the blob itself was solid. Deliberately capped below the NN's implicit
    1.0 so fusion prefers a real network box on a tie.
    """
    fill = min(1.0, max(0.0, float(fill)))
    color_frac = min(1.0, max(0.0, float(color_frac)))
    raw = (fill * color_frac) ** 0.5
    # Cap so a classical proposal never out-ranks a genuine NN detection.
    return min(0.85, raw)


def _clip_box(
    xyxy: tuple[float, float, float, float], frame_hw: tuple[int, int]
) -> tuple[float, float, float, float]:
    """Clamp a box to the frame rectangle [0,W]x[0,H]."""
    fh, fw = frame_hw
    x1, y1, x2, y2 = xyxy
    x1 = min(max(0.0, x1), float(fw))
    x2 = min(max(0.0, x2), float(fw))
    y1 = min(max(0.0, y1), float(fh))
    y2 = min(max(0.0, y2), float(fh))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def _iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """Intersection-over-union of two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _dedup_boxes(props: list[dict], iou_thresh: float = 0.55) -> list[dict]:
    """Greedy NMS over proposals (highest score first).

    Red and blue blobs that touch (a red robot next to a blue robot) have different
    `alliance`, so we only suppress a lower-scored box by a higher-scored one when
    they share an alliance — two adjacent different-color robots must both survive.
    Pure function (operates on the proposal dicts only). Tested directly.
    """
    order = sorted(props, key=lambda p: p["score"], reverse=True)
    kept: list[dict] = []
    for p in order:
        drop = False
        for k in kept:
            if k["alliance"] == p["alliance"] and _iou(k["xyxy"], p["xyxy"]) >= iou_thresh:
                drop = True
                break
        if not drop:
            kept.append(p)
    return kept


# =========================================================================== #
# The cv2-backed proposer.
# =========================================================================== #

class ClassicalRobotProposer:
    """Propose robot boxes from saturated red/blue bumper blobs.

    Cheap, model-free recall booster. Construct once; call propose() per frame.
    All cv2 use is lazy so importing this class never requires OpenCV.
    """

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.min_area_frac = float(cfg.get("min_area_frac", _DEF_MIN_AREA_FRAC))
        self.max_area_frac = float(cfg.get("max_area_frac", _DEF_MAX_AREA_FRAC))
        self.min_aspect = float(cfg.get("min_aspect", _DEF_MIN_ASPECT))
        self.max_aspect = float(cfg.get("max_aspect", _DEF_MAX_ASPECT))
        self.min_fill = float(cfg.get("min_fill", _DEF_MIN_FILL))
        self.min_color_frac = float(cfg.get("min_color_frac", _DEF_MIN_BLOB_COLOR_FRAC))
        # Morphological-close kernel size (px). Glues the two bumper bands of one
        # robot into a single component. Small + odd.
        self.close_ksize = int(cfg.get("close_ksize", 7))
        self.sat_min = int(cfg.get("sat_min", _SAT_MIN))
        self.val_min = int(cfg.get("val_min", _VAL_MIN))

    # ----- HSV masks -------------------------------------------------------- #
    def _color_masks(self, frame) -> tuple["np.ndarray", "np.ndarray"]:
        """Return (red_mask, blue_mask) as uint8 0/255 arrays for a BGR frame."""
        import cv2

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h = hsv[..., 0]
        s = hsv[..., 1]
        v = hsv[..., 2]
        sat = (s > self.sat_min) & (v > self.val_min)
        red = sat & ((h <= _RED_LO_HI) | (h >= _RED_HI_LO))
        blue = sat & (h >= _BLUE_LO) & (h <= _BLUE_HI)
        return (
            (red.astype(np.uint8) * 255),
            (blue.astype(np.uint8) * 255),
        )

    def _proposals_from_mask(
        self,
        mask,
        alliance: Alliance,
        frame_hw: tuple[int, int],
    ) -> list[dict]:
        """Connected-components on one color mask -> gated proposal dicts."""
        import cv2

        # Glue the (top + bottom) bumper bands of one robot into a single blob.
        if self.close_ksize >= 2:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.close_ksize, self.close_ksize)
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out: list[dict] = []
        for lbl in range(1, num):  # 0 is background
            x = stats[lbl, cv2.CC_STAT_LEFT]
            y = stats[lbl, cv2.CC_STAT_TOP]
            w = stats[lbl, cv2.CC_STAT_WIDTH]
            hgt = stats[lbl, cv2.CC_STAT_HEIGHT]
            blob_area = int(stats[lbl, cv2.CC_STAT_AREA])
            xyxy = _clip_box((float(x), float(y), float(x + w), float(y + hgt)), frame_hw)

            if not _passes_gates(
                xyxy,
                frame_hw,
                min_area_frac=self.min_area_frac,
                max_area_frac=self.max_area_frac,
                min_aspect=self.min_aspect,
                max_aspect=self.max_aspect,
            ):
                continue

            bx1, by1, bx2, by2 = xyxy
            bbox_area = max(1.0, (bx2 - bx1) * (by2 - by1))
            fill = blob_area / bbox_area
            if fill < self.min_fill:
                continue

            # color_frac: how much of the *bounding box* is this color (re-measured
            # on the bbox crop of the morph-closed mask). A solid bumper region
            # scores high; a sparse scatter that happened to span a big bbox is cut.
            crop = mask[int(by1):int(by2), int(bx1):int(bx2)]
            color_frac = float((crop > 0).sum()) / bbox_area if crop.size else 0.0
            if color_frac < self.min_color_frac:
                continue

            out.append(
                {
                    "xyxy": xyxy,
                    "alliance": alliance,
                    "score": _box_score(fill, color_frac),
                }
            )
        return out

    # ----- public API ------------------------------------------------------- #
    def propose(self, frame, field_mask=None) -> list[dict]:
        """Propose robot boxes from bumper blobs in a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray (H, W, 3) BGR
        field_mask : optional uint8 0/255 array (H, W). When given (e.g. from
            FieldModel.field_mask once a pose is solved), color detection is
            restricted to the carpet — this kills red/blue *crowd* and sponsor
            signage in the stands, the dominant false-positive source. When None
            (cold start, no pose yet) the whole frame is searched.

        Returns
        -------
        list of {"xyxy": (x1,y1,x2,y2), "alliance": "red"|"blue", "score": float}
        Deduplicated within each alliance. Empty list on an empty/degenerate frame.
        """
        import cv2  # noqa: F401  (kept here so propose() fails loudly if cv2 absent)

        if frame is None or getattr(frame, "size", 0) == 0:
            return []
        frame_hw = (int(frame.shape[0]), int(frame.shape[1]))

        red_mask, blue_mask = self._color_masks(frame)

        if field_mask is not None:
            fm = field_mask
            if fm.dtype != np.uint8:
                fm = (fm > 0).astype(np.uint8) * 255
            # Match shape defensively; if it mismatches, ignore the mask rather
            # than crash the whole frame.
            if fm.shape[:2] == red_mask.shape[:2]:
                import cv2 as _cv2

                red_mask = _cv2.bitwise_and(red_mask, fm)
                blue_mask = _cv2.bitwise_and(blue_mask, fm)

        props = (
            self._proposals_from_mask(red_mask, "red", frame_hw)
            + self._proposals_from_mask(blue_mask, "blue", frame_hw)
        )
        return _dedup_boxes(props)
