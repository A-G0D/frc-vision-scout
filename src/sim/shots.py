"""
shots.py — camera-cut detection and segment labelling for FRC REBUILT (2026)
broadcast footage.

Algorithm
---------
1. Sample one frame per second (step_s).
2. Compute a per-frame histogram signature over the field-of-play ROI
   (skip the scoreboard bar and barrier/reflection strip).
   Signature: 32-bin Hue + 16-bin Value, each channel normalised independently,
   using cv2.calcHist + cv2.normalize so the result works directly with
   cv2.compareHist(HISTCMP_BHATTACHARYYA).
3. Compute cv2.compareHist(HISTCMP_BHATTACHARYYA) between consecutive signatures.
   This returns values in [0, ~1.0]; near-zero = very similar, high = very different.
4. Threshold to detect hard cuts (default 0.20 — empirically verified on match_full.mp4;
   the one real cut scores ~0.56; highest in-scene motion scores ~0.10).
5. Merge tiny inter-cut segments (< MIN_SEG_S) into their neighbours.
6. Label each segment:
   - 'wide'  — duration >= wide_min_s AND the midpoint frame looks like a field shot
               (not a full-screen graphic, not dominated by saturated crowd colours).
   - 'other' — short or non-field segment.

Verified on data/sim/match_full.mp4 (Upper Bracket R4 M11, 178s, 640×360, 30fps):
  Expected output:
    seg 0 — start_f=0,    end_f=4166,  dur=138.9s,  kind='wide'
    seg 1 — start_f=4166, end_f=5340,  dur=39.1s,   kind='other'
  This was confirmed by visual inspection of the transition frames.

Public API
----------
  segment_shots(video_path, *, step_s=1.0, cut_threshold=0.20,
                wide_min_s=10.0) -> list[dict]

Each dict: {'start_f': int, 'end_f': int, 'kind': 'wide' | 'other'}
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Tunable parameters (all exposed as keyword args on segment_shots too)
# ---------------------------------------------------------------------------

_DEFAULT_STEP_S      = 1.0    # sample every N seconds
_DEFAULT_CUT_THRESH  = 0.20   # Bhattacharyya distance that signals a cut
                               # (verified: real cut ~0.56, intra-scene max ~0.10)
_DEFAULT_WIDE_MIN_S  = 10.0   # minimum seconds for a segment to be 'wide'
_MIN_SEG_S           = 2.0    # short segments shorter than this are merged


# ---------------------------------------------------------------------------
# Field-of-play ROI (excludes scoreboard bar and barrier/reflection strip)
# ---------------------------------------------------------------------------

def _roi_crop(frame: np.ndarray) -> np.ndarray:
    """Crop to the mid-frame field-of-play region.

    Excludes:
      - Top 13%  : scoreboard bar
      - Bottom 20%: barrier, reflected region, operator consoles
      - Sides 10% each: partial exclusion of side crowds/stands
    """
    h, w = frame.shape[:2]
    y0 = int(0.13 * h)
    y1 = int(0.80 * h)
    x0 = int(0.10 * w)
    x1 = int(0.90 * w)
    return frame[y0:y1, x0:x1]


# ---------------------------------------------------------------------------
# Histogram signature — two normalised histograms ready for compareHist
# ---------------------------------------------------------------------------

def _hist_signature(frame: np.ndarray):
    """Return (hue_hist, val_hist) — each a normalised cv2 histogram ndarray.

    Using 32 Hue bins and 16 Value bins; computed over the field-of-play ROI.
    Both histograms are normalised so cv2.compareHist(HISTCMP_BHATTACHARYYA)
    returns a proper distance in [0, 1].
    """
    roi = _roi_crop(frame)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256])

    cv2.normalize(h_hist, h_hist)
    cv2.normalize(v_hist, v_hist)

    return h_hist, v_hist


def _hist_distance(sig_a, sig_b) -> float:
    """Average Bhattacharyya distance across hue and value channels.

    Returns values in [0, ~1.0]; 0 = identical scenes, high = hard cut.
    """
    hh_a, vh_a = sig_a
    hh_b, vh_b = sig_b
    dh = cv2.compareHist(hh_a, hh_b, cv2.HISTCMP_BHATTACHARYYA)
    dv = cv2.compareHist(vh_a, vh_b, cv2.HISTCMP_BHATTACHARYYA)
    return (dh + dv) / 2.0


# ---------------------------------------------------------------------------
# Cut detection by sampling the video at step_s intervals
# ---------------------------------------------------------------------------

def _detect_cuts(cap: cv2.VideoCapture,
                 step_s: float,
                 threshold: float) -> List[int]:
    """Return list of frame indices where a hard cut is detected.

    Each returned frame index is the FIRST frame of the new scene.
    The cut is placed at the frame where the histogram distance to the
    previous sample exceeds `threshold`.  Because we sample at step_s
    intervals (not frame-by-frame), the detected cut frame is at most
    step_s seconds late.  For this match, step_s=1.0s is sufficient since
    the cut is a hard instantaneous switch.
    """
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step_f = max(1, int(round(fps * step_s)))

    prev_sig = None
    cut_frames: List[int] = []

    for fi in range(0, total, step_f):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            break
        sig = _hist_signature(frame)
        if prev_sig is not None:
            dist = _hist_distance(prev_sig, sig)
            if dist > threshold:
                cut_frames.append(fi)
        prev_sig = sig

    return cut_frames


# ---------------------------------------------------------------------------
# Segment builder
# ---------------------------------------------------------------------------

def _build_segments(cut_frames: List[int],
                    total_frames: int,
                    fps: float,
                    min_seg_s: float) -> List[Dict]:
    """Turn a list of cut-start frame indices into contiguous segments.

    Segments shorter than min_seg_s are merged into their next neighbour
    (forward merge preference).  Returns list of {'start_f', 'end_f'}.
    """
    boundaries = sorted(set([0] + cut_frames + [total_frames]))
    segs = [{"start_f": boundaries[i], "end_f": boundaries[i + 1]}
            for i in range(len(boundaries) - 1)]

    min_f = int(round(min_seg_s * fps))
    changed = True
    while changed:
        changed = False
        merged: List[Dict] = []
        i = 0
        while i < len(segs):
            seg = segs[i]
            length = seg["end_f"] - seg["start_f"]
            if length < min_f and i + 1 < len(segs):
                # Merge into next segment (absorb the short one).
                segs[i + 1] = {"start_f": seg["start_f"],
                                "end_f":   segs[i + 1]["end_f"]}
                changed = True
                i += 1
            else:
                merged.append(seg)
                i += 1
        segs = merged

    return segs


# ---------------------------------------------------------------------------
# 'wide' vs 'other' classifier
# ---------------------------------------------------------------------------

def _is_wide_segment(cap: cv2.VideoCapture,
                     seg: Dict,
                     fps: float,
                     wide_min_s: float) -> bool:
    """Return True if this segment should be labelled 'wide'.

    Criteria (both must be met):
    (a) Duration >= wide_min_s seconds.
    (b) Midpoint frame is a field view — not a full-screen graphic.
        Field views have:
          - A significant grey carpet region (low saturation, mid value).
          - The ROI is NOT overwhelmingly saturated (no garish full-screen
            graphic or close-up with solid colour blocks).

    For match_full.mp4, this cleanly separates the 138s match-play segment
    (wide) from the 39s post-match animation+results segment (other).
    """
    duration_s = (seg["end_f"] - seg["start_f"]) / fps
    if duration_s < wide_min_s:
        return False

    # Sample the midpoint frame.
    mid_f = (seg["start_f"] + seg["end_f"]) // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_f)
    ok, frame = cap.read()
    if not ok:
        return True  # Assume wide if unreadable.

    roi = _roi_crop(frame)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    s_ch = hsv[..., 1].astype(np.float32)
    v_ch = hsv[..., 2].astype(np.float32)

    # Fraction highly saturated (bright coloured pixels): S>160, V>60.
    sat_frac = float(((s_ch > 160) & (v_ch > 60)).mean())

    # Fraction of grey-ish pixels (carpet-like): S<60, V in [50,210].
    grey_frac = float(((s_ch < 60) & (v_ch > 50) & (v_ch < 210)).mean())

    # Field view: NOT overwhelmingly saturated AND has visible grey carpet.
    return sat_frac < 0.60 and grey_frac > 0.08


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment_shots(
    video_path: str,
    *,
    step_s: float = _DEFAULT_STEP_S,
    cut_threshold: float = _DEFAULT_CUT_THRESH,
    wide_min_s: float = _DEFAULT_WIDE_MIN_S,
) -> List[Dict]:
    """Detect camera cuts and label segments as 'wide' or 'other'.

    Parameters
    ----------
    video_path     : Path to the input video file.
    step_s         : Histogram-sample interval in seconds (default 1.0).
    cut_threshold  : Bhattacharyya distance threshold for cut detection.
                     0.20 works for this broadcast — the one hard cut scores
                     ~0.56 while the highest in-scene transition scores ~0.10.
    wide_min_s     : Minimum duration (seconds) for a segment to be labelled
                     'wide'.  Shorter segments are always 'other'.

    Returns
    -------
    List of dicts: [{'start_f': int, 'end_f': int, 'kind': 'wide'|'other'}]
    Sorted by start_f, non-overlapping, covering [0, total_frames).

    Verified on match_full.mp4 (Upper Bracket R4 M11):
      - 2 segments: 0–4166 'wide' (138.9s), 4166–5340 'other' (39.1s)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 1. Detect hard cuts.
        cut_frames = _detect_cuts(cap, step_s, cut_threshold)

        # 2. Build segments (merge tiny fragments).
        segs = _build_segments(cut_frames, total, fps, _MIN_SEG_S)

        # 3. Label each segment.
        labeled: List[Dict] = []
        for seg in segs:
            kind = "wide" if _is_wide_segment(cap, seg, fps, wide_min_s) else "other"
            labeled.append({
                "start_f": seg["start_f"],
                "end_f":   seg["end_f"],
                "kind":    kind,
            })

    finally:
        cap.release()

    return labeled


# ---------------------------------------------------------------------------
# CLI / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    video = (sys.argv[1] if len(sys.argv) > 1
             else "C:/Users/agodavarthy/Claude/projects/frc-vision-scout/data/sim/match_full.mp4")

    print(f"Analysing: {video}\n")
    segments = segment_shots(video)

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"{'#':>3}  {'start_f':>8}  {'end_f':>8}  {'start_t':>8}  {'end_t':>7}  {'dur_s':>7}  kind")
    print("-" * 60)
    for i, seg in enumerate(segments):
        t0 = seg["start_f"] / fps
        t1 = seg["end_f"] / fps
        dur = t1 - t0
        print(f"{i:>3}  {seg['start_f']:>8}  {seg['end_f']:>8}  "
              f"{t0:>7.1f}s  {t1:>6.1f}s  {dur:>7.1f}  {seg['kind']}")

    wide  = [s for s in segments if s["kind"] == "wide"]
    other = [s for s in segments if s["kind"] == "other"]
    wide_s  = sum((s["end_f"] - s["start_f"]) / fps for s in wide)
    other_s = sum((s["end_f"] - s["start_f"]) / fps for s in other)

    print(f"\nTotal segments : {len(segments)}")
    print(f"  wide         : {len(wide)}   ({wide_s:.1f}s covered)")
    print(f"  other        : {len(other)}   ({other_s:.1f}s covered)")
    print(f"Coverage       : {wide_s + other_s:.1f}s / {total/fps:.1f}s total")

    # Save JSON alongside the video.
    out_path = Path(video).with_name("shots.json")
    with open(out_path, "w") as f:
        json.dump(segments, f, indent=2)
    print(f"\nSegment list saved: {out_path}")
