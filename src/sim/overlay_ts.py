"""
overlay_ts.py — Read the live scoreboard bar at every N seconds of footage and
write a per-second time-series to data/sim/overlay_ts.json.

This is THE ORACLE for downstream verification: the overlay fuel/score counter
increments in real-time so the JSON gives ground-truth score at each second.

Public API
----------
read_timeseries(video_path, every_s=1.0) -> list[dict]
    Each dict: {t, blue_score, red_score, blue_fuel, red_fuel, timer}
    Also writes data/sim/overlay_ts.json (path relative to the project root,
    which is detected as the directory two levels above this file).

Frame-type detection
--------------------
Two overlay layouts exist in this video:

  LIVE BAR (t=0..~103s): standard scoreboard strip at top (y~0.07-0.13).
    Detected by is_live_bar() — blue block left, red block right at y~0.09.

  RESULTS SCREEN (t~150..178s): post-match full-screen results graphic.
    Large white score numbers centered on blue/red panels at y~0.12-0.38.
    Detected by is_results_screen() — blue/red color fractions in center band.

  CAMERA FIELD (t~104..148s): raw field view with no overlay bar.
    Neither live bar nor results screen → forward-fill from last known values.

Sanity checks (logged to stderr, not raised):
  - scores are non-decreasing (monotone)
  - final blue ≈ 268, final red ≈ 506 (TBA ground truth for this match)

Usage
-----
    python src/sim/overlay_ts.py [video_path] [--every SECS]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Project layout helpers
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_SRC  = _HERE.parent.parent          # .../src/
_ROOT = _SRC.parent                  # project root
_DATA_DIR = _ROOT / "data" / "sim"
_OUT_JSON = _DATA_DIR / "overlay_ts.json"

# ---------------------------------------------------------------------------
# Import from src/overlay_live
# ---------------------------------------------------------------------------
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from overlay_live import LiveBarReader, is_live_bar  # noqa: E402


# ---------------------------------------------------------------------------
# Results-screen detection and reading
# ---------------------------------------------------------------------------
# ROI fractions measured on 640x360 results screen (frame at t~150s):
#   '268' centroid: (0.406, 0.225)  '506' centroid: (0.594, 0.228)
# Blue score panel: left half center; Red score panel: right half center.
_RES_BLUE_ROI  = (0.28, 0.12, 0.49, 0.32)   # contains "268"
_RES_RED_ROI   = (0.49, 0.12, 0.70, 0.32)   # contains "506"
# Detection probe: center band y~0.10-0.35 for blue/red saturation
_RES_BLUE_PROBE = (0.20, 0.10, 0.48, 0.35)
_RES_RED_PROBE  = (0.48, 0.10, 0.78, 0.35)


def _crop(img: np.ndarray, roi: tuple) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = roi
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def _color_frac(bgr: np.ndarray, which: str) -> float:
    if bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sat = (s > 80) & (v > 60)
    if which == "blue":
        m = sat & (h >= 100) & (h <= 130)
    else:
        m = sat & ((h <= 10) | (h >= 170))
    return float(m.mean())


def is_results_screen(frame: np.ndarray) -> bool:
    """Detect the post-match results screen.

    The results screen has LARGE SOLID-COLOR blue/red panels that fill the
    live-bar probe regions nearly completely (color fraction ≈ 0.95+), whereas
    the actual live bar only uses those regions partially (color fraction 0.4-0.8).

    Thresholds measured on this video:
      live bar:        lb ≈ 0.69,  lr ≈ 0.74
      results screen:  lb ≈ 0.97,  lr ≈ 0.97
    """
    from overlay_live import _BLUE_BLK, _RED_BLK
    # Very high saturation in the live-bar probe zones = results screen solid panels
    lb = _color_frac(_crop(frame, _BLUE_BLK), "blue")
    lr = _color_frac(_crop(frame, _RED_BLK), "red")
    return lb >= 0.90 and lr >= 0.90


def _read_results_scores(reader: LiveBarReader, frame: np.ndarray) -> tuple[int | None, int | None]:
    """Read blue and red scores from the post-match results screen."""
    blue_crop = _crop(frame, _RES_BLUE_ROI)
    red_crop  = _crop(frame, _RES_RED_ROI)
    # Use _read_score (picks the tallest digit group)
    b = reader._read_score(blue_crop)
    r = reader._read_score(red_crop)
    # Sanity bounds: match scores are 0-9999
    if b is not None and not (0 <= b <= 9999):
        b = None
    if r is not None and not (0 <= r <= 9999):
        r = None
    return b, r


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def _forward_fill(ts: list[dict]) -> list[dict]:
    """For any None field, carry the last known value forward.
    Scores are non-decreasing so this is safe — a missed read means
    no change in score, not a reset."""
    last: dict = {
        "blue_score": None,
        "red_score":  None,
        "blue_fuel":  None,
        "red_fuel":   None,
        "timer":      None,
    }
    for row in ts:
        for k in last:
            if row[k] is not None:
                last[k] = row[k]
            else:
                row[k] = last[k]
    return ts


def _repair_monotone(ts: list[dict]) -> tuple[list[dict], list[str]]:
    """Clamp any score decrease (OCR glitch -> wrong digit) to the previous valid value.
    Returns (repaired_ts, list_of_warning_messages)."""
    notes: list[str] = []
    prev_b = prev_r = 0
    for row in ts:
        b = row["blue_score"]
        r = row["red_score"]
        if b is not None:
            if b < prev_b:
                notes.append(
                    f"t={row['t']:.1f}s: blue_score decreased {prev_b}->{b} (clamped)"
                )
                row["blue_score"] = prev_b
            else:
                prev_b = b
        if r is not None:
            if r < prev_r:
                notes.append(
                    f"t={row['t']:.1f}s: red_score decreased {prev_r}->{r} (clamped)"
                )
                row["red_score"] = prev_r
            else:
                prev_r = r
    return ts, notes


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def read_timeseries(
    video_path: str | Path,
    every_s: float = 1.0,
    *,
    verbose: bool = True,
) -> list[dict]:
    """
    Sample the video at `every_s` second intervals, read the live scoreboard
    bar (or post-match results screen) at each sample, forward-fill missing
    reads, enforce monotone scores, write data/sim/overlay_ts.json, and return
    the list.

    Returns a list of dicts, one per sample:
        {t, frame, blue_score, red_score, blue_fuel, red_fuel, timer,
         live_bar, results_screen, read_ok}
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    if verbose:
        print(
            f"[overlay_ts] {video_path.name}  fps={fps:.1f}  "
            f"frames={total_frames}  duration={duration:.1f}s",
            file=sys.stderr,
        )

    # Build sample timestamps: 0, every_s, 2*every_s, ...
    times: list[float] = []
    t = 0.0
    while t <= duration + 0.01:
        times.append(t)
        t += every_s

    reader = LiveBarReader(gpu=False)
    results: list[dict] = []
    failed_frames: list[float] = []
    non_live_frames: list[float] = []
    results_screen_frames: list[float] = []

    t0_wall = time.perf_counter()
    for i, ts in enumerate(times):
        frame_idx = min(int(round(ts * fps)), total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            failed_frames.append(ts)
            results.append(
                {
                    "t": round(ts, 3),
                    "frame": frame_idx,
                    "blue_score": None,
                    "red_score": None,
                    "blue_fuel": None,
                    "red_fuel": None,
                    "timer": None,
                    "live_bar": False,
                    "results_screen": False,
                    "read_ok": False,
                }
            )
            continue

        # Check results screen FIRST — it also triggers is_live_bar (solid panels)
        res_screen = is_results_screen(frame)
        live = is_live_bar(frame) and not res_screen

        blue_s: int | None = None
        red_s: int | None = None
        blue_f: int | None = None
        red_f: int | None = None
        timer_s: str | None = None

        if res_screen:
            results_screen_frames.append(ts)
            b, r = _read_results_scores(reader, frame)
            blue_s = b
            red_s  = r
            # Fuel and timer not available on results screen
        elif live:
            bar = reader.read(frame)
            blue_s = bar.blue_score
            red_s  = bar.red_score
            blue_f = bar.blue_fuel
            red_f  = bar.red_fuel
            timer_s = bar.timer
        else:
            non_live_frames.append(ts)

        read_ok = (blue_s is not None) or (red_s is not None)

        results.append(
            {
                "t": round(ts, 3),
                "frame": frame_idx,
                "blue_score": blue_s,
                "red_score": red_s,
                "blue_fuel": blue_f,
                "red_fuel": red_f,
                "timer": timer_s,
                "live_bar": live,
                "results_screen": res_screen,
                "read_ok": read_ok,
            }
        )

        if verbose and (i % 20 == 0 or i == len(times) - 1):
            elapsed = time.perf_counter() - t0_wall
            pct = 100.0 * (i + 1) / len(times)
            eta = (elapsed / (i + 1)) * (len(times) - i - 1)
            ftype = "LIVE" if live else ("RESULTS" if res_screen else "FIELD")
            print(
                f"  t={ts:6.1f}s  B={blue_s}  R={red_s}"
                f"  bf={blue_f}  rf={red_f}  T={timer_s}"
                f"  [{ftype}]  [{pct:.0f}% ETA {eta:.0f}s]",
                file=sys.stderr,
            )

    cap.release()

    # ---- Post-processing ------------------------------------------------
    results = _forward_fill(results)
    results, mono_notes = _repair_monotone(results)

    # ---- Sanity checks --------------------------------------------------
    san_notes: list[str] = []

    n_ok     = sum(1 for r in results if r["read_ok"])
    n_live   = sum(1 for r in results if r["live_bar"])
    n_res    = sum(1 for r in results if r["results_screen"])
    n_field  = sum(1 for r in results if not r["live_bar"] and not r["results_screen"])
    san_notes.append(
        f"Samples: {len(results)}  read_ok={n_ok}  live_bar={n_live}"
        f"  results_screen={n_res}  raw_field={n_field}"
        f"  failed_decode={len(failed_frames)}"
    )

    final_b = next((r["blue_score"] for r in reversed(results) if r["blue_score"] is not None), None)
    final_r = next((r["red_score"] for r in reversed(results) if r["red_score"] is not None), None)
    san_notes.append(f"Final blue_score={final_b}  final_red_score={final_r}")

    TBA_BLUE, TBA_RED = 268, 506
    if final_b is not None:
        diff_b = abs(final_b - TBA_BLUE)
        tag = "OK" if diff_b <= 5 else "WARN"
        san_notes.append(f"[{tag}] blue vs TBA({TBA_BLUE}): diff={diff_b}")
    if final_r is not None:
        diff_r = abs(final_r - TBA_RED)
        tag = "OK" if diff_r <= 5 else "WARN"
        san_notes.append(f"[{tag}] red vs TBA({TBA_RED}): diff={diff_r}")

    if failed_frames:
        san_notes.append(f"  decode failures at t={failed_frames}")
    if non_live_frames:
        san_notes.append(
            f"  raw_field frames (no overlay) at t={non_live_frames[:10]}"
            + (" ..." if len(non_live_frames) > 10 else "")
        )
    if results_screen_frames:
        san_notes.append(
            f"  results_screen frames at t={results_screen_frames[:5]}"
            + (" ..." if len(results_screen_frames) > 5 else "")
        )
    san_notes.extend(mono_notes)

    for note in san_notes:
        print(f"[overlay_ts] {note}", file=sys.stderr)

    # ---- Write JSON -------------------------------------------------------
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "video": str(video_path),
        "fps": fps,
        "every_s": every_s,
        "n_samples": len(results),
        "final_blue": final_b,
        "final_red": final_r,
        "tba_blue": TBA_BLUE,
        "tba_red": TBA_RED,
        "sanity_notes": san_notes,
        "timeseries": results,
    }
    with open(_OUT_JSON, "w") as fh:
        json.dump(payload, fh, indent=2)

    if verbose:
        print(f"[overlay_ts] Wrote {_OUT_JSON}  ({len(results)} rows)", file=sys.stderr)

    # Return just the time-series list (caller interface)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Read FRC overlay time series from video")
    ap.add_argument("video_path", nargs="?",
                    default=str(_DATA_DIR / "match_full.mp4"),
                    help="Path to the match video (default: data/sim/match_full.mp4)")
    ap.add_argument("--every", type=float, default=1.0,
                    help="Sample interval in seconds (default: 1.0)")
    ap.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = ap.parse_args()

    ts = read_timeseries(args.video_path, every_s=args.every, verbose=not args.quiet)
    print(f"\nDone. {len(ts)} samples written to {_OUT_JSON}")
