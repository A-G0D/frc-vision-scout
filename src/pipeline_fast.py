"""
pipeline_fast.py — ONE fast annotated pass over an FRC REBUILT (2026) broadcast video.

Design goal: process a full 9-hour broadcast in ~15 minutes of wall-clock.

Speed strategy
--------------
1. DECODE-TIME DECIMATION + DOWNSCALE.  We never decode every frame in OpenCV.
   Instead ffmpeg does `-vf "fps=SAMPLE_FPS,scale=640:360"` and streams raw
   bgr24 bytes on stdout, which we read in fixed-size (W*H*3) chunks straight
   into numpy.  ffmpeg seeks/decodes far faster than per-frame cv2 grabs and the
   fps filter throws away ~96-98% of frames before they ever reach Python.

2. CLASSIFY EVERY SAMPLED FRAME (cheap HSV stats, ~1-3 ms) -> tint the frame
   GREEN/YELLOW/RED for live/info/filler.

3. HEAVY OBJECT DETECTORS (fuel/robots/structures) run ONLY on 'live' frames.

4. OCR THROTTLE.  easyocr is the single expensive step (tens-to-hundreds of ms).
   The live scoreboard changes slowly, so we OCR-read the bar only once every
   OCR_EVERY_S seconds of footage and reuse the cached read on intervening live
   frames.  This keeps mean throughput high while still surfacing the live score.
   Set OCR_EVERY_S very large (or ocr=False) to disable OCR entirely for max speed.

fps arithmetic (9 h = 32400 s; 15 min target = 900 s wall-clock):
    SAMPLE_FPS = 1.0  -> 32400 sampled frames -> need >= 36.0 frames/s end-to-end.
    SAMPLE_FPS = 0.5  -> 16200 sampled frames -> need >= 18.0 frames/s end-to-end.
We default to SAMPLE_FPS = 1.0 and verify the measured throughput clears 36 fps.

CLI:
    python src/pipeline_fast.py --benchmark           # 3-min span, no sample dump
    python src/pipeline_fast.py --samples             # dump annotated live/info/filler
    python src/pipeline_fast.py --ss 600 --t 180 ...  # custom span
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import detect_fuel
import detect_robots
import detect_structures
import frame_classify

# ------------------------------------------------------------------ config ----

FFMPEG = (
    r"C:\Users\agodavarthy\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe"
)
if not os.path.exists(FFMPEG):
    FFMPEG = "ffmpeg"  # fall back to PATH

VIDEO = r"C:\Users\agodavarthy\Claude\projects\frc-vision-scout\data\ngvsHrmoW8U_360p.mp4"

W, H = 640, 360
FRAME_BYTES = W * H * 3

SAMPLE_FPS = 1.0          # frames sampled per second of footage
# OCR (easyocr) is the one heavy step (~1.1-1.5 s/read for the 5 score ROIs), so we
# throttle it hard.  Reading the live bar once per OCR_EVERY_S seconds of FOOTAGE
# (not frames) is plenty — scores change slowly and matches run ~2.5 min — while
# keeping the OCR time budget small enough to clear the 15-min/9h target.
OCR_EVERY_S = 60.0        # read the live scoreboard at most this often (seconds of footage)

FULL_VIDEO_HOURS = 9.0    # extrapolation target

TINT = {                  # BGR overlay colors
    "live":   (0, 200, 0),    # green
    "info":   (0, 200, 255),  # yellow
    "filler": (0, 0, 200),    # red
}
TINT_ALPHA = 0.22         # translucency of the whole-frame tint


# ----------------------------------------------------------- ffmpeg stream ----

def ffmpeg_frames(video: str, sample_fps: float, ss: float | None = None,
                  t: float | None = None):
    """Yield downscaled bgr24 numpy frames decoded+decimated by ffmpeg.

    -ss before -i is a fast (keyframe) seek; -t bounds the span.  The fps filter
    decimates and scale guarantees a fixed 640x360 so each chunk is FRAME_BYTES.
    """
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error"]
    if ss is not None:
        cmd += ["-ss", str(ss)]
    cmd += ["-i", video]
    if t is not None:
        cmd += ["-t", str(t)]
    cmd += [
        "-vf", f"fps={sample_fps},scale={W}:{H}",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-an", "-sn", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, bufsize=FRAME_BYTES * 4)
    try:
        while True:
            buf = proc.stdout.read(FRAME_BYTES)
            if len(buf) < FRAME_BYTES:
                break
            yield np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.wait()


# --------------------------------------------------------------- annotate ----

def _tint(frame: np.ndarray, label: str) -> np.ndarray:
    """Blend a translucent full-frame color wash for the class."""
    color = np.full_like(frame, TINT[label])
    return cv2.addWeighted(frame, 1.0 - TINT_ALPHA, color, TINT_ALPHA, 0)


def _put(img, text, org, color, scale=0.5, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thick, cv2.LINE_AA)


def annotate_live(frame, balls, robots, structs, score):
    """Tint green + draw every detected object + counts + live score read."""
    out = _tint(frame, "live")

    # FUEL — yellow circles
    for x, y, r in balls:
        cv2.circle(out, (x, y), r + 1, (0, 255, 255), 1)
    # ROBOTS — alliance-colored boxes
    for rb in robots:
        x, y, w, h = rb["bbox"]
        col = (255, 60, 0) if rb["alliance"] == "blue" else (0, 0, 255)
        cv2.rectangle(out, (x, y), (x + w, y + h), col, 2)
    # STRUCTURES (HUBs) — orange boxes
    for sd in structs:
        x, y, w, h = sd["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 165, 255), 2)
        _put(out, "HUB", (x, max(10, y - 3)), (0, 165, 255), 0.4, 1)

    nb = sum(1 for r in robots if r["alliance"] == "blue")
    nr = sum(1 for r in robots if r["alliance"] == "red")

    # Corner counts
    _put(out, "LIVE", (10, 22), (0, 255, 0), 0.7, 2)
    _put(out, f"FUEL:{len(balls)}  ROBOTS b{nb}/r{nr}  HUBS:{len(structs)}",
         (10, 44), (255, 255, 255), 0.5, 1)

    # Live score read (cached OCR)
    if score is not None:
        st = (f"SCORE  B:{score.get('blue_score')}  R:{score.get('red_score')}"
              f"  T:{score.get('timer')}"
              f"  fuel b:{score.get('blue_fuel')}/r:{score.get('red_fuel')}")
        _put(out, st, (10, H - 12), (255, 255, 255), 0.45, 1)
    return out


def annotate_simple(frame, label):
    out = _tint(frame, label)
    _put(out, label.upper(), (10, 22), TINT[label], 0.7, 2)
    return out


# -------------------------------------------------------------- pipeline ----

@dataclass
class Stats:
    frames: int = 0
    live: int = 0
    info: int = 0
    filler: int = 0
    ocr_calls: int = 0
    decode_s: float = 0.0
    work_s: float = 0.0


def run(video=VIDEO, sample_fps=SAMPLE_FPS, ss=None, t=None,
        ocr=True, ocr_every_s=OCR_EVERY_S, dump_dir=None, dump_budget=None):
    """Single annotated pass.  Returns (Stats, wall_seconds, dumped_paths)."""
    # Lazy OCR reader: only built if we actually OCR (keeps benchmark honest /
    # avoids the multi-second easyocr init when ocr=False).
    reader = None
    if ocr:
        try:
            from overlay_live import LiveBarReader
            reader = LiveBarReader()
        except Exception as e:  # easyocr missing -> degrade gracefully
            print(f"[warn] OCR disabled: {e}", file=sys.stderr)
            reader = None

    ocr_every_frames = max(1, int(round(ocr_every_s * sample_fps)))
    cached_score = None
    frames_since_ocr = 10 ** 9

    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
    want = {"live": 0, "info": 0, "filler": 0}  # how many of each dumped
    dumped = []

    st = Stats()
    wall0 = time.perf_counter()
    gen = ffmpeg_frames(video, sample_fps, ss=ss, t=t)

    # Pull-with-timing loop: time spent blocked on the generator ~ decode time.
    while True:
        td = time.perf_counter()
        frame = next(gen, None)
        st.decode_s += time.perf_counter() - td
        if frame is None:
            break

        tw = time.perf_counter()
        st.frames += 1
        res = frame_classify.classify(frame)
        label = res.label

        if label == "live":
            st.live += 1
            balls, _ = detect_fuel.detect(frame, draw=False)
            robots, _ = detect_robots.detect(frame, draw=False)
            structs, _ = detect_structures.detect(frame, draw=False)

            if reader is not None and frames_since_ocr >= ocr_every_frames:
                try:
                    lb = reader.read(frame)
                    cached_score = {
                        "blue_score": lb.blue_score, "red_score": lb.red_score,
                        "timer": lb.timer, "blue_fuel": lb.blue_fuel,
                        "red_fuel": lb.red_fuel,
                    }
                    st.ocr_calls += 1
                    frames_since_ocr = 0
                except Exception:
                    pass
            else:
                frames_since_ocr += 1

            need_dump = dump_dir and want["live"] < (dump_budget or 0)
            if need_dump:
                ann = annotate_live(frame, balls, robots, structs, cached_score)
        else:
            frames_since_ocr += 1
            if label == "info":
                st.info += 1
            else:
                st.filler += 1
            need_dump = dump_dir and want[label] < (dump_budget or 0)
            if need_dump:
                ann = annotate_simple(frame, label)

        st.work_s += time.perf_counter() - tw

        if dump_dir and need_dump:
            p = os.path.join(dump_dir, f"{label}_{want[label]:02d}_f{st.frames:05d}.png")
            cv2.imwrite(p, ann)
            want[label] += 1
            dumped.append(p)

    wall = time.perf_counter() - wall0
    return st, wall, dumped


def project_minutes(throughput_fps: float, sample_fps: float, hours: float) -> float:
    total_frames = hours * 3600.0 * sample_fps
    return (total_frames / throughput_fps) / 60.0


# ------------------------------------------------------------------- main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=VIDEO)
    ap.add_argument("--ss", type=float, default=None)
    ap.add_argument("--t", type=float, default=None)
    ap.add_argument("--fps", type=float, default=SAMPLE_FPS)
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--ocr-every", type=float, default=OCR_EVERY_S,
                    help="seconds of footage between live-score OCR reads")
    ap.add_argument("--benchmark", action="store_true",
                    help="3-min span, throughput + extrapolation, no dump")
    ap.add_argument("--samples", action="store_true",
                    help="dump a few annotated live/info/filler frames")
    ap.add_argument("--dump-dir",
                    default=r"C:\Users\agodavarthy\Claude\projects\frc-vision-scout\frames\dataset\pipeline_annot")
    ap.add_argument("--dump-budget", type=int, default=4)
    args = ap.parse_args()

    ss = args.ss
    t = args.t
    if args.benchmark and t is None:
        ss = ss if ss is not None else 1800.0  # ~30 min in, mid-event = real gameplay
        t = 180.0

    dump = args.dump_dir if args.samples else None
    budget = args.dump_budget if args.samples else None

    print(f"video      = {args.video}")
    print(f"sample_fps = {args.fps}")
    print(f"span       = ss={ss} t={t}")
    print(f"ocr        = {not args.no_ocr} (every {args.ocr_every}s footage)")
    print("running...")

    st, wall, dumped = run(
        video=args.video, sample_fps=args.fps, ss=ss, t=t,
        ocr=not args.no_ocr, ocr_every_s=args.ocr_every,
        dump_dir=dump, dump_budget=budget,
    )

    thr = st.frames / wall if wall > 0 else 0.0
    proj = project_minutes(thr, args.fps, FULL_VIDEO_HOURS)
    needed = (FULL_VIDEO_HOURS * 3600 * args.fps) / (15 * 60)

    print("\n=== RESULTS ===")
    print(f"frames sampled : {st.frames}")
    print(f"  live/info/filler = {st.live}/{st.info}/{st.filler}")
    print(f"  ocr calls        = {st.ocr_calls}")
    print(f"wall seconds   : {wall:.2f}")
    print(f"  decode time      = {st.decode_s:.2f}s ({100*st.decode_s/wall:.0f}%)")
    print(f"  cv/work time     = {st.work_s:.2f}s ({100*st.work_s/wall:.0f}%)")
    print(f"throughput     : {thr:.2f} frames/s end-to-end")
    print(f"need >= {needed:.1f} fps for 15-min/{FULL_VIDEO_HOURS}h target")
    print(f"projected {FULL_VIDEO_HOURS}h @ {args.fps}fps : {proj:.1f} min "
          f"({'MEETS' if proj <= 15 else 'MISSES'} 15-min target)")
    if dumped:
        print("\ndumped:")
        for p in dumped:
            print("  ", p)


if __name__ == "__main__":
    main()
