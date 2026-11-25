"""
render_overlay.py — render an annotated overlay video from a clip.

For each frame: classify (GREEN=live / YELLOW=info / RED=filler whole-frame tint), and on
LIVE FIELD frames draw a circle on every FUEL ball, a box on every ROBOT (blue/red), a box
on every HUB, plus a live score line. Output is H.264 mp4 (browser-playable).

Adds a FIELD-VIEW GATE on top of the frame classifier: 'live' only means the score bar is
present, which also covers crowd/pit cutaways during a match. The gate additionally requires
grey-carpet evidence in the field band before running the object detectors, so cutaways keep
their green tint but don't get spurious boxes.

Score OCR (easyocr) is slow, so it is run once per ~1s of footage and cached onto intervening
frames. Everything else is fast classical CV.

Usage: python src/render_overlay.py <in_clip.mp4> <out.mp4> [--fps 15]
"""

from __future__ import annotations

import subprocess
import sys

import cv2
import numpy as np

import frame_classify
import detect_fuel
import detect_robots
import detect_structures
import overlay_live

TINT = {"live": (0, 190, 0), "info": (0, 195, 255), "filler": (0, 0, 215)}  # BGR

# Field-view gate: grey carpet sits at low saturation, mid value. Crowd/pit shots are
# higher-saturation / more textured, so the carpet fraction in the field band drops.
FIELD_BAND = (0.0, 0.20, 1.0, 0.60)   # upper field where carpet dominates on a real field view
CARPET_MIN_FRAC = 0.16


def _carpet_frac(frame) -> float:
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = FIELD_BAND
    band = frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    return float(((s < 55) & (v > 70) & (v < 200)).mean())


def is_field_view(frame) -> bool:
    return _carpet_frac(frame) >= CARPET_MIN_FRAC


def annotate(frame, score_cache):
    out = cv2.addWeighted(frame, 0.80, np.full_like(frame, (0, 0, 0)), 0.0, 0)
    cls = frame_classify.classify(frame)
    label = cls.label
    tint = np.full_like(frame, TINT.get(label, (0, 0, 0)))
    out = cv2.addWeighted(frame, 0.82, tint, 0.18, 0)

    nb = nr = nh = 0
    field = label == "live" and is_field_view(frame)
    if field:
        for (x, y, r) in detect_fuel.detect(frame)[0]:
            cv2.circle(out, (x, y), max(2, r), (0, 255, 255), 1)
            nb += 1
        for d in detect_robots.detect(frame)[0]:
            x, y, w, h = d["bbox"]
            col = (255, 90, 0) if d["alliance"] == "blue" else (0, 0, 255)
            cv2.rectangle(out, (x, y), (x + w, y + h), col, 2)
            nr += 1
        for d in detect_structures.detect(frame)[0]:
            x, y, w, h = d["bbox"]
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 165, 255), 2)
            nh += 1

    # top-left status chip
    chip = {"live": "LIVE", "info": "INFO", "filler": "FILLER"}[label]
    if label == "live" and not field:
        chip = "LIVE (cutaway)"
    cv2.rectangle(out, (0, 0), (frame.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(out, f"{chip}   FUEL:{nb}  ROBOTS:{nr}  HUBS:{nh}", (6, 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # bottom score line (cached OCR)
    sc = score_cache.get("val")
    if sc:
        txt = f"SCORE  BLUE {sc.get('blue_score')}   RED {sc.get('red_score')}   {sc.get('timer') or ''}"
        cv2.rectangle(out, (0, frame.shape[0] - 16), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.putText(out, txt, (6, frame.shape[0] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return out


def render(in_path, out_path, out_fps=15):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise FileNotFoundError(in_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, int(round(src_fps / out_fps)))

    # pipe annotated BGR frames to ffmpeg -> browser-playable H.264
    ff = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(out_fps),
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", out_path],
        stdin=subprocess.PIPE)

    score_cache, idx, written = {"val": None}, 0, 0
    ocr_every = step * out_fps  # ~once per second of footage
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok2, frame = cap.retrieve()
            if not ok2:
                break
            if written % out_fps == 0:  # refresh score ~1x/sec
                try:
                    bar = overlay_live.read_bar(frame)
                    if bar and (bar.get("blue_score") is not None or bar.get("red_score") is not None):
                        score_cache["val"] = bar
                except Exception:
                    pass
            ff.stdin.write(annotate(frame, score_cache).tobytes())
            written += 1
        idx += 1
    cap.release()
    ff.stdin.close()
    ff.wait()
    return written


if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else "notebook/playoff_clip.mp4"
    outp = sys.argv[2] if len(sys.argv) > 2 else "notebook/final_overlay.mp4"
    fps = 15
    if "--fps" in sys.argv:
        fps = int(sys.argv[sys.argv.index("--fps") + 1])
    n = render(inp, outp, fps)
    print(f"rendered {n} annotated frames @ {fps}fps -> {outp}")
