"""
render_final.py — fast annotated overlay using the REFINED engines, score from the oracle.

Annotates a time window of the match video: frame-classify tint (green/yellow/red) + refined
FUEL circles + ROBOT boxes (carpet-restricted) + HUB boxes + a score line read from the
pre-computed overlay oracle (data/sim/overlay_ts.json) instead of slow per-frame OCR. Output
is browser-playable H.264.

Usage: python src/render_final.py <in_video> <out.mp4> <t0> <t1> [--fps 15]
"""
from __future__ import annotations
import json, subprocess, sys
import cv2, numpy as np

import detect_fuel
import frame_classify
try:
    import detect_robots
except Exception:
    detect_robots = None
try:
    import detect_structures
except Exception:
    detect_structures = None

TINT = {"live": (0, 190, 0), "info": (0, 195, 255), "filler": (0, 0, 215)}


def _load_oracle(path="data/sim/overlay_ts.json"):
    try:
        rows = json.load(open(path))
        return {int(r["t"]): r for r in rows}
    except Exception:
        return {}


def _in_carpet(cx, cy, w, h):
    poly = np.array([(int(x * w), int(y * h)) for x, y in detect_fuel.FIELD_POLY_NORM], np.int32)
    return cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0


def render(vid, out, t0, t1, out_fps=15):
    oracle = _load_oracle()
    cap = cv2.VideoCapture(vid)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(3)); h = int(cap.get(4))
    f0, f1 = int(t0 * src_fps), int(t1 * src_fps)
    step = max(1, int(round(src_fps / out_fps)))
    poly = np.array([(int(x * w), int(y * h)) for x, y in detect_fuel.FIELD_POLY_NORM], np.int32)

    ff = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(out_fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", out], stdin=subprocess.PIPE)

    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    idx, written = f0, 0
    while idx < f1:
        ok = cap.grab()
        if not ok:
            break
        if (idx - f0) % step == 0:
            ok2, frame = cap.retrieve()
            if not ok2:
                break
            label = frame_classify.classify(frame).label
            out_img = cv2.addWeighted(frame, 0.82, np.full_like(frame, TINT.get(label, (0, 0, 0))), 0.18, 0)
            nb = nr = nh = 0
            if label == "live":
                cv2.polylines(out_img, [poly], True, (255, 255, 255), 1)
                for (x, y, r) in detect_fuel.detect(frame)[0]:
                    cv2.circle(out_img, (x, y), max(2, r), (0, 255, 255), 1); nb += 1
                if detect_structures:
                    for d in detect_structures.detect(frame)[0]:
                        x, y, ww, hh = d["bbox"]; cv2.rectangle(out_img, (x, y), (x + ww, y + hh), (0, 165, 255), 2); nh += 1
                if detect_robots:
                    for d in detect_robots.detect(frame)[0]:
                        x, y, ww, hh = d["bbox"]; cx, cy = x + ww // 2, y + hh // 2
                        if not _in_carpet(cx, cy, w, h):
                            continue
                        col = (255, 90, 0) if d["alliance"] == "blue" else (0, 0, 255)
                        cv2.rectangle(out_img, (x, y), (x + ww, y + hh), col, 2); nr += 1
            cv2.rectangle(out_img, (0, 0), (w, 17), (0, 0, 0), -1)
            cv2.putText(out_img, f"{label.upper()}   FUEL:{nb}  ROBOTS:{nr}  HUBS:{nh}", (5, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)
            tt = int(idx / src_fps)
            sc = oracle.get(tt) or oracle.get(tt - 1) or oracle.get(tt + 1)
            if sc:
                cv2.rectangle(out_img, (0, h - 16), (w, h), (0, 0, 0), -1)
                cv2.putText(out_img, f"SCORE  BLUE {sc.get('blue_score')}   RED {sc.get('red_score')}   t={sc.get('timer','')}",
                            (5, h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            ff.stdin.write(out_img.tobytes()); written += 1
        idx += 1
    cap.release(); ff.stdin.close(); ff.wait()
    return written


if __name__ == "__main__":
    vid = sys.argv[1]; out = sys.argv[2]; t0 = float(sys.argv[3]); t1 = float(sys.argv[4])
    fps = int(sys.argv[sys.argv.index("--fps") + 1]) if "--fps" in sys.argv else 15
    n = render(vid, out, t0, t1, fps)
    print(f"rendered {n} frames @ {fps}fps -> {out}")
