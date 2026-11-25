#!/usr/bin/env python3
"""Score/timer OCR accuracy + frames/s on a labelled synthetic live-bar set.

No broadcast footage ships in the repo, so frames are drawn with cv2 in the ROI layout
of detect/overlay_ocr.py (640x360, blue banner left, red banner right). Fonts: Hershey
SIMPLEX/DUPLEX are in the reader's templates, TRIPLEX is held out. Degradations: gaussian
noise, blur, brightness, JPEG. Usage: bench_overlay_ocr.py [N] [--out results/overlay_ocr.json]
"""
import json, platform, random, sys, time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from frcscout.detect import digit_ocr
from frcscout.detect.overlay_ocr import _LIVE_BAR_ROIS as R, OverlayReader

FONTS = {"simplex": cv2.FONT_HERSHEY_SIMPLEX, "duplex": cv2.FONT_HERSHEY_DUPLEX,
         "triplex": cv2.FONT_HERSHEY_TRIPLEX}


def make_frame(rng, level):
    W, H = 640, 360
    f = np.full((H, W, 3), (40, 90, 60), np.uint8)  # field-ish background
    cv2.rectangle(f, (0, 0), (W, 29), (35, 35, 35), -1)
    cv2.rectangle(f, (0, 0), (int(.15 * W), 29), (200, 90, 20), -1)   # blue (BGR)
    cv2.rectangle(f, (int(.85 * W), 0), (W, 29), (30, 30, 200), -1)    # red
    font = rng.choice(list(FONTS))
    score = rng.randint(0, 299)
    mm, ss = rng.randint(0, 2), rng.randint(0, 59)
    timer = f"{mm}:{ss:02d}" if rng.random() < .8 else f"{ss}"
    gt_t = mm * 60 + ss if ":" in timer else ss

    def put(txt, roi, scale):
        x0, y0 = int(roi[0] * W), int(roi[1] * H)
        w = int((roi[2] - roi[0]) * W)
        (tw, th), _ = cv2.getTextSize(txt, FONTS[font], scale, 2)
        cv2.putText(f, txt, (x0 + (w - tw) // 2, 24), FONTS[font], scale, (245, 245, 245), 2, cv2.LINE_AA)
    put(str(score), R["blue_score"], .8)
    put(timer, R["timer"], .7)
    f = f.astype(np.float32)
    f = f * rng.uniform(.85, 1.1) + rng.uniform(-10, 10)
    f += np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, 4 + 5 * level, f.shape)
    f = np.clip(f, 0, 255).astype(np.uint8)
    if level:
        f = cv2.GaussianBlur(f, (3, 3), 0.8)
    q = rng.randint(35, 60) if level else rng.randint(70, 95)
    f = cv2.imdecode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 1)
    return f, score, float(gt_t), font


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = int(args[0]) if args else 100
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    rng = random.Random(0)
    set_ = [make_frame(rng, lvl) for lvl in (0, 1) for _ in range(n // 2)]
    t0 = time.perf_counter()
    preds = []
    reader = OverlayReader({"ocr": "template"})  # the product's reader, template mode
    for i, (f, *_) in enumerate(set_):
        st = reader.read(f, i)
        preds.append((st.blue_score, st.timer_s))
    dt = time.perf_counter() - t0
    res = {"frames": len(set_), "fps": round(len(set_) / dt, 1)}
    for name, sel in [("all", lambda i, g: True), ("clean", lambda i, g: i < n // 2),
                      ("degraded", lambda i, g: i >= n // 2), ("held_out_font", lambda i, g: g[3] == "triplex")]:
        ix = [i for i, g in enumerate(set_) if sel(i, g)]
        res[name] = {"n": len(ix),
                     "score_exact": round(sum(preds[i][0] == set_[i][1] for i in ix) / len(ix), 4),
                     "timer_exact": round(sum(preds[i][1] == set_[i][2] for i in ix) / len(ix), 4)}
    res["machine"] = f"{platform.platform()} {platform.processor()} cv2 {cv2.__version__}"
    print(json.dumps(res, indent=1))
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
