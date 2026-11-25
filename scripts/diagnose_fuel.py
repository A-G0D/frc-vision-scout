"""
Diagnostic script to understand frame geometry and yellow pixel distribution.
Saves debug annotated frames to understand where detections are happening.
"""
import cv2
import numpy as np
import os
import sys

sys.path.insert(0, "src")

# ---- load a representative frame
paths = [
    "frames/dataset/live_00_qm52_b100.png",
    "frames/dataset/live_01_qm53_b70.png",
    "frames/dataset/live_04_qm55_b70.png",
    "frames/dataset/live_09_qm59_b100.png",
]

OUT = "frames/dataset/fuel_annot/diag"
os.makedirs(OUT, exist_ok=True)

YELLOW_LO = np.array([18,  90, 100], dtype=np.uint8)
YELLOW_HI = np.array([40, 255, 255], dtype=np.uint8)

for path in paths:
    frame = cv2.imread(path)
    if frame is None:
        continue
    h, w = frame.shape[:2]
    name = os.path.basename(path).replace(".png", "")

    # 1. Show yellow mask across entire frame (no ROI)
    blur = cv2.GaussianBlur(frame, (3, 3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    raw = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)

    # Save yellow mask
    cv2.imwrite(f"{OUT}/{name}_ymask.png", raw)

    # 2. Overlay mask on original with row sums plotted
    ann = frame.copy()
    # Draw mask pixels as green overlay
    ann[raw > 0] = [0, 255, 0]

    # Draw ROI lines
    y0_cur = int(0.21 * h)
    y1_cur = int(0.91 * h)
    cv2.line(ann, (0, y0_cur), (w, y0_cur), (0, 255, 255), 2)
    cv2.line(ann, (0, y1_cur), (w, y1_cur), (0, 255, 255), 2)

    # Also show 50% height line
    cv2.line(ann, (0, h//2), (w, h//2), (255, 0, 255), 1)

    # Count yellow pixels per row
    row_sums = raw.sum(axis=1) / 255
    for y_idx in range(0, h, 2):
        bar_len = int(row_sums[y_idx] * 2)
        if bar_len > 0:
            cv2.line(ann, (0, y_idx), (min(bar_len, w-1), y_idx), (0, 128, 255), 1)

    cv2.putText(ann, f"h={h} w={w}", (8, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.imwrite(f"{OUT}/{name}_diag.png", ann)
    print(f"{name}: h={h} w={w}, yellow pixels in top 50%: {raw[:h//2].sum()//255}, bottom 50%: {raw[h//2:].sum()//255}")

    # 3. Per-horizontal-strip yellow count
    nstrips = 20
    strip_h = h // nstrips
    for s in range(nstrips):
        y_s = s * strip_h
        y_e = y_s + strip_h
        cnt = raw[y_s:y_e].sum() // 255
        frac = y_s / h
        print(f"  strip {s:2d} ({frac:.2f}-{y_e/h:.2f}): {cnt:6d} yellow px")

print("Done.")
