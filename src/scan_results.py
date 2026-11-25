"""
scan_results.py — find every post-match results screen in a video and extract its data.

Two-stage, OCR only where it pays:
  1. Walk the video at SAMPLE_FPS, run the cheap color/brightness is_results_screen() on
     each sampled frame. Group consecutive hits into CLUSTERS (each match's results graphic
     is shown ~10-15s, so one cluster ≈ one match).
  2. For each cluster, OCR-extract the sharpest middle frame with ResultsExtractor.

Output: data/<stem>_results.json — a list of detected matches with teams + scores, keyed by
the OCR'd match label. This is what verify_tba.py checks against the TBA oracle.

Deterministic + offline: detection is OpenCV, digit reading is local easyocr. No API key.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import cv2

from results_extractor import is_results_screen, ResultsExtractor

SAMPLE_FPS = 0.5          # classify one frame every 2s
MIN_CLUSTER_S = 4         # ignore results-flashes shorter than this
BRIDGE_GAP_S = 6          # merge hits separated by <= this (camera cuts during the graphic)


def _clusters(hits: list[float], dt: float) -> list[tuple[float, float]]:
    """Group sorted hit-timestamps into [start, end] clusters, bridging small gaps."""
    if not hits:
        return []
    out = []
    s = p = hits[0]
    for t in hits[1:]:
        if t - p <= BRIDGE_GAP_S:
            p = t
        else:
            out.append((s, p)); s = p = t
    out.append((s, p))
    return [(a, b) for a, b in out if b - a >= MIN_CLUSTER_S or len(hits) == 1]


def scan(video_path: str, sample_fps: float = SAMPLE_FPS, verbose: bool = True) -> list[dict]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / sample_fps)))

    # stage 1: cheap classification over the whole video
    hits, pos = [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if pos % step == 0:
            ok2, frame = cap.retrieve()
            if not ok2:
                break
            if is_results_screen(frame):
                hits.append(pos / fps)
        pos += 1
    clusters = _clusters(hits, 1.0 / sample_fps)
    if verbose:
        print(f"stage1: {len(hits)} results-frames -> {len(clusters)} clusters")

    # stage 2: OCR the middle frame of each cluster
    ex = ResultsExtractor()
    results = []
    for i, (a, b) in enumerate(clusters):
        mid = (a + b) / 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(mid * fps))
        ok, frame = cap.read()
        if not ok:
            continue
        r = ex.extract(frame)
        rec = asdict(r)
        rec["t"] = round(mid, 1)
        rec["cluster"] = [round(a, 1), round(b, 1)]
        results.append(rec)
        if verbose:
            print(f"  {i+1:3d} t={mid:7.1f}  {r.match_label or '?':18} "
                  f"blue={r.blue_teams} red={r.red_teams} {r.blue_score}-{r.red_score}")
    cap.release()
    return results


if __name__ == "__main__":
    import sys
    vid = sys.argv[1] if len(sys.argv) > 1 else "data/ngvsHrmoW8U_360p.mp4"
    res = scan(vid)
    out = Path(vid).resolve().parent / f"{Path(vid).stem}_results.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nwrote {len(res)} detected matches -> {out}")
