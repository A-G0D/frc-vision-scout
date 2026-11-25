"""
scan_timeline.py — find every match in a multi-hour FRC stream and timestamp it.

Pipeline (all local, after one low-res download):
  1. Sample the video at SAMPLE_FPS (default every 2s) -> frames.
  2. Run overlay_detect on each frame -> boolean "match overlay present".
  3. Smooth the boolean track and merge into match SEGMENTS [t_start, t_end].
     (a match is a run of overlay-present samples lasting >= MIN_MATCH_S, with small
      gaps bridged -- replays/camera cuts briefly drop the overlay.)
  4. For each segment, pull a clean mid-match frame and crop the OCR ROIs
     (blue/red team panels, match label, fuel boxes) for the extraction stage.
  5. Optionally align the ordered segments to the ground-truth match list
     (match_list.py) so each segment gets labelled Qual N with its 6 team numbers.

The team-number OCR itself is pluggable (see ocr_backend); it is intentionally the LAST
stage because segmentation + alignment already recovers the match schedule structure, and
alignment to TBA gives the teams even if OCR is weak at low resolution.

Output: data/<id>_segments.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import cv2

from overlay_detect import detect_overlay, crop_norm, OVERLAY_ROIS

SAMPLE_FPS = 0.5        # sample one frame every 2 seconds
MIN_MATCH_S = 30        # ignore overlay blips shorter than this
BRIDGE_GAP_S = 12       # merge match runs separated by <= this many seconds


@dataclass
class Segment:
    index: int
    t_start: float
    t_end: float
    n_samples: int
    mean_score: float
    # filled by later stages:
    match_label: str | None = None
    red: list[int] = field(default_factory=list)
    blue: list[int] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    def hms(self, t: float) -> str:
        h, rem = divmod(int(t), 3600)
        m, s = divmod(rem, 60)
        return f"{h:d}:{m:02d}:{s:02d}"


def scan(video_path: str, sample_fps: float = SAMPLE_FPS,
         thresh: float = 0.12) -> list[Segment]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps / sample_fps)))

    track: list[tuple[float, bool, float]] = []  # (t, is_match, score)
    # Sequential read with grab()-skip is far faster than seeking per sample on long files.
    pos = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if pos % step == 0:
            ok2, frame = cap.retrieve()
            if not ok2:
                break
            v = detect_overlay(frame, thresh)
            track.append((pos / fps, v.is_match, v.score))
        pos += 1
    cap.release()
    return _segments_from_track(track, sample_fps)


def _segments_from_track(track, sample_fps) -> list[Segment]:
    dt = 1.0 / sample_fps
    bridge = int(round(BRIDGE_GAP_S / dt))
    flags = [m for _, m, _ in track]

    # bridge short gaps of False between True runs
    i = 0
    n = len(flags)
    while i < n:
        if not flags[i]:
            j = i
            while j < n and not flags[j]:
                j += 1
            left = i > 0 and flags[i - 1]
            right = j < n and j - i <= bridge
            if left and right:
                for k in range(i, j):
                    flags[k] = True
            i = j
        else:
            i += 1

    segs: list[Segment] = []
    i = 0
    while i < n:
        if flags[i]:
            j = i
            while j < n and flags[j]:
                j += 1
            t0, t1 = track[i][0], track[j - 1][0]
            if t1 - t0 >= MIN_MATCH_S:
                scores = [track[k][2] for k in range(i, j) if track[k][1]]
                segs.append(Segment(
                    index=len(segs) + 1, t_start=round(t0, 1), t_end=round(t1, 1),
                    n_samples=j - i,
                    mean_score=round(sum(scores) / max(1, len(scores)), 3),
                ))
            i = j
        else:
            i += 1
    return segs


def dump_roi_crops(video_path: str, segs: list[Segment], out_dir: str) -> None:
    """Save the OCR ROIs from a mid-match frame of each segment for inspection/OCR."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    for s in segs:
        mid = (s.t_start + s.t_end) / 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(mid * fps))
        ok, frame = cap.read()
        if not ok:
            continue
        for name in ("blue_panel", "red_panel", "match_label", "blue_fuel", "red_fuel"):
            crop = crop_norm(frame, OVERLAY_ROIS[name])
            crop = cv2.resize(crop, None, fx=6, fy=6, interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(str(out / f"seg{s.index:03d}_{name}.png"), crop)
    cap.release()


def align_to_schedule(segs: list[Segment], matches) -> None:
    """
    Label segments by order against the ground-truth schedule. Streams play matches in
    schedule order, so the k-th detected match segment == the k-th scheduled match.
    OCR (when available) should be used to verify/repair this alignment, not replace it.
    """
    for seg, m in zip(segs, matches):
        seg.match_label = m.label
        seg.red, seg.blue = m.red, m.blue


if __name__ == "__main__":
    import sys
    vid = sys.argv[1] if len(sys.argv) > 1 else "data/ngvsHrmoW8U_144p.mp4"
    segs = scan(vid)
    print(f"detected {len(segs)} match segments")
    for s in segs:
        print(f"  seg {s.index:3d}  {s.hms(s.t_start)} - {s.hms(s.t_end)} "
              f"({int(s.duration)}s, score={s.mean_score})")
    # write next to the video file so cwd doesn't matter
    out = Path(vid).resolve().parent / f"{Path(vid).stem}_segments.json"
    json.dump([asdict(s) for s in segs], open(out, "w"), indent=2)
    print("wrote", out)
