"""
render_sim.py — Annotated H.264 MP4 rendering for the FRC REBUILT (2026)
playoff match simulation (Upper Bracket R4 M11).

Renders: data/sim/match_full.mp4 -> data/sim/match_sim_overlay.mp4

Annotation layer (per frame of the wide segment):
  - Robot bounding boxes with stable ID labels (ID:alliance_initial)
  - Short motion trails (last 30 positions in a fading colour)
  - Ball markers (yellow circle; cyan if flagged as a 'shot' track)
  - Event flashes: PICKUP (green ring + text), SHOT (magenta arrow),
    SCORE (yellow banner with fuel delta)
  - Live simulated-vs-actual score HUD (bottom bar)
  - Frame / time stamp

Post-wide-segment frames (camera cut, results screen) are passed through
with just a minimal HUD (time stamp + final score).

Interface:
    render(video_path, out_path) -> annotated H.264 mp4
    Can also be invoked from the CLI or imported as a function.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_SRC  = _HERE.parent.parent
_ROOT = _SRC.parent
_DATA = _ROOT / "data" / "sim"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))


# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------
# Per-alliance base colours (BGR)
_BLUE_COL  = (255, 80,   0)    # vivid orange-blue -> use blue channel
_BLUE_COL  = (255, 100,  30)   # bright blue
_RED_COL   = (40,  40,  230)   # vivid red
_BALL_COL  = (0,   230, 230)   # yellow ball -> cyan so it stands out on yellow carpet
_SHOT_COL  = (200, 0,   220)   # magenta shot vector
_PICKUP_COL = (0,  220, 80)    # green pickup ring
_SCORE_COL  = (0,  220, 220)   # yellow score flash
_HUB_BLUE_COL = (255, 60,  0)
_HUB_RED_COL  = (0,   40, 220)

# Trail length in frames (at 30fps = 1s)
_TRAIL_LEN = 30

# Event flash duration (frames)
_FLASH_DUR = 20

# HUD bar height in output frame
_HUD_H = 24

# Font params
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SM   = 0.36
_FONT_MD   = 0.50
_FONT_LG   = 0.65


# ---------------------------------------------------------------------------
# Colour palette for robot IDs (deterministic)
# ---------------------------------------------------------------------------
def _id_color(robot_id: int, alliance: str) -> tuple:
    """Return a distinctive BGR colour per robot ID, tinted by alliance."""
    np.random.seed(robot_id * 7 + 13)
    base = [int(c) for c in np.random.randint(120, 255, 3)]
    if alliance == "blue":
        # Make sure Blue channel dominates
        base[0] = max(base[0], 200)
        base[1] = min(base[1], 120)
        base[2] = min(base[2], 120)
    else:
        # Make sure Red channel dominates
        base[0] = min(base[0], 80)
        base[1] = min(base[1], 80)
        base[2] = max(base[2], 200)
    return tuple(base)


# ---------------------------------------------------------------------------
# Helper: draw bordered text (black outline + coloured fill)
# ---------------------------------------------------------------------------
def _text(img, txt, pt, scale=_FONT_SM, color=(255, 255, 255), thickness=1):
    ox, oy = pt
    cv2.putText(img, txt, (ox + 1, oy + 1), _FONT, scale, (0, 0, 0), thickness + 1,
                cv2.LINE_AA)
    cv2.putText(img, txt, (ox, oy), _FONT, scale, color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------
def render(
    video_path: str | Path,
    out_path: str | Path,
    *,
    robot_tracks_by_frame: dict[int, list] | None = None,
    ball_tracks_by_frame: dict[int, list] | None = None,
    events: list[dict] | None = None,
    sim: dict | None = None,
    overlay_ts: list[dict] | None = None,
    hub_regions: list[dict] | None = None,
    wide_start: int = 0,
    wide_end: int = 4170,
    scale: float = 1.0,     # output scale (1.0 = original 640x360)
) -> Path:
    """Render annotated video.

    If robot/ball track dicts are not supplied, tries to load them from the
    pipeline's saved JSON files in data/sim/. If those are missing, runs the
    full pipeline first.
    """
    video_path = Path(video_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Load pipeline outputs if not provided ---------------------------
    if robot_tracks_by_frame is None or sim is None:
        print("[render] Running pipeline to get tracking data...", file=sys.stderr)
        from sim.pipeline_sim import run_pipeline
        pipe = run_pipeline(video_path)
        robot_tracks_by_frame = pipe["robot_tracks_by_frame"]
        ball_tracks_by_frame  = pipe["ball_tracks_by_frame"]
        events       = pipe["events"]
        sim          = pipe["sim"]
        overlay_ts   = pipe["overlay_ts"]
        hub_regions  = pipe["hub_regions"]
        wide_start   = pipe["wide_start"]
        wide_end     = pipe["wide_end"]

    if ball_tracks_by_frame is None:
        ball_tracks_by_frame = {}
    if events is None:
        events = []
    if overlay_ts is None:
        overlay_ts = []

    # ---- Build overlay oracle lookup: frame -> (blue_score, red_score) ---
    oracle_lookup: dict[int, tuple[int, int]] = {}
    _lb, _lr = 0, 0
    for row in sorted(overlay_ts, key=lambda r: r.get("t", 0)):
        fi_row = int(row.get("frame", 0))
        b = row.get("blue_score")
        r = row.get("red_score")
        if b is not None:
            _lb = int(b)
        if r is not None:
            _lr = int(r)
        oracle_lookup[fi_row] = (_lb, _lr)

    # ---- Build sim timeline lookup: frame -> (blue_score, red_score) -----
    sim_timeline = sim.get("timeline", []) if sim else []
    sim_lookup: dict[int, tuple[int, int]] = {}
    _fps_ref = 30.0
    for entry in sim_timeline:
        fi_entry = int(entry.get("t", 0) * _fps_ref)
        sim_lookup[fi_entry] = (
            int(entry.get("blue_score", 0)),
            int(entry.get("red_score", 0)),
        )

    # ---- Build event flash registry: frame -> list of events -------------
    # Each event causes a flash that lasts _FLASH_DUR frames.
    flash_events: dict[int, list[dict]] = {}
    for ev in events:
        fi_ev = int(ev.get("frame", 0))
        for df in range(_FLASH_DUR):
            fkey = fi_ev + df
            flash_events.setdefault(fkey, []).append(ev)

    # ---- Open video + set up writer -------------------------------------
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_w = max(1, int(round(orig_w * scale)))
    out_h = max(1, int(round(orig_h * scale))) + _HUD_H

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer at {out_path}")

    print(f"[render] Writing {out_path}  ({out_w}x{out_h} @ {fps:.0f}fps "
          f"{total_frames} frames)...", file=sys.stderr)

    # ---- Per-robot trail buffer (ring of recent centroids) --------------
    trails: dict[int, list[tuple[int, int]]] = {}

    # ---- Forward-fill helpers for HUD -----------------------------------
    _cur_oracle = [0, 0]   # [blue, red]
    _cur_sim    = [0, 0]

    # ---- Last seen per-robot dict for coasting trails -------------------
    _last_robots: dict[int, dict] = {}

    # ---- Frame loop -------------------------------------------------------
    fi = 0
    log_every = max(1, total_frames // 20)
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if scale != 1.0:
            frame = cv2.resize(frame, (out_w, orig_h + _HUD_H - _HUD_H),
                               interpolation=cv2.INTER_LINEAR)
        ann = frame.copy()
        h, w = ann.shape[:2]

        in_wide = (wide_start <= fi < wide_end)

        # -- Update oracle + sim scores (forward-fill) --------------------
        if fi in oracle_lookup:
            _cur_oracle[0], _cur_oracle[1] = oracle_lookup[fi]
        # Nearest sim entry
        if fi in sim_lookup:
            _cur_sim[0], _cur_sim[1] = sim_lookup[fi]
        else:
            # linear search for nearest
            best_dt, best_val = 9999, None
            for sfi, sv in sim_lookup.items():
                dt = abs(sfi - fi)
                if dt < best_dt:
                    best_dt = dt
                    best_val = sv
                elif dt > best_dt:
                    break
            if best_val is not None and best_dt <= 90:
                _cur_sim[0], _cur_sim[1] = best_val

        if in_wide:
            # -- Draw HUB regions -------------------------------------------
            if hub_regions:
                for hub in hub_regions:
                    hx, hy, hw_, hh_ = hub["bbox"]
                    col = _HUB_BLUE_COL if hub["alliance"] == "blue" else _HUB_RED_COL
                    cv2.rectangle(ann, (hx, hy), (hx + hw_, hy + hh_), col, 1)

            # -- Get robot + ball tracks for this frame -------------------
            rtrks = robot_tracks_by_frame.get(fi, [])
            btrks = ball_tracks_by_frame.get(fi, [])

            # Update last seen
            for rt in rtrks:
                _last_robots[rt["id"]] = rt

            # -- Draw ball tracks -----------------------------------------
            for bt in btrks:
                bcx, bcy = bt["centroid"]
                br = max(2, bt.get("r", 4))
                is_shot = bt.get("shot", False)
                col = _SHOT_COL if is_shot else _BALL_COL
                cv2.circle(ann, (int(bcx), int(bcy)), br + 1, col, 1, cv2.LINE_AA)
                if is_shot:
                    vx, vy = bt.get("vel", (0, 0))
                    spd = (vx**2 + vy**2) ** 0.5
                    if spd > 0:
                        scale_v = 6.0
                        ex = int(bcx + vx * scale_v)
                        ey = int(bcy + vy * scale_v)
                        cv2.arrowedLine(ann, (int(bcx), int(bcy)), (ex, ey),
                                        _SHOT_COL, 1, tipLength=0.35, line_type=cv2.LINE_AA)

            # -- Draw robot tracks (boxes + labels + trails) ---------------
            for rt in rtrks:
                tid = rt["id"]
                al = rt.get("alliance", "blue")
                col = _id_color(tid, al)

                x, y, bw_, bh_ = rt["bbox"]
                cv2.rectangle(ann, (x, y), (x + bw_, y + bh_), col, 2, cv2.LINE_AA)
                label = f"{tid}:{al[0].upper()}"
                _text(ann, label, (x, max(8, y - 3)), scale=_FONT_SM, color=col)

                # Trail
                cx_t, cy_t = rt["centroid"]
                trails.setdefault(tid, []).append((int(cx_t), int(cy_t)))
                if len(trails[tid]) > _TRAIL_LEN:
                    trails[tid].pop(0)
                pts_trail = trails[tid]
                n_trail = len(pts_trail)
                for k in range(1, n_trail):
                    alpha = k / n_trail
                    tc = tuple(int(c * alpha) for c in col)
                    cv2.line(ann, pts_trail[k - 1], pts_trail[k], tc, 1, cv2.LINE_AA)

            # -- Draw event flashes at this frame -------------------------
            flashes = flash_events.get(fi, [])
            y_flash = 20
            for ev in flashes:
                age = fi - ev.get("frame", fi)    # frames since event
                if age > _FLASH_DUR:
                    continue
                fade = 1.0 - age / _FLASH_DUR
                ev_type = ev.get("type", "")

                if ev_type == "pickup":
                    detail = ev.get("detail", {})
                    bp = detail.get("ball_pos")
                    if bp:
                        bpx, bpy = int(bp[0]), int(bp[1])
                        r_ring = int(10 + age * 0.7)
                        alpha_c = max(0, int(220 * fade))
                        cv2.circle(ann, (bpx, bpy), r_ring,
                                   (_PICKUP_COL[0], _PICKUP_COL[1], alpha_c), 2, cv2.LINE_AA)
                        if age == 0:
                            _text(ann, f"PICK r{ev.get('robot_id', '?')}",
                                  (bpx - 8, bpy - 12), color=_PICKUP_COL)

                elif ev_type == "shot":
                    detail = ev.get("detail", {})
                    bp = detail.get("ball_pos")
                    if bp and age < 8:
                        bpx, bpy = int(bp[0]), int(bp[1])
                        vx_, vy_ = detail.get("vel", (0, 0))
                        scale_v2 = 8.0
                        ex2 = int(bpx + vx_ * scale_v2)
                        ey2 = int(bpy + vy_ * scale_v2)
                        cv2.arrowedLine(ann, (bpx, bpy), (ex2, ey2),
                                        _SHOT_COL, 2, tipLength=0.3, line_type=cv2.LINE_AA)
                        _text(ann, f"SHOT r{ev.get('robot_id','?')}→{detail.get('target_hub','?')}",
                              (bpx - 8, bpy - 14), color=_SHOT_COL)

                elif ev_type == "score":
                    detail = ev.get("detail", {})
                    delta = detail.get("fuel_delta", detail.get("pts", 0))
                    al2 = ev.get("alliance", "?")
                    score_str = f"SCORE +{delta} {al2.upper()}"
                    col2 = _BLUE_COL if al2 == "blue" else _RED_COL
                    _text(ann, score_str, (5, y_flash), scale=_FONT_MD, color=col2)
                    y_flash += 18

        # -- HUD bar at bottom -------------------------------------------
        hud = np.zeros((_HUD_H, out_w, 3), dtype=np.uint8)
        hud[:] = (30, 30, 30)

        # Sim scores
        sim_b_str = str(_cur_sim[0])
        sim_r_str = str(_cur_sim[1])
        # Oracle scores
        ora_b_str = str(_cur_oracle[0])
        ora_r_str = str(_cur_oracle[1])

        ts_s = fi / fps
        time_str = f"{int(ts_s // 60):02d}:{int(ts_s % 60):02d} f{fi:05d}"

        # Blue: left side
        _text(hud, f"SIM B:{sim_b_str}", (4, _HUD_H - 5), scale=_FONT_SM, color=_BLUE_COL)
        _text(hud, f"OCR B:{ora_b_str}", (4 + 80, _HUD_H - 5), scale=_FONT_SM, color=(180, 180, 255))
        # Timer center
        _text(hud, time_str, (out_w // 2 - 40, _HUD_H - 5), scale=_FONT_SM, color=(220, 220, 220))
        # Red: right side
        _text(hud, f"SIM R:{sim_r_str}", (out_w - 155, _HUD_H - 5), scale=_FONT_SM, color=_RED_COL)
        _text(hud, f"OCR R:{ora_r_str}", (out_w - 75, _HUD_H - 5), scale=_FONT_SM, color=(180, 100, 100))

        # Combine
        combined = np.zeros((orig_h + _HUD_H, out_w, 3), dtype=np.uint8)
        combined[:orig_h, :out_w] = ann
        combined[orig_h:, :out_w] = hud

        writer.write(combined)

        if fi % log_every == 0:
            pct = 100.0 * fi / max(1, total_frames)
            print(f"  [render] f{fi}/{total_frames} {pct:.0f}%"
                  f"  t={ts_s:.1f}s", file=sys.stderr)

        fi += 1

    cap.release()
    writer.release()
    print(f"[render] Done -> {out_path}", file=sys.stderr)
    return out_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else str(_DATA / "match_full.mp4")
    out   = sys.argv[2] if len(sys.argv) > 2 else str(_DATA / "match_sim_overlay.mp4")

    render(video, out)
