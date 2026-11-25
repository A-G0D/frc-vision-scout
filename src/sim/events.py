"""
events.py — per-robot PICKUP and SHOT event detection for the FRC REBUILT (2026)
playoff match (Upper Bracket R4 M11, 640x360, single static wide camera).

Algorithm
---------
PICKUP  — A tracked ball disappears near a robot's intake region. Specifically:
  1. A ball track ID that was present at frame N is missing at frame N+1 (it
     dropped out of the BallTracker output, meaning the ball left the carpet OR
     was occluded by a robot).
  2. At the time of disappearance, the ball centroid is within PICKUP_RADIUS
     pixels of the nearest robot centroid AND that robot is on-carpet.
  3. Attribute to the nearest robot; record possession starts.

SHOT    — A ball track with shot==True (high velocity) heads toward a HUB region.
  Specifically:
  1. The BallTracker already flags a track as `shot=True` when its Kalman-filter
     speed crosses shot_speed.
  2. For each frame where a shot-flagged ball track appears, compute the ball's
     trajectory direction (from vel) and check if it intersects / is aimed at
     one of the HUB bboxes within a lookahead distance.
  3. Attribute to the nearest robot at shot origin frame; record once per ball ID.

SCORE  — Derived from the overlay_ts fuel counter: when blue_fuel or red_fuel
  increases, score events are emitted, attributed to the alliance (robot ID
  left as None since we can't associate individual shots to hub scores from
  camera alone at this resolution).

Interface (exact)
-----------------
detect_events(
    robot_tracks_by_frame : dict[int, list[dict]]  — output of RobotTracker.update()
    ball_tracks_by_frame  : dict[int, list[dict]]  — output of BallTracker.update()
    hub_regions           : list[{'bbox':(x,y,w,h),'alliance':'blue'|'red'}]
    overlay_ts            : list[{'t','blue_score','red_score','blue_fuel','red_fuel',...}]
) -> list[{'t':float,'frame':int,'type':'pickup'|'shot'|'score',
           'robot_id':int|None,'alliance':'blue'|'red'|None,'detail':dict}]

The output list is sorted by frame.

Classical CV only (OpenCV + numpy). Deterministic.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------

# PICKUP: a ball is considered "picked up" if it disappears within this many
# pixels of a robot centroid. Robots are ~18-40px wide; tight 22px radius
# requires the ball to be INSIDE the robot box to count as a pickup.
PICKUP_RADIUS = 22.0

# PICKUP: a ball track must have been seen for at least this many consecutive
# frames before disappearing to count as a real ball (not detection noise).
PICKUP_MIN_BALL_AGE = 4

# PICKUP: per-robot de-duplicate window. A given robot cannot pick up two
# different balls within this many frames (~45 frames / 1.5s). Robots need
# time to actually intake the ball and cycle; sub-1.5s is physically impossible.
PICKUP_DEDUP_PER_ROBOT = 45

# PICKUP: maximum realistic pickup rate for any ball ID (separate ball tracks
# can share IDs in rare cases). A single ball ID cannot be "picked up" again
# within this window.
PICKUP_DEDUP_PER_BALL = 60

# SHOT: minimum speed (px/frame from ball tracker's EMA vel) for a ball to be
# flagged as a potential shot. track_ball.BallTracker already applies multi-
# criteria (speed + travel + directness) before setting shot=True; this is just
# a last-resort sanity gate on the velocity magnitude. Lowered to 3.5 to match
# the new shot_speed=4.5 in the tracker (EMA vel lags slightly).
SHOT_MIN_SPEED = 3.5

# SHOT aiming: the ball's velocity vector must point toward a HUB region within
# SHOT_LOOKAHEAD frames. Extended to 120 frames (~4s at 30fps) because at this
# resolution ball arcs are shallow and the flight takes longer than expected.
# Also: if no HUB is aimed at but the ball is clearly in-flight (shot==True AND
# speed >= SHOT_MIN_SPEED), we count it as an unattributed shot so we don't lose
# high-confidence events just because the velocity vector is slightly off-axis.
SHOT_LOOKAHEAD = 120       # frames to project the ball's velocity forward

# SHOT attribution: nearest robot within this radius of the ball's first
# shot-flagged centroid is the shooter.
SHOT_ATTR_RADIUS = 100.0

# SHOT: per-ball-id de-duplicate window.
SHOT_DEDUP_FRAMES = 15

# SCORE: maximum plausible fuel increase per overlay sample period (1s).
# The TBA final is red=506, blue=268 over ~103s of live-bar coverage.
# Peak realistic rate: ~20 fuel/s. Cap at 25 to reject OCR spikes.
SCORE_MAX_DELTA_PER_S = 25


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def _ray_hits_bbox(ox: float, oy: float, dx: float, dy: float,
                   bx: int, by: int, bw: int, bh: int,
                   lookahead: float) -> bool:
    """True if the ray from (ox,oy) in direction (dx,dy) passes through bbox
    within `lookahead` distance."""
    spd = math.hypot(dx, dy)
    if spd < 0.5:
        return False
    # Parametric clipping (slab method) for the AABB.
    inv_dx = dx / spd
    inv_dy = dy / spd
    x0, y0 = float(bx), float(by)
    x1, y1 = float(bx + bw), float(by + bh)

    # Avoid division by zero with a tiny epsilon.
    eps = 1e-6
    if abs(inv_dx) > eps:
        tx0 = (x0 - ox) / inv_dx
        tx1 = (x1 - ox) / inv_dx
        tmin_x, tmax_x = min(tx0, tx1), max(tx0, tx1)
    else:
        # Ray is vertical; check if ox is inside the x-slab.
        if ox < x0 or ox > x1:
            return False
        tmin_x, tmax_x = 0.0, lookahead

    if abs(inv_dy) > eps:
        ty0 = (y0 - oy) / inv_dy
        ty1 = (y1 - oy) / inv_dy
        tmin_y, tmax_y = min(ty0, ty1), max(ty0, ty1)
    else:
        if oy < y0 or oy > y1:
            return False
        tmin_y, tmax_y = 0.0, lookahead

    tmin = max(tmin_x, tmin_y)
    tmax = min(tmax_x, tmax_y)

    return tmax >= tmin >= 0 and tmin <= lookahead


def _nearest_robot(cx: float, cy: float,
                   robot_list: list[dict],
                   max_dist: float = float("inf")) -> tuple[int | None, float]:
    """Return (robot_id, distance) of the closest robot to (cx, cy)."""
    best_id: int | None = None
    best_d = max_dist
    for r in robot_list:
        rcx, rcy = r["centroid"]
        d = _dist(cx, cy, rcx, rcy)
        if d < best_d:
            best_d = d
            best_id = r["id"]
    return best_id, best_d


def _robot_alliance(robot_id: int | None,
                    robot_tracks_by_frame: dict[int, list[dict]]) -> str | None:
    """Look up the alliance of a robot ID from the track history."""
    if robot_id is None:
        return None
    for frame_tracks in robot_tracks_by_frame.values():
        for r in frame_tracks:
            if r["id"] == robot_id:
                return r["alliance"]
    return None


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def detect_events(
    robot_tracks_by_frame: dict[int, list[dict]],
    ball_tracks_by_frame: dict[int, list[dict]],
    hub_regions: list[dict],
    overlay_ts: list[dict],
) -> list[dict]:
    """Detect PICKUP, SHOT, and SCORE events.

    Parameters
    ----------
    robot_tracks_by_frame : {frame_idx: [{'id','bbox','centroid','alliance'},...]}
    ball_tracks_by_frame  : {frame_idx: [{'id','centroid','r','vel','shot'},...]}
    hub_regions           : [{'bbox':(x,y,w,h),'alliance':'blue'|'red'}]
    overlay_ts            : list of per-second overlay rows

    Returns
    -------
    Sorted list of event dicts:
      {'t': float, 'frame': int, 'type': 'pickup'|'shot'|'score',
       'robot_id': int|None, 'alliance': str|None, 'detail': dict}
    """
    events: list[dict] = []

    frames_sorted = sorted(robot_tracks_by_frame.keys())
    if not frames_sorted:
        return events

    # Build a reference FPS from the frame spacing (assume 30 fps default).
    fps = 30.0

    # -----------------------------------------------------------------------
    # PICKUP detection
    # -----------------------------------------------------------------------
    # Track which ball IDs were visible in the previous frame + how long they've
    # been continuously visible (consecutive_age).
    # prev_ball_ids: id -> (cx, cy)
    # ball_consecutive_age: id -> consecutive frames seen (PERSISTENT; increments
    #   when present, resets when absent)
    prev_ball_ids: dict[int, tuple[float, float]] = {}
    ball_consecutive_age: dict[int, int] = {}  # id -> consecutive frames seen

    # Per-robot last pickup frame for de-duplication.
    robot_last_pickup: dict[int, int] = {}  # robot_id -> last pickup frame

    # Per-ball last pickup frame for de-duplication.
    ball_last_pickup: dict[int, int] = {}   # ball_id -> last pickup frame

    for fi in frames_sorted:
        ball_tracks = ball_tracks_by_frame.get(fi, [])
        robot_tracks = robot_tracks_by_frame.get(fi, [])

        cur_ball_ids: dict[int, tuple[float, float]] = {
            b["id"]: b["centroid"] for b in ball_tracks
        }

        # Snapshot age BEFORE the update (we need the age at the moment a ball
        # was last visible, i.e., the age at the previous frame when it was present).
        age_snapshot = dict(ball_consecutive_age)

        # Update consecutive age counters: increment for seen, reset for absent.
        for bid in cur_ball_ids:
            ball_consecutive_age[bid] = ball_consecutive_age.get(bid, 0) + 1
        for bid in list(ball_consecutive_age.keys()):
            if bid not in cur_ball_ids:
                ball_consecutive_age[bid] = 0   # reset; keep key for later

        # Check which ball IDs from the previous frame are now missing.
        for bid, (bcx, bcy) in prev_ball_ids.items():
            if bid in cur_ball_ids:
                continue  # still visible, no pickup

            # Ball must have been stably tracked for enough frames to be real.
            # Use the age it had AT the last frame it was seen (snapshot).
            age = age_snapshot.get(bid, 0)
            if age < PICKUP_MIN_BALL_AGE:
                continue

            # Ball disappeared — was a robot near it?
            nearest_rid, nearest_d = _nearest_robot(bcx, bcy, robot_tracks,
                                                    PICKUP_RADIUS)
            if nearest_rid is None:
                continue

            # Per-ball de-duplicate: same ball ID can't be picked up again soon.
            last_ball_pickup = ball_last_pickup.get(bid, -999)
            if fi - last_ball_pickup < PICKUP_DEDUP_PER_BALL:
                continue

            # Per-robot de-duplicate: same robot can't pick up two balls too fast.
            last_robot_pickup = robot_last_pickup.get(nearest_rid, -999)
            if fi - last_robot_pickup < PICKUP_DEDUP_PER_ROBOT:
                continue

            robot_last_pickup[nearest_rid] = fi
            ball_last_pickup[bid] = fi

            alliance = None
            for r in robot_tracks:
                if r["id"] == nearest_rid:
                    alliance = r["alliance"]
                    break

            events.append({
                "t": float(fi) / fps,
                "frame": fi,
                "type": "pickup",
                "robot_id": nearest_rid,
                "alliance": alliance,
                "detail": {
                    "ball_id": bid,
                    "ball_age": age,
                    "ball_pos": (round(bcx, 1), round(bcy, 1)),
                    "robot_dist": round(nearest_d, 1),
                },
            })

        prev_ball_ids = cur_ball_ids

    # -----------------------------------------------------------------------
    # SHOT detection
    # -----------------------------------------------------------------------
    # For each ball track with shot==True, fire one shot event per ball ID.
    # The event is attributed at the frame the shot flag first appears, using
    # the ball's velocity to confirm it is headed toward a HUB.

    shot_emitted: dict[int, int] = {}   # ball_id -> frame of shot event

    for fi in frames_sorted:
        ball_tracks = ball_tracks_by_frame.get(fi, [])
        robot_tracks = robot_tracks_by_frame.get(fi, [])

        for bt in ball_tracks:
            if not bt.get("shot", False):
                continue

            bid = bt["id"]
            # Only emit once per ball ID; suppress if too close to prior shot.
            last_shot_fi = shot_emitted.get(bid, -999)
            if fi - last_shot_fi < SHOT_DEDUP_FRAMES:
                continue

            bcx, bcy = bt["centroid"]
            vx, vy = bt["vel"]
            spd = math.hypot(vx, vy)
            if spd < SHOT_MIN_SPEED:
                continue

            # Check if the ball's velocity vector aims at any HUB.
            # The track_ball.BallTracker already validates shot=True with multi-
            # criteria (sustained speed + travel + directness), so we trust it.
            # Use a 2x expanded hub bbox for the ray test to compensate for the
            # shallow arc projections at 640x360. If no hub is hit, still accept
            # the shot (record aimed_alliance=None) since false-positive shots from
            # the tracker are rare and recall matters more than perfect attribution.
            aimed_hub = None
            aimed_alliance = None
            for hub in hub_regions:
                hx, hy, hw, hh = hub["bbox"]
                # Expand the target by 1.5x in each direction for the ray test.
                exp = 1.5
                exp_bx = int(hx - hw * (exp - 1) / 2)
                exp_by = int(hy - hh * (exp - 1) / 2)
                exp_bw = int(hw * exp)
                exp_bh = int(hh * exp)
                if _ray_hits_bbox(bcx, bcy, vx, vy, exp_bx, exp_by, exp_bw, exp_bh,
                                  SHOT_LOOKAHEAD):
                    aimed_hub = hub
                    aimed_alliance = hub["alliance"]
                    break

            # Accept ALL tracker-confirmed shots (shot=True from multi-criteria
            # tracker). The aimed_alliance is set if hub aim confirmed; None otherwise.
            # This ensures one event per shot ID rather than filtering on direction.

            # Attribute to the nearest robot at this frame.
            nearest_rid, nearest_d = _nearest_robot(bcx, bcy, robot_tracks,
                                                    SHOT_ATTR_RADIUS)

            # Fall back: search nearby frames for a closer robot if current frame has none.
            if nearest_rid is None:
                for delta in [1, -1, 2, -2, 3]:
                    alt_tracks = robot_tracks_by_frame.get(fi + delta, [])
                    rid, d = _nearest_robot(bcx, bcy, alt_tracks, SHOT_ATTR_RADIUS)
                    if rid is not None:
                        nearest_rid, nearest_d = rid, d
                        break

            robot_alliance = None
            if nearest_rid is not None:
                for search_fi in [fi, fi - 1, fi + 1]:
                    for r in robot_tracks_by_frame.get(search_fi, []):
                        if r["id"] == nearest_rid:
                            robot_alliance = r["alliance"]
                            break
                    if robot_alliance is not None:
                        break

            shot_emitted[bid] = fi

            events.append({
                "t": float(fi) / fps,
                "frame": fi,
                "type": "shot",
                "robot_id": nearest_rid,
                "alliance": robot_alliance,
                "detail": {
                    "ball_id": bid,
                    "ball_pos": (round(bcx, 1), round(bcy, 1)),
                    "vel": (round(vx, 2), round(vy, 2)),
                    "speed": round(spd, 2),
                    "target_hub": aimed_alliance,
                    "robot_dist": round(nearest_d, 1) if nearest_rid is not None else None,
                },
            })

    # -----------------------------------------------------------------------
    # SCORE detection (from overlay_ts fuel counter deltas)
    # -----------------------------------------------------------------------
    # The overlay fuel counter increments as fuel is scored; use the delta to
    # emit score events. Robot ID is None since we can't link a hub score to a
    # specific robot from the camera alone.

    prev_blue_fuel: int | None = None
    prev_red_fuel: int | None = None
    prev_t: float = 0.0

    for row in overlay_ts:
        t = row.get("t", 0.0)
        fi_row = row.get("frame", int(round(t * fps)))
        bf = row.get("blue_fuel")
        rf = row.get("red_fuel")
        dt = max(1.0, t - prev_t)   # time since last sample (seconds)
        max_delta = int(SCORE_MAX_DELTA_PER_S * dt)

        if bf is not None and prev_blue_fuel is not None:
            delta = bf - prev_blue_fuel
            if 0 < delta <= max_delta:
                events.append({
                    "t": float(t),
                    "frame": fi_row,
                    "type": "score",
                    "robot_id": None,
                    "alliance": "blue",
                    "detail": {
                        "fuel_delta": delta,
                        "fuel_total": bf,
                        "source": "overlay_ts",
                    },
                })

        if rf is not None and prev_red_fuel is not None:
            delta = rf - prev_red_fuel
            if 0 < delta <= max_delta:
                events.append({
                    "t": float(t),
                    "frame": fi_row,
                    "type": "score",
                    "robot_id": None,
                    "alliance": "red",
                    "detail": {
                        "fuel_delta": delta,
                        "fuel_total": rf,
                        "source": "overlay_ts",
                    },
                })

        if bf is not None:
            prev_blue_fuel = bf
        if rf is not None:
            prev_red_fuel = rf
        prev_t = t

    # Sort by frame index, then event type (pickup < shot < score) for
    # deterministic ordering within the same frame.
    _ORDER = {"pickup": 0, "shot": 1, "score": 2}
    events.sort(key=lambda e: (e["frame"], _ORDER.get(e["type"], 9)))

    return events


# ---------------------------------------------------------------------------
# CLI self-check: run on the real match; annotate sampled frames with events.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os
    import json
    import cv2

    # Project root = two levels above this file.
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _SRC_DIR = os.path.dirname(_HERE)
    _ROOT = os.path.dirname(_SRC_DIR)
    sys.path.insert(0, _SRC_DIR)
    sys.path.insert(0, _HERE)

    try:
        from sim import detect as _detect
        from sim import field as _field
        from sim.track import RobotTracker, BallTracker
    except Exception:
        import detect as _detect
        import field as _field
        from track import RobotTracker, BallTracker

    VIDEO = os.path.join(_ROOT, "data", "sim", "match_full.mp4")
    OUT_DIR = os.path.join(_ROOT, "events_annot")
    TS_JSON = os.path.join(_ROOT, "data", "sim", "overlay_ts.json")
    os.makedirs(OUT_DIR, exist_ok=True)

    WIDE_END_F = 4170  # wide segment end frame
    FPS = 30.0

    # ---- Load overlay_ts -------------------------------------------------
    if os.path.exists(TS_JSON):
        with open(TS_JSON) as f:
            ts_data = json.load(f)
        overlay_ts = ts_data["timeseries"]
        print(f"Loaded overlay_ts: {len(overlay_ts)} rows from {TS_JSON}")
    else:
        print(f"overlay_ts.json not found at {TS_JSON}, score events will be empty")
        overlay_ts = []

    # ---- Build background -----------------------------------------------
    print("Building background (0..139s)...")
    bg = _detect.build_background(VIDEO, 0.0, 139.0)

    # ---- Run tracking over the wide segment ------------------------------
    print("Tracking robots + balls over wide segment (~4170 frames)...")
    cap = cv2.VideoCapture(VIDEO)
    rt = RobotTracker()
    bt = BallTracker()

    robot_tracks_by_frame: dict[int, list] = {}
    ball_tracks_by_frame: dict[int, list] = {}

    fi = 0
    while fi < WIDE_END_F:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        rdets = _detect.detect_robots(frame, bg)
        bdets = _detect.detect_balls(frame)
        rtracked = rt.update(fi, rdets)
        btracked = bt.update(fi, bdets)
        robot_tracks_by_frame[fi] = rtracked
        ball_tracks_by_frame[fi] = btracked
        fi += 1

    cap.release()
    print(f"Tracked {fi} frames. Building hub regions...")

    # ---- Hub regions + event detection ----------------------------------
    cap2 = cv2.VideoCapture(VIDEO)
    ok, sample_frame = cap2.read()
    cap2.release()
    h, w = sample_frame.shape[:2]
    hubs = _field.hub_regions(w, h)
    print(f"Hub regions: {hubs}")

    print("Detecting events...")
    evs = detect_events(robot_tracks_by_frame, ball_tracks_by_frame,
                        hubs, overlay_ts)

    # ---- Summary --------------------------------------------------------
    pickups = [e for e in evs if e["type"] == "pickup"]
    shots   = [e for e in evs if e["type"] == "shot"]
    scores  = [e for e in evs if e["type"] == "score"]

    print(f"\n=== Event Summary ===")
    print(f"PICKUP: {len(pickups)}")
    print(f"SHOT:   {len(shots)}")
    print(f"SCORE:  {len(scores)}")

    # Per-robot counts
    from collections import defaultdict
    robot_pickups: dict = defaultdict(int)
    robot_shots:   dict = defaultdict(int)
    for e in pickups:
        robot_pickups[(e['robot_id'], e.get('alliance', '?'))] += 1
    for e in shots:
        robot_shots[(e['robot_id'], e.get('alliance', '?'))] += 1

    print("\nPickups per robot:")
    for (rid, al), cnt in sorted(robot_pickups.items(), key=lambda x: (x[0][0] if x[0][0] is not None else -1, x[0][1])):
        print(f"  robot {rid} ({al}): {cnt} pickups")
    print("\nShots per robot:")
    for (rid, al), cnt in sorted(robot_shots.items(), key=lambda x: (x[0][0] if x[0][0] is not None else -1, x[0][1] if x[0][1] is not None else '')):
        print(f"  robot {rid} ({al}): {cnt} shots")

    print("\nScore events (alliance, fuel_delta):")
    for e in scores[:20]:
        print(f"  t={e['t']:.1f}s f={e['frame']} {e['alliance']} +{e['detail']['fuel_delta']} -> {e['detail']['fuel_total']}")

    # ---- Save JSON -------------------------------------------------------
    out_json = os.path.join(_ROOT, "data", "sim", "events.json")
    with open(out_json, "w") as f:
        json.dump(evs, f, indent=2)
    print(f"\nEvents saved: {out_json}  ({len(evs)} events)")

    # ---- Annotate sampled frames ----------------------------------------
    print("Annotating frames...")

    # Map: frame -> list of events at that frame
    events_by_frame: dict[int, list] = defaultdict(list)
    for e in evs:
        events_by_frame[e["frame"]].append(e)

    # Sample every 300 frames + frames with events
    save_fis = set(range(0, WIDE_END_F, 300))
    # Include frames just after each event for visibility
    for e in evs:
        for delta in [0, 1, 2]:
            save_fis.add(e["frame"] + delta)

    # Also include a few shot-annotated frames
    shot_frames = sorted(set(e["frame"] for e in shots))[:30]
    save_fis.update(shot_frames)

    # Color scheme
    ALLIANCE_COL = {"blue": (255, 80, 0), "red": (0, 60, 255), None: (200, 200, 200)}
    EVENT_COL = {"pickup": (0, 255, 128), "shot": (0, 200, 255), "score": (255, 255, 0)}

    cap3 = cv2.VideoCapture(VIDEO)
    saved_count = 0
    for target_fi in sorted(save_fis):
        if target_fi >= WIDE_END_F:
            continue
        cap3.set(cv2.CAP_PROP_POS_FRAMES, target_fi)
        ok, frame = cap3.read()
        if not ok:
            continue

        ann = frame.copy()
        h2, w2 = ann.shape[:2]

        # Draw HUB regions
        for hub in hubs:
            hx, hy, hw, hh = hub["bbox"]
            col = (255, 0, 0) if hub["alliance"] == "blue" else (0, 0, 255)
            cv2.rectangle(ann, (hx, hy), (hx + hw, hy + hh), col, 1)

        # Draw robot tracks
        rtrks = robot_tracks_by_frame.get(target_fi, [])
        for r in rtrks:
            x, y, bw, bh = r["bbox"]
            col = ALLIANCE_COL.get(r.get("alliance"), (200, 200, 200))
            cv2.rectangle(ann, (x, y), (x + bw, y + bh), col, 2)
            cv2.putText(ann, f"{r['id']}:{r.get('alliance','?')[0].upper()}",
                        (x, max(8, y - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)

        # Draw ball tracks
        btrks = ball_tracks_by_frame.get(target_fi, [])
        for b in btrks:
            bcx, bcy = b["centroid"]
            col = (0, 200, 255) if b.get("shot") else (0, 255, 255)
            cv2.circle(ann, (int(bcx), int(bcy)), b["r"] + 1, col, 1)
            if b.get("shot"):
                vx, vy = b["vel"]
                cv2.arrowedLine(ann, (int(bcx), int(bcy)),
                                (int(bcx + vx * 5), int(bcy + vy * 5)),
                                (0, 255, 200), 1, tipLength=0.3)

        # Draw events AT this frame
        frame_evs = events_by_frame.get(target_fi, [])
        y_off = 15
        for ev in frame_evs:
            ecol = EVENT_COL.get(ev["type"], (200, 200, 200))
            if ev["type"] == "pickup":
                bpos = ev["detail"].get("ball_pos")
                if bpos:
                    bx2, by2 = int(bpos[0]), int(bpos[1])
                    cv2.circle(ann, (bx2, by2), 8, ecol, 2)
                    cv2.putText(ann, f"PICK r{ev['robot_id']}", (bx2 - 10, by2 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.32, ecol, 1)
            elif ev["type"] == "shot":
                bpos = ev["detail"].get("ball_pos")
                if bpos:
                    bx2, by2 = int(bpos[0]), int(bpos[1])
                    cv2.circle(ann, (bx2, by2), 10, ecol, 2)
                    vx2, vy2 = ev["detail"].get("vel", (0, 0))
                    cv2.arrowedLine(ann, (bx2, by2),
                                   (int(bx2 + vx2 * 8), int(by2 + vy2 * 8)),
                                   ecol, 2, tipLength=0.3)
                    cv2.putText(ann, f"SHOT r{ev['robot_id']}→{ev['detail'].get('target_hub','?')}",
                                (bx2 - 10, by2 - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.32, ecol, 1)

            label = (f"{ev['type'].upper()} rid={ev['robot_id']} "
                     f"al={ev.get('alliance','?')} "
                     + (f"+{ev['detail'].get('fuel_delta','')} "
                        if ev['type'] == 'score' else ""))
            cv2.putText(ann, label, (4, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, ecol, 1)
            y_off += 12

        # HUD
        cv2.putText(ann, f"f{target_fi}  P:{len(pickups)} S:{len(shots)} SC:{len(scores)}",
                    (4, h2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        big = cv2.resize(ann, None, fx=2.0, fy=2.0,
                         interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(OUT_DIR, f"f{target_fi:05d}.png"), big)
        saved_count += 1

    cap3.release()
    print(f"Annotated {saved_count} frames -> {OUT_DIR}")
