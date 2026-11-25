"""
track_ball.py — persistent frame-to-frame tracking of FUEL balls for the single
static wide camera of this FRC REBUILT (2026) playoff match.

The integrator re-exports ``BallTracker`` from ``track.py``; it lives here to
avoid a file collision with the RobotTracker (which is authored separately in
track.py). The ``update()`` contract matches the RobotTracker:

    tracker = make_ball_tracker()                 # or BallTracker()
    tracks  = tracker.update(frame_idx, detections)

where ``detections`` is the output of ``detect.detect_balls(frame)`` —
a list of ``{'centroid':(cx,cy), 'r':int}`` — and ``tracks`` is a list of

    {'id':int, 'centroid':(cx,cy), 'r':int, 'vel':(vx,vy),
     'shot':bool, 'age':int, 'hits':int}

Approach (classical, deterministic — no ML)
-------------------------------------------
There are dozens of mostly-still yellow balls littering the carpet plus a
handful that are in flight after a robot shoots. Per-frame detection is noisy
(clusters split/merge, recall flickers), so a tracker must:

  * Predict each live track forward with a constant-velocity model (a light
    Kalman-style position+velocity filter; velocity is an exponential moving
    average of measured frame-to-frame displacement).
  * Associate detections to predictions greedily by nearest distance within a
    speed-aware gate (a fast ball can jump far between frames, so the gate
    grows with the track's current speed).
  * Birth new tracks for unmatched detections, and let unmatched tracks COAST
    for a few frames (re-acquire after brief occlusion / detection dropout)
    before they die.

Shot identification
-------------------
A "shot" is a ball that accelerates away from the carpet clutter and flies
(toward a HUB). The vast majority of balls are static or jostling slowly. We
flag a track as a SHOT when its smoothed speed crosses SHOT_SPEED for at least
SHOT_MIN_FRAMES consecutive frames AND it has travelled a real distance
(SHOT_MIN_TRAVEL) from where the fast motion began — this rejects detector
jitter on a stationary ball and one-frame association errors. Once flagged, a
track keeps ``shot=True`` for the rest of its life so downstream code can read
the whole trajectory.

Self-contained: a CLI at the bottom runs the tracker on the real video, draws
ball tracks + highlighted shot trajectories, and writes annotated frames so the
result can be eyeballed.
"""

from __future__ import annotations

import math
from typing import List, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Tunables (pixel units at 640x360, 30 fps)
# ---------------------------------------------------------------------------
# Association gate. A still ball moves ~0 px/frame; a flying shot can cover
# 15-40 px/frame at this zoom. Base gate handles slow jostle + detector jitter;
# the speed term lets a fast track reach its next detection.
_GATE_BASE = 18.0          # px: base match radius for a slow/new track
_GATE_SPEED_K = 1.8        # gate grows by k * current_speed
_GATE_MAX = 70.0           # px: hard cap so we never match across the field

# Velocity smoothing (EMA). Higher alpha = snappier (follows real accel of a
# shot faster) but noisier; lower = smoother but laggy. A flying ball changes
# velocity fast and only lives a handful of frames, so we keep this high (0.8)
# to latch onto shot motion before the track is lost; static-ball jitter is
# rejected later by the travel + sustained-speed requirements.
_VEL_ALPHA = 0.8

# Track lifecycle
_MAX_COAST = 6             # frames a track may survive with no detection
_MIN_HITS_REPORT = 1       # report a track once it has >= this many hits

# Association bias: a moving track gets a discount on its match cost so a flying
# ball passing over the dense carpet-ball field is not "stolen" by a closer
# static neighbour (greedy NN otherwise reassigns the fast ID to clutter).
_FAST_BIAS = 0.6           # multiply cost by this for tracks above _SHOT_SPEED

# Shot classification. A shot ball flickers in/out of the detector as it crosses
# clusters, so we tolerate brief speed dips (fast_run only decays, not resets)
# and count a run of "mostly fast" frames rather than strictly consecutive ones.
_SHOT_SPEED = 6.5          # px/frame smoothed speed to be "in flight"
_SHOT_MIN_FRAMES = 3       # accumulated fast frames required
_SHOT_MIN_TRAVEL = 28.0    # px straight-line travel since fast motion began
_SHOT_MIN_DIRECT = 0.55    # net/path displacement ratio (flight is directional)


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class _BallTrack:
    """One tracked ball with a constant-velocity prediction model."""

    __slots__ = (
        "id", "x", "y", "r", "vx", "vy",
        "age", "hits", "misses",
        "fast_run", "shot", "shot_origin", "shot_path", "recent",
    )

    def __init__(self, tid: int, cx: float, cy: float, r: int):
        self.id = tid
        self.x = float(cx)
        self.y = float(cy)
        self.r = int(r)
        self.vx = 0.0
        self.vy = 0.0
        self.age = 0           # frames since birth
        self.hits = 1          # detections associated
        self.misses = 0        # consecutive frames without a detection
        self.fast_run = 0.0    # accumulated fast frames (decays on slow frame)
        self.shot = False      # sticky once detected
        self.shot_origin: Optional[Tuple[float, float]] = None
        self.shot_path: List[Tuple[int, int]] = [(int(cx), int(cy))]
        # short ring of recent confirmed positions, for directionality test
        self.recent: List[Tuple[float, float]] = [(float(cx), float(cy))]

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    def predict(self) -> Tuple[float, float]:
        """Predicted centroid for the next frame (constant velocity)."""
        return (self.x + self.vx, self.y + self.vy)

    def gate(self) -> float:
        return min(_GATE_MAX, _GATE_BASE + _GATE_SPEED_K * self.speed)

    def update_with(self, cx: float, cy: float, r: int):
        """Correct the track with an associated detection."""
        # Measured displacement from the PREVIOUS confirmed position.
        mdx = cx - self.x
        mdy = cy - self.y
        # If the track coasted for k misses, the measured step spans k+1 frames;
        # normalise to a per-frame velocity so a re-acquired shot keeps its speed.
        span = self.misses + 1
        mvx = mdx / span
        mvy = mdy / span
        self.vx = _VEL_ALPHA * mvx + (1.0 - _VEL_ALPHA) * self.vx
        self.vy = _VEL_ALPHA * mvy + (1.0 - _VEL_ALPHA) * self.vy
        self.x = float(cx)
        self.y = float(cy)
        self.r = int(r)
        self.age += 1
        self.hits += 1
        self.misses = 0
        self.recent.append((self.x, self.y))
        if len(self.recent) > 8:
            self.recent = self.recent[-8:]
        self._update_shot_state()

    def coast(self):
        """No detection this frame: advance by velocity, decay a little."""
        self.x += self.vx
        self.y += self.vy
        # Mild drag so a coasting (occluded) ball doesn't fly forever.
        self.vx *= 0.85
        self.vy *= 0.85
        self.age += 1
        self.misses += 1
        # A coasting fast ball is still mid-flight; keep extending the path so
        # the rendered shot trajectory is continuous through brief dropouts.
        if self.shot:
            self.shot_path.append((int(self.x), int(self.y)))

    def _directness(self) -> float:
        """Net displacement / path length over the recent window (1=straight)."""
        if len(self.recent) < 3:
            return 1.0
        net = _dist(self.recent[0], self.recent[-1])
        path = 0.0
        for k in range(1, len(self.recent)):
            path += _dist(self.recent[k - 1], self.recent[k])
        return (net / path) if path > 1e-6 else 1.0

    def _update_shot_state(self):
        spd = self.speed
        if spd >= _SHOT_SPEED:
            if self.fast_run <= 0.0:
                self.shot_origin = (self.x, self.y)
            self.fast_run += 1.0
        else:
            # decay rather than reset: tolerate a brief speed dip (occlusion /
            # detector flicker) without losing an in-progress shot run.
            self.fast_run -= 1.0
            if self.fast_run <= 0.0:
                self.fast_run = 0.0
                if not self.shot:
                    self.shot_origin = None

        if not self.shot and self.fast_run >= _SHOT_MIN_FRAMES \
                and self.shot_origin is not None:
            travel = _dist(self.shot_origin, (self.x, self.y))
            if travel >= _SHOT_MIN_TRAVEL and self._directness() >= _SHOT_MIN_DIRECT:
                self.shot = True

        if self.shot:
            self.shot_path.append((int(self.x), int(self.y)))
            if len(self.shot_path) > 60:
                self.shot_path = self.shot_path[-60:]

    def as_dict(self) -> Dict:
        return {
            "id": self.id,
            "centroid": (int(round(self.x)), int(round(self.y))),
            "r": int(self.r),
            "vel": (round(self.vx, 2), round(self.vy, 2)),
            "shot": bool(self.shot),
            "age": int(self.age),
            "hits": int(self.hits),
        }


class BallTracker:
    """Persistent multi-ball tracker. Same update() contract as RobotTracker."""

    def __init__(
        self,
        gate_base: float = _GATE_BASE,
        gate_speed_k: float = _GATE_SPEED_K,
        max_coast: int = _MAX_COAST,
        shot_speed: float = _SHOT_SPEED,
        shot_min_frames: int = _SHOT_MIN_FRAMES,
        shot_min_travel: float = _SHOT_MIN_TRAVEL,
    ):
        self._next_id = 0
        self._tracks: List[_BallTrack] = []
        self.gate_base = gate_base
        self.gate_speed_k = gate_speed_k
        self.max_coast = max_coast
        self.shot_speed = shot_speed
        self.shot_min_frames = shot_min_frames
        self.shot_min_travel = shot_min_travel
        # Record of every track that was ever flagged a shot (for reporting).
        self._shot_ids: set = set()
        self.last_frame_idx = -1

    def update(self, frame_idx: int, detections: List[Dict]) -> List[Dict]:
        """Advance all tracks one frame and associate ``detections``.

        Parameters
        ----------
        frame_idx  : int frame number (monotonic; used only for bookkeeping).
        detections : list of {'centroid':(cx,cy), 'r':int} from detect_balls.

        Returns
        -------
        list of {'id','centroid','r','vel','shot','age','hits'} for every live
        track with enough hits to report.
        """
        self.last_frame_idx = frame_idx

        dets = [
            (float(d["centroid"][0]), float(d["centroid"][1]),
             int(d.get("r", 4)))
            for d in detections
        ]

        # --- Predict + build a cost-ordered candidate list -------------------
        preds = [t.predict() for t in self._tracks]
        candidates = []  # (cost, track_idx, det_idx)
        for ti, t in enumerate(self._tracks):
            px, py = preds[ti]
            spd = t.speed
            g = min(_GATE_MAX, self.gate_base + self.gate_speed_k * spd)
            bias = _FAST_BIAS if spd >= self.shot_speed else 1.0
            for di, (dx, dy, _r) in enumerate(dets):
                geo = math.hypot(px - dx, py - dy)
                if geo <= g:
                    # bias only orders the greedy pick; the gate uses true geo
                    candidates.append((geo * bias, ti, di))
        candidates.sort(key=lambda z: z[0])

        # --- Greedy association (closest pairs first) ------------------------
        track_used = [False] * len(self._tracks)
        det_used = [False] * len(dets)
        for c, ti, di in candidates:
            if track_used[ti] or det_used[di]:
                continue
            dx, dy, dr = dets[di]
            self._tracks[ti].update_with(dx, dy, dr)
            track_used[ti] = True
            det_used[di] = True

        # --- Coast unmatched tracks; cull the stale ones ---------------------
        survivors: List[_BallTrack] = []
        for ti, t in enumerate(self._tracks):
            if not track_used[ti]:
                t.coast()
            if t.misses <= self.max_coast:
                survivors.append(t)
        self._tracks = survivors

        # --- Birth tracks for unmatched detections ---------------------------
        for di, used in enumerate(det_used):
            if used:
                continue
            dx, dy, dr = dets[di]
            self._tracks.append(_BallTrack(self._next_id, dx, dy, dr))
            self._next_id += 1

        # Record shots seen this frame.
        for t in self._tracks:
            if t.shot:
                self._shot_ids.add(t.id)

        return [t.as_dict() for t in self._tracks
                if t.hits >= _MIN_HITS_REPORT]

    # -- convenience accessors -------------------------------------------------
    def live_shots(self) -> List[Dict]:
        """Currently-live tracks flagged as shots (with their path)."""
        out = []
        for t in self._tracks:
            if t.shot:
                d = t.as_dict()
                d["path"] = list(t.shot_path)
                out.append(d)
        return out

    @property
    def shot_ids(self) -> set:
        """All track IDs ever flagged as a shot during this tracker's life."""
        return set(self._shot_ids)


def make_ball_tracker(**kwargs) -> BallTracker:
    """Factory mirroring the contract name. See BallTracker for kwargs."""
    return BallTracker(**kwargs)


# ---------------------------------------------------------------------------
# CLI self-check: run on the real video, draw tracks + shots, save frames
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import os
    import cv2
    import numpy as np

    # Import the sibling detector (works whether run as module or script).
    try:
        from . import detect as _detect  # type: ignore
    except Exception:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import detect as _detect  # type: ignore

    video = sys.argv[1] if len(sys.argv) > 1 else "data/sim/match_full.mp4"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "ball_track_annot"
    # Process the wide segment only (0..139s -> frames 0..4166).
    start_f = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    end_f = int(sys.argv[4]) if len(sys.argv) > 4 else 4166
    save_every = int(sys.argv[5]) if len(sys.argv) > 5 else 15

    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    tracker = make_ball_tracker()

    # Stable per-id color for drawing.
    rng = np.random.default_rng(7)
    colors: dict = {}

    def color_for(i):
        if i not in colors:
            colors[i] = tuple(int(v) for v in rng.integers(60, 256, size=3))
        return colors[i]

    shot_events = []   # (frame, id, origin, end, travel)
    seen_shot_ids: set = set()
    n_frames = 0
    import time
    t_start = time.perf_counter()

    fi = start_f
    while fi < end_f:
        ok, frame = cap.read()
        if not ok:
            break
        dets = _detect.detect_balls(frame)
        tracks = tracker.update(fi, dets)
        n_frames += 1

        # Detect newly-promoted shots for reporting.
        for t in tracker._tracks:  # noqa: SLF001 (self-check only)
            if t.shot and t.id not in seen_shot_ids:
                seen_shot_ids.add(t.id)
                org = t.shot_origin or (t.x, t.y)
                shot_events.append(
                    (fi, t.id, (int(org[0]), int(org[1])),
                     (int(t.x), int(t.y)),
                     round(_dist(org, (t.x, t.y)), 1)))

        if fi % save_every == 0:
            ann = frame.copy()
            # all live tracks: small marker + short velocity vector
            for d in tracks:
                cx, cy = d["centroid"]
                col = color_for(d["id"])
                if d["shot"]:
                    continue  # drawn below, on top
                cv2.circle(ann, (cx, cy), max(2, d["r"]), col, 1)
                vx, vy = d["vel"]
                if abs(vx) + abs(vy) > 1.5:
                    cv2.arrowedLine(ann, (cx, cy),
                                    (int(cx + vx * 2), int(cy + vy * 2)),
                                    col, 1, tipLength=0.4)
            # shots: bright trajectory + ring
            for s in tracker.live_shots():
                path = s["path"]
                for k in range(1, len(path)):
                    cv2.line(ann, path[k - 1], path[k], (0, 0, 255), 2)
                cx, cy = s["centroid"]
                cv2.circle(ann, (cx, cy), max(4, s["r"] + 2), (0, 0, 255), 2)
                cv2.putText(ann, f"SHOT {s['id']}", (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            n_live = len(tracks)
            n_shot = len(tracker.live_shots())
            cv2.putText(ann,
                        f"f{fi} balls:{len(dets)} tracks:{n_live} "
                        f"shots(total):{len(seen_shot_ids)} live:{n_shot}",
                        (6, 352), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 255, 255), 1)
            big = cv2.resize(ann, None, fx=2.0, fy=2.0,
                             interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(out_dir, f"f{fi:05d}.png"), big)
        fi += 1

    cap.release()
    elapsed = time.perf_counter() - t_start

    print(f"\nProcessed {n_frames} frames [{start_f}..{fi}) "
          f"in {elapsed:.1f}s ({1000*elapsed/max(1,n_frames):.1f} ms/frame)")
    print(f"Total distinct SHOT tracks flagged: {len(seen_shot_ids)}")
    print(f"{'frame':>6} {'t(s)':>6} {'id':>4} {'origin':>12} "
          f"{'end':>12} {'travel':>7}")
    for (f, i, org, end, tr) in shot_events:
        print(f"{f:>6} {f/30.0:>6.1f} {i:>4} "
              f"{str(org):>12} {str(end):>12} {tr:>7}")
    print(f"\nAnnotated frames -> {out_dir}")
