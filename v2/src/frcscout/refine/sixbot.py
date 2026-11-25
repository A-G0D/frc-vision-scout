"""SixRobotTracker — the structural-prior tracker (THE key module).

Every FRC match has EXACTLY six robots on the field: three red and three blue.
That hard count is the strongest prior we have, far stronger than any per-frame
detector confidence. This tracker turns that prior into identity stability.

What it does
------------
We maintain *up to* three persistent tracks per alliance (six total). Each frame:

  1. predict every track forward with a constant-velocity Kalman (so a track that
     is briefly occluded or has left the visible frame keeps a plausible position);
  2. assign this frame's detections to existing tracks with the Hungarian algorithm
     (``scipy.optimize.linear_sum_assignment``; a greedy nearest-neighbour fallback
     is used when SciPy is unavailable). Cost is centroid distance discounted by
     IoU, and the assignment is *GATED by alliance* — a red detection can only ever
     bind to a red track, a blue to a blue. Pairs beyond the gate are forbidden;
  3. an unmatched detection BIRTHS a new track for its alliance, capped at three per
     alliance. If that alliance is already full, the weakest existing track (fewest
     hits, breaking ties by longest unseen) is evicted so the fresh, currently-seen
     robot takes its slot — the count is conserved, identity follows the evidence;
  4. an unmatched *track* is NOT dropped. Robots do not leave a match. The track
     COASTS on its Kalman prediction and is still emitted, flagged ``est=True``
     (estimate) so consumers know the box is predicted rather than measured. When the
     robot reappears, association re-binds it to the same track -> the SAME id.

Output: up to six dicts, one per live track::

    {"id": int, "alliance": "red"|"blue",
     "xyxy": (x1,y1,x2,y2), "centroid": (cx,cy),
     "est": bool,        # True while coasting (no detection matched this frame)
     "hits": int,        # successful measurement updates over the track's life
     "age": int}         # frames since birth

Design / dependency notes
-------------------------
* All timing is in SECONDS and converted with ``seconds_to_frames`` against the real
  capture fps — never a hardcoded frame count (see ``coast_grace_s``; note that, per
  the structural prior, coasting tracks are never culled for *absence*, so the grace
  window only governs when a coasting track may be *evicted* to make room for a birth).
* The Kalman filter is implemented in PURE PYTHON (a small decoupled per-axis
  constant-velocity filter). The 4-D state is tiny, so NumPy would only add overhead,
  and — critically — it lets the whole tracker run and be unit-tested with NO heavy
  dependency installed. NumPy is used *only* to build the cost matrix when present;
  SciPy *only* for the optimal assignment. Both are import-guarded so this module
  PARSES and the pure logic runs even if neither is installed.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

from frcscout.io.config import seconds_to_frames
from frcscout.io.types import Alliance, Detection

# ── Heavy deps are OPTIONAL. Guard so the file parses / runs without them. ────
try:  # NumPy only accelerates cost-matrix construction; not required.
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - exercised only when numpy is absent
    _np = None

try:  # SciPy gives the optimal assignment; we fall back to greedy without it.
    from scipy.optimize import linear_sum_assignment as _lsa  # type: ignore
except Exception:  # pragma: no cover - exercised only when scipy is absent
    _lsa = None

HAS_NUMPY = _np is not None
HAS_SCIPY = _lsa is not None

ALLIANCES: tuple[Alliance, Alliance] = ("red", "blue")
MAX_PER_ALLIANCE = 3
# A cost at/above this means "forbidden" (used to gate cross-alliance and out-of-gate
# pairs in the dense matrix the assignment solver consumes).
_FORBIDDEN = 1e9


# ─────────────────────────────── geometry helpers ────────────────────────────
def _centroid(xyxy: Sequence[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _wh(xyxy: Sequence[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (abs(x2 - x1), abs(y2 - y1))


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two xyxy boxes (0 when they do not overlap)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ─────────────────────────── constant-velocity Kalman ────────────────────────
class _CVTrack:
    """Per-robot constant-velocity Kalman track, implemented in pure Python.

    State is the centroid and its velocity, modelled as two INDEPENDENT 2-D
    (position, velocity) filters — one for x, one for y. With axis-aligned process
    and measurement noise this decoupling is exact and avoids any matrix library.
    The last measured box size (w, h) is held so a coasting track can still emit a
    plausibly-sized box at its predicted centre.
    """

    __slots__ = (
        "tid", "alliance", "dt",
        "x", "vx", "px", "pvx",
        "y", "vy", "py", "pvy",
        "w", "h",
        "age", "hits", "time_since_update", "last_frame",
        "q_pos", "q_vel", "r_meas",
    )

    def __init__(self, tid: int, alliance: Optional[Alliance],
                 det: Detection, frame_idx: int, dt: float = 1.0,
                 q_pos: float = 1.0, q_vel: float = 0.01, r_meas: float = 1.0):
        self.tid = tid
        self.alliance = alliance
        self.dt = float(dt)

        cx, cy = _centroid(det.xyxy)
        w, h = _wh(det.xyxy)
        # x-axis filter: position x, velocity vx (+ their variances)
        self.x, self.vx = cx, 0.0
        self.px, self.pvx = 10.0, 100.0     # large initial velocity uncertainty
        # y-axis filter
        self.y, self.vy = cy, 0.0
        self.py, self.pvy = 10.0, 100.0
        self.w, self.h = max(1.0, w), max(1.0, h)

        self.age = 0
        self.hits = 1
        self.time_since_update = 0          # frames since last measurement
        self.last_frame = frame_idx

        self.q_pos = float(q_pos)            # process noise on position
        self.q_vel = float(q_vel)            # process noise on velocity
        self.r_meas = float(r_meas)          # measurement noise (px^2)

    # -- motion: advance one frame (constant velocity) -------------------------
    def predict(self) -> None:
        dt = self.dt
        # x: position integrates velocity; position variance grows with the velocity
        # variance over dt plus process noise (the F P F^T + Q step, decoupled).
        self.x += self.vx * dt
        self.px = self.px + (dt * dt) * self.pvx + self.q_pos
        self.pvx = self.pvx + self.q_vel
        # y
        self.y += self.vy * dt
        self.py = self.py + (dt * dt) * self.pvy + self.q_pos
        self.pvy = self.pvy + self.q_vel

        self.age += 1
        self.time_since_update += 1

    def _update_axis(self, pos: float, vel: float, p_pos: float, p_vel: float,
                     z: float) -> tuple[float, float, float, float]:
        """Scalar Kalman correction of one (position, velocity) axis from a position
        measurement ``z``. Velocity is observed only through the position innovation,
        scaled by the cross-term we approximate with the velocity variance and dt."""
        dt = self.dt
        s = p_pos + self.r_meas                  # innovation variance
        k_pos = p_pos / s                         # gain on position
        k_vel = (p_vel * dt) / s                  # gain on velocity (via dt coupling)
        resid = z - pos
        pos = pos + k_pos * resid
        vel = vel + k_vel * resid
        p_pos = (1.0 - k_pos) * p_pos
        p_vel = (1.0 - k_vel * dt) * p_vel
        # keep variances positive / sane
        p_pos = max(1e-3, p_pos)
        p_vel = max(1e-3, p_vel)
        return pos, vel, p_pos, p_vel

    def update(self, det: Detection, frame_idx: int) -> None:
        cx, cy = _centroid(det.xyxy)
        w, h = _wh(det.xyxy)
        self.x, self.vx, self.px, self.pvx = self._update_axis(
            self.x, self.vx, self.px, self.pvx, cx)
        self.y, self.vy, self.py, self.pvy = self._update_axis(
            self.y, self.vy, self.py, self.pvy, cy)
        self.w, self.h = max(1.0, w), max(1.0, h)
        self.hits += 1
        self.time_since_update = 0
        self.last_frame = frame_idx

    # -- views -----------------------------------------------------------------
    @property
    def centroid(self) -> tuple[float, float]:
        return (self.x, self.y)

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x - self.w / 2.0, self.y - self.h / 2.0,
                self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def est(self) -> bool:
        """True when this frame produced no measurement (the track is coasting)."""
        return self.time_since_update > 0


# ─────────────────────────────── the tracker ─────────────────────────────────
class SixRobotTracker:
    """Six-robot structural-prior tracker: <=3 persistent tracks per alliance.

    Parameters
    ----------
    cfg : dict | None
        Optional knobs:
          ``gate_px``        base association gate radius in pixels (default 120);
          ``gate_speed_k``   extra gate px per px/frame of track speed (default 1.5);
          ``iou_weight``     how strongly box IoU discounts distance cost (default 80);
          ``coast_grace_s``  seconds a coasting (unseen) track is protected from being
                             evicted to make room for a birth (default 1.0);
          ``q_pos`` / ``q_vel`` / ``r_meas``  Kalman noise terms.
    fps : float
        The capture's REAL frames-per-second (used for all seconds->frames timing).
    """

    def __init__(self, cfg: Optional[dict] = None, fps: float = 60.0):
        self.cfg = cfg or {}
        self.fps = float(fps)
        self.dt = 1.0  # state advances exactly one frame per update() call

        self.gate_px = float(self.cfg.get("gate_px", 120.0))
        self.gate_speed_k = float(self.cfg.get("gate_speed_k", 1.5))
        self.coast_widen = float(self.cfg.get("coast_widen_px", 8.0))
        self.iou_weight = float(self.cfg.get("iou_weight", 80.0))
        self.q_pos = float(self.cfg.get("q_pos", 1.0))
        self.q_vel = float(self.cfg.get("q_vel", 0.01))
        self.r_meas = float(self.cfg.get("r_meas", 1.0))

        # Seconds -> frames via the REAL fps (never a hardcoded frame count).
        self.coast_grace = seconds_to_frames(
            float(self.cfg.get("coast_grace_s", 1.0)), self.fps)

        # ── HONESTY (R9): the prior provides up to six SLOTS, but a slot is only a
        #   CONFIDENT robot while it has recent in-field evidence. A track coasting beyond
        #   ``confident_grace_s`` is still emitted (so a short occlusion keeps its id) but
        #   flagged confident=False — consumers must not treat it as a measured robot. A
        #   track coasting beyond ``drop_after_s`` is DROPPED entirely rather than invented
        #   indefinitely (so we never fill all six slots with stale phantoms when the camera
        #   is only showing two robots). drop_after_s defaults LONG so brief occlusions and
        #   the existing coast/eviction tests are unaffected; tune it down for production.
        self.confident_grace = seconds_to_frames(
            float(self.cfg.get("confident_grace_s", 0.5)), self.fps)
        self.drop_after = seconds_to_frames(
            float(self.cfg.get("drop_after_s", 8.0)), self.fps)

        # tracks bucketed by alliance so the 3-per-alliance cap is structural
        self._tracks: dict[Alliance, list[_CVTrack]] = {a: [] for a in ALLIANCES}
        self._next_id = 0

    # ───────────────────────────── association ───────────────────────────────
    def _gate(self, t: _CVTrack) -> float:
        # Base gate + speed term + a COAST term: a track coasting (unseen) for several
        # frames is positionally uncertain, so widen its gate so the robot re-binds to the
        # SAME id when it reappears instead of birthing a new one.
        speed = math.hypot(t.vx, t.vy)
        return (self.gate_px + self.gate_speed_k * speed
                + self.coast_widen * t.time_since_update)

    def _cost(self, t: _CVTrack, det: Detection) -> float:
        """Assignment cost: centroid distance, discounted by IoU. Returns
        ``_FORBIDDEN`` when the pair is outside the track's (speed-aware) gate so the
        solver never picks an implausible match."""
        tc = t.centroid
        dc = _centroid(det.xyxy)
        dist = math.hypot(tc[0] - dc[0], tc[1] - dc[1])
        if dist > self._gate(t):
            return _FORBIDDEN
        iou = _iou(t.xyxy, det.xyxy)
        return dist - self.iou_weight * iou

    def _assign(self, tracks: list[_CVTrack], dets: list[Detection]):
        """Optimal (Hungarian) one-to-one assignment within the per-alliance group,
        falling back to greedy nearest-cost if SciPy is unavailable. Pairs whose cost
        is ``_FORBIDDEN`` (outside the gate) are rejected even if the solver returns
        them. Returns (matches[(track_idx, det_idx)], unmatched_tracks, unmatched_dets).
        """
        nt, nd = len(tracks), len(dets)
        if nt == 0 or nd == 0:
            return [], list(range(nt)), list(range(nd))

        cost = [[self._cost(t, d) for d in dets] for t in tracks]

        if HAS_SCIPY:
            rows, cols = self._hungarian_scipy(cost, nt, nd)
        else:
            rows, cols = self._greedy(cost, nt, nd)

        matches: list[tuple[int, int]] = []
        used_t: set[int] = set()
        used_d: set[int] = set()
        for ti, di in zip(rows, cols):
            if cost[ti][di] >= _FORBIDDEN:        # gated out -> not a real match
                continue
            matches.append((ti, di))
            used_t.add(ti)
            used_d.add(di)

        unmatched_t = [i for i in range(nt) if i not in used_t]
        unmatched_d = [i for i in range(nd) if i not in used_d]
        return matches, unmatched_t, unmatched_d

    @staticmethod
    def _hungarian_scipy(cost: list[list[float]], nt: int, nd: int):
        """Optimal assignment via scipy. ``linear_sum_assignment`` handles
        rectangular matrices, assigning min(nt, nd) pairs."""
        mat = _np.asarray(cost, dtype=float)
        r, c = _lsa(mat)
        return list(r), list(c)

    @staticmethod
    def _greedy(cost: list[list[float]], nt: int, nd: int):
        """Greedy nearest-cost fallback: repeatedly take the globally cheapest
        remaining (track, det) pair. Approximate, but identical to scipy on the small,
        well-separated 3x3 problems this tracker poses."""
        pairs = sorted(
            ((cost[ti][di], ti, di) for ti in range(nt) for di in range(nd)),
            key=lambda p: p[0],
        )
        used_t: set[int] = set()
        used_d: set[int] = set()
        rows: list[int] = []
        cols: list[int] = []
        for _c, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            rows.append(ti)
            cols.append(di)
            used_t.add(ti)
            used_d.add(di)
        return rows, cols

    # ───────────────────────────── main step ─────────────────────────────────
    def update(self, frame_idx: int, detections: Sequence[Detection],
               alliances: Optional[Sequence[Optional[Alliance]]] = None) -> list[dict]:
        """Advance the tracker one frame and return up to six robot dicts.

        Parameters
        ----------
        frame_idx : int
            Index of the current frame.
        detections : sequence of Detection
            This frame's robot detections (image pixels).
        alliances : sequence of ("red"|"blue"|None), optional
            Per-detection bumper-colour label, parallel to ``detections``. A detection
            can only ever bind to (or birth) a track of its own alliance. Detections
            with an unknown alliance (``None`` or missing) are ignored, because the
            structural prior is defined per alliance. If omitted entirely, every
            detection is treated as unknown and nothing is tracked.
        """
        dets = list(detections)
        if alliances is None:
            allys: list[Optional[Alliance]] = [None] * len(dets)
        else:
            allys = list(alliances)
            if len(allys) < len(dets):              # pad defensively
                allys += [None] * (len(dets) - len(allys))

        # 1) predict every track forward one frame
        for bucket in self._tracks.values():
            for t in bucket:
                t.predict()

        # 2) per-alliance association + birth (the gate is structural: red<->red only)
        for alliance in ALLIANCES:
            idx = [i for i, a in enumerate(allys) if a == alliance]
            grp_dets = [dets[i] for i in idx]
            tracks = self._tracks[alliance]

            matches, _unmatched_t, unmatched_d = self._assign(tracks, grp_dets)
            for ti, di in matches:
                tracks[ti].update(grp_dets[di], frame_idx)

            # 3) births from unmatched detections, capped at 3/alliance
            for di in unmatched_d:
                self._birth(alliance, grp_dets[di], frame_idx)

        # 4) HONEST death: a track is normally kept while coasting (the 6-robot prior, so a
        #    brief occlusion keeps its id), BUT a track that has coasted longer than
        #    ``drop_after`` has no in-field evidence for too long to keep ASSERTING it
        #    exists — we drop it rather than invent a confident robot from nothing. This is
        #    the R9 honesty fix: we no longer FORCE six; we emit only slots with evidence.
        if self.drop_after > 0:
            for alliance in ALLIANCES:
                self._tracks[alliance] = [
                    t for t in self._tracks[alliance]
                    if t.time_since_update <= self.drop_after]

        # 5) emit up to six dicts (a coasting track within drop_after is still emitted, but
        #    flagged confident=False once it is past the confident-grace window).
        out: list[dict] = []
        for alliance in ALLIANCES:
            for t in self._tracks[alliance]:
                out.append(self._emit(t))
        return out

    # ───────────────────────────── birth / evict ─────────────────────────────
    def _birth(self, alliance: Alliance, det: Detection, frame_idx: int) -> None:
        """Start a track for an unmatched detection. If the alliance already holds the
        maximum of three, evict the weakest CURRENTLY-COASTING track to conserve the
        count; if every slot is a freshly-seen track within the grace window we keep
        them all (the spurious detection is dropped rather than displacing a good id)."""
        bucket = self._tracks[alliance]
        if len(bucket) >= MAX_PER_ALLIANCE:
            victim = self._weakest_evictable(bucket)
            if victim is None:
                return  # all three are healthy & seen -> ignore the extra detection
            bucket.remove(victim)
        bucket.append(_CVTrack(
            self._next_id, alliance, det, frame_idx, dt=self.dt,
            q_pos=self.q_pos, q_vel=self.q_vel, r_meas=self.r_meas))
        self._next_id += 1

    def _weakest_evictable(self, bucket: list[_CVTrack]) -> Optional[_CVTrack]:
        """The track most deserving of eviction: prefer ones coasting beyond the grace
        window; among candidates pick fewest hits, then longest unseen. A track that is
        seen this frame, or still within its coast-grace window, is protected — so a
        momentary occlusion never costs a robot its identity."""
        # ONLY evict a track that has coasted BEYOND the grace window (truly gone — e.g. a
        # mis-birthed spurious track). A track seen within the grace window is NEVER evicted,
        # so a brief occlusion can't cost a robot its id — this was the #1 ID-churn source
        # (a single missed frame + a spurious detection elsewhere re-minted the id).
        evictable = [t for t in bucket if t.time_since_update > self.coast_grace]
        if not evictable:
            return None
        # weakest = fewest hits, then most stale
        return min(evictable, key=lambda t: (t.hits, -t.time_since_update))

    # ───────────────────────────── emit ──────────────────────────────────────
    def _emit(self, t: _CVTrack) -> dict:
        # confident == seen recently (within the confident-grace window). A track coasting
        # past that grace is still emitted (id continuity) but confident=False so a consumer
        # can choose not to count it as a measured robot — we never DROP it within
        # drop_after, but we also never CLAIM it as confidently present without evidence.
        return {
            "id": t.tid,
            "alliance": t.alliance,
            "xyxy": t.xyxy,
            "centroid": t.centroid,
            "est": t.est,
            "confident": t.time_since_update <= self.confident_grace,
            "hits": t.hits,
            "age": t.age,
        }

    # ───────────────────────────── introspection ─────────────────────────────
    @property
    def n_tracks(self) -> int:
        return sum(len(b) for b in self._tracks.values())

    def tracks_for(self, alliance: Alliance) -> list[_CVTrack]:
        return list(self._tracks.get(alliance, []))
