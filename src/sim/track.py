"""
track.py — persistent multi-object tracking for the FRC REBUILT (2026) playoff
match shot from the single static wide camera (Upper Bracket R4 M11, 640x360).

Two trackers, both classical (OpenCV/numpy, deterministic):

RobotTracker
------------
Assigns STABLE integer IDs to the up-to-6 alliance robots across the whole wide
segment. Each track carries a constant-velocity Kalman filter (state = [cx, cy,
vx, vy], measurement = [cx, cy]) plus its last bbox size. Per frame:

  1. PREDICT every live track one step.
  2. ASSOCIATE detections to live predictions greedily by a cost that blends IOU
     (against the predicted box) and centroid distance, gated by a max distance
     and softly biased so that a detection of the SAME alliance is preferred.
  3. UPDATE matched tracks (Kalman correct + bbox EMA + alliance vote), spawn
     new tracks for unmatched detections, and COAST unmatched tracks on their
     prediction for up to ``max_age`` frames so an ID survives brief occlusion /
     a missed detection and is re-acquired when the robot reappears nearby.
  4. GRAVEYARD RE-ID: recently-deleted tracks are kept in a graveyard for up
     to ``reacquire_frames`` frames. When a new detection cannot match any live
     track, try to revive the best graveyard entry of the same alliance.

     Key improvement over the naive version: the graveyard entry stores the
     track's last Kalman velocity so the predicted re-emergence position is
     extrapolated forward by the full gap, not anchored to the stale last-seen
     position. The search gate also scales with the square root of the gap so
     a robot that was occluded for 90 frames gets a wider gate than one that
     was occluded for 5 frames. Appearance similarity (bumper hue histogram,
     from reid.py) is used as a tie-breaker when multiple graveyard entries of
     the same alliance could match.

Alliance per ID is fixed by MAJORITY VOTE over every frame the track was
measured (a momentary mis-classified bumper colour cannot flip a settled ID).

Output of update(): only CONFIRMED tracks (>= min_hits measurements and
currently visible) are returned, so transient false detections don't get IDs in
the public stream — but their internal ids are still stable.

BallTracker
-----------
Lighter IOU/centroid tracker for the yellow FUEL balls (no bbox, just centroid
+ radius). Maintains a velocity estimate and flags a track as a 'shot' when its
speed crosses a threshold (a fired ball moves much faster than a rolling/at-rest
ball). Ball IDs churn (balls are many, identical and frequently occluded); the
useful product is the per-track velocity and the shot flag.

Interface (exact):
  class RobotTracker:
      update(frame_idx:int, detections:list) ->
          [{'id':int,'bbox':(x,y,w,h),'centroid':(cx,cy),'alliance':'blue'|'red'}]
  class BallTracker:
      update(frame_idx:int, detections:list) ->
          [{'id':int,'centroid':(cx,cy),'r':int,'vel':(vx,vy),'shot':bool}]
"""

from __future__ import annotations

import math
import numpy as np


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix = max(ax, bx)
    iy = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix)
    ih = max(0.0, iy2 - iy)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Constant-velocity Kalman filter (state [x, y, vx, vy], meas [x, y])
# ---------------------------------------------------------------------------
class _KalmanCV:
    def __init__(self, cx: float, cy: float, dt: float = 1.0):
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=np.float64)
        # State covariance: confident in position, uncertain in velocity.
        self.P = np.diag([10.0, 10.0, 1e3, 1e3])
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)
        # Process noise: allow modest acceleration (robots change direction).
        q = 2.0
        self.Q = np.diag([0.25, 0.25, q, q])
        # Measurement noise: detection centroid jitter.
        self.R = np.diag([6.0, 6.0])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0], self.x[1]

    def correct(self, cx: float, cy: float):
        z = np.array([cx, cy], dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    @property
    def pos(self):
        return float(self.x[0]), float(self.x[1])

    @property
    def vel(self):
        return float(self.x[2]), float(self.x[3])


# ---------------------------------------------------------------------------
# Robot track
# ---------------------------------------------------------------------------
class _RobotTrack:
    __slots__ = ("id", "kf", "bw", "bh", "alliance", "votes",
                 "hits", "age", "time_since_update", "last_frame",
                 "history", "last_seen_pos", "hue_hist")

    def __init__(self, tid: int, det: dict, frame_idx: int,
                 frame=None):
        cx, cy = det["centroid"]
        x, y, w, h = det["bbox"]
        self.id = tid
        self.kf = _KalmanCV(cx, cy)
        self.bw = float(w)
        self.bh = float(h)
        self.alliance = det["alliance"]
        self.votes = {"blue": 0, "red": 0}
        self.votes[det["alliance"]] += 1
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.last_frame = frame_idx
        self.history = [(cx, cy)]
        self.last_seen_pos = (cx, cy)
        self.hue_hist = _bumper_hue_hist(frame, det["bbox"]) if frame is not None else None

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, det: dict, frame_idx: int, frame=None):
        cx, cy = det["centroid"]
        _, _, w, h = det["bbox"]
        self.kf.correct(cx, cy)
        # EMA the bbox size so the predicted box tracks scale changes smoothly.
        a = 0.4
        self.bw = (1 - a) * self.bw + a * float(w)
        self.bh = (1 - a) * self.bh + a * float(h)
        self.votes[det["alliance"]] += 1
        self.alliance = "blue" if self.votes["blue"] >= self.votes["red"] else "red"
        self.hits += 1
        self.time_since_update = 0
        self.last_frame = frame_idx
        self.last_seen_pos = (float(cx), float(cy))
        px, py = self.kf.pos
        self.history.append((px, py))
        if len(self.history) > 64:
            self.history.pop(0)
        # EMA update appearance histogram
        if frame is not None:
            new_hist = _bumper_hue_hist(frame, det["bbox"])
            if new_hist is not None and new_hist.sum() > 0:
                if self.hue_hist is None or self.hue_hist.sum() == 0:
                    self.hue_hist = new_hist
                else:
                    blended = 0.7 * self.hue_hist + 0.3 * new_hist
                    s = blended.sum()
                    self.hue_hist = (blended / s).astype(np.float32) if s > 0 else blended

    def pred_bbox(self):
        cx, cy = self.kf.pos
        return (cx - self.bw / 2.0, cy - self.bh / 2.0, self.bw, self.bh)


# ---------------------------------------------------------------------------
# Bumper hue histogram for appearance-based re-ID
# ---------------------------------------------------------------------------
_HUE_BINS = 12
_BUMPER_LOWER_FRAC = 0.30
_SAT_FLOOR = 80
_VAL_FLOOR = 70


def _bumper_hue_hist(frame, bbox):
    """Normalized Hue histogram of vivid bumper pixels in lower band of bbox.

    Returns float32 array of length _HUE_BINS summing to 1, or None if frame
    is None or patch has no vivid pixels.
    """
    if frame is None:
        return None
    try:
        import cv2
    except ImportError:
        return None
    h_img, w_img = frame.shape[:2]
    x, y, w, h = [int(v) for v in bbox]
    x0 = max(0, x)
    y0 = max(0, y + int(_BUMPER_LOWER_FRAC * h))
    x1 = min(w_img, x + w)
    y1 = min(h_img, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = frame[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1]
    val = hsv[..., 2]
    vivid = (sat >= _SAT_FLOOR) & (val >= _VAL_FLOOR)
    if int(vivid.sum()) < 3:
        return None
    hue = hsv[..., 0][vivid].astype(np.float32)
    hist, _ = np.histogram(hue, bins=_HUE_BINS, range=(0.0, 180.0))
    hist = hist.astype(np.float32)
    s = float(hist.sum())
    if s > 0:
        hist /= s
    return hist


def _hist_similarity(a, b) -> float:
    """Histogram intersection similarity in [0,1]; returns 0.5 if either is None."""
    if a is None or b is None:
        return 0.5
    sa, sb = float(a.sum()), float(b.sum())
    if sa <= 0.0 or sb <= 0.0:
        return 0.5
    return float(np.minimum(a, b).sum())


# ---------------------------------------------------------------------------
# Graveyard entry: preserves key track state after deletion for re-ID
# ---------------------------------------------------------------------------
class _Grave:
    """Snapshot of a deleted track, held for re-acquisition.

    Stores the last Kalman velocity so the predicted re-emergence position can
    be extrapolated forward by the full gap duration (not stuck at last_seen).
    """
    __slots__ = ("id", "pos", "vel", "bw", "bh", "alliance", "votes",
                 "hits", "died_frame", "history", "hue_hist")

    def __init__(self, tr: _RobotTrack, frame_idx: int):
        self.id = tr.id
        self.pos = tr.last_seen_pos
        self.vel = tr.kf.vel          # Kalman velocity at death
        self.bw = tr.bw
        self.bh = tr.bh
        self.alliance = tr.alliance
        self.votes = dict(tr.votes)
        self.hits = tr.hits
        self.died_frame = frame_idx
        self.history = list(tr.history)
        self.hue_hist = tr.hue_hist   # appearance signature

    def predicted_pos(self, frame_idx: int):
        """Extrapolate position from death to ``frame_idx`` using stored velocity."""
        gap = frame_idx - self.died_frame
        px = self.pos[0] + self.vel[0] * gap
        py = self.pos[1] + self.vel[1] * gap
        return float(px), float(py)

    def revive(self, det: dict, frame_idx: int) -> _RobotTrack:
        """Resurrect this entry as a live track, seeded from the detection."""
        cx, cy = det["centroid"]
        x, y, w, h = det["bbox"]
        tr = object.__new__(_RobotTrack)
        tr.id = self.id
        tr.kf = _KalmanCV(cx, cy)
        tr.bw = (self.bw + float(w)) / 2.0
        tr.bh = (self.bh + float(h)) / 2.0
        tr.alliance = self.alliance
        tr.votes = dict(self.votes)
        tr.votes[det["alliance"]] += 1
        tr.alliance = "blue" if tr.votes["blue"] >= tr.votes["red"] else "red"
        tr.hits = self.hits + 1
        tr.age = 1
        tr.time_since_update = 0
        tr.last_frame = frame_idx
        tr.last_seen_pos = (float(cx), float(cy))
        tr.history = list(self.history)
        tr.history.append((float(cx), float(cy)))
        if len(tr.history) > 64:
            tr.history.pop(0)
        tr.hue_hist = self.hue_hist
        return tr


# ---------------------------------------------------------------------------
# Robot tracker
# ---------------------------------------------------------------------------

# Base gate for live-track association (px)
_MAX_DIST_DEFAULT = 90.0
# Minimum gate for graveyard re-acquisition (px)
_REACQUIRE_DIST_MIN = 80.0
# Maximum gate for graveyard re-acquisition (px)
_REACQUIRE_DIST_MAX = 280.0
# Per-frame-of-gap scaling factor for the dynamic gate (px / sqrt(frame))
_REACQUIRE_SCALE = 22.0


class RobotTracker:
    """Persistent IOU + Kalman tracker for the alliance robots.

    Parameters
    ----------
    max_age : int
        Frames a track may coast unmatched before deletion (occlusion budget).
    min_hits : int
        Measurements before a track is reported publicly (confirmation).
    max_dist : float
        Hard gate on predicted-to-detection centroid distance (pixels).
    iou_weight : float
        Blend between IOU and distance in the association cost.
    max_tracks : int
        Cap on simultaneously CONFIRMED tracks reported (6 robots on field).
    reacquire_frames : int
        How long (frames) a deleted track stays in the graveyard for re-ID.
    reacquire_dist : float
        Base distance (px) for graveyard re-acquisition gate; the actual gate
        is dynamically scaled by sqrt(gap) so long absences get wider search.
    """

    def __init__(self, max_age: int = 60, min_hits: int = 4,
                 max_dist: float = _MAX_DIST_DEFAULT, iou_weight: float = 0.5,
                 max_tracks: int = 6,
                 reacquire_frames: int = 600,
                 reacquire_dist: float = 220.0):
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_dist = max_dist
        self.iou_weight = iou_weight
        self.max_tracks = max_tracks
        self.reacquire_frames = reacquire_frames
        self.reacquire_dist = reacquire_dist
        self.tracks: list[_RobotTrack] = []
        self.graveyard: list[_Grave] = []
        self._next_id = 0
        self.id_switches = 0  # informational; bumped on alliance-vote flips

    def _match(self, dets: list, tracks: list, max_dist: float):
        """Greedy association between tracks and detections.

        Returns (matches, un_track_idxs, un_det_idxs).
        """
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))

        n_t, n_d = len(tracks), len(dets)
        cost = np.full((n_t, n_d), 1e6, dtype=np.float64)
        for ti, tr in enumerate(tracks):
            pbox = tr.pred_bbox()
            px, py = tr.kf.pos
            for di, d in enumerate(dets):
                dcx, dcy = d["centroid"]
                dist = float(np.hypot(px - dcx, py - dcy))
                if dist > max_dist:
                    continue
                iou = _iou(pbox, d["bbox"])
                # Lower cost = better.
                c = (1 - self.iou_weight) * (dist / max_dist) \
                    + self.iou_weight * (1.0 - iou)
                # Soft alliance bias.
                if d["alliance"] != tr.alliance:
                    c += 0.25
                cost[ti, di] = c

        matches = []
        used_t, used_d = set(), set()
        flat = [(cost[ti, di], ti, di)
                for ti in range(n_t) for di in range(n_d)
                if cost[ti, di] < 1e5]
        flat.sort(key=lambda z: (z[0], z[1], z[2]))
        for c, ti, di in flat:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            matches.append((ti, di))
        un_t = [ti for ti in range(n_t) if ti not in used_t]
        un_d = [di for di in range(n_d) if di not in used_d]
        return matches, un_t, un_d

    def _reacquire(self, unmatched_dets: list, detections: list,
                   frame_idx: int, frame=None) -> list[int]:
        """Try to re-ID unmatched detections against the graveyard.

        Key changes vs. old version:
        - Predicted position uses stored Kalman velocity extrapolated over
          the full gap (dynamic prediction, not stale last-seen).
        - Gate is velocity-extrapolated distance + dynamic margin that grows
          with sqrt(gap) so longer absences tolerate wider excursions.
        - Appearance similarity (bumper hue histogram) breaks ties when
          multiple graveyard entries of the same alliance are in range.
        - The best graveyard match is chosen globally (minimise cost over all
          un_det x graveyard pairs) rather than per-detection greedily, to
          avoid two detections both claiming the same grave.

        Returns the subset of unmatched_det indices that were re-acquired
        (and thus should NOT spawn fresh tracks).
        """
        if not self.graveyard or not unmatched_dets:
            return []

        # Build cost matrix: rows = unmatched dets, cols = graveyard entries
        n_d = len(unmatched_dets)
        n_g = len(self.graveyard)
        cost = np.full((n_d, n_g), 1e9, dtype=np.float64)

        for row_i, di in enumerate(unmatched_dets):
            d = detections[di]
            dcx, dcy = d["centroid"]
            for gi, grave in enumerate(self.graveyard):
                # Hard alliance gate
                if grave.alliance != d["alliance"]:
                    continue
                gap = frame_idx - grave.died_frame
                # Predicted position: extrapolate with stored velocity
                ppx, ppy = grave.predicted_pos(frame_idx)
                pred_dist = float(np.hypot(ppx - dcx, ppy - dcy))

                # Dynamic gate: base dist + scale * sqrt(gap), capped
                dyn_gate = min(
                    _REACQUIRE_DIST_MAX,
                    max(_REACQUIRE_DIST_MIN,
                        self.reacquire_dist + _REACQUIRE_SCALE * math.sqrt(gap))
                )

                if pred_dist > dyn_gate:
                    # Also check last-seen pos for slow/stopped robots
                    lx, ly = grave.pos
                    last_dist = float(np.hypot(lx - dcx, ly - dcy))
                    if last_dist > dyn_gate:
                        continue
                    # Use min of predicted and last-seen distances
                    pred_dist = min(pred_dist, last_dist)

                # Position score: 1 at prediction, 0 at gate edge
                pos_score = max(0.0, 1.0 - pred_dist / dyn_gate)

                # Appearance score (bumper hue histogram similarity)
                if frame is not None and grave.hue_hist is not None:
                    det_hist = _bumper_hue_hist(frame, d["bbox"])
                    app_score = _hist_similarity(grave.hue_hist, det_hist)
                else:
                    app_score = 0.5  # neutral when no appearance info

                # Gap penalty: slight penalty for very long absences
                gap_pen = min(0.3, gap * 0.001)

                # Blended cost (lower = better match)
                score = 0.6 * pos_score + 0.4 * app_score - gap_pen
                cost[row_i, gi] = 1.0 - score  # convert to cost

        # Global greedy matching (best cost first)
        reacquired = []
        used_graves = set()
        used_dets = set()

        flat = [(cost[ri, gi], ri, gi)
                for ri in range(n_d) for gi in range(n_g)
                if cost[ri, gi] < 0.99]
        flat.sort(key=lambda z: z[0])

        for c, row_i, gi in flat:
            if row_i in used_dets or gi in used_graves:
                continue
            # Threshold: cost < 0.65 means blended score > 0.35
            if c >= 0.65:
                continue
            di = unmatched_dets[row_i]
            grave = self.graveyard[gi]
            revived = grave.revive(detections[di], frame_idx)
            self.tracks.append(revived)
            used_graves.add(gi)
            used_dets.add(row_i)
            reacquired.append(di)

        # Remove consumed grave entries
        self.graveyard = [g for gi, g in enumerate(self.graveyard)
                          if gi not in used_graves]
        return reacquired

    def update(self, frame_idx: int, detections: list, frame=None) -> list:
        """Update tracker with new detections.

        Parameters
        ----------
        frame_idx : int
            Current frame index.
        detections : list
            Output of detect_robots() — list of dicts with 'bbox', 'centroid',
            'alliance'.
        frame : ndarray or None
            The current BGR frame (optional). If provided, bumper appearance
            histograms are computed and used for graveyard re-ID.
        """
        # 1. Predict all live tracks
        for tr in self.tracks:
            tr.predict()

        # 2. Associate detections to live tracks
        matches, un_t, un_d = self._match(detections, self.tracks, self.max_dist)

        # 3. Update matched tracks
        for ti, di in matches:
            tr = self.tracks[ti]
            prev_alliance = tr.alliance
            tr.update(detections[di], frame_idx, frame)
            if tr.hits > self.min_hits and tr.alliance != prev_alliance:
                self.id_switches += 1

        # 4. Age out dead tracks -> put them in the graveyard
        alive = []
        for tr in self.tracks:
            if tr.time_since_update > self.max_age:
                if tr.hits >= self.min_hits:  # only worthwhile tracks are grave'd
                    self.graveyard.append(_Grave(tr, frame_idx))
            else:
                alive.append(tr)
        self.tracks = alive

        # 5. Expire old graveyard entries
        self.graveyard = [g for g in self.graveyard
                          if (frame_idx - g.died_frame) <= self.reacquire_frames]

        # 6. Re-acquire unmatched detections against the graveyard (with velocity
        #    extrapolation + appearance for disambiguation)
        reacquired = self._reacquire(un_d, detections, frame_idx, frame)
        remaining_un_d = [di for di in un_d if di not in reacquired]

        # 7. Spawn fresh tracks for truly unmatched detections.
        # Tight cap: only allow new spawns when the confirmed track count is
        # below max_tracks. Count confirmed tracks (hits >= min_hits) to avoid
        # letting transient fragments prevent real new tracks from seeding.
        confirmed_live = sum(1 for tr in self.tracks if tr.hits >= self.min_hits)
        live = len(self.tracks)
        # Allow at most 1 in-flight unconfirmed candidate at a time.
        unconfirmed_live = live - confirmed_live
        for di in remaining_un_d:
            if confirmed_live >= self.max_tracks:
                break  # all 6 robots accounted for — don't spawn anything new
            if unconfirmed_live >= 1:
                break  # already have one candidate cooking
            self.tracks.append(_RobotTrack(self._next_id, detections[di],
                                           frame_idx, frame))
            self._next_id += 1
            live += 1
            unconfirmed_live += 1

        # 8. Report confirmed + currently-visible tracks; keep at most max_tracks,
        #    the longest-lived (most hits) winning ties so stable IDs dominate.
        visible = [tr for tr in self.tracks
                   if tr.time_since_update == 0
                   and tr.hits >= self.min_hits]
        visible.sort(key=lambda t: (-t.hits, t.id))
        visible = visible[: self.max_tracks]
        visible.sort(key=lambda t: t.id)

        out = []
        for tr in visible:
            cx, cy = tr.kf.pos
            bbox = (int(round(cx - tr.bw / 2.0)), int(round(cy - tr.bh / 2.0)),
                    int(round(tr.bw)), int(round(tr.bh)))
            out.append({
                "id": tr.id,
                "bbox": bbox,
                "centroid": (float(cx), float(cy)),
                "alliance": tr.alliance,
            })
        return out

    # --- introspection helpers for self-verify / reporting -----------------
    def all_tracks(self):
        return self.tracks


# ---------------------------------------------------------------------------
# Ball track / tracker
# ---------------------------------------------------------------------------
class _BallTrack:
    __slots__ = ("id", "kf", "r", "hits", "time_since_update",
                 "last_frame", "shot", "max_speed")

    def __init__(self, tid: int, det: dict, frame_idx: int):
        cx, cy = det["centroid"]
        self.id = tid
        self.kf = _KalmanCV(cx, cy)
        self.r = int(det.get("r", 4))
        self.hits = 1
        self.time_since_update = 0
        self.last_frame = frame_idx
        self.shot = False
        self.max_speed = 0.0

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1

    def update(self, det: dict, frame_idx: int, shot_speed: float):
        cx, cy = det["centroid"]
        self.kf.correct(cx, cy)
        self.r = int(det.get("r", self.r))
        self.hits += 1
        self.time_since_update = 0
        self.last_frame = frame_idx
        vx, vy = self.kf.vel
        sp = float(np.hypot(vx, vy))
        self.max_speed = max(self.max_speed, sp)
        if sp >= shot_speed:
            self.shot = True


class BallTracker:
    """IOU/centroid tracker for FUEL balls; flags fast 'shot' tracks.

    Ball detections have no bbox (centroid + radius), so association is purely
    by gated centroid distance.
    """

    def __init__(self, max_age: int = 6, max_dist: float = 26.0,
                 min_hits: int = 2, shot_speed: float = 9.0):
        self.max_age = max_age
        self.max_dist = max_dist
        self.min_hits = min_hits
        self.shot_speed = shot_speed
        self.tracks: list[_BallTrack] = []
        self._next_id = 0

    def _match(self, dets: list):
        tracks = self.tracks
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        flat = []
        for ti, tr in enumerate(tracks):
            px, py = tr.kf.pos
            for di, d in enumerate(dets):
                dcx, dcy = d["centroid"]
                dist = float(np.hypot(px - dcx, py - dcy))
                if dist <= self.max_dist:
                    flat.append((dist, ti, di))
        flat.sort(key=lambda z: (z[0], z[1], z[2]))
        used_t, used_d, matches = set(), set(), []
        for dist, ti, di in flat:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            matches.append((ti, di))
        un_t = [ti for ti in range(len(tracks)) if ti not in used_t]
        un_d = [di for di in range(len(dets)) if di not in used_d]
        return matches, un_t, un_d

    def update(self, frame_idx: int, detections: list) -> list:
        for tr in self.tracks:
            tr.predict()
        matches, un_t, un_d = self._match(detections)
        for ti, di in matches:
            self.tracks[ti].update(detections[di], frame_idx, self.shot_speed)
        for di in un_d:
            self.tracks.append(_BallTrack(self._next_id, detections[di],
                                          frame_idx))
            self._next_id += 1
        self.tracks = [tr for tr in self.tracks
                       if tr.time_since_update <= self.max_age]
        out = []
        for tr in self.tracks:
            if tr.time_since_update != 0 or tr.hits < self.min_hits:
                continue
            cx, cy = tr.kf.pos
            vx, vy = tr.kf.vel
            out.append({
                "id": tr.id,
                "centroid": (float(cx), float(cy)),
                "r": int(tr.r),
                "vel": (float(vx), float(vy)),
                "shot": bool(tr.shot),
            })
        return out


# ---------------------------------------------------------------------------
# CLI self-check: run over the wide segment, render boxes+IDs+trails, and
# report stable-track / id-switch statistics.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import os
    import cv2

    try:
        from . import detect as _detect  # type: ignore
    except Exception:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import detect as _detect  # type: ignore

    video = sys.argv[1] if len(sys.argv) > 1 else "data/sim/match_full.mp4"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "track_annot"
    os.makedirs(out_dir, exist_ok=True)

    WIDE_END = 4170  # from shots.json
    print("Building background over wide segment...")
    bg = _detect.build_background(video, 0.0, 139.0)

    cap = cv2.VideoCapture(video)
    rt = RobotTracker()

    # palette per stable id
    rng = np.random.RandomState(0)
    palette = {}

    def color_for(i):
        if i not in palette:
            palette[i] = tuple(int(c) for c in rng.randint(60, 255, 3))
        return palette[i]

    save_frames = set(int(x) for x in
                      [300, 600, 1200, 1800, 2400, 3000, 3600, 4100])
    trails = {}  # id -> list of centroids

    # lifetime stats
    track_life = {}   # id -> [first_frame, last_frame, hits]
    confirmed_per_frame = []

    fi = 0
    while fi < WIDE_END:
        ok, fr = cap.read()
        if not ok:
            break
        dets = _detect.detect_robots(fr, bg)
        tracked = rt.update(fi, dets, fr)
        confirmed_per_frame.append(len(tracked))

        for t in tracked:
            tid = t["id"]
            cx, cy = t["centroid"]
            trails.setdefault(tid, []).append((int(cx), int(cy)))
            if len(trails[tid]) > 25:
                trails[tid].pop(0)
            st = track_life.get(tid)
            if st is None:
                track_life[tid] = [fi, fi, 1]
            else:
                st[1] = fi
                st[2] += 1

        if fi in save_frames:
            ann = fr.copy()
            for t in tracked:
                x, y, w, h = t["bbox"]
                tid = t["id"]
                col = color_for(tid)
                cv2.rectangle(ann, (x, y), (x + w, y + h), col, 2)
                lbl = f"{tid}:{t['alliance'][0].upper()}"
                cv2.putText(ann, lbl, (x, max(9, y - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
                pts = trails.get(tid, [])
                for k in range(1, len(pts)):
                    cv2.line(ann, pts[k - 1], pts[k], col, 1)
            cv2.putText(ann, f"f{fi} tracks:{len(tracked)}", (6, 350),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            big = cv2.resize(ann, None, fx=2.0, fy=2.0,
                             interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(out_dir, f"f{fi:05d}.png"), big)
        fi += 1
    cap.release()

    # ---- report ----
    long_lived = {i: v for i, v in track_life.items()
                  if (v[1] - v[0]) >= 90}  # alive >= ~3s span
    print(f"total ids ever spawned (reported): {len(track_life)}")
    print(f"long-lived tracks (span >= 90 frames): {len(long_lived)}")
    for i, v in sorted(long_lived.items(),
                       key=lambda kv: -(kv[1][1] - kv[1][0]))[:12]:
        span = v[1] - v[0]
        print(f"  id {i:3d}: frames {v[0]:4d}-{v[1]:4d} span {span:4d} "
              f"hits {v[2]:4d}")
    cpf = np.array(confirmed_per_frame)
    print(f"confirmed tracks/frame: mean {cpf.mean():.2f} "
          f"median {np.median(cpf):.0f} max {cpf.max()}")
    print(f"alliance-vote flips after confirm: {rt.id_switches}")
    print("annotated frames ->", out_dir)
