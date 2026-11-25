"""scout_pipeline.py — the ONE source of truth for the synchronous per-frame scouting
pipeline.

This class is the behavior-preserving extraction of the per-frame body that used to live
inline in ``scripts/run_scout.py``'s ``run_sync``. Both the file/demo renderer
(``run_sync``) and the genuinely-live app (``scripts/live_app.py``) now drive the SAME
:class:`ScoutPipeline`, so there is exactly one place the per-frame logic lives.

Everything the ``--sync`` path did per frame is preserved here, bit-for-bit:

  * AprilTag pose refresh (~4 Hz) shared by robots/balls/blob detector
  * ONE multi-class YOLO model @960 → split class 0 (ball) / class 1 (robot)
  * classical color-prior blob fuel detector fused with the net via WBF
  * the tight FIELD_POLY ball crowd-gate (kills crowd-circle FPs)
  * the dense-pile AREA estimate + magenta contours + "FUEL ~N (area est.)" headline
  * the WIDER ROBOT_POLY robot gate (floor-contact point), crowd_rejected counter
  * robot ByteTrack + the in-MatchRunner 6-robot prior
  * robot field_xy projection INCLUDING the A6.3 edge-pose clamp (tag + px fallback)
  * NET-AIRBORNE flight path (net ball dets in the arc band → FlightTracker → SHOT marker)
  * periodic HUD OCR + the fuel-pulse → overlay-score attribution
  * the bin_scorer / HUB funnel polygons drawn on the overlay
  * the phase state machine via MatchRunner.process

The class owns NO video IO: it is constructed with a fixed ``frame_wh`` + ``fps`` and its
:meth:`process` takes a decoded BGR frame + its frame index and returns the rendered
annotated BGR frame. The caller owns capture, the writer, export, and progress printing.

``process`` ALWAYS renders the overlay (the live app needs it every frame; the renderer is
deterministic and never touches export state, so this is identical to the old "render only
when a writer exists" behavior for the bytes that get written / exported).
"""
from __future__ import annotations

import json as _json
import os
import time as _time
from collections import Counter

import numpy as _np

from frcscout.io.types import Detection, FrameState, OverlayState, Track


# Pseudo image->meter scale for the no-AprilTag-pose broadcast case. The absolute value
# only sets metric UNITS + the intake floor's px size; the intake catchment in px is
# ~intake_k * robot_width_px and is scale-invariant, so proximity intake is robust.
MPP = 0.01  # ~1 cm / px


class _BallTracker:
    """Minimal persistent-ID tracker for inventory balls: greedy nearest-centroid
    association within a gate, fresh ids for unmatched detections, cull after a gap.
    Emits only CURRENTLY-detected balls (with stable ids) so the scoring tracker sees a
    ball 'vanish' (its id drops out) when consumed — that's the intake signal."""

    def __init__(self, gate_px: float = 70.0, max_gap: int = 8):
        self.gate = float(gate_px)
        self.max_gap = int(max_gap)
        self._tracks: dict = {}   # tid -> {"c": (x,y), "last": frame_idx}
        self._next = 0

    @staticmethod
    def _c(xyxy):
        return ((xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0)

    def update(self, dets, frame_idx: int):
        import math
        used: set = set()
        out = []  # (tid, det)
        for d in dets:
            cx, cy = self._c(d.xyxy)
            best, bd = None, self.gate
            for tid, tk in self._tracks.items():
                if tid in used:
                    continue
                dist = math.hypot(cx - tk["c"][0], cy - tk["c"][1])
                if dist < bd:
                    bd, best = dist, tid
            if best is None:
                best = self._next
                self._next += 1
            used.add(best)
            self._tracks[best] = {"c": (cx, cy), "last": frame_idx}
            out.append((best, d))
        for tid in [t for t, tk in self._tracks.items()
                    if frame_idx - tk["last"] > self.max_gap]:
            del self._tracks[tid]
        return out


def _project_h(H, x, y):
    """Apply a 3x3 image->ground homography to an image point -> (fx, fy) field meters."""
    if H is None:
        return None
    import numpy as np
    v = np.asarray(H, dtype=float) @ np.array([float(x), float(y), 1.0])
    if abs(v[2]) < 1e-9:
        return None
    return (float(v[0] / v[2]), float(v[1] / v[2]))


def _yellow_frac(frame, xyxy) -> float:
    """Fraction of FUEL-yellow pixels in a box. The airborne 'arc band' (y 0.12-0.33)
    overlaps the STANDS, so a net false-ball on a crowd head/shirt would otherwise become
    a phantom SHOT. Real fuel is bright yellow; crowd objects are not -> gate flight
    detections on this. Cheap HSV check on the (small) crop."""
    import cv2
    import numpy as np
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return 0.0
    crop = frame[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    yellow = (h >= 18) & (h <= 38) & (s > 80) & (v > 80)
    return float(yellow.mean())


class ScoutPipeline:
    """The reusable synchronous scouting pipeline (one source of truth).

    Parameters
    ----------
    cfg : Config
        Loaded config (``frcscout.io.config.Config``). Device must already be forced onto
        the detectors if desired (``run_scout.force_device``); ``device`` below is the
        per-predict device string.
    frame_wh : tuple[int, int]
        ``(W, H)`` of the incoming frames. Drives all the image-space polygons.
    fps : float
        Source fps — drives pose/HUD refresh cadence and the metric clocks.
    device : str
        Per-frame YOLO predict device (e.g. ``"cpu"``, ``"intel:cpu"``).
    a63_dump : str | None
        Optional path to append per-robot raw+clamped field_xy JSONL (the A6.3 verification
        hook). When None, honors the ``A63_DUMP`` env var for parity with the old run_sync.

    Attributes (refreshed every :meth:`process` call)
    ----------
    last_state : FrameState
        The post-``MatchRunner.process`` state (``res["state"]``).
    last_res : dict
        The full MatchRunner result (``{"state","six","phase"}``).
    last_pile : object
        The dense-pile estimate for the frame.
    """

    def __init__(self, cfg, frame_wh, fps, device, a63_dump: str | None = None,
                 render: bool = True):
        from ultralytics import YOLO

        from frcscout.track.bytetrack import RobotTracker
        from frcscout.track.ball_kalman import FlightTracker
        from frcscout.detect.flight import FlightDetector
        from frcscout.detect.ball_blob import BallBlobDetector
        from frcscout.geometry.field import FieldModel
        from frcscout.geometry.apriltag import PoseEstimator
        from frcscout.pipeline.match_runner import MatchRunner

        self.cfg = cfg
        self.device = device
        self.render_enabled = bool(render)
        W, H = int(frame_wh[0]), int(frame_wh[1])
        self.W, self.H = W, H
        self.fps = float(fps) if fps else 60.0

        self.field = FieldModel(cfg.get("field_config"))
        # A6.3 verification hook: when set, append per-robot raw + clamped field_xy_m as
        # JSONL so the visual-verifier can measure off-field counts. Off by default.
        dump_path = a63_dump if a63_dump is not None else os.environ.get("A63_DUMP")
        self._A63_DUMP = open(dump_path, "w") if dump_path else None

        # ONE multi-class model serves both balls (cls 0) and robots (cls 1).
        self.model = YOLO(cfg.get("detectors.ball_inventory.model"), task="detect")
        self.ball_imgsz = int(cfg.get("detectors.ball_inventory.imgsz", 960))
        self.ball_conf = float(cfg.get("detectors.ball_inventory.conf", 0.15))
        self.robot_conf = float(cfg.get("detectors.robot.conf", 0.12))
        self.gate = min(self.ball_conf, self.robot_conf)

        # Fixed field-region polygon (fractions of W,H) for the main pane — the play area
        # between the alliance structures, below the stands, above the pane divider. The
        # few clustered AprilTags can't give a reliable back-projected field extent, so on
        # this ~fixed wide field shot a fixed image polygon is the robust precision tool
        # that kills crowd-circle ball FPs. (Override via cfg 'field_poly_frac'.)
        _POLY = cfg.get("field_poly_frac") or [(0.07, 0.31), (0.93, 0.31), (0.99, 0.62), (0.01, 0.62)]
        self.FIELD_POLY = _np.array([(int(x * W), int(y * H)) for x, y in _POLY],
                                    dtype=_np.int32).reshape(-1, 1, 2)
        # Robots use a WIDER field polygon than balls. Robot false positives are filtered
        # downstream (bumper-colour alliance + the 6-robot prior), whereas ball FPs are not
        # — so the ball polygon stays tight (crowd-clean) while the robot polygon reaches
        # the field edges + near side to catch edge/near robots the tight ball polygon
        # clipped. Per-video via cfg 'robot_poly_frac'.
        _RPOLY = cfg.get("robot_poly_frac") or [(0.02, 0.25), (0.98, 0.25), (1.00, 0.64), (0.00, 0.64)]
        self.ROBOT_POLY = _np.array([(int(x * W), int(y * H)) for x, y in _RPOLY],
                                    dtype=_np.int32).reshape(-1, 1, 2)

        self.robot_tracker = RobotTracker(cfg.get("tracking.robot"), fps=self.fps)
        self.flight_det = FlightDetector(cfg.get("detectors.ball_flight"), self.field)
        self.flight_trk = FlightTracker(cfg.get("tracking.ball_flight"), fps=self.fps)
        self.ball_tracker = _BallTracker()
        self.blob = BallBlobDetector()                  # classical color-prior fuel detector (R2)
        self.pose_est = PoseEstimator(self.field)
        self.cur_pose = None
        self.MAIN_PANE = (0.0, float(cfg.get("main_pane_frac", 0.52)))   # top pane = main view
        self.pose_every = max(1, int(round(self.fps / 4.0)))            # ~4 Hz pose refresh
        self.pose_frames = 0

        # AprilTags ARE visible -> a REAL per-frame pose is recovered and used to project
        # real field_xy. Climb still needs a true HEIGHT z, which a single ground
        # homography (z=0) cannot give, so the monocular bumper-px climb heuristic stays
        # OFF (1e9) until multi-view triangulation supplies z; intake/hold/defense/positions
        # use the real xy. 1e9 (not just high): _climb_level uses growth = max(r, 1/r), so a
        # noisy bbox that momentarily SHRINKS yields a large growth that can exceed a merely-
        # high threshold once and latch the monotonic climb. 1e9 is truly unreachable.
        for _k in ("climb_px_l1_ratio", "climb_px_l2_ratio", "climb_px_l3_ratio"):
            cfg[_k] = 1e9
        self.runner = MatchRunner(field_model=self.field, hub_localizer=self.field,
                                  cfg=cfg, fps=self.fps)
        # Robots are already gated by ROBOT_POLY (image-space) BEFORE the tracker, so the
        # in-process MatchRunner pane gate would only DOUBLE-cut. Intentionally NOT calling
        # runner.set_frame_h(H): that leaves the in-process pane test disabled (y_main=inf)
        # so only the harmless field back-projection remains, and ROBOT_POLY stays authoritative.
        if getattr(self.runner.phase, "bin_scorer", None) is not None:
            self.runner.phase.bin_scorer.set_frame_wh(W, H)   # scale the HUB-opening polygons to px

        # Broadcast HUD reader (calibrated ROIs) — fills real teams/scores/fuel/timer. Read
        # every ~0.5s and vote the rosters over the match to beat transient OCR misreads.
        try:
            try:
                from hud_read import HudReader   # scripts/ on path (run_scout run dir)
            except ModuleNotFoundError:
                # Any non-scripts/ caller (e.g. live/run_live.py): hud_read lives in
                # <v2_root>/scripts. Add it to sys.path so the HUD reader works from any cwd.
                import sys as _sys
                from pathlib import Path as _Path
                _scripts = _Path(__file__).resolve().parents[3] / "scripts"
                if _scripts.is_dir() and str(_scripts) not in _sys.path:
                    _sys.path.insert(0, str(_scripts))
                from hud_read import HudReader
            self.hud = HudReader()
        except Exception as e:               # easyocr missing / model dl failed -> degrade
            print(f"[sync] HUD reader off ({type(e).__name__}: {e}); overlay stays empty", flush=True)
            self.hud = None
        self.blue_votes, self.red_votes = Counter(), Counter()
        self.cur_ov = OverlayState()
        self.hud_every = max(1, int(round(self.fps / 2.0)))   # ~2 Hz
        self.prev_fuel = {"blue": None, "red": None}

        # rolling diagnostics the driver may surface
        self.crowd_rejected = 0
        self.max_six = 0
        self.phase_hist: dict[str, int] = {}

        # last-frame outputs (live app + driver read these)
        self.last_state = None
        self.last_res = None
        self.last_pile = None
        self.last_ball_det_count = 0   # gated inventory ball detections this frame
        self.last_flight_count = 0     # flight tracker tracks this frame (pre render-filter)
        self._loop_i = 0   # internal counter for the %-cadence (pose/HUD), mirrors run_sync's `i`

        # ── LIVE-MODE cadence/fast knobs (default OFF -> run_scout --sync is byte-identical).
        # When `live_mode` is True, the HEAVY detectors only run on a sub-cadence and their
        # last outputs are CACHED; on the OFF frames only the cheap trackers (which predict
        # forward) + the overlay render run, so boxes/markers keep MOVING smoothly via
        # prediction at the display rate. `fast` additionally drops the heaviest classical
        # bits (blob + dense-pile) entirely for max speed (net-only balls).
        self.live_mode = False
        self.fast = False
        # cadences (in PROCESS calls): run the block when (_loop_i % every == 0).
        self.net_every = 1     # YOLO net (balls+robots)
        self.blob_every = 1    # classical blob detect + WBF fuse
        self.pile_every = 1    # dense-pile AREA estimate
        self.ocr_every = self.hud_every   # HUD OCR
        self.pose_every_live = self.pose_every  # AprilTag pose (overrides pose_every in live mode)

        # caches for the OFF frames (live mode)
        self._cache_net_ball_dets: list = []   # last net ball dets (raw, for flight band)
        self._cache_robot_dets: list = []      # last gated robot dets (for re-feed continuity)
        self._cache_pile = None                # last dense-pile estimate
        self._cache_blob_dets: list = []       # last classical blob dets
        self._cache_ball_dets: list = []       # last fused+gated inventory ball dets

        # Online static red/blue structure mask. For the first warmup window, count pixels
        # that persistently look like saturated alliance-color structure; once built, pass
        # the mask into the robot precision filter and stop doing per-frame mask work.
        self._robot_static_mask_warmup = int(
            cfg.get("detectors.robot.static_mask_warmup_frames", 120)
        )
        self._robot_static_mask_persistence = float(
            cfg.get("detectors.robot.static_mask_persistence", 0.60)
        )
        self._robot_static_mask_dilate = int(
            cfg.get("detectors.robot.static_mask_dilate", 5)
        )
        self._robot_static_mask_scale = float(
            cfg.get("detectors.robot.static_mask_scale", 0.25)
        )
        self._robot_static_counts = None
        self._robot_static_frames = 0
        self._robot_static_mask = None

        # profiling: SCOUT_PROFILE=1 turns on per-block timers; print after `prof_window` frames.
        self._profile = bool(int(os.environ.get("SCOUT_PROFILE", "0") or "0"))
        self._prof: dict[str, float] = {}      # block -> summed ms
        self._prof_n = 0                        # frames accumulated
        self._prof_window = int(os.environ.get("SCOUT_PROFILE_FRAMES", "40") or "40")
        self._prof_printed = False

        if bool(cfg.get("sync_performance.warmup_net", False)):
            self.model.predict(_np.zeros((H, W, 3), dtype=_np.uint8),
                               imgsz=self.ball_imgsz, conf=self.gate,
                               device=self.device, verbose=False)

    def configure_live(self, live_mode=None, fast=None, net_every=None, blob_every=None,
                       pile_every=None, ocr_every=None, pose_every=None):
        """Configure the live cadence/fast knobs (STEP 2). All args optional; only the ones
        passed are changed. Safe to call between frames (the live app's cadence selector uses
        it). Leaving everything default keeps the full per-frame pipeline (run_scout --sync)."""
        if live_mode is not None:
            self.live_mode = bool(live_mode)
        if fast is not None:
            self.fast = bool(fast)
        if net_every is not None:
            self.net_every = max(1, int(net_every))
        if blob_every is not None:
            self.blob_every = max(1, int(blob_every))
        if pile_every is not None:
            self.pile_every = max(1, int(pile_every))
        if ocr_every is not None:
            self.ocr_every = max(1, int(ocr_every))
        if pose_every is not None:
            self.pose_every_live = max(1, int(pose_every))

    def _update_robot_static_mask(self, frame):
        """Accumulate the first N processed frames into the online structure mask."""
        if self._robot_static_mask is not None:
            return
        if self._robot_static_mask_warmup <= 0:
            return
        if self._robot_static_frames >= self._robot_static_mask_warmup:
            return

        from frcscout.detect.robot_filter import (
            build_static_structure_mask,
            saturated_red_or_blue,
        )
        import cv2

        scale = min(1.0, max(0.05, self._robot_static_mask_scale))
        if scale < 1.0:
            small_w = max(1, int(round(self.W * scale)))
            small_h = max(1, int(round(self.H * scale)))
            sample = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
        else:
            sample = frame

        vote = saturated_red_or_blue(sample)
        if self._robot_static_counts is None or self._robot_static_counts.shape != vote.shape:
            self._robot_static_counts = _np.zeros(vote.shape, dtype=_np.uint16)
            self._robot_static_frames = 0
        self._robot_static_counts += vote
        self._robot_static_frames += 1

        if self._robot_static_frames >= self._robot_static_mask_warmup:
            mask = build_static_structure_mask(
                self._robot_static_counts,
                self._robot_static_frames,
                persistence=self._robot_static_mask_persistence,
                dilate=self._robot_static_mask_dilate,
            )
            if mask.shape[:2] != (self.H, self.W):
                mask = cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            self._robot_static_mask = mask

    def _t0(self):
        """perf_counter when profiling, else None (so the no-profile path has zero overhead)."""
        return _time.perf_counter() if self._profile else None

    def _acc(self, block: str, t0):
        """Accumulate elapsed ms for ``block`` since ``t0`` (no-op when not profiling)."""
        if t0 is not None:
            self._prof[block] = self._prof.get(block, 0.0) + (_time.perf_counter() - t0) * 1000.0

    def _maybe_print_profile(self):
        """Print the mean ms/block breakdown once ``_prof_window`` frames are accumulated."""
        if not self._profile or self._prof_printed:
            return
        self._prof_n += 1
        if self._prof_n < self._prof_window:
            return
        n = self._prof_n
        total = sum(self._prof.values()) / n
        print(f"\n===== SCOUT_PROFILE  (mean ms/block over {n} frames) =====", flush=True)
        order = ["static_mask", "pose", "net_predict", "blob_detect", "wbf_fuse", "pile_estimate",
                 "robot_gate_track", "flight", "hud_ocr", "phase_matchrunner", "render"]
        seen = set()
        for k in order + [k for k in self._prof if k not in order]:
            if k in self._prof and k not in seen:
                seen.add(k)
                ms = self._prof[k] / n
                print(f"  {k:<20s} {ms:8.2f} ms   ({100.0*ms/total if total else 0:5.1f}%)",
                      flush=True)
        print(f"  {'TOTAL':<20s} {total:8.2f} ms   -> {1000.0/total if total else 0:.2f} fps",
              flush=True)
        print("==========================================================\n", flush=True)
        self._prof_printed = True

    def process(self, frame, frame_idx: int):
        """Run ALL per-frame logic on ``frame`` (BGR) and return the annotated BGR overlay.

        ``frame_idx`` is the absolute frame index (``start + i`` in the old run_sync), used
        for tracker timing + diagnostics. The internal ``_loop_i`` counter (incremented per
        call) drives the modulo pose/HUD cadence exactly as run_sync's loop variable did.

        When ``live_mode`` is OFF (the default), every block runs every frame and the output
        is byte-identical to run_scout --sync. When ``live_mode`` is ON, the HEAVY detectors
        (pose / net / blob+fuse / pile / OCR) only run on their configured cadence and their
        last outputs are CACHED; the cheap trackers still run every frame and PREDICT forward,
        so boxes/markers keep moving smoothly at the display rate. ``fast`` additionally drops
        the blob+pile classical work entirely (net-only balls) for maximum speed.
        """
        import cv2

        from frcscout.refine.detect_fuse import fuse as fuse_dets
        i = self._loop_i
        fidx = frame_idx
        live = self.live_mode
        # cadence predicates: in non-live mode everything is "on" (every % 1 == 0).
        pose_on = (i % (self.pose_every_live if live else self.pose_every) == 0)
        net_on = (not live) or (i % self.net_every == 0)
        blob_on = ((not live) or (i % self.blob_every == 0)) and not self.fast
        pile_on = ((not live) or (i % self.pile_every == 0)) and not self.fast
        ocr_on = (i % (self.ocr_every if live else self.hud_every) == 0)

        _t = self._t0()
        self._update_robot_static_mask(frame)
        self._acc("static_mask", _t)

        # --- POSE FIRST (refreshed ~4Hz; camera ~fixed but pans/zooms) so robots, balls,
        #     and the classical blob detector all share one image->field homography. ---
        _t = self._t0()
        if pose_on:
            p = self.pose_est.estimate(frame, fidx, region=self.MAIN_PANE)
            if p is not None and p.valid:
                self.cur_pose = p
        self._acc("pose", _t)
        Hpose = self.cur_pose.H if (self.cur_pose is not None and self.cur_pose.valid) else None
        if Hpose is not None:
            self.pose_frames += 1
        y_main = self.MAIN_PANE[1] * self.H

        # --- NET (YOLO multi-class). On live OFF-frames reuse the cached dets. ---
        _t = self._t0()
        if net_on:
            r = self.model.predict(frame, imgsz=self.ball_imgsz, conf=self.gate,
                                   device=self.device, verbose=False)[0]
            net_ball_dets, robot_dets_raw = [], []
            if r.boxes is not None and len(r.boxes):
                xyxy = r.boxes.xyxy.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy().astype(int)
                cfd = r.boxes.conf.cpu().numpy()
                for b, c, cc in zip(xyxy, cls, cfd):
                    box = tuple(map(float, b))
                    if c == 0 and cc >= self.ball_conf:
                        net_ball_dets.append(Detection("ball", box, float(cc), fidx))
                    elif c == 1 and cc >= self.robot_conf:
                        robot_dets_raw.append(Detection("robot", box, float(cc), fidx))
            self._cache_net_ball_dets = net_ball_dets
        else:
            net_ball_dets = self._cache_net_ball_dets
            robot_dets_raw = []   # robots: coast the tracker on OFF frames (empty -> KF predict)
        self._acc("net_predict", _t)

        # FUEL (R2): classical color-prior blob detector (recovers spread/ground balls the
        # net structurally misses) FUSED with the net via WBF. The blob detector is
        # crowd-safe (Lab-b + circularity + field-projected size gate); fuse() down-weights
        # single-source boxes. Returns dicts -> wrap back to Detection for the ball tracker.
        _t = self._t0()
        if blob_on:
            blob_dets = self.blob.detect(frame, fidx, pose=self.cur_pose)
            self._cache_blob_dets = blob_dets
        elif self.fast:
            blob_dets = []          # fast mode: net-only balls
        else:
            blob_dets = self._cache_blob_dets
        self._acc("blob_detect", _t)

        _t = self._t0()
        if net_on or blob_on:
            if self.fast and not blob_dets:
                # fast mode: blob is OFF, so the WBF fuse is a no-op re-wrap of the net dets
                # — skip it and use the net ball dets directly (saves ~6 ms/frame, the margin
                # that clears the 15 fps detection floor on CPU). Net-only balls is the
                # documented fast-mode tradeoff.
                ball_dets = list(net_ball_dets)
            else:
                fused = fuse_dets([net_ball_dets, blob_dets], iou_thr=0.3)
                ball_dets = [Detection("ball", tuple(map(float, d["xyxy"])),
                                       float(d.get("conf", d.get("score", 0.5))), fidx) for d in fused]
            # Gate INVENTORY balls to the field-region POLYGON (play area; excludes the stands
            # + bottom pane). This is the precision fix for the crowd-circle FPs: the clustered
            # tags can't give a reliable back-projected extent, but a fixed image polygon is
            # robust on this near-fixed wide field shot. Airborne shots are handled separately
            # by the flight path.
            ball_dets = [_d for _d in ball_dets
                         if cv2.pointPolygonTest(
                             self.FIELD_POLY,
                             ((_d.xyxy[0] + _d.xyxy[2]) / 2.0, (_d.xyxy[1] + _d.xyxy[3]) / 2.0),
                             False) >= 0]
            self._cache_ball_dets = ball_dets
        else:
            # live OFF-frame: re-feed the cached inventory dets so the static carpet balls
            # stay shown (inventory balls are near-stationary litter; the greedy id-tracker
            # holds their ids). Robots/flight still get real Kalman prediction below.
            ball_dets = self._cache_ball_dets
        self._acc("wbf_fuse", _t)

        # Dense-pile AREA estimate (FUEL): the merged, occluded heap can't be boxed per
        # ball, so count it honestly by field-gated yellow AREA / a Hough-calibrated
        # ball-face area. total_est = these polygon-gated isolated balls + the pile; the
        # label is flagged "~N (area est.)" whenever a pile contributed (never faked exact).
        _t = self._t0()
        if pile_on:
            pile = self.blob.estimate_pile(frame, fidx, isolated_dets=ball_dets,
                                           field_poly_px=self.FIELD_POLY)
            self._cache_pile = pile
        elif self.fast:
            # fast mode: no dense-pile classical work — honest net-only count (isolated balls,
            # no area estimate). PileEstimate.is_estimate is False, so the label is a plain int.
            from frcscout.detect.ball_blob import PileEstimate
            pile = PileEstimate(count_est=0, components=[], single_ball_area_px=0.0,
                                single_ball_source="default", isolated_count=len(ball_dets))
        else:
            # live OFF-frame (not fast): reuse the cached pile estimate.
            pile = self._cache_pile if self._cache_pile is not None else \
                self.blob.estimate_pile(frame, fidx, isolated_dets=ball_dets,
                                        field_poly_px=self.FIELD_POLY)
            self._cache_pile = pile
        self._acc("pile_estimate", _t)

        # ROBOTS (R1): gate detections to the FIELD-REGION POLYGON by floor-contact point
        # (cx, y2) — the same robust image-space region used for balls. Image-space, so it
        # works even when the pose is momentarily invalid.
        _t = self._t0()
        _before = len(robot_dets_raw)
        robot_dets = [_d for _d in robot_dets_raw
                      if cv2.pointPolygonTest(
                          self.ROBOT_POLY,
                          ((_d.xyxy[0] + _d.xyxy[2]) / 2.0, float(_d.xyxy[3])),
                          False) >= 0]
        self.crowd_rejected += _before - len(robot_dets)
        # PRECISION FILTER — the measured winner of the detector fleet (eval vs hand-labeled
        # ground truth: robot F1 0.25 -> 0.51, precision 0.30 -> 0.90). filter_robot_dets does:
        # bumper-colour gate (rejects balls/people/grey structures — no red/blue bumper),
        # tighten-to-bumper-footprint (the net over-boxes), person/HUB/size/aspect reject, NMS.
        # This is what stops "every box is a ball or a player". See detect/robot_filter.py +
        # scripts/eval_robots.py. Brief occlusion drops are coasted by the tracker's 6-robot prior.
        from frcscout.detect.robot_filter import filter_robot_dets
        _pre_color = len(robot_dets)
        _filtered = filter_robot_dets(
            frame, [_d.xyxy for _d in robot_dets], self.W, self.H,
            confs=[_d.conf for _d in robot_dets],
            static_mask=self._robot_static_mask,
        )
        # The tracker uses high/low confidence tiers for association. The filter has already
        # made the precision decision, so feed accepted robots as high-confidence detections.
        robot_dets = [Detection("robot", (b[0], b[1], b[2], b[3]), 1.0, fidx) for b in _filtered]
        self.robot_noncolor_rejected = getattr(self, "robot_noncolor_rejected", 0) \
            + (_pre_color - len(robot_dets))
        robots = self.robot_tracker.update(fidx, robot_dets, frame)

        for rb in robots:
            cx, cy = rb.centroid_px
            fxy = _project_h(Hpose, cx, rb.xyxy[3]) if (Hpose is not None and cy < y_main) else None
            if fxy is not None:
                # A6.3: an edge robot past the recovered-tag hull back-projects to absurd
                # off-field meters (homography extrapolation). Clamp the reported field XY
                # into the field rectangle (+0.5 m tol); center-field fixes are untouched.
                fxy_clamped = self.field.clamp_field_xy(fxy)
                if self._A63_DUMP is not None:
                    self._A63_DUMP.write(_json.dumps(
                        {"f": fidx, "rid": rb.tid, "raw": list(fxy),
                         "clamped": list(fxy_clamped), "geo": "tag"}) + "\n")
                rb.field_xy_m = fxy_clamped
                bl = _project_h(Hpose, rb.xyxy[0], rb.xyxy[3])
                br = _project_h(Hpose, rb.xyxy[2], rb.xyxy[3])
                rb.field_wh_m = (abs(br[0] - bl[0]) if (bl and br) else 0.8, 0.8)
                rb.geo = "tag"
            else:
                # A6.3: the px fallback (cx*MPP) also runs off-field — a robot near the
                # right frame edge maps to cx*0.01 ~ 19 m on a 16.5 m field. Clamp it into
                # the field rectangle (+0.5 m tol) like the tag path; in-bounds untouched.
                px_raw = (cx * MPP, cy * MPP)
                rb.field_xy_m = self.field.clamp_field_xy(px_raw)
                rb.field_wh_m = ((rb.xyxy[2] - rb.xyxy[0]) * MPP, (rb.xyxy[3] - rb.xyxy[1]) * MPP)
                rb.geo = "px"
                if self._A63_DUMP is not None:
                    self._A63_DUMP.write(_json.dumps(
                        {"f": fidx, "rid": rb.tid, "raw": list(px_raw),
                         "clamped": list(rb.field_xy_m), "geo": "px"}) + "\n")

        tracked_balls = self.ball_tracker.update(ball_dets, fidx)
        ball_tracks = []
        for tid, d in tracked_balls:
            c = ((d.xyxy[0] + d.xyxy[2]) / 2.0, (d.xyxy[1] + d.xyxy[3]) / 2.0)
            bt = Track(tid=tid, cls="ball", xyxy=d.xyxy, centroid_px=c)
            fxy = _project_h(Hpose, c[0], c[1]) if (Hpose is not None and c[1] < y_main) else None
            bt.field_xy_m = fxy if fxy is not None else (c[0] * MPP, c[1] * MPP)
            ball_tracks.append(bt)
        self._acc("robot_gate_track", _t)

        # AIRBORNE shots: feed the flight tracker the NET ball detections in the shooting-
        # arc band ABOVE the carpet. The inventory polygon rejects these (too high) and the
        # old MOG2 flight detector only caught ~10% — but the net reliably detects the
        # airborne balls (same outlines it gives ground balls; ~14-20/frame measured). The
        # tracker's velocity gate then confirms the MOVING ones as in-flight; a ball merely
        # resting above the carpet stays below the speed gate and is not marked a shot.
        _t = self._t0()
        _air_top, _air_bot = 0.12 * self.H, 0.33 * self.H
        flight_dets = [d for d in net_ball_dets
                       if _air_top <= (d.xyxy[1] + d.xyxy[3]) / 2.0 < _air_bot
                       and 0.04 * self.W <= (d.xyxy[0] + d.xyxy[2]) / 2.0 <= 0.96 * self.W
                       and _yellow_frac(frame, d.xyxy) >= 0.25]
        flight_tracks = self.flight_trk.update(fidx, flight_dets)
        self.last_ball_det_count = len(ball_dets)
        self.last_flight_count = len(flight_tracks)
        self._acc("flight", _t)

        # Periodic HUD OCR -> persistent voted overlay (teams) + latest scalars.
        _t = self._t0()
        if self.hud is not None and ocr_on:
            ov = self.hud.read(frame, fidx)
            for t in ov.blue_teams:
                self.blue_votes[t] += 1
            for t in ov.red_teams:
                self.red_votes[t] += 1
            self.cur_ov.blue_teams = [t for t, _ in self.blue_votes.most_common(3)]
            self.cur_ov.red_teams = [t for t, _ in self.red_votes.most_common(3)]
            for attr in ("blue_score", "red_score", "blue_fuel", "red_fuel",
                         "timer_s", "match_label"):
                v = getattr(ov, attr)
                if v is not None:
                    setattr(self.cur_ov, attr, v)
            self.runner._overlay_teams = {"red": list(self.cur_ov.red_teams),
                                          "blue": list(self.cur_ov.blue_teams)}
            # R3: an overlay FUEL increment = a made shot for that alliance (authoritative,
            # TBA-grounded). Attribute via the embedded scoring tracker.
            for _al in ("blue", "red"):
                _f = getattr(self.cur_ov, f"{_al}_fuel")
                if _f is not None and self.prev_fuel[_al] is not None and _f > self.prev_fuel[_al]:
                    for _ in range(int(_f - self.prev_fuel[_al])):
                        self.runner.note_overlay_score(fidx, _al)
                if _f is not None:
                    self.prev_fuel[_al] = _f
        self._acc("hud_ocr", _t)

        _t = self._t0()
        st = FrameState(frame_idx=fidx, t_s=fidx / self.fps, robots=robots,
                        balls_inventory=ball_tracks, balls_flight=flight_tracks,
                        overlay=self.cur_ov, pose=self.cur_pose)
        res = self.runner.process(st)
        self.max_six = max(self.max_six, len(st.robots))
        for rec in res["phase"].values():
            p = rec.get("phase")
            if p:
                self.phase_hist[p] = self.phase_hist.get(p, 0) + 1
        self._acc("phase_matchrunner", _t)

        # Render the scouting overlay (ALWAYS — the live app needs it every frame; the
        # renderer is deterministic and never mutates export state, so the produced bytes
        # are identical to the old "render only when a writer existed" path).
        # Render only CONFIRMED-flight tracks. FlightDetector now AND-s motion with the
        # fuel-color mask, so is_flight tracks are real moving yellow balls (shots), not
        # crowd/pan noise — safe to draw (R4).
        _t = self._t0()
        res["state"].balls_flight = [t for t in (res["state"].balls_flight or [])
                                     if getattr(t, "is_flight", False)]
        ann = frame
        if self.render_enabled:
            from frcscout.viz.scout_render import draw_scout_overlay
            ann = draw_scout_overlay(frame, res["state"], res["six"], res["phase"],
                                     self.runner.phase.metrics(), fps=self.fps)
        if self.render_enabled and ann is not None:
            # Dense-pile region(s) + honest fuel headline ("~N (area est.)"). The pile is
            # what the per-ball boxer structurally can't count; drawn as a magenta outline
            # so it reads as "this heap, ~N balls", not N false precise boxes.
            for _comp in pile.components:
                if _comp.contour is not None and len(_comp.contour) >= 3:
                    cv2.polylines(ann, [_comp.contour.reshape(-1, 1, 2)], True,
                                  (255, 0, 255), 2)
            _ftxt = "FUEL " + pile.label()
            cv2.putText(ann, _ftxt, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(ann, _ftxt, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2, cv2.LINE_AA)
            # The two static HUB openings (the "collector bins") robots score fuel into.
            if getattr(self.runner.phase, "bin_scorer", None) is not None:
                for _al, _poly in self.runner.phase.bin_scorer.bin.polygons_px((self.W, self.H)).items():
                    _bcol = (0, 0, 220) if _al == "red" else (220, 80, 0)
                    _pp = _np.asarray(_poly, dtype=_np.int32).reshape(-1, 1, 2)
                    cv2.polylines(ann, [_pp], True, _bcol, 2)
                    _bc = _pp.reshape(-1, 2).mean(axis=0).astype(int)
                    cv2.putText(ann, "HUB", (int(_bc[0]) - 14, int(_bc[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _bcol, 2, cv2.LINE_AA)

        self._acc("render", _t)

        # stash last-frame outputs for the driver / live app
        self.last_state = res["state"]
        self.last_res = res
        self.last_pile = pile
        self._loop_i += 1
        self._maybe_print_profile()
        return ann

    def close(self):
        """Release the A6.3 dump file if open. Idempotent."""
        if self._A63_DUMP is not None:
            try:
                self._A63_DUMP.close()
            finally:
                self._A63_DUMP = None
