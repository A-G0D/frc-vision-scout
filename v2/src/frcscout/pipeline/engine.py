"""The live 60fps engine — the orchestrator that ties every layer together.

Hot path (per frame, CPU, must fit in ~16.6ms):
    1. pull decoded frame (QSV, overlapped)
    2. scheduler.tick(): launch any due detector ASYNC (robot/ball/overlay) — non-blocking
    3. ball_flight: classical motion detect on the field trapezoid (cheap, 60Hz)
    4. consume LATEST detector snapshots (robot boxes, ball inventory, overlay)
    5. trackers update (robot ByteTrack, flight Kalman) — predict between detector refreshes
    6. geometry: image -> field coords for all ground-plane objects
    7. fuse: events + attribution + state
    8. (optional) render

The heavy nets (robot on NPU, ball on iGPU) run in scheduler worker threads and never
block this loop. See scheduler.py for why p99 stays under budget.

Run:
    python -m frcscout.pipeline.engine --config configs/deploy.yaml --source match.mp4
    python -m frcscout.pipeline.engine --config configs/deploy.yaml --source <url> --benchmark
"""
from __future__ import annotations

import argparse
import time
from typing import Optional

import cv2
import numpy as np

from frcscout.io.config import Config
from frcscout.io.types import FrameState, Track
from frcscout.pipeline.decode import FrameSource
from frcscout.pipeline.scheduler import AsyncScheduler, RateMeter


class Engine:
    def __init__(self, cfg: Config, source: str):
        self.cfg = cfg
        self.src = FrameSource(source, backend=cfg.get("decode.backend", "qsv"))
        self.fps = self.src.fps
        self.target_fps = cfg.get("target_fps", 60)
        self.budget_ms = cfg.get("frame_budget_ms", 16.6)

        # --- Lazy imports keep torch/ultralytics OUT of the deploy path (OpenVINO only).
        from frcscout.detect.robot_net import RobotNet
        from frcscout.detect.ball_net import BallInventoryNet
        from frcscout.detect.flight import FlightDetector
        from frcscout.detect.overlay_ocr import OverlayReader
        from frcscout.track.bytetrack import RobotTracker
        from frcscout.track.ball_kalman import FlightTracker
        from frcscout.geometry.apriltag import PoseEstimator
        from frcscout.geometry.field import FieldModel
        from frcscout.fuse.state import StateFuser

        self.field = FieldModel(cfg.get("field_config"))
        self.pose_est = PoseEstimator(self.field)
        self.robot_net = RobotNet(cfg.get("detectors.robot"))
        self.ball_net = BallInventoryNet(cfg.get("detectors.ball_inventory"))
        self.flight = FlightDetector(cfg.get("detectors.ball_flight"), self.field)
        self.overlay = OverlayReader(cfg.get("detectors.overlay"))
        self.robot_tracker = RobotTracker(cfg.get("tracking.robot"), fps=self.fps)
        self.flight_tracker = FlightTracker(cfg.get("tracking.ball_flight"), fps=self.fps)
        self.fuser = StateFuser(self.field, events_config="configs/events.yaml", fps=self.fps)

        # Async detectors (NOT ball_flight — that's a cheap 60Hz CPU step on the hot path).
        self.sched = AsyncScheduler(self.target_fps, max_workers=4)
        self.sched.register("robot", self.robot_net.infer, cfg.get("detectors.robot.cadence_hz", 60))
        self.sched.register("ball", self.ball_net.infer, cfg.get("detectors.ball_inventory.cadence_hz", 5))
        self.sched.register("overlay", self.overlay.read, cfg.get("detectors.overlay.cadence_hz", 2))
        self.meter = RateMeter()
        self._pose = None

    # ----------------------------------------------------------------------- #
    def step(self, frame, frame_idx: int) -> FrameState:
        t0 = time.perf_counter()
        st = FrameState(frame_idx=frame_idx, t_s=frame_idx / self.fps)

        # 2. launch async detectors if due (non-blocking)
        self.sched.tick(frame, frame_idx)

        # 3. 60Hz classical flight detection (cheap, on the hot path)
        flight_dets = self.flight.detect(frame, frame_idx)

        # 4. consume latest async snapshots
        robot_dets, _ = self.sched.latest("robot")
        ball_dets, _ = self.sched.latest("ball")
        overlay_state, _ = self.sched.latest("overlay")
        if overlay_state:
            st.overlay = overlay_state

        # 1b. recover/refresh camera pose occasionally (cheap; cached)
        self._pose = self.pose_est.maybe_update(frame, frame_idx, self._pose)
        st.pose = self._pose

        # 5. trackers (every frame; interpolate between detector refreshes)
        st.robots = self.robot_tracker.update(frame_idx, robot_dets or [], frame)
        st.balls_flight = self.flight_tracker.update(frame_idx, flight_dets)
        st.balls_inventory = self.ball_net.as_tracks(ball_dets or [])

        # 6. geometry: image px -> field meters for ground-plane objects
        if self._pose and self._pose.valid:
            self.field.project_tracks(st.robots, self._pose)
            self.field.project_tracks(st.balls_inventory, self._pose)
            self.field.project_tracks(st.balls_flight, self._pose, airborne=True)

        # 7. fuse events + probabilistic attribution + game state
        st.events = self.fuser.update(st)

        st.timings_ms["hot_path"] = (time.perf_counter() - t0) * 1000.0
        return st

    # ----------------------------------------------------------------------- #
    def run(
        self,
        render: bool | None = None,
        max_frames: int | None = None,
        save_path: Optional[str] = None,
    ):
        render = self.cfg.get("render.enabled", False) if render is None else render
        # Live sources run indefinitely; max_frames is only honoured for file sources.
        live = getattr(self.src, "is_live", False)

        writer: Optional[cv2.VideoWriter] = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                save_path, fourcc, self.fps, (self.src.w, self.src.h)
            )

        try:
            for i, frame in enumerate(self.src.frames()):
                # Skip max_frames limit when running against a live stream.
                if max_frames and not live and i >= max_frames:
                    break
                st = self.step(frame, i)
                fps = self.meter.mark()
                if render or writer:
                    annotated = self._draw(frame, st, fps)
                    if writer and annotated is not None:
                        writer.write(annotated)
        finally:
            if writer:
                writer.release()
            self.sched.shutdown()
            self.src.release()

    def benchmark(self, n: int = 600) -> dict:
        """Measure sustained fps + p99 frame time against the 16.6ms budget."""
        for i, frame in enumerate(self.src.frames()):
            if i >= n:
                break
            self.step(frame, i)
            self.meter.mark()
        self.sched.shutdown()
        self.src.release()
        report = {
            "ema_fps": round(self.meter.ema_fps or 0, 1),
            "p99_frame_ms": round(self.meter.p99_ms(), 2),
            "budget_ms": self.budget_ms,
            "passes_budget": self.meter.p99_ms() <= self.budget_ms,
        }
        return report

    # ---------------------------------------------------------------------- #
    # Render helpers
    # ---------------------------------------------------------------------- #

    # BGR colour palette
    _RED_BGR    = (0,   0,   220)
    _BLUE_BGR   = (220, 80,  0)
    _YELLOW_BGR = (0,   220, 220)
    _WHITE_BGR  = (255, 255, 255)
    _GRAY_BGR   = (120, 120, 120)
    _GREEN_BGR  = (0,   200, 0)

    # Palette of 20 track-ID colours (cycles via tid % len)
    _TID_COLORS = [
        (255, 80,  80),  (80,  80,  255), (80,  255, 80),  (255, 200, 0),
        (0,   200, 200), (200, 0,   200), (255, 128, 0),   (0,   255, 128),
        (128, 0,   255), (255, 0,   128), (0,   128, 255), (128, 255, 0),
        (200, 100, 0),   (0,   100, 200), (100, 0,   200), (200, 0,   100),
        (80,  160, 255), (255, 160, 80),  (80,  255, 160), (160, 80,  255),
    ]

    @staticmethod
    def _tid_color(tid: int) -> tuple:
        return Engine._TID_COLORS[tid % len(Engine._TID_COLORS)]

    @staticmethod
    def _alliance_color(alliance) -> tuple:
        if alliance == "red":
            return Engine._RED_BGR
        if alliance == "blue":
            return Engine._BLUE_BGR
        return Engine._GRAY_BGR

    def _draw(self, frame, st: FrameState, fps: float):
        """Annotate *frame* with tracking/event overlays and return the annotated copy.

        Headless-safe: ``cv2.imshow`` is guarded — it only runs when a display is
        available (i.e. when ``render.enabled`` is True AND the build includes highgui).
        Returns the annotated frame (BGR ndarray) so the caller can write it to a file.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        render_cfg = self.cfg.get("render", {}) or {}
        draw_layers = render_cfg.get("draw", []) if isinstance(render_cfg, dict) else []
        # Default: draw everything when the list is absent from config.
        if not draw_layers:
            draw_layers = ["robots", "ball_flight", "ball_inventory", "field_grid", "events"]

        # ── 1. Field grid (faint, drawn first so everything else sits on top) ──────
        if "field_grid" in draw_layers and st.pose and getattr(st.pose, "valid", False):
            self._draw_field_grid(out, st.pose)

        # ── 2. Robot boxes ────────────────────────────────────────────────────────
        if "robots" in draw_layers:
            for t in st.robots:
                color = self._alliance_color(t.alliance)
                x1, y1, x2, y2 = (int(v) for v in t.xyxy)
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                label = f"R{t.tid}"
                if t.alliance:
                    label += f" [{t.alliance[0].upper()}]"
                cv2.putText(out, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        # ── 3. Flight balls — circles + velocity arrows ───────────────────────────
        if "ball_flight" in draw_layers:
            for t in st.balls_flight:
                cx, cy = (int(v) for v in t.centroid_px)
                r = max(8, (int(t.xyxy[2]) - int(t.xyxy[0])) // 2)
                cv2.circle(out, (cx, cy), r, self._YELLOW_BGR, 2)
                # Velocity arrow (scaled from m/s to pixels heuristically at ~50px/m)
                vx, vy = t.vel_mps
                scale = 50.0
                ex, ey = int(cx + vx * scale), int(cy + vy * scale)
                if abs(vx) + abs(vy) > 0.05:
                    cv2.arrowedLine(out, (cx, cy), (ex, ey),
                                    self._YELLOW_BGR, 2, tipLength=0.3)
                cv2.putText(out, f"f{t.tid}", (cx + r + 2, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, self._YELLOW_BGR, 1, cv2.LINE_AA)

        # ── 4. Inventory balls (faint, semi-transparent circles) ──────────────────
        if "ball_inventory" in draw_layers:
            overlay = out.copy()
            for t in st.balls_inventory:
                cx, cy = (int(v) for v in t.centroid_px)
                r = max(6, (int(t.xyxy[2]) - int(t.xyxy[0])) // 2)
                cv2.circle(overlay, (cx, cy), r, self._GREEN_BGR, -1)
            cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)

        # ── 5. Event labels ───────────────────────────────────────────────────────
        if "events" in draw_layers:
            y_offset = h - 30
            for ev in st.events[-5:]:           # show last 5 events to avoid clutter
                color = self._alliance_color(ev.alliance)
                label = f"{ev.type.upper()}"
                if ev.robot_id is not None:
                    label += f" R{ev.robot_id}"
                cv2.putText(out, label, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
                y_offset -= 22

        # ── 6. FPS / FUEL HUD ─────────────────────────────────────────────────────
        self._draw_hud(out, st, fps)

        # ── 7. Display (headless-safe) ─────────────────────────────────────────────
        show = self.cfg.get("render.enabled", False)
        if show:
            try:
                cv2.imshow("FRC Vision Scout", out)
                cv2.waitKey(1)
            except cv2.error:
                # Headless build — no highgui; silently skip.
                pass

        return out

    # ---------------------------------------------------------------------- #
    def _draw_field_grid(self, out, pose) -> None:
        """Draw a faint perspective grid over the field using the homography."""
        h_img, w_img = out.shape[:2]
        import numpy as np

        H = np.array(pose.H, dtype=np.float64)   # pose.H maps image-px → field-m; invert for field → image
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return

        # 2026 REBUILT: field is ~16.5m × 8.2m; draw 1m grid lines
        grid_color = (60, 60, 60)
        for x_m in range(0, 18):
            pts_field = np.array([[x_m, 0, 1], [x_m, 9, 1]], dtype=np.float64).T
            pts_img = H_inv @ pts_field
            pts_img /= pts_img[2]
            pts = pts_img[:2].T.astype(int)
            cv2.line(out, tuple(pts[0]), tuple(pts[1]), grid_color, 1)

        for y_m in range(0, 10):
            pts_field = np.array([[0, y_m, 1], [17, y_m, 1]], dtype=np.float64).T
            pts_img = H_inv @ pts_field
            pts_img /= pts_img[2]
            pts = pts_img[:2].T.astype(int)
            cv2.line(out, tuple(pts[0]), tuple(pts[1]), grid_color, 1)

    def _draw_hud(self, out, st: FrameState, fps: float) -> None:
        """Draw the FPS / score / fuel HUD bar at the top of the frame."""
        h, w = out.shape[:2]
        ov = st.overlay

        # Semi-transparent top bar
        bar = out[:36, :].copy()
        cv2.rectangle(bar, (0, 0), (w, 36), (20, 20, 20), -1)
        cv2.addWeighted(bar, 0.7, out[:36, :], 0.3, 0, out[:36, :])

        # FPS counter (top-right)
        fps_str = f"{fps:.1f} fps"
        cv2.putText(out, fps_str, (w - 110, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, self._WHITE_BGR, 2, cv2.LINE_AA)

        # Match label (top-center)
        if ov.match_label:
            lbl_sz, _ = cv2.getTextSize(ov.match_label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.putText(out, ov.match_label, ((w - lbl_sz[0]) // 2, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, self._WHITE_BGR, 2, cv2.LINE_AA)

        # Blue score / fuel (top-left)
        blue_txt = "B"
        if ov.blue_score is not None:
            blue_txt += f":{ov.blue_score}"
        if ov.blue_fuel is not None:
            blue_txt += f"  F{ov.blue_fuel}"
        cv2.putText(out, blue_txt, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, self._BLUE_BGR, 2, cv2.LINE_AA)

        # Red score / fuel
        red_txt = "R"
        if ov.red_score is not None:
            red_txt += f":{ov.red_score}"
        if ov.red_fuel is not None:
            red_txt += f"  F{ov.red_fuel}"
        cv2.putText(out, red_txt, (200, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, self._RED_BGR, 2, cv2.LINE_AA)

        # Timer
        if ov.timer_s is not None:
            timer_str = f"{ov.timer_s:.0f}s"
            cv2.putText(out, timer_str, (w // 2 + 80, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, self._WHITE_BGR, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(
        description="FRC Vision Scout v2 — live 60fps engine"
    )
    ap.add_argument("--config", default="configs/deploy.yaml")
    ap.add_argument("--source", required=True,
                    help="video file path, direct stream URL, or YouTube URL")
    ap.add_argument("--benchmark", action="store_true",
                    help="run 600-frame benchmark and print p99 stats")
    ap.add_argument("--frames", type=int, default=None,
                    help="max frames to process (file sources only; live runs indefinitely)")
    ap.add_argument("--save", metavar="OUTPUT.mp4", default=None,
                    help="write annotated frames to this video file")
    a = ap.parse_args()
    eng = Engine(Config.load(a.config), a.source)
    if a.benchmark:
        import json
        print(json.dumps(eng.benchmark(), indent=2))
    else:
        # Live streams ignore --frames (run indefinitely); file sources respect it.
        eng.run(max_frames=a.frames, save_path=a.save)


if __name__ == "__main__":
    main()
