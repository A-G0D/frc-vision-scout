"""live_app.py — a GENUINELY-LIVE side-by-side scout app (NOT pre-render-and-replay).

This emulates the eventual product: a live broadcast feed comes IN, the ScoutPipeline
processes it in REAL TIME, and a side-by-side window shows

    LEFT  = the RAW feed advancing at real time (src/feed fps)
    RIGHT = the annotated scout output, refreshed at the feed rate — its boxes/markers keep
            MOVING smoothly via the cheap trackers (Kalman predict) even on frames where the
            heavy detectors are skipped by the cadence.

Real-time discipline (it does NOT hide the truth):
  * PRODUCER reads at real time and keeps only the NEWEST frame (drops backlog = stays live).
  * CONSUMER pulls the NEWEST frame and runs ScoutPipeline.process. In LIVE cadence mode the
    heavy detectors (net / blob+pile / OCR / pose) run only on their cadence and are CACHED;
    on the OFF frames only the cheap trackers + the overlay render run, so the RIGHT pane is
    smooth at (near) the feed rate while detections refresh at the best achievable Hz.
  * DISPLAY shows LEFT=newest raw, RIGHT=newest processed, and a HUD with feed-fps,
    DISPLAY fps, DETECTION fps, latency, drop%.

GUI controls (display mode):
    RESTART        — re-seek the source to frame 0 (or --start)
    Feed FPS       — 15 / 30 / 60 chooser (re-paces the producer)
    Detect cadence — full / cadence / fast (live ScoutPipeline mode)

Usage
-----
    # live window from a local file (paced to 60fps, cadence detection, open at an action):
    PYTHONPATH=src python scripts/live_app.py --source data/scout_out/qual59_action.mp4 \
        --feed-fps 60 --mode cadence --start 0

    # HEADLESS validation (no display): run ~8s, save paired left/right frames + latency log:
    PYTHONPATH=src python scripts/live_app.py --source data/scout_out/qual59_action.mp4 \
        --device cpu --capture data/scout_out/live_capture --capture-secs 8 \
        --feed-fps 60 --mode cadence
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

# make `frcscout` importable when run as a plain script (PYTHONPATH=src also works)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from frcscout.io.config import Config


# Cadence presets (process-call modulos). "cadence" is tuned to clear the >=15fps detection
# floor on CPU while keeping the heavy classical detectors refreshing often enough to track
# the dense ball field; OCR is rare (teams/score change slowly) to avoid a multi-100ms stall.
MODE_PRESETS = {
    "full":    dict(live_mode=False, fast=False),
    # cadence: net EVERY frame (detect == display) so the detection floor is met; the heavy
    # classical blob+pile run at 1/6 and OCR is rare. DISPLAY stays smooth because the net is
    # the per-frame work and the trackers predict between blob refreshes.
    "cadence": dict(live_mode=True,  fast=False, net_every=1, blob_every=8, pile_every=8,
                    ocr_every=180, pose_every=12),
    # fast: drop blob+pile entirely (net-only balls) AND net every frame -> highest detect fps.
    "fast":    dict(live_mode=True,  fast=True,  net_every=1, blob_every=999, pile_every=999,
                    ocr_every=180, pose_every=12),
}


def force_device(cfg, device):
    # "deploy" = DON'T override — keep the per-detector devices from deploy.yaml
    # (robot -> intel:npu, ball -> intel:gpu), i.e. genuinely use NPU + iGPU together.
    if device in ("deploy", None, ""):
        return cfg
    dets = cfg.get("detectors", {}) or {}
    for k in ("robot", "ball_inventory"):
        if isinstance(dets.get(k), dict):
            dets[k]["device"] = device
    return cfg


def _resolve_source(source: str) -> str:
    """Resolve a YouTube/yt-dlp URL to a direct stream (the livestream path), else passthrough.
    Mirrors run_scout.run_sync so the live app pulls from a stream exactly the same way."""
    if "://" in source and ("youtu" in source):
        try:
            from frcscout.pipeline.decode import resolve_yt_url
            out = resolve_yt_url(source)
            print("[live] resolved YouTube URL -> direct stream", flush=True)
            return out
        except Exception as e:
            print(f"[live] URL resolve failed ({type(e).__name__}: {e}); using raw source", flush=True)
    return source


class LiveScout:
    """Owns the producer + consumer threads and the shared latest-frame slots.

    The producer timestamps every captured frame with ``time.perf_counter()``. The consumer
    carries that capture timestamp through the pipeline, so the REAL end-to-end latency the
    HUD shows is ``now − capture_time_of_the_frame_now_on_the_right`` — the honest seconds the
    scout output trails the live feed.
    """

    def __init__(self, source: str, cfg, device: str, feed_fps: float | None = None,
                 queue_max: int = 2, mode: str = "full", start: int = 0,
                 analyze_fps: str | float | None = None):
        import cv2

        self.cv2 = cv2
        self._source = source
        self._resolved = _resolve_source(source)
        self.is_url = "://" in source
        self.start_frame = int(start)
        self.cap = cv2.VideoCapture(self._resolved)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open source: {source}")
        src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 60.0
        self._src_fps = float(src_fps)
        self.feed_fps = float(feed_fps) if feed_fps else float(src_fps)
        # Dynamic even-decimation ("analyze 1 of every N frames, progressing at real-time speed"):
        #   None  -> legacy drop-oldest (analysis = newest available, bursty real-time sync)
        #   "auto"-> stride recomputed each loop from the consumer's measured proc_fps so the
        #            analyzed subset stays in REAL-TIME lockstep with the source (the requested behaviour)
        #   <num> -> fixed target analyze-fps; stride = round(src_fps / num)
        self.analyze_fps: str | float | None = (
            analyze_fps if (analyze_fps is None or analyze_fps == "auto") else float(analyze_fps))
        self._stride = 1                     # live decimation stride (1 of every _stride frames)
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.start_frame:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        # ScoutPipeline is the ONE source of truth (same class run_scout --sync uses).
        from frcscout.pipeline.scout_pipeline import ScoutPipeline
        self.pipe = ScoutPipeline(cfg, frame_wh=(self.W, self.H), fps=self.feed_fps,
                                  device=device)
        self.mode = mode
        self.set_mode(mode)

        self.queue_max = int(queue_max)
        self._buf: list = []
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

        # shared display slots
        self._latest_raw = None              # (idx, frame) newest captured frame (LEFT)
        self._latest_proc = None             # (idx, capture_time, annotated) (RIGHT)
        self._latest_proc_latency = 0.0      # seconds the RIGHT frame trails the live feed

        # counters / stats
        self.captured = 0
        self.processed = 0
        self.dropped = 0
        self._proc_times: list = []          # rolling per-frame process wall-times
        self._detect_stamps: list = []       # wall-times of DETECTION-refresh frames (net ran)
        self._disp_stamps: list = []         # wall-times the RIGHT pane was refreshed
        self._stop = threading.Event()
        self._restart = threading.Event()
        self._t_start = None

        self._producer = threading.Thread(target=self._produce, daemon=True, name="producer")
        self._consumer = threading.Thread(target=self._consume, daemon=True, name="consumer")

    # ------------------------------------------------------------------ controls
    def set_mode(self, mode: str):
        """Apply a cadence preset to the live pipeline (full | cadence | fast)."""
        preset = MODE_PRESETS.get(mode, MODE_PRESETS["full"])
        self.mode = mode
        self.pipe.configure_live(**preset)

    def set_feed_fps(self, fps: float):
        self.feed_fps = float(fps)

    def _compute_stride(self) -> int:
        """Decimation stride: analyze 1 of every N source frames. With analyze_fps='auto',
        N tracks the consumer's MEASURED throughput so the analyzed subset stays in real-time
        lockstep (e.g. 60fps source + ~3fps analysis -> N≈20). With a number, N=round(src/num)."""
        if self.analyze_fps == "auto":
            avg = (sum(self._proc_times) / len(self._proc_times)) if self._proc_times else 0.0
            proc_fps = (1.0 / avg) if avg > 0 else self._src_fps
            target = max(0.5, proc_fps)
        elif isinstance(self.analyze_fps, (int, float)):
            target = max(0.5, float(self.analyze_fps))
        else:
            return 1
        return max(1, int(round(self._src_fps / target)))

    def request_restart(self):
        """Re-seek the source to the start frame (thread-safe; producer handles it)."""
        self._restart.set()

    # ------------------------------------------------------------------ threads
    def start(self):
        self._t_start = time.perf_counter()
        self._producer.start()
        self._consumer.start()

    def stop(self):
        self._stop.set()
        with self._cv:
            self._cv.notify_all()

    def _produce(self):
        """Read frames at REAL TIME (paced to feed_fps). On a full queue, DROP the oldest
        (count it) — a real live stream cannot pause the camera to wait for our CPU."""
        next_t = time.perf_counter()
        idx = 0
        while not self._stop.is_set():
            if self._restart.is_set():
                self._restart.clear()
                self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, self.start_frame)
                idx = 0
                next_t = time.perf_counter()
            period = 1.0 / max(1.0, self.feed_fps)   # re-read each loop so the FPS chooser is live
            ok, frame = self.cap.read()
            if not ok:
                if self.is_url:
                    time.sleep(0.05)
                    continue
                # file EOF -> loop back to start so the live window keeps running
                self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, self.start_frame)
                idx = 0
                continue
            now = time.perf_counter()
            # Even-decimation: analyze only 1 of every _stride frames so the analyzed subset
            # keeps REAL-TIME pace (the LEFT pane still updates every frame -> smooth video;
            # the source advances at real time -> "progressing at the same speed").
            if self.analyze_fps is not None:
                period = 1.0 / max(1.0, self._src_fps)     # real-time playback of the source
                self._stride = self._compute_stride()
                feed_this = (idx % self._stride == 0)
            else:
                feed_this = True
            with self._cv:
                self._latest_raw = (idx, frame)
                if feed_this:
                    if len(self._buf) >= self.queue_max:
                        self._buf.pop(0)
                        self.dropped += 1
                    self._buf.append((idx, now, frame))
                    self._cv.notify()
            self.captured += 1
            idx += 1
            next_t += period
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()
        with self._cv:
            self._stop.set()
            self._cv.notify_all()

    def _consume(self):
        """Pull the NEWEST queued frame, process it, publish annotated + REAL latency.

        Tracks DETECTION-refresh timestamps separately from DISPLAY (RIGHT-pane) timestamps:
        in cadence/fast mode the RIGHT pane refreshes every consumed frame (smooth, trackers
        predict), but a real DETECTION (net inference) only happens on the cadence frames."""
        while not self._stop.is_set():
            with self._cv:
                while not self._buf and not self._stop.is_set():
                    self._cv.wait(timeout=0.2)
                if self._stop.is_set() and not self._buf:
                    break
                if not self._buf:
                    continue
                stale = len(self._buf) - 1
                if stale > 0:
                    self.dropped += stale
                idx, cap_t, frame = self._buf[-1]
                self._buf.clear()
            t0 = time.perf_counter()
            # did the heavy net run on THIS process() call? (cadence predicate, pre-call)
            net_ran = (not self.pipe.live_mode) or (self.pipe._loop_i % self.pipe.net_every == 0)
            try:
                ann = self.pipe.process(frame, idx)
            except Exception as e:  # never let a bad frame kill the live loop
                print(f"[live] process error on f{idx}: {type(e).__name__}: {e}", flush=True)
                continue
            t1 = time.perf_counter()
            self._proc_times.append(t1 - t0)
            if len(self._proc_times) > 30:
                self._proc_times.pop(0)
            self._disp_stamps.append(t1)
            if net_ran:
                self._detect_stamps.append(t1)
            # keep ~2s windows
            cutoff = t1 - 2.0
            self._disp_stamps = [s for s in self._disp_stamps if s >= cutoff]
            self._detect_stamps = [s for s in self._detect_stamps if s >= cutoff]
            latency = t1 - cap_t   # REAL end-to-end: now - capture time of THIS frame
            with self._cv:
                self._latest_proc = (idx, cap_t, ann)
                self._latest_proc_latency = latency
            self.processed += 1

    # ------------------------------------------------------------------ readout
    @staticmethod
    def _rate(stamps):
        """Frames/sec from a list of recent wall-time stamps over the span they cover."""
        if len(stamps) < 2:
            return 0.0
        span = stamps[-1] - stamps[0]
        return (len(stamps) - 1) / span if span > 0 else 0.0

    def stats(self) -> dict:
        elapsed = (time.perf_counter() - self._t_start) if self._t_start else 0.0
        with self._cv:
            qd = len(self._buf)
            lat = self._latest_proc_latency
            raw = self._latest_raw
            proc = self._latest_proc
            disp = list(self._disp_stamps)
            det = list(self._detect_stamps)
        avg_pt = sum(self._proc_times) / len(self._proc_times) if self._proc_times else 0.0
        proc_fps = (1.0 / avg_pt) if avg_pt > 0 else 0.0
        drop_rate = (self.dropped / self.captured) if self.captured else 0.0
        return {
            "elapsed_s": elapsed,
            "captured": self.captured,
            "processed": self.processed,
            "dropped": self.dropped,
            "drop_rate": drop_rate,
            "proc_fps": proc_fps,                 # raw per-frame throughput (1/avg process time)
            "display_fps": self._rate(disp),      # RIGHT-pane refresh rate (smooth, w/ predict)
            "detect_fps": self._rate(det),        # real DETECTION (net inference) rate
            "feed_fps": self.feed_fps,
            "latency_s": lat,
            "queue_depth": qd,
            "mode": self.mode,
            "stride": self._stride,
            "analyze_mode": ("auto" if self.analyze_fps == "auto"
                             else (f"{self.analyze_fps:g}fps" if self.analyze_fps else "drop")),
            "raw": raw,
            "proc": proc,
        }

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass
        try:
            self.pipe.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# HEADLESS validation mode — prove it works without a display (STEP 4)
# --------------------------------------------------------------------------- #
def run_headless(live: LiveScout, capture_dir: str, secs: float, tag: str = "") -> dict:
    import cv2

    os.makedirs(capture_dir, exist_ok=True)
    pfx = (tag + "_") if tag else ""
    log_path = os.path.join(capture_dir, f"{pfx}latency_log.csv")
    live.start()
    t_end = time.perf_counter() + secs
    saved = 0
    rows = ["t_s,raw_idx,proc_idx,latency_s,display_fps,detect_fps,proc_fps,"
            "queue_depth,captured,processed,dropped"]
    save_at = {int(secs * 0.30), int(secs * 0.60), int(secs * 0.90)}  # ~3 paired snapshots
    saved_secs: set = set()
    saved_paths: list = []
    # let the heavy first-frame warmup pass before timing the steady state
    warmup_until = time.perf_counter() + 1.2
    while time.perf_counter() < warmup_until and not live._stop.is_set():
        time.sleep(0.05)
    # reset the rolling fps windows + drop accounting so warmup doesn't skew the report
    with live._cv:
        live._disp_stamps.clear()
        live._detect_stamps.clear()
    steady_dropped0 = live.dropped
    steady_cap0 = live.captured
    steady_t0 = time.perf_counter()
    while time.perf_counter() < t_end and not live._stop.is_set():
        s = live.stats()
        t_rel = s["elapsed_s"]
        raw = s["raw"]
        proc = s["proc"]
        raw_idx = raw[0] if raw else -1
        proc_idx = proc[0] if proc else -1
        rows.append(f"{t_rel:.3f},{raw_idx},{proc_idx},{s['latency_s']:.3f},"
                    f"{s['display_fps']:.2f},{s['detect_fps']:.2f},{s['proc_fps']:.2f},"
                    f"{s['queue_depth']},{s['captured']},{s['processed']},{s['dropped']}")
        sec_bucket = int(t_rel)
        if (sec_bucket in save_at and sec_bucket not in saved_secs
                and raw is not None and proc is not None):
            lp = os.path.join(capture_dir, f"{pfx}pair_t{sec_bucket}s_LEFT_raw_f{raw[0]}.png")
            rp = os.path.join(capture_dir, f"{pfx}pair_t{sec_bucket}s_RIGHT_scout_f{proc[0]}.png")
            cv2.imwrite(lp, raw[1])
            cv2.imwrite(rp, proc[2])
            saved_secs.add(sec_bucket)
            saved_paths += [lp, rp]
            saved += 2
        time.sleep(0.1)
    # steady-state drop% (excludes warmup capture/drops)
    steady_dropped = live.dropped - steady_dropped0
    steady_cap = max(1, live.captured - steady_cap0)
    steady_drop_rate = steady_dropped / steady_cap
    live.stop()
    time.sleep(0.3)
    s = live.stats()
    with open(log_path, "w") as f:
        f.write("\n".join(rows) + "\n")
    live.release()

    report = {
        "mode": s["mode"], "feed_fps": s["feed_fps"],
        "display_fps": s["display_fps"], "detect_fps": s["detect_fps"],
        "proc_fps": s["proc_fps"], "latency_s": s["latency_s"],
        "drop_rate": steady_drop_rate, "dropped": steady_dropped, "captured": steady_cap,
        "saved_paths": saved_paths, "log_path": log_path,
    }
    print(f"\n========  LIVE HEADLESS VALIDATION  [mode={s['mode']}]  ========", flush=True)
    print(f"  feed_fps (real-time input)   : {s['feed_fps']:.1f}", flush=True)
    print(f"  DISPLAY fps (RIGHT refresh)  : {s['display_fps']:.2f} fps", flush=True)
    print(f"  DETECTION fps (net refresh)  : {s['detect_fps']:.2f} fps", flush=True)
    print(f"  raw proc throughput          : {s['proc_fps']:.2f} fps", flush=True)
    print(f"  END-TO-END latency           : {s['latency_s']:.2f} s", flush=True)
    print(f"  DROP RATE (steady, vs feed)  : {100.0*steady_drop_rate:.1f}%  "
          f"({steady_dropped}/{steady_cap})", flush=True)
    print(f"  latency log                  : {log_path}", flush=True)
    print(f"  paired frames saved          : {saved} files", flush=True)
    for p in saved_paths:
        print(f"      {p}", flush=True)
    print("================================================================\n", flush=True)
    return report


# --------------------------------------------------------------------------- #
# Tk display mode (STEP 3 GUI)
# --------------------------------------------------------------------------- #
def run_display(live: LiveScout) -> int:
    import cv2
    import tkinter as tk
    from PIL import Image, ImageTk

    live.start()
    root = tk.Tk()
    root.title("FRC Vision Scout v2 — LIVE (left=raw feed · right=scout, smooth via predict)")
    root.geometry("1320x720")
    root.configure(bg="#0c0c0c")

    # ── control bar ──
    ctrl = tk.Frame(root, bg="#161616")
    ctrl.pack(fill="x", side="top")
    tk.Button(ctrl, text="⟲ RESTART", command=live.request_restart,
              bg="#2a2a2a", fg="#f0f0f0", relief="flat", padx=10).pack(side="left", padx=6, pady=6)

    tk.Label(ctrl, text="Feed FPS:", bg="#161616", fg="#bbb").pack(side="left", padx=(14, 2))
    feed_var = tk.StringVar(value=str(int(live.feed_fps)))
    for f in (15, 30, 60):
        tk.Radiobutton(ctrl, text=str(f), variable=feed_var, value=str(f),
                       command=lambda: live.set_feed_fps(float(feed_var.get())),
                       bg="#161616", fg="#ddd", selectcolor="#333",
                       activebackground="#161616").pack(side="left")

    tk.Label(ctrl, text="Detect:", bg="#161616", fg="#bbb").pack(side="left", padx=(14, 2))
    mode_var = tk.StringVar(value=live.mode)
    for m in ("full", "cadence", "fast"):
        tk.Radiobutton(ctrl, text=m, variable=mode_var, value=m,
                       command=lambda: live.set_mode(mode_var.get()),
                       bg="#161616", fg="#ddd", selectcolor="#333",
                       activebackground="#161616").pack(side="left")

    panes = tk.Frame(root, bg="#0c0c0c")
    panes.pack(fill="both", expand=True)
    left_lbl = tk.Label(panes, bg="black")
    left_lbl.pack(side="left", fill="both", expand=True, padx=2, pady=2)
    right_lbl = tk.Label(panes, bg="black")
    right_lbl.pack(side="left", fill="both", expand=True, padx=2, pady=2)

    status = tk.Label(root, anchor="w", fg="#d8d8d8", bg="#0c0c0c", font=("Consolas", 10))
    status.pack(fill="x", padx=8, pady=(0, 6))

    imgs: dict = {}

    def _to_tk(frame, target_w):
        h, w = frame.shape[:2]
        scale = target_w / float(w) if w else 1.0
        if abs(scale - 1.0) > 1e-3:
            frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
        return ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    def tick():
        s = live.stats()
        tw = max(320, panes.winfo_width() // 2 - 8)
        if s["raw"] is not None:
            imgs["L"] = _to_tk(s["raw"][1], tw)
            left_lbl.config(image=imgs["L"])
        if s["proc"] is not None:
            imgs["R"] = _to_tk(s["proc"][2], tw)
            right_lbl.config(image=imgs["R"])
        gap = ((s["raw"][0] - s["proc"][0]) if (s["raw"] and s["proc"]) else 0)
        status.config(text=(
            f"mode {s['mode']:<7s} | feed {s['feed_fps']:.0f}  DISPLAY {s['display_fps']:4.1f}  "
            f"DETECT {s['detect_fps']:4.1f} fps  |  decimate[{s['analyze_mode']}] "
            f"1/{s['stride']}  |  latency {s['latency_s']:5.2f}s "
            f"({gap} fr)  |  drop {100*s['drop_rate']:.0f}%  |  q {s['queue_depth']}/{live.queue_max}"
        ))
        root.after(16, tick)  # ~60 Hz UI refresh (display only; pipeline runs in its thread)

    def on_close():
        live.stop()
        root.after(150, root.destroy)

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.bind("<Escape>", lambda e: on_close())
    tick()
    root.mainloop()
    live.release()
    return 0


def _build(a, mode):
    cfg = Config.load(a.config)
    force_device(cfg, a.device)
    live = LiveScout(a.source, cfg, device=a.device, feed_fps=a.feed_fps,
                     queue_max=a.queue_max, mode=mode, start=a.start,
                     analyze_fps=a.analyze_fps)
    print(f"[live] source={a.source} {live.W}x{live.H} feed_fps={live.feed_fps:.1f} "
          f"device={a.device} mode={mode} start={a.start} queue_max={a.queue_max}", flush=True)
    return live


def main() -> int:
    ap = argparse.ArgumentParser(description="Genuinely-live side-by-side FRC scout app.")
    ap.add_argument("--config", default="configs/deploy.yaml")
    ap.add_argument("--source", required=True, help="local file OR livestream/YouTube URL")
    ap.add_argument("--device", default="cpu", help="cpu|intel:cpu|intel:gpu|intel:npu "
                    "(default cpu — device does not change the CPU-bound per-frame cost)")
    ap.add_argument("--feed-fps", dest="feed_fps", type=float, default=None,
                    help="override real-time input pacing (default: source fps)")
    ap.add_argument("--analyze-fps", dest="analyze_fps", default=None,
                    help="even-decimation: 'auto' (track measured throughput so analysis keeps "
                         "real-time pace) or a number (target analyze-fps). Source plays real-time; "
                         "1 of every N frames is analyzed. Default off = legacy drop-oldest.")
    ap.add_argument("--queue-max", dest="queue_max", type=int, default=2)
    ap.add_argument("--start", type=int, default=0, help="seek the source to this frame "
                    "(open on an action window)")
    ap.add_argument("--mode", default="full", choices=list(MODE_PRESETS.keys()),
                    help="detection cadence preset (full=every-frame, cadence, fast)")
    ap.add_argument("--capture", default=None,
                    help="HEADLESS validation: dir to save paired frames + latency log (no display)")
    ap.add_argument("--capture-secs", dest="capture_secs", type=float, default=8.0)
    ap.add_argument("--capture-all-modes", action="store_true",
                    help="HEADLESS: run full, cadence AND fast back-to-back and summarize")
    a = ap.parse_args()

    if a.capture and a.capture_all_modes:
        reports = []
        for m in ("full", "cadence", "fast"):
            live = _build(a, m)
            reports.append(run_headless(live, a.capture, a.capture_secs, tag=m))
        print("\n############  MODE COMPARISON  ############", flush=True)
        print(f"{'mode':<9}{'feed':>6}{'DISPLAY':>10}{'DETECT':>9}{'latency':>10}{'drop%':>8}",
              flush=True)
        for r in reports:
            print(f"{r['mode']:<9}{r['feed_fps']:>6.0f}{r['display_fps']:>10.1f}"
                  f"{r['detect_fps']:>9.1f}{r['latency_s']:>9.2f}s{100*r['drop_rate']:>7.1f}%",
                  flush=True)
        print("###########################################\n", flush=True)
        return 0

    live = _build(a, a.mode)
    if a.capture:
        run_headless(live, a.capture, a.capture_secs, tag=a.mode)
        return 0
    return run_display(live)


if __name__ == "__main__":
    raise SystemExit(main())
