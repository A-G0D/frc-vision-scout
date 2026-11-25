"""livesim.py — emulate a LIVE broadcast vs the SCOUTED output, side by side.

LEFT  = the raw incoming broadcast, playing in (near) real-time — the "feed".
RIGHT = the rendered/annotated scout output, trailing the left by an adjustable
        PROCESSING DELAY. The right pane shows "buffering / processing…" until the
        delay has elapsed, then plays the annotation of frames that are `delay`
        seconds old — so you can watch what the live latency actually looks/feels like.

    python scripts/livesim.py --raw data/footage/qual59.mp4 --raw-start 8700 \
        --rendered data/scout_out/<demo>.mp4 --delay 2.0

CONTRACT: the rendered video must be the frame-for-frame annotation of the raw segment
that begins at --raw-start (rendered frame i == annotated raw frame raw_start+i). That is
exactly what run_scout.py --sync --start <raw_start> produces.

Controls: Space play/pause · R restart · the Delay slider (applied on restart) · ←/→ nudge
playback speed. The status bar shows LIVE time, SCOUTED time, and the live gap.
"""
from __future__ import annotations

import argparse
import os
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk


class LiveSim:
    def __init__(self, root: tk.Tk, raw: str, raw_start: int, rendered: str,
                 delay_s: float, fps: float | None, speed: float):
        self.root = root
        self.raw_path = raw
        self.raw_start = int(raw_start)
        self.rendered_path = rendered
        self.delay_s = float(delay_s)
        self.speed = float(speed)

        self.cap_raw = cv2.VideoCapture(raw)
        self.cap_ren = cv2.VideoCapture(rendered)
        if not self.cap_raw.isOpened():
            raise SystemExit(f"cannot open raw: {raw}")
        if not self.cap_ren.isOpened():
            raise SystemExit(f"cannot open rendered: {rendered}")
        self.src_fps = self.cap_raw.get(cv2.CAP_PROP_FPS) or 60.0
        self.fps = float(fps) if fps else self.src_fps
        self.n_raw = max(0, int(self.cap_raw.get(cv2.CAP_PROP_FRAME_COUNT)) - self.raw_start)
        self.n_ren = int(self.cap_ren.get(cv2.CAP_PROP_FRAME_COUNT))

        self.clock = 0        # raw frames consumed since start
        self.ren_idx = 0      # rendered frames consumed
        self.playing = False
        self._imgs: dict = {}     # keep PhotoImage refs alive
        self._raw_last = None     # freeze last raw frame when the feed ends

        self._build_ui()
        self._seek_start()
        self.toggle()             # autoplay

    # ── geometry helpers ─────────────────────────────────────────────────────
    def delay_frames(self) -> int:
        return int(round(self.delay_s * self.fps))

    # ── UI ───────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.root.title("FRC Vision Scout — LIVE stream simulator")
        self.root.geometry("1380x560")
        self.root.configure(bg="#0c0c0c")

        head = tk.Frame(self.root, bg="#0c0c0c"); head.pack(fill="x")
        tk.Label(head, text="●  LIVE  —  raw broadcast feed", font=("Segoe UI", 11, "bold"),
                 fg="#ff5252", bg="#0c0c0c").pack(side="left", padx=(16, 0), pady=6)
        self.delay_lbl = tk.Label(head, text="", font=("Segoe UI", 11, "bold"),
                                  fg="#54c7ec", bg="#0c0c0c")
        self.delay_lbl.pack(side="right", padx=(0, 16))

        panes = tk.Frame(self.root, bg="#0c0c0c"); panes.pack(fill="both", expand=True, padx=10)
        self.left = tk.Label(panes, bg="black"); self.left.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.right = tk.Label(panes, bg="black"); self.right.pack(side="left", fill="both", expand=True, padx=(5, 0))

        ctl = tk.Frame(self.root, bg="#0c0c0c"); ctl.pack(fill="x", pady=6)
        self.btn = tk.Button(ctl, text="⏸ Pause", width=9, command=self.toggle); self.btn.pack(side="left", padx=8)
        tk.Button(ctl, text="⟲ Restart", command=self.restart).pack(side="left", padx=4)
        tk.Label(ctl, text="Δ delay (s):", fg="#bbb", bg="#0c0c0c").pack(side="left", padx=(16, 4))
        self.delay_var = tk.DoubleVar(value=self.delay_s)
        ttk.Scale(ctl, from_=0.0, to=8.0, variable=self.delay_var, length=260,
                  command=lambda v: self._delay_preview()).pack(side="left")
        self.delay_read = tk.Label(ctl, text=f"{self.delay_s:.1f}s  (Restart to apply)",
                                   fg="#54c7ec", bg="#0c0c0c", width=22); self.delay_read.pack(side="left", padx=8)
        self.status = tk.Label(self.root, text="", anchor="w", fg="#9aa", bg="#0c0c0c",
                               font=("Consolas", 9)); self.status.pack(fill="x", padx=12, pady=(0, 6))

        self.root.bind("<space>", lambda e: self.toggle())
        self.root.bind("r", lambda e: self.restart())
        self.root.bind("<Right>", lambda e: self._nudge(1.25))
        self.root.bind("<Left>", lambda e: self._nudge(0.8))

    def _delay_preview(self):
        self.delay_read.config(text=f"{self.delay_var.get():.1f}s  (Restart to apply)")

    def _nudge(self, factor):
        self.speed = max(0.1, min(8.0, self.speed * factor))

    # ── playback ──────────────────────────────────────────────────────────────
    def _seek_start(self):
        self.cap_raw.set(cv2.CAP_PROP_POS_FRAMES, self.raw_start)
        self.cap_ren.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.clock = 0
        self.ren_idx = 0
        self._raw_last = None

    def restart(self):
        self.delay_s = float(self.delay_var.get())
        self._seek_start()
        if not self.playing:
            self.toggle()

    def toggle(self):
        self.playing = not self.playing
        self.btn.config(text="⏸ Pause" if self.playing else "▶ Play")
        if self.playing:
            self.loop()

    def _show(self, widget, frame, tag):
        w = max(320, widget.winfo_width()); h = max(240, widget.winfo_height())
        fh, fw = frame.shape[:2]
        s = min(w / fw, h / fh)
        if s > 0 and abs(s - 1.0) > 1e-3:
            frame = cv2.resize(frame, (max(1, int(fw * s)), max(1, int(fh * s))))
        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self._imgs[tag] = img  # keep ref
        widget.config(image=img)

    def _buffering(self, widget, frac):
        import numpy as np
        w = max(320, widget.winfo_width()); h = max(240, widget.winfo_height())
        canvas = np.zeros((h, w, 3), np.uint8) + 16
        msg = "PROCESSING…  buffering scout output"
        cv2.putText(canvas, msg, (max(10, w // 2 - 230), h // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 200, 240), 2, cv2.LINE_AA)
        bw = int(w * 0.5); x0 = (w - bw) // 2; y0 = h // 2 + 20
        cv2.rectangle(canvas, (x0, y0), (x0 + bw, y0 + 16), (60, 60, 60), 1)
        cv2.rectangle(canvas, (x0, y0), (x0 + int(bw * frac), y0 + 16), (84, 199, 236), -1)
        img = ImageTk.PhotoImage(Image.fromarray(canvas))
        self._imgs["rbuf"] = img
        widget.config(image=img)

    def loop(self):
        if not self.playing:
            return
        df = self.delay_frames()

        # LEFT — the live feed advances every tick until the raw segment ends
        if self.clock < self.n_raw:
            ok, fr = self.cap_raw.read()
            if ok:
                self._raw_last = fr
                self._show(self.left, fr, "left")
            self.clock += 1
        elif self._raw_last is not None:
            fr = self._raw_last.copy()
            cv2.putText(fr, "LIVE FEED ENDED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (80, 80, 255), 2, cv2.LINE_AA)
            self._show(self.left, fr, "left")

        # RIGHT — trails by `df` frames; buffers until then
        if self.clock > df:
            if self.ren_idx < self.n_ren:
                ok2, fr2 = self.cap_ren.read()
                if ok2:
                    self._show(self.right, fr2, "right")
                    self.ren_idx += 1
        else:
            self._buffering(self.right, min(1.0, self.clock / max(1, df)))

        live_t = self.clock / self.fps
        scout_t = max(0, self.ren_idx) / self.fps
        gap = live_t - scout_t
        self.delay_lbl.config(text=f"SCOUTED  ◀ Δ {self.delay_s:.1f}s behind  ({self.speed:.1f}× speed)")
        self.status.config(text=(f"LIVE t={live_t:5.1f}s  frame {self.raw_start + self.clock}   |   "
                                 f"SCOUTED t={scout_t:5.1f}s   |   live gap {gap:4.1f}s   |   "
                                 f"{'BUFFERING' if self.clock <= df else 'streaming'}"))

        more = (self.clock < self.n_raw) or (self.ren_idx < self.n_ren)
        if more:
            self.root.after(max(1, int(1000 / (self.fps * self.speed))), self.loop)
        else:
            self.playing = False
            self.btn.config(text="▶ Play")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/footage/qual59.mp4")
    ap.add_argument("--raw-start", type=int, default=0)
    ap.add_argument("--rendered", required=True)
    ap.add_argument("--delay", type=float, default=2.0, help="processing delay in seconds")
    ap.add_argument("--fps", type=float, default=None, help="playback fps (default = source fps)")
    ap.add_argument("--speed", type=float, default=1.0, help="real-time multiplier (1.0 = real-time)")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(here, "..")
    raw = a.raw if os.path.isabs(a.raw) else os.path.join(base, a.raw)
    rendered = a.rendered if os.path.isabs(a.rendered) else os.path.join(base, a.rendered)
    root = tk.Tk()
    LiveSim(root, raw, a.raw_start, rendered, a.delay, a.fps, a.speed)
    root.mainloop()


if __name__ == "__main__":
    main()
