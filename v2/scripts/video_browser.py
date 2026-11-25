"""video_browser.py — click through all the FRC Vision Scout demo clips in ONE window.

    python scripts/video_browser.py

Left = clip list (newest first). Right = player. Space=play/pause, ←/→ = step frame,
the slider scrubs, Prev/Next switch clips. No per-file opening needed.
"""
from __future__ import annotations

import glob
import os
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "scout_out")

# Friendly labels for the known clips (falls back to the filename stem).
LABELS = {
    "qual59_air": "qual59 — ★ LATEST: airborne SHOT recall fix (net-detected balls + velocity gate) — heavy-scoring window",
    "qual59_FINAL": "qual59 — 6 robots red/blue · pile area-est · airborne shots · HUB bins · intake/hold/shoot",
    "qual59_state2": "qual59 — interaction states (flight-gated SHOOTING) + HUB bins",
    "m1080_rob": "match_1080 — robot fix GENERALIZES: 6/6 detected, red/blue",
    "qual59_robots3": "qual59 — robot double-gate fix (1-2 → 5 robots)",
    "qual59_v7": "qual59 — balls: crowd-clean + pile area-estimate + airborne SHOT tracking",
    "qual59_v6": "qual59 — pile area-estimate (before airborne-marker fix)",
    "qual59_v4": "qual59 — field-gated robots + fused balls + geo (no clutter)",
    "qual59_v3": "qual59 — flight-clutter fix",
    "qual59_v2": "qual59 — first integration (field gate + classical balls)",
    "qual56_full": "qual56 — every-feature (with field minimap)",
    "qual56_geo": "qual56 — real geolocation",
    "qual56_demo_h264": "qual56 — main demo (H264, plays anywhere)",
    "qual59_demo_h264": "qual59 — earlier demo (H264)",
    "qual56_demo": "qual56 — earlier (mp4v)",
    "qual59_demo": "qual59 — earlier (mp4v)",
}


def label_for(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return LABELS.get(stem, stem)


class Browser:
    def __init__(self, root: tk.Tk, videos: list[str]):
        self.root = root
        self.videos = videos
        self.cap = None
        self.playing = False
        self.fps = 30.0
        self.total = 1
        self.cur = None
        self._scrub_user = True

        root.title("FRC Vision Scout — demo browser")
        root.geometry("1200x780")
        root.configure(bg="#111")

        left = tk.Frame(root, width=340, bg="#1b1b1b")
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        tk.Label(left, text="CLIPS  (newest first)", font=("Segoe UI", 11, "bold"),
                 fg="#eee", bg="#1b1b1b").pack(anchor="w", padx=10, pady=(10, 4))
        self.lb = tk.Listbox(left, font=("Segoe UI", 9), activestyle="dotbox",
                             bg="#222", fg="#ddd", selectbackground="#3a6ea5",
                             highlightthickness=0, bd=0)
        for v in videos:
            self.lb.insert("end", "  " + label_for(v))
        self.lb.pack(fill="both", expand=True, padx=8, pady=6)
        self.lb.bind("<<ListboxSelect>>", self._on_select)

        right = tk.Frame(root, bg="#111")
        right.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Label(right, bg="black")
        self.canvas.pack(fill="both", expand=True)

        ctl = tk.Frame(right, bg="#111")
        ctl.pack(fill="x", pady=5)
        self.btn_play = tk.Button(ctl, text="▶ Play", width=9, command=self.toggle)
        self.btn_play.pack(side="left", padx=4)
        tk.Button(ctl, text="⟲ Restart", command=self.restart).pack(side="left", padx=4)
        tk.Button(ctl, text="◀ Prev", command=lambda: self.step_clip(-1)).pack(side="left", padx=4)
        tk.Button(ctl, text="Next ▶", command=lambda: self.step_clip(1)).pack(side="left", padx=4)
        self.scrub = ttk.Scale(ctl, from_=0, to=100, command=self._on_scrub)
        self.scrub.pack(side="left", fill="x", expand=True, padx=8)

        self.status = tk.Label(right, text="select a clip", anchor="w",
                               fg="#bbb", bg="#111", font=("Segoe UI", 9))
        self.status.pack(fill="x", padx=6, pady=(0, 4))

        root.bind("<space>", lambda e: self.toggle())
        root.bind("<Right>", lambda e: self.step_frame(1))
        root.bind("<Left>", lambda e: self.step_frame(-1))

        if videos:
            self.lb.selection_set(0)
            self.load(0)

    def load(self, idx: int):
        if self.cap is not None:
            self.cap.release()
        self.cur = idx
        self.cap = cv2.VideoCapture(self.videos[idx])
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        self.scrub.configure(to=max(1, self.total - 1))
        self.playing = True
        self.btn_play.config(text="⏸ Pause")
        self.loop()

    def _on_select(self, _e):
        sel = self.lb.curselection()
        if sel and sel[0] != self.cur:
            self.load(sel[0])

    def step_clip(self, d: int):
        if self.cur is None:
            return
        i = (self.cur + d) % len(self.videos)
        self.lb.selection_clear(0, "end")
        self.lb.selection_set(i)
        self.lb.see(i)
        self.load(i)

    def toggle(self):
        self.playing = not self.playing
        self.btn_play.config(text="⏸ Pause" if self.playing else "▶ Play")
        if self.playing:
            self.loop()

    def restart(self):
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._render()

    def step_frame(self, d: int):
        if self.cap is None:
            return
        f = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) + d - 1
        f = max(0, min(self.total - 1, f))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        self.playing = False
        self.btn_play.config(text="▶ Play")
        self._render()

    def _on_scrub(self, val):
        if not self._scrub_user or self.cap is None:
            return
        f = max(0, min(self.total - 1, int(float(val))))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        self._render()

    def _render(self) -> bool:
        if self.cap is None:
            return False
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.playing = False
            self.btn_play.config(text="▶ Play")
            return False
        f = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        cw = max(320, self.canvas.winfo_width())
        ch = max(240, self.canvas.winfo_height())
        h, w = frame.shape[:2]
        scale = min(cw / w, ch / h) if (w and h) else 1.0
        if scale > 0 and abs(scale - 1.0) > 1e-3:
            frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.tkimg = ImageTk.PhotoImage(img)
        self.canvas.config(image=self.tkimg)
        self._scrub_user = False
        self.scrub.set(f)
        self._scrub_user = True
        self.status.config(
            text=f"{label_for(self.videos[self.cur])}    |    frame {f}/{self.total}    {self.fps:.0f} fps")
        return True

    def loop(self):
        if not self.playing:
            return
        if self._render():
            self.root.after(int(1000 / max(1.0, self.fps)), self.loop)


def main():
    videos = sorted(glob.glob(os.path.join(OUT, "*.mp4")), key=os.path.getmtime, reverse=True)
    if not videos:
        print("No .mp4 clips found in", os.path.abspath(OUT))
        return
    root = tk.Tk()
    Browser(root, videos)
    root.mainloop()


if __name__ == "__main__":
    main()
