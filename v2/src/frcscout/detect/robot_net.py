"""Robot detector — tiny YOLO11n INT8 on the NPU, every frame. 6 large objects.

Alliance (red/blue) is NOT a separate class — it's a post-hoc bumper-color vote on the
detected box, so the net stays single-class and cheap. This was a v1 failure mode
(YOLO-World produced 0 robot labels); a dedicated trained single-class net + color
head is robust and small.
"""
from __future__ import annotations

import numpy as np

from frcscout.io.types import Alliance, Detection, Track


class RobotNet:
    def __init__(self, cfg: dict):
        self.cfg = cfg or {}
        self.imgsz = self.cfg.get("imgsz", 512)
        self.conf = self.cfg.get("conf", 0.30)
        self.device = self.cfg.get("device", "intel:npu")
        # When pointed at a MULTI-class model (e.g. the YOLO11s ball model whose `robot`
        # class generalises far better than the single-class robot net), keep only this
        # class id. None = single-class model, keep every box.
        self.class_id = self.cfg.get("class_id", None)
        self._model = None  # lazy — OpenVINO compile happens on first infer

    def _ensure(self):
        if self._model is None:
            from ultralytics import YOLO   # ultralytics wraps the OpenVINO runtime
            self._model = YOLO(self.cfg["model"], task="detect")

    def infer(self, frame, frame_idx: int) -> list[Detection]:
        self._ensure()
        r = self._model.predict(frame, imgsz=self.imgsz, conf=self.conf,
                                device=self.device, verbose=False)[0]
        out: list[Detection] = []
        if r.boxes is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = (r.boxes.cls.cpu().numpy().astype(int)
                   if (self.class_id is not None and r.boxes.cls is not None) else None)
            for i, b in enumerate(xyxy):
                if cls is not None and cls[i] != self.class_id:
                    continue
                # conf forced to 1.0: the predict() conf gate already filtered, and the
                # downstream 6-robot prior (not a per-box threshold) handles spurious hits.
                out.append(Detection("robot", tuple(map(float, b)), 1.0, frame_idx))
        return out

    @staticmethod
    def bumper_alliance(frame, xyxy) -> Alliance | None:
        """Alliance from the bumper *ring* — calibrated to ABSTAIN when unclear.

        Bumpers form a saturated red/blue band around the robot's lower perimeter.
        We sample a border ring (dropping the inner mechanism core, which is grey/
        black/ball-yellow and only dilutes the vote) and score red vs blue with
        BOTH HSV hue and the LAB a*/b* axes — LAB survives the broadcast colour
        wash that collapses HSV saturation on small overhead robots (the camouflage
        failure: blue bumpers on blue walls, red on red ramps).

        Returns None (abstain) unless a clear winner owns a real share of the
        coloured pixels AND decisively beats the other alliance. The per-track
        temporal vote in bytetrack only counts non-None frames, so abstaining on
        ambiguous frames is strictly better than coin-flipping.
        """
        import cv2
        x1, y1, x2, y2 = (int(v) for v in xyxy)
        if x2 - x1 < 6 or y2 - y1 < 6:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        H, W = crop.shape[:2]

        # Bumper ring: lower ~65% of the box, perimeter only (carve the core out).
        ring = np.zeros((H, W), dtype=bool)
        top = int(H * 0.35)
        ring[top:, :] = True
        my, mx = int(H * 0.22), int(W * 0.22)
        if H - 2 * my > top and W - 2 * mx > 0:
            ring[top + my:H - my, mx:W - mx] = False
        if ring.sum() < 12:                 # tiny box — fall back to the lower band
            ring[:] = False
            ring[top:, :] = True

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        a = lab[..., 1].astype(np.int16) - 128   # >0 red, <0 green
        b = lab[..., 2].astype(np.int16) - 128   # >0 yellow, <0 blue

        # Lowered sat floor (washed bumpers) — LAB carries the weak-colour signal.
        colored = ring & (s > 60) & (v > 45)
        n = int(colored.sum())
        if n < 20 or n < 0.015 * H * W:
            return None

        red = int((colored & ((h <= 12) | (h >= 168))).sum()) \
            + int((colored & (a >= 20) & (b > -15)).sum())      # red, not blue-purple
        blue = int((colored & (h >= 95) & (h <= 135)).sum()) \
            + int((colored & (b <= -18)).sum())                 # decisively blue

        hi, lo = max(red, blue), min(red, blue)
        if hi < 0.20 * n:            # winner must own a real share of coloured px
            return None
        if hi < 1.5 * max(lo, 1):    # and clearly beat the other alliance, else abstain
            return None
        return "red" if red > blue else "blue"
