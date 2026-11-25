"""
FRC Vision Scout — LIVE inference on Intel CPU / Arc iGPU / NPU via OpenVINO (no CUDA).

Runs the custom-trained FUEL+robot detector (exported to OpenVINO INT8 on Colab) in real time
on the Intel Core Ultra. FUEL = yellow circles, robots = boxes with persistent track IDs.

Setup (one time, on your machine):
    pip install ultralytics openvino supervision opencv-python

Usage:
    # live window on the integrated Arc GPU (recommended), playing a file or a live stream URL
    python src/live_openvino.py frc_model_openvino_model --device intel:gpu --source data/match_1080p.mp4

    # benchmark all Intel devices on one clip (no window) to pick the fastest
    python src/live_openvino.py frc_model_openvino_model --benchmark --source data/match_1080p.mp4

    # contest day: point --source at the live HLS/stream URL (yt-dlp -g '<live url>')
    python src/live_openvino.py frc_model_openvino_model --device intel:gpu --source "<stream_url>" --save out.mp4

Device strings: intel:cpu | intel:gpu (Arc iGPU) | intel:npu (AI Boost). iGPU is usually the
best speed/quality balance; NPU is most power-efficient; CPU is the universal fallback.
"""
import argparse, time, sys
import cv2
import numpy as np

BALL_COL = (0, 255, 255)    # yellow
ROBOT_COL = (255, 90, 0)    # blue-ish


def load(model_dir, imgsz):
    from ultralytics import YOLO
    m = YOLO(model_dir, task="detect")
    return m


def benchmark(model_dir, source, imgsz, conf, n_frames=200):
    """Time each Intel device on the same clip and print fps."""
    from ultralytics import YOLO
    for dev in ("intel:cpu", "intel:gpu", "intel:npu"):
        try:
            m = YOLO(model_dir, task="detect")
            cap = cv2.VideoCapture(source)
            # warmup
            ok, fr = cap.read()
            if not ok:
                print(f"{dev}: cannot read source"); cap.release(); continue
            for _ in range(3):
                m.predict(fr, imgsz=imgsz, conf=conf, device=dev, verbose=False)
            t0 = time.time(); i = 0
            while i < n_frames:
                ok, fr = cap.read()
                if not ok:
                    break
                m.predict(fr, imgsz=imgsz, conf=conf, device=dev, verbose=False)
                i += 1
            cap.release()
            dt = time.time() - t0
            print(f"{dev:>10}: {i/dt:6.1f} fps  ({i} frames, {dt:.1f}s) @ imgsz {imgsz}")
        except Exception as e:
            print(f"{dev:>10}: unavailable — {str(e)[:90]}")


def run_live(model_dir, source, device, imgsz, conf, save):
    m = load(model_dir, imgsz)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Cannot open source: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(3)); H = int(cap.get(4)); cap.release()

    writer = None
    if save:
        writer = cv2.VideoWriter(save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    # ultralytics handles tracking + stream decode; ByteTrack is lightweight (good on CPU)
    results = m.track(source=source, stream=True, device=device, imgsz=imgsz,
                      conf=conf, iou=0.5, tracker="bytetrack.yaml", persist=True, verbose=False)

    t0 = time.time(); i = 0; ema = None
    show = not save  # if saving, skip the window (e.g. headless / stream)
    for r in results:
        f = r.orig_img
        nb = nr = 0
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            ids = (r.boxes.id.cpu().numpy().astype(int)
                   if r.boxes.id is not None else [-1] * len(xyxy))
            for (x1, y1, x2, y2), c, tid in zip(xyxy.astype(int), cls, ids):
                if c == 0:  # ball
                    cv2.circle(f, ((x1 + x2) // 2, (y1 + y2) // 2),
                               max(3, (x2 - x1) // 2), BALL_COL, 2); nb += 1
                else:       # robot
                    cv2.rectangle(f, (x1, y1), (x2, y2), ROBOT_COL, 2)
                    cv2.putText(f, f"R{tid}", (x1, y1 - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ROBOT_COL, 1); nr += 1
        i += 1
        inst = i / (time.time() - t0)
        ema = inst if ema is None else 0.9 * ema + 0.1 * inst
        cv2.putText(f, f"{device}  {ema:4.1f} fps   FUEL {nb}  ROBOTS {nr}",
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if writer:
            writer.write(f)
        if show:
            cv2.imshow("FRC Vision Scout — live (q to quit)", f)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    if writer:
        writer.release()
    if show:
        try: cv2.destroyAllWindows()
        except cv2.error: pass
    print(f"done — {i} frames, mean {i/(time.time()-t0):.1f} fps on {device}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to *_openvino_model dir (unzipped from Colab)")
    ap.add_argument("--source", default="data/match_1080p.mp4", help="video file or stream URL")
    ap.add_argument("--device", default="intel:gpu",
                    help="intel:cpu | intel:gpu | intel:npu")
    ap.add_argument("--imgsz", type=int, default=768, help="must match DEPLOY_IMGSZ from export")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--save", default="", help="write annotated mp4 instead of showing a window")
    ap.add_argument("--benchmark", action="store_true", help="time all Intel devices and exit")
    a = ap.parse_args()
    if a.benchmark:
        benchmark(a.model, a.source, a.imgsz, a.conf)
    else:
        run_live(a.model, a.source, a.device, a.imgsz, a.conf, a.save)
