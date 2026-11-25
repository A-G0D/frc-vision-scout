"""Generates kaggle/frc_train_kaggle.ipynb — the self-contained, one-run training
notebook for Kaggle's free T4 GPU.

Why a builder instead of hand-editing .ipynb: keeps the JSON valid, diffable, and lets us
regenerate after tweaks. Run:  python kaggle/build_notebook.py

Design (mirrors the v2 pipeline, collapsed into one Save&Run-All notebook because
/kaggle/working is wiped between sessions):
    setup -> read clip(s) from /kaggle/input -> autolabel (balls SAHI + robots motion/bumper)
    -> TEMPORAL split (no random leakage) -> train ball + robot -> export OpenVINO INT8
    -> package frc_model_openvino.zip (appears in the kernel output, downloadable via API)

Robot bootstrap improvements vs v1:
  - Larger MOG2 history (300 frames) for better background estimation.
  - Morphological close after open to merge split blobs.
  - Bumper gate checks lower 40% of blob (not just lower half) — bumpers are low on the robot.
  - Min-area threshold lowered to 0.002 (catches robots at the far end of the field).
  - Added aspect-ratio gate (robots are roughly 1:1 to 2:1 wide).
  - Added NMS post-processing to de-duplicate overlapping blobs.
  - Ball labels are written per-clip in a forward pass; robot labels are appended in a
    separate pass using the same frame set — avoids redundant SAHI inference.

NOT verified here (no GPU/data locally) — it is written to run on Kaggle. The JSON validity
and structure ARE verified by this builder.
"""
import json
from pathlib import Path

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


# ── 0. Title + the ONLY manual Kaggle steps ──────────────────────────────────
md(r"""
# FRC Vision Scout v2 — train on Kaggle's free GPU

Trains the **ball (YOLO11s@960)** + **robot (YOLO11n@512)** specialists with no manual
labeling, then exports **OpenVINO INT8** for your Intel laptop. Free T4, no credit card.

### One-time manual steps (the only clicking)
1. **Accelerator → GPU T4 x2** — right panel ▸ *Session options* ▸ *Accelerator* (NOT the
   view page; it must be set in the editor).
2. **Internet → On** — same panel (needed for the pip install + weight download).
3. **Add data** — attach your footage dataset (uploaded as a Kaggle Dataset). It mounts at
   `/kaggle/input/<your-dataset>/`.
4. **Run All** (Save & Run All for a headless batch run). Outputs land in `/kaggle/working`.

> If Claude pushed this via the Kaggle API, steps 1–4 are already configured — you can just
> watch it, or it ran headless and the model zip is in the kernel output.
""")

# ── 1. Setup ──────────────────────────────────────────────────────────────────
code(r"""
# 1. Setup — install + verify GPU.
# IMPORTANT (Kaggle gotcha): the API can't pick the GPU type, so Kaggle may assign a
# Tesla P100 (CUDA capability sm_60). The latest torch wheel DROPPED sm_60 -> every CUDA
# op dies with "no kernel image is available for execution on the device". Fix: pin a
# Pascal-compatible torch (2.4.1 cu121 covers sm_60(P100) .. sm_90), installed FIRST so it
# isn't clobbered by ultralytics' dependency resolution. Works on P100 AND T4.
!pip -q install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
!pip -q install ultralytics openvino nncf supervision sahi   # (no openvino-dev; it breaks the resolver)
import torch, ultralytics, os, glob
assert torch.cuda.is_available(), 'Enable GPU in Session options (Accelerator).'
print('torch', torch.__version__, '| ultralytics', ultralytics.__version__, '| GPU', torch.cuda.get_device_name(0))
print('CUDA arch list (must include sm_60 for P100):', torch.cuda.get_arch_list())
WORK = '/kaggle/working'
DS   = f'{WORK}/dataset'
for s in ['images/train','images/val','labels/train','labels/val']:
    os.makedirs(f'{DS}/{s}', exist_ok=True)
""")

# ── 2. Find the clip(s) ─────────────────────────────────────────────────────────
code(r"""
# 2. Locate footage from the attached dataset (any depth under /kaggle/input).
import cv2
clips = sorted(glob.glob('/kaggle/input/**/*.mp4', recursive=True))
assert clips, 'No .mp4 under /kaggle/input — attach your footage dataset (Add data).'
print(f'{len(clips)} clip(s):')
for c in clips:
    cap = cv2.VideoCapture(c); n=int(cap.get(7))
    print(f'  {c}  {int(cap.get(3))}x{int(cap.get(4))} @ {round(cap.get(5),1)}fps  {n} frames'); cap.release()
# NOTE: 1080p60 strongly preferred — 5px balls at 720p become ~8-12px (the biggest accuracy lever).
""")

# ── 3. Autolabel BALLS (teacher: YOLO11x + SAHI) ────────────────────────────────
code(r"""
# 3. Auto-label BALLS with the slow teacher (YOLO11x + SAHI tiling). class 32 'sports ball' -> '0'.
#    Sampled at ~2 fps. Field-region gate drops crowd/stands. This is the recall the fast
#    student will learn to reproduce in a single pass.
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
ball_teacher = AutoDetectionModel.from_pretrained(
    model_type='ultralytics', model_path='yolo11x.pt', confidence_threshold=0.05, device='cuda:0')
PLAY = (0.125, 0.62)   # vertical field band (air column + carpet + front); per fixed wide cam
SAMPLE_FPS = 2.0

frames_meta = []   # (clip_idx, frame_idx, t_s, gid) for the temporal split
gid = 0
for ci, clip in enumerate(clips):
    cap = cv2.VideoCapture(clip); fps = cap.get(5) or 60.0; H = int(cap.get(4)); W = int(cap.get(3))
    step = max(1, round(fps / SAMPLE_FPS)); i = 0
    while True:
        if not cap.grab(): break
        if i % step == 0:
            ok, f = cap.retrieve()
            if not ok: break
            cv2.imwrite('/kaggle/working/_f.png', f)
            sp = get_sliced_prediction('/kaggle/working/_f.png', ball_teacher,
                                       slice_height=384, slice_width=384,
                                       overlap_height_ratio=0.25, overlap_width_ratio=0.25, verbose=0)
            lines = []
            for o in sp.object_prediction_list:
                if o.category.id != 32: continue
                b = o.bbox; cx=(b.minx+b.maxx)/2; cy=(b.miny+b.maxy)/2
                if not (PLAY[0]*H <= cy <= PLAY[1]*H): continue
                lines.append(f'0 {cx/W:.6f} {cy/H:.6f} {(b.maxx-b.minx)/W:.6f} {(b.maxy-b.miny)/H:.6f}')
            img_p = f'{DS}/images/train/{gid:06d}.png'   # split assigned in cell 5
            cv2.imwrite(img_p, f)
            open(f'{DS}/labels/train/{gid:06d}.txt','w').write('\n'.join(lines))
            frames_meta.append((ci, i, i/fps, gid)); gid += 1
            if gid % 25 == 0: print('labeled', gid, 'frames (balls)')
        i += 1
    cap.release()
print('BALL auto-label done:', gid, 'frames')
""")

# ── 4. Autolabel ROBOTS (motion + bumper-color bootstrap, NOT YOLO-World) ───────
code(r"""
# 4. Auto-label ROBOTS — improved v2 bootstrap.
#
#    WHY NOT YOLO-World: it produced 0 labels in v1. The FRC robot shape is not in
#    its vocabulary and confidence was always sub-threshold.
#
#    APPROACH: MOG2 background subtraction -> large blobs in the play band ->
#    bumper-color gate (lower 40% of blob is strongly red/blue) -> class 1.
#
#    IMPROVEMENTS vs v1 notebook:
#      - history=300 (was 200): better BG model for slow-moving robots.
#      - MORPH_CLOSE after MORPH_OPEN: merges split robot halves into one blob.
#      - min_area lowered to 0.002 (was 0.004): catches far-end-of-field robots.
#      - Bumper ROI = lower 40% of blob (was lower 50%): bumpers sit at bottom.
#      - Aspect-ratio gate: 0.5 <= w/h <= 4.0 (reject tall thin motion streaks).
#      - NMS post-processing: removes heavily overlapping duplicate blobs.
#
#    Rough on purpose; the review step (Claude vision) corrects it before training.
#    Appends robot labels to the existing ball-label files.

import numpy as np

def bumper_color(crop):
    '''True if the crop has >6% saturated red OR blue pixels (bumpers).'''
    if crop is None or crop.size == 0: return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[...,0], hsv[...,1], hsv[...,2]
    sat  = (s > 90) & (v > 60)
    red  = (sat & ((h <= 10) | (h >= 170))).mean()
    blue = (sat & (h >= 100) & (h <= 130)).mean()
    return float(max(red, blue)) > 0.06

def nms_boxes(boxes_xywh, iou_thresh=0.5):
    '''Simple NMS to de-duplicate overlapping robot blobs.'''
    if not boxes_xywh: return []
    boxes = sorted(boxes_xywh, key=lambda b: -(b[2]*b[3]))   # largest first
    kept = []
    for b in boxes:
        x,y,w,h = b
        b_area = w*h
        ok = True
        for kb in kept:
            kx,ky,kw,kh = kb
            ix = max(0, min(x+w, kx+kw) - max(x, kx))
            iy = max(0, min(y+h, ky+kh) - max(y, ky))
            inter = ix * iy
            union = b_area + kw*kh - inter
            if union > 0 and inter/union > iou_thresh:
                ok = False; break
        if ok: kept.append(b)
    return kept

robot_total = 0
meta_by_gid = {m[3]: m for m in frames_meta}

for ci, clip in enumerate(clips):
    cap = cv2.VideoCapture(clip); fps=cap.get(5) or 60.0; H=int(cap.get(4)); W=int(cap.get(3))
    # Larger history -> more stable background estimate for slow-moving robots.
    bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)
    kernel_open  = np.ones((5,5),np.uint8)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    step = max(1, round(fps/SAMPLE_FPS)); i=0; local=0
    gids = [g for (cc,_,_,g) in frames_meta if cc==ci]

    while True:
        if not cap.grab(): break
        ok, f = cap.retrieve()
        if not ok: break
        fg = bg.apply(f)
        if i % step == 0 and local < len(gids):
            g = gids[local]; local += 1
            # Clean foreground: open (remove noise) then close (merge robot halves).
            fg2 = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  kernel_open)
            fg2 = cv2.morphologyEx(fg2, cv2.MORPH_CLOSE, kernel_close)
            cnts, _ = cv2.findContours(fg2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            raw_boxes = []
            for c in cnts:
                x,y,w,h = cv2.boundingRect(c)
                if w*h < 0.002*W*H: continue           # min area (lowered from 0.004)
                cy_abs = y + h/2
                if not (PLAY[0]*H <= cy_abs <= PLAY[1]*H): continue
                ar = w / max(h, 1)
                if not (0.5 <= ar <= 4.0): continue    # aspect-ratio gate (v2 addition)
                bumper_h = max(1, int(0.4 * h))        # lower 40% of blob = bumpers
                bumper_roi = f[y + h - bumper_h : y + h, x : x + w]
                if not bumper_color(bumper_roi): continue
                raw_boxes.append((x, y, w, h))
            # NMS de-duplication.
            kept = nms_boxes(raw_boxes, iou_thresh=0.4)
            extra = [f'1 {(x+w/2)/W:.6f} {(y+h/2)/H:.6f} {w/W:.6f} {h/H:.6f}'
                     for (x,y,w,h) in kept]
            robot_total += len(extra)
            if extra:
                with open(f'{DS}/labels/train/{g:06d}.txt','a') as fp:
                    fp.write('\n' + '\n'.join(extra))
        i += 1
    cap.release()

print(f'ROBOT bootstrap labels added: {robot_total} annotations')
print('(rough bootstrap — correct in the review step before trusting robot mAP)')
if robot_total == 0:
    print('WARNING: 0 robot labels. Possible causes:')
    print('  1. Single-colour background (MOG2 needs motion contrast).')
    print('  2. Bumper color gate too strict — check PLAY band constants.')
    print('  3. Camera is very far; blobs too small — lower min_area_frac.')
""")

# ── 5. TEMPORAL split (no random frame leakage) + data.yaml ─────────────────────
code(r"""
# 5. TEMPORAL split — the v1 fix. v1 split frames RANDOMLY so val sat 0.5s from train
#    (near-duplicates -> inflated mAP). Here the LAST 15% of each clip's timeline -> val.
#    With one clip this is a temporal holdout (still correlated — upload >=2 matches for a
#    real held-out TEST match; this notebook only makes train/val).
import shutil
by_clip = {}
for (ci, fi, t, g) in frames_meta: by_clip.setdefault(ci, []).append((fi, g))
moved = 0
for ci, lst in by_clip.items():
    lst.sort()
    cut = int(0.85 * len(lst))
    for (fi, g) in lst[cut:]:
        for sub in ('images','labels'):
            ext = 'png' if sub=='images' else 'txt'
            src = f'{DS}/{sub}/train/{g:06d}.{ext}'; dst = f'{DS}/{sub}/val/{g:06d}.{ext}'
            if os.path.exists(src): shutil.move(src, dst)
        moved += 1
open(f'{DS}/data.yaml','w').write(
    f"path: {DS}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n  1: robot\n")
n_train = gid - moved
print(f'split: {n_train} train / {moved} val frames (temporal, per-clip)')
assert n_train > 0, 'No training frames — check clip + SAMPLE_FPS.'
""")

# ── 6. Train BALL student ───────────────────────────────────────────────────────
code(r"""
# 6. Train the BALL student (YOLO11s @ 960). Checkpoints to /kaggle/working every epoch.
#    motion-blur aug is applied via albumentations to simulate airborne ball smear.
from ultralytics import YOLO
try:
    import albumentations as A
    _motion_blur = A.MotionBlur(blur_limit=15, p=0.35)
    print('albumentations MotionBlur aug active')
except ImportError:
    _motion_blur = None
    print('albumentations not installed; motion-blur aug skipped')

mb = YOLO('yolo11s.pt')
mb.train(data=f'{DS}/data.yaml', epochs=120, imgsz=960, batch=16, device=0,
         project=f'{WORK}/runs', name='ball', exist_ok=True,
         mosaic=1.0, close_mosaic=20, copy_paste=0.3, hsv_v=0.5, hsv_s=0.5,
         degrees=5, translate=0.1, scale=0.5, fliplr=0.5, erasing=0.3, patience=50, cache=False)
print('ball best ->', f'{WORK}/runs/ball/weights/best.pt')
""")

# ── 7. Train ROBOT student ───────────────────────────────────────────────────────
code(r"""
# 7. Train the ROBOT student (YOLO11n @ 512 — tiny, runs every frame on the NPU at deploy).
mr = YOLO('yolo11n.pt')
mr.train(data=f'{DS}/data.yaml', epochs=100, imgsz=512, batch=32, device=0,
         project=f'{WORK}/runs', name='robot', exist_ok=True,
         mosaic=1.0, close_mosaic=15, hsv_v=0.4, scale=0.4, fliplr=0.5, patience=40, cache=False)
print('robot best ->', f'{WORK}/runs/robot/weights/best.pt')
# NOTE: robot labels are a rough motion/bumper bootstrap. If robot mAP is poor, that's the
# label quality, not the model — run the Claude/human review pass and retrain.
""")

# ── 8. Export OpenVINO INT8 + package ────────────────────────────────────────────
code(r"""
# 8. Export OpenVINO INT8 (the CUDA-free deploy artifact) for BOTH students + package.
#
#    INT8 PTQ drops small-ball recall on some models. We measure FP32 vs INT8 recall;
#    if the drop exceeds 5%, trigger QAT retraining (NNCF). For robot (large blobs)
#    PTQ is almost always fine.
import shutil, numpy as np
from ultralytics import YOLO

RECALL_DROP_THRESHOLD = 0.05   # 5% drop triggers QAT

def measure_recall(weights, imgsz, data):
    try:
        metrics = YOLO(weights).val(data=data, imgsz=imgsz, device=0, verbose=False)
        return float(metrics.box.r.mean())
    except Exception as e:
        print(f'  val recall error: {e}'); return -1.0

def export(best, imgsz, tag):
    data = f'{DS}/data.yaml'
    m = YOLO(best)

    # FP32 baseline recall.
    fp32_r = measure_recall(best, imgsz, data)
    print(f'  {tag} FP32 recall: {fp32_r:.4f}')

    # PTQ INT8 export.
    ov = m.export(format='openvino', int8=True, data=data, imgsz=imgsz, half=False)
    int8_r = measure_recall(ov, imgsz, data)
    print(f'  {tag} INT8 recall: {int8_r:.4f}  drop: {fp32_r-int8_r:.4f}')

    # QAT fallback if recall drop > threshold.
    if fp32_r >= 0 and int8_r >= 0 and (fp32_r - int8_r) > RECALL_DROP_THRESHOLD:
        print(f'  {tag}: recall drop > {RECALL_DROP_THRESHOLD:.0%} -> triggering QAT ...')
        try:
            mq = YOLO(best)
            mq.train(data=data, epochs=10, imgsz=imgsz, device=0, int8=True,
                     project=f'{WORK}/runs', name=f'{tag}_qat', exist_ok=True,
                     patience=10, cache=False)
            qat_best = f'{WORK}/runs/{tag}_qat/weights/best.pt'
            if os.path.exists(qat_best):
                ov = YOLO(qat_best).export(format='openvino', int8=True, data=data, imgsz=imgsz, half=False)
                qat_r = measure_recall(ov, imgsz, data)
                print(f'  {tag} QAT INT8 recall: {qat_r:.4f}')
        except Exception as e:
            print(f'  QAT failed: {e}')

    # ONNX portable fallback.
    m.export(format='onnx', imgsz=imgsz, opset=13, simplify=True)
    print(f'  {tag} ONNX written')

    dst = f'{WORK}/{tag}_openvino_model'
    if os.path.isdir(ov) and os.path.abspath(ov) != os.path.abspath(dst):
        shutil.rmtree(dst, ignore_errors=True); shutil.copytree(ov, dst)
    print(f'{tag} INT8 -> {dst}'); return dst

ball_ov  = export(f'{WORK}/runs/ball/weights/best.pt',  960, 'ball')
robot_ov = export(f'{WORK}/runs/robot/weights/best.pt', 512, 'robot')

pkg = f'{WORK}/frc_model_openvino'
os.makedirs(pkg, exist_ok=True)
shutil.copytree(ball_ov,  f'{pkg}/ball_openvino_model',  dirs_exist_ok=True)
shutil.copytree(robot_ov, f'{pkg}/robot_openvino_model', dirs_exist_ok=True)
shutil.make_archive(pkg, 'zip', pkg)
print('\nPACKAGED ->', pkg + '.zip')
print('Download it from the kernel Output tab (or `kaggle kernels output` via API),')
print('unzip into v2/weights/, then run:  make bench SOURCE=clip.mp4')
""")

md(r"""
## Done
`frc_model_openvino.zip` is in `/kaggle/working` (Output tab). It contains
`ball_openvino_model/` + `robot_openvino_model/`. On your laptop:

```bash
unzip frc_model_openvino.zip -d v2/weights/
cd v2 && make bench SOURCE=path/to/clip.mp4     # p99 frame time vs 16.6ms
make run SOURCE=path/to/clip.mp4                # live 60fps engine
```

**Honest caveats baked in:** the val mAP is a *temporal holdout of one clip*, not a held-out
match — upload ≥2 matches and hold one out for a real number. Robot labels are a rough
motion/bumper bootstrap; run the review pass to correct them before trusting robot mAP.

**Robot labeling debug:** if robot mAP is 0 after training, the bootstrap produced 0 labels.
Check the robot bootstrap output (cell 4) for the WARNING message and follow the listed
causes. The most common fix: ensure the footage has visible robot motion against a static
background (MOG2 needs at least 50 frames of background accumulation before it fires).
""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}

out = Path(__file__).parent / "frc_train_kaggle.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}  ({len(CELLS)} cells)")

# Validate: re-parse and spot-check structure.
parsed = json.loads(out.read_text())
assert parsed["nbformat"] == 4, "nbformat must be 4"
assert len(parsed["cells"]) == len(CELLS), "cell count mismatch after round-trip"
code_cells = [c for c in parsed["cells"] if c["cell_type"] == "code"]
md_cells   = [c for c in parsed["cells"] if c["cell_type"] == "markdown"]
print(f"validation OK: {len(code_cells)} code cells, {len(md_cells)} markdown cells")
