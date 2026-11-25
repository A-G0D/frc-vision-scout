"""Generates ball_dense.ipynb — a FOCUSED, FAST ball-only retrain that fixes the
dense-pile failure (a 20-ball clump reading ZERO).

Key changes vs the main train notebook:
  - SAHI tiles 384->256 + overlap .25->.3  → the teacher resolves individual balls
    inside dense piles, so the student gets dense-pile labels (not one blob/none).
  - copy_paste 0.3->0.5 + mosaic            → synthesize more dense overlapping piles.
  - PLAY band widened slightly (0.10–0.66)  → catch piles at the front barrier.
  - BALL ONLY (robot stays robot2) + 90 epochs → finishes ~3–4 h (well under the cap),
    unlike the 8 h ball+robot run.
Outputs: ball2dense_openvino_model + ball2dense_best.pt + ball2dense_results.csv.
NOT verified locally (no GPU) — JSON validity is verified by this builder.
"""
import json
from pathlib import Path

C = []
def code(t): C.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                       "outputs": [], "source": t.strip("\n").splitlines(keepends=True)})

code(r"""
# Pascal-compatible torch FIRST (Kaggle P100/sm_60) — same pin as the main train
!pip -q install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
!pip -q install ultralytics sahi
import torch, os, glob, cv2, numpy as np
assert torch.cuda.is_available(), 'enable GPU'
print('torch', torch.__version__, '| GPU', torch.cuda.get_device_name(0), '| arch', torch.cuda.get_arch_list()[:3], '...')
WORK = '/kaggle/working'; DS = f'{WORK}/ds'
for s in ['images/train','images/val','labels/train','labels/val']:
    os.makedirs(f'{DS}/{s}', exist_ok=True)
clips = sorted(glob.glob('/kaggle/input/**/*.mp4', recursive=True))
assert clips, 'attach frc-footage'
print(len(clips), 'clips')
""")

code(r"""
# DENSE-PILE auto-label: SMALL SAHI tiles (256, overlap .3) so each ball in a 20-ball
# clump becomes its own label. This is the fix for piles reading ZERO.
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
teach = AutoDetectionModel.from_pretrained(model_type='ultralytics', model_path='yolo11x.pt',
                                           confidence_threshold=0.05, device='cuda:0')
PLAY = (0.10, 0.66); SAMPLE_FPS = 2.0
meta = []; gid = 0
for ci, clip in enumerate(clips):
    cap = cv2.VideoCapture(clip); fps = cap.get(5) or 60.0; H = int(cap.get(4)); W = int(cap.get(3))
    step = max(1, round(fps / SAMPLE_FPS)); i = 0
    while True:
        if not cap.grab(): break
        if i % step == 0:
            ok, f = cap.retrieve()
            if not ok: break
            cv2.imwrite(f'{WORK}/_f.png', f)
            sp = get_sliced_prediction(f'{WORK}/_f.png', teach, slice_height=256, slice_width=256,
                                       overlap_height_ratio=0.3, overlap_width_ratio=0.3, verbose=0)
            lines = []
            for o in sp.object_prediction_list:
                if o.category.id != 32: continue
                b = o.bbox; cx = (b.minx+b.maxx)/2; cy = (b.miny+b.maxy)/2
                if not (PLAY[0]*H <= cy <= PLAY[1]*H): continue
                lines.append(f'0 {cx/W:.6f} {cy/H:.6f} {(b.maxx-b.minx)/W:.6f} {(b.maxy-b.miny)/H:.6f}')
            cv2.imwrite(f'{DS}/images/train/{gid:06d}.png', f)
            open(f'{DS}/labels/train/{gid:06d}.txt', 'w').write('\n'.join(lines))
            meta.append((ci, gid, len(lines))); gid += 1
            if gid % 25 == 0: print('labeled', gid, 'frames')
        i += 1
    cap.release()
tot = sum(m[2] for m in meta)
print(f'DENSE auto-label done: {gid} frames, {tot} balls, mean {tot/max(1,gid):.1f}/frame (was ~159 w/ 384 tiles)')
""")

code(r"""
import shutil
by = {}
for (ci, g, n) in meta: by.setdefault(ci, []).append(g)
mv = 0
for ci, gs in by.items():
    cut = int(0.85 * len(gs))
    for g in gs[cut:]:
        for sub, ext in (('images','png'), ('labels','txt')):
            s = f'{DS}/{sub}/train/{g:06d}.{ext}'; d = f'{DS}/{sub}/val/{g:06d}.{ext}'
            if os.path.exists(s): shutil.move(s, d)
        mv += 1
open(f'{DS}/data.yaml', 'w').write(f"path: {DS}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n")
print(f'split: {gid-mv} train / {mv} val')
assert gid - mv > 0
""")

code(r"""
from ultralytics import YOLO
m = YOLO('yolo11s.pt')
m.train(data=f'{DS}/data.yaml', epochs=90, imgsz=960, batch=16, device=0,
        project=f'{WORK}/runs', name='ball2', exist_ok=True,
        mosaic=1.0, close_mosaic=15, copy_paste=0.5, hsv_v=0.5, hsv_s=0.5,
        degrees=5, translate=0.1, scale=0.5, fliplr=0.5, erasing=0.3, patience=30, cache=False)
best = f'{WORK}/runs/ball2/weights/best.pt'
import shutil
mb = YOLO(best)
ov = mb.export(format='openvino', int8=True, data=f'{DS}/data.yaml', imgsz=960, half=False)
shutil.copytree(ov, f'{WORK}/ball2dense_openvino_model', dirs_exist_ok=True)
shutil.copy(best, f'{WORK}/ball2dense_best.pt')
shutil.copy(f'{WORK}/runs/ball2/results.csv', f'{WORK}/ball2dense_results.csv')
try: print('ball2dense val mAP50-95:', YOLO(best).val(data=f'{DS}/data.yaml').box.map)
except Exception as e: print('val skip', str(e)[:80])
print('DONE -> ball2dense_openvino_model, ball2dense_best.pt, ball2dense_results.csv')
""")

nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "ball_dense.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(C)} cells)")
