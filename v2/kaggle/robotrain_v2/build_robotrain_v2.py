"""Generates robotrain_v2.ipynb — a from-scratch ROBOT detector retrain on a LARGE,
DIVERSE public dataset, fixing the v1 failure where single-class robot nets trained on
~175 same-event heuristic-labeled frames detected ~0 robots on downscaled broadcast.

WHY this should work where v1 didn't:
  - DATA: nithilanr/2024-frc-notedetection-dataset = 32,425 real FRC images (~25,680 train /
    2,522 val / 1,202 test), ~53,626 Roboflow boxes, YOLO format. Diverse events, cameras,
    distances — the generalization v1 lacked.
    !!! The dataset's own data.yaml is WRONG: it claims single-class "0: robot", but the
    labels are actually 2-class -> 0 = NOTE (small, ~0.03 area), 1 = ROBOT (large, ~0.19
    area). VERIFIED via box-geometry over all 28k labels + visual inspection of frames. This
    builder KEEPS only class 1, REMAPS it to 0, DROPS class 0 (note) -> a clean single-class
    robot detector. Note-only frames become background negatives (fewer false positives).
  - RESOLUTION: imgsz=960 (NOT 640). The 640 variant failed on broadcast; the production
    engine runs the robot class @960, so we match that.
  - SCALE: ~110 epochs, batch 16 (fits T4 16GB @960 for YOLO11s), real held-out val split.

Robustness: the dataset is a Roboflow export and the on-disk split layout (flat
`<split>/*.jpg|*.txt` vs `<split>/images` + `<split>/labels`) varies by export version. The
notebook AUTO-DETECTS the layout at runtime, symlinks images+labels into a clean
images/{train,val} + labels/{train,val} tree, and writes its own data.yaml. So it can't
break on a wrong path guess.

Outputs (in /kaggle/working, for a clean `kaggle kernels output` pull):
  robot_v2_best.pt          <- best.pt (use this in the engine; class 0 = robot)
  robot_v2_last.pt          <- last.pt
  robot_v2_results.csv      <- training curves (per-epoch metrics)
  robot_v2_openvino_model/  <- INT8 OpenVINO export @960 (CPU deploy, matches engine)
  robot_v2_plots/           <- results.png, PR/F1 curves, confusion matrix, val batches
  robot_v2_val_metrics.txt  <- final mAP50 / mAP50-95 on the dataset val split

NOT verified locally (no GPU here). JSON validity is verified by this builder + a Kaggle
dry-import check. The class scheme (single `robot`, index 0) is verified from the dataset's
own data.yaml before this kernel is pushed.

Run: python kaggle/robotrain_v2/build_robotrain_v2.py
"""
import json
from pathlib import Path

C = []
def md(t): C.append({"cell_type": "markdown", "metadata": {},
                     "source": t.strip("\n").splitlines(keepends=True)})
def code(t): C.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                       "outputs": [], "source": t.strip("\n").splitlines(keepends=True)})

md(r"""
# FRC robot detector v2 — single-class `robot` @960 on the 32k-image public set
Fixes v1 (≈175 same-event frames, robot mAP50≈0.48, ~0 detections on broadcast).
Dataset: `nithilanr/2024-frc-notedetection-dataset` (attached). Its data.yaml is WRONG
(claims single-class robot); labels are really 2-class: **0=note, 1=robot**. We keep only
robot (src class 1), remap to 0, drop notes. Verified by geometry + visual inspection.
""")

code(r"""
# Torch pinned cu121 — works on Kaggle T4 (Turing/sm_75) AND P100 (Pascal/sm_60).
# Same pin as the proven ball@960 kernel.
!pip -q install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
!pip -q install ultralytics
import torch, os, glob, shutil
from pathlib import Path
assert torch.cuda.is_available(), 'enable GPU on this kernel'
print('torch', torch.__version__, '| GPU', torch.cuda.get_device_name(0),
      '| arch', torch.cuda.get_arch_list()[-3:])
WORK = '/kaggle/working'
""")

code(r"""
# Locate the attached dataset root (don't hard-code the mount name).
cands = sorted(glob.glob('/kaggle/input/*'))
print('inputs:', cands)
ROOT = None
for c in cands:
    # the FRC note/robot set has split dirs train/valid(or val)/test and a data.yaml
    if glob.glob(c + '/data.yaml') or glob.glob(c + '/**/data.yaml', recursive=True):
        ROOT = c; break
if ROOT is None and cands:
    ROOT = cands[0]
assert ROOT, 'no dataset attached — add nithilanr/2024-frc-notedetection-dataset as input'
# if data.yaml sits one level down, descend to it
dy = glob.glob(ROOT + '/data.yaml') or glob.glob(ROOT + '/**/data.yaml', recursive=True)
if dy:
    ROOT = str(Path(dy[0]).parent)
print('DATASET ROOT =', ROOT)
print('source data.yaml:')
print(open(dy[0]).read() if dy else '(none found — will infer)')
""")

code(r"""
# AUTO-DETECT split layout and build a clean YOLO tree via symlinks (no copy = fast, no extra disk).
# Handles BOTH Roboflow layouts:
#   (A) flat:     <split>/*.jpg  +  <split>/*.txt   (labels beside images)
#   (B) subdirs:  <split>/images/*.jpg  +  <split>/labels/*.txt
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
DS = f'{WORK}/rds'
for s in ('images/train', 'images/val', 'labels/train', 'labels/val'):
    os.makedirs(f'{DS}/{s}', exist_ok=True)

def find_split_dir(root, names):
    for n in names:
        p = os.path.join(root, n)
        if os.path.isdir(p):
            return p
    return None

train_src = find_split_dir(ROOT, ['train'])
val_src   = find_split_dir(ROOT, ['valid', 'val'])
assert train_src and val_src, f'could not find train/ and valid/ under {ROOT} (saw {os.listdir(ROOT)})'

# !!! VERIFIED CLASS SCHEME (do NOT trust the dataset's data.yaml — it wrongly says single
# "0: robot"). The actual labels are TWO classes:  0 = NOTE (game piece, small ~0.03 area),
# 1 = ROBOT (large ~0.19 area). Confirmed by box-geometry stats over all 28k labels AND by
# eyeballing annotated frames. For a single-class ROBOT detector we KEEP only class 1, REMAP
# it to 0, and DROP class 0 (note). Images whose robot-filtered label ends up empty are kept
# as background negatives (they teach "a note is not a robot" -> fewer false positives).
SRC_ROBOT_ID = '1'   # robot class index IN THE SOURCE labels

def build_split(src_dir, dst_img, dst_lbl):
    # locate image dir + label dir within this split (flat or images/labels)
    img_dir = os.path.join(src_dir, 'images') if os.path.isdir(os.path.join(src_dir, 'images')) else src_dir
    lbl_dir = os.path.join(src_dir, 'labels') if os.path.isdir(os.path.join(src_dir, 'labels')) else src_dir
    imgs = [f for f in glob.glob(img_dir + '/*') if os.path.splitext(f)[1].lower() in IMG_EXT]
    n_img = n_robot_boxes = n_note_dropped = n_bg = 0
    for ip in imgs:
        stem = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(lbl_dir, stem + '.txt')
        di = os.path.join(dst_img, os.path.basename(ip))
        try:
            os.symlink(ip, di)
        except FileExistsError:
            pass
        except OSError:
            shutil.copy(ip, di)
        n_img += 1
        # ALWAYS write a label file (empty = background negative); filter+remap to robot-only
        out_lines = []
        if os.path.exists(lp):
            for line in open(lp):
                p = line.split()
                if len(p) >= 5:
                    if p[0] == SRC_ROBOT_ID:
                        out_lines.append('0 ' + ' '.join(p[1:5]))   # remap robot 1 -> 0
                        n_robot_boxes += 1
                    else:
                        n_note_dropped += 1
        if not out_lines:
            n_bg += 1
        with open(os.path.join(dst_lbl, stem + '.txt'), 'w') as fh:
            fh.write('\n'.join(out_lines))
    return dict(imgs=n_img, robot_boxes=n_robot_boxes, notes_dropped=n_note_dropped,
                backgrounds=n_bg, img_dir=img_dir, lbl_dir=lbl_dir)

tr = build_split(train_src, f'{DS}/images/train', f'{DS}/labels/train')
va = build_split(val_src,   f'{DS}/images/val',   f'{DS}/labels/val')
print('TRAIN:', tr)
print('VAL  :', va)
assert tr['imgs'] > 1000 and tr['robot_boxes'] > 1000, 'too few train robot boxes — layout/filter failed'
assert va['imgs'] > 100  and va['robot_boxes'] > 100,  'too few val robot boxes — layout/filter failed'

# Sanity: confirm the remapped labels now contain ONLY class 0.
ids = set()
for lp in glob.glob(f'{DS}/labels/train/*.txt')[:1000]:
    for line in open(lp):
        line = line.strip()
        if line:
            ids.add(line.split()[0])
print('class ids in REMAPPED train labels (must be just {"0"}):', sorted(ids))
assert ids <= {'0'}, f'remap failed, saw {ids}'

open(f'{DS}/data.yaml', 'w').write(
    f"path: {DS}\ntrain: images/train\nval: images/val\nnames:\n  0: robot\n")
print('wrote', f'{DS}/data.yaml')
print(open(f'{DS}/data.yaml').read())
""")

code(r"""
# TRAIN YOLO11s @960, single class robot, batch 16 (T4 16GB).
# ~25.7k train imgs @960 on a T4 ≈ 7-9 min/epoch, so 110 epochs would blow past Kaggle's
# 12h GPU session cap. We therefore set a high epoch ceiling (100) AND a hard wall-clock
# budget time=10.5 (hours): Ultralytics stops at whichever comes first and still writes
# best.pt/last.pt — so we ALWAYS get weights out under the cap, however fast the T4 runs.
# patience=30 lets it stop early if mAP plateaus.
from ultralytics import YOLO
m = YOLO('yolo11s.pt')
m.train(
    data=f'{DS}/data.yaml', epochs=100, time=10.5, imgsz=960, batch=16, device=0,
    project=f'{WORK}/runs', name='robot_v2', exist_ok=True,
    mosaic=1.0, close_mosaic=15, hsv_v=0.4, hsv_s=0.4, hsv_h=0.015,
    degrees=5, translate=0.1, scale=0.5, fliplr=0.5, erasing=0.3,
    patience=30, cache=False, workers=4, seed=0, save_period=10,
)
print('training complete')
""")

code(r"""
# Collect deploy artifacts into /kaggle/working root for a clean output pull.
import shutil, glob
run = f'{WORK}/runs/robot_v2'
best = f'{run}/weights/best.pt'
last = f'{run}/weights/last.pt'
shutil.copy(best, f'{WORK}/robot_v2_best.pt')
if os.path.exists(last): shutil.copy(last, f'{WORK}/robot_v2_last.pt')
if os.path.exists(f'{run}/results.csv'): shutil.copy(f'{run}/results.csv', f'{WORK}/robot_v2_results.csv')

# plots (results.png, PR/F1 curves, confusion matrix, train/val batch mosaics)
os.makedirs(f'{WORK}/robot_v2_plots', exist_ok=True)
for p in glob.glob(f'{run}/*.png') + glob.glob(f'{run}/*.jpg'):
    shutil.copy(p, f'{WORK}/robot_v2_plots/{os.path.basename(p)}')

# OpenVINO INT8 @960 — matches the CPU/OpenVINO production engine
try:
    ov = YOLO(best).export(format='openvino', int8=True, data=f'{DS}/data.yaml', imgsz=960, half=False)
    shutil.copytree(ov, f'{WORK}/robot_v2_openvino_model', dirs_exist_ok=True)
    print('exported OpenVINO ->', f'{WORK}/robot_v2_openvino_model')
except Exception as e:
    print('OpenVINO export failed (best.pt still saved):', str(e)[:160])

# final val metrics on the held-out dataset val split
try:
    r = YOLO(best).val(data=f'{DS}/data.yaml', imgsz=960)
    txt = f'mAP50={r.box.map50:.4f}\nmAP50-95={r.box.map:.4f}\nprecision={r.box.mp:.4f}\nrecall={r.box.mr:.4f}\n'
    open(f'{WORK}/robot_v2_val_metrics.txt', 'w').write(txt)
    print(txt)
except Exception as e:
    print('val skipped:', str(e)[:160])

print('DONE -> robot_v2_best.pt, robot_v2_last.pt, robot_v2_results.csv, '
      'robot_v2_openvino_model/, robot_v2_plots/, robot_v2_val_metrics.txt')
""")

nb = {"cells": C,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "robotrain_v2.ipynb"
out.write_text(json.dumps(nb, indent=1))
# sanity: re-parse what we wrote
json.loads(out.read_text())
print(f"wrote {out} ({len(C)} cells) — JSON valid")
