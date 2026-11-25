"""Builds robotrain.ipynb — fix the robot detector via PERSON-NEGATIVE auto-labeling, then
retrain a single-class robot model.

Root cause of the bad robot model: motion+bumper bootstrap fired on PEOPLE (refs/spectators in
alliance colors) and on static red/blue STRUCTURES. Fix:
  candidate robot = MOG2 motion blob, large, red/blue bumper in lower half, in the field band,
                    AND NOT overlapping a COCO 'person' detection.
Motion rejects static HUB/ramp structures; person-negative rejects humans; field band rejects
stands/scorer table. Cleaner labels, at scale, no manual boxing.

Outputs: robot_openvino_model + robot_best.pt + 8 annotated QA sample frames (to eyeball that
it now boxes robots, not people).

Run: python kaggle/robotrain/build_robotrain.py
"""
import json
from pathlib import Path

CELLS = []
def md(t): CELLS.append({"cell_type":"markdown","metadata":{},"source":t.strip("\n").splitlines(keepends=True)})
def code(t): CELLS.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":t.strip("\n").splitlines(keepends=True)})

md(r"""# Robot detector retrain — person-negative auto-label, single-class""")

code(r"""
# Pascal-compatible torch FIRST (Kaggle P100/sm_60)
!pip -q install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
!pip -q install ultralytics
import torch, glob, os, cv2, numpy as np
from ultralytics import YOLO
assert torch.cuda.is_available()
clips = sorted(glob.glob('/kaggle/input/**/*.mp4', recursive=True))
assert clips, 'attach frc-footage dataset'
WORK='/kaggle/working'; DS=f'{WORK}/rds'
for s in ['images/train','images/val','labels/train','labels/val','qa']:
    os.makedirs(f'{DS}/{s}', exist_ok=True)
print(len(clips), 'clips')
""")

code(r"""
# AUTO-LABEL: person-negative + motion + bumper + field band
person = YOLO('yolo11x.pt')           # COCO; class 0 = person (negative filter)
PLAY = (0.18, 0.63)                   # field band (drop stands top / scorer foreground)
def iou(a,b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
    iw=max(0,x2-x1);ih=max(0,y2-y1);inter=iw*ih
    if inter==0: return 0.0
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)
def bumper(crop):
    if crop.size==0: return False
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV);h,s,v=hsv[...,0],hsv[...,1],hsv[...,2];sat=(s>90)&(v>60)
    return max(float((sat&((h<=10)|(h>=170))).mean()), float((sat&(h>=100)&(h<=130)).mean())) > 0.10

SAMPLE_FPS=1.0; gid=0; meta=[]; qa=0
for ci,clip in enumerate(clips):
    cap=cv2.VideoCapture(clip); fps=cap.get(5) or 60; W=int(cap.get(3)); H=int(cap.get(4))
    bg=cv2.createBackgroundSubtractorMOG2(history=250, varThreshold=30, detectShadows=False)
    step=max(1,round(fps/SAMPLE_FPS)); i=0
    while True:
        ok,f=cap.read()
        if not ok: break
        fgmask=bg.apply(f)
        if i%step==0 and i>step:          # skip first frame (bg not warm)
            persons=[b for b in person.predict(f,classes=[0],conf=0.3,imgsz=960,verbose=False)[0].boxes.xyxy.cpu().numpy()]
            fg=cv2.morphologyEx(fgmask,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
            fg=cv2.morphologyEx(fg,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
            cnts,_=cv2.findContours(fg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            lines=[]; boxes=[]
            for c in cnts:
                x,y,w,h=cv2.boundingRect(c); area=(w*h)/(W*H); ar=w/max(1,h); cy=y+h/2
                if not (0.004<=area<=0.13 and 0.55<=ar<=3.0 and PLAY[0]*H<=cy<=PLAY[1]*H): continue
                if not bumper(f[y+int(0.55*h):y+h, x:x+w]): continue
                bx=[x,y,x+w,y+h]
                if any(iou(bx,p)>0.25 for p in persons): continue   # PERSON-NEGATIVE
                lines.append(f'0 {(x+w/2)/W:.6f} {(y+h/2)/H:.6f} {w/W:.6f} {h/H:.6f}'); boxes.append(bx)
            sub='val' if (ci==len(clips)-1) else 'train'   # last clip = val (match-level)
            cv2.imwrite(f'{DS}/images/{sub}/{gid:06d}.png',f)
            open(f'{DS}/labels/{sub}/{gid:06d}.txt','w').write('\n'.join(lines))
            meta.append((ci,gid,len(lines)))
            if qa<8 and boxes:
                ann=f.copy()
                for bx in boxes: cv2.rectangle(ann,(bx[0],bx[1]),(bx[2],bx[3]),(0,255,0),3)
                cv2.imwrite(f'{DS}/qa/qa_{qa:02d}.png',ann); qa+=1
            gid+=1
        i+=1
    cap.release()
open(f'{DS}/data.yaml','w').write(f"path: {DS}\ntrain: images/train\nval: images/val\nnames:\n  0: robot\n")
import numpy as np
tot=sum(m[2] for m in meta)
print(f'auto-labeled {gid} frames, {tot} robot boxes (mean {tot/max(1,gid):.1f}/frame), {qa} QA samples')
""")

code(r"""
# TRAIN single-class robot (YOLO11n @640) — robots are large/few so this is fast
m=YOLO('yolo11n.pt')
m.train(data=f'{DS}/data.yaml', epochs=60, imgsz=640, batch=16, device=0,
        project=f'{WORK}/runs', name='robot2', exist_ok=True,
        mosaic=1.0, close_mosaic=10, hsv_v=0.4, scale=0.4, fliplr=0.5, patience=25, cache=False)
best=f'{WORK}/runs/robot2/weights/best.pt'
import shutil
m2=YOLO(best); ov=m2.export(format='openvino', int8=True, data=f'{DS}/data.yaml', imgsz=640, half=False)
shutil.copytree(ov, f'{WORK}/robot2_openvino_model', dirs_exist_ok=True)
shutil.copy(best, f'{WORK}/robot2_best.pt')
try: print('robot2 val mAP50-95:', m2.val(data=f'{DS}/data.yaml').box.map)
except Exception as e: print('val skipped', str(e)[:80])
# keep only the small deploy artifacts + QA in working root (clean pull)
print('done -> robot2_openvino_model, robot2_best.pt, rds/qa/*.png')
""")

nb={"cells":CELLS,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"},"accelerator":"GPU"},"nbformat":4,"nbformat_minor":5}
out=Path(__file__).parent/"robotrain.ipynb"; out.write_text(json.dumps(nb,indent=1)); print(f"wrote {out} ({len(CELLS)} cells)")
