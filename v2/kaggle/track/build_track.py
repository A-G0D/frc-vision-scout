"""Builds track.ipynb (v2) — GPU render with the 4 fixes:
  1. EFFICIENT robot tracking: bumper-color + size/aspect gate rejects people/refs; an inline
     greedy tracker with constant-velocity coast + Re-ID merge + a CAP of <=3 tracks/alliance
     collapses the fragment explosion (v1 made 32 IDs for 6 robots).
  2. Ball color = SOURCE robot: a ball that enters a robot's (size-scaled) intake radius and then
     departs is locked to that robot's color.
  3. Circle MAX SIZE clamp: oversize ball detections (the giant-circle glitch) are dropped, and
     the drawn radius is clamped.
  4. Intake metrics (radius SCALED TO ROBOT SIZE): balls-in-radius, intake attempts/successes,
     shots (balls that left a robot) — shown live per robot. "Made" needs HUB localization (next
     layer) so it is NOT claimed here.

Run: python kaggle/track/build_track.py
"""
import json
from pathlib import Path

CELLS = []
def md(t): CELLS.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)})
def code(t): CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(keepends=True)})

md(r"""# Robot tracking (gated + capped + predicted) · balls colored by source robot · intake metrics""")

code(r"""
# Pascal-compatible torch FIRST (Kaggle P100/sm_60) — same pin as training
!pip -q install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
!pip -q install ultralytics
import glob, os, cv2, math, numpy as np
from ultralytics import YOLO
clip = (glob.glob('/kaggle/input/**/final1_demo30.mp4', recursive=True) + glob.glob('/kaggle/input/**/*.mp4', recursive=True))[0]
def pick(*names):
    for n in names:
        c = glob.glob(f'/kaggle/input/**/{n}', recursive=True)
        if c: return c[0]
    raise FileNotFoundError(names)
ball_pt  = pick('ball_best.pt',  'runs/ball/weights/best.pt',  'ball*/weights/best.pt')
robot_pt = pick('robot2_best.pt','robot_best.pt','runs/robot2/weights/best.pt','runs/robot/weights/best.pt')
ROBOT_CLS = 0 if 'robot2' in robot_pt else 1   # robot2 = single-class (0=robot); v1 = 2-class (1=robot)
print('clip', clip, '\nball', ball_pt, '\nrobot', robot_pt, 'cls', ROBOT_CLS)
MAX_FRAMES = 720
""")

code(r"""
PAL = {'red':[(0,0,255),(0,90,255),(80,80,255)], 'blue':[(255,80,0),(255,160,0),(255,90,90)]}
WHITE=(220,220,220)

def bumper_alliance(crop):
    if crop is None or crop.size == 0: return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV); h,s,v = hsv[...,0],hsv[...,1],hsv[...,2]
    sat = (s>90)&(v>60)
    red = float((sat&((h<=10)|(h>=170))).mean()); blue = float((sat&(h>=100)&(h<=130)).mean())
    if max(red,blue) < 0.10: return None       # no strong bumper colour -> not a robot (rejects most people)
    return 'red' if red>=blue else 'blue'

def robot_gate(frame, box):
    x1,y1,x2,y2 = box; w,h = x2-x1, y2-y1; H,W = frame.shape[:2]
    area = (w*h)/(W*H); ar = w/max(1.0,h)
    al = bumper_alliance(frame[int(y1+0.55*h):int(y2), int(x1):int(x2)])  # bumpers = lower 45%
    if ROBOT_CLS == 0:                                  # robot2 already EXCLUDES people -> trust it
        if not (0.002 <= area <= 0.22): return None     # loose size sanity only
        return al or ('blue' if (x1+x2)/2 < W/2 else 'red')   # alliance: bumper colour, else field side
    # old 2-class model path: strict gate required to reject people
    if not (0.004 <= area <= 0.13): return None
    if not (0.55 <= ar <= 3.0):     return None
    return al

class RT:
    _n = 0
    def __init__(s, cx,cy,w,h,al,fi):
        s.id=RT._n; RT._n+=1; s.cx=cx;s.cy=cy;s.w=w;s.h=h;s.vx=0;s.vy=0;s.al=al;s.hits=1;s.seen=fi
        s.in_radius=0; s.intake_att=0; s.intakes=0; s.shots=0; s.tt_sum=0.0; s.tt_n=0
    def radius(s): return 1.2*max(s.w,s.h)          # intake radius SCALED to robot size
    def color(s):
        order=[t for t in TRACKS if t.al==s.al]; order.sort(key=lambda t:t.id)
        return PAL[s.al][min(2, order.index(s) if s in order else 0)] if s.al else WHITE

TRACKS=[]
def associate(cands, fi):
    for t in TRACKS: t.predict_done=False
    for (cx,cy,w,h,al) in cands:
        best=None; bd=1e9
        for t in TRACKS:
            if t.al!=al or getattr(t,'used',False): continue
            d=math.hypot(cx-t.cx,cy-t.cy)
            if d < min(bd, 1.6*max(t.w,t.h)+30): bd=d; best=t
        if best is not None:
            best.vx=0.6*(cx-best.cx)+0.4*best.vx; best.vy=0.6*(cy-best.cy)+0.4*best.vy
            best.cx,best.cy,best.w,best.h=cx,cy,w,h; best.hits+=1; best.seen=fi; best.used=True
        else:
            t=RT(cx,cy,w,h,al,fi); t.used=True; TRACKS.append(t)
    # Re-ID merge: same-alliance tracks that overlap -> keep the one with more hits
    for a in TRACKS:
        for b in TRACKS:
            if a is not b and a.al==b.al and a in TRACKS and b in TRACKS:
                if math.hypot(a.cx-b.cx,a.cy-b.cy) < 0.5*max(a.w,a.h):
                    lo = a if a.hits<b.hits else b
                    if lo in TRACKS: TRACKS.remove(lo)

def predict_unseen(fi):
    for t in TRACKS:
        if t.seen!=fi: t.cx+=t.vx; t.cy+=t.vy

def active_tracks():   # CAP: top-3 hits per alliance -> collapses fragments to <=6
    out=[]
    for al in ('red','blue'):
        ts=sorted([t for t in TRACKS if t.al==al], key=lambda t:-t.hits)[:3]
        out+=ts
    return out
""")

code(r"""
cap=cv2.VideoCapture(clip); fps=cap.get(5) or 60.0; W=int(cap.get(3)); H=int(cap.get(4)); cap.release()
MAX_BALL_W = 0.030*W       # a FUEL ball can't be wider than ~3% of frame -> oversize = glitch, drop
MAX_BALL_R = 0.016*W       # clamp drawn radius
vw=cv2.VideoWriter('/kaggle/working/_t.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (W,H))
robot_model=YOLO(robot_pt); ball_model=YOLO(ball_pt)

def dashed(img,p1,p2,c,step=14):
    x1,y1,x2,y2=int(p1[0]),int(p1[1]),int(p2[0]),int(p2[1])
    for x in range(x1,x2,step): cv2.line(img,(x,y1),(min(x+step//2,x2),y1),c,2); cv2.line(img,(x,y2),(min(x+step//2,x2),y2),c,2)
    for y in range(y1,y2,step): cv2.line(img,(x1,y),(x1,min(y+step//2,y2)),c,2); cv2.line(img,(x2,y),(x2,min(y+step//2,y2)),c,2)

ball_src={}; ball_in={}; prev_ball={}   # bid->rid locked ; bid->(rid,enter_fi) pending-in-radius ; bid->(cx,cy)
COAST=45; fi=0
for res in ball_model.track(source=clip, stream=True, persist=True, conf=0.15, imgsz=960, classes=[0], tracker='bytetrack.yaml', verbose=False):
    if fi>=MAX_FRAMES: break
    frame=res.orig_img; f=frame.copy()
    # ---- robots: detect -> gate -> track -> cap ----
    rr=robot_model.predict(frame, conf=0.25, classes=[ROBOT_CLS], imgsz=640, verbose=False)[0]
    cands=[]
    if rr.boxes is not None:
        for b in rr.boxes.xyxy.cpu().numpy():
            al=robot_gate(frame,b)
            if al: x1,y1,x2,y2=b; cands.append(((x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1,al))
    for t in TRACKS: t.used=False
    associate(cands, fi); predict_unseen(fi)
    TRACKS[:] = [t for t in TRACKS if fi-t.seen<=COAST]
    active=active_tracks()
    for t in active: t.in_radius=0
    # ---- balls: clamp/drop -> source attribution (scaled radius) -> intake/shot events ----
    cur={}
    if res.boxes is not None and res.boxes.id is not None:
        for bb,bid in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.cpu().numpy().astype(int)):
            bx1,by1,bx2,by2=bb; bw=bx2-bx1
            if bw>MAX_BALL_W: continue                       # FIX 3: drop oversize glitch
            bcx,bcy=(bx1+bx2)/2,(by1+by2)/2; r=int(min(max(3,bw/2), MAX_BALL_R))   # clamp radius
            cur[bid]=(bcx,bcy)
            inside=[(math.hypot(bcx-t.cx,bcy-t.cy),t) for t in active if math.hypot(bcx-t.cx,bcy-t.cy)<=t.radius()]
            if inside:
                d,t=min(inside); t.in_radius+=1
                if bid not in ball_in and bid not in ball_src:
                    t.intake_att+=1; ball_in[bid]=(t.id, fi)      # entered radius (attempt)
            else:
                # departed a robot it had entered -> that's a shot / it "left" that robot
                if bid in ball_in and bid not in ball_src:
                    rid,_=ball_in.pop(bid)
                    tt=next((x for x in TRACKS if x.id==rid),None)
                    if tt: tt.shots+=1
                    ball_src[bid]=rid                              # FIX 2: lock colour to source robot
            c = (next((t.color() for t in TRACKS if t.id==ball_src[bid]), WHITE) if bid in ball_src else WHITE)
            cv2.circle(f,(int(bcx),int(bcy)),r,c,2)
    # intake SUCCESS: a ball that was inside a radius disappeared this frame -> possessed
    for bid in list(ball_in.keys()):
        if bid not in cur:
            rid,efi=ball_in.pop(bid); tt=next((x for x in TRACKS if x.id==rid),None)
            if tt: tt.intakes+=1; tt.tt_sum+=(fi-efi)/fps; tt.tt_n+=1
    prev_ball=cur
    # ---- draw robots (solid if seen, dashed ghost if predicted) + per-robot HUD ----
    y0=24
    for t in active:
        c=t.color(); x1,y1,x2,y2=t.cx-t.w/2,t.cy-t.h/2,t.cx+t.w/2,t.cy+t.h/2
        if t.seen==fi: cv2.rectangle(f,(int(x1),int(y1)),(int(x2),int(y2)),c,3); lab=f'R{t.id} {t.al}'
        else: dashed(f,(x1,y1),(x2,y2),c); lab=f'R{t.id} predicted'
        cv2.circle(f,(int(t.cx),int(t.cy)),int(t.radius()),c,1)   # intake radius (scaled)
        cv2.putText(f,lab,(int(x1),int(y1)-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,c,2)
        tt = (t.tt_sum/t.tt_n) if t.tt_n else 0
        cv2.putText(f,f'R{t.id}{t.al[0].upper()}: inR {t.in_radius} | intake {t.intakes}/{t.intake_att} ({tt:.1f}s) | shots {t.shots}',
                    (8,y0),cv2.FONT_HERSHEY_SIMPLEX,0.5,c,1); y0+=18
    cv2.putText(f,'balls colored by SOURCE robot | dashed=predicted | circle=intake radius (scaled to robot)',
                (8,H-12),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1)
    vw.write(f); fi+=1
vw.release()
print(f'rendered {fi} frames | robot identities kept {len(TRACKS)} | active<=6 | balls attributed {len(ball_src)}')
os.system('ffmpeg -y -hide_banner -loglevel error -i /kaggle/working/_t.mp4 -c:v libx264 -pix_fmt yuv420p /kaggle/working/tracking_demo.mp4')
os.remove('/kaggle/working/_t.mp4'); print('done -> tracking_demo.mp4')
""")

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name":"Python 3","language":"python","name":"python3"}, "language_info":{"name":"python"}, "accelerator":"GPU"}, "nbformat":4, "nbformat_minor":5}
out = Path(__file__).parent / "track.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(CELLS)} cells)")
