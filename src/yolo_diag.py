"""Diagnose ball recall: run YOLO UNGATED, mark on-carpet (yellow) vs off-carpet (red=airborne/edge)."""
import sys, cv2, numpy as np
from ultralytics import YOLO
import detect_fuel_yolo as yf

m = YOLO("yolo11x.pt")
poly_norm = yf.FIELD_POLY_NORM
cap = cv2.VideoCapture("data/sim/match_full.mp4"); fps = cap.get(cv2.CAP_PROP_FPS)
tiles = []
for t in (55, 95):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps)); ok, f = cap.read()
    if not ok: continue
    h, w = f.shape[:2]
    poly = np.array([(int(x*w), int(y*h)) for x, y in poly_norm], np.int32)
    r = m.predict(f, imgsz=1920, conf=0.04, classes=[32], verbose=False)[0]
    on = off = 0
    ann = f.copy(); cv2.polylines(ann, [poly], True, (255,255,255), 1)
    for b in r.boxes.xyxy.cpu().numpy():
        x1,y1,x2,y2 = b; cx,cy = int((x1+x2)/2), int((y1+y2)/2); rad = max(3,int((x2-x1)/2))
        inside = cv2.pointPolygonTest(poly,(float(cx),float(cy)),False) >= 0
        if inside: cv2.circle(ann,(cx,cy),rad,(0,255,255),1); on += 1
        else: cv2.circle(ann,(cx,cy),rad,(0,0,255),2); off += 1   # RED = currently dropped
    cv2.putText(ann, f"t={t}s  on-carpet:{on}  OFF(dropped):{off}", (6,16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
    tiles.append(ann); print(f"t={t}s: on-carpet={on}  off-carpet(dropped)={off}  total={on+off}")
cap.release()
if len(tiles) == 2:
    cv2.imwrite("data/sim/_yolo_diag.png", np.vstack([tiles[0], tiles[1]]))
    print("wrote data/sim/_yolo_diag.png")
