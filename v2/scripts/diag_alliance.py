"""Quantify the bumper_alliance() red/blue skew on REAL robot boxes.

For every class-1 robot det (conf>=0.10) across the sampled frames, run
RobotNet.bumper_alliance and tally red/blue/None. The export shows all-red; this
measures whether the vote itself is red-biased (the `red >= blue` tie rule) or whether
real blue robots are being mislabeled red.
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cv2, numpy as np
from ultralytics import YOLO
from frcscout.detect.robot_net import RobotNet

ROOT = os.path.join(os.path.dirname(__file__), "..")
ball = YOLO(os.path.join(ROOT, "weights", "ball_openvino_model"), task="detect")
VIDEOS = ["qual50", "qual56", "qual59", "final1", "match_1080", "ubr1m1"]
FRACS = [0.15, 0.30, 0.45, 0.60, 0.75]

tot = {"red":0,"blue":0,"None":0}
# Also recompute the raw red vs blue pixel masses to see how often it's a near-tie / weak.
import cv2 as _cv2
def raw_counts(frame, xyxy):
    x1,y1,x2,y2 = (int(v) for v in xyxy)
    crop = frame[(y1+y2)//2:y2, x1:x2]
    if crop.size==0: return 0,0,0
    hsv=_cv2.cvtColor(crop,_cv2.COLOR_BGR2HSV)
    h,s,v=hsv[...,0],hsv[...,1],hsv[...,2]
    sat=(s>90)&(v>60)
    red=int((sat&((h<=10)|(h>=170))).sum())
    blue=int((sat&(h>=100)&(h<=130)).sum())
    return red, blue, crop.shape[0]*crop.shape[1]

near_tie=0; weak=0; n=0
for v in VIDEOS:
    cap=cv2.VideoCapture(os.path.join(ROOT,"data","footage",f"{v}.mp4"))
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frac in FRACS:
        f=int(total*frac); cap.set(cv2.CAP_PROP_POS_FRAMES,f); ok,fr=cap.read()
        if not ok: continue
        r=ball.predict(fr,imgsz=960,conf=0.10,device="cpu",verbose=False)[0]
        if r.boxes is None: continue
        xyxy=r.boxes.xyxy.cpu().numpy(); cl=r.boxes.cls.cpu().numpy().astype(int)
        for b,c in zip(xyxy,cl):
            if c!=1: continue
            n+=1
            al=RobotNet.bumper_alliance(fr,tuple(map(float,b)))
            tot[str(al)]+=1
            red,blue,area=raw_counts(fr,tuple(map(float,b)))
            thr=0.02*area
            if max(red,blue)<thr: weak+=1
            elif abs(red-blue)<0.25*max(red,blue,1): near_tie+=1
    cap.release()
print(json.dumps({"n_robot_dets":n,"alliance_vote":tot,
    "weak_signal(below 2% thr->None)":weak,
    "near_tie(<25% margin)":near_tie}), flush=True)
print(f"red fraction of decided = {tot['red']/max(1,tot['red']+tot['blue']):.0%}", flush=True)
