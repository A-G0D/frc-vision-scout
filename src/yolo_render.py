"""Short YOLO ball-tracking showcase clip: green tint + deep-model FUEL circles + score from oracle."""
import json, subprocess, sys
import cv2
import detect_fuel_yolo as yf

def oracle(p="data/sim/overlay_ts.json"):
    try: return {int(r["t"]): r for r in json.load(open(p))}
    except Exception: return {}

def render(vid, out, t0, t1, fps=5, imgsz=1536):
    ts = oracle()
    cap = cv2.VideoCapture(vid); sfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(3)); h = int(cap.get(4))
    f0, f1 = int(t0 * sfps), int(t1 * sfps); step = max(1, int(round(sfps / fps)))
    ff = subprocess.Popen(["ffmpeg","-hide_banner","-loglevel","error","-y","-f","rawvideo",
        "-pix_fmt","bgr24","-s",f"{w}x{h}","-r",str(fps),"-i","-","-c:v","libx264",
        "-pix_fmt","yuv420p","-crf","23",out], stdin=subprocess.PIPE)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0); idx = f0; n = 0
    import numpy as np
    while idx < f1:
        if not cap.grab(): break
        if (idx - f0) % step == 0:
            ok, frame = cap.retrieve()
            if not ok: break
            balls, _ = yf.detect(frame, draw=False, imgsz=imgsz)
            out_img = cv2.addWeighted(frame, 0.82, np.full_like(frame, (0,190,0)), 0.18, 0)
            for (x,y,r) in balls: cv2.circle(out_img,(x,y),max(2,r),(0,255,255),1)
            cv2.rectangle(out_img,(0,0),(w,17),(0,0,0),-1)
            cv2.putText(out_img,f"YOLO FUEL: {len(balls)}",(5,12),cv2.FONT_HERSHEY_SIMPLEX,0.44,(255,255,255),1)
            sc = ts.get(int(idx/sfps)) or {}
            if sc:
                cv2.rectangle(out_img,(0,h-16),(w,h),(0,0,0),-1)
                cv2.putText(out_img,f"SCORE  BLUE {sc.get('blue_score')}  RED {sc.get('red_score')}",(5,h-4),cv2.FONT_HERSHEY_SIMPLEX,0.42,(255,255,255),1)
            ff.stdin.write(out_img.tobytes()); n += 1
            print(f"frame {n}: {len(balls)} balls", flush=True)
        idx += 1
    cap.release(); ff.stdin.close(); ff.wait(); return n

if __name__ == "__main__":
    a = sys.argv
    n = render(a[1], a[2], float(a[3]), float(a[4]),
               int(a[a.index("--fps")+1]) if "--fps" in a else 5,
               int(a[a.index("--imgsz")+1]) if "--imgsz" in a else 1536)
    print(f"rendered {n} frames -> {a[2]}")
