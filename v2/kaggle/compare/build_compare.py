"""Builds compare.ipynb — a GPU render kernel that overlays BOTH the teacher
(YOLO11x + SAHI) and the trained student on the held-out final1 slice, so we can eyeball
teacher-vs-student-vs-raw AND get the quantitative mean-balls/frame recall comparison.

Run: python kaggle/compare/build_compare.py
"""
import json
from pathlib import Path

CELLS = []
def md(t): CELLS.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)})
def code(t): CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(keepends=True)})

md(r"""
# Teacher vs Student vs Raw — recall comparison on held-out `final1`
Renders ball overlays from the slow teacher (YOLO11x+SAHI) and the fast trained student on a
held-out clip, and prints mean balls/frame for each (the "did the student keep recall" number).
""")

code(r"""
# Pascal-compatible torch FIRST (Kaggle may assign a P100/sm_60; latest torch dropped it) — same pin as training
!pip -q install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
!pip -q install ultralytics sahi
import glob, os, cv2, numpy as np
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
clip = (glob.glob('/kaggle/input/**/final1_demo30.mp4', recursive=True) + glob.glob('/kaggle/input/**/*.mp4', recursive=True))[0]
ball_pt = (glob.glob('/kaggle/input/**/ball_best.pt', recursive=True)
           + glob.glob('/kaggle/input/**/runs/ball/weights/best.pt', recursive=True)
           + glob.glob('/kaggle/input/**/ball*/weights/best.pt', recursive=True))[0]
yolox = (glob.glob('/kaggle/input/**/yolo11x.pt', recursive=True) + ['yolo11x.pt'])[0]
MAX_FRAMES = 720   # ~12s @60fps; the teacher's SAHI pass is the bottleneck
print('clip   :', clip); print('student:', ball_pt); print('teacher:', yolox)
""")

code(r"""
cap = cv2.VideoCapture(clip); fps = cap.get(5) or 60.0; W = int(cap.get(3)); H = int(cap.get(4))
frames = []
while len(frames) < MAX_FRAMES:
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
cap.release()
print(len(frames), 'frames', W, 'x', H, '@', round(fps,1))

student = YOLO(ball_pt)
teacher = AutoDetectionModel.from_pretrained(model_type='ultralytics', model_path=yolox,
                                             confidence_threshold=0.05, device='cuda:0')

def draw(img, pts, color, label):
    out = img.copy()
    for (cx, cy, r) in pts:
        cv2.circle(out, (int(cx), int(cy)), max(3, int(r)), color, 2)
    cv2.putText(out, f'{label}: {len(pts)}', (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    return out

sw = cv2.VideoWriter('/kaggle/working/_s.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
tw = cv2.VideoWriter('/kaggle/working/_t.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
s_counts, t_counts = [], []
for i, f in enumerate(frames):
    r = student.predict(f, imgsz=960, conf=0.15, classes=[0], verbose=False)[0]
    spts = [((b[0]+b[2])/2, (b[1]+b[3])/2, (b[2]-b[0])/2) for b in (r.boxes.xyxy.cpu().numpy() if r.boxes is not None else [])]
    sw.write(draw(f, spts, (0,255,255), 'STUDENT YOLO11s')); s_counts.append(len(spts))

    cv2.imwrite('/kaggle/working/_frame.png', f)
    sp = get_sliced_prediction('/kaggle/working/_frame.png', teacher, slice_height=384, slice_width=384,
                               overlap_height_ratio=0.25, overlap_width_ratio=0.25, verbose=0)
    tpts = [(((o.bbox.minx+o.bbox.maxx)/2), ((o.bbox.miny+o.bbox.maxy)/2), (o.bbox.maxx-o.bbox.minx)/2)
            for o in sp.object_prediction_list if o.category.id == 32]
    tw.write(draw(f, tpts, (0,140,255), 'TEACHER YOLO11x+SAHI')); t_counts.append(len(tpts))
    if i % 60 == 0: print(f'frame {i}: student {len(spts)}  teacher {len(tpts)}')
sw.release(); tw.release()

mt, ms = float(np.mean(t_counts)), float(np.mean(s_counts))
print(f'\n=== MEAN balls/frame ===  teacher {mt:.1f}   student {ms:.1f}   student/teacher recall ratio {ms/max(1,mt):.2f}')
os.system('ffmpeg -y -hide_banner -loglevel error -i /kaggle/working/_s.mp4 -c:v libx264 -pix_fmt yuv420p /kaggle/working/student_overlay.mp4')
os.system('ffmpeg -y -hide_banner -loglevel error -i /kaggle/working/_t.mp4 -c:v libx264 -pix_fmt yuv420p /kaggle/working/teacher_overlay.mp4')
for p in ('/kaggle/working/_s.mp4','/kaggle/working/_t.mp4','/kaggle/working/_frame.png'):
    if os.path.exists(p): os.remove(p)
print('done -> student_overlay.mp4 , teacher_overlay.mp4')
""")

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "compare.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(CELLS)} cells)")
