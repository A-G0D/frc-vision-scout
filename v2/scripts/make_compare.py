"""Controlled before/after: same frames, raw net (what 'sucked') vs the integrated filter.
Draws raw-net boxes (left) and filtered boxes (right) with GT boxes (thin white) for truth.
Writes downscaled JPEGs + a self-contained HTML to the scratchpad for viewing.
"""
import base64
import json
import sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from eval_robots import det_net, det_net_color  # noqa: E402

GT = json.loads((ROOT / "data/robot_gt/qual59_diag_labels.json").read_text())["frames"]
FRAMES = ["f006.png", "f021.png", "f011.png"]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/scout_out/compare.html"
COL = {"red": (0, 0, 255), "blue": (255, 0, 0), None: (0, 200, 255)}


def draw(frame, dets, title, gt):
    img = frame.copy()
    for g in gt:                      # truth = thin white
        cv2.rectangle(img, (g[0], g[1]), (g[2], g[3]), (255, 255, 255), 1)
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), COL.get(d[4]), 3)
    cv2.rectangle(img, (0, 0), (img.shape[1], 46), (0, 0, 0), -1)
    cv2.putText(img, title, (12, 33), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return img


def b64(img, w=900):
    h = int(img.shape[0] * w / img.shape[1])
    small = cv2.resize(img, (w, h))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode()


rows = []
for fn in FRAMES:
    frame = cv2.imread(str(ROOT / "matrix_org/robot_diag" / fn))
    if frame is None:
        continue
    gt = GT.get(fn, [])
    left = draw(frame, det_net(frame), "RAW net (before)", gt)
    right = draw(frame, det_net_color(frame), "filtered (after)", gt)
    rows.append((fn, b64(left), b64(right), len(gt)))
    print(f"{fn}: rendered")

STYLE = """
<style>
  :root{
    --bg:#0d1014; --panel:#151a21; --line:#232b35; --ink:#e9eef5; --mut:#8a97a6;
    --blue:#2d7dff; --red:#ff4d4d; --good:#3fb950;
    --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
    --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
       line-height:1.5;-webkit-font-smoothing:antialiased}
  .wrap{max-width:1080px;margin:0 auto;padding:40px 24px 80px}
  .eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.18em;text-transform:uppercase;
           color:var(--mut)}
  h1{font-size:30px;line-height:1.15;margin:.3em 0 .1em;text-wrap:balance;font-weight:650}
  .sub{color:var(--mut);max-width:62ch}
  .scoreboard{display:flex;gap:14px;flex-wrap:wrap;margin:26px 0 8px}
  .stat{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 18px;min-width:150px}
  .stat .k{font-family:var(--mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--mut)}
  .stat .v{font-family:var(--mono);font-size:22px;font-variant-numeric:tabular-nums;margin-top:4px}
  .stat .v b{color:var(--good)} .stat .v s{color:var(--mut);text-decoration:none;opacity:.6}
  .arrow{color:var(--mut)}
  .legend{font-family:var(--mono);font-size:12.5px;color:var(--mut);display:flex;gap:18px;flex-wrap:wrap;
          border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:12px 0;margin:24px 0}
  .legend i{font-style:normal} .sw{display:inline-block;width:11px;height:11px;border-radius:2px;vertical-align:middle;margin-right:6px}
  .frame{margin:30px 0}
  .frame h3{font-family:var(--mono);font-size:13px;font-weight:600;letter-spacing:.04em;color:var(--mut);margin:0 0 8px}
  .pair{display:grid;grid-template-columns:1fr 1fr;gap:8px}
  .pair figure{margin:0} .pair img{width:100%;display:block;border:1px solid var(--line);border-radius:8px}
  .pair figcaption{font-family:var(--mono);font-size:11px;color:var(--mut);margin-top:6px}
  .caveat{background:var(--panel);border:1px solid var(--line);border-left:3px solid #b3892f;
          border-radius:8px;padding:16px 20px;margin-top:34px;color:var(--mut);font-size:14px}
  .caveat b{color:var(--ink)}
  @media(max-width:720px){.pair{grid-template-columns:1fr}}
</style>
"""
html = [STYLE, '<div class="wrap">',
        '<div class="eyebrow">FRC Vision Scout · qual59 · robot detection</div>',
        "<h1>Same frames, raw detector vs. the integrated filter</h1>",
        '<p class="sub">Measured against 24 hand-labeled ground-truth frames. The raw YOLO '
        '“robot” class boxed balls, people, and HUB structures; the new filter keeps only '
        'bumper-coloured, robot-shaped, deduplicated boxes.</p>',
        '<div class="scoreboard">'
        '<div class="stat"><div class="k">F1</div><div class="v"><s>0.25</s> <span class="arrow">&rarr;</span> <b>0.44</b></div></div>'
        '<div class="stat"><div class="k">Precision</div><div class="v"><s>0.30</s> <span class="arrow">&rarr;</span> <b>0.61</b></div></div>'
        '<div class="stat"><div class="k">Alliance acc</div><div class="v"><b>0.70</b></div></div>'
        '<div class="stat"><div class="k">Pipeline speed</div><div class="v"><s>2.2</s> <span class="arrow">&rarr;</span> <b>11.9</b> fps</div></div>'
        '</div>',
        '<div class="legend">'
        '<i><span class="sw" style="background:#fff"></span>ground truth (hand-labeled)</i>'
        '<i><span class="sw" style="background:var(--blue)"></span>detector &mdash; blue alliance</i>'
        '<i><span class="sw" style="background:var(--red)"></span>detector &mdash; red alliance</i>'
        '</div>']
for fn, l, r, n in rows:
    html.append(f'<div class="frame"><h3>{fn} &mdash; {n} robots labeled visible</h3>'
                '<div class="pair">'
                f'<figure><img src="data:image/jpeg;base64,{l}"><figcaption>BEFORE &mdash; raw net</figcaption></figure>'
                f'<figure><img src="data:image/jpeg;base64,{r}"><figcaption>AFTER &mdash; integrated filter</figcaption></figure>'
                '</div></div>')
html.append('<div class="caveat"><b>Honest limits.</b> qual59 is the worst-case clip &mdash; the '
            'dense ball pile and HUB towers leave only 2&ndash;4 of 6 robots visible per frame, so '
            'recall is capped by occlusion, not the model. Precision 0.61 means ~4 of 10 boxes can '
            'still be wrong; a per-clip static-region mask recovers the 0.90 seen with qual59-specific '
            'tuning. The crowd &ldquo;SHOT&rdquo; markers are a separate flight-detector issue, fix pending.</div>')
html.append('</div>')
OUT.write_text("\n".join(html), encoding="utf-8")
print(f"wrote {OUT}")
