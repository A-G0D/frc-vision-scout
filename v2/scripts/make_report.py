"""Build a self-contained results report (data/scout_out/report.html) with all images
base64-embedded, so it opens locally (file://) with no server and no external assets.

    python scripts/make_report.py
"""
from __future__ import annotations

import base64
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "data", "scout_out")


def img_tag(path, style=""):
    if not os.path.exists(path):
        return f'<div class="missing">[missing: {os.path.basename(path)}]</div>'
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="{style}">'


def rows_from(match_key):
    p = os.path.join(OUT, f"match_{match_key}.json")
    if not os.path.exists(p):
        return ""
    rec = json.load(open(p))
    robots = rec["robots"]
    ov = rec.get("overlay", {})
    head = ("<tr><th>robot</th><th>alliance</th><th>phase</th><th>intakes</th>"
            "<th>mean t→intake (s)</th><th>max held</th><th>possession (s)</th>"
            "<th>jitter</th><th>fuel_scored</th><th>shot acc</th></tr>")
    body = []
    for rid in sorted(robots, key=lambda k: (robots[k]["alliance"] != "red", int(k))):
        r = robots[rid]
        def g(k):
            v = r.get(k)
            if isinstance(v, dict):
                return v.get("value") if v.get("available") else "n/a"
            return v
        ja = r.get("jitter_index", {})
        body.append(
            f"<tr><td>{r['alliance'][0].upper()}{rid}</td><td>{r['alliance']}</td>"
            f"<td>{r.get('phase')}</td><td>{g('intake_attempts')}</td>"
            f"<td>{g('mean_time_to_intake_s')}</td><td>{g('max_held')}</td>"
            f"<td>{g('possession_time_s')}</td>"
            f"<td>{ja.get('value') if isinstance(ja, dict) else ja}</td>"
            f"<td>{g('fuel_scored')}</td><td>{g('shot_accuracy')}</td></tr>")
    teams = f"blue {ov.get('blue_teams')} vs red {ov.get('red_teams')}"
    return f"<p class='cap'>{match_key}: {teams}</p><table>{head}{''.join(body)}</table>"


HTML = f"""<!doctype html><html><head><meta charset="utf-8"><title>FRC Vision Scout v2 — Results</title>
<style>
 body{{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:24px auto;padding:0 18px;color:#1a1a1a}}
 h1{{font-size:26px;margin-bottom:2px}} h2{{margin-top:30px;border-bottom:2px solid #eee;padding-bottom:4px}}
 .sub{{color:#666}} img{{max-width:100%;border:1px solid #ddd;border-radius:6px}}
 table{{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}}
 th,td{{border:1px solid #ddd;padding:5px 8px;text-align:center}} th{{background:#f4f4f4}}
 .cap{{color:#444;font-weight:600;margin:14px 0 2px}} code{{background:#f2f2f2;padding:1px 5px;border-radius:4px}}
 .grid{{display:flex;gap:12px;flex-wrap:wrap}} .grid>div{{flex:1;min-width:340px}}
 .ok{{color:#1ca84f}} .na{{color:#b06000}} pre{{background:#f6f8fa;padding:12px;border-radius:6px;overflow-x:auto;font-size:13px}}
 .missing{{color:#b00;font-style:italic}}
</style></head><body>
<h1>FRC Vision Scout v2 — Results</h1>
<p class="sub">Broadcast (file or live YouTube URL) → 6-robot tracking + scouting metrics → match JSON/CSV + annotated video + dashboard. CPU/OpenVINO, no CUDA. Footage: 2026ctwat (NE District Waterbury). Every metric measured; un-derivable ones marked <span class="na">n/a</span>, never faked.</p>

<h2>Results dashboard (real measured data)</h2>
{img_tag(os.path.join(OUT, "scout_dashboard.png"))}

<h2>Annotated demo — what the operator sees</h2>
<div class="grid">
 <div><div class="cap">qual56 (dense pile @74 balls, 6 robots, intake/hold)</div>{img_tag(os.path.join(OUT, "final2_f570.png"))}</div>
 <div><div class="cap">qual59 (2nd match — generality; scores 19/58)</div>{img_tag(os.path.join(OUT, "qual59_f.png"))}</div>
</div>
<p class="sub">Magenta = detected balls (dense pile reads 74, never 0). Boxes colored by alliance with stable IDs + phase. Right panel = per-robot scouting metrics. Top = real HUD OCR (teams/score/timer).</p>

<h2>Per-robot scouting metrics</h2>
{rows_from("qual56")}
{rows_from("qual59")}

<h2>Run it</h2>
<pre>cd v2
# a match file:
PYTHONPATH=src python scripts/run_scout.py --sync --source data/footage/qual56.mp4 \\
    --start 2000 --frames 600 --out-mp4 data/scout_out/qual56_demo.mp4 \\
    --out-dir data/scout_out --match qual56
# directly from a live/VOD YouTube URL (force 720p60 so team OCR is legible):
PYTHONPATH=src python scripts/run_scout.py --sync \\
    --source "https://www.youtube.com/watch?v=&lt;LIVE_ID&gt;" --out-dir data/scout_out --match live</pre>

<h2>Honest limitations</h2>
<ul>
 <li><span class="ok">Measured:</span> 6-robot stable-ID tracking + coast, dense balls (≤74/frame), phases (traverse/intake/hold), intake attempts + time-to-intake, max held, possession time, jitter, HUD teams/score/timer (qual56 teams match the TBA-verified rosters → OCR cross-validated).</li>
 <li><span class="na">n/a (gated, never faked):</span> per-robot fuel_scored / shot accuracy (need a hub pose or unambiguous shot; alliance fuel/score IS read from the HUD); climb level (only a fool-able monocular heuristic without a real z — disabled); exact field positions (no AprilTag pose on broadcast → image-approximate).</li>
 <li>Intake over-fires in <em>ball-flooded endgames</em> (every robot always near a ball); team OCR robustness varies by match/window.</li>
</ul>
<p class="sub">Tests: 543 passed (PYTHONPATH=src python -m pytest). See RESULTS.md for full detail.</p>
</body></html>"""


def main():
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "report.html"), "w", encoding="utf-8").write(HTML)
    print("wrote", os.path.join(OUT, "report.html"))


if __name__ == "__main__":
    main()
