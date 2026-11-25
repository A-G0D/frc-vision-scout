"""
pipeline.py — end-to-end: video file -> match segments -> Claude overlay reads -> schedule.

Ties the modules together:
  1. scan_timeline.scan()          : whole video -> match SEGMENTS (overlay color signature)
  2. ocr_overlay.read_segment()    : per segment -> teams / scores / label (Claude vision)
  3. match_list.get_matches()      : ground-truth schedule (TBA key or Statbotics)
  4. reconcile                     : align detected matches to the schedule, flag mismatches

Each stage is independently runnable; this just sequences them and writes a result JSON.
The Claude stage needs ANTHROPIC_API_KEY; stages 1 and 3 do not.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import scan_timeline
import ocr_overlay


def run(video_path: str, event_key: str | None = None,
        ocr_samples: int = 3, limit_segments: int | None = None) -> dict:
    video_path = str(video_path)

    # 1. Segment the stream (no OCR, no key needed)
    print(f"[1/3] scanning {video_path} for match segments…")
    segments = scan_timeline.scan(video_path)
    print(f"      found {len(segments)} segments")
    if limit_segments:
        segments = segments[:limit_segments]

    # 2. Read the overlay for each segment via Claude vision (needs key)
    have_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if have_key:
        print(f"[2/3] reading overlays with Claude vision ({len(segments)} segments)…")
        for s in segments:
            r = ocr_overlay.read_segment(video_path, s.t_start, s.t_end, n_samples=ocr_samples)
            s.match_label = r.match_label
            s.blue = r.blue_teams
            s.red = r.red_teams
            print(f"      seg {s.index:3d} {s.hms(s.t_start)}  {r.match_label or '?'}  "
                  f"blue={r.blue_teams} red={r.red_teams} "
                  f"score={r.blue_score}-{r.red_score}{' [playoff]' if r.is_playoff else ''}")
    else:
        print("[2/3] SKIPPED — ANTHROPIC_API_KEY not set (overlay reading needs it)")

    # 3. Align to the ground-truth schedule if we can fetch it
    schedule = []
    if event_key:
        try:
            import match_list
            print(f"[3/3] fetching schedule for {event_key}…")
            schedule = match_list.get_matches(event_key)
            scan_timeline.align_to_schedule(segments, schedule)
            print(f"      aligned {min(len(segments), len(schedule))} segments to schedule")
        except Exception as e:
            print(f"[3/3] schedule fetch failed: {e}")
    else:
        print("[3/3] SKIPPED — no event_key provided")

    out = {
        "video": video_path,
        "event_key": event_key,
        "segment_count": len(segments),
        "ocr_ran": have_key,
        "schedule_matches": len(schedule),
        "segments": [asdict(s) for s in segments],
    }
    out_path = Path(video_path).resolve().parent / f"{Path(video_path).stem}_pipeline.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"wrote {out_path}")
    return out


if __name__ == "__main__":
    import sys
    vid = sys.argv[1] if len(sys.argv) > 1 else "data/ngvsHrmoW8U_360p.mp4"
    ek = sys.argv[2] if len(sys.argv) > 2 else None
    lim = int(sys.argv[3]) if len(sys.argv) > 3 else None
    run(vid, event_key=ek, limit_segments=lim)
