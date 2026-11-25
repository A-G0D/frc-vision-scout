# Feasibility: extracting match data from FRC livestreams — VERDICT: ✅ feasible

Tested against a real, full-length official stream and documented what works, what's hard,
and the exact knobs that mattered. Everything below was run end-to-end on this machine.

## Test subject
- **Video:** `ngvsHrmoW8U` — *FIRST Robotics Competition REBUILT 2026 — NE FIRST Waterbury District Event Day 2* (official NE FIRST channel), **9h 31m**.
- A second candidate confirmed: `i1cedzJVzzg` — *NE FIRST District Championship Burns Division Day 1*, 8h 15m.

## What we proved works
| Step | Result |
|------|--------|
| Pull metadata (title/channel/duration) | ✅ yt-dlp |
| Download a full low-res copy | ✅ **354 MB @ 144p in 44s** (format 160, H.264 DASH) |
| Sample frames locally with ffmpeg | ✅ instant, arbitrary timestamps |
| **Read full-screen graphics (team numbers)** | ✅ **legible even at 144p** (Alliance Selection screen team grid read cleanly) |
| Detect "match vs filler" with no OCR | ✅ color-signature on the overlay bar, validated on real frames |
| Segment the whole 9.5h stream into matches | ✅ **66 overlay segments (34 full matches + 32 recaps/replays)** |

## The key findings that shape the design
1. **The broadcast overlay is a structured scoreboard we can read.** During every match a persistent top bar shows:
   - `Qualification N of M` (top center) → **the match identifier, for free**
   - **Blue teams (left) + Red teams (right)** → the competitor list
   - a center **timer** → match clock/phase
   - `X / 100` boxes per side → the **live FUEL count vs the ENERGIZED RP threshold (100)** — i.e. the live score is literally rendered on screen.
   This means we get match ID + teams + live score + clock from OCR of one bar, *before* doing any physical robot/ball detection.

2. **Match/filler segmentation needs zero OCR.** The bar's left third is blue-dominant and right third red-dominant during a match; absent otherwise. On real frames: in-match ≈ 0.15–0.20 color-mass on both sides, filler ≈ 0.00. Clean separation → reliable timeline cutting.

3. **Two detection layers, not one** (see FRAMEWORK.md):
   - **Layer A "read the broadcast"** — OCR the overlay. Cheap, high-value, gets schedule + scores. *This alone satisfies the original ask* ("extract competitors per game").
   - **Layer B "watch the field"** — detect FUEL/robots/climbs directly. Needed for scouting granularity the overlay never shows (per-robot cycles, defense, sourcing).

## The gotchas (and the fixes)
| Gotcha | Fix |
|--------|-----|
| yt-dlp throttled ~50× + hid formats | `--js-runtimes node --remote-components ejs:github` (node v24 present; solves the n-challenge) |
| Former-live VODs only expose **HLS (m3u8)**; `--download-sections` over HLS is unusably slow | **Download one full DASH (`https`) low-res copy, sample locally.** Don't section-download, don't remote-seek. |
| ffmpeg remote-seeking into a 9.5h DASH file stalls | Same fix — pull the whole low-res file once (it's small at 144–240p). |
| Team numbers in the *in-match* bar are tiny at 144p | Use **240p/360p** for production OCR; full-screen graphics are fine even at 144p. |
| OCR not installed (no tesseract/easyocr) | Layer-A OCR is the one remaining install; segmentation + schedule-alignment already recover teams without it. |
| Ground-truth match list (TBA/Statbotics) | TBA needs a free auth key; Statbotics API was returning 500s during testing. `match_list.py` supports both; set `TBA_AUTH_KEY`. |

## Cost / scale estimate (per event-day stream)
- Download: ~350 MB (144p) – ~800 MB (240p), under a minute on this link.
- Full-timeline overlay scan at 0.5 fps: completed within a couple of minutes for 9.5h.
- So a whole event (multiple day-streams) is a handful of GB and minutes of compute — very tractable.

## Bottom line
Extracting "every match and its competitors" from these streams is **clearly feasible** and most of it doesn't even need ML — the broadcast hands us the structured data. The harder, more valuable layer (physical scoring/scouting from the field view) is a genuine CV problem and is scoped in FRAMEWORK.md.
