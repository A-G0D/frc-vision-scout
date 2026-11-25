"""Label review/correction — the step where CLAUDE does the tedious labeling work so
you don't sit in a labeling tool.

Two operating modes (pick in configs/review.yaml):

  mode: claude_read   (zero extra infra)
    The script writes out an ORDERED worklist of frame crops + the teacher's draft boxes
    as side-by-side review images into frames/review/. Claude opens each with the Read
    tool (Claude has native vision), compares draft vs image, and emits corrected YOLO
    label files. Good for building the HELD-OUT TEST SET and auditing the teacher.
    NOTE: keep Claude OUT of the teacher ensemble so this stays an INDEPENDENT check, not
    a circular one. This is dev-time only — no API in the deployed inference path.

  mode: vision_mcp    (batch, higher throughput)
    Same task, but frames are sent through a vision MCP / batch endpoint instead of the
    interactive Read tool. Use when correcting hundreds of frames.

  mode: human         (you, only for final spot-check)
    Emits an Label Studio / CVAT task file for a human pass on the N lowest-confidence
    frames. Optional — the point is to MINIMIZE this.

Active learning: after a training round, `--mine` re-runs the student on unlabeled frames
and surfaces only its DISAGREEMENTS/low-confidence frames for the next review round, so
review effort goes where it moves mAP.
"""
from __future__ import annotations

import argparse

from frcscout.io.config import Config


def build_worklist(cfg: Config) -> list[str]:
    """Rank frames for review: teacher/student disagreement, low conf, dense piles,
    flagged-impossible counts. Returns ordered frame paths + writes review composites."""
    # TODO: emit frames/review/<id>.png (image | draft-overlay) + a manifest.json
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/review.yaml")
    ap.add_argument("--mine", action="store_true", help="active-learning: surface student errors")
    a = ap.parse_args()
    cfg = Config.load(a.config)
    work = build_worklist(cfg)
    print(f"{len(work)} frames queued for review -> frames/review/")
    print("mode:", cfg.get("mode", "claude_read"))
    # In claude_read mode Claude now iterates frames/review/ with Read and writes
    # corrected labels/*.txt; this script just prepares + later validates them.


if __name__ == "__main__":
    main()
