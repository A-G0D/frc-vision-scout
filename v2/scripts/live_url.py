#!/usr/bin/env python3
"""Resolve a YouTube (or any yt-dlp-supported) URL to a direct stream URL.

Contest-day usage — point the engine at a live FRC stream:

    # 1. Resolve the URL once and inspect it:
    python scripts/live_url.py "https://www.youtube.com/watch?v=<LIVE_MATCH_ID>"

    # 2. Feed the resolved URL directly to the engine (avoids the per-process yt-dlp call):
    RESOLVED=$(python scripts/live_url.py "https://www.youtube.com/watch?v=<ID>")
    python -m frcscout.pipeline.engine --config configs/deploy.yaml --source "$RESOLVED"

    # 3. Or let the engine resolve it automatically (same result, one command):
    python -m frcscout.pipeline.engine --config configs/deploy.yaml \
        --source "https://www.youtube.com/watch?v=<ID>" \
        --save out.mp4

Format preference: 720p60 HLS (format 300), then 1080p60 HLS (301), then best available.
Force-IPv4 is always on to avoid contest-venue IPv6 edge-cases.
"""
from __future__ import annotations

import argparse
import sys

# Ensure the package root is on the path when invoked as a script.
from pathlib import Path
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from frcscout.pipeline.decode import resolve_yt_url  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Resolve a YouTube/stream URL to a direct playback URL via yt-dlp."
    )
    ap.add_argument("url", help="YouTube URL or any yt-dlp-supported stream URL")
    ap.add_argument(
        "-f", "--format",
        default=None,
        help="yt-dlp format string (default: 300/301/best[ext=mp4]/best)",
    )
    a = ap.parse_args()

    try:
        if a.format:
            # Allow override without changing the module constant.
            import subprocess
            cmd = ["yt-dlp", "--force-ipv4", "-f", a.format, "-g", a.url]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                print(f"yt-dlp error: {result.stderr.strip()}", file=sys.stderr)
                sys.exit(1)
            lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
            if not lines:
                print("yt-dlp returned no URL", file=sys.stderr)
                sys.exit(1)
            print(lines[0])
        else:
            print(resolve_yt_url(a.url))
    except FileNotFoundError:
        print(
            "yt-dlp is not installed.  Install it with:  pip install yt-dlp",
            file=sys.stderr,
        )
        sys.exit(1)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
