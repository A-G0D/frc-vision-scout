"""Download match footage with yt-dlp — fully automated, no browser (residential IP).

Bakes in the v1 hard-won flags: HLS format 300/301, node JS runtime, force-ipv4, and
--download-sections for fast windowed clipping (HLS pulls only the window's fragments).
DASH + sections hangs ffmpeg on a 12GB remote seek — do not use.

Claude runs this directly via Bash. Edit configs/sources.yaml to add matches.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from frcscout.io.config import Config


def fetch(url: str, start: str, end: str, out: str, fmt: str = "300") -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp", "-f", fmt,
        "--js-runtimes", "node",          # mandatory; without it formats show "not available"
        "--force-ipv4",                   # googlevideo CDN failed IPv6 DNS
        "--download-sections", f"*{start}-{end}",
        "-o", out, url,
    ]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)
    sz = out_path.stat().st_size if out_path.exists() else 0
    if sz < 100_000:                      # 262-byte stub trap from v1
        raise RuntimeError(f"{out} is {sz} bytes — failed-download stub, not a real clip.")
    print(f"OK {out} ({sz/1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sources.yaml")
    a = ap.parse_args()
    cfg = Config.load(a.config)
    for m in cfg.get("matches", []):
        fetch(m["url"], m["start"], m["end"], m["out"], m.get("format", "300"))


if __name__ == "__main__":
    main()
