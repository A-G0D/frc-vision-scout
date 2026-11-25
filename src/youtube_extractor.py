"""
youtube_extractor.py — thin, reliable wrapper around yt-dlp + ffmpeg for FRC streams.

Why this exists:
  FRC event broadcasts are multi-hour YouTube streams. We need to (a) pull metadata,
  (b) download low-res sections (or whole streams) cheaply, and (c) sample frames for
  computer-vision analysis. yt-dlp on this machine needs two non-obvious flags to avoid
  getting throttled to a crawl by YouTube's "n-challenge":
        --js-runtimes node            (a JS runtime must exist; node v24 is installed)
        --remote-components ejs:github (fetches the signature/n-challenge solver)
  Without these, downloads are throttled ~50-100x slower and many formats are hidden.

Design choices:
  - Frame sampling for the WHOLE stream is done by downloading a low-res (<=480p) copy
    once, then decimating with ffmpeg to a target fps (e.g. 0.5 fps). For a 9h stream a
    480p AV1/H264 copy is a few hundred MB; sampling at 0.5fps yields ~16k frames, which
    is tractable. We never need audio -> video-only formats.
  - Section download (download_section) is for cheap probing: grab a tiny window anywhere
    in the timeline without pulling the whole file.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# --- flags that make yt-dlp behave on this machine ---------------------------------
YTDLP = "yt-dlp"
JS_FLAGS = ["--js-runtimes", "node", "--remote-components", "ejs:github"]


def _run(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


@dataclass
class VideoMeta:
    id: str
    title: str
    channel: str
    duration_s: int

    @property
    def duration_hms(self) -> str:
        h, rem = divmod(self.duration_s, 3600)
        m, s = divmod(rem, 60)
        return f"{h:d}:{m:02d}:{s:02d}"


def url_for(video: str) -> str:
    """Accept a bare video id or a full URL."""
    if video.startswith("http"):
        return video
    return f"https://www.youtube.com/watch?v={video}"


def get_meta(video: str, timeout: int = 120) -> VideoMeta:
    """Fetch metadata without downloading media."""
    url = url_for(video)
    fmt = "%(id)s\t%(title)s\t%(channel)s\t%(duration)s"
    cp = _run([YTDLP, *JS_FLAGS, "--skip-download", "--print", fmt, url], timeout=timeout)
    line = next((l for l in cp.stdout.splitlines() if "\t" in l), None)
    if not line:
        raise RuntimeError(f"could not get metadata:\n{cp.stderr[-800:]}")
    vid, title, channel, dur = line.split("\t")
    return VideoMeta(id=vid, title=title, channel=channel, duration_s=int(float(dur)))


def download_section(video: str, start: str, end: str, out_dir: str | Path,
                     max_height: int = 480, prefer_h264: bool = True,
                     timeout: int = 900) -> Path:
    """
    Download a single time window [start, end] (HH:MM:SS strings) at <= max_height.
    Returns the path to the written mp4. Cheap way to probe a stream.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h264 = f"bestvideo[height<={max_height}][vcodec^=avc]/" if prefer_h264 else ""
    fmt = f"{h264}bestvideo[height<={max_height}]/best[height<={max_height}]/best"
    out_tmpl = str(out_dir / "sec_%(id)s_%(section_start)s.%(ext)s")
    cmd = [YTDLP, *JS_FLAGS,
           "--download-sections", f"*{start}-{end}",
           "-f", fmt, "--force-keyframes-at-cuts",
           "-o", out_tmpl, url_for(video)]
    cp = _run(cmd, timeout=timeout)
    if cp.returncode != 0:
        raise RuntimeError(f"section download failed:\n{cp.stderr[-1200:]}")
    files = sorted(out_dir.glob(f"sec_*"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise RuntimeError("section download produced no file")
    return files[-1]


def download_lowres(video: str, out_path: str | Path, max_height: int = 480,
                    prefer_h264: bool = True, timeout: int = 7200) -> Path:
    """
    Download the ENTIRE stream as a video-only low-res file for frame scanning.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h264 = f"bestvideo[height<={max_height}][vcodec^=avc]/" if prefer_h264 else ""
    fmt = f"{h264}bestvideo[height<={max_height}]/best[height<={max_height}]/best"
    cmd = [YTDLP, *JS_FLAGS, "-f", fmt, "-o", str(out_path), url_for(video)]
    cp = _run(cmd, timeout=timeout)
    if cp.returncode != 0:
        raise RuntimeError(f"download failed:\n{cp.stderr[-1200:]}")
    return out_path


def sample_frames(video_path: str | Path, out_dir: str | Path, fps: float = 0.5,
                  scale_w: int | None = None) -> int:
    """
    Decimate a local video file into JPEG frames at `fps` frames/sec using ffmpeg.
    Frame filenames encode their index: frame_000001.jpg etc. Returns frame count.
    To map a frame back to a timestamp: t_seconds = (index - 1) / fps.
    """
    video_path, out_dir = Path(video_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vf = f"fps={fps}"
    if scale_w:
        vf += f",scale={scale_w}:-1"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(video_path), "-vf", vf,
           "-q:v", "3", str(out_dir / "frame_%06d.jpg")]
    cp = _run(cmd)
    if cp.returncode != 0:
        raise RuntimeError(f"ffmpeg frame sampling failed:\n{cp.stderr[-800:]}")
    return len(list(out_dir.glob("frame_*.jpg")))


def frame_index_to_timestamp(index: int, fps: float) -> str:
    t = (index - 1) / fps
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


if __name__ == "__main__":
    import sys
    m = get_meta(sys.argv[1] if len(sys.argv) > 1 else "ngvsHrmoW8U")
    print(f"{m.id} | {m.channel} | {m.duration_hms} | {m.title}")
