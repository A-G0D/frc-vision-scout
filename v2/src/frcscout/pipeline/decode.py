"""Frame source with hardware decode. Reads fps from the stream (never assume 30).

QSV (Quick Sync) offloads H.264/HEVC decode to the iGPU media engine, keeping the
CPU free for the 60Hz hot path. Falls back to the default OpenCV backend if QSV is
unavailable. Works on a file OR a live stream URL (contest day: `yt-dlp -g <live>`).

Live-stream support:
    Pass a YouTube URL (or any yt-dlp-resolvable URL) as `source`.  FrameSource will
    detect the URL scheme, resolve it to a direct HLS/DASH stream URL via
    ``yt-dlp -g`` (force-IPv4, prefers 720p60/1080p60 HLS formats 300/301), then open
    the resolved URL with the FFMPEG backend.

    A reconnect loop inside :meth:`frames` re-opens the stream whenever the capture
    stalls (EOF on a live stream is a stall, not the end of a match).

    ``is_live`` is set to True whenever the source looks like a URL rather than a
    local path, so callers can skip ``max_frames`` / progress-bar logic.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Iterator, Optional

import cv2

# yt-dlp format preference: 720p60 HLS (300) or 1080p60 HLS (301) — direct HTTP
# streams that FFmpeg can open without further demux tricks.  Falls back to the best
# available HLS if neither is present.
_YT_FORMAT = "300/301/best[ext=mp4]/best"
_RECONNECT_DELAY_S = 2.0          # seconds to wait before re-opening a live source
_MAX_RECONNECT = 20               # give up after this many consecutive re-open attempts


def _looks_like_url(source: str) -> bool:
    return source.startswith(("http://", "https://", "rtsp://", "rtmp://"))


def resolve_yt_url(url: str) -> str:
    """Resolve a YouTube (or other yt-dlp-supported) URL to a direct stream URL.

    Uses ``yt-dlp -g`` (print URL, no download) with:
    - ``--force-ipv4``     — avoids IPv6 flakiness on contest-day networks
    - ``-f <fmt>``         — prefers 720p60/1080p60 HLS (formats 300/301)

    Returns the first non-empty line of stdout, which is the direct stream URL.
    Raises :class:`RuntimeError` on failure.
    """
    cmd = [
        "yt-dlp",
        "--force-ipv4",
        "-f", _YT_FORMAT,
        "-g",           # print the direct URL instead of downloading
        url,
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp failed (rc={result.returncode}): {result.stderr.strip()}"
        )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("yt-dlp returned no URL")
    return lines[0]


class FrameSource:
    """Unified frame source for local files and live/VOD YouTube streams.

    Parameters
    ----------
    source:
        Local file path, a direct stream URL, or a YouTube URL.
        YouTube URLs are auto-resolved via :func:`resolve_yt_url`.
    backend:
        ``"qsv"`` (default) — opens with ``cv2.CAP_FFMPEG`` + hardware-decode hint.
        Any other string — ``cv2.VideoCapture(source)`` default backend.
    """

    def __init__(self, source: str, backend: str = "qsv"):
        self.source = source
        self.backend = backend
        # Resolve YouTube/stream URLs to a direct playback URL.
        self._resolved_url: Optional[str] = None
        self.is_live: bool = False

        if _looks_like_url(source):
            self.is_live = True
            # Try to resolve via yt-dlp; if yt-dlp is unavailable, try the URL directly.
            try:
                self._resolved_url = resolve_yt_url(source)
            except (FileNotFoundError, RuntimeError):
                # yt-dlp not installed or resolution failed — attempt raw URL.
                self._resolved_url = source
        else:
            self._resolved_url = source

        self.cap = self._open(self._resolved_url, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")

        # Trust frame READS, not CAP_PROP_FRAME_COUNT (unreliable on some encodings — v1 gotcha).
        self.fps: float = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.w: int = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h: int = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.w == 0 or self.h == 0:
            # Probe by reading one frame (some streams report 0 until first decode).
            ok, fr = self.cap.read()
            if not ok:
                raise RuntimeError(f"Source opened but yields no frames: {source}")
            self.h, self.w = fr.shape[:2]
            self._primed = fr
        else:
            self._primed = None

    @staticmethod
    def _open(source: str, backend: str) -> "cv2.VideoCapture":
        if backend == "qsv":
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            # Hint FFmpeg toward hardware decode; harmless if unsupported.
            try:
                cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
            except Exception:
                pass
            if cap.isOpened():
                return cap
        return cv2.VideoCapture(source)

    def _reopen(self) -> bool:
        """Re-open the resolved URL.  Returns True on success."""
        self.cap.release()
        time.sleep(_RECONNECT_DELAY_S)
        url = self._resolved_url or self.source
        self.cap = self._open(url, self.backend)
        return self.cap.isOpened()

    def frames(self) -> Iterator:
        """Yield decoded BGR frames indefinitely (live) or until EOF (file).

        On a live stream, EOF is treated as a stall: the source is re-opened up to
        ``_MAX_RECONNECT`` consecutive times before giving up.
        """
        if self._primed is not None:
            yield self._primed
            self._primed = None

        consecutive_fails = 0
        while True:
            ok, fr = self.cap.read()
            if ok:
                consecutive_fails = 0
                yield fr
            else:
                # EOF on a file → done; EOF on a live stream → reconnect.
                if not self.is_live:
                    break
                consecutive_fails += 1
                if consecutive_fails > _MAX_RECONNECT:
                    break
                if not self._reopen():
                    # Back-off and retry the reopen itself.
                    time.sleep(_RECONNECT_DELAY_S)

    def release(self) -> None:
        self.cap.release()
