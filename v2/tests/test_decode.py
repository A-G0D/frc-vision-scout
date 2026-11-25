"""Tests for decode.py — FrameSource and live-URL resolver.

Covers:
  - Opening a synthetic mp4 file (generated with cv2.VideoWriter)
  - fps > 0 and frames actually read
  - Live-URL resolver: subprocess args and URL parsing (mocked — no network)
  - is_live flag set for URL sources, not set for file sources
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Guard: skip the whole module if cv2 is not available.
try:
    import cv2
    import numpy as np
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# Ensure the package is importable when running from the repo root.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #

def _make_synthetic_mp4(path: str, n_frames: int = 10, fps: float = 30.0) -> None:
    """Write a tiny synthetic mp4 using cv2.VideoWriter (solid-colour frames)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (320, 240))
    for i in range(n_frames):
        color = ((i * 25) % 256, (i * 13) % 256, (i * 7) % 256)
        frame = np.full((240, 320, 3), color, dtype=np.uint8)
        writer.write(frame)
    writer.release()


# --------------------------------------------------------------------- #
# Tests: FrameSource with a local file
# --------------------------------------------------------------------- #

@unittest.skipUnless(_HAS_CV2, "cv2 not available")
class TestFrameSourceFile(unittest.TestCase):

    def setUp(self):
        from frcscout.pipeline.decode import FrameSource
        self.FrameSource = FrameSource
        self._tmpdir = tempfile.mkdtemp()
        self._mp4 = os.path.join(self._tmpdir, "synth.mp4")
        _make_synthetic_mp4(self._mp4, n_frames=10, fps=30.0)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_opens_and_fps_positive(self):
        src = self.FrameSource(self._mp4, backend="cpu")
        try:
            self.assertGreater(src.fps, 0.0, "fps should be > 0")
        finally:
            src.release()

    def test_reads_frames(self):
        src = self.FrameSource(self._mp4, backend="cpu")
        frames = []
        try:
            for f in src.frames():
                frames.append(f)
        finally:
            src.release()
        self.assertGreater(len(frames), 0, "should yield at least one frame")

    def test_frames_have_correct_shape(self):
        src = self.FrameSource(self._mp4, backend="cpu")
        first = None
        try:
            for f in src.frames():
                first = f
                break
        finally:
            src.release()
        self.assertIsNotNone(first)
        self.assertEqual(len(first.shape), 3, "frame should be H x W x C")
        self.assertEqual(first.shape[2], 3, "frame should have 3 channels (BGR)")

    def test_width_height_set(self):
        src = self.FrameSource(self._mp4, backend="cpu")
        try:
            self.assertGreater(src.w, 0)
            self.assertGreater(src.h, 0)
        finally:
            src.release()

    def test_is_live_false_for_file(self):
        src = self.FrameSource(self._mp4, backend="cpu")
        try:
            self.assertFalse(src.is_live, "is_live should be False for a local file")
        finally:
            src.release()


# --------------------------------------------------------------------- #
# Tests: live-URL resolver (subprocess mocked — no network)
# --------------------------------------------------------------------- #

@unittest.skipUnless(_HAS_CV2, "cv2 not available")
class TestResolveYtUrl(unittest.TestCase):

    def setUp(self):
        from frcscout.pipeline.decode import resolve_yt_url, _YT_FORMAT
        self.resolve_yt_url = resolve_yt_url
        self._expected_format = _YT_FORMAT

    def _make_completed_process(self, stdout: str, returncode: int = 0):
        mock = MagicMock(spec=subprocess.CompletedProcess)
        mock.returncode = returncode
        mock.stdout = stdout
        mock.stderr = ""
        return mock

    @patch("frcscout.pipeline.decode.subprocess.run")
    def test_calls_yt_dlp_with_correct_args(self, mock_run):
        """resolve_yt_url must call yt-dlp with --force-ipv4 and -g."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        fake_stream = "https://manifest.googlevideo.com/api/manifest/hls_playlist/faketoken"
        mock_run.return_value = self._make_completed_process(fake_stream + "\n")

        result = self.resolve_yt_url(url)

        self.assertTrue(mock_run.called, "subprocess.run should have been called")
        call_args = mock_run.call_args
        cmd = call_args[0][0]   # first positional arg is the command list

        self.assertEqual(cmd[0], "yt-dlp", "first arg should be yt-dlp")
        self.assertIn("--force-ipv4", cmd, "--force-ipv4 must be present")
        self.assertIn("-g", cmd, "-g (print URL) flag must be present")
        self.assertIn(url, cmd, "the target URL must be in the command")

    @patch("frcscout.pipeline.decode.subprocess.run")
    def test_uses_preferred_format(self, mock_run):
        """The format string passed to yt-dlp must include format 300 or 301."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        mock_run.return_value = self._make_completed_process(
            "https://stream.example.com/live.m3u8\n"
        )
        self.resolve_yt_url(url)
        cmd = mock_run.call_args[0][0]
        # -f flag must be present and its value must mention 300 and 301
        self.assertIn("-f", cmd)
        fmt_idx = cmd.index("-f") + 1
        fmt_str = cmd[fmt_idx]
        self.assertIn("300", fmt_str, "format 300 (720p60 HLS) must be in the format string")
        self.assertIn("301", fmt_str, "format 301 (1080p60 HLS) must be in the format string")

    @patch("frcscout.pipeline.decode.subprocess.run")
    def test_returns_first_url_line(self, mock_run):
        """Result should be the first non-empty line of yt-dlp stdout."""
        expected = "https://stream.example.com/live.m3u8"
        mock_run.return_value = self._make_completed_process(
            f"\n{expected}\nhttps://other.example.com/other.m3u8\n"
        )
        result = self.resolve_yt_url("https://www.youtube.com/watch?v=fake")
        self.assertEqual(result, expected)

    @patch("frcscout.pipeline.decode.subprocess.run")
    def test_raises_on_nonzero_returncode(self, mock_run):
        """Should raise RuntimeError when yt-dlp exits non-zero."""
        mock_run.return_value = self._make_completed_process("", returncode=1)
        mock_run.return_value.stderr = "ERROR: Video unavailable"
        with self.assertRaises(RuntimeError):
            self.resolve_yt_url("https://www.youtube.com/watch?v=fake")

    @patch("frcscout.pipeline.decode.subprocess.run")
    def test_raises_on_empty_output(self, mock_run):
        """Should raise RuntimeError when yt-dlp returns empty stdout."""
        mock_run.return_value = self._make_completed_process("   \n  \n")
        with self.assertRaises(RuntimeError):
            self.resolve_yt_url("https://www.youtube.com/watch?v=fake")


# --------------------------------------------------------------------- #
# Tests: is_live flag for URL sources
# --------------------------------------------------------------------- #

@unittest.skipUnless(_HAS_CV2, "cv2 not available")
class TestIsLiveFlag(unittest.TestCase):

    def setUp(self):
        from frcscout.pipeline.decode import _looks_like_url
        self._looks_like_url = _looks_like_url

    def test_http_url_is_live(self):
        self.assertTrue(self._looks_like_url("https://www.youtube.com/watch?v=abc"))

    def test_rtsp_is_live(self):
        self.assertTrue(self._looks_like_url("rtsp://192.168.1.1/stream"))

    def test_rtmp_is_live(self):
        self.assertTrue(self._looks_like_url("rtmp://live.example.com/app/key"))

    def test_file_path_not_live(self):
        self.assertFalse(self._looks_like_url("/path/to/match.mp4"))
        self.assertFalse(self._looks_like_url("match.mp4"))
        self.assertFalse(self._looks_like_url(r"C:\Users\me\match.mp4"))


if __name__ == "__main__":
    unittest.main()
