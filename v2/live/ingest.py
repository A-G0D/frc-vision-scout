"""live/ingest.py — YouTube live-stream ingestion layer for the FRC Vision Scout pipeline.

Turns a YouTube LIVE URL into a real-time frame stream that a downstream consumer
(ScoutPipeline or any CV pipeline) can pull from via a simple blocking ``get()`` call.
A ``FileSource`` fallback provides the same API over a local video file so the pipeline
can be tested without a network connection.

Architecture
------------
::

    YouTube URL
        │
        ▼
    yt-dlp --get-url         (resolve HLS/DASH manifest; re-run on every reconnect)
        │
        ▼
    ffmpeg rawvideo pipe     (demux, decode, decimate to target_fps, scale, bgr24)
        │
        ▼
    ProducerThread           (read pipe bytes → numpy, real-time clock gate, enqueue)
        │
        ▼
    Queue(maxsize=2)         (bounded; drop-oldest on overflow)
        │
        ▼
    caller's consume loop    (ingest.get() → (frame, capture_t))

The ingest layer owns ProducerThread and WatchdogThread only.  The consumer runs in
the caller's own thread and is never imported here.  Import is side-effect-free; no
threads are started until ``start()`` is called.

Usage — live stream
-------------------
::

    from live.ingest import YouTubeLiveIngest
    import time

    ingest = YouTubeLiveIngest("https://www.youtube.com/watch?v=LIVE_ID")
    ingest.start()
    try:
        while True:
            item = ingest.get(timeout=0.5)
            if item is None:
                continue
            frame, capture_t = item
            latency = time.perf_counter() - capture_t
            # ... pipeline.process(frame) ...
    finally:
        ingest.stop()

    # Or use as a context manager:
    with YouTubeLiveIngest("https://...") as ingest:
        for frame, capture_t in ingest:
            process(frame)

Usage — file fallback (no network)
-----------------------------------
::

    from live.ingest import FileSource

    with FileSource("data/footage/qual59.mp4", target_fps=30.0) as src:
        for frame, capture_t in src:
            process(frame)

Exact yt-dlp + ffmpeg command
------------------------------
Step 1 — resolve manifest (re-run on every reconnect to get a fresh token)::

    yt-dlp --no-warnings --get-url \
        --format "bestvideo[height<=720][fps<=30][ext=mp4]/bestvideo[height<=720][fps<=30]/bestvideo[height<=720]" \
        "https://www.youtube.com/watch?v=LIVE_ID"

Step 2 — pipe raw BGR24 frames to stdout::

    ffmpeg -loglevel error \
        -reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5 \
        -i "MANIFEST_URL" \
        -vf "fps=fps=30:round=down,scale=1280:720:flags=fast_bilinear" \
        -vcodec rawvideo -pix_fmt bgr24 \
        -an \
        -f rawvideo pipe:1

Key flag rationale:
  -reconnect / -reconnect_streamed   ffmpeg handles TCP drops without dying
  -reconnect_delay_max 5             caps retry to 5 s (default 120 s is unusable)
  fps=fps=N:round=down               real-time decimation — DROPS frames, never
                                     duplicates to fill.  Keeps playback real-time.
  scale=WxH:fast_bilinear            constant geometry so numpy buffer size is stable
  -pix_fmt bgr24                     OpenCV-native; no secondary cvtColor in hot path
  -an                                drop audio — saves ~5 % CPU

Dependencies
------------
- yt-dlp   (``pip install yt-dlp``)
- ffmpeg   (static binary; set FFMPEG_BIN env var or place on PATH)
- numpy    (already in scout env)
- opencv-python (already in scout env; used only by FileSource)
"""
from __future__ import annotations

import logging
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Generator, Iterator, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class FrameItem:
    """A single decoded video frame with provenance timestamps.

    All timestamps are ``time.perf_counter()`` values so latency arithmetic is
    straightforward: ``time.perf_counter() - capture_t`` gives pipeline latency.
    """

    frame: np.ndarray
    """H×W×3 uint8 BGR array — OpenCV-native format, no further conversion needed."""

    capture_t: float
    """Wall time when the last byte of this frame was read from the ffmpeg pipe."""

    enqueue_t: float
    """Wall time when this item was placed into the queue."""

    segment_seq: int = -1
    """HLS segment sequence number if available (parsed from ffmpeg stderr); else -1."""


@dataclass
class IngestStats:
    """Snapshot of ingest health metrics — safe to read from any thread."""

    state: str
    """One of: ``"idle"``, ``"connecting"``, ``"streaming"``, ``"reconnecting"``, ``"stopped"``."""

    captured: int
    """Total frames successfully read from the pipe since start()."""

    dropped: int
    """Frames discarded (queue overflow or real-time clock gate skips)."""

    reconnects: int
    """Number of times the ingest layer reconnected (stall or EOF)."""

    drop_rate: float
    """``dropped / max(1, captured + dropped)``."""

    latency_hint_s: float
    """Estimated glass-to-Python latency in seconds.

    This is the *minimum* measurable component: time from frame enqueue to now.
    Excludes HLS segment buffer delay (~4–6 s depending on ``hls_live_edge``).
    """

    queue_depth: int
    """Current number of frames waiting in the queue."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_BACKOFF_BASE = 0.5    # seconds for attempt 0
_BACKOFF_MAX  = 30.0   # seconds cap

_DEFAULT_MAX_RETRIES = 10  # stop after this many consecutive reconnect attempts


def _backoff(attempt: int) -> float:
    """Exponential back-off: 0.5, 1, 2, 4, … capped at 30 s."""
    return min(_BACKOFF_MAX, _BACKOFF_BASE * (2 ** attempt))


def _compute_decimation_period(source_fps: float, target_fps: float) -> float:
    """Return the minimum wall-clock period between accepted frames.

    When ``target_fps >= source_fps`` every frame is accepted (period = 0).
    When ``target_fps < source_fps`` we keep 1-in-N frames, N = source/target.

    This is the pure math used by the real-time clock gate; tested independently
    of any subprocess or queue.

    Parameters
    ----------
    source_fps:
        Frame rate actually arriving from the pipe (must be > 0).
    target_fps:
        Desired output frame rate (must be > 0).

    Returns
    -------
    float
        Minimum seconds between kept frames.  0.0 means keep every frame.
    """
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError(f"fps values must be > 0, got source={source_fps} target={target_fps}")
    if target_fps >= source_fps:
        return 0.0
    return 1.0 / target_fps


def _resolve_manifest(
    url: str,
    resolution: Tuple[int, int],
    ytdlp_bin: str = "yt-dlp",
) -> str:
    """Call yt-dlp --get-url to resolve a YouTube live URL to an HLS manifest URL.

    Parameters
    ----------
    url:
        YouTube watch URL, e.g. ``"https://www.youtube.com/watch?v=…"``.
    resolution:
        ``(width, height)`` target; used to build the format selector.
    ytdlp_bin:
        Path to the yt-dlp executable (default: ``"yt-dlp"`` on PATH).

    Returns
    -------
    str
        Direct manifest or segment URL suitable for passing to ffmpeg ``-i``.

    Raises
    ------
    RuntimeError
        If yt-dlp exits non-zero or returns empty output.
    """
    _, h = resolution
    # Prefer mp4 at or below the target height, <=30 fps.  Falls back to best available.
    fmt = (
        f"bestvideo[height<={h}][fps<=30][ext=mp4]"
        f"/bestvideo[height<={h}][fps<=30]"
        f"/bestvideo[height<={h}]"
    )
    cmd = [
        ytdlp_bin,
        "--no-warnings",
        "--get-url",
        "--format", fmt,
        url,
    ]
    log.debug("yt-dlp resolve: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"yt-dlp not found at '{ytdlp_bin}'. Install via: pip install yt-dlp"
        ) from None
    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp exited {result.returncode}: {result.stderr.strip()}"
        )
    lines = [ln.strip() for ln in result.stdout.strip().splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"yt-dlp returned no URL for: {url}")
    manifest_url = lines[0]
    log.info("Resolved manifest: %s…", manifest_url[:80])
    return manifest_url


def _build_ffmpeg_cmd(
    manifest_url: str,
    resolution: Tuple[int, int],
    target_fps: float,
    ffmpeg_bin: str = "ffmpeg",
) -> list:
    """Build the ffmpeg command that pipes raw BGR frames to stdout.

    The ``fps=fps=N:round=down`` filter performs real-time decimation: it drops
    source frames to hit the target rate.  It does NOT duplicate frames.

    Parameters
    ----------
    manifest_url:
        HLS/DASH manifest URL from ``_resolve_manifest``.
    resolution:
        ``(width, height)`` output geometry; kept constant so the numpy buffer size
        never changes between reconnects.
    target_fps:
        Target output frame rate for the decimation filter.
    ffmpeg_bin:
        Path to ffmpeg executable.
    """
    w, h = resolution
    vf = (
        f"fps=fps={target_fps}:round=down,"
        f"scale={w}:{h}:flags=fast_bilinear"
    )
    return [
        ffmpeg_bin,
        "-loglevel", "error",
        # Reconnect flags: ffmpeg handles TCP drops without dying
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", "5",
        "-i", manifest_url,
        "-vf", vf,
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-an",            # drop audio
        "-f", "rawvideo",
        "pipe:1",
    ]


# ---------------------------------------------------------------------------
# Core class — YouTubeLiveIngest
# ---------------------------------------------------------------------------


class YouTubeLiveIngest:
    """Real-time YouTube live-stream ingestion layer.

    Spawns a ProducerThread that reads raw BGR frames from an ffmpeg pipe and
    places them into a bounded queue.  A WatchdogThread monitors for stalls and
    drives the reconnect state machine.

    The caller's consume loop calls ``get()`` — a simple blocking call that returns
    a ``(frame, capture_t)`` tuple or ``None`` on timeout.

    Parameters
    ----------
    url:
        YouTube watch URL.
    target_fps:
        Desired output frame rate.  ffmpeg decimates to this rate; the Python-side
        clock gate provides a second layer of protection against burst delivery.
    resolution:
        ``(width, height)`` output geometry.
    queue_maxsize:
        Maximum frames in the internal queue.  Keep at 2 for minimum latency:
        one in the consumer, one pre-fetched.
    stall_timeout:
        Seconds without a new frame before the watchdog triggers reconnect.
    max_retries:
        Maximum consecutive reconnect attempts before the producer gives up and
        transitions to ``"stopped"``.  Set to 0 for unlimited retries.
    ytdlp_bin:
        Path to yt-dlp executable (default: ``"yt-dlp"`` on PATH).
    ffmpeg_bin:
        Path to ffmpeg executable.  Falls back to ``FFMPEG_BIN`` env var, then
        ``"ffmpeg"`` on PATH.
    """

    # State machine values
    _STATE_IDLE         = "idle"
    _STATE_CONNECTING   = "connecting"
    _STATE_STREAMING    = "streaming"
    _STATE_RECONNECTING = "reconnecting"
    _STATE_STOPPED      = "stopped"

    def __init__(
        self,
        url: str,
        target_fps: float = 30.0,
        resolution: Tuple[int, int] = (1280, 720),
        queue_maxsize: int = 2,
        stall_timeout: float = 3.0,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        ytdlp_bin: str = "yt-dlp",
        ffmpeg_bin: Optional[str] = None,
    ) -> None:
        self.url = url
        self.target_fps = float(target_fps)
        self.resolution = resolution
        self.queue_maxsize = int(queue_maxsize)
        self.stall_timeout = float(stall_timeout)
        self.max_retries = int(max_retries)
        self.ytdlp_bin = ytdlp_bin
        self.ffmpeg_bin = ffmpeg_bin or os.environ.get("FFMPEG_BIN", "ffmpeg")

        self._queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._state = self._STATE_IDLE
        self._state_lock = threading.Lock()

        # Counters — written by producer, read by any thread (GIL protects int ops)
        self._captured: int = 0
        self._dropped: int = 0
        self._reconnects: int = 0

        # Timing
        self._last_frame_t: float = 0.0
        self._frame_bytes: int = resolution[0] * resolution[1] * 3

        # Subprocess handle (producer owns; watchdog only signals, never reaps)
        self._proc: Optional[subprocess.Popen] = None

        # Stop signal
        self._stop_event = threading.Event()

        # Threads (created in start())
        self._producer_thread: Optional[threading.Thread] = None
        self._watchdog_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the producer and watchdog threads.

        Safe to call once.  Calling again after ``stop()`` is not supported.
        """
        if self._state != self._STATE_IDLE:
            raise RuntimeError(
                "start() called on a non-idle ingest (already started or stopped)"
            )
        self._set_state(self._STATE_CONNECTING)
        self._producer_thread = threading.Thread(
            target=self._producer_loop,
            daemon=True,
            name="ingest-producer",
        )
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            daemon=True,
            name="ingest-watchdog",
        )
        self._producer_thread.start()
        self._watchdog_thread.start()
        log.info(
            "Ingest started for %s @ %dx%d %.0ffps",
            self.url, *self.resolution, self.target_fps,
        )

    def stop(self, join_timeout: float = 3.0) -> None:
        """Signal threads to stop and wait for them to exit.

        Safe to call from any thread.  Idempotent.
        """
        self._stop_event.set()
        self._terminate_proc()
        if self._producer_thread and self._producer_thread.is_alive():
            self._producer_thread.join(timeout=join_timeout)
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=join_timeout)
        self._set_state(self._STATE_STOPPED)
        log.info(
            "Ingest stopped. captured=%d dropped=%d reconnects=%d",
            self._captured, self._dropped, self._reconnects,
        )

    def get(self, timeout: float = 0.5) -> Optional[Tuple[np.ndarray, float]]:
        """Pull the next frame from the queue.

        Returns
        -------
        ``(frame, capture_t)`` where ``frame`` is an ``H×W×3`` uint8 BGR numpy
        array and ``capture_t`` is a ``perf_counter()`` timestamp.

        Returns ``None`` if no frame is available within ``timeout`` seconds.

        Notes
        -----
        Drains any backlog before processing: if the queue has more than one frame
        waiting, all but the newest are discarded (counted as drops) so the consumer
        always receives the most recent frame and stays real-time.
        """
        # Drain backlog: keep only the newest frame
        item: Optional[FrameItem] = None
        while True:
            try:
                candidate = self._queue.get_nowait()
                if item is not None:
                    self._dropped += 1   # count skipped frames as drops
                item = candidate
            except queue.Empty:
                break

        if item is not None:
            return item.frame, item.capture_t

        # Queue was empty; block for up to timeout
        try:
            item = self._queue.get(timeout=timeout)
            return item.frame, item.capture_t
        except queue.Empty:
            return None

    def __iter__(self) -> Iterator[Tuple[np.ndarray, float]]:
        """Iterate over ``(frame, capture_t)`` tuples until the ingest is stopped.

        Yields frames indefinitely; breaks when ``state == "stopped"`` and the
        queue is empty.

        Example
        -------
        ::

            with YouTubeLiveIngest("https://...") as ingest:
                for frame, capture_t in ingest:
                    latency = time.perf_counter() - capture_t
                    process(frame)
        """
        while True:
            if self._state == self._STATE_STOPPED and self._queue.empty():
                return
            result = self.get(timeout=0.5)
            if result is not None:
                yield result

    def __enter__(self) -> "YouTubeLiveIngest":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    def stats(self) -> IngestStats:
        """Return a snapshot of ingest health metrics (safe to call from any thread)."""
        total = self._captured + self._dropped
        drop_rate = self._dropped / max(1, total)
        q_depth = self._queue.qsize()
        latency_hint = (
            time.perf_counter() - self._last_frame_t if self._last_frame_t else 0.0
        )
        return IngestStats(
            state=self._state,
            captured=self._captured,
            dropped=self._dropped,
            reconnects=self._reconnects,
            drop_rate=drop_rate,
            latency_hint_s=latency_hint,
            queue_depth=q_depth,
        )

    # ------------------------------------------------------------------
    # Internal — state machine
    # ------------------------------------------------------------------

    def _set_state(self, state: str) -> None:
        with self._state_lock:
            self._state = state
        log.debug("Ingest state -> %s", state)

    # ------------------------------------------------------------------
    # Internal — process management
    # ------------------------------------------------------------------

    def _start_proc(self, manifest_url: str) -> subprocess.Popen:
        """Spawn a new ffmpeg process piping raw BGR frames to stdout."""
        cmd = _build_ffmpeg_cmd(
            manifest_url,
            resolution=self.resolution,
            target_fps=self.target_fps,
            ffmpeg_bin=self.ffmpeg_bin,
        )
        log.debug("ffmpeg cmd: %s", " ".join(cmd))
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            # DEVNULL, not PIPE: stderr was captured "for stall diagnosis" but is never
            # read, so on a verbose reconnect the 64KB pipe buffer fills and ffmpeg blocks
            # while the producer blocks on stdout.read() — a classic pipe deadlock. We do
            # not parse stderr (segment_seq is always -1), so discard it.
            stderr=subprocess.DEVNULL,
            bufsize=self._frame_bytes * 4,
        )

    def _terminate_proc(self) -> None:
        """Terminate the ffmpeg subprocess if running.

        On Windows, ``terminate()`` sends ``TerminateProcess``.  We then wait up to
        2 s so the OS reclaims the socket before we reconnect.
        """
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception as exc:
                log.warning("ffmpeg terminate error: %s", exc)
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = None

    # ------------------------------------------------------------------
    # Internal — producer thread
    # ------------------------------------------------------------------

    def _producer_loop(self) -> None:
        """Main loop for the producer thread.

        Manages the yt-dlp -> ffmpeg pipeline lifecycle, performs real-time
        clock-gating, and places FrameItems into the queue.  Stops after
        ``max_retries`` consecutive failures if ``max_retries > 0``.
        """
        consecutive_failures = 0

        while not self._stop_event.is_set():
            # --- Check max-retries cap ---
            if self.max_retries > 0 and consecutive_failures >= self.max_retries:
                log.error(
                    "Ingest giving up after %d consecutive failures (max_retries=%d)",
                    consecutive_failures, self.max_retries,
                )
                self._set_state(self._STATE_STOPPED)
                return

            # --- Resolve manifest ---
            self._set_state(self._STATE_CONNECTING)
            try:
                manifest_url = _resolve_manifest(
                    self.url,
                    resolution=self.resolution,
                    ytdlp_bin=self.ytdlp_bin,
                )
            except Exception as exc:
                delay = _backoff(consecutive_failures)
                log.error(
                    "Manifest resolution failed: %s — retry in %.1fs (attempt %d/%s)",
                    exc, delay, consecutive_failures + 1,
                    str(self.max_retries) if self.max_retries > 0 else "unlimited",
                )
                self._stop_event.wait(timeout=delay)
                consecutive_failures += 1
                continue

            # --- Spawn ffmpeg ---
            try:
                self._proc = self._start_proc(manifest_url)
            except Exception as exc:
                delay = _backoff(consecutive_failures)
                log.error("ffmpeg spawn failed: %s — retry in %.1fs", exc, delay)
                self._stop_event.wait(timeout=delay)
                consecutive_failures += 1
                continue

            # Reset the freshness clock to 0.0 BEFORE announcing STREAMING so the
            # watchdog (which skips while _last_frame_t == 0.0) waits for the first
            # real frame of this connection instead of comparing against the stale
            # pre-stall timestamp and killing freshly-spawned ffmpeg (reconnect storm).
            self._last_frame_t = 0.0
            self._set_state(self._STATE_STREAMING)
            consecutive_failures = 0   # successful connect resets the failure counter

            # --- Read loop ---
            clean_eof = self._read_frames(self._proc)

            # --- Cleanup after EOF / error ---
            self._terminate_proc()
            if self._stop_event.is_set():
                break

            self._set_state(self._STATE_RECONNECTING)
            self._reconnects += 1
            consecutive_failures += 1
            delay = _backoff(self._reconnects)
            log.warning(
                "Stream ended (eof=%s) — reconnecting in %.1fs (reconnect #%d)",
                clean_eof, delay, self._reconnects,
            )
            self._stop_event.wait(timeout=delay)

    def _read_frames(self, proc: subprocess.Popen) -> bool:
        """Read raw frames from ``proc.stdout`` until EOF or stop signal.

        Applies a real-time clock gate: if the pipe is delivering frames faster
        than ``target_fps``, excess frames are discarded without being queued.
        If the pipe falls behind and the gate resets, ``next_t`` is snapped to
        ``now`` so we never accumulate scheduling debt.

        Parameters
        ----------
        proc:
            The running ffmpeg subprocess.

        Returns
        -------
        bool
            ``True`` if loop exited cleanly on EOF; ``False`` if interrupted by
            stop signal or read error.
        """
        period = 1.0 / self.target_fps
        next_t = time.perf_counter()

        while not self._stop_event.is_set():
            # Read exactly one frame's worth of bytes
            try:
                raw = proc.stdout.read(self._frame_bytes)
            except Exception as exc:
                log.error("Pipe read error: %s", exc)
                return False

            if len(raw) < self._frame_bytes:
                log.info(
                    "ffmpeg pipe EOF (got %d bytes, expected %d)",
                    len(raw), self._frame_bytes,
                )
                return True

            now = time.perf_counter()
            self._last_frame_t = now

            # Real-time clock gate: drop frames that arrive ahead of schedule.
            # This is the Python-side guard; ffmpeg's fps filter is the primary one.
            if now < next_t:
                self._dropped += 1
                continue

            # Advance schedule; if already behind, snap forward (no catch-up debt)
            next_t += period
            if next_t < now:
                next_t = now

            # Decode bytes -> numpy (zero-copy reshape)
            w, h = self.resolution
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
            capture_t = now

            item = FrameItem(
                frame=frame,
                capture_t=capture_t,
                enqueue_t=time.perf_counter(),
            )

            self._enqueue(item)
            self._captured += 1

        return False

    def _enqueue(self, item: FrameItem) -> None:
        """Put ``item`` into the queue, dropping the oldest frame if full."""
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._dropped += 1
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                self._dropped += 1   # defensive; should not happen

    # ------------------------------------------------------------------
    # Internal — watchdog thread
    # ------------------------------------------------------------------

    def _watchdog_loop(self) -> None:
        """Monitor for frame stalls and force reconnect when the pipe goes silent.

        A "stall" is defined as ``stall_timeout`` seconds passing without the
        producer updating ``_last_frame_t``.  When detected, the watchdog terminates
        the ffmpeg process; the producer loop then advances through its reconnect
        state machine.

        Poll interval is ``stall_timeout / 3`` to detect stalls within ~one interval
        of the threshold.
        """
        poll_interval = max(0.5, self.stall_timeout / 3.0)
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=poll_interval)
            if self._stop_event.is_set():
                break
            if self._state != self._STATE_STREAMING:
                continue
            if self._last_frame_t == 0.0:
                continue
            age = time.perf_counter() - self._last_frame_t
            if age > self.stall_timeout:
                log.warning(
                    "Stall detected: no frame for %.1fs (threshold %.1fs) — killing ffmpeg",
                    age, self.stall_timeout,
                )
                self._set_state(self._STATE_RECONNECTING)
                self._terminate_proc()
                # producer_loop detects proc is gone and re-enters reconnect path


# ---------------------------------------------------------------------------
# FileSource — same API, reads from a local video file
# ---------------------------------------------------------------------------


class FileSource:
    """Drop-in replacement for ``YouTubeLiveIngest`` that reads from a local file.

    Provides the identical ``get()`` / ``__iter__`` / context-manager API so the
    rest of the pipeline can be tested without a network connection or a live stream.

    The source decodes via OpenCV at the file's native frame rate, applying the
    same real-time clock gate as the live source (drops excess frames; does NOT
    slow down if ``target_fps < source_fps``).

    Parameters
    ----------
    path:
        Path to the video file (e.g. ``"data/footage/qual59.mp4"``).
    target_fps:
        Target frame rate for decimation.  If 0 or ``None``, uses the file's
        native fps without decimation.
    queue_maxsize:
        Bound on the internal queue depth.
    loop:
        If ``True``, restart from frame 0 when the file ends (useful for
        long-running pipeline tests without a large test file).
    resolution:
        If provided, resize each frame to ``(width, height)``.  If ``None``,
        frames are returned at the file's native resolution.
    """

    def __init__(
        self,
        path: str,
        target_fps: Optional[float] = None,
        queue_maxsize: int = 2,
        loop: bool = False,
        resolution: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.path = path
        self.target_fps = target_fps   # None means use native fps
        self.queue_maxsize = int(queue_maxsize)
        self.loop = loop
        self.resolution = resolution

        self._queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._stop_event = threading.Event()
        self._producer_thread: Optional[threading.Thread] = None
        self._state = "idle"

        # Counters
        self._captured: int = 0
        self._dropped: int = 0

    # ------------------------------------------------------------------
    # Public API (mirrors YouTubeLiveIngest)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start reading frames in a background thread."""
        if self._state != "idle":
            raise RuntimeError("FileSource.start() called on a non-idle source")
        self._state = "streaming"
        self._producer_thread = threading.Thread(
            target=self._producer_loop,
            daemon=True,
            name="filesource-producer",
        )
        self._producer_thread.start()

    def stop(self, join_timeout: float = 3.0) -> None:
        """Stop the producer thread."""
        self._stop_event.set()
        if self._producer_thread and self._producer_thread.is_alive():
            self._producer_thread.join(timeout=join_timeout)
        self._state = "stopped"

    def get(self, timeout: float = 0.5) -> Optional[Tuple[np.ndarray, float]]:
        """Pull the next frame; returns ``(frame, capture_t)`` or ``None`` on timeout."""
        item: Optional[FrameItem] = None
        while True:
            try:
                candidate = self._queue.get_nowait()
                if item is not None:
                    self._dropped += 1
                item = candidate
            except queue.Empty:
                break

        if item is not None:
            return item.frame, item.capture_t

        try:
            item = self._queue.get(timeout=timeout)
            return item.frame, item.capture_t
        except queue.Empty:
            return None

    def __iter__(self) -> Iterator[Tuple[np.ndarray, float]]:
        """Iterate over ``(frame, capture_t)`` until the file ends (or ``stop()`` called)."""
        while True:
            if self._state == "stopped" and self._queue.empty():
                return
            result = self.get(timeout=0.5)
            if result is not None:
                yield result

    def __enter__(self) -> "FileSource":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    def stats(self) -> IngestStats:
        """Return a health snapshot in the same shape as ``YouTubeLiveIngest.stats()``."""
        total = self._captured + self._dropped
        drop_rate = self._dropped / max(1, total)
        return IngestStats(
            state=self._state,
            captured=self._captured,
            dropped=self._dropped,
            reconnects=0,
            drop_rate=drop_rate,
            latency_hint_s=0.0,
            queue_depth=self._queue.qsize(),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _producer_loop(self) -> None:
        import cv2

        while not self._stop_event.is_set():
            cap = cv2.VideoCapture(self.path)
            if not cap.isOpened():
                log.error("FileSource: cannot open %s", self.path)
                self._state = "stopped"
                return

            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            effective_target = self.target_fps if self.target_fps else src_fps
            period = _compute_decimation_period(src_fps, effective_target)

            # The file is read as fast as possible; we clock-gate to target_fps.
            next_t = time.perf_counter()

            while not self._stop_event.is_set():
                ok, bgr = cap.read()
                if not ok:
                    break   # EOF

                now = time.perf_counter()

                # Real-time clock gate — same logic as YouTubeLiveIngest._read_frames
                if period > 0 and now < next_t:
                    self._dropped += 1
                    continue

                if period > 0:
                    next_t += period
                    if next_t < now:
                        next_t = now

                if self.resolution is not None:
                    w, h = self.resolution
                    bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)

                capture_t = now
                item = FrameItem(
                    frame=bgr,
                    capture_t=capture_t,
                    enqueue_t=time.perf_counter(),
                )
                self._enqueue(item)
                self._captured += 1

            cap.release()

            if not self.loop:
                break

        self._state = "stopped"

    def _enqueue(self, item: FrameItem) -> None:
        """Put item into queue, dropping oldest on overflow."""
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._dropped += 1
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                self._dropped += 1


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def open_stream(
    url: str,
    target_fps: float = 30.0,
    resolution: Tuple[int, int] = (1280, 720),
    **kwargs,
) -> YouTubeLiveIngest:
    """Create and start a ``YouTubeLiveIngest`` in one call.

    Returns a started ingest object.  Caller is responsible for calling ``stop()``.
    Use as a context manager via ``YouTubeLiveIngest`` directly for automatic cleanup::

        with YouTubeLiveIngest("https://youtube.com/watch?v=X") as ingest:
            for frame, capture_t in ingest:
                process(frame)
    """
    ingest = YouTubeLiveIngest(
        url=url, target_fps=target_fps, resolution=resolution, **kwargs
    )
    ingest.start()
    return ingest
