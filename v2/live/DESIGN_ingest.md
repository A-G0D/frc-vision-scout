# Live YouTube Ingestion Layer — Design Notes

**Scope:** Turn a YouTube LIVE URL into a real-time frame stream that feeds the FRC
Vision Scout CV pipeline on Windows (Intel Core Ultra 7 255U, CPU/OpenVINO).
Target: ≤ 2 s glass-to-frame latency, ≥ 99 % uptime across network hiccups.

---

## 1. Resolving a YouTube Live URL to a Raw Stream

### Why not `cv2.VideoCapture(url)` directly?

`VideoCapture` on a YouTube URL calls libavformat's lavf demuxer, which has no
YouTube-aware HLS manifest logic. It breaks on rate-limit redirects, geo-token
rotation, and adaptive-bitrate playlist switching. `yt-dlp` + `streamlink` both
handle all of these and expose a stable, re-resolable HLS/DASH URL.

### Preferred path: yt-dlp → ffmpeg → stdout pipe → OpenCV

```
YouTube URL
    │
    ▼
yt-dlp --get-url          ← resolves the best HLS manifest URL, no download
    │  (m3u8 URL or direct mp4 segment URL)
    ▼
ffmpeg -i <manifest_url>  ← demux + decode HLS/DASH, handles segment rotation
       -vf fps=30          ← decimate/upsample to target fps IN the demuxer
       -vcodec rawvideo
       -pix_fmt bgr24      ← OpenCV native format, zero extra conversion
       -f rawvideo
       pipe:1              ← stdout
    │
    ▼
Python reads width×height×3 bytes per frame from stdout pipe → numpy array
```

### Exact command lines

#### Step 1 — resolve manifest URL (run once, re-run on reconnect)

```powershell
yt-dlp --no-warnings --get-url `
    --format "bestvideo[height<=720][fps<=30][ext=mp4]/bestvideo[height<=720]" `
    "https://www.youtube.com/watch?v=LIVE_ID"
```

- `--get-url` prints one or more direct segment/manifest URLs and exits immediately.
- The format selector picks **720 p ≤ 30 fps mp4** first — this is the sweet spot for
  the Core Ultra 255U: 1080p decoding adds ~8 ms/frame CPU overhead with no scouting
  benefit; 480p loses OCR accuracy on scoreboard digits.
- Falls back to best available ≤ 720 if the exact codec is absent.
- Output is one line, e.g. `https://manifest.googlevideo.com/api/manifest/hls/...`

#### Step 2 — ffmpeg rawvideo pipe

```powershell
ffmpeg -loglevel error `
    -reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5 `
    -i "MANIFEST_URL" `
    -vf "fps=fps=30,scale=1280:720:flags=fast_bilinear" `
    -vcodec rawvideo -pix_fmt bgr24 `
    -an `
    -f rawvideo pipe:1
```

Flag rationale:

| Flag | Purpose |
|------|---------|
| `-reconnect 1 -reconnect_streamed 1` | ffmpeg reconnects on TCP drop without dying |
| `-reconnect_delay_max 5` | caps retry back-off at 5 s; default is 120 s (unusable) |
| `fps=fps=30` | real-time decimation filter — **drops frames to hit 30 fps, never duplicates to fill**; use `fps=fps=N:round=down` to match exactly |
| `scale=1280:720:flags=fast_bilinear` | lock output geometry so the numpy buffer size is constant |
| `-pix_fmt bgr24` | OpenCV-native; avoids a secondary `cvtColor` in the hot path |
| `-an` | drop audio stream; saves ~5 % CPU |

#### Step 3 — Python reads the pipe

```python
import subprocess, numpy as np

W, H = 1280, 720
frame_bytes = W * H * 3

proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=frame_bytes * 4)

while True:
    raw = proc.stdout.read(frame_bytes)
    if len(raw) < frame_bytes:
        break                         # EOF / ffmpeg died → trigger reconnect
    frame = np.frombuffer(raw, np.uint8).reshape((H, W, 3))
    yield frame
```

### Alternative: streamlink

Use `streamlink` instead of yt-dlp when the stream is behind Cloudflare or has
aggressive token rotation (yt-dlp's `--get-url` sometimes returns a short-lived
token that expires before ffmpeg can start):

```powershell
streamlink --stdout `
    --hls-live-edge 2 `
    --hls-segment-stream-data `
    "https://www.youtube.com/watch?v=LIVE_ID" best `
| ffmpeg -loglevel error -i pipe:0 `
    -vf "fps=fps=30,scale=1280:720:flags=fast_bilinear" `
    -vcodec rawvideo -pix_fmt bgr24 -an -f rawvideo pipe:1
```

`--hls-live-edge 2` minimises buffering to 2 segments (the lowest the server
permits); this is the primary knob for reducing stream latency. Default is 3
segments (~6 s at 2 s/segment).

### Resolution / FPS selection logic

```
Priority                 Condition
─────────────────────────────────────────────────────
720p30                   Default; OCR + detection are calibrated to this
1080p30                  Only if score-OCR accuracy becomes a blocker
480p30                   Low-bandwidth fallback (auto-selected on repeated reconnects)
target fps override      --feed-fps arg; decimation happens in ffmpeg, not Python
```

---

## 2. Real-Time Discipline

### The core invariant

**The pipeline must track wall-clock time, not frame count.** Every design
decision flows from this: we are a live viewer, not a video player.

### Bounded queue with drop-oldest

```
ffmpeg pipe → ProducerThread → Queue(maxsize=2) → ConsumerThread → pipeline
                                     ↑
                             drop oldest on overflow
                             (count as "dropped")
```

- `maxsize=2`: one frame "in flight" in the consumer + one pre-fetched. Any deeper
  and latency creeps up by `N × frame_period`.
- `queue.put_nowait()` with a `try/except Full: queue.get_nowait(); dropped += 1`
  pattern is faster than `put(block=False)` + polling.
- The consumer always drains any backlog before processing: `while not q.empty(): q.get_nowait(); dropped += 1` then take the last one.

### Real-time decimation (NOT slow-motion)

The `fps=fps=30` ffmpeg filter is the primary decimation point. It:

1. Reads source segments at whatever rate the manifest delivers (often 60 fps).
2. Drops frames to emit exactly N frames per second of **wall-clock time**.
3. Does **not** duplicate frames to fill gaps — it uses the `round=down` mode.

In Python, an additional decimation guard prevents the queue from racing ahead
on fast networks:

```python
# Producer: pace to target_fps using perf_counter
period = 1.0 / target_fps
next_t = time.perf_counter()
for frame in _read_pipe(proc):
    now = time.perf_counter()
    if now < next_t:
        continue                  # decimate: discard ahead-of-schedule frames
    _enqueue(frame, now)
    next_t += period
    if next_t < now:              # fell behind: reset clock, do NOT accumulate debt
        next_t = now
```

The `next_t = now` reset on fall-behind is the key real-time correction: it
prevents a slow consumer from causing the producer to flood the queue trying to
"catch up" to a schedule it already missed.

### Reconnect / backoff on stall or EOF

```
State machine:

  CONNECTING ──(ffmpeg started)──► STREAMING
      ▲                                │
      │                         (EOF / pipe read < frame_bytes
      │                          OR no frame for >stall_timeout)
      │                                │
      └─────── RECONNECTING ◄──────────┘
                    │
             exponential backoff:
             attempt 0 → 0.5 s
             attempt 1 → 1.0 s
             attempt 2 → 2.0 s
             attempt N → min(2^N × 0.5, 30.0) s
                    │
             re-run yt-dlp --get-url   ← get a fresh manifest (token rotation)
             re-spawn ffmpeg
```

Stall detection: a watchdog thread checks `time.perf_counter() - last_frame_t`.
If `> stall_timeout` (default 3 s), it sends `SIGTERM` to ffmpeg and triggers
RECONNECTING.

On Windows, `subprocess.Popen.terminate()` sends `TerminateProcess`; `proc.kill()`
is the same. Use `proc.communicate(timeout=2)` after terminate to reap the child.

---

## 3. Producer / Consumer API

### Module contract

```python
# live/ingest.py — the only public surface the pipeline imports

from live.ingest import YouTubeLiveIngest

ingest = YouTubeLiveIngest(
    url="https://www.youtube.com/watch?v=LIVE_ID",
    target_fps=30,
    resolution=(1280, 720),
    queue_maxsize=2,
    stall_timeout=3.0,
)

ingest.start()                       # spawns producer + watchdog threads

# Consumer loop (runs in the pipeline's own thread):
while True:
    item = ingest.get(timeout=0.5)   # blocks up to timeout; returns None on timeout
    if item is None:
        continue
    frame, capture_t = item          # numpy BGR array + perf_counter capture time
    result = pipeline.process(frame)
    latency = time.perf_counter() - capture_t

ingest.stop()                        # graceful shutdown; joins threads
```

### Key properties the consumer can read

```python
ingest.state          # "connecting" | "streaming" | "reconnecting" | "stopped"
ingest.stats()        # dict: captured, dropped, drop_rate, reconnects, latency_hint_s
```

### Thread ownership

```
Main thread      → constructs, calls start() / stop()
ProducerThread   → reads ffmpeg pipe, decimates, enqueues
WatchdogThread   → detects stalls, manages reconnect state machine
ConsumerThread   → NOT owned by ingest; caller runs its own consume loop
```

This keeps the ingest layer dependency-free: the pipeline does not need to know
about reconnect logic, and the ingest layer does not import any CV or scout code.

---

## 4. Latency Measurement Hooks

### Glass-to-frame definition

```
glass_time      ← the moment a photon left the physical scene
capture_time    ← time.perf_counter() when the frame bytes finished reading from pipe
pipeline_time   ← time.perf_counter() when pipeline.process() returns
display_time    ← time.perf_counter() when the annotated frame is composited to screen
```

We cannot measure `glass_time` directly. The **measurable proxy** is:

```
stream_delay  ≈  (HLS segment duration × hls_live_edge) + ffmpeg buffer
                ≈  2 s × 2 segments = ~4 s  (streamlink --hls-live-edge 2)
                ≈  2 s × 3 segments = ~6 s  (default)

glass_to_frame_estimate = stream_delay + (capture_time - segment_fetch_time)
```

### Instrumented timestamps in the frame item

```python
@dataclass
class FrameItem:
    frame: np.ndarray          # H×W×3 uint8 BGR
    capture_t: float           # perf_counter() at pipe-read completion
    enqueue_t: float           # perf_counter() when put into queue
    segment_seq: int           # ffmpeg segment number (parsed from stderr if enabled)
```

### Latency log format (CSV)

```
wall_t, capture_t, enqueue_t, dequeue_t, process_t, display_t,
queue_depth, dropped_total, reconnects, state
```

`wall_t - capture_t` measures producer scheduling jitter.
`dequeue_t - enqueue_t` measures queue wait (= pipeline compute time).
`display_t - capture_t` is the pipeline's end-to-end latency, excluding stream delay.

### Enabling ffmpeg pts timestamps

Add `-vf "fps=fps=30,setpts=N/(30*TB),scale=..."` and parse `[Parsed_setpts_...]`
lines from ffmpeg's stderr to correlate segment sequence numbers. This is optional
instrumentation for debugging, not required for production.

---

## 5. Windows-Specific Notes

- **ffmpeg binary**: Use a static Windows build from
  `https://www.gyan.dev/ffmpeg/builds/` (ffmpeg-release-essentials.zip).
  Place `ffmpeg.exe` on PATH or set `FFMPEG_BIN` env var.
- **yt-dlp**: `pip install yt-dlp` or standalone `.exe` from the yt-dlp releases.
- **streamlink**: `pip install streamlink`.
- **Pipe buffer on Windows**: The default Windows named pipe buffer is 64 KB.
  `bufsize=frame_bytes * 4` in `Popen` sets the Python-side read buffer; ffmpeg's
  output buffer is controlled by `-bufsize 512k` if needed.
- **Process cleanup**: Always call `proc.terminate()` then `proc.wait(timeout=3)`
  in a `finally` block. Orphaned ffmpeg processes on Windows will keep the network
  socket open and prevent clean reconnect.
- **OpenVINO device affinity**: The ingest layer is CPU-only (pipe I/O + numpy
  reshape). OpenVINO inference runs in the consumer thread; the ingest layer is
  intentionally decoupled from device selection.

---

## 6. Dependency Summary

```
yt-dlp        >= 2024.01.01   pip install yt-dlp
streamlink    >= 6.5.0        pip install streamlink  (optional, alternative to yt-dlp)
ffmpeg        >= 6.0          static binary on PATH
numpy         >= 1.24         already in scout env
opencv-python >= 4.8          already in scout env
```

No new Python packages are required beyond what the scout pipeline already uses,
except `yt-dlp` (and optionally `streamlink`).
