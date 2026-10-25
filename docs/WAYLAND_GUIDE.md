# Showing DeepStream output on a Wayland desktop, step by step

*DeepStream 7.0 (x86, dGPU) in Docker · deepstream-test5-app · Triton · YOLO11 PGIE + YOLO11-cls SGIE · Ubuntu on Wayland · tested 2026-10-25*

Ubuntu now uses **Wayland** for its desktop by default. Most DeepStream examples assume **X11**, so the first
question is: *how do I get the DeepStream window onto my screen?* There are two ways. This post walks through both,
including the three problems I hit on the native route and how each was fixed.

| | Route A: through XWayland | Route B: native Wayland |
|---|---|---|
| Video sink | `nveglglessink` (NVIDIA, X11 only) | `waylandsink` (standard GStreamer) |
| Works with `deepstream-test5-app` | ✅ yes (`[sink] type=2`) | ❌ no Wayland sink type, so use `gst-launch-1.0` or your own app |
| Speed (4 × 1080p, GTX 1650 Ti) | 30 FPS per stream (real time) | ~28 FPS at the end of a 30 s run, 19 FPS average; some frames dropped |
| Frames stay on GPU | ✅ | ❌ copied to CPU memory for Wayland |
| Script | `scripts/run_deepstream.sh test5_yolo11_triton_display.txt` | `scripts/run_wayland_native.sh` |

**Short answer:** use **Route A** day to day. It's faster and works with `deepstream-test5-app`. Use **Route B** when you need
a true Wayland client, e.g. no XWayland on the system, or a kiosk/compositor setup.

---

## Step 0: Check that you're on Wayland

```bash
echo $XDG_SESSION_TYPE        # wayland
echo $WAYLAND_DISPLAY         # wayland-0
echo $XDG_RUNTIME_DIR         # /run/user/1000
ls -la $XDG_RUNTIME_DIR/wayland-0   # srwxrwxr-x ... wayland-0   <- the Wayland socket
echo $DISPLAY                 # :0  <- XWayland is also running (needed for Route A)
ls /tmp/.X11-unix             # X0 ...
```

Wayland is just a **socket file**. A program that can reach `/run/user/1000/wayland-0` and is allowed to talk to it
can open a window. Docker containers can't see that file unless you mount it in.

---

## Route A: XWayland (recommended)

```
DeepStream (nveglglessink, X11) ──► /tmp/.X11-unix/X0 ──► XWayland (:0) ──► Wayland desktop
```

**Why.** NVIDIA's `nveglglessink` on x86 only knows X11. Its own help text says window creation
*"is currently only supported when the sink is used under X11"*. Ubuntu runs **XWayland**, a translator that pretends
to be an X11 screen and shows those windows on the Wayland desktop.

### A1. Allow local programs to open windows
```bash
xhost +local:
```

### A2. Give the container the X11 screen
```bash
docker run ... \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  ...
```

### A3. Use the EGL sink in the deepstream-test5-app config
```ini
[sink2]
enable=1
type=2        # EglSink
sync=1
```

### A4. Run
```bash
./scripts/run_deepstream.sh test5_yolo11_triton_display.txt
```
That's the whole route. It ran at 30 FPS per stream with labels on every box (`deepstream-test5-app -t`).

---

## Route B: native Wayland with `waylandsink`

```
DeepStream ─► nvdsosd ─► nvvideoconvert (GPU→CPU, BGRx) ─► waylandsink ─► /tmp/wayland-0 ─► Wayland desktop
```

### B1. Check that the plugin is in the image
```bash
docker run --rm --entrypoint gst-inspect-1.0 nvcr.io/nvidia/deepstream:7.0-triton-multiarch waylandsink
```
It's there (`waylandsink: wayland video sink`), along with `libwayland-client`.

### B2. Smoke test: a test pattern in a Wayland window
```bash
docker run --rm \
  -u $(id -u):$(id -g) \
  -e WAYLAND_DISPLAY=wayland-0 \
  -e XDG_RUNTIME_DIR=/tmp \
  -v $XDG_RUNTIME_DIR/wayland-0:/tmp/wayland-0 \
  --entrypoint gst-launch-1.0 nvcr.io/nvidia/deepstream:7.0-triton-multiarch \
  videotestsrc num-buffers=90 ! video/x-raw,width=640,height=360,framerate=30/1 ! waylandsink
```
A color-bar window should appear for 3 seconds. Three settings matter here:

| Setting | Why |
|---|---|
| `-v $XDG_RUNTIME_DIR/wayland-0:/tmp/wayland-0` | puts the Wayland socket inside the container |
| `-u $(id -u):$(id -g)` | the socket only accepts **your** user, not root |
| `-e XDG_RUNTIME_DIR=/tmp` | must be a **writable** folder (see Problem 1) |

### B3. The DeepStream pipeline
`deepstream-test5-app` can't output to Wayland, so build the same pipeline with `gst-launch-1.0`:
```bash
gst-launch-1.0 -e \
  nvstreammux name=m batch-size=4 width=1920 height=1080 batched-push-timeout=40000 ! \
  nvinferserver config-file-path=config_pgie_yolo11_triton_nocudashare.txt batch-size=4 ! \
  nvtracker ll-lib-file=.../libnvds_nvmultiobjecttracker.so ll-config-file=.../config_tracker_NvDCF_perf.yml ! \
  nvinferserver config-file-path=config_sgie_yolo11_cls_triton_nocudashare.txt ! \
  nvmultistreamtiler rows=2 columns=2 width=1280 height=720 ! \
  nvvideoconvert ! nvdsosd ! \
  nvvideoconvert ! "video/x-raw,format=BGRx" ! waylandsink sync=true \
  uridecodebin uri=file:///.../sample_1080p_h264.mp4 ! m.sink_0 \
  ...  (sink_1 .. sink_3)
```
Why each piece is there:
- **`nvvideoconvert ! video/x-raw,format=BGRx`**: DeepStream frames live in GPU memory (NVMM). `waylandsink`
  needs ordinary CPU memory in a format Wayland understands, so this element copies and converts them.
- **`*_nocudashare.txt` configs**: same as the normal configs but with `enable_cuda_buffer_sharing: false` (see Problem 3).

### B4. Run it
```bash
./scripts/start_triton_server.sh     # if Triton isn't running yet
./scripts/run_wayland_native.sh      # opens a native Wayland window
```
Stop with Ctrl+C or `docker stop ds-wayland`.

**Result:** 901 frames rendered in a 30-second test, about 28 FPS per tile by the end and 19 FPS on average. Some
frames were dropped, because `sync=true` throws away frames that arrive late, mostly during startup.

---

## Problems on the native route and how they were solved

### Problem 1: `waylandsink` failed with `failed to activate bufferpool`
```
ERROR: from element /GstPipeline:pipeline0/GstVideoConvert:videoconvert0: failed to activate bufferpool
```
**What it means.** `waylandsink` shares frames with the desktop through shared-memory files, which it creates in
`$XDG_RUNTIME_DIR`. I had mounted the socket at `/tmp/xdg/wayland-0`, so Docker **created `/tmp/xdg` as root**.
The container runs as my user, couldn't write there, and the buffer pool failed.

**How I proved it.** I ran the same test-pattern pipeline twice:
- `XDG_RUNTIME_DIR=/tmp/xdg` (root-owned) → `Internal data stream error`
- `XDG_RUNTIME_DIR=/tmp` (writable) → `Execution ended after 0:00:03` ✅

**Fix.** Mount the socket at `/tmp/wayland-0` and set `XDG_RUNTIME_DIR=/tmp`.

### Problem 2: "Permission denied" on the socket (avoided up front)
The socket is `srwxrwxr-x`, so only your user and group can connect. A container running as root, or as any other user,
can't open the window. **Fix:** always run the container with `-u $(id -u):$(id -g)`.

### Problem 3: CUDA buffer sharing with Triton breaks under `gst-launch`
```
ERROR: infer_grpc_client.cpp:224 Failed to register CUDA shared memory.
... failed to open CUDA IPC handle: invalid device context
```
**What it means.** With `enable_cuda_buffer_sharing: true`, DeepStream doesn't send frames over the network.
It hands Triton a pointer into GPU memory, and Triton couldn't open that pointer.

**What I tested:**
- Restarting Triton → still failed, so it wasn't leftover state.
- The same configs in `deepstream-app` and `deepstream-test5-app` → **work fine**, about 70 FPS per stream. So the configs and Triton are OK.
- Sharing off for the SGIE only → the PGIE then failed the same way.
- Sharing off for both → works.

**Fix.** The `*_nocudashare.txt` copies of the PGIE/SGIE configs set `enable_cuda_buffer_sharing: false`, and tensors go
through gRPC instead. It costs some speed but is reliable. The root cause isn't confirmed. My best guess is that
the reference apps (`deepstream-app` / `deepstream-test5-app`) set up the CUDA context before the inference elements start and `gst-launch` doesn't. A custom
C++/Python app could test that.

This problem has nothing to do with Wayland. It's about `gst-launch` vs the reference apps.
It showed up here because Route B can't use `deepstream-test5-app`.

---

## Cheat sheet

```bash
# Route A: XWayland (fast, deepstream-test5-app)
xhost +local:
docker run ... -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix ...   # + [sink] type=2

# Route B: native Wayland (gst-launch + waylandsink)
docker run ... -u $(id -u):$(id -g) \
  -e WAYLAND_DISPLAY=wayland-0 -e XDG_RUNTIME_DIR=/tmp \
  -v $XDG_RUNTIME_DIR/wayland-0:/tmp/wayland-0 ...
# pipeline end:  ... nvdsosd ! nvvideoconvert ! "video/x-raw,format=BGRx" ! waylandsink
```

| Symptom | Cause | Fix |
|---|---|---|
| `failed to activate bufferpool` (waylandsink) | `XDG_RUNTIME_DIR` not writable | `XDG_RUNTIME_DIR=/tmp`, socket at `/tmp/wayland-0` |
| `Could not connect to Wayland display` / permission denied | socket not mounted, or wrong user | mount the socket, `-u $(id -u):$(id -g)` |
| `Failed to register CUDA shared memory … invalid device context` | CUDA IPC under gst-launch | `enable_cuda_buffer_sharing: false` |
| Window works but no labels (Route A) | `deepstream-test5-app` hides text in tiled view | `deepstream-test5-app -t` |
| No window at all (Route A) | X access not allowed | `xhost +local:`, `DISPLAY` set, `/tmp/.X11-unix` mounted |
