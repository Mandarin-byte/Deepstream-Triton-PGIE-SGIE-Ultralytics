#!/usr/bin/env bash
# Show the DeepStream output in a NATIVE Wayland window (waylandsink), no X11/XWayland involved.
#   PGIE yolo11n -> NvDCF tracker -> SGIE yolo11n_cls  (both on the Triton container over gRPC)
#   -> 2x2 tiler -> OSD -> nvvideoconvert (GPU -> CPU, BGRx) -> waylandsink
# deepstream-app has no Wayland sink type, so this uses a gst-launch-1.0 pipeline instead.
# Usage: ./scripts/run_wayland_native.sh [uri]   (default: DeepStream sample video, shown 4 times)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
URI=${1:-file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4}
WL_SOCK="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/${WAYLAND_DISPLAY:-wayland-0}"
DS=/opt/nvidia/deepstream/deepstream

[ -S "$WL_SOCK" ] || { echo "No Wayland socket at $WL_SOCK - are you in a Wayland session?"; exit 1; }
curl -sf localhost:8000/v2/health/ready >/dev/null || { echo "Triton not ready - run scripts/start_triton_server.sh"; exit 1; }

TTY=""; if [ -t 0 ]; then TTY="-it"; fi
# -u <you>             : the Wayland socket only accepts connections from your user
# XDG_RUNTIME_DIR=/tmp : must be WRITABLE, waylandsink creates its shared-memory buffers there
# *_nocudashare.txt    : CUDA buffer sharing with Triton fails under gst-launch (works in deepstream-app)
docker run --rm $TTY --name ds-wayland --gpus all --net=host --ipc=host \
  -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -e WAYLAND_DISPLAY=wayland-0 -e XDG_RUNTIME_DIR=/tmp -v "$WL_SOCK":/tmp/wayland-0 \
  -v "$ROOT":/app -w /app/configs \
  --entrypoint gst-launch-1.0 nvcr.io/nvidia/deepstream:7.0-triton-multiarch -e \
  nvstreammux name=m batch-size=4 width=1920 height=1080 batched-push-timeout=40000 ! \
  nvinferserver config-file-path=config_pgie_yolo11_triton_nocudashare.txt batch-size=4 ! \
  nvtracker ll-lib-file=$DS/lib/libnvds_nvmultiobjecttracker.so \
            ll-config-file=$DS/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml \
            tracker-width=960 tracker-height=544 ! \
  nvinferserver config-file-path=config_sgie_yolo11_cls_triton_nocudashare.txt ! \
  nvmultistreamtiler rows=2 columns=2 width=1280 height=720 ! \
  nvvideoconvert ! nvdsosd ! \
  nvvideoconvert ! "video/x-raw,format=BGRx" ! \
  waylandsink sync=true \
  uridecodebin uri="$URI" ! m.sink_0 \
  uridecodebin uri="$URI" ! m.sink_1 \
  uridecodebin uri="$URI" ! m.sink_2 \
  uridecodebin uri="$URI" ! m.sink_3
