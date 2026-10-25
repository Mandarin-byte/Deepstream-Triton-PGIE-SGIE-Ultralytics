#!/usr/bin/env bash
# Step 5: run the pipeline with deepstream-test5-app in the DS 7.0 container, talking to the Triton container.
#   --net=host : reach Triton gRPC at 127.0.0.1:8001 and expose RTSP on host port 8554
#   --ipc=host : CUDA buffer sharing with the Triton container
#   DISPLAY + /tmp/.X11-unix : on-screen window for the *_display.txt config
#   --net=host also lets the *_redis.txt config reach Redis at localhost:6379 (scripts/start_redis.sh)
# Usage: ./scripts/run_deepstream.sh [config]   (default test5_yolo11_triton.txt = RTSP out)
#        ./scripts/run_deepstream.sh test5_yolo11_triton_display.txt   (window on screen)
#        ./scripts/run_deepstream.sh test5_yolo11_triton_file.txt      (MP4 file)
#        ./scripts/run_deepstream.sh test5_yolo11_triton_redis.txt     (RTSP + JSON messages to Redis)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG=${1:-test5_yolo11_triton.txt}
curl -sf localhost:8000/v2/health/ready >/dev/null || { echo "Triton not ready - run scripts/start_triton_server.sh"; exit 1; }
xhost +local: >/dev/null 2>&1 || true
TTY=""; if [ -t 0 ]; then TTY="-it"; fi
docker run --rm $TTY --name ds-yolo --gpus all --net=host --ipc=host \
  -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -e DISPLAY="${DISPLAY:-:0}" -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$ROOT":/app -w /app/configs \
  nvcr.io/nvidia/deepstream:7.0-triton-multiarch \
  deepstream-test5-app -t -c "$(basename "$CFG")"   # -t: draw labels in tiled (multi-source) view
