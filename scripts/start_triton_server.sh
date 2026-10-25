#!/usr/bin/env bash
# Step 4: run Triton Inference Server (from the DS 7.0 image) serving ./triton_model_repo.
#   8000 = HTTP, 8001 = gRPC (used by DeepStream nvinferserver), 8002 = Prometheus metrics
# --ipc=host is required for enable_cuda_buffer_sharing between the DeepStream and Triton containers.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME=triton-yolo
docker rm -f $NAME >/dev/null 2>&1 || true
docker run -d --name $NAME --gpus all --ipc=host --shm-size=2g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$ROOT/triton_model_repo":/models \
  --entrypoint /opt/tritonserver/bin/tritonserver \
  nvcr.io/nvidia/deepstream:7.0-triton-multiarch \
  --model-repository=/models --disable-auto-complete-config --log-verbose=0
echo "Waiting for Triton to be ready..."
for i in $(seq 60); do
  curl -sf localhost:8000/v2/health/ready >/dev/null && { echo "Triton READY"; break; }
  sleep 2
done
curl -s -X POST localhost:8000/v2/repository/index; echo
