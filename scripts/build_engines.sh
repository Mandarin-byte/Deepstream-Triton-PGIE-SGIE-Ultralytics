#!/usr/bin/env bash
# Step 2: ONNX -> TensorRT FP16 plans in the Triton model repo.
# Run INSIDE nvcr.io/nvidia/deepstream:7.0-triton-multiarch so the plan's TensorRT version
# matches the one Triton loads. Usage: ./scripts/build_engines.sh [--force]
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRTEXEC=/usr/src/tensorrt/bin/trtexec

# build <onnx> <triton_model> <CxHxW> <opt_batch> <max_batch>
build() {
  local onnx=$ROOT/convert/$1 dir=$ROOT/triton_model_repo/$2 chw=$3 opt=$4 max=$5
  local plan=$dir/1/model.plan
  if [ -f "$plan" ] && [ "$FORCE" != 1 ]; then echo "[skip] $plan"; return; fi
  echo "[build] $2 : $(basename $onnx) fp16 batch 1/$opt/$max (a few minutes)..."
  $TRTEXEC --onnx="$onnx" --saveEngine="$plan" --fp16 \
    --minShapes=input:1x$chw --optShapes=input:${opt}x$chw --maxShapes=input:${max}x$chw \
    > "$dir/build.log" 2>&1 || { echo "FAILED, see $dir/build.log"; rm -f "$plan"; exit 1; }
  rm -f "$dir/build.log"
  echo "[ok] $plan"
}

[ "$1" = "--force" ] && FORCE=1
build yolo11n.onnx     yolo11n     3x640x640 4 4     # max = max_batch_size in config.pbtxt
build yolo11n-cls.onnx yolo11n_cls 3x224x224 8 16
