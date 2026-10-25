#!/usr/bin/env bash
# Step 1a: YOLO11 detector -> ONNX using DeepStream-Yolo's utils/export_yolo11.py (unmodified).
# torch>=2.9 defaults torch.onnx.export to the dynamo exporter, which produces a broken graph
# for this model (and ignores --opset), so we force the classic TorchScript exporter.
set -e
cd "$(dirname "$0")"
WEIGHTS=${1:-yolo11n.pt}
# export_yolo11.py needs the .pt on disk; let Ultralytics download official weights first
[ -f "$WEIGHTS" ] || python3 -c "from ultralytics import YOLO; YOLO('$WEIGHTS')"
python3 - "$WEIGHTS" <<'PY'
import functools, runpy, sys, torch
torch.onnx.export = functools.partial(torch.onnx.export, dynamo=False)
sys.argv = ["export_yolo11.py", "-w", sys.argv[1], "--dynamic", "--simplify", "--opset", "17"]
runpy.run_path("export_yolo11.py", run_name="__main__")
PY
