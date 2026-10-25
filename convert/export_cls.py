#!/usr/bin/env python3
"""Step 1b: YOLO11-cls -> ONNX in the same style as DeepStream-Yolo (input/output names, dynamic batch).
Output = softmax probabilities [B, num_classes]. Also writes labels_cls.txt (one line, ';' separated,
which is the classifier label format nvinfer/nvinferserver expect)."""
import argparse
from copy import deepcopy

import onnx
import onnxslim
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import Classify

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", default="yolo11n-cls.pt")
ap.add_argument("-s", "--size", type=int, default=224)
a = ap.parse_args()

yolo = YOLO(a.weights)
model = deepcopy(yolo.model).float().eval().fuse()
for m in model.modules():
    if isinstance(m, Classify):
        m.export = True  # return softmax only

with open("labels_cls.txt", "w", encoding="utf-8") as f:
    f.write(";".join(yolo.names[i].replace(";", ",") for i in range(len(yolo.names))) + "\n")

out = a.weights.rsplit(".", 1)[0] + ".onnx"
torch.onnx.export(model, torch.zeros(1, 3, a.size, a.size), out, dynamo=False, opset_version=17,
                  do_constant_folding=True, input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
onnx.save(onnxslim.slim(onnx.load(out)), out)
print("Done:", out)
