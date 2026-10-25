#!/usr/bin/env python3
"""Draws docs/pipeline_flowchart.png: build phase (offline) + run phase (DeepStream <-> Triton)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HOST, DSC, TRT, SINK, NOTE = "#dbeafe", "#dcfce7", "#ffedd5", "#ede9fe", "#fef9c3"
EDGE = {"#dbeafe": "#2563eb", "#dcfce7": "#16a34a", "#ffedd5": "#ea580c", "#ede9fe": "#7c3aed",
        "#fef9c3": "#ca8a04", "white": "#475569"}

fig, ax = plt.subplots(figsize=(14, 21))
ax.set_xlim(0, 100)
ax.set_ylim(0, 150)
ax.axis("off")


def group(x0, y0, x1, y1, color, label):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=color, ec=EDGE[color], lw=2, alpha=0.45, zorder=0))
    ax.text(x0 + 1.2, y1 - 1.2, label, ha="left", va="top", fontsize=11, fontweight="bold",
            color=EDGE[color], zorder=1)


def box(cx, cy, w, h, text, color="white", fs=9.5, bold_first=True):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=color, ec=EDGE.get(color, "#475569"), lw=1.6, zorder=2))
    lines = text.split("\n")
    if bold_first and len(lines) > 1:
        ax.text(cx, cy + h / 2 - 1.0, lines[0], ha="center", va="top", fontsize=fs + 0.5,
                fontweight="bold", zorder=3)
        ax.text(cx, cy - 0.6, "\n".join(lines[1:]), ha="center", va="center", fontsize=fs - 0.5,
                zorder=3, linespacing=1.35)
    else:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold_first else None,
                zorder=3, linespacing=1.35)


def arrow(x0, y0, x1, y1, label=None, color="#334155", lpos=0.5, loff=(0, 0.9), fs=8.5):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color=color, mutation_scale=16), zorder=4)
    if label:
        ax.text(x0 + (x1 - x0) * lpos + loff[0], y0 + (y1 - y0) * lpos + loff[1], label, ha="center",
                va="center", fontsize=fs, color=color, zorder=5,
                bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.9))


# ---------------------------------------------------------------- title
ax.text(50, 148.2, "DeepStream 7.0 + Triton  ·  YOLO11 detector (PGIE) + YOLO11-cls classifier (SGIE)",
        ha="center", fontsize=15, fontweight="bold")
ax.text(50, 145.6, "Conversion + parser: marcoslucianops/DeepStream-Yolo     ·     Triton-over-gRPC + RTSP out: "
        "levipereira/deepstream-yolo-triton-server-rtsp-out", ha="center", fontsize=9.5, color="#475569")

# ================================================================ PHASE 1: BUILD
ax.text(3, 142.5, "PHASE 1 — BUILD  (offline, once)", fontsize=13, fontweight="bold", color="#0f172a")

group(3, 111, 97, 140.5, HOST, "HOST  ·  Python + Ultralytics 8.4  (steps 1a / 1b)")
box(25, 133.5, 30, 3.6, "yolo11n.pt  (detector, 80 COCO classes)", bold_first=True, fs=9.5)
box(75, 133.5, 30, 3.6, "yolo11n-cls.pt  (classifier, 1000 ImageNet)", bold_first=True, fs=9.5)
box(25, 125.5, 34, 7, "convert/export_det.sh\nruns DeepStream-Yolo export_yolo11.py\nforces classic ONNX exporter, opset 17",
    HOST)
box(75, 125.5, 34, 7, "convert/export_cls.py\nsame style: input / output names\ndynamic batch, opset 17", HOST)
box(25, 116, 34, 6.5, "yolo11n.onnx  + labels.txt\nin  [B,3,640,640]\nout [B,8400,6] = x1,y1,x2,y2,score,class",
    "white")
box(75, 116, 34, 6.5, "yolo11n-cls.onnx  + labels_cls.txt\nin  [B,3,224,224]\nout [B,1000] softmax", "white")
arrow(25, 131.7, 25, 129.0)
arrow(75, 131.7, 75, 129.0)
arrow(25, 122.0, 25, 119.25)
arrow(75, 122.0, 75, 119.25)

group(3, 86, 97, 108.5, DSC, "DS 7.0 CONTAINER  (steps 2 / 3)")
ax.text(95.8, 87.0, "TensorRT 8.6  ·  CUDA 12.2  ·  same versions Triton uses", ha="right", va="bottom",
        fontsize=8.5, color="#15803d", style="italic")
box(35, 101, 46, 4, "scripts/build_engines.sh  →  trtexec --fp16", DSC, fs=9.5)
box(81, 101, 30, 4.6, "DeepStream-Yolo source\nnvdsinfer_custom_impl_Yolo/  (C++ / CUDA)", "white", fs=8.5)
box(19, 92, 28, 7, "triton_model_repo/yolo11n\n1/model.plan + config.pbtxt\nbatch ≤ 4 · dynamic batching", "white")
box(50, 92, 28, 7, "triton_model_repo/yolo11n_cls\n1/model.plan + config.pbtxt\nbatch ≤ 16 · dynamic batching", "white")
box(81, 92, 30, 7, "make (CUDA_VER=12.2)\nlibnvdsinfer_custom_impl_Yolo.so\nNvDsInferParseYolo", "white")
arrow(36, 112.75, 36, 103.0, "ONNX", lpos=0.45)
arrow(75, 112.75, 50, 103.0, "ONNX", lpos=0.45)
arrow(22, 99.0, 19, 95.5)
arrow(48, 99.0, 50, 95.5)
arrow(81, 99.0, 81, 95.5, "make")

# ================================================================ PHASE 2: RUN
ax.text(3, 83.2, "PHASE 2 — RUN  (two containers talking over gRPC)", fontsize=13, fontweight="bold",
        color="#0f172a")

group(3, 1.5, 62, 81, DSC, "CONTAINER  ds-yolo   ·   deepstream-test5-app -t -c <config>")
group(67, 27, 97, 72, TRT, "CONTAINER  triton-yolo\ntritonserver  :8000 / :8001 / :8002")

cx, w = 32.5, 50
chain = [
    (75.5, 4.2, "4 × video file   /   RTSP camera", "white"),
    (68.5, 4.2, "nvurisrcbin  —  GPU decode (NVDEC)", "white"),
    (61.5, 4.2, "nvstreammux  —  batch 4 frames @ 1920×1080", "white"),
    (52.0, 8.0, "PGIE  ·  nvinferserver  →  yolo11n\nletterbox 640 (centered), RGB, ÷255\n"
                "NvDsInferParseYolo → NMS  (IoU 0.45, conf 0.25)", DSC),
    (42.5, 4.2, "NvDCF tracker  —  gives each object a track ID", "white"),
    (33.0, 8.0, "SGIE  ·  nvinferserver  →  yolo11n_cls\ncrops of car / motorcycle / bus / truck  (≥ 64 px)\n"
                "async, re-classify every 30 frames", DSC),
    (23.5, 4.2, "nvmultistreamtiler  —  2×2 grid, 1280×720", "white"),
    (16.0, 5.6, "nvdsosd  —  draws box + text\n“car 381 racer”  =  class · track ID · classifier", "white"),
]
for cy, h, text, color in chain:
    box(cx, cy, w, h, text, color, fs=9.5, bold_first=("\n" in text))
for (y0, h0, *_), (y1, h1, *_) in zip(chain, chain[1:]):
    arrow(cx, y0 - h0 / 2, cx, y1 + h1 / 2)

sinks = [(10.5, "RTSP out\n:8554/ds-test\ntest5_…triton.txt"),
         (25.2, "EGL window\nyour screen\n…_display.txt"),
         (39.9, "MP4 file\noutput/out.mp4\n…_file.txt"),
         (54.6, "Redis (JSON)\nmessage broker\n…_redis.txt")]
for sx, text in sinks:
    box(sx, 6.5, 13.6, 7, text, SINK, fs=8.2)
    arrow(cx, 16.0 - 2.8, sx, 10.0)

box(82, 52, 24, 8, "yolo11n\nTensorRT FP16 plan\nbatch ≤ 4", TRT)
box(82, 33, 24, 8, "yolo11n_cls\nTensorRT FP16 plan\nbatch ≤ 16", TRT)
arrow(57.5, 53.6, 70, 53.6, "gRPC :8001  frames", color="#c2410c", loff=(0, 1.3))
arrow(70, 50.4, 57.5, 50.4, "[8400,6] tensor", color="#15803d", loff=(0, -1.3))
arrow(57.5, 34.6, 70, 34.6, "gRPC :8001  crops", color="#c2410c", loff=(0, 1.3))
arrow(70, 31.4, 57.5, 31.4, "[1000] softmax", color="#15803d", loff=(0, -1.3))
ax.text(82, 42.5, "dynamic batching merges\nrequests into GPU batches", ha="center", va="center",
        fontsize=8.5, color="#9a3412", style="italic")

box(82, 21, 30, 6, "--ipc=host on both containers\n→ CUDA buffer sharing\n(frames are not copied over the network)",
    NOTE, fs=8.5)
box(82, 8.5, 30, 12, "Fixes along the way\n① torch 2.10 dynamo export broke the ONNX\n    → classic exporter (dynamo=False)\n"
    "② no labels in tiled 4-source view\n    → -t  flag (tiledtext)\n"
    "③ run script died without a TTY\n    → fixed check under set -e", NOTE, fs=8.5)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_flowchart.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print("saved", out)
