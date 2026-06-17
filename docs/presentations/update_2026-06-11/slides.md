---
marp: true
theme: gaia
paginate: true
title: Edge Detection on QCS6490 — Benchmark
---

<!-- _class: lead -->

# Edge Detection on QCS6490

### Model × backend benchmark

YOLOv8 · RF-DETR · RTMDet · Detectron2 (w8a8 / w8a16)
across **CPU · GPU · NPU**

2026-06-11

---

## Setup

- **Device:** Qualcomm QCS6490, Hexagon **v68** HTP, Adreno A642L GPU
- **Models:** 5 configs — `yolo_v8`, `rf_detr`, `rtm_det`, `detectron2_w8a8`, `detectron2_w8a16`
- **Backends:** CPU, GPU, NPU (QAIRT / QNN)
- **Artifacts:** Qualcomm AI Hub — quantized **DLC** + AOT **context binaries**
- **Workload:** 50 COCO val2017 images × **3 load/infer/unload cycles**
- **Metrics:** per-stage latency (load / preproc / infer / post / unload) + AP50 proxy, on-device

---

## What is AP50?

<style scoped>section { font-size: 23px; }</style>

- **AP** = Average Precision = area under the precision–recall curve, detections ranked by confidence (rewards real objects, punishes false positives).
- **AP50** = AP counting a box correct when **IoU with a ground-truth box ≥ 0.5**.
- Real **COCO mAP** averages AP over IoU 0.5→0.95 **and per class** — stricter.

**Our number is a proxy** (`_ap50`): greedy match preds→GT at IoU ≥ 0.5, **class-agnostic** (labels ignored), interpolated P–R area, averaged over images → a **relative trend signal**, not validation mAP.

---

## The models

<style scoped>section { font-size: 22px; } table { font-size: 20px; }</style>

| Model | Family | Type | Input | Quant |
|---|---|---|---|---|
| **YOLOv8** (n) | YOLO | 1-stage anchor-free CNN | 640² | w8a8 |
| **RF-DETR** | DETR | 1-stage transformer | 560² | float |
| **RTMDet** | RTM | 1-stage CNN | 640² | float |
| **Detectron2** | Faster R-CNN R50-C4 | **2-stage** (RPN + ROI head) | 800² | w8a8 / w8a16 |

- All COCO 80-class; exported from **Qualcomm AI Hub** (DLC + context binaries)
- *1-stage*: single forward pass
- *Detectron2*: backbone → proposals → per-RoI head (heavier)

---

## What runs where

![h:360](plots/support_matrix.png)

YOLOv8 = all three · RF-DETR / RTMDet = **CPU only** · Detectron2 = **CPU + NPU**

---

## Inference latency (the headline)

![h:380](plots/infer_latency.png)

NPU: **YOLOv8 15 ms** · Detectron2 **356 / 582 ms** (w8a8 / w8a16). CPU 30–50× slower.

---

## NPU speedup vs CPU

![h:360](plots/npu_speedup.png)

**YOLOv8 35× · Detectron2 w8a8 52× · w8a16 32×.** YOLOv8 NPU is 7.4× faster than its own GPU.

---

## Latency breakdown (per cycle)

![h:360](plots/latency_breakdown.png)

Detectron2 = **two-stage** (2 NPU graphs + CPU proposal-NMS on critical path); NPU load ≈ 0.9–1.3 s.

---

## Throughput

![h:380](plots/throughput_fps.png)

**YOLOv8 NPU ≈ 67 FPS** (real-time) · **Detectron2 NPU ≈ 2–3 FPS** (accuracy-first).

---

## Accuracy (approximate)

![h:340](plots/accuracy_ap50.png)

**Caveat:** crude class-agnostic IoU proxy (labels ignored), **not** COCO mAP. Note YOLOv8-GPU collapse.

---

## What works / what doesn't

<style scoped>section { font-size: 22px; } table { font-size: 21px; }</style>

| Model | CPU | GPU | NPU |
|---|---|---|---|
| YOLOv8 | ✅ | ⚠️ runs, accuracy collapses | ✅ 15 ms |
| RF-DETR | ✅ | ❌ `Tile` | ❌ float-only |
| RTMDet | ✅ | ❌ `Cast` | ❌ float decode head |
| Detectron2 w8a8/w8a16 | ✅ | ❌ `Exp` | ✅ |

**GPU:** QNN GPU OpPackage rejects a different op per model (`Tile` / `Cast` / `Exp`) → won't compose.
**NPU blocks:** RF-DETR float-only; RTMDet float decode head (`argmax→float`) not quantizable → v68 rejects float I/O.

---

## Why Detectron2 *can* use the NPU

- Exposes **full-integer** precisions (`w8a8`, `w8a16`) → **integer graph I/O** → passes the
  v68 context-binary linker.
- RF-DETR (float-only) and RTMDet (float decode head) **cannot** — same wall blocks both.
- On NPU, **w8a8 is faster** than w8a16 (356 vs 582 ms) — prefer w8a8 when latency matters.

---

## Limitations

<style scoped>section { font-size: 23px; }</style>

- **AP50 is a crude proxy:** class-agnostic, not validation mAP.
- **GPU very limited** on this SoC: per-op OpPackage gaps, YOLOv8 composes but accuracy collapses.
- **RF-DETR / RTMDet are CPU-only:** no NPU path on v68 (float decode / float-only).
- **Detectron2 is heavy:** CPU ≈ 18 s/frame; on NPU the CPU proposal-NMS sits between the two graphs.
