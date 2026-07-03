---
marp: true
theme: gaia
paginate: true
title: Edge Detection + LLM Reasoning on QCS6490 — Update 2026-07-02
---

<!-- _class: lead -->

# Edge Detection + LLM Reasoning on QCS6490

### Update 2026-07-02

---

## Today's update

1. **Real-data benchmark** — actual annotated video clips, not synthetic frames/boxes
2. **LLM pipeline** — 1-FPS frames → **Detectron2 NPU** detection → text prompt → 7 edge LLMs
3. **VLM pipeline** — 1-FPS frames straight into 5 edge VLMs, no detector
4. **Small Qwen3 models rarely answer** — reasoning budget runs out inside `<think>` before a verdict

---

## Setup

<style scoped>section { font-size: 20px; }</style>

**Data:** 12 real video clips, 3 apps (violence, eating, animal threat), 4 clips/app, balanced positive/negative, each clip trimmed to its annotated ROI window (start_s/end_s).

**Frame extraction:** 1-FPS sampling from the ROI window (4–17 frames/clip depending on length), resized to ≤480p to fit the CPU image tower constraint.

**LLM pipeline:** frames → **Detectron2 (NPU, w8a16)** detection per frame → aggregate detections across frames (highest-confidence instance per label, frame count per label) → spatial-context text prompt (bbox-derived zone/depth) → LLM (llama-server, CPU or GPU) → streamed YES/NO + reasoning.

**VLM pipeline:** frames base64-JPEG-encoded → passed directly to the VLM, no detector in the loop.

**Workload:** 3 apps × 4 clips × 3 cycles = 36 inferences/model, full load/infer/unload timing captured via MetricsCollector spans.

**Metrics:** keyword recall (expected keywords present in response) + Y/N accuracy where an answer is detectable in the text.

---

<!-- _class: lead -->

# LLM benchmark

Detectron2 (NPU) detections → text prompt → 7 edge LLMs

---

## LLM: answer rate

<style scoped>section { font-size: 21px; }</style>

![h:330](plots/answer_rate.png)

`--max-tokens 128` isn't enough for reasoning models: **Qwen3-0.6B/1.7B/4B** spend the budget inside `<think>...</think>` and cut off before answering. **Gemma3-270M** never produces a parseable YES/NO at all — echoes the prompt back instead. (Details: backup slide.)

---

## LLM: accuracy — answered-only vs. overall

<style scoped>section { font-size: 21px; }</style>

![h:330](plots/yn_accuracy_overall.png)

**Phi-3.5-Mini** best either way (64%, 100% answer rate). **Qwen3-0.6B/4B** look strong on answered-only (100%/27%, tiny n) but collapse to 8% overall once non-answers count as wrong — answer rate decides the real-world number, not conditioned accuracy.

---

## LLM: accuracy by app

<style scoped>section { font-size: 20px; }</style>

![h:330](plots/yn_accuracy_by_app.png)

**Eating** is the strongest app for every model with enough samples (up to 100% at n=12, Phi-3.5-Mini 92%). **Animal threat** weakest across the board (0–50%). Cells with n≤3 (mostly Qwen3 family) are single-digit-sample noise, not a real signal — read those as "unknown," not "good."

---

## LLM: latency breakdown

<style scoped>section { font-size: 21px; }</style>

![h:330](plots/latency_breakdown.png)

**Qwen2-1.5B** cheapest end-to-end (41s: mostly GPU load). **Phi-3.5-Mini** heaviest (140s) but only model that reliably finishes reasoning. **Gemma3-270M** load-dominated (89s) despite being tiny — GPU load cost, unrelated to size. (Detectron2/NPU detection runs separately, not counted in this load figure.)

---

## LLM: streaming cuts decision latency 4.6–7.3×

<style scoped>section { font-size: 20px; }</style>

![h:330](plots/ttfyd_speedup.png)

Streaming lets the pipeline stop reading the moment YES/NO commits, instead of waiting for the full response. **Phi-3.5-Mini**: 97s → 13.3s (7.3×). **Gemma3-1B**: 9.8s → 1.7s (5.9×). **Qwen2-1.5B**: 3.6s → 772ms (4.6×). Qwen3 family and Gemma3-270M excluded — the real-time detector doesn't strip `<think>`, so it never fires even on runs that eventually answer.

---

## LLM: recall by app

<style scoped>section { font-size: 21px; }</style>

![h:330](plots/recall_heatmap.png)

**Eating** far easier than violence/animal-threat for every model (0.27–0.37 vs 0.06–0.20) — Detectron2's COCO classes (food-adjacent objects, person) map cleanly onto the eating prompt; violence/animal-threat need behavior inference the detector can't provide.

---

<!-- _class: lead -->

# VLM benchmark

1-FPS frames → VLM directly, no detector

---

## VLM: response quality

<style scoped>section { font-size: 21px; }</style>

![h:330](plots/vlm_quality.png)

**Moondream2 completely broken** on real multi-frame clips — 100% empty responses (worked fine on synthetic single-frame input previously). **SmolVLM2-500M** only model with zero empty responses.

---

## VLM: accuracy — answered-only vs. overall

<style scoped>section { font-size: 21px; }</style>

![h:330](plots/vlm_yn_accuracy_overall.png)

**SmolVLM2-500M** best overall (44%) — highest answer rate carries it despite a lower conditioned accuracy than Qwen2.5-VL-3B (57%, but only n=7). **Moondream2, InternVL3-1B, SmolVLM2-256M** all at exactly 0% overall — either never answer or guess wrong on every answer they do give.

---

## VLM: accuracy by app

<style scoped>section { font-size: 20px; }</style>

![h:330](plots/vlm_yn_accuracy_by_app.png)

**InternVL3-1B** wrong on every answered app cell — 0% everywhere it answers. Most cells sit at n≤3; only **SmolVLM2-500M**'s animal-threat cell (n=12) is a large enough sample to trust. No app is reliably solved by any VLM here.

---

## VLM: streaming cuts decision latency 3.4–11×

<style scoped>section { font-size: 20px; }</style>

![h:330](plots/vlm_ttfyd_speedup.png)

Same early-exit benefit as the LLM path. **SmolVLM2-500M**: 21.2s → 1.9s (11×, n=25 — the trustworthy one). **InternVL3-1B**: 16s → 1.9s (8.4×, n=6). **Qwen2.5-VL-3B** and **SmolVLM2-256M** speedups are real but built on n=7 and n=2 — directionally right, not precise. Moondream2 excluded (never answers).

---

## VLM: latency

<style scoped>section { font-size: 21px; }</style>

![h:330](plots/vlm_latency.png)

**SmolVLM2-256M** cheapest (18s) but barely answers (6%, see quality chart). **Qwen2.5-VL-3B** slowest (70s/clip) for a 19% answer rate — worst latency/accuracy trade of the group.

---

## VLM: recall by app

<style scoped>section { font-size: 21px; }</style>

![h:330](plots/vlm_recall_heatmap.png)

Same pattern as the LLM path — eating is easiest, violence/animal-threat hardest, independent of pipeline architecture. Recall is uniformly low (≤0.29) — real video is materially harder than the earlier synthetic benchmark.

---

<!-- _class: lead -->

# Ego4D dataset

896 GB of egocentric video, ready for the next benchmark round

---

## Ego4D: what's available locally

<style scoped>section { font-size: 20px; }</style>

**Location:** `~/cedarp-fromero7-0/ego4d/v2/` (896 GB total)

**Raw clips** (`v2/clips/`, 15,134 files, `.mp4`) — pre-segmented, de-identified first-person camera footage, each named by a `clip_uid`. This is what the pipeline's sensor stage would ingest.

**Annotations** (`v2/annotations/`, 34 JSON/CSV files, one per benchmark task, each split `_train`/`_val`/`_test_unannotated`):

- **`moments_{train,val}.json`** (948 / 329 videos) — most directly useful: per-video list of clips with human-labeled action category, start/end window, primary flag. Ground-truth "moment → action label" pairs — exactly what the pipeline is trying to produce.
- **`narration.json`** (9,611 videos, ~1–2 GB, keyed directly by `video_uid`) — dense timestamped free-text narration ("C picks up a mug"). Finer-grained than moments; candidate weak supervision or LLM/VLM text source.
- **`fho_main.json`** (1,725 videos, 2 GB+, Forecasting Hands & Objects) — hand-object interaction, state-change (PNR) frames, long-term action anticipation. More specialized, not needed for a first pass.
- **`nlq_train.json`** (933 videos) / **`av_train.json`** (153 videos) / `vq_*` / `goalstep_*` — natural-language query localization, visual query localization, audio-visual diarization, hierarchical goal/step annotations.
- **Taxonomy files + `manifest.csv`** — controlled vocabularies and S3 source mapping.

*(Video-level counts — each video holds multiple clips/annotations, so clip/label counts run higher.)*

---

## Ego4D: structure + recommended path

<style scoped>section { font-size: 21px; }</style>

**Structure:** everything keyed by `video_uid` → `clip_uid`; one source video can split into multiple clips, each with its own start/end offsets (`video_start_sec/frame` vs. `clip_start_sec/frame`). Multiple annotators can label the same clip independently (redundancy in `moments`). Pre-split into train/val/test — test labels withheld (official held-out set).

**Recommended starting point:** `moments_val.json` (small, 1.4 MB) paired with the matching clips in `v2/clips/` — ready-made ground-truth video-segment → action-label pairs to run the pipeline against and score directly, same shape as the current `annotations.json` benchmark harness. `narration.json` is a good secondary source once the pipeline leans harder on the LLM reasoning stage — far denser than the moment labels.

`moments_val.json` alone covers **329 videos** — enough for a solid first evaluation set without touching the full 896 GB corpus.

---

## Summary

<style scoped>section { font-size: 21px; } table { font-size: 19px; }</style>

| Use case | Best pick | Key metric |
|---|---|---|
| LLM (best accuracy) | Phi-3.5-Mini | 64% acc, 100% answer rate, 140s |
| LLM (fastest, reliable) | Qwen2-1.5B | 100% answer rate, 41s, ~chance accuracy |
| LLM (avoid as-is) | Gemma3-270M, Qwen3-0.6B/1.7B | 0–8% answer rate at 128 tokens |
| VLM (most reliable) | SmolVLM2-500M | 0% empty, 69% answer rate, 37s |
| VLM (avoid as-is) | Moondream2 | 100% empty on real multi-frame video |

**Root cause to fix next:** 128-token budget starves reasoning models before they emit an answer — either raise `max_tokens` for Qwen3 or switch to a non-thinking chat template.

**Next:** re-run Qwen3 family with a larger token budget; debug Moondream2 empty-response regression on multi-frame input.

---

## Backup: confirming the 128-token truncation

<style scoped>section { font-size: 22px; } table { font-size: 20px; }</style>

llama.cpp returns `predicted_n` (tokens actually generated) per response. If a model is hitting the `max_tokens` cap rather than stopping naturally, `predicted_n` should equal 128 on (almost) every run.

| Model | Runs hitting predicted_n=128 | predicted_n range |
|---|---|---|
| Qwen3-0.6B | 36/36 (100%) | 128–128 |
| Qwen3-1.7B | 36/36 (100%) | 128–128 |
| Qwen3-4B | 34/36 (94%) | 115–128 |
| Phi-3.5-Mini | 2/36 (6%) | 25–128 |

Qwen3-0.6B and Qwen3-1.7B hit the cap on **every single run** — never once stopped on their own. Phi-3.5-Mini (non-thinking template) hits it on 2/36 — most responses finish naturally, well under budget. This confirms the answer-rate gap is a token-budget artifact, not a capability gap.
