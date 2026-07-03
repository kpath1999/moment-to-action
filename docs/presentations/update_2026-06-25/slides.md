---
marp: true
theme: gaia
paginate: true
title: LLM + VLM Scene Classification on QCS6490 — Update 2026-06-25
---

<!-- _class: lead -->

# LLM + VLM Scene Classification on QCS6490

### Update 2026-06-25

---

## Today's update

1. **LLM text-classification** — YOLO detections → LLM → YES / NO, 5 models on GPU
2. **Y/N accuracy + recall** — per-model and per-app breakdown
3. **Latency** — TTFT, ITL, load/infer/unload, TTFYD speedup
4. **Hardware profile** — GPU memory, RSS, CPU utilization
5. **VLM benchmark** — 4 models, raw video frames, no YOLO intermediary
6. **CPU status** — not benchmarked yet

---

## Methodology

**Input:** YOLO detections → structured text prompt (5 apps × 2 scenes × 3 runs = 30 inferences/model)

**Models (GPU, llama-server):** Qwen2-1.5B, Qwen2-7B, Qwen3-4B, Phi-3.5-Mini, Moondream2

**Metrics:**
- **Y/N accuracy** — did model answer YES/NO correctly?
- **Recall** — fraction of expected keywords in response
- **TTFT** — time to first token; **ITL** — inter-token latency
- **TTFYD** — time until YES/NO decision appears in stream

---

## Y/N accuracy

![h:380](plots/yn_accuracy.png)

Qwen3-4B leads (80%). Moondream2 generates no usable YES/NO with structured scene context (0%).

---

## Y/N accuracy per application

![h:400](plots/yn_per_app.png)

PPE excluded (no binary ground truth). Eating detection hardest across models.

---

## Time to first token + inter-token latency

![h:400](plots/ttft_itl.png)

Qwen2-1.5B: 1.6 s TTFT, 206 ms ITL — only model viable for near-real-time. Qwen2-7B: 65 s TTFT.

---

## Latency breakdown

![h:400](plots/latency_breakdown.png)

Inference dominates for large models. Qwen2-1.5B cheapest end-to-end.

---

## Recall by model × application

![h:400](plots/recall_heatmap.png)

PPE easiest (0.5–0.7). Violence and fall hardest. Qwen3-4B and Phi-3.5-Mini tied on recall (0.475).

---

## Decision speed: TTFYD vs full response

![h:400](plots/ttfyd_speedup.png)

Phi-3.5-Mini commits to YES/NO at 4.6 s vs 5.4 s full response. Qwen2-7B: 57 s to decision vs 65 s full.

---

## Hardware profile

![h:400](plots/hw_profile.png)

Moondream2 highest GPU memory (5.9 GB) despite worst accuracy. Phi-3.5-Mini most memory-efficient (192 MB RSS).

---

## Model notes

<style scoped>section { font-size: 22px; }</style>

**Qwen2-1.5B** — Best real-time option: 1.6 s TTFT, 75% Y/N, 2.8 GB GPU.

**Qwen3-4B** — Best accuracy (80% Y/N, 0.475 recall), 37 s TTFT — async/batch only.

**Phi-3.5-Mini** — Uses `Answer: YES` format (detected via regex). 5.4 s TTFT, 75% Y/N.

**Qwen2-7B** — 65 s TTFT, same accuracy as Qwen2-1.5B. Not recommended.

**Moondream2** — 0% Y/N with scene context. Long structured prompts trigger near-immediate EOS. Under investigation.

---

## VLM benchmark: latency

![h:390](plots/vlm_latency.png)

Qwen2.5-VL-3B fastest at ~16 min/run. Qwen3-VL 2B/4B ~30 min — token rate bottleneck, not model size.

---

## VLM benchmark: recall

![h:390](plots/vlm_recall_heatmap.png)

Qwen3-VL-2B leads (0.339). PPE best across all models. Moondream2 generates nothing (0.000 recall).

---

## CPU status

**CPU benchmarking not completed.**

- Tested on Rubik Pi **#2**
- Models loaded in ~10 ms (suspicious) — inference returned 0 tokens
- Root cause: suspected **llama install error** on Rubik Pi #2
- Fix: re-install llama-server on Pi #2, verify with a basic `/completion` curl before re-running

GPU results above from Rubik Pi #1.

---

## Next steps

1. **Moondream2 LLM** — strip system prompt, try short-form prompt; determine if text model works
2. **CPU benchmark** — re-install llama-server on Rubik Pi #2, re-run GPU models on CPU
3. **VLM optimization** — profile token generation bottleneck; try quantized VLM variants
4. **Real video** — re-run VLM bench with real video clips (flag exists in benchmark script)
