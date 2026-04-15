# Model Taxonomy & Benchmarking

## I) Object Perception

| Tier | Model | Notes |
|------|-------|-------|
| Edge | **YOLOv8n** | Fastest, ultra-lightweight |
| Edge | **YOLOv8s / YOLOv8m** | Step up in accuracy, still edge-viable |
| Edge | **RF-DETR Nano** | Roboflow's transformer-based nano detector  [digitalocean](https://www.digitalocean.com/community/tutorials/best-object-detection-models-guide) |
| Edge | **MobileNet-SSD** | Classic depthwise-conv detection, TFLite-friendly  [reddit](https://www.reddit.com/r/computervision/comments/1qha8bm/good_detection_models_for_edge_deployment_in_2026/) |
| Edge | **YOLOX-Nano** | Open-source YOLO variant, strong mobile perf  [reddit](https://www.reddit.com/r/computervision/comments/1qha8bm/good_detection_models_for_edge_deployment_in_2026/) |
| Oracle | **Grounding DINO** ✓ | Open-vocabulary, strong zero-shot |
| Oracle+ | **DINO-X** | Stronger open-world extension of GDINO  [arxiv](https://arxiv.org/html/2411.14347v1) |

## II) Semantic Retrieval

| Tier | Model | Notes |
|------|-------|-------|
| Edge | **MobileCLIPs2** ✓ | Fast image-text embedding |
| Edge | **TinyCLIP** | Distilled CLIP, further reduced params |
| Edge | **e5-small** | Best latency/accuracy tradeoff for retrieval  [aimultiple](https://aimultiple.com/open-source-embedding-models) |
| Edge | **EmbeddingGemma-300M** | Google DeepMind, strong MTEB perf at 300M params  [bentoml](https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models) |
| Oracle | **SigLIP 2** ✓ | Multilingual, strong zero-shot VL |
| Oracle+ | **OpenCLIP ViT-H/G** | Largest public CLIP variants for strongest baseline |

## III) Video Understanding

| Tier | Model | Notes |
|------|-------|-------|
| Edge | **SmolVLM2** ✓ | Multi-frame, device-friendly |
| Edge | **Moondream2** | Very small (~1.8B), image+light video QA  [blog.roboflow](https://blog.roboflow.com/local-vision-language-models/) |
| Edge | **Qwen2.5-VL-3B** | Handles video >1hr even at small size  [sourceforge](https://sourceforge.net/software/product/SmolVLM/alternatives) |
| Edge | **LLaVA-1.5-7B** (frame-wise) | Strong VQA applied per-frame |
| Oracle | **Qwen2.5-VL-72B** | Open-weight, strong temporal reasoning  [sourceforge](https://sourceforge.net/software/product/SmolVLM/alternatives) |
| Oracle | **GPT-4V / GPT-4o** | Good choice but closed; prefer open if fully offline  [aimultiple](https://aimultiple.com/large-vision-models) |

> For a truly offline oracle, **Qwen2.5-VL-72B** or **InternVL2-76B** are better choices than GPT-4V since they can be self-hosted.

## IV) Reasoning Engine

| Tier | Model | Notes |
|------|-------|-------|
| Edge | **Qwen3.5-4B** ✓ | Reasoning + planning |
| Edge | **Gemma 3-4B** | Competitive on shared benchmarks vs Qwen3.5-4B  [maniac](https://www.maniac.ai/blog/qwen-3-5-vs-gemma-4-benchmarks-by-size) |
| Edge | **Phi-4-mini** | Microsoft, strong at structured output / chain-of-thought |
| Edge | **SmolLM2-1.7B** | HuggingFace's smallest capable LLM for planning tasks  [arxiv](https://arxiv.org/html/2502.02737v1) |
| Oracle | **Qwen3.5-72B** ✓ | Same family, dramatically higher reasoning ceiling |
| Oracle+ | **Qwen3-Max-Thinking** | Extended CoT, 80K thinking tokens  [slashdot](https://slashdot.org/software/p/Qwen3.5/alternatives) |

***

## V) Missing: Audio Understanding 🔊

This is a real gap if your use case involves detecting sounds (impacts, screaming, ambient aggression cues):

| Tier | Model | Notes |
|------|-------|-------|
| Edge | **Whisper-tiny / small** | Fast ASR, good for speech signals |
| Edge | **wav2vec2-base** | Lightweight speech representation |
| Edge | **YAMNet** | Google's audio event classifier, mobile-first |
| Oracle | **Whisper-large-v3** | SoTA ASR, much higher accuracy |
| Oracle | **WavLM-large** | Strong general audio understanding, SoTA on many audio tasks |

***

## How to think about model types broadly

Your current stack maps well to a **perception → representation → interpretation → decision** pipeline:

```
Audio / visual sensors
       ↓
Object Perception      ← "What objects are present, where?"
       ↓
Semantic Retrieval     ← "What do these look/sound like conceptually?"
       ↓
Video Understanding    ← "What actions/sequences are happening?"
       ↓
Audio Understanding    ← "What does the soundscape confirm?"  ← missing
       ↓
Reasoning Engine       ← "What is the probability, what should I do next?"
```

The two additional types worth considering for completeness:

- **Pose / keypoint estimation** (e.g., YOLOv8-Pose edge → ViTPose oracle): directly captures body positions and physical contact geometry, highly relevant for push/fall type interactions.
- **OCR / document understanding**: less relevant for your use case unless you need to read signage or text in video frames.

## Things to improve about benchmarking

1) Conducting a clean sweep (how many CPUs and which ones/cores)
2) 218 ms for the Adreno GPU is really high. Inference latency does not include model loading
* Before you runm warm-up (first few runs are garbage)
3) Why is NPU so high for MobileCLIP?? The GPU numbers make more sense now though
4) Why is the memory so high (MB)?
* Process RSS, PyTorch (baseline at start; when it's loaded)
* Memory usage at different times, YOLO is pretty light with a few KB images

## Pareto-optimal
* Combination of models taking different amounts of latency
* Show me the spectrum
* Accuracy graph (y-axis), latency (x-axis)

Pareto-front of different combinations; accuracy will take care of the candidates
