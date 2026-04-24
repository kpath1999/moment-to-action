# Model Taxonomy & Benchmarking

## I) Object Perception

| Tier | Model | Notes |
|------|-------|-------|
| Candidate | **SSD-MobileNet-v2** | Lightweight CNN baseline, TFLite/edge-friendly |
| Candidate | **YOLO-v12-n** | Fastest single-stage detector in the set |
| Candidate | **RF-DETR-n** | Transformer-based detector, architecturally distinct from the CNN and YOLO baselines |

**Planned evaluation dataset:** COCO val2017

**Planned metric focus:** object detection accuracy and latency on the same validation split, using COCO bounding boxes.

**Why these three:** they should expose useful tradeoffs in latency, accuracy, and model architecture while staying in a practical edge-deployment envelope.

## II) Semantic Retrieval

| Tier | Model | Notes |
|------|-------|-------|
| Candidate | **TinyCLIP 8M (ViT-B/16)** | Smallest embedding model in the set |
| Candidate | **MobileCLIP-S2** | Fast image-text embedding baseline for on-device retrieval |
| Candidate | **SigLIP (ViT-B/16)** | Stronger image-text alignment baseline with a larger accuracy ceiling |

**Planned evaluation dataset:** COCO val2017

**Planned metric focus:** image-text alignment on COCO captions rather than detection boxes. The image split stays aligned with object detection, but the supervision source changes from bounding boxes to paired captions.

## III) Video Understanding

| Tier | Model | Notes |
|------|-------|-------|
| Candidate | **SmolVLM2-256M** | Smallest video-language model in the set, optimized for edge constraints |
| Candidate | **InternVL-3B** | Mid-scale multimodal model with stronger capacity than the tiny baseline |
| Candidate | **Qwen2.5-VL-3B** | Compact VLM with stronger temporal and instruction-following capability |

## IV) Reasoning Engine

| Tier | Model | Notes |
|------|-------|-------|
| Candidate | **SmolLM2-1.7B** | Smallest reasoning baseline, useful for tight edge budgets |
| Candidate | **Phi4-mini-reasoning** | Compact reasoning-specialized model |
| Candidate | **Qwen3-8B** | Larger reasoning baseline with a higher quality ceiling |

***

## V) Audio Understanding

This remains an important modality if the system needs to capture impacts, speech, or ambient context alongside vision:

| Tier | Model | Notes |
|------|-------|-------|
| Candidate | **Whisper Tiny** | Fast ASR baseline for speech-heavy audio |
| Candidate | **Parakeet TDT** | Strong speech/audio modeling candidate |
| Candidate | **Step-Audio R1.1** | Higher-capacity audio reasoning / understanding candidate |

***

## Benchmark Framing

The current proposal keeps the benchmark intentionally architecturally diverse:

- Detection compares a lightweight CNN detector, a YOLO detector, and a transformer detector.
- Retrieval compares compact CLIP-style embeddings against a stronger alignment-oriented baseline.
- Reasoning, audio, and video each span small-to-larger candidates to expose practical latency versus quality tradeoffs.

The evaluation pipeline still maps cleanly to a **perception → representation → interpretation → decision** stack:

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

The two additional types still worth considering for completeness:

- **Pose / keypoint estimation** (e.g., YOLO-pose edge → ViTPose oracle): directly captures body positions and physical contact geometry, highly relevant for push/fall type interactions.
- **OCR / document understanding**: less relevant for your use case unless you need to read signage or text in video frames.

## Things to improve about benchmarking

1. Define a cleaner hardware sweep: CPU core selection, GPU/NPU routing, and run-to-run isolation.
2. Separate model load time from steady-state inference latency, and make warm-up policy explicit.
3. Clarify why some accelerators underperform on specific models, especially retrieval workloads.
4. Tighten memory reporting so process baseline, model residency, and peak inference memory are distinguishable.

## Pareto-optimal

- Compare combinations of models that occupy different latency budgets.
- Show the spectrum rather than only the single "best" point.
- Plot accuracy on the y-axis and latency on the x-axis.

Pareto-front of different combinations; accuracy will take care of the candidates
