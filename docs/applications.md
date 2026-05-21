# Applications

Five target applications for the moment-to-action research pipeline.
Each is evaluated against a VLM baseline on E2E latency and classification accuracy across compute units (CPU, NPU, GPU).

Priority is on **egocentric / wearable** use cases where low-latency on-device inference has the clearest deployment story. Ego4D is the primary dataset anchor; other sources supplement where needed.

---

## 1. Violence Detection

- **Input:** video + audio
- **Task:** binary classification — fight / no fight
- **Dataset:** [Real Life Violence Situations](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset); Ego4D (violence-relevant clips); backup — [PoliceActivity YouTube channel](https://www.youtube.com/@PoliceActivity) (redacted body-camera footage, egocentric)
- **Metrics:** E2E latency, accuracy

Violence detection in video is a high-stakes public safety problem with real deployment constraints (body cameras, edge devices). This app tests whether a lightweight pipeline combining visual and audio signals can match VLM accuracy at a fraction of the compute cost. The audio channel (shouting, impact sounds) is expected to improve recall in visually ambiguous scenes. Ego4D likely contains relevant clips; the PoliceActivity channel provides egocentric body-camera footage as a supplementary source if needed.

---

## 2. Fall Detection

- **Input:** video
- **Task:** binary classification — fall / no fall
- **Dataset:** [Fall Video Dataset](https://www.kaggle.com/datasets/payutch/fall-video-dataset)
- **Metrics:** E2E latency, accuracy
- **Note:** Lower priority than egocentric apps. Most existing datasets are fixed-camera (surveillance); value proposition is weaker there. Retain for benchmarking breadth but do not block on it.

Automated fall detection is a critical assistive technology for elderly care and remote monitoring. Low latency is a hard requirement — a multi-second delay before alerting a caregiver reduces the intervention window. This app benchmarks how much accuracy a fast pipeline sacrifices vs. a VLM.

---

## 3. Animal Threat / Attack Detection

- **Input:** video (+ audio TBD)
- **Task:** binary classification — threat / no threat
- **Dataset:** TBD — dataset search in progress
- **Metrics:** E2E latency, accuracy

Detection of animal attacks (dog bites, wildlife encounters) has direct applications in outdoor safety and livestock monitoring. The challenge is that threat context is often determined by relative motion and proximity, not appearance alone. Dataset and exact label taxonomy are still being finalized; this app is contingent on finding a suitable dataset.

---

## 4. Eating Detection

- **Input:** video (egocentric / wearable view)
- **Task:** binary classification — eating / not eating
- **Dataset:** [Ego4D](https://ego4d-data.org/) (egocentric video benchmark)
- **Metrics:** E2E latency, accuracy

Eating detection is relevant for dietary monitoring and behavioral health applications. The egocentric (first-person) viewpoint is the natural capture angle for a wearable device and Ego4D provides a large-scale dataset that fits directly. Occlusion from hands and variable lighting are key challenges the pipeline must handle.

---

## 5. Workplace Safety / PPE Compliance

- **Input:** video
- **Task:** multi-label classification — per-item PPE presence (helmet, vest, gloves, boots) + overall compliance
- **Dataset:** [Construction Site Safety (Roboflow)](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow)
- **Metrics:** E2E latency, per-item accuracy, overall compliance accuracy

PPE compliance monitoring is a high-value industrial safety application. A VLM can reason about compliance in a single pass but is too slow and compute-heavy for continuous site monitoring. This app tests whether a staged pipeline can achieve comparable accuracy while meeting real-time throughput requirements.
