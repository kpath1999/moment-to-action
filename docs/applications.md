# Applications

Six target applications for the moment-to-action research pipeline.
Each is evaluated against a VLM baseline on E2E latency and classification accuracy across compute units (CPU, NPU, GPU).

---

## 1. Violence Detection

- **Input:** video + audio
- **Task:** binary classification — fight / no fight
- **Dataset:** [Real Life Violence Situations](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)
- **Metrics:** E2E latency, accuracy

Violence detection in video is a high-stakes public safety problem with real deployment constraints (surveillance cameras, edge devices). This app tests whether a lightweight pipeline combining visual and audio signals can match VLM accuracy at a fraction of the compute cost. The audio channel (shouting, impact sounds) is expected to improve recall in visually ambiguous scenes.

---

## 2. Fall Detection

- **Input:** video
- **Task:** binary classification — fall / no fall
- **Dataset:** [Fall Video Dataset](https://www.kaggle.com/datasets/payutch/fall-video-dataset)
- **Metrics:** E2E latency, accuracy

Automated fall detection is a critical assistive technology for elderly care and remote monitoring. Low latency is a hard requirement — a multi-second delay before alerting a caregiver reduces the intervention window. This app benchmarks how much accuracy a fast pipeline sacrifices vs. a VLM.

---

## 3. Animal Threat / Attack Detection

- **Input:** video (+ audio TBD)
- **Task:** binary classification — threat / no threat
- **Dataset:** TBD
- **Metrics:** E2E latency, accuracy

Detection of animal attacks (dog bites, wildlife encounters) has direct applications in outdoor safety and livestock monitoring. The challenge is that threat context is often determined by relative motion and proximity, not appearance alone. Dataset and exact label taxonomy are still being finalized.

---

## 4. Infant Monitoring

- **Input:** video
- **Task:** multi-class — safe / distress / intervention needed
- **Dataset:** TBD
- **Metrics:** E2E latency, accuracy

Infant monitoring is a safety-critical application where continuous, low-latency inference matters more than peak accuracy. Detecting distress postures, uncovered states, or roll-over risk in real time is difficult for current VLMs at interactive frame rates. Dataset and label taxonomy are still being finalized.

---

## 5. Eating Detection

- **Input:** video (distance / third-person view)
- **Task:** binary classification — eating / not eating
- **Dataset:** [Ego4D](https://ego4d-data.org/) (egocentric video benchmark)
- **Metrics:** E2E latency, accuracy

Eating detection is relevant for dietary monitoring and behavioral health applications. The egocentric (first-person) viewpoint is the natural capture angle but introduces heavy occlusion from hands and variable lighting; we adopt a third-person "distance view" for this study and treat egocentric as a stretch goal. Ego4D provides a large-scale egocentric dataset for grounding evaluation.

---

## 6. Workplace Safety / PPE Compliance

- **Input:** video
- **Task:** multi-label classification — per-item PPE presence (helmet, vest, gloves, boots) + overall compliance
- **Dataset:** [Construction Site Safety (Roboflow)](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow)
- **Metrics:** E2E latency, per-item accuracy, overall compliance accuracy

PPE compliance monitoring is a high-value industrial safety application. A VLM can reason about compliance in a single pass but is too slow and compute-heavy for continuous site monitoring. This app tests whether a staged pipeline can achieve comparable accuracy while meeting real-time throughput requirements.
