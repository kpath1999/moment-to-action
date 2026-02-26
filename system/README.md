A multimodal violence detection system. Given a video clip, it extracts audio and visual signals, fuses them into a timeline, queries an LLM to identify moments of interest, and uses MobileCLIP to retrieve the most relevant frames.

Run from the `system/` directory with the `m2a` conda environment active.

```
conda activate m2a
python3 minimodal.py --mode full --dataset ../data/mini --output ./output_mini
```

---

**minimodal.py**: Main entry point and pipeline orchestrator. Selects a random video from the dataset, runs each stage in sequence (YAMNet, YOLO, LLM, MobileCLIP), and writes all results to the output directory.

**audiomodule.py**: Loads YAMNet from TF Hub and classifies overlapping audio segments. Also handles speech transcription via Groq Whisper if speech is detected.

**yolosync.py**: Runs YOLOv8 on evenly-sampled frames to produce `VisualEvent` objects. Also defines the shared dataclasses (`AudioEvent`, `AudioSpeech`, `VisualEvent`, `FusedMoment`) and the timeline alignment and text rendering utilities used by the LLM stage.

**llmquery.py**: Aligns audio, speech, and visual events into a fused timeline, converts it to text, and sends it to Groq Llama. Returns a structured JSON response with violence probability, a summary, and MobileCLIP search queries for moments of interest.

**retrieveframe.py**: Extracts frames from a video at a fixed interval. Given a text query from the LLM, uses MobileCLIP to score and return the top 5 most relevant frames.