#!/usr/bin/env python3
"""
Take one video clip from the RLVS dataset (randomly pick one; fight/no-fight)

1. Extract audio
-- run YAMNet on it
-- audio classification across segments

2. Randomly sample image frames
-- run YOLO
-- BB classifications, COCO-style

3. Sequence frames from the video clip
-- run MobileCLIP
-- find relevant frames corresponding to fighting

Sample terminal command:
python3 minimodal.py \
    --mode full \
    --dataset "../data/mini" \
    --output ./output
"""

import argparse
import logging
import os
import random
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import subprocess
import tempfile
import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
import json

from audiomodule import analyze_audio_segments, transcribe_speech, preload_whisper
from llmquery import analyze_video_with_llm
from retrieveframe import extract_frames, get_best_frames
from yolosync import run_yolo_tracking, run_pose_detection
from movinet import run_movinet_inference, detect_fight_trigger, stream_segment_inference, SegmentPrediction


# ========================== KEYS SETUP ==========================

def load_keys():
    """Load API keys from keys/list.txt if not already in environment."""
    # Look for keys/list.txt in the workspace root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    keys_file = os.path.join(base_dir, "keys", "list.txt")
    
    if os.path.exists(keys_file):
        with open(keys_file, "r") as f:
            for line in f:
                if line.startswith("export "):
                    # strip 'export ' and whitespace
                    entry = line[7:].strip()
                    if "=" in entry:
                        key, val = entry.split("=", 1)
                        # strip potential quotes around the value
                        val = val.strip("'").strip('"')
                        if key not in os.environ:
                            os.environ[key] = val


# ========================== LOGGING SETUP ==========================

def setup_logging(clip_output_dir, clip_name):
    """Setup file + console logging in the clip-specific directory."""
    os.makedirs(clip_output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(clip_output_dir, f"violence_detection_{clip_name}_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


# ========================== VIDEO PICKER ==========================

def select_random_video(directory: str) -> str | None:
    """Pick a random video file from the dataset directory."""
    extensions = {".mp4", ".avi", ".mov", ".mkv"}
    candidates = [p for p in Path(directory).rglob("*") if p.suffix.lower() in extensions]
    if not candidates:
        return None
    return str(random.choice(candidates))


def has_audio_stream(video_path: str) -> bool:
    """Return True if the video contains at least one audio stream."""
    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        video_path,
    ]

    try:
        result = subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
        return bool(result.stdout.strip())
    except subprocess.CalledProcessError:
        return False


# ========================== AUDIO EXTRACTION ==========================

def extract_audio_from_video(video_path: str, output_path: str = None) -> str:
    """Extract audio from video using ffmpeg and return path to audio file."""
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".wav")
    
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        "-y", output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to extract audio: {e.stderr.decode()}")


def create_audio_segments(audio_path: str, segment_duration: float = 0.975, overlap: float = 0.5):
    """Split audio into overlapping segments for YAMNet analysis.
    
    Args:
        audio_path: Path to audio file
        segment_duration: Duration of each segment in seconds (YAMNet expects 0.975s)
        overlap: Overlap ratio (0.5 = 50% overlap)
    
    Yields:
        Tuples of (start_time, end_time, waveform_tensor)
    """
    # Load audio at 16kHz (YAMNet's expected sample rate)
    waveform, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
    
    segment_samples = int(segment_duration * sample_rate)
    hop_samples = int(segment_samples * (1 - overlap))
    
    start = 0
    while start < len(waveform):
        end = min(start + segment_samples, len(waveform))
        segment = waveform[start:end]
        
        # Pad if necessary to reach expected length
        if len(segment) < segment_samples:
            segment = np.pad(segment, (0, segment_samples - len(segment)))
        
        # Convert to tensor
        segment_tensor = tf.convert_to_tensor(segment, dtype=tf.float32)
        
        start_time = start / sample_rate
        end_time = end / sample_rate
        
        yield (start_time, end_time, segment_tensor)
        
        start += hop_samples
        if end >= len(waveform):
            break


# ========================== UTILITY FUNCTIONS ==========================

def get_clip_name_from_path(video_path: str) -> str:
    """Extract a clean clip name from the video file path."""
    return Path(video_path).stem

# ========================== PIPELINE ==========================

def run_pipeline(args):
    video_path = None
    audio_path = None
    clip_output_dir = None
    logger = None

    # Pre-load Whisper in the background if using local Whisper as primary.
    # Skip if Groq is primary to avoid unnecessary loading time.
    use_groq = os.environ.get("USE_GROQ_TRANSCRIPTION", "true").lower() in ("true", "1", "yes")
    if not use_groq:
        print("Pre-loading Whisper model in background...")
        threading.Thread(target=preload_whisper, daemon=True).start()
    else:
        print("Using Groq for transcription (set USE_GROQ_TRANSCRIPTION=false to use local Whisper)")

    for attempt in range(1, args.max_tries + 1):
        candidate = select_random_video(args.dataset)
        if not candidate:
            print(f"No videos found in dataset: {args.dataset}")
            return

        if not has_audio_stream(candidate):
            print(f"Attempt {attempt}: {candidate} has no audio track. Retrying...")
            continue

        print(f"Selected video: {candidate}")
        
        # Create clip-specific output directory and setup logging
        clip_name = get_clip_name_from_path(candidate)
        clip_output_dir = os.path.join(args.output, clip_name)
        os.makedirs(clip_output_dir, exist_ok=True)
        
        # Setup logging in the clip-specific directory
        logger = setup_logging(clip_output_dir, clip_name)
        logger.info(f"Selected video: {candidate}")
        logger.info(f"Created output directory: {clip_output_dir}")

        logger.info("Extracting audio from video...")
        try:
            audio_path = extract_audio_from_video(candidate)
            video_path = candidate
            logger.info(f"Audio extracted to: {audio_path}")
            break
        except RuntimeError as err:
            if logger:
                logger.warning(f"Attempt {attempt}: audio extraction failed for {candidate} -> {err}")
            else:
                print(f"Attempt {attempt}: audio extraction failed for {candidate} -> {err}")

    if not video_path or not audio_path or not clip_output_dir:
        if logger:
            logger.error("Unable to find a video with extractable audio after multiple attempts.")
        else:
            print("Unable to find a video with extractable audio after multiple attempts.")
        return

    # ---- Stage 0: YOLO Pose Detection (lightweight first-stage trigger) ----
    logger.info(
        f"Running YOLO pose detection (threshold={args.pose_likelihood_threshold}, "
        f"min_frames={args.pose_min_frames})..."
    )
    _t0 = time.perf_counter()
    pose_triggered, pose_hit_count, pose_frame_results = run_pose_detection(
        video_path,
        fight_model_path=args.pose_fight_model,
        yolo_model_path=args.pose_yolo_model,
        likelihood_threshold=args.pose_likelihood_threshold,
        min_trigger_frames=args.pose_min_frames,
    )
    pose_dt = time.perf_counter() - _t0
    logger.info(
        f"YOLO pose detection completed in {pose_dt:.2f}s — "
        f"{pose_hit_count} frame(s) met fight_likelihood>={args.pose_likelihood_threshold} "
        f"(need {args.pose_min_frames})"
    )

    # Save pose results
    pose_output = os.path.join(clip_output_dir, "pose_detection.json")
    with open(pose_output, "w") as f:
        json.dump(
            {
                "triggered": pose_triggered,
                "high_likelihood_count": pose_hit_count,
                "likelihood_threshold": args.pose_likelihood_threshold,
                "min_trigger_frames": args.pose_min_frames,
                "frames": pose_frame_results,
            },
            f,
            indent=2,
        )
    logger.info(f"Pose detection results saved to {pose_output}")

    if not pose_triggered:
        logger.info(
            "Pose detection did not trigger — insufficient fight evidence "
            f"({pose_hit_count}/{args.pose_min_frames} frames). Pipeline terminated early."
        )
        return

    logger.info(
        f"Pose detection TRIGGERED ({pose_hit_count} frames >= {args.pose_min_frames}). "
        "Proceeding to MoviNet..."
    )

    # Pose-only mode: stop after pose stage
    if args.mode == "pose":
        logger.info("Pose-only mode — pipeline complete.")
        return

    # Run MoviNet segment-based fight detection
    logger.info(f"Running MoviNet on {int(args.segment_fraction * 100)}%-length segments ({args.consecutive_fights} consecutive fight segments = {int(args.segment_fraction * args.consecutive_fights * 100)}% of clip required to trigger)...")
    _t0 = time.perf_counter()
    triggered, segment_predictions, trigger_segment = detect_fight_trigger(
        video_path,
        threshold=args.fight_threshold,
        n_frames=args.movinet_frames,
        segment_fraction=args.segment_fraction,
        consecutive_segments=args.consecutive_fights,
    )
    movinet_dt = time.perf_counter() - _t0
    
    # Log per-segment predictions
    for pred in segment_predictions:
        logger.info(
            f"Segment {pred.segment_index} [{pred.start_time:.1f}s-{pred.end_time:.1f}s]: "
            f"{pred.label} (Fight: {pred.fight_prob:.3f}, No_Fight: {pred.no_fight_prob:.3f})"
        )
    
    # Save MoviNet results
    movinet_output = os.path.join(clip_output_dir, "movinet_segments.json")
    with open(movinet_output, "w") as f:
        json.dump(
            [
                {
                    "segment": p.segment_index,
                    "start_time": p.start_time,
                    "end_time": p.end_time,
                    "label": p.label,
                    "fight_prob": p.fight_prob,
                    "no_fight_prob": p.no_fight_prob,
                }
                for p in segment_predictions
            ],
            f,
            indent=2,
        )
    logger.info(f"MoviNet results saved to {movinet_output}")
    
    if not triggered:
        logger.info(f"MoviNet completed in {movinet_dt:.2f}s - No sustained fight detected (threshold: {args.fight_threshold}, required: {args.consecutive_fights} consecutive {int(args.segment_fraction * 100)}%-length segments)")
        logger.info("Pipeline terminated early - no violence trigger.")
        return
    
    trigger_time = segment_predictions[trigger_segment].start_time if trigger_segment is not None else 0.0
    logger.info(f"MoviNet FIGHT TRIGGERED at segment {trigger_segment} ({trigger_time:.1f}s) after {args.consecutive_fights} consecutive fight detections in {movinet_dt:.2f}s")
    logger.info("Continuing with full analysis pipeline...")

    # If movinet-only mode, stop here after logging the trigger
    if args.mode == "movinet":
        logger.info(f"MoviNet-only mode - {len(segment_predictions)} segments analysed, pipeline complete.")
        return

    # Create audio segments for YAMNet analysis
    logger.info("Creating audio segments...")
    audio_segments = list(create_audio_segments(audio_path))
    logger.info(f"Created {len(audio_segments)} audio segments")

    # Run YAMNet on audio segments
    logger.info("Running YAMNet audio classification...")
    _t0 = time.perf_counter()
    yamnet_events = analyze_audio_segments(audio_segments)
    yamnet_dt = time.perf_counter() - _t0
    logger.info(f"YAMNet detected {len(yamnet_events)} audio events in {yamnet_dt:.2f}s")
    
    # Log sample of detected audio events
    if yamnet_events:
        logger.info(f"Sample audio events: {[(e.label, f'{e.confidence:.2f}') for e in yamnet_events[:5]]}")
    
    # Save YAMNet results
    yamnet_output = os.path.join(clip_output_dir, "yamnet_events.json")
    with open(yamnet_output, "w") as f:
        json.dump([{"start": e.start, "end": e.end, "label": e.label, "confidence": e.confidence} for e in yamnet_events], f, indent=2)
    logger.info(f"YAMNet results saved to {yamnet_output}")
    
    # Run speech transcription if speech detected
    logger.info("Checking for speech transcription...")
    _t0 = time.perf_counter()
    speech_events = transcribe_speech(yamnet_events, audio_segments, audio_path)
    speech_dt = time.perf_counter() - _t0
    logger.info(f"Transcribed {len(speech_events)} speech segments in {speech_dt:.2f}s")
    
    # Log sample of transcribed speech
    if speech_events:
        logger.info(f"Sample transcriptions: {[e.label[:50] + '...' if len(e.label) > 50 else e.label for e in speech_events[:3]]}")
    
    # Save speech transcription results
    speech_output = os.path.join(clip_output_dir, "speech_events.json")
    with open(speech_output, "w") as f:
        json.dump([{"start": e.start, "end": e.end, "text": e.label, "confidence": e.confidence} for e in speech_events], f, indent=2)
    logger.info(f"Speech transcription results saved to {speech_output}")
    
    # Run YOLO on video
    logger.info(f"Running YOLO object detection on {args.yolo_frames} sampled frames...")
    _t0 = time.perf_counter()
    yolo_events = run_yolo_tracking(video_path, display=args.mode == "yolo", num_frames=args.yolo_frames)
    yolo_dt = time.perf_counter() - _t0
    logger.info(f"YOLO detected {len(yolo_events)} visual events in {yolo_dt:.2f}s")
    
    # Log sample of detected objects
    if yolo_events:
        sample_objs = set()
        for e in yolo_events[:3]:
            sample_objs.update(e.objects)
        logger.info(f"Sample detected objects: {list(sample_objs)[:10]}")
    
    # Save YOLO results
    yolo_output = os.path.join(clip_output_dir, "yolo_events.json")
    with open(yolo_output, "w") as f:
        json.dump([{"time": e.time, "objects": e.objects, "boxes": e.boxes, "confidences": e.confidences} for e in yolo_events], f, indent=2)
    logger.info(f"YOLO results saved to {yolo_output}")

    if args.mode == "yamnet":
        logger.info(f"YAMNet events: {len(yamnet_events)}")
        return

    if args.mode == "yolo":
        logger.info(f"YOLO events: {len(yolo_events)}")
        return

    # LLM reasoning (used by mobileclip/full modes)
    logger.info("Querying LLM for scene analysis and MobileCLIP query generation...")

    # Build movinet segment dicts for the LLM
    movinet_segment_dicts = [
        {
            "segment": p.segment_index,
            "start_time": p.start_time,
            "end_time": p.end_time,
            "label": p.label,
            "fight_prob": p.fight_prob,
            "no_fight_prob": p.no_fight_prob,
        }
        for p in segment_predictions
    ]

    _t0 = time.perf_counter()
    llm_response = analyze_video_with_llm(
        yamnet_predictions=yamnet_events,
        whisper_predictions=speech_events,
        yolo_predictions=yolo_events,
        pose_frame_results=pose_frame_results,
        movinet_segments=movinet_segment_dicts,
        question=args.question,
    )
    llm_dt = time.perf_counter() - _t0
    logger.info(f"LLM analysis completed in {llm_dt:.2f}s")

    logger.info(f"LLM Summary: {llm_response.get('summary', 'N/A')}")
    for i, m in enumerate(llm_response.get('moments_of_interest', [])):
        logger.info(f"  Moment {i}: [{m.get('start', 0):.1f}s–{m.get('end', 0):.1f}s] {m.get('description', '')}")
        for q in m.get('mobileclip_queries', []):
            logger.info(f"    query: {q}")
    
    # Save LLM response
    llm_output = os.path.join(clip_output_dir, "llm_response.json")
    with open(llm_output, "w") as f:
        json.dump(llm_response, f, indent=2)
    logger.info(f"LLM analysis saved to {llm_output}")

    if args.mode == "mobileclip":
        frame_paths = extract_frames(video_path, os.path.join(clip_output_dir, "frames"))

        # Use the flat top-level mobileclip_queries list from LLM response (2-3 queries total)
        all_best_frames = []
        _t0 = time.perf_counter()
        for query in llm_response.get("mobileclip_queries", []):
            logger.info(f"Searching frames for: {query}")
            best_frames = get_best_frames(frame_paths, query, args.mobileclip_weights, top_k=2)
            all_best_frames.extend(best_frames)
        mobileclip_dt = time.perf_counter() - _t0

        logger.info(f"Selected {len(set(all_best_frames))} unique frames")
        logger.info(f"Frames: {list(set(all_best_frames))}")
        logger.info(f"MobileCLIP inference completed in {mobileclip_dt:.2f}s")
        return

    # full pipeline
    frame_paths = extract_frames(video_path, os.path.join(clip_output_dir, "frames"))
    logger.info(f"Extracted {len(frame_paths)} frames")

    # Use the flat top-level mobileclip_queries list from LLM response (2-3 queries total)
    all_best_frames = []
    _t0 = time.perf_counter()
    for query in llm_response.get("mobileclip_queries", []):
        logger.info(f"Searching frames for: {query}")
        best_frames = get_best_frames(frame_paths, query, args.mobileclip_weights, top_k=2)
        all_best_frames.extend(best_frames)
    mobileclip_dt = time.perf_counter() - _t0
    logger.info("Pipeline complete")
    logger.info(f"Summary: {llm_response.get('summary', 'N/A')}")
    logger.info(f"Selected {len(set(all_best_frames))} unique frames")
    logger.info(f"Frames: {list(set(all_best_frames))}")
    logger.info(f"MobileCLIP inference completed in {mobileclip_dt:.2f}s")


# ========================== MAIN ==========================

def main():
    parser = argparse.ArgumentParser(
        description="Violence Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="pose | movinet | yamnet | yolo | mobileclip | full\n\nPipeline order: YOLO Pose Detection → MoviNet → YAMNet → YOLO BB → LLM → MobileCLIP.\nYOLO pose runs first; if <pose-min-frames frames reach pose-likelihood-threshold, the pipeline terminates early.",
    )

    # Load keys before starting the pipeline
    load_keys()
    
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["yamnet", "yolo", "mobileclip", "movinet", "pose", "full"],
        help="Operation mode: pose, yamnet, yolo, mobileclip, movinet, or full",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="/Volumes/KAUSAR/kaggle/Real Life Violence Dataset",
        help="Path to training dataset directory",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Output directory for models, logs, and results",
    )

    # gotta modify this; .pt file is in the weights folder
    parser.add_argument(
        "--mobileclip-weights",
        dest="mobileclip_weights",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml-mobileclip", "checkpoints", "mobileclip_s1.pt")),
        help="Path to MobileCLIP weights",
    )

    parser.add_argument(
        "--question",
        type=str,
        default="Describe the violent actions visible in this clip and produce MobileCLIP queries.",
        help="Question passed to the LLM",
    )

    parser.add_argument(
        "--max-tries",
        dest="max_tries",
        type=int,
        default=20,
        help="Maximum number of videos to sample when searching for one with audio",
    )

    parser.add_argument(
        "--yolo-frames",
        dest="yolo_frames",
        type=int,
        default=10,
        help="Number of frames to sample for YOLO detection (default: 10)",
    )

    parser.add_argument(
        "--fight-threshold",
        dest="fight_threshold",
        type=float,
        default=0.5,
        help="Fight probability threshold per segment to trigger full pipeline (default: 0.5)",
    )

    parser.add_argument(
        "--segment-fraction",
        dest="segment_fraction",
        type=float,
        default=0.1,
        help="Each MoviNet segment as a fraction of total video length (default: 0.1 = 10%%)",
    )

    parser.add_argument(
        "--consecutive-fights",
        dest="consecutive_fights",
        type=int,
        default=3,
        help="Number of consecutive fight segments required to trigger downstream pipeline (default: 3)",
    )

    parser.add_argument(
        "--movinet-frames",
        dest="movinet_frames",
        type=int,
        default=12,
        help="Number of frames sampled per segment for MoviNet (default: 12)",
    )

    parser.add_argument(
        "--pose-fight-model",
        dest="pose_fight_model",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "model", "fight", "fight-model.pth")),
        help="Path to FightDetector .pth weights used by YOLO pose stage",
    )

    parser.add_argument(
        "--pose-yolo-model",
        dest="pose_yolo_model",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "model", "yolo", "yolov8n-pose.pt")),
        help="Path to YOLOv8-pose .pt weights used by YOLO pose stage",
    )

    parser.add_argument(
        "--pose-likelihood-threshold",
        dest="pose_likelihood_threshold",
        type=float,
        default=0.30,
        help="Per-frame fight_likelihood score that counts as a hit (default: 0.30)",
    )

    parser.add_argument(
        "--pose-min-frames",
        dest="pose_min_frames",
        type=int,
        default=3,
        help="Minimum number of frames meeting the likelihood threshold to trigger MoviNet (default: 3)",
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 60)
    print("Violence Detection System Started")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Output directory: {args.output}")

    try:
        run_pipeline(args)
        print("=" * 60)
        print("Violence Detection System Completed Successfully")
        print("=" * 60)
    except Exception as e:
        print(f"Error occurred: {e}")
        raise


if __name__ == "__main__":
    main()