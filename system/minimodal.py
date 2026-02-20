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
    --dataset "/Volumes/KAUSAR/kaggle/Real Life Violence Dataset" \
    --output ./output_full
"""

import argparse
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import subprocess
import tempfile
import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
import json

from audiomodule import analyze_audio_segments, transcribe_speech
from llmquery import analyze_video_with_llm
from retrieveframe import extract_frames, get_best_frames
from yolosync import run_yolo_tracking


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

def setup_logging(log_dir):
    """Setup file + console logging."""
    log_subdir = os.path.join(log_dir, "logs")
    os.makedirs(log_subdir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_subdir, f"violence_detection_{timestamp}.log")

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


# ========================== PIPELINE ==========================

def run_pipeline(args, logger):
    video_path = None
    audio_path = None

    for attempt in range(1, args.max_tries + 1):
        candidate = select_random_video(args.dataset)
        if not candidate:
            logger.error(f"No videos found in dataset: {args.dataset}")
            return

        if not has_audio_stream(candidate):
            logger.warning(f"Attempt {attempt}: {candidate} has no audio track. Retrying...")
            continue

        logger.info(f"Selected video: {candidate}")

        logger.info("Extracting audio from video...")
        try:
            audio_path = extract_audio_from_video(candidate)
            video_path = candidate
            logger.info(f"Audio extracted to: {audio_path}")
            break
        except RuntimeError as err:
            logger.warning(f"Attempt {attempt}: audio extraction failed for {candidate} -> {err}")

    if not video_path or not audio_path:
        logger.error("Unable to find a video with extractable audio after multiple attempts.")
        return

    # Create audio segments for YAMNet analysis
    logger.info("Creating audio segments...")
    audio_segments = list(create_audio_segments(audio_path))
    logger.info(f"Created {len(audio_segments)} audio segments")

    # Run YAMNet on audio segments
    logger.info("Running YAMNet audio classification...")
    yamnet_events = analyze_audio_segments(audio_segments)
    logger.info(f"YAMNet detected {len(yamnet_events)} audio events")
    
    # Log sample of detected audio events
    if yamnet_events:
        logger.info(f"Sample audio events: {[(e.label, f'{e.confidence:.2f}') for e in yamnet_events[:5]]}")
    
    # Save YAMNet results
    yamnet_output = os.path.join(args.output, "yamnet_events.json")
    with open(yamnet_output, "w") as f:
        json.dump([{"start": e.start, "end": e.end, "label": e.label, "confidence": e.confidence} for e in yamnet_events], f, indent=2)
    logger.info(f"YAMNet results saved to {yamnet_output}")
    
    # Run speech transcription if speech detected
    logger.info("Checking for speech transcription...")
    speech_events = transcribe_speech(yamnet_events, audio_segments, audio_path)
    logger.info(f"Transcribed {len(speech_events)} speech segments")
    
    # Log sample of transcribed speech
    if speech_events:
        logger.info(f"Sample transcriptions: {[e.label[:50] + '...' if len(e.label) > 50 else e.label for e in speech_events[:3]]}")
    
    # Save speech transcription results
    speech_output = os.path.join(args.output, "speech_events.json")
    with open(speech_output, "w") as f:
        json.dump([{"start": e.start, "end": e.end, "text": e.label, "confidence": e.confidence} for e in speech_events], f, indent=2)
    logger.info(f"Speech transcription results saved to {speech_output}")
    
    # Run YOLO on video
    logger.info(f"Running YOLO object detection on {args.yolo_frames} sampled frames...")
    yolo_events = run_yolo_tracking(video_path, display=args.mode == "yolo", num_frames=args.yolo_frames)
    logger.info(f"YOLO detected {len(yolo_events)} visual events")
    
    # Log sample of detected objects
    if yolo_events:
        sample_objs = set()
        for e in yolo_events[:3]:
            sample_objs.update(e.objects)
        logger.info(f"Sample detected objects: {list(sample_objs)[:10]}")
    
    # Save YOLO results
    yolo_output = os.path.join(args.output, "yolo_events.json")
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
    logger.info("Querying LLM for violence analysis...")
    llm_response = analyze_video_with_llm(
        yamnet_predictions=yamnet_events,
        whisper_predictions=speech_events,
        yolo_predictions=yolo_events,
        question=args.question,
    )

    logger.info(f"LLM Violence Probability: {llm_response.get('violence_probability', 'N/A')}")
    logger.info(f"LLM Summary: {llm_response.get('summary', 'N/A')}")
    
    # Save LLM response
    llm_output = os.path.join(args.output, "llm_response.json")
    with open(llm_output, "w") as f:
        json.dump(llm_response, f, indent=2)
    logger.info(f"LLM analysis saved to {llm_output}")

    if args.mode == "mobileclip":
        frame_paths = extract_frames(video_path, os.path.join(args.output, "frames"))
        
        # Use MobileCLIP queries from LLM response
        all_best_frames = []
        for moment in llm_response.get("moments_of_interest", []):
            for query in moment.get("mobileclip_queries", []):
                logger.info(f"Searching frames for: {query}")
                best_frames = get_best_frames(frame_paths, query, args.mobileclip_weights)
                all_best_frames.extend(best_frames)
        
        logger.info(f"Selected {len(set(all_best_frames))} unique frames")
        logger.info(f"Frames: {list(set(all_best_frames))}")
        return

    # full pipeline
    frame_paths = extract_frames(video_path, os.path.join(args.output, "frames"))
    logger.info(f"Extracted {len(frame_paths)} frames")
    
    # Use MobileCLIP queries from LLM response
    all_best_frames = []
    for moment in llm_response.get("moments_of_interest", []):
        for query in moment.get("mobileclip_queries", []):
            logger.info(f"Searching frames for: {query}")
            best_frames = get_best_frames(frame_paths, query, args.mobileclip_weights)
            all_best_frames.extend(best_frames)
    
    logger.info("Pipeline complete")
    logger.info(f"Violence probability: {llm_response.get('violence_probability', 0.0)}")
    logger.info(f"Summary: {llm_response.get('summary', 'N/A')}")
    logger.info(f"Selected {len(set(all_best_frames))} unique frames")
    logger.info(f"Frames: {list(set(all_best_frames))}")


# ========================== MAIN ==========================

def main():
    parser = argparse.ArgumentParser(
        description="Violence Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="yamnet | yolo | mobileclip | full",
    )

    # Load keys before starting the pipeline
    load_keys()
    
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["yamnet", "yolo", "mobileclip", "full"],
        help="Operation mode: yamnet, yolo, mobileclip, or full",
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
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weights", "mobileclip_s1.pt")),
        help="Path to MobileCLIP weights",
    )

    parser.add_argument(
        "--question",
        type=str,
        default="Is violence likely?",
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

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    log_dir = os.path.dirname(os.path.abspath(__file__))
    logger = setup_logging(log_dir)

    logger.info("=" * 60)
    logger.info("Violence Detection System Started")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Output directory: {args.output}")

    try:
        run_pipeline(args, logger)
        logger.info("=" * 60)
        logger.info("Violence Detection System Completed Successfully")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()