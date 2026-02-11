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
"""

import os
import sys
import argparse
import logging
import os
import cv2
import numpy as np
from datetime import datetime

import tensorflow as tf
import tensorflow_hub as hub
import csv
import io

import torch
from PIL import Image
import mobileclip

# ========================== CONFIG ==========================

IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 16

# ========================== LOGGING SETUP ==========================

def setup_logging(log_dir):
    """Setup comprehensive logging"""
    log_subdir = os.path.join(log_dir, 'logs')
    os.makedirs(log_subdir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_subdir, f"violence_detection_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

# ========================== VIDEO PARSING ==========================

def count_video_frames(video_path):
    """Count total frames in a video"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def select_random_video(directory):
    """
    can split off randomly into fight/no-fight
    within the sub-directory, it will randomly select a video
    """
    pass

# ========================== FRAME EXTRACTION ==========================

def frames_extraction(video_path, logger):
    """Extract frames from video with proper error handling"""
    frames_list = []
    
    video_reader = cv2.VideoCapture(video_path)
    
    if not video_reader.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return frames_list
    
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if video_frames_count == 0:
        logger.warning(f"Video has 0 frames: {video_path}")
        video_reader.release()
        return frames_list
    
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)
    
    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        
        if not success:
            break
        
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255.0
        frames_list.append(normalized_frame)
    
    video_reader.release()
    return frames_list

# =================== AUDIO CLASSIFICATION ===================

def generate_audio_label(segment):
    # load the model
    model = hub.load('https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1')

    # input: certain amount of audio from the video clip
    """
    we could create a 50% overlap across the audio segments,
    get a list of the most common audio labels
    return the top-3
    """
    waveform = segment  # some kind of sampling can be put in place

    # run the model, check the output
    scores, embeddings, log_mel_spectrogram = model(waveform)
    scores.shape.assert_is_compatible_with([None, 521])
    embeddings.shape.assert_is_compatible_with([None, 1024])
    log_mel_spectrogram.shape.assert_is_compatible_with([None, 64])

    class_map_path = model.class_map_path().numpy()
    class_names = class_names_from_csv(tf.io.read_file(class_map_path).numpy().decode('utf-8'))

    return class_names[scores.numpy().mean(axis=0).argmax()]  # will need to return something here; i'm guessing the label with confidence

# find the name of class with the top score when mean-aggregated across frames
def class_names_from_csv(class_map_csv_text):
    """return a list of class names corresponding to score vector"""
    class_map_csv = io.StringIO(class_map_csv_text)
    class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
    class_names = class_names[1:]   # skip csv header
    return class_names


# =================== OBJECT DETECTION ===================
"""using YOLO here"""


"""
random idea but the audio classification labels and the yolo detections-
can be combined into one by an LLM-
LLM could hypothesize what is occurring based on the transcription (whisper), sound, and images (yolo)
and then it could subsequently prompt mobileclip-
asking it to find any instances of violence leveraging the previous context
"""


# =================== LLM PARSER =========================
"""YAMNet + YOLO => Llama"""



# =================== FRAME RETRIEVAL ===================
"""
FULL IMAGE SET + PROMPT => SPECIFIC FRAMES
full image set created from the frames in the video clip
the prompt was created by Llama using yolo bb's and yamnet audio classes
unknown: idk what the prompt is going to look like
the above two are then fed into mobileclip to find instances of violence
"""

def get_best_frames(video_path, llm_query):
    model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s1', pretrained='/path/to/mobileclip_s1.pt')
    tokenizer = mobileclip.get_tokenizer('mobileclip_s1')

    # create the frames; could use the frame extraction function
    frame_paths = None

    # step 1- encode all frames once
    images = torch.stack([preprocess(Image.open(p).convert('RGB')) for p in frame_paths])
    with torch.no_grad():
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    
    # step 2- encode ONE query text
    query = tokenizer([llm_query])   # this is a string; maybe we could have a list of words as well, get confidences

    with torch.no_grad():
        text_features = model.encode_text(query)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # step 3- similarity search
    scores = image_features @ text_features.T
    topk = scores.squeeze().topk(5)

    best_frames = [frame_paths[i] for i in topk.indices]

    return best_frames      # the 5 most relevant frames to the text prompt


# ========================== MAIN ==========================

def main():
    parser = argparse.ArgumentParser(
        description='Violence Detection',
        formatter_classer=argparse.RawDescriptionHelpFormatter,
        epilog="yamnet | yolo | mobileclip | full"
    )

    parser.add_argument('--mode', type=str, required=True,
                        choices=['yamnet', 'yolo', 'mobileclip', 'full'],
                        help='Operation mode: yamnet, yolo, mobileclip, or full')
    
    parser.add_argument('--dataset', type=str,
                        default='/Volumes/KAUSAR/kaggle/Real Life Violence Dataset',
                        help='Path to training dataset directory')
    
    parser.add_argument('--output', type=str,
                        default='./output',
                        help='Output directory for models, logs, and results')

    args = parser.parse_args()

    # set up the directory
    os.makedirs(args.output, exist_ok=True)

    # set up logging
    log_dir = os.path.dirname(os.path.abspath(__file__))
    logger = setup_logging(log_dir)

    logger.info("=" * 60)
    logger.info("Violence Detection System Started")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Output directory: {args.output}")

    try:
        """
        calling the appropriate sub-modules
        can also run the full sequence if so specified
        """

        logger.info("=" * 60)
        logger.info("Violence Detection System Completed Successfully")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()