#!/usr/bin/env python3
"""Violence detection: train, evaluate, infer."""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ========================== CONFIG ==========================

IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 16
CLASSES_LIST = ["NonViolence", "Violence"]

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

def scan_videos_in_directory(directory, logger):
    """Scan directory for video files and return statistics"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    videos = []
    
    path = Path(directory)
    if not path.exists():
        logger.warning(f"Directory not found: {directory}")
        return videos
    
    logger.info(f"Scanning directory: {directory}")
    
    for file in path.rglob("*"):
        if file.suffix.lower() in video_extensions and not file.name.startswith('.'):
            try:
                frame_count = count_video_frames(str(file))
                videos.append({
                    'path': str(file),
                    'name': file.name,
                    'frames': frame_count
                })
                logger.debug(f"Found video: {file.name} ({frame_count} frames)")
            except Exception as e:
                logger.error(f"Error processing {file.name}: {e}")
    
    logger.info(f"Found {len(videos)} videos in {directory}")
    return videos

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

# ========================== DATASET CREATION ==========================

def create_dataset(dataset_dir, logger):
    """Create dataset from directory structure"""
    features = []
    labels = []
    video_paths = []
    
    logger.info(f"Creating dataset from: {dataset_dir}")
    
    for class_index, class_name in enumerate(CLASSES_LIST):
        class_dir = os.path.join(dataset_dir, class_name)
        
        if not os.path.exists(class_dir):
            logger.warning(f"Class directory not found: {class_dir}")
            continue
        
        logger.info(f"Processing class: {class_name}")
        files_list = [f for f in os.listdir(class_dir) if not f.startswith('.')]
        
        for idx, file_name in enumerate(files_list):
            video_path = os.path.join(class_dir, file_name)
            
            frames = frames_extraction(video_path, logger)
            
            if len(frames) == SEQUENCE_LENGTH:
                features.append(frames)
                labels.append(class_index)
                video_paths.append(video_path)
                
                if (idx + 1) % 50 == 0:
                    logger.info(f"Processed {idx + 1}/{len(files_list)} videos for {class_name}")
        
        logger.info(f"Completed {class_name}: {len([l for l in labels if l == class_index])} valid videos")
    
    features = np.asarray(features)
    labels = np.array(labels)
    
    logger.info(f"Dataset created: {features.shape[0]} samples")
    return features, labels, video_paths

def load_or_create_dataset(dataset_dir, cache_dir, force_recreate, logger):
    """Load cached dataset or create new one"""
    os.makedirs(cache_dir, exist_ok=True)
    
    features_file = os.path.join(cache_dir, "features.npy")
    labels_file = os.path.join(cache_dir, "labels.npy")
    paths_file = os.path.join(cache_dir, "video_paths.npy")
    
    if not force_recreate and all(os.path.exists(f) for f in [features_file, labels_file, paths_file]):
        logger.info("Loading cached dataset...")
        features = np.load(features_file)
        labels = np.load(labels_file)
        video_paths = np.load(paths_file, allow_pickle=True)
        logger.info(f"Loaded dataset: {features.shape[0]} samples")
    else:
        logger.info("Creating new dataset...")
        features, labels, video_paths = create_dataset(dataset_dir, logger)
        
        np.save(features_file, features)
        np.save(labels_file, labels)
        np.save(paths_file, video_paths)
        logger.info("Dataset saved to cache")
    
    return features, labels, video_paths

# ========================== MODEL CREATION ==========================

def create_model(logger):
    """Create LRCN model with MobileNetV2 and BiLSTM"""
    logger.info("Creating model architecture...")
    
    mobilenet = MobileNetV2(include_top=False, weights="imagenet")
    
    mobilenet.trainable = True
    for layer in mobilenet.layers[:-40]:
        layer.trainable = False
    
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
        TimeDistributed(mobilenet),
        Dropout(0.25),
        TimeDistributed(Flatten()),
        Bidirectional(LSTM(units=32)),
        Dropout(0.25),
        Dense(256, activation='relu'),
        Dropout(0.25),
        Dense(128, activation='relu'),
        Dropout(0.25),
        Dense(64, activation='relu'),
        Dropout(0.25),
        Dense(32, activation='relu'),
        Dropout(0.25),
        Dense(len(CLASSES_LIST), activation='softmax')
    ])
    
    logger.info("Model architecture created")
    model.summary(print_fn=lambda x: logger.info(x))
    
    return model

# ========================== TRAINING ==========================

def train_model(model, features_train, labels_train, output_dir, logger):
    """Train the model"""
    logger.info("Starting model training...")
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.6,
        patience=5,
        min_lr=5e-5,
        verbose=1
    )
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='sgd',
        metrics=['accuracy']
    )
    
    history = model.fit(
        x=features_train,
        y=labels_train,
        epochs=50,
        batch_size=8,
        shuffle=True,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    model_path = os.path.join(output_dir, 'violence_detection_model.h5')
    model.save(model_path)
    logger.info(f"Model saved to: {model_path}")
    
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)
    logger.info(f"Training history saved to: {history_path}")
    
    return history

def plot_training_metrics(history, output_dir, logger):
    """Plot and save training metrics"""
    logger.info("Plotting training metrics...")
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    loss_path = os.path.join(output_dir, 'loss_plot.png')
    plt.savefig(loss_path)
    plt.close()
    logger.info(f"Loss plot saved to: {loss_path}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    acc_path = os.path.join(output_dir, 'accuracy_plot.png')
    plt.savefig(acc_path)
    plt.close()
    logger.info(f"Accuracy plot saved to: {acc_path}")

# ========================== EVALUATION ==========================

def evaluate_model(model, features_test, labels_test, output_dir, logger):
    """Evaluate model on test set"""
    logger.info("Evaluating model on test set...")
    
    predictions = model.predict(features_test)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(labels_test, axis=1)
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                xticklabels=CLASSES_LIST,
                yticklabels=CLASSES_LIST)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.4f})')
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    logger.info(f"Confusion matrix saved to: {cm_path}")
    
    report = classification_report(true_labels, predicted_labels, target_names=CLASSES_LIST)
    logger.info(f"Classification Report:\n{report}")
    
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    logger.info(f"Classification report saved to: {report_path}")
    
    return accuracy

# ========================== RWF-2000 LABEL ANALYSIS ==========================

def extract_rwf_label(video_path):
    """Extract ground truth label from RWF-2000 directory structure"""
    path_parts = video_path.replace('\\', '/').split('/')
    
    for part in path_parts:
        if 'Fight' in part and 'NonFight' not in part:
            return 'Fight'
        elif 'NonFight' in part:
            return 'NonFight'
    
    return None

def map_rwf_label_to_class(label):
    """Map RWF-2000 label to Violence/NonViolence"""
    if label == 'Fight':
        return 'Violence'
    elif label == 'NonFight':
        return 'NonViolence'
    return None

def analyze_predictions(results_file, output_dir, logger):
    """Analyze predictions against RWF-2000 ground truth labels"""
    logger.info("="*60)
    logger.info("ANALYZING PREDICTIONS - RWF-2000 DATASET")
    logger.info("="*60)
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    y_true = []
    y_pred = []
    y_conf = []
    labeled_videos = []
    unlabeled_count = 0
    
    for result in results:
        video_path = result.get('video_path', '')
        video_name = result['video_name']
        predicted_class = result['predicted_class']
        confidence = result['confidence']
        
        # Extract ground truth from path
        rwf_label = extract_rwf_label(video_path)
        ground_truth = map_rwf_label_to_class(rwf_label)
        
        if ground_truth and predicted_class:
            y_true.append(ground_truth)
            y_pred.append(predicted_class)
            y_conf.append(confidence)
            labeled_videos.append({
                'name': video_name,
                'path': video_path,
                'ground_truth': ground_truth,
                'predicted': predicted_class,
                'confidence': confidence,
                'correct': ground_truth == predicted_class
            })
        else:
            unlabeled_count += 1
    
    if not y_true:
        logger.warning("No videos with valid labels found")
        return
    
    logger.info(f"Videos with labels: {len(y_true)}")
    logger.info(f"Videos without labels: {unlabeled_count}")
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    logger.info(f"\nOverall Metrics:")
    logger.info(f"  Accuracy:  {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1 Score:  {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=['NonViolence', 'Violence'])
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"                True NonViolence | True Violence")
    logger.info(f"Pred NonViolence      {cm[0][0]:7d} | {cm[0][1]:6d}")
    logger.info(f"Pred Violence         {cm[1][0]:7d} | {cm[1][1]:6d}")
    
    # Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NonViolence', 'Violence'],
                yticklabels=['NonViolence', 'Violence'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual (RWF-2000)')
    plt.title(f'RWF-2000 Evaluation (Acc: {accuracy:.4f})')
    cm_path = os.path.join(output_dir, 'rwf2000_confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    logger.info(f"\nConfusion matrix saved to: {cm_path}")
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=['NonViolence', 'Violence'])
    logger.info(f"\nClassification Report:\n{report}")
    
    # Save detailed report
    report_path = os.path.join(output_dir, 'rwf2000_analysis.txt')
    with open(report_path, 'w') as f:
        f.write("RWF-2000 Dataset Evaluation\n")
        f.write("="*60 + "\n\n")
        f.write(f"Videos analyzed: {len(y_true)}\n")
        f.write(f"Videos without labels: {unlabeled_count}\n\n")
        f.write(f"Overall Metrics:\n")
        f.write(f"  Accuracy:  {accuracy:.4f}\n")
        f.write(f"  Precision: {precision:.4f}\n")
        f.write(f"  Recall:    {recall:.4f}\n")
        f.write(f"  F1 Score:  {f1:.4f}\n\n")
        f.write(f"Classification Report:\n{report}\n")
        f.write(f"\nMisclassified Videos:\n")
        f.write("="*60 + "\n")
        for vid in labeled_videos:
            if not vid['correct']:
                f.write(f"  {vid['name']}\n")
                f.write(f"    Truth: {vid['ground_truth']}\n")
                f.write(f"    Predicted: {vid['predicted']} (Conf: {vid['confidence']:.4f})\n\n")
    
    logger.info(f"Analysis report saved to: {report_path}")
    
    # Summary statistics
    correct_count = sum(1 for v in labeled_videos if v['correct'])
    logger.info(f"\nSummary: {correct_count}/{len(labeled_videos)} correct predictions")
    logger.info("="*60)

# ========================== INFERENCE ==========================

def predict_single_video(model, video_path, logger):
    """Predict violence in a single video"""
    logger.debug(f"Processing video: {video_path}")
    
    # Extract frames
    frames = frames_extraction(video_path, logger)
    
    if len(frames) != SEQUENCE_LENGTH:
        logger.warning(f"Insufficient frames ({len(frames)}) in video: {video_path}")
        return None, None
    
    # Predict
    frames_array = np.expand_dims(frames, axis=0)
    predictions = model.predict(frames_array, verbose=0)[0]
    
    predicted_label = np.argmax(predictions)
    predicted_class = CLASSES_LIST[predicted_label]
    confidence = predictions[predicted_label]
    
    return predicted_class, confidence

def batch_inference(model, video_directories, output_dir, logger):
    """Run inference on multiple directories of videos"""
    logger.info("Starting batch inference...")
    
    all_results = []
    
    for directory in video_directories:
        if not os.path.exists(directory):
            logger.warning(f"Directory not found: {directory}")
            continue
        
        logger.info(f"Processing directory: {directory}")
        videos = scan_videos_in_directory(directory, logger)
        
        for video_info in videos:
            video_path = video_info['path']
            predicted_class, confidence = predict_single_video(model, video_path, logger)
            
            result = {
                'video_path': video_path,
                'video_name': video_info['name'],
                'total_frames': video_info['frames'],
                'predicted_class': predicted_class,
                'confidence': float(confidence) if confidence is not None else None
            }
            all_results.append(result)
            
            if predicted_class:
                logger.info(f"{video_info['name']}: {predicted_class} (confidence: {confidence:.4f})")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f'inference_results_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Inference results saved to: {results_file}")
    
    # Summary statistics
    violence_count = sum(1 for r in all_results if r['predicted_class'] == 'Violence')
    nonviolence_count = sum(1 for r in all_results if r['predicted_class'] == 'NonViolence')
    failed_count = sum(1 for r in all_results if r['predicted_class'] is None)
    
    logger.info("=" * 60)
    logger.info("INFERENCE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total videos processed: {len(all_results)}")
    logger.info(f"Violence detected: {violence_count}")
    logger.info(f"Non-violence detected: {nonviolence_count}")
    logger.info(f"Failed to process: {failed_count}")
    logger.info("=" * 60)
    
    return all_results

# ========================== MAIN ==========================

def main():
    parser = argparse.ArgumentParser(
        description='Violence Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="train | inference | full | analyze"
    )
    
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'inference', 'full', 'analyze'],
                        help='Operation mode: train, inference, full, or analyze')
    
    parser.add_argument('--dataset', type=str,
                        default='/Volumes/KAUSAR/kaggle/Real Life Violence Dataset',
                        help='Path to training dataset directory')
    
    parser.add_argument('--test-dirs', type=str, nargs='+',
                        default=['/Volumes/KAUSAR/1-1004', '/Volumes/KAUSAR/videos/'],
                        help='Directories containing test videos for inference')
    
    parser.add_argument('--model', type=str,
                        help='Path to saved model file (for inference mode)')
    
    parser.add_argument('--output', type=str,
                        default='./output',
                        help='Output directory for models, logs, and results')
    
    parser.add_argument('--cache', type=str,
                        default='./cache',
                        help='Directory for cached dataset')
    
    parser.add_argument('--force-recreate', action='store_true',
                        help='Force recreation of dataset from scratch')
    
    parser.add_argument('--test-split', type=float, default=0.1,
                        help='Test set split ratio (default: 0.1)')
    
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze latest inference results against weak labels')
    
    args = parser.parse_args()
    
    # Setup directories
    os.makedirs(args.output, exist_ok=True)
    
    # Setup logging
    log_dir = os.path.dirname(os.path.abspath(__file__))
    logger = setup_logging(log_dir)
    
    logger.info("=" * 60)
    logger.info("Violence Detection System Started")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Output directory: {args.output}")
    
    try:
        # Mode: Analyze only
        if args.mode == 'analyze':
            results_files = sorted([f for f in os.listdir(args.output) 
                                   if f.startswith('inference_results_') and f.endswith('.json')])
            if results_files:
                latest_results = os.path.join(args.output, results_files[-1])
                logger.info(f"Analyzing results from: {os.path.basename(latest_results)}")
                analyze_predictions(latest_results, args.output, logger)
            else:
                logger.error("No inference results found for analysis")
            return
        
        # Mode: Train or Full
        if args.mode in ['train', 'full']:
            # Load/create dataset
            features, labels, video_paths = load_or_create_dataset(
                args.dataset, args.cache, args.force_recreate, logger
            )
            
            if len(features) == 0:
                logger.error("No valid samples in dataset. Exiting.")
                return
            
            # Encode labels
            one_hot_labels = to_categorical(labels)
            
            # Split dataset
            logger.info(f"Splitting dataset (test_size={args.test_split})...")
            features_train, features_test, labels_train, labels_test = train_test_split(
                features, one_hot_labels,
                test_size=args.test_split,
                shuffle=True,
                random_state=42
            )
            
            logger.info(f"Training set: {features_train.shape}")
            logger.info(f"Test set: {features_test.shape}")
            
            # Create and train model
            model = create_model(logger)
            history = train_model(model, features_train, labels_train, args.output, logger)
            
            # Plot metrics
            plot_training_metrics(history, args.output, logger)
            
            # Evaluate
            evaluate_model(model, features_test, labels_test, args.output, logger)
            
            # Set model path for inference
            model_path = os.path.join(args.output, 'violence_detection_model.h5')
        
        # Mode: Inference or Full
        if args.mode in ['inference', 'full']:
            # Load model
            if args.mode == 'inference':
                if not args.model:
                    logger.error("--model argument required for inference mode")
                    return
                model_path = args.model
            
            logger.info(f"Loading model from: {model_path}")
            model = keras.models.load_model(model_path)
            logger.info("Model loaded successfully")
            
            # Run batch inference
            batch_inference(model, args.test_dirs, args.output, logger)
            
            if args.analyze:
                results_files = sorted([f for f in os.listdir(args.output) 
                                       if f.startswith('inference_results_') and f.endswith('.json')])
                if results_files:
                    latest_results = os.path.join(args.output, results_files[-1])
                    analyze_predictions(latest_results, args.output, logger)
                else:
                    logger.warning("No inference results found for analysis")
        
        logger.info("=" * 60)
        logger.info("Violence Detection System Completed Successfully")
        logger.info("=" * 60)
    
    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
