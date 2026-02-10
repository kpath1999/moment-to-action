import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re
from sklearn.metrics import confusion_matrix, classification_report

# Setup style for professional looking plots
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

# Define paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '../output')
SAVE_DIR = SCRIPT_DIR

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_rlvs_training_performance():
    """1. Analyzing MobileNetV2 + BiLSTM training performance on the RLVS dataset"""
    print("Generating RLVS Training Performance graphs...")
    history_path = os.path.join(DATA_DIR, 'training_history.json')
    
    if not os.path.exists(history_path):
        print(f"Warning: {history_path} not found. Skipping training graphs.")
        return

    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
    except Exception as e:
        print(f"Error reading {history_path}: {e}")
        return

    # Accuracy Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history['accuracy'], label='Training Accuracy', linewidth=2, linestyle='-')
    plt.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2, linestyle='--')
    plt.title('RLVS Training Performance: Accuracy\n(MobileNetV2 + BiLSTM)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim(0.4, 1.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'rlvs_training_accuracy.png'), dpi=300)
    plt.close()

    # Loss Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss', linewidth=2, linestyle='-')
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2, linestyle='--')
    plt.title('RLVS Training Performance: Loss\n(MobileNetV2 + BiLSTM)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'rlvs_training_loss.png'), dpi=300)
    plt.close()

def parse_and_plot_rlvs_report():
    """Visualize existing classification report for RLVS"""
    print("Generating RLVS Classification Report visualization...")
    report_path = os.path.join(DATA_DIR, 'classification_report.txt')
    
    if not os.path.exists(report_path):
        print(f"Warning: {report_path} not found. Skipping RLVS report viz.")
        return

    classes = []
    data = []
    
    try:
        with open(report_path, 'r') as f:
            lines = f.readlines()
            
        # Parse logic for standard sklearn classification_report text
        for line in lines:
            parts = line.split()
            # Look for lines with class names and metrics
            if len(parts) >= 5 and parts[0] in ['NonViolence', 'Violence']:
                classes.append(parts[0])
                # Precision, Recall, F1-score, Support
                data.append([float(x) for x in parts[1:5]])
                
        if classes:
            df = pd.DataFrame(data, columns=['Precision', 'Recall', 'F1-Score', 'Support'], index=classes)
            
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.axis('off')
            table = ax.table(
                cellText=df.values,
                rowLabels=df.index,
                colLabels=df.columns,
                cellLoc='center',
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.3)
            ax.set_title('RLVS Classification Metrics', pad=20)
            fig.tight_layout()
            fig.savefig(os.path.join(SAVE_DIR, 'rlvs_classification_report_table.png'), dpi=300)
            plt.close(fig)
        else:
            print("Could not parse class data from classification_report.txt")
            
    except Exception as e:
        print(f"Error processing RLVS report: {e}")

def analyze_rwf2000_performance():
    """2. Looking into the test performance of the .h5 model on the RWF2000 dataset"""
    print("Generating RWF2000 Performance analysis...")
    
    # Locate the JSON file
    target_json = 'inference_results_20260209_191627.json'
    json_path = os.path.join(DATA_DIR, target_json)
    
    if not os.path.exists(json_path):
        # Fallback to any inference json
        json_files = [f for f in os.listdir(DATA_DIR) if f.startswith('inference_results') and f.endswith('.json')]
        if json_files:
            json_path = os.path.join(DATA_DIR, sorted(json_files)[-1])
            print(f"Target JSON not found, using latest: {json_path}")
        else:
            print("Warning: No inference results JSON found. Skipping RWF2000 analysis.")
            return
    
    try:
        with open(json_path, 'r') as f:
            results = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return

    y_true = []
    y_pred = []
    
    for item in results:
        path = item.get('video_path', '')
        pred = item.get('predicted_class', '')
        
        # Infer Ground Truth from path (RWF naming convention)
        gt = None
        if 'NonFight' in path:
            gt = 'NonViolence'
        elif 'Fight' in path:
            gt = 'Violence'
        
        if gt and pred:
            y_true.append(gt)
            y_pred.append(pred)
            
    if not y_true:
        print("No valid data extracted from inference JSON.")
        return

    labels = ['NonViolence', 'Violence']
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(8, 6))
    heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                         xticklabels=labels, yticklabels=labels,
                         annot_kws={"size": 14})
    plt.title('RWF2000 Test Performance: Confusion Matrix\n(Model .h5 on RWF2000)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'rwf2000_confusion_matrix.png'), dpi=300)
    plt.close()
    
    # 2. Classification Report Visualization
    report_dict = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    
    # Create DataFrame for visualization
    report_data = []
    rows = []
    for key, value in report_dict.items():
        if key in labels or key in ['macro avg', 'weighted avg', 'accuracy']:
             rows.append(key)
             if key == 'accuracy':
                 # accuracy is a scalar in output_dict
                 report_data.append([value, value, value]) 
             else:
                 report_data.append([value['precision'], value['recall'], value['f1-score']])
    
    # Handle accuracy row structure differing in newer sklearn
    if 'accuracy' in report_dict and isinstance(report_dict['accuracy'], float):
        # Already handled above if we iterate, but 'accuracy' key might not be iterated well if we filter keys
        pass

    # Easier way to dataframe
    df_rep = pd.DataFrame(report_dict).transpose()
    # Filter rows we want
    target_rows = labels + ['macro avg', 'weighted avg']
    cols = ['precision', 'recall', 'f1-score']
    
    if 'accuracy' in df_rep.index:
        # metrics for accuracy are not prec/rec/f1 exactly, but let's just show the summary rows
        pass
        
    viz_df = df_rep.loc[target_rows, cols]
    
    fig, ax = plt.subplots(figsize=(8, 3))
    display_df = viz_df.copy()
    display_df['support'] = df_rep.loc[target_rows, 'support']
    for column in cols:
        display_df[column] = display_df[column].apply(lambda x: f"{x:.2f}" if pd.notna(x) else '-')
    display_df['support'] = display_df['support'].apply(lambda x: f"{int(x)}" if pd.notna(x) else '-')
    ax.axis('off')
    table = ax.table(
        cellText=display_df.values,
        rowLabels=display_df.index,
        colLabels=display_df.columns.str.title(),
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.3)
    ax.set_title('RWF2000 Classification Report Metrics', pad=20)
    ax.text(0, -0.25, f"Accuracy: {report_dict['accuracy']:.2f}", transform=ax.transAxes, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'rwf2000_classification_report_table.png'), dpi=300)
    plt.close(fig)
    
    # Print summary text
    print(f"RWF2000 Analysis Complete. Analyzed {len(y_true)} samples.")
    print("Metrics saved to plots.")

if __name__ == "__main__":
    ensure_dir(SAVE_DIR)
    print(f"Saving visualizations to: {SAVE_DIR}")
    
    plot_rlvs_training_performance()
    parse_and_plot_rlvs_report()
    analyze_rwf2000_performance()
    
    print("Done.")
