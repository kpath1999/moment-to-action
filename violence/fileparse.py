import cv2
import os
from pathlib import Path

def count_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def process_directory(dir_path):
    results = {}
    total_frames = 0
    video_count = 0
    
    path = Path(dir_path)
    if not path.exists():
        print(f"Directory not found: {dir_path}")
        return results, total_frames, video_count
    
    for file in path.glob("*"):
        if file.name.startswith('.'):
            continue
        if file.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']:
            try:
                frames = count_frames(str(file))
                results[file.name] = frames
                total_frames += frames
                video_count += 1
            except Exception as e:
                print(f"Error processing {file.name}: {e}")
    
    return results, total_frames, video_count

def main():
    base_path = "/Volumes/KAUSAR/kaggle/Real Life Violence Dataset"
    nonviolence_path = os.path.join(base_path, "NonViolence")
    violence_path = os.path.join(base_path, "Violence")
    
    print("="*60)
    print("Non-Violence Dataset")
    print("="*60)
    nv_results, nv_total, nv_count = process_directory(nonviolence_path)
    for filename, frames in nv_results.items():
        print(f"{filename}: {frames} frames")
    
    print("\n" + "="*60)
    print("Violence Dataset")
    print("="*60)
    v_results, v_total, v_count = process_directory(violence_path)
    for filename, frames in v_results.items():
        print(f"{filename}: {frames} frames")
    
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    print(f"Non-Violence Videos: {nv_count}")
    print(f"Non-Violence Total Frames: {nv_total}")
    if nv_count > 0:
        print(f"NonViolence Avg Frames per Video: {nv_total/nv_count:.2f}")
    
    print(f"\nViolence Videos: {v_count}")
    print(f"Violence Total Frames: {v_total}")
    if v_count > 0:
        print(f"Violence Avg Frames per Video: {v_total/v_count:.2f}")
    
    print(f"\nTotal Videos: {nv_count + v_count}")
    print(f"Total Frames: {nv_total + v_total}")
    if (nv_count + v_count) > 0:
        print(f"Overall Avg Frames per Video: {(nv_total + v_total)/(nv_count + v_count):.2f}")
    print("="*60)

if __name__ == "__main__":
    main()
