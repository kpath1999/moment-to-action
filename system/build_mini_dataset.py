#!/usr/bin/env python3
"""
Scan the Real Life Violence Dataset and copy only the videos that contain
an audio stream into <dataset>/mini/, preserving the Violence / NonViolence
sub-directory structure.

Usage:
    python3 build_mini_dataset.py \
        --dataset "/Volumes/KAUSAR/kaggle/Real Life Violence Dataset"

Optional flags:
    --dry-run        Print what would be copied without actually copying
    --workers N      Parallel ffprobe workers (default: 4)
"""

import argparse
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
SPLIT_DIRS = {"Violence", "NonViolence"}


def has_audio_stream(video_path: Path) -> bool:
    """Return True if the video file contains at least one audio stream."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return bool(result.stdout.strip())
    except subprocess.CalledProcessError:
        return False


def collect_candidates(dataset_root: Path) -> list[Path]:
    """Return all video files that live under a Violence or NonViolence sub-dir."""
    candidates = []
    for split in SPLIT_DIRS:
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            print(f"[WARN] Expected sub-directory not found: {split_dir}", file=sys.stderr)
            continue
        for p in split_dir.iterdir():
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
                candidates.append(p)
    return candidates


def get_directory_size(directory: Path) -> int:
    """Return total size in bytes of all files in the directory tree."""
    total_size = 0
    try:
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
    except (OSError, PermissionError):
        pass  # Skip files we can't access
    return total_size


def format_bytes(size_bytes: int) -> str:
    """Format bytes into human readable format (KB, MB, GB)."""
    if size_bytes == 0:
        return "0 B"
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def probe_and_copy(
    video_path: Path,
    dataset_root: Path,
    mini_root: Path,
    dry_run: bool,
) -> tuple[Path, bool]:
    """Check audio and, if present, copy to the mini directory tree."""
    if not has_audio_stream(video_path):
        return video_path, False

    # Reconstruct relative path: e.g. Violence/V_17.mp4
    relative = video_path.relative_to(dataset_root)
    dest = mini_root / relative

    if not dry_run:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(video_path, dest)

    return video_path, True


def main():
    parser = argparse.ArgumentParser(
        description="Copy audio-bearing videos from RLVS dataset into a mini/ sub-directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/Volumes/KAUSAR/kaggle/Real Life Violence Dataset",
        help="Path to the Real Life Violence Dataset root",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be copied; do not write any files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel ffprobe workers (default: 4)",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset).expanduser().resolve()
    if not dataset_root.is_dir():
        print(f"[ERROR] Dataset directory not found: {dataset_root}", file=sys.stderr)
        sys.exit(1)

    mini_root = dataset_root / "mini"
    if args.dry_run:
        print(f"[DRY RUN] Would create mini directory at: {mini_root}")
    else:
        mini_root.mkdir(parents=True, exist_ok=True)
        print(f"Mini directory: {mini_root}")

    candidates = collect_candidates(dataset_root)
    total = len(candidates)
    print(f"Found {total} video files across Violence / NonViolence directories.")

    if total == 0:
        print("Nothing to do.")
        return

    copied = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(probe_and_copy, p, dataset_root, mini_root, args.dry_run): p
            for p in candidates
        }

        for i, future in enumerate(as_completed(futures), start=1):
            video_path, had_audio = future.result()
            relative = video_path.relative_to(dataset_root)
            status = "COPY" if had_audio else "SKIP"
            label = "[DRY RUN] " if args.dry_run and had_audio else ""
            print(f"[{i:4d}/{total}] {label}{status}  {relative}")
            if had_audio:
                copied += 1
            else:
                skipped += 1

    print()
    print(f"Done.  Copied: {copied}  |  Skipped (no audio): {skipped}  |  Total: {total}")
    if not args.dry_run:
        print(f"Output: {mini_root}")
        # Check and report mini directory size
        if mini_root.exists():
            size_bytes = get_directory_size(mini_root)
            print(f"Mini directory size: {format_bytes(size_bytes)}")


if __name__ == "__main__":
    main()
