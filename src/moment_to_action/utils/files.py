"""File system utilities."""

from pathlib import Path


def clear_directory(path: Path) -> int:
    """Recursively clear all files in a directory and return the total size cleared.

    Args:
        path: The directory to clear.

    Returns:
        The total size of the cleared files in bytes.
    """
    total_size = 0
    for item in path.iterdir():
        if item.is_file():
            total_size += item.stat().st_size
            item.unlink()
        elif item.is_dir():
            total_size += clear_directory(item)

    # After clearing all contents, remove the now-empty directory
    path.rmdir()
    return total_size
