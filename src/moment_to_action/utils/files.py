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


def disk_size(path: Path) -> int:
    """Calculate the total size of a file or directory.

    Args:
        path: The file or directory to calculate the size of.

    Returns:
        The total size in bytes.
    """
    if path.is_file():
        return path.stat().st_size

    if path.is_dir():
        total_size = 0
        for item in path.iterdir():
            total_size += disk_size(item)
        return total_size

    msg = f"Path {path} is neither a file nor a directory."
    raise ValueError(msg)
