from pathlib import Path


class DataManager:
    """Manager for application data on disk."""

    def __init__(self, data_dir: Path) -> None:
        """Initialize the data manager with the given data directory.

        Args:
            data_dir: The directory where application data will be stored.
        """
        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Simple subpaths
        self._qairt_dir = self._data_dir / "qairt"

    @property
    def data_dir(self) -> Path:
        """Return the data directory."""
        return self._data_dir

    @property
    def qairt_dir(self) -> Path:
        """Return the directory for QAIRT data."""
        self._qairt_dir.mkdir(parents=True, exist_ok=True)
        return self._qairt_dir
