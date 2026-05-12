from pathlib import Path

from platformdirs import PlatformDirs

from moment_to_action._version import VERSION

from ._cache import CacheManager
from ._data import DataManager


class PathManager:
    """Manages paths for the Moment to Action project."""

    def __init__(self, app_name: str = "MomentToAction", author: str = "GeorgiaTech") -> None:
        self._dirs = PlatformDirs(
            appname=app_name, appauthor=author, version=VERSION, ensure_exists=True
        )

        # Create managers
        self._cache_manager = CacheManager(self._dirs.user_cache_path)
        self._data_manager = DataManager(self._dirs.user_data_path)

    @property
    def cache(self) -> CacheManager:
        """Return the cache manager."""
        return self._cache_manager

    @property
    def data(self) -> DataManager:
        """Return the data manager."""
        return self._data_manager

    @property
    def logs_dir(self) -> Path:
        """Return the directory for log files."""
        return self._dirs.user_log_path

    @property
    def app_config_file(self) -> Path:
        """Return the path to the application config file."""
        return self._dirs.user_config_path / "config.json"
