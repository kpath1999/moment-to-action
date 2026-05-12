from platformdirs import PlatformDirs

from moment_to_action._version import VERSION

from ._cache import CacheManager


class PathManager:
    """Manages paths for the Moment to Action project."""

    def __init__(self, app_name: str = "MomentToAction", author: str = "GeorgiaTech") -> None:
        self._dirs = PlatformDirs(appname=app_name, appauthor=author, version=VERSION)

        # Create managers
        self._cache_manager = CacheManager(self._dirs.user_cache_path)

    @property
    def cache(self) -> CacheManager:
        """Return the cache manager."""
        return self._cache_manager
