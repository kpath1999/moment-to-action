# Path Management

All filesystem paths in the application must go through the path managers. **Do not create app data directories manually.** This ensures consistent, platform-appropriate path handling and enables comprehensive tracking of all app state.

## PathManager

Top-level coordinator for all path operations.

**Location:** `src/moment_to_action/paths/_manager.py`

**Usage:**
```python
from moment_to_action.paths import PathManager

path_mgr = PathManager()
```

**Provides:**
- `path_mgr.cache` → `CacheManager` for cached data (models, etc.)
- `path_mgr.data` → `DataManager` for persistent app data
- `path_mgr.logs_dir` → Directory for log files
- `path_mgr.app_config_file` → Path to `config.json`

Platform-aware locations via `platformdirs`:
- Linux: `~/.cache/MomentToAction`, `~/.local/share/MomentToAction`, `~/.local/state/MomentToAction`
- macOS: `~/Library/Caches/MomentToAction`, `~/Library/Application Support/MomentToAction`, `~/Library/Logs/MomentToAction`
- Windows: `%APPDATA%\MomentToAction`, `%LOCALAPPDATA%\MomentToAction`

---

## CacheManager

Manages cached files (downloads, temporary data). Cache can be cleared without breaking app state.

**Location:** `src/moment_to_action/paths/_cache/_manager.py`

**Access:** `path_mgr.cache`

**Provides:**
- `cache.cache_dir` → Root cache directory
- `cache.models` → `ModelCacheManager` for ML model cache
- `cache.clear_cache()` → Removes all cached files

**Submanager:**
- `ModelCacheManager` handles downloaded ML models and their lifecycle

---

## DataManager

Manages persistent application data. **Never delete this directory.** It contains critical app state.

**Location:** `src/moment_to_action/paths/_data/_manager.py`

**Access:** `path_mgr.data`

**Provides:**
- `data.data_dir` → Root data directory
- `data.qairt_dir` → QAIRT-related data storage

**Adding a new data subdirectory:**
1. Add a property to `DataManager` that calls `mkdir(parents=True, exist_ok=True)`
2. Access via `path_mgr.data.<name>`

Example:
```python
@property
def qairt_dir(self) -> Path:
    self._qairt_dir.mkdir(parents=True, exist_ok=True)
    return self._qairt_dir
```

---

## ModelManager

Manages ML model discovery, caching, and resolution. Handles both vendored models (shipped with package) and downloaded models (HuggingFace Hub).

**Location:** `src/moment_to_action/models/_manager.py`

**Usage:**
```python
from moment_to_action.models import ModelManager

model_mgr = ModelManager()
path = model_mgr.get_path(ModelID.YOLO_V8)  # Auto-downloads if needed
```

**Provides:**
- `get_path(model_id)` → Path to model file (downloads from HF if needed)
- `is_available(model_id)` → Check without downloading
- `list_models()` → Status of all known models
- `clear_cache()` → Remove downloaded (non-vendored) models
- `cache_dir` → Where downloaded models are stored

**Model Sources:**
- **Vendored:** Included in package at `_vendored/` subdirectories
- **Downloadable:** Downloaded from HuggingFace Hub and cached locally

---

## Critical Rules

1. **All app paths go through managers.** No hardcoded `Path.home()`, `~/.config`, or direct mkdir calls for app directories.
2. **Never create manually.** Let managers initialize directories via `mkdir(parents=True, exist_ok=True)`.
3. **Track everything.** Path managers enable monitoring of what gets stored and where.
4. **Data is persistent.** Respect `data_dir` as application state; cache is ephemeral.

---

## Example: Adding a New Data Path

Don't do this:
```python
# ❌ Wrong
config_path = Path.home() / ".config" / "myapp" / "settings.json"
config_path.parent.mkdir(parents=True, exist_ok=True)
```

Do this:
```python
# ✓ Right
path_mgr = PathManager()
config_path = path_mgr.app_config_file  # Already exists, directory created
```

Or extend `DataManager`:
```python
# In DataManager class:
@property
def settings_dir(self) -> Path:
    settings = self._data_dir / "settings"
    settings.mkdir(parents=True, exist_ok=True)
    return settings

# Usage:
settings_path = path_mgr.data.settings_dir / "config.json"
```
