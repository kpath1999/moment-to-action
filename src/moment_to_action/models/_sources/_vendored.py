from pathlib import Path

import attrs

from moment_to_action.models._formats import ModelFormat


@attrs.frozen
class VendoredSource:
    """Model files included directly in the repository."""

    format: ModelFormat
    """Format of the model file (e.g. ONNX, TFLite)."""

    path: Path
    """Path to the model (file or directory) within the _vendored/ directory."""


def resolve_vendored_source(source: VendoredSource) -> Path:
    """Resolve the path of a vendored source to an absolute path on disk."""
    # _vendored.py is at src/moment_to_action/models/_sources/_vendored.py
    # _vendored/ dir is at src/moment_to_action/models/_sources/_vendored/
    #
    # the first parent is _sources, the second parent is models
    # then, we append _vendored/ and the source path to get the full path to the model files
    base_dir = (Path(__file__).parent.parent / "_vendored").resolve()
    return base_dir / source.path
