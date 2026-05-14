# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Dev tooling

Task runner is [`just`](https://just.systems); package/env manager is [`uv`](https://docs.astral.sh/uv/). One-time setup: `just setup` (installs `uv`, syncs deps, installs pre-commit). Python is pinned to `>=3.10,<3.11` in `pyproject.toml`.

Common recipes (see `justfile`):

```bash
just                       # list recipes
just format                # ruff format + ruff --fix
just lint                  # ruff format --check, ruff check, mypy (src/tests/scripts)
just test                  # full suite, includes slow, with coverage
just test-fast             # unit + integration, no slow
just test-unit             # unit only
just test-int              # integration only
just test-k <expr> [args]  # pytest -k <expr>
just coverage-html         # HTML coverage on http://localhost:8000
```

To run a single test: `just test-k "test_name"` or `uv run pytest path/to/test.py::test_name`. To run a script: `uv run python scripts/<name>.py`.

### Non-negotiable per-change checks

- **`just lint && just test` must both pass** before declaring work done. Lint is not optional — pre-commit runs format + lint + mypy; CI runs the same plus tests on push/PR to `main`.
- **100% coverage on every file in `src/`** must be maintained. If a touched commit drops a file below 100%, fix the gap — even in files you did not edit. Coverage omits test dirs (see `[tool.coverage.run]` in `pyproject.toml`).
- Ruff is `select = ["ALL"]` with a small ignore list; see `[tool.ruff.lint]` for the rationale on each ignore. Per-file ignores exist for `scripts/*` (allows `print`) and `tests/**` (allows `assert`, magic numbers, private access, etc.).
- mypy runs with `disallow_untyped_defs` and `check_untyped_defs` — type every def.

### Pytest setup

Markers (registered in `tests/conftest.py` and `pyproject.toml`):
- `unit` — fast, isolated
- `integration` — uses real models
- `slow` — heavyweight; **excluded by default** via `addopts = ["-m", "not slow"]`. Pass `-m ""` or use `just test` to include.

`pytest-randomly` randomizes order; `pytest-antilru` clears LRU caches between tests. Two warnings are filtered (protobuf gencode mismatch from TF vs. qairt-dev pin, and matplotlib pyparsing deprecation) — see `[tool.pytest.ini_options].filterwarnings`.

## Architecture

Data flows one direction: **Sensor → Message → Pipeline → Stages → Message out**.

```
BaseSensor.read() ─► Message ─► Pipeline.run(msg, metrics=…)
                                  ├─► Stage[0].process(msg, metrics)
                                  ├─► Stage[1].process(msg, metrics)
                                  └─► Stage[N] → Message | None
```

### Core types

- **`Pipeline`** (`pipeline.py`) — holds ordered `list[Stage]`. Any stage returning `None` short-circuits the rest. Wraps the whole run in a `SpanType.PIPELINE` metrics span.
- **`Stage`** (`stages/_base.py`) — ABC. Subclasses implement `_process(msg, metrics) -> Message | None`. The base `process()` wraps `_process` in a `SpanType.STAGE` span, stamps `latency_ms` on the result via `model_copy`, and logs. Stages do **not** store their index.
- **Messages** (`messages/`) — immutable Pydantic `BaseModel` subclasses inheriting `BaseMessage` (`timestamp`, `latency_ms`). `messages.Message` is a `TypeAlias` union of every concrete message type — use it for `isinstance` and exhaustive `match`.
- **`MetricsCollector`** (`metrics/`) — optional; `Pipeline`/`Stage` fall back to `NullMetricsCollector` when none is passed, so stage code never null-checks. Public types come from `_types.py` (`Span`, `SpanType`, `Trace`, `MetricsReport`); collector logic lives in `_collector.py`.
- **`ComputeBackend`** (`hardware/`) — the *only* allowed entry point to inference runtimes. Detects platform at construction, picks an `InferenceBackend` subclass under `hardware/_platforms/<chip>/`. Model handles are opaque `object`s. **Nothing outside `hardware/` may import LiteRT / ONNX / SNPE directly.**
- **`ModelManager`** + `MODEL_REGISTRY` (`models/`) — resolves model IDs to download sources (`HuggingFaceSource`, `VendoredSource`) and writes them into the model cache.
- **`PathManager`** (`paths/`) — only legitimate way to get filesystem paths. Wraps `platformdirs` versioned dirs. `path_mgr.cache` → `CacheManager` (+ `models` submanager); `path_mgr.data` → `DataManager`; `path_mgr.logs_dir`; `path_mgr.app_config_file`. **Do not create app data directories manually.** See `docs/paths.md`.
- **`AppConfig`** (`config.py`) — pydantic model persisted as JSON at `path_mgr.app_config_file`. `load_config()` writes defaults on first run and re-normalizes on every load.

### CLI

Entry point `m2a` (`pyproject.toml` `[project.scripts]`) → `moment_to_action._cli:cli`.

- Built on `rich_click` with a custom **`AutoRichGroup`** (`_cli/_auto_group.py`) that auto-loads subcommands by globbing `cmd_*` files/dirs and importing the matching object (e.g. `cmd_config.py` must export `config`).
- Root callback in `_cli/__init__.py` constructs `PathManager`, loads `AppConfig`, inits logging, and stashes a `GlobalData(log, path_manager, config)` on `ctx.obj`. Subcommands retrieve it via `pass_global_data` or `get_global_data(ctx)`.
- New CLI commands: drop a `cmd_<name>.py` (or `cmd_<name>/` package) under `src/moment_to_action/_cli/commands/` exporting `<name>` as a `click.Command`/`Group`. No manual registration needed.

### Stages, by category

- `stages.video` — `PreprocessorStage`, `YOLOStage`, `ClipBufferStage`
- `stages.vlm` — `MobileCLIPStage`, `SmolVLM2Stage`
- `stages.llm` — `ReasoningStage`
- `stages.audio` — placeholder
- `stages/_preprocess.py` — generic `BasePreprocessor[InputT, OutputT]`. Subclasses call `self._dispatch(fn, *args)` (not `fn(*args)`) so DSP/CPU routing works when a Hexagon backend lands.

## Conventions

### Style
- `from __future__ import annotations` at the top of every file.
- Line length 100. Google-style docstrings (`[tool.ruff.lint.pydocstyle].convention = "google"`).
- Use `object` instead of `Any` except where a runtime handle is genuinely opaque (`_ModelHandle.raw` is the one sanctioned `Any`).
- Type-checker-only imports go under `if TYPE_CHECKING:`.

### Data classes
- **`@attrs.define`** — mutable, slotted (replaces `@dataclass(slots=True)`).
- **`@attrs.frozen`** — immutable, slotted (replaces `@dataclass(frozen=True, slots=True)`).
- `attrs.Factory(dict)` instead of `field(default_factory=dict)`. Serialize with `attrs.asdict`.
- Pydantic `BaseModel` is reserved for **pipeline messages**, not config-internal types or metrics types.

### File layout
- `_base.py` — abstract base for a subsystem.
- `_types.py` — pure data types / enums for a subsystem (no logic).
- `__init__.py` — re-exports the public API; private helpers stay `_`-prefixed.
- Platform-specific code lives under `hardware/_platforms/<chip>/`.

### Avoiding circular imports
Import from `_types.py` submodules instead of from the top-level subpackage when wiring cross-module types.

## Docs

In-repo design docs live in `docs/`. Consult before touching the relevant subsystem and keep them in sync when behavior changes.

- [`docs/paths.md`](docs/paths.md) — `PathManager` / `CacheManager` / `DataManager` contract and platform-specific directory layout. Required reading before adding any filesystem path.

## Contributing flow

1. Branch off `main` as `<your_name>/<feature>` (e.g. `nikola/add-logging`).
2. Write code **and tests**; keep coverage at 100%.
3. `just lint && just test` clean locally.
4. Open PR to `main` using `.github/pull_request_template.md` **EXACTLY**; GitHub Actions runs lint + tests.
5. Merge as a **squash commit**.
