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

Data flows one direction through a **lazy generator chain**: **Sensor → Message stream → Pipeline → Stages → Message stream out**. Stages are generator transformers, not single-shot functions — a stage's `process()` pulls from its input iterator and yields to its output iterator, so the whole pipeline is one composed generator. Breaking out of the consumer loop propagates `GeneratorExit` up through every stage, aborting any in-flight upstream work (e.g. stops LLM token generation the moment a downstream stage stops pulling).

```
sensor.stream() ─► Iterator[Message] ─► Pipeline.run(source)
                                          ├─► Stage[0].process(stream)  (window/stride/ready/drop)
                                          ├─► Stage[1].process(stream)
                                          └─► Stage[N] → Iterator[Message]  (0..N per input)
```

### Core types

- **`Moment2Action`** (`app/`) — the single import consumers of this library need. Owns path/config/logging/`Platform` setup; `PathManager`, `ModelManager`, and the raw `Pipeline` are not exposed. `new_pipeline(name).add_stage(...).build()` constructs stages/models (each pipeline gets its own `MetricsCollector`/`ModelManager`) and returns a `PipelineHandle`; `load_pipeline(handle)`/`unload_pipeline(handle)`/`remove_pipeline(handle)`/`metrics_report(handle)` take that handle directly (default to the active pipeline where it makes sense), with only one pipeline loaded at a time. `get_pipeline(name)` looks a handle up by name. See `docs/app.md`.
- **`Pipeline`** (`pipeline.py`) — holds ordered `list[Stage]`; `run(source: Iterator[Message]) -> Generator[Message, None, None]` chains each stage's `process()` onto the previous stage's output and lazily yields from the last one. Wraps the whole drain in a `SpanType.PIPELINE` metrics span (stays open across the entire generator's lifetime). `metrics` is a constructor-only dependency — pass the same `MetricsCollector` instance used to construct every stage/model in the pipeline so spans nest under one trace.
- **`Stage`** (`stages/_base.py`) — ABC. `__init__(*, window=1, stride=None, ready=None, drop=None, metrics=None)` configures buffering: `window` is how many recent messages are buffered before `_process` runs; `stride` (defaults to `window`) gates how many new messages are required between subsequent emissions once the buffer is full; `ready(items) -> bool` fully overrides the count/stride emit check for custom conditions (e.g. scene boundaries); `drop(msg) -> bool` discards unwanted inputs before they buffer. Subclasses implement `_process(items: list[Message]) -> Iterator[Message]` (0..N outputs — this is what makes token streaming and multi-frame windowing possible without a separate buffering stage). The base `process()` owns windowing and opens a `SpanType.STAGE` span per emission; it does **not** post-stamp `latency_ms` on results (that would require draining a streamed multi-yield `_process`, which defeats streaming) — authoritative per-emission timing lives in the metrics report instead. `load()`/`unload()` are no-ops by default (overridden by model-backed stages).
- **`ModelStage[_ModelT]`** (`stages/_base.py`) — generic `Stage` subclass for stages that wrap exactly one model (`LLMStage`, `VLMDescriptionStage`, `ImageStage` and its subclasses `ImageDetectionStage`/`ImageClassificationStage`). Centralizes what every model-backed stage would otherwise repeat: `__init__` raises `ValueError` if the model is already loaded (models must be unloaded at stage construction — `Moment2Action` loads them later, uniformly, via `load_pipeline`), and `load()`/`unload()` delegate to the model. Subclasses parametrize with their concrete model type (e.g. `ModelStage[LlamaGGUFModel]`) so `self._model` keeps its specific type, and just call `super().__init__(model, ...)`.
- **Messages** (`messages/`) — immutable Pydantic `BaseModel` subclasses inheriting `BaseMessage` (`timestamp`, `latency_ms`). `messages.Message` is a `TypeAlias` union of every concrete message type — use it for `isinstance` and exhaustive `match`.
- **`MetricsCollector`** (`metrics/`) — constructor dependency threaded through `ModelManager` → models → `Stage`/`Pipeline`; `None` defaults to a per-instance `NullMetricsCollector` so everything stays standalone-constructable (useful for tests) without null-checks in stage/model code. `timed_stream(tokens, *, yn_predicate=None)` wraps a token generator, stamping `ttft_ms`/`mean_itl_ms`/`std_itl_ms` (and `ttfyd_ms` if `yn_predicate` fires) onto the currently open span in a `finally` block, so metrics are recorded even if the caller closes the generator early. Public types come from `_types.py` (`Span`, `SpanType`, `Trace`, `MetricsReport`); collector logic lives in `_collector.py`.
- **`ComputeBackend`** (`hardware/`) — the *only* allowed entry point to inference runtimes. Detects platform at construction, picks an `InferenceBackend` subclass under `hardware/_platforms/<chip>/`. Model handles are opaque `object`s. **Nothing outside `hardware/` may import LiteRT / ONNX / SNPE directly.**
- **`ModelManager`** + `MODEL_REGISTRY` (`models/`) — resolves model IDs to download sources (`HuggingFaceSource`, `VendoredSource`) and writes them into the model cache. Takes a `metrics` collector at construction and forwards it into every model built via `get_model()`.
- **`PathManager`** (`paths/`) — only legitimate way to get filesystem paths. Wraps `platformdirs` versioned dirs. `path_mgr.cache` → `CacheManager` (+ `models` submanager); `path_mgr.data` → `DataManager`; `path_mgr.logs_dir`; `path_mgr.app_config_file`. **Do not create app data directories manually.** See `docs/paths.md`.
- **`AppConfig`** (`config.py`) — pydantic model persisted as JSON at `path_mgr.app_config_file`. `load_config()` writes defaults on first run and re-normalizes on every load.

### CLI

Entry point `m2a` (`pyproject.toml` `[project.scripts]`) → `moment_to_action._cli:cli`.

- Built on `rich_click` with a custom **`AutoRichGroup`** (`_cli/_auto_group.py`) that auto-loads subcommands by globbing `cmd_*` files/dirs and importing the matching object (e.g. `cmd_config.py` must export `config`).
- Root callback in `_cli/__init__.py` constructs `PathManager`, loads `AppConfig`, inits logging, and stashes a `GlobalData(log, path_manager, config)` on `ctx.obj`. Subcommands retrieve it via `pass_global_data` or `get_global_data(ctx)`.
- New CLI commands: drop a `cmd_<name>.py` (or `cmd_<name>/` package) under `src/moment_to_action/_cli/commands/` exporting `<name>` as a `click.Command`/`Group`. No manual registration needed.

### Stages, by category

- `stages.image` — `ImageStage` (marker base), `ImageDetectionStage`, `ImageClassificationStage`. Both use `window=1` with a `drop` predicate that discards non-`RawFrameMessage` input and dropped frames (`frame is None`); `ImageDetectionStage` special-cases `EndOfClipMessage` in that predicate to let it through unchanged. Also `DetectionAggregationStage`: keeps a running highest-confidence-per-label accumulation as instance state (`window=1`, real time, no buffering, no fixed clip length) and flushes it as one `DetectionMessage` when an `EndOfClipMessage` arrives — promotes what used to be bench-local `_aggregate_detections` logic into a reusable stage.
- `stages.llm` — `LLMStage` (streams a `LlamaGGUFModel`'s response to a detection-derived prompt as `GenerationMessage` partials, splitting `<think>...</think>` via an internal `_ThinkResponseRouter`, then an `EndOfClipMessage`) and `DecisionStage` (a separate downstream stage — not a subclass — that reads a grammar-constrained `GenerationMessage` stream and emits a `DecisionMessage` the moment `YES`/`NO` is unambiguous, then `DecisionReasoningMessage` partials, then forwards the upstream `EndOfClipMessage` unchanged; a verdict-only sink can stop pulling right after `DecisionMessage` to abort the rest of generation). The question is **not** fixed at `LLMStage` construction — it comes from `DetectionMessage.question` on each incoming message, so one loaded model/stage instance serves any question.
- `stages.vlm` — `VLMDescriptionStage` (streams a `LlamaVLModel`'s response over base64-JPEG-encoded frames from a `RawFrameMessage`/`VideoClipMessage`; no think phase, always `type="response"`, terminated by an `EndOfClipMessage` same as `LLMStage`). Same message-borne-question design as `LLMStage`: the task comes from the incoming message's `question` field, not a constructor arg. Frame encoding lives in `stages/vlm/_encode.py`.
- `stages.video`, `stages.audio` — placeholders (`__all__ = []`); frame/clip windowing is base `Stage` config (`window=N, stride=…`), not a dedicated buffering stage.
- `prompting.YES_NO_GRAMMAR` — GBNF grammar forcing a leading `YES`/`NO` token; pass to `LLMStage(..., grammar=YES_NO_GRAMMAR)` so `DecisionStage` can read the verdict unambiguously as soon as it arrives. Both `LlamaGGUFModel.stream`/`LlamaVLModel.stream` accept an optional `grammar: str | None` forwarded straight into the llama.cpp `/completion` payload.
- **Bounded-stream sentinel, not `done`/`is_last` flags**: `messages.control.EndOfClipMessage` is a payload-free control message sent once, after the last real message in a bounded run (a clip's frames, or one prompt's generation), matched by `isinstance` — not a boolean field on every payload message. One type serves both cases: it carries no clip ID or prompt because a pipeline processes one bounded run to completion before the next starts, so there's never more than one sentinel live to disambiguate. A stage that must accumulate across such a run keeps the accumulation as instance state and flushes on the sentinel; a stage that doesn't care just never matches it.

## Conventions

### Style
- `from __future__ import annotations` at the top of every file.
- Line length 100. Google-style docstrings (`[tool.ruff.lint.pydocstyle].convention = "google"`).
- **Every function and method must have a docstring** describing its purpose, arguments, return value, and exceptions. Include `__init__` docstrings with all parameters. Use Google style: `Args:`, `Returns:`, `Raises:` sections.
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
- [`docs/app.md`](docs/app.md) — `Moment2Action` usage: building/loading/running/unloading pipelines, metrics reporting, and when you need to drive stages directly instead of one chained `run()`.

## Contributing flow

1. Branch off `main` as `<your_name>/<feature>` (e.g. `nikola/add-logging`).
2. Write code **and tests**; keep coverage at 100%.
3. `just lint && just test` clean locally.
4. Open PR to `main` using `.github/pull_request_template.md` **EXACTLY**; GitHub Actions runs lint + tests.
5. Merge as a **squash commit**.
