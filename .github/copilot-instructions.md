# Copilot Instructions

## Commands

```bash
uv run ruff format src          # format
uv run ruff check src           # lint
uv run mypy src                 # type-check
uv run python scripts/<name>.py # run a pipeline script
```

There is no test suite. CI runs format + lint + mypy on push/PR to `main`.

## Architecture

Data flows in one direction: **Sensor → Pipeline → Stages → Message out**.

```
BaseSensor.read()
    └─► Message
            └─► Pipeline.run(msg)
                    ├─► Stage[0].process(msg, stage_idx=0, metrics=…)
                    ├─► Stage[1].process(msg, stage_idx=1, metrics=…)
                    └─► Stage[N] → final Message | None
```

**`Pipeline`** (`_pipeline.py`) holds an ordered `list[Stage]`. It drives
`enumerate` to assign each stage its index — stages do **not** store their own
index. Returning `None` from any stage short-circuits the rest.

**Messages** (`messages/`) are immutable Pydantic `BaseModel` subclasses. Every
`Stage.process()` call stamps `latency_ms` onto the result via `model_copy`.
`Message` is a `type` alias (Python 3.12 soft-keyword) over all concrete types.

**Hardware** (`hardware/`) is accessed exclusively through `ComputeBackend`.
Nothing outside `hardware/` should import LiteRT, ONNX, or SNPE directly.
`ComputeBackend` detects the platform at construction, picks the right
`InferenceBackend` subclass (`QCS6490Backend`), and delegates every call.
Model handles are opaque `object`s — the only place `Any` is permitted is
`_ModelHandle.raw`.

**Metrics** (`metrics/`) are entirely optional. Pass a `MetricsCollector` to
`Pipeline`; it flows through to each `Stage.process()` call automatically.
Types live in `_types.py`; `_collector.py` contains only `MetricsCollector`.

**Preprocessors** (`stages/_preprocess.py`) are generic `BasePreprocessor[InputT, OutputT]`.
Subclasses call `self._dispatch(fn, *args)` instead of `fn(*args)` to get
DSP/CPU routing for free when a Hexagon backend is added.

**`edgeperceive/`** is the original reference implementation. **Never modify it.**
It exists only as a historical reference.

## Conventions

### Python version & style
- Requires Python ≥ 3.12. Use 3.12 features freely: `type` aliases, `class Foo[T]` generics, `match` statements.
- Every file starts with `from __future__ import annotations`.
- Line length 100. Google-style docstrings. Ruff `select = ["ALL"]`.

### Imports
- Anything only needed by the type checker goes under `TYPE_CHECKING`.
- Exception: `Callable` used in a runtime signature must be imported at module level with `# noqa: TC003`.
- Circular imports are broken by importing from `_types.py` submodules, not from the top-level package.

### Data classes
- **`@attrs.define`** — mutable, slotted (replaces `@dataclass(slots=True)`).
- **`@attrs.frozen`** — immutable + slotted (replaces `@dataclass(frozen=True, slots=True)`).
- `attrs.Factory(dict)` replaces `field(default_factory=dict)`.
- Serialise with `attrs.asdict(...)`, not `dataclasses.asdict`.
- Pydantic `BaseModel` is used only for **messages** (pipeline data), not for config or metrics types.

### Type annotations
- Use `object` instead of `Any` everywhere except `_ModelHandle.raw`.
- `ParamSpec` + `Callable[_P, _R]` for functions that wrap arbitrary callables (see `ComputeDispatcher`).

### File layout patterns
- `_base.py` — abstract base class for a subsystem.
- `_types.py` — pure data types / enums for a subsystem (no logic).
- `__init__.py` — re-exports the public API; internal helpers stay private (`_`-prefixed).
- Platform-specific code lives under `hardware/_platforms/<chip>/`.

### Stage index
Stage index is assigned by `Pipeline` via `enumerate` and passed as `stage_idx`
to `Stage.process()`. It is **not** stored as a member variable on the stage.
