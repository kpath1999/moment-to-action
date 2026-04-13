# Plan: INFaaS-Style Benchmark Harness & Variant Registry

## TL;DR

Add a `benchmark/` module that profiles model variants across compute engines (CPU/GPU/NPU) and stores INFaaS-style metadata in a persistent, queryable variant registry. Also add a Qwen3-4B-Instruct model entry to the existing model registry. The approach follows existing codebase conventions: `@attrs.frozen` for data types, `_types.py`/`_base.py` file layout, `object` over `Any`, Python 3.12 features, 100% test coverage.

Ideas for accuracy measurement:
* YOLO_V8: COCO 2017 val, mAP@[0.5:0.95]
* MOBILECLIP_S2: ImageNet-1k val, zero-shot top-1
* SMOLVLM2_2_2B: TextVQA val, VQA accuracy
* QWEN3_4B: IFEval, instruction-following score

---

## Phase 1: Qwen3 Model Registry Entry

**Goal**: Add `QWEN3_4B` to the existing model system so the benchmark harness can profile it.

### Steps

1. Add `QWEN3_4B = "qwen3_4b"` to `ModelID` enum in `models/_types.py`
2. Add registry entry to `MODEL_REGISTRY` in `models/_registry.py`:
   ```
   ModelID.QWEN3_4B → ModelInfo(
       id=ModelID.QWEN3_4B,
       filename="__UNUSED__",
       source=TransformersSource(hf_repo_id="Qwen/Qwen3-4B-Instruct-2507"),
   )
   ```
3. Update `tests/unit/models/test_types.py`:
   - Change `test_model_id_enum_count` assertion from 3 to 4
   - Add `QWEN3_4B` to parametrized test lists
   - Add `TestModelRegistry` entries for Qwen3 (source type, hf_repo, filename sentinel)

**Files modified**:
- `src/moment_to_action/models/_types.py` — add enum member
- `src/moment_to_action/models/_registry.py` — add dict entry
- `tests/unit/models/test_types.py` — update counts, add Qwen3 tests

---

## Phase 2: Benchmark Types & Foundation

**Goal**: Define the data model and abstract base class for the benchmark subsystem.

### Step 1: Create `benchmark/_types.py`

INFaaS-style variant profile data types, all `@attrs.frozen`:

- **`VariantID`**: Composite key `(model_id: ModelID, compute_unit: ComputeUnit)`. Hashable (frozen attrs), usable as dict key.
- **`CostProfile`**: `power_mw: float | None`, `energy_per_inference_mj: float | None`.
- **`VariantProfile`**: The core INFaaS profile record with:
  - `variant_id: VariantID`
  - `accuracy: float | None` — quality metric (mAP, accuracy, BLEU; model-specific, optional)
  - `load_latency_ms: float` — time to load model onto target hardware
  - `inference_mean_ms: float` — mean inference latency
  - `inference_p50_ms: float`
  - `inference_p95_ms: float`
  - `inference_p99_ms: float`
  - `peak_memory_mb: float` — peak RSS during inference
  - `max_batch_size: int` — largest batch the hardware can service
  - `hardware_target: str` — detected platform string (e.g., `"x86_64"`, `"qcs6490"`)
  - `cost: CostProfile`
  - `model_size_bytes: int` — model file/dir size on disk
  - `n_runs: int` — number of inference runs in the profiling session
  - `profiled_at: datetime` — ISO timestamp of when profiling was done
  - `json() -> dict[str, Any]` — JSON-serializable dict (following `Trace.json()` / `Span.json()` pattern)
- **`BenchmarkConfig`**: `n_warmup: int = 5`, `n_runs: int = 20`, `batch_sizes: list[int] = [1]` — configuration for a benchmarking session.

### Step 2: Create `benchmark/_base.py`

Abstract base class `ModelBenchmark`:

```
class ModelBenchmark(ABC):
    @property
    @abstractmethod
    def model_id(self) -> ModelID: ...

    def profile(
        self,
        backend: ComputeBackend,
        manager: ModelManager,
        config: BenchmarkConfig | None = None,
    ) -> VariantProfile:
        """Template method: load → warmup → benchmark → measure memory → return profile."""
        # 1. Measure load latency via _load_model()
        # 2. Run _warmup() iterations
        # 3. Run _run_inference() n_runs times, collect latencies
        # 4. Measure peak memory via psutil
        # 5. Call _evaluate_accuracy() (optional, returns None by default)
        # 6. Compute percentiles with numpy
        # 7. Probe max_batch_size via _probe_max_batch_size()
        # 8. Build and return VariantProfile

    @abstractmethod
    def _load_model(self, backend, manager) -> object: ...

    @abstractmethod
    def _make_dummy_input(self, handle: object, batch_size: int = 1) -> object: ...

    @abstractmethod
    def _run_inference(self, handle: object, inputs: object, backend: ComputeBackend) -> None: ...

    def _evaluate_accuracy(self, handle, backend, manager) -> float | None:
        return None  # override in subclasses that have quality metrics

    def _probe_max_batch_size(self, handle, backend, max_probe: int = 32) -> int:
        # Try increasing batch sizes via _make_dummy_input() until OOM/error
        ...
```

The template method `profile()` measures load latency, runs inference, samples memory (via `psutil.Process().memory_info()`), and assembles the `VariantProfile`. Uses `time.perf_counter()` matching all existing patterns.

**Files created**:
- `src/moment_to_action/benchmark/_types.py`
- `src/moment_to_action/benchmark/_base.py`

---

## Phase 3: Variant Registry (Persistence & Querying)

**Goal**: Build a persistent store for variant profiles that supports structured queries.

### Create `benchmark/_variant_registry.py`

Class `VariantRegistry`:

- **Storage**: In-memory `dict[VariantID, VariantProfile]` backed by a JSON file.
- **`register(profile: VariantProfile)`** — upsert a profile.
- **`get(variant_id: VariantID) -> VariantProfile | None`** — exact lookup.
- **`query(...) -> list[VariantProfile]`** — filter by any combination of:
  - `model_id: ModelID | None`
  - `compute_unit: ComputeUnit | None`
  - `max_latency_ms: float | None`
  - `min_accuracy: float | None`
  - `hardware_target: str | None`
- **`best_variant(model_id: ModelID, objective: str) -> VariantProfile | None`** — returns the optimal variant for an objective: `"latency"` (lowest mean), `"accuracy"` (highest), `"efficiency"` (lowest energy_per_inference).
- **`save(path: Path) -> None`** — serialize to JSON using `attrs.asdict()` (convention from copilot-instructions).
- **`load(path: Path) -> None`** — deserialize from JSON, rebuild `VariantID` keys.
- **`all_profiles() -> list[VariantProfile]`** — return all registered profiles.
- Default persistence path: `platformdirs.user_cache_path("moment_to_action", "GATech") / "variant_registry.json"` (matching `ModelManager` convention).

**Files created**:
- `src/moment_to_action/benchmark/_variant_registry.py`

---

## Phase 4: Concrete Model Benchmarks

**Goal**: Implement `ModelBenchmark` subclasses for each model family.

All four benchmarks follow the same `_load_model` / `_make_dummy_input` / `_run_inference` pattern. Grouped by loading mechanism:

### ComputeBackend-based (ONNX/TFLite)

**`benchmark/_yolo.py` — `YOLOBenchmark`**
- `_load_model`: `backend.load_model(manager.get_path(ModelID.YOLO_V8))`
- `_make_dummy_input`: `np.zeros((batch_size, 3, 640, 640), dtype=np.float32)` (matching `PreprocessorStage` target size)
- `_run_inference`: `backend.run(handle, input_tensor)`
- Reference: `YOLOStage.__init__()` and `YOLOStage._process()` for model loading and input format

**`benchmark/_mobileclip.py` — `MobileCLIPBenchmark`**
- `_load_model`: `backend.load_model(manager.get_path(ModelID.MOBILECLIP_S2))`
- `_make_dummy_input`: dict with `"serving_default_args_0:0"` (image `[1, 3, 256, 256]` float32) and `"serving_default_args_1:0"` (token `[1, 77]` int64) matching `MobileCLIPStage._process()`
- `_run_inference`: `backend.run(handle, inputs_dict)`

### Transformers-based

**`benchmark/_smolvlm2.py` — `SmolVLM2Benchmark`**
- `_load_model`: `AutoModelForImageTextToText.from_pretrained()` + `AutoProcessor.from_pretrained()` matching `SmolVLM2Stage.__init__()`
- `_make_dummy_input`: Processor-generated dummy inputs (small synthetic image + short prompt)
- `_run_inference`: `model.generate()` with `torch.inference_mode()`
- Uses `backend.resolve_torch_policy()` for device/dtype

**`benchmark/_qwen3.py` — `Qwen3Benchmark`**
- `_load_model`: `AutoModelForCausalLM.from_pretrained()` + `AutoTokenizer.from_pretrained()` from `Qwen/Qwen3-4B-Instruct-2507`
- `_make_dummy_input`: Tokenizer-generated dummy inputs (short prompt)
- `_run_inference`: `model.generate()` with `torch.inference_mode()`
- Uses `backend.resolve_torch_policy()` for device/dtype

**Files created**:
- `src/moment_to_action/benchmark/_yolo.py`
- `src/moment_to_action/benchmark/_mobileclip.py`
- `src/moment_to_action/benchmark/_smolvlm2.py`
- `src/moment_to_action/benchmark/_qwen3.py`

---

## Phase 5: Harness Orchestrator & Public API

**Goal**: Wire everything together into a usable harness.

### Create `benchmark/_harness.py`

Class `BenchmarkHarness`:
- `__init__(self, backend: ComputeBackend, manager: ModelManager, registry: VariantRegistry | None = None)`
- `register_benchmark(benchmark: ModelBenchmark)` — add a benchmark to the harness
- `run_all(config: BenchmarkConfig | None = None) -> list[VariantProfile]` — profile all registered benchmarks on the active compute unit, store results in registry
- `run_model(model_id: ModelID, config: BenchmarkConfig | None = None) -> VariantProfile` — profile a single model

### Create `benchmark/__init__.py`

Re-export public API following `__init__.py` convention:
- `BenchmarkConfig`, `BenchmarkHarness`, `CostProfile`, `ModelBenchmark`, `MobileCLIPBenchmark`, `Qwen3Benchmark`, `SmolVLM2Benchmark`, `VariantID`, `VariantProfile`, `VariantRegistry`, `YOLOBenchmark`

**Files created**:
- `src/moment_to_action/benchmark/_harness.py`
- `src/moment_to_action/benchmark/__init__.py`

---

## Phase 6: Tests

**Goal**: 100% coverage for all new code, following existing test patterns.

Test files (all under `tests/unit/benchmark/`):

| Test file | Covers | Key patterns |
|---|---|---|
| `test_types.py` | `VariantID`, `CostProfile`, `VariantProfile`, `BenchmarkConfig` | Frozen immutability, field storage, `json()` round-trip, equality checks |
| `test_base.py` | `ModelBenchmark` ABC, `profile()` template method | Concrete test subclass, mock backend/manager, verify latency/memory measurement |
| `test_variant_registry.py` | `VariantRegistry` register/query/save/load/best_variant | Temp file JSON round-trip, query filtering, edge cases (empty registry, no match) |
| `test_harness.py` | `BenchmarkHarness` run_all/run_model | Mock benchmarks, verify registry integration |
| `test_yolo.py` | `YOLOBenchmark` | Mock `ComputeBackend`, verify input shape, load path |
| `test_mobileclip.py` | `MobileCLIPBenchmark` | Mock `ComputeBackend`, verify dict input keys |
| `test_smolvlm2.py` | `SmolVLM2Benchmark` | Mock Transformers, verify torch policy usage |
| `test_qwen3.py` | `Qwen3Benchmark` | Mock Transformers, verify model loading |

All tests use `@pytest.mark.unit` marker. Mock `ComputeBackend`, `ModelManager`, `psutil.Process` as needed. Follow the heavy mocking patterns from `test_manager.py`.

**Files created**:
- `tests/unit/benchmark/__init__.py`
- `tests/unit/benchmark/test_types.py`
- `tests/unit/benchmark/test_base.py`
- `tests/unit/benchmark/test_variant_registry.py`
- `tests/unit/benchmark/test_harness.py`
- `tests/unit/benchmark/test_yolo.py`
- `tests/unit/benchmark/test_mobileclip.py`
- `tests/unit/benchmark/test_smolvlm2.py`
- `tests/unit/benchmark/test_qwen3.py`

Update existing test:
- `tests/unit/models/test_types.py` — Qwen3 assertions (Phase 1)

---

## Relevant files

**New files (18)**:
- `src/moment_to_action/benchmark/__init__.py` — public API re-exports
- `src/moment_to_action/benchmark/_types.py` — `VariantID`, `VariantProfile`, `CostProfile`, `BenchmarkConfig`
- `src/moment_to_action/benchmark/_base.py` — `ModelBenchmark` ABC with `profile()` template method
- `src/moment_to_action/benchmark/_harness.py` — `BenchmarkHarness` orchestrator
- `src/moment_to_action/benchmark/_variant_registry.py` — `VariantRegistry` with query/persist
- `src/moment_to_action/benchmark/_yolo.py` — `YOLOBenchmark` (uses `ComputeBackend`)
- `src/moment_to_action/benchmark/_mobileclip.py` — `MobileCLIPBenchmark` (uses `ComputeBackend`)
- `src/moment_to_action/benchmark/_smolvlm2.py` — `SmolVLM2Benchmark` (uses Transformers)
- `src/moment_to_action/benchmark/_qwen3.py` — `Qwen3Benchmark` (uses Transformers)
- `tests/unit/benchmark/__init__.py` + 8 test files

**Modified files (3)**:
- `src/moment_to_action/models/_types.py` — add `QWEN3_4B` to `ModelID` enum
- `src/moment_to_action/models/_registry.py` — add Qwen3 entry to `MODEL_REGISTRY`
- `tests/unit/models/test_types.py` — update count, add Qwen3 test cases

**Reference files** (read-only, patterns to follow):
- `src/moment_to_action/hardware/_backend.py` — `BenchmarkResult`, `ComputeBackend.benchmark()` for latency measurement pattern
- `src/moment_to_action/hardware/_types.py` — `ComputeUnit` enum, `ComputeUnitUsageSample` for resource pattern
- `src/moment_to_action/hardware/_platforms/_base.py` — `ResourceMonitor.used_memory_mb()` for memory sampling
- `src/moment_to_action/metrics/_types.py` — `Span.json()`, `Trace.json()` for serialization pattern
- `src/moment_to_action/stages/video/_yolo.py` — `YOLOStage.__init__()` for model loading reference
- `src/moment_to_action/stages/vlm/_mobileclip.py` — `MobileCLIPStage.__init__()` for multi-input model reference
- `src/moment_to_action/stages/vlm/_smolvlm2.py` — `SmolVLM2Stage.__init__()` for Transformers loading reference
- `tests/unit/models/test_manager.py` — mock patterns for `ModelManager`, HF hub

---

## Verification

1. `just test-unit` — all existing + new unit tests pass
2. `just lint` — ruff format, ruff check, mypy all pass
3. Verify model count: `ModelID` has 4 members, `MODEL_REGISTRY` has 4 entries
4. Verify `VariantProfile.json()` round-trips through `VariantRegistry.save()`/`load()`
5. Verify `VariantRegistry.query()` filters correctly on all supported fields
6. Verify `ModelBenchmark.profile()` measures load latency, inference latency, and peak memory
7. `just test` with `--cov` confirms no coverage regression

---

## Decisions

- **`@attrs.frozen` over Pydantic** for all benchmark types (Pydantic reserved for messages per conventions)
- **JSON persistence** for variant registry (not SQLite) — matches codebase simplicity; in-memory filtering sufficient for expected scale
- **Single `ModelBenchmark` ABC** rather than separate compute-backend vs. transformers base classes — fewer abstractions, each subclass owns its loading logic
- **`VariantID = (ModelID, ComputeUnit)`** as the composite key — simple, extensible later to include quantization/format
- **Scope includes**: benchmark module, variant registry, Qwen3 model entry, comprehensive tests
- **Scope excludes**: CLI commands for benchmarking (future), integration tests requiring real models/hardware, query optimizer implementation (the registry is the foundation for it)
- **Qwen3 entry**: Uses `TransformersSource` with `filename="__UNUSED__"` to match the `SMOLVLM2_2_2B` pattern
- **`hardware_target`**: Detected via existing `_detection.detect_platform()` — stored as `Platform.value` string

## Dependencies Between Phases

- Phase 1 is independent (can run first or in parallel with Phase 2)
- Phase 2 is independent of Phase 1
- Phase 3 depends on Phase 2 (needs `_types.py`)
- Phase 4 depends on Phases 1 and 2 (needs `ModelID.QWEN3_4B` + `_base.py`)
- Phase 5 depends on Phases 2, 3, 4
- Phase 6 tests should be written alongside each phase
