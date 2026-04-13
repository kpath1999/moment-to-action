# Moment2Action

An Edge-AI inference platform and framework.

## Getting Started

### Prerequisites

We use [`just`](https://just.systems) to run tasks in this repository. It works like Make, but
without the issues with dependencies when only running tasks. The task definitions can be found
in the [`justfile`](./justfile), and can be viewed by running `just` or `just help` in this
repository.

There are [many ways](https://github.com/casey/just#packages) to install `just`, but the simplest
is with Cargo. First install Rust and Cargo using [rustup.rs](https://rustup.rs/), then run
```bash
$ cargo install just
```

### Repo Setup

We use [`uv`](https://docs.astral.sh/uv/) for project management. Its is an improved version of all
other python package managers (pip, Poetry, etc.), written in Rust for performance.

To get started with the repository, run the following command. It will help you install `uv`,
then set up the rest of your repository.
```bash
$ just setup
```

## Contributing

To contribute code to this repository, do the following (assuming you've already set up the repo):
1. Create a branch off of `main` for your code, named `<your_name>/<feature_name>` (e.g.,
   `nikola/add-logging`).
2. Write your code and push it, *INCLUDING TESTS*. We would like to maintain 100% test coverage.
3. Open a PR to `main` for your branch, following the template that shows up.
4. Ensure all GitHub Actions pass for tests and linting.
5. Once approved, merge your code with a *SQUASH commit*.

## Terminal recs

### for smolvlm2 video

```
uv sync
uv run python scripts/run_smolvlm2_pipeline.py \
  --device images/smoke_test.mp4
```

### yolo plus reasoning

```
uv run python scripts/run_yolo_pipeline.py \
  --image images/weapon.jpg \
  --device cpu
```

### mobileclip

```
uv run python scripts/run_mobileclip_pipeline.py \
  --image images/fighting.jpg \
  --device cpu
```

## Benchmark module

The repository now includes an INFaaS-style benchmark subsystem under
`moment_to_action.benchmark` for profiling model variants across compute units
and storing queryable variant profiles.

### What it provides

- `BenchmarkHarness` to run one or many benchmarks.
- `ModelBenchmark` base class with a template `profile()` flow.
- Built-in benchmarks for:
  - `YOLOBenchmark`
  - `MobileCLIPBenchmark`
  - `SmolVLM2Benchmark`
  - `Qwen3Benchmark`
- `VariantRegistry` for persistent JSON storage and querying.

### Quick usage

```python
from moment_to_action.benchmark import (
    BenchmarkConfig,
    BenchmarkHarness,
    MobileCLIPBenchmark,
    Qwen3Benchmark,
    SmolVLM2Benchmark,
    VariantRegistry,
    YOLOBenchmark,
)
from moment_to_action.hardware import ComputeBackend, ComputeUnit
from moment_to_action.models import ModelID, ModelManager

backend = ComputeBackend(preferred_unit=ComputeUnit.CPU)
manager = ModelManager()
registry = VariantRegistry()

harness = BenchmarkHarness(backend=backend, manager=manager, registry=registry)
harness.register_benchmark(YOLOBenchmark())
harness.register_benchmark(MobileCLIPBenchmark())
harness.register_benchmark(SmolVLM2Benchmark())
harness.register_benchmark(Qwen3Benchmark())

config = BenchmarkConfig(n_warmup=3, n_runs=10, batch_sizes=[1])
profiles = harness.run_all(config=config)

# Query and persist
fastest_yolo = registry.best_variant(ModelID.YOLO_V8, objective="latency")
registry.save()  # saves to the default cache location
```

### Testing the benchmark module

Use the existing task helpers:

```bash
just test-unit
just lint
```

For benchmark-focused tests only:

```bash
uv run pytest tests/unit/benchmark
```

Notes:
- Unit tests for the benchmark module mock backends and model loaders, so they
  do not require real hardware accelerators or large model downloads.
- Real-world latency/accuracy numbers should be collected in your own runtime
  environment with the target hardware.
