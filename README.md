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

### for smolvlm2 video on Jetson Nano (CUDA)

First, install `uv` if it is not already present (the setup script does this
automatically, but you can also run it manually):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.zshrc   # or ~/.bashrc / restart the shell
```

At the moment, direct `uv sync` on Jetson is blocked by an ABI mismatch:
NVIDIA's JetPack PyTorch index currently publishes only a `cp310` wheel for
`aarch64`.

Note: NVIDIA's Jetson PyTorch wheel is published as a prerelease-looking build
(`2.5.0a0+...nv...`). The project is configured to allow that build on
`aarch64` while using normal stable `torch` releases on other platforms.
Also note that `open-clip-torch` is excluded on Jetson for now because its
`torchvision` dependency does not resolve against NVIDIA's CUDA-enabled PyTorch
wheel. This does not affect the SmolVLM2 pipeline, but MobileCLIP is not
currently supported in this Jetson environment.

Current status for Jetson CUDA:

- NVIDIA publishes `torch-...-cp310-cp310-linux_aarch64.whl` for this index.
- This repo now supports Python `>=3.10,<3.14` and pins `.python-version` to `3.10`.
- `uv` will provision Python 3.10 and install the CUDA-enabled NVIDIA torch wheel on Jetson.
- Jetson torch wheel metadata has a known filename/version mismatch, so use
  `UV_SKIP_WHEEL_FILENAME_CHECK=1` for both `uv sync` and `uv run`.

```
export UV_SKIP_WHEEL_FILENAME_CHECK=1
uv sync
uv run python scripts/run_smolvlm2_pipeline.py \
  --device images/smoke_test.mp4 \
  --torch-device cuda \
  --max-new-tokens 128 \
  --clip-len 16 \
  --max-images 4
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
