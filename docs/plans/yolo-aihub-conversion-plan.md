# Plan: Ultralytics source, AI Hub export, per-backend context binaries

**Status:** proposed
**Author:** generated 2026-06-10
**Scope:** YOLOv8 conversion overhaul + general per-backend context-binary infra.

This document is self-contained. It captures the investigation that motivated the
work, the decisions already locked, and a detailed, file-by-file implementation
plan for the six requested changes.

---

## 0. Background (why this work exists)

`m2a model run yolo_v8 --variant qcs6490 ... --backend cpu` returned **zero
detections** on every machine. Root cause, proven exhaustively:

- The DLC class scores collapse to ~0 (`sigmoid`/`cls` max **0.0036**, near-constant).
- This happens because **`qairt.convert` (the QAIRT 2.45 Python / AIMET INT8
  quantizer)** mis-quantizes YOLOv8's detection head: the head logits come out
  all-negative, so every class probability sigmoids to ~0. A handful of "hot"
  anchors among 8400 get treated as outliers and crushed.
- This is **independent** of: ONNX surgery (minimal split vs our heavy
  `boxes/cls/class_idx`), input layout, calibration method (min-max / percentile /
  mse / sqnr), bias bitwidth (8/32), per-channel on/off, and CPU-vs-HTP. All collapse.
- `qairt-quantizer` (classic CLI) collapses identically. Only the **legacy
  `qnn-onnx-converter`** quantizer survives (sigmoid 0.757, real detections) — but
  its DLLs do not load in the user's environment.
- mobilenet works through `qairt.convert` because a plain classifier head is not
  sensitive the way the detection head is.

### The fix that works: Qualcomm AI Hub

`qai_hub_models` runs Qualcomm's **production cloud quantizer** and returns a
portable `.dlc`. Verified on x86 QNN CPU backend:

```
qairt.load("yolov8_det.dlc") → handle(inputs=x).data
KEYS: boxes (1,8400,4), scores (1,8400), class_idx (1,8400)   [float32]
scores max 0.891, 17 anchors >0.25  →  person 0.891, bbox [176,124,303,455]
```

Healthy scores. And the output names/shapes are **exactly** what
`YOLOModel._decode` already expects. See
`memory/project_yolo_qairt_int8_collapse.md` for the full evidence trail.

### Key AI Hub facts (memorize these — they bit us)

- Token lives in project `.env` as `QAI_HUB_API_TOKEN` (gitignored). Configure the
  CLI once: `qai-hub configure --api_token <token>`.
- Install deps via the model extra: `uv run --with "qai-hub-models[yolov8-det]"`
  (pulls `ultralytics`, `pycocotools`, `aiofiles`, `scikit-image`, ...).
- **Do NOT pass `--num-calibration-samples`** — their parser leaves it a `str`,
  causing `TypeError: 'str' // 'int'` inside `get_calibration_data`.
- The CLI sometimes exits cleanly after the COCO download without submitting the
  quantize job. Re-run; compile/quantize jobs are cached server-side.
- Working command:
  ```bash
  uv run --with "qai-hub-models[yolov8-det]" \
    python -m qai_hub_models.models.yolov8_det.export \
    --precision w8a8 --target-runtime qnn_dlc --chipset qualcomm-qcs6490 \
    --skip-profiling --skip-inferencing --skip-summary --output-dir <out>
  ```
- Output artifact: `<out>/yolov8_det-qnn_dlc-w8a8/{yolov8_det.dlc, labels.txt, metadata.json}`.

### AI Hub YOLOv8 I/O contract (drives the NHWC change)

From `metadata.json` of the produced DLC:

- **Input** `image`: shape **`[1, 640, 640, 3]` — NHWC**, dtype uint8, quant
  scale `1/255`, value_range `[0,1]`. We feed **float32 NHWC in `[0,1]`** and qairt
  quantizes internally (verified). This is the single biggest behavioral change:
  our current `YOLOModel.prepare()` emits **NCHW `[1,3,640,640]`**.
- **Outputs** (qairt returns them dequantized as float32):
  - `boxes` `[1, 8400, 4]` — **x1,y1,x2,y2 in 640-space** (already decoded).
  - `scores` `[1, 8400]` — max class prob per anchor (reduction done in-graph).
  - `class_idx` `[1, 8400]` — argmax class per anchor.
- Consequence: **no ONNX surgery and no Python `cls.max`** for this variant. AI
  Hub emits the three tensors directly; `run()` returns
  `[boxes, scores, class_idx]` straight from `infer_dlc`; `_decode` is unchanged.

---

## 1. Decisions locked

1. **Artifact strategy: per-backend context `.bin` + `.dlc` fallback.**
   Variant directory layout:
   ```
   <model>/<variant>/
     model.dlc          # portable master — verify, debug, fallback
     model.cpu.bin      # AOT QNN context binary, CPU backend
     model.gpu.bin      # AOT QNN context binary, GPU backend
     model.npu.bin      # AOT QNN context binary, HTP backend (qcs6490-pinned)
     reference_outputs/ # for `m2a model verify`
   ```
   Loader tries `model.<backend>.bin`; if absent, falls back to `model.dlc`.
   Rationale: NPU/HTP graph-prepare is seconds *per load* from a `.dlc`; an AOT
   context binary loads in ms. CPU/GPU prepare is cheap so the `.dlc` fallback is
   fine there. HTP context binaries are SoC-specific (qcs6490 only) — acceptable,
   that is our only NPU target.

2. **YOLOv8 default source = Ultralytics export** (replaces the vendored ONNX),
   mirroring `../m2a-models/cli/src/m2a_models_cli/sources/ultralytics.py`.

3. **YOLOv8 `qcs6490` quantized variant = AI Hub `w8a8` DLC**, *not* the
   `qairt.convert` output (which collapses). The AI Hub path supersedes the local
   convert path for YOLO. Keep `m2a model convert` for models where `qairt.convert`
   works (e.g. mobilenet) and for the task-5 general converter.

4. **Per-backend context binaries come from AI Hub when available.** AI Hub compiles
   the device (qcs6490) artifacts in the cloud — **no local cross-compile, no `.so`,
   no `qnn-*` DLLs.** The `qcs6490` "cpu/gpu/npu" backends are all compute units *on
   the qcs6490 device* (aarch64 CPU / Adreno GPU / Hexagon HTP), not x86. The
   **non-AI-Hub fallback is the existing `m2a model convert`** (local `qairt.convert`
   → `.dlc`), which stays for models where that quantizer works (e.g. mobilenet). We
   add a **warning** to `convert` about its limits (mis-quantizes some heads; emits
   only a `.dlc`, no context bins). No separate stub function — local context-binary
   generation, if ever needed, is a future evolution of `convert` itself. The full
   local recipe is documented in §5a-fallback for that future work.

5. **`qai_hub_models` goes in the existing `[host]` extra** (alongside `qairt-dev`),
   with an explanatory comment. Not a core dep.

---

## 2. Task-by-task plan

### Task 1 — Ultralytics source for YOLOv8 default

**Goal:** download + ONNX-export YOLOv8 from `ultralytics` at resolve time instead
of shipping a vendored ONNX.

**New file:** `src/moment_to_action/models/_sources/_ultralytics.py`

```python
@attrs.frozen
class UltralyticsSource:
    format: ModelFormat          # ModelFormat.ONNX
    name: str                    # e.g. "yolov8n"
    filename: str = "model.onnx" # local storage name under variant_dir

def resolve_ultralytics_source(
    source, variant_dir, *, download=False, progress=True
) -> Path | None:
    target = variant_dir / source.filename
    if target.exists():
        return target
    if not download:
        return None
    from ultralytics import YOLO            # lazy import (optional dep)
    yolo = YOLO(f"{source.name}.pt")
    exported = Path(yolo.export(format="onnx", dynamic=False))
    target.parent.mkdir(parents=True, exist_ok=True)
    exported.replace(target)
    Path(f"{source.name}.pt").unlink(missing_ok=True)
    return target
```

**Wire-up:**
- `src/moment_to_action/models/_sources/__init__.py`: add `UltralyticsSource` to the
  `ModelSource` union, add a `case UltralyticsSource()` to `resolve_model_source`,
  export it in `__all__`.
- `src/moment_to_action/models/_registry.py`: change `ModelID.YOLO_V8` `DEFAULT_KEY`
  from `VendoredSource(... yolo/model.onnx)` to
  `UltralyticsSource(format=ModelFormat.ONNX, name="yolov8n")`.
- Delete the vendored ONNX once unreferenced:
  `src/moment_to_action/models/_sources/_vendored/yolo/model.onnx`
  (confirm nothing else references it first).

**Dependencies:** `ultralytics` is heavy (torch). Make it an **optional extra**, not
a core dep:
- `pyproject.toml`: add `[project.optional-dependencies] yolo-export = ["ultralytics>=8.3"]`
  (or fold into a broader `convert`/`export` extra).
- Lazy-import inside the resolver; raise a clear error if missing
  (`"install with: uv sync --extra yolo-export"`).

**Gotchas / decisions:**
- Ultralytics `export(format="onnx")` produces a **single `output0` Concat** graph
  (boxes + cls mixed). That is fine for the *default* (ONNX/CPU) runtime — ONNX
  Runtime gives correct scores (0.895). It only matters for the local
  `qairt.convert` DLC path (which we are abandoning for YOLO). Confirm the default
  `YOLOModel.prepare`/`run`/`decode` still work against the fresh ultralytics ONNX
  output names (it may export `output0 (1,84,8400)` rather than the current
  `boxes/scores/class_idx`). **If the ultralytics ONNX output differs from what
  `YOLOModel.run` (ONNX path) expects, add a tiny post-export reshape or update
  `run()`'s ONNX branch.** Verify with `m2a model run yolo_v8 images/pedestrian.jpg
  --backend cpu` after the swap.
- Pin the ultralytics model version for reproducibility (record `yolov8n` weights
  release in a comment / lockfile note).

**Tests:** unit test the resolver (mock `ultralytics.YOLO`), `download=False`
returns None when absent, returns path when present. Keep `src/` at 100% coverage.

---

### Task 2 — `m2a model convert-aihub` command

**Goal:** a CLI command that produces the AI Hub `w8a8` DLC (+ context binaries) for
a registered model.

**Command:** `m2a model convert-aihub <model_id> [opts]`

**New file:** `src/moment_to_action/_cli/commands/cmd_model/cmd_convert_aihub.py`
exporting **`convert_aihub`** (a `click.Command`; auto-loaded by `AutoRichGroup`,
which maps the `convert_aihub` underscore name to the `convert-aihub` command).

**Design — thin orchestrator over `qai_hub_models`:**
- `qai_hub_models` is a very heavy optional dep (torch + per-model extras). It lives
  in the **`[host]` extra** (DECIDED), next to `qairt-dev`, with a comment:
  ```toml
  # Qualcomm AI Hub — cloud quantize/compile for models that the local QAIRT
  # quantizer mis-handles (e.g. YOLOv8 detection head). Heavy (torch); host-only.
  "qai-hub-models>=0.40",
  ```
  Per-model export deps (ultralytics, pycocotools, ...) come from the model's own
  extra and are pulled at call time.
- **Call shape:** import `qai_hub_models` **lazily inside the command** (so importing
  the CLI never drags in torch). Reuse the proven flow. We hit a CLI quirk where
  `export_model` exits after COCO download without submitting the quantize job, so
  prefer driving the job functions directly (`compile_model` → `quantize_model` →
  `compile_model` → `download`) with explicit `.wait()`/`get_target_model()`, which
  we verified works. Lazy import keeps the `[host]`-only boundary clean.

**Command surface:**
```
m2a model convert-aihub <model_id>
  --precision w8a8            (default; choices w8a8/w8a16/float)
  --runtime qnn_dlc           (default; we post-process to per-backend .bin in task 5)
  --chipset qualcomm-qcs6490  (default)
  --output-dir <dir>          (default: a temp/build dir)
```
- Read token from `.env` / `QAI_HUB_API_TOKEN`; fail with a clear message + signup
  URL if absent. Ensure `qai-hub configure` has run (or set the token via the
  `QAI_HUB_CLIENT_INI`/env the SDK reads).
- Maintain a mapping `ModelID -> qai_hub_models module id` (e.g.
  `YOLO_V8 -> "yolov8_det"`) and the pip extra (`"yolov8-det"`). Keep it small and
  explicit.
- Bake in the gotchas: never pass `--num-calibration-samples`; if the subprocess
  exits 0 with no downloaded artifact, retry once (cached jobs).
- On success, copy `<out>/<...>/yolov8_det.dlc` to
  `<output-dir>/model.dlc` and keep `metadata.json`/`labels.txt`.

**Tests:** mock the subprocess / `export_model`; assert correct args, token-missing
error path, artifact relocation. 100% coverage.

---

### Task 3 — Produce the YOLOv8 artifact and publish it

**One-time data op (not code):**
1. Run the task-2 command (or the raw working command in §0) to get
   `yolov8_det.dlc` for `qualcomm-qcs6490`.
2. Get the per-backend **device** context binaries **from AI Hub** (task 5a): at
   minimum `model.npu.bin` (HTP); also `model.cpu.bin`/`model.gpu.bin` if AI Hub can
   target those compute units on the device. Whatever AI Hub does not emit simply
   falls back to `model.dlc` via the loader — acceptable. No local cross-compile.
3. Capture `reference_outputs/` (inputs + the 3 outputs) for `m2a model verify`.
   - NOTE: reference captured with **NHWC** inputs and AI Hub output order
     `boxes(0)/scores(1)/class_idx(2)`.
4. Place under `../m2a-models/` (existing `yolo_qcs/` dir or new `yolo_aihub/`);
   **`git add` + commit + push**. (DECIDED: `../m2a-models` git push is the publish —
   it mirrors the HF repo the registry reads. Commit+push is sufficient; no separate
   `hf upload`.)
5. Update `src/moment_to_action/models/_registry.py` `ModelID.YOLO_V8`
   `"qcs6490"` variant: new `hf_subdir`, `files=[model.dlc, model.npu.bin,
   (model.cpu.bin/model.gpu.bin if produced), reference_outputs/...]`, and the new
   `revision` (the m2a-models commit hash).

**Commit/push:** only after the user confirms (outward-facing). Follow the m2a-models
repo conventions; do not add attribution lines.

---

### Task 4 — Hash/integrity check in the HF "available" check

**Today:** `resolve_hugging_face_source` only checks **file existence**. A pinned
`revision` gives upstream reproducibility but does not catch local corruption /
partial downloads, and `is_available` can report a half-written file as present.

**Goal:** verify each file's content hash, re-download on mismatch.

**Approach (preferred): HF etag via HEAD.**
- HF already exposes per-file metadata: `get_hf_file_metadata(hf_hub_url(...))`
  returns `.etag`. For LFS blobs the etag is the **sha256**; for small files it is
  the git blob sha. We already call `get_hf_file_metadata` for size during download.
- On download: write a sidecar `<file>.etag` (and/or a single
  `<variant_dir>/.hashes.json` manifest mapping `relpath -> etag`).
- On availability check (`resolve_hugging_face_source` with `download=False` and in
  `ModelManager.is_available`): for each file, compare the stored sidecar etag
  against a freshly-fetched HEAD etag **when online**; if mismatch → treat as
  missing (so a subsequent `download=True` re-fetches). When **offline / HEAD
  fails**, fall back to existence + (optionally) a local recompute of the sha256 vs
  the sidecar to catch corruption without network.

**Fallback (if HEAD/etag proves unreliable):** write a `.version` / `.hash` file
keyed by `revision` containing expected sha256 per file (computed at publish time,
stored alongside the registry entry). Check local file sha256 against it. This is
fully offline and deterministic; the cost is computing/storing hashes at publish.

**Implementation notes:**
- Put a small helper `verify_files(variant_dir, expected: dict[str,str]) -> set[str]`
  returning the set of files that are missing/mismatched.
- Keep the network call **best-effort**: never make `is_available` hang offline.
- Add `expected_hashes: dict[str,str] | None = None` to `HuggingFaceSource` for the
  deterministic fallback path; when set, it is authoritative and needs no network.

**Tests:** corrupt a cached file → check reports unavailable → re-download fixes it;
offline path uses sidecar; etag-mismatch triggers re-fetch (mock
`get_hf_file_metadata`). 100% coverage.

---

### Task 5 — Context-binary converter + per-backend loader helper

Two pieces: **(5a)** a converter that emits per-backend `.bin`, and **(5b)** a
runtime helper that loads the right file for a requested backend.

#### 5a. Per-backend context binaries — from AI Hub (DECIDED)

We do **not** generate context binaries locally. Reasons (locked):
- Local generation needs `qnn-onnx-converter` → `qnn-model-lib-generator` (a `.so`,
  **cross-compiled to `aarch64-oe-linux`** for the device) → a **device-built HTP
  context** — painful and DLL-fragile in the user's env.
- The pure-Python `qairt.load(dlc).compile(...).save(...)` route was rejected: it
  goes through the same QAIRT quantizer/compile stack that already failed for us, and
  it would still need the cross-compiled `.so` for the device target.
- AI Hub compiles all device artifacts **in the cloud**, for the exact qcs6490
  backends, with zero local toolchain. → "use AI Hub whenever possible."

**Mechanism:** extend the task-2 AI Hub command to also request context binaries:
- `--target-runtime qnn_context_binary` (runs the link/AOT step) for the qcs6490
  device → produces a device context `.bin`.
- Produce one per requested **compute unit** if AI Hub exposes backend selection in
  `compile_options` (HTP is the default/primary; CPU/GPU on-device may need a
  `--compute_unit`/QNN-backend option — verify in AI Hub's
  `get_hub_compile_options`). At minimum ship **`model.npu.bin`** (HTP); let cpu/gpu
  fall back to `model.dlc` via the loader.
- Rename downloaded artifacts to `model.<unit>.bin` in the variant dir.

This makes task 5 almost entirely the **loader helper (5b)** plus a few extra flags
on the task-2 command. No `QairtSDKManager` context-gen, no `.so`, no cross-compile.

#### 5a-fallback. Non-AI-Hub path = the existing `m2a model convert` (with a warning)

No separate stub. The local fallback for models not on AI Hub **is** `m2a model
convert` — it already does the local `qairt.convert` → `.dlc`. The future "local
context-binary generation" is just an evolution of that same command, so a dedicated
`generate_context_binaries` stub would be redundant. (DECIDED: drop the stub.)

Instead, **add a warning to `m2a model convert`** so its limitations are explicit:

```python
# in cmd_convert.py, near the top of convert()
click.echo(click.style(
    "warning: local convert uses the QAIRT INT8 quantizer, which mis-handles some "
    "models (e.g. the YOLOv8 detection head collapses to ~0 scores) and emits only a "
    "portable .dlc — no per-backend context binaries. For AI Hub-supported models "
    "prefer 'm2a model convert-aihub'.",
    fg="yellow",
))
```

- Keep the full local context-binary recipe (qnn-onnx-converter →
  qnn-model-lib-generator cross-compiled to `aarch64-oe-linux` →
  qnn-context-binary-generator per QNN backend lib; HTP needs a device-built context;
  env = venv py3.10, `QNN_SDK_ROOT`/`QAIRT_SDK_ROOT`,
  `LD_LIBRARY_PATH=<uv-libpython-dir>:$SDK/lib/...`, `PYTHONPATH=$SDK/lib/python`)
  **here in this doc** for whoever later extends `convert`. We validated it manually;
  see `memory/project_yolo_qairt_int8_collapse.md`.
- No new module, no `NotImplementedError` stub, no extra test scaffolding.

#### 5b. Loader helper (runtime, used by every model)

**New helper** (e.g. `src/moment_to_action/models/_artifacts.py` or on the model
base): given a `variant_dir` and a `ComputeUnit`, return the artifact path:

```python
_BIN_BY_UNIT = {ComputeUnit.CPU: "model.cpu.bin",
                ComputeUnit.GPU: "model.gpu.bin",
                ComputeUnit.NPU: "model.npu.bin",
                ComputeUnit.DSP: "model.npu.bin"}

def resolve_backend_artifact(variant_dir: Path, unit: ComputeUnit) -> Path:
    cand = variant_dir / _BIN_BY_UNIT[unit]
    if cand.exists():
        return cand
    dlc = variant_dir / "model.dlc"
    if dlc.exists():
        return dlc
    raise FileNotFoundError(...)
```

- Wire into the DLC-loading model path. Currently `YOLOModel.load` calls
  `backend.load_model_dlc(self._path / "model.dlc")`. Change it to
  `backend.load_model_dlc(resolve_backend_artifact(self._path, backend.preferred_unit))`.
  `ComputeBackend.preferred_unit` already exists.
- `qairt.load` already accepts **both** `.dlc` and context `.bin` and returns a
  callable handle (`Model` or `CompiledModel`); `infer_dlc` does `handle(inputs=x).data`
  for both. A `CompiledModel` may need `.initialize(backend=...)` — handle that in
  `load_model_dlc` (try/branch on type). Verify `unload`/`destroy` works for both.
- This helper is **reused by future models** (mobilenet etc.), so keep it
  model-agnostic and put it next to the source resolvers.

**Tests:** helper picks `.bin` when present per unit, falls back to `.dlc`, raises
when neither; `load_model_dlc` handles `CompiledModel` init. 100% coverage.

---

### Task 6 — `--variant` flag on `verify`

**File:** `src/moment_to_action/_cli/commands/cmd_model/cmd_verify.py`

**Today:** verify hardcodes `DEFAULT_VARIANT_KEY` for cpu/gpu and
`_find_dlc_variant(mid)` (first DLC variant) for npu. The ref dir is always the
default variant's `reference_outputs`.

**Change:**
- Add `--variant` option (default `None`). When set:
  - Use it as the variant for **all** requested backends (skip the
    `_find_dlc_variant` auto-pick; the chosen variant must exist + be the right
    format for the backend).
  - Load `reference_outputs` from **that variant's** dir, not the default's.
    (AI Hub variants ship their own NHWC reference outputs.)
- When unset: preserve current behavior (default for cpu/gpu, auto DLC for npu).
- Validate: if `--variant` given but not cached / wrong format for a backend,
  report a clear per-backend FAIL row (matches existing `results` pattern).

**Interplay with per-backend `.bin`:** verify should load via the task-5 helper too,
so NPU verify exercises `model.npu.bin` when present (closest to production).

**Tests:** `--variant qcs6490` loads the right ref dir + artifact; unset keeps old
behavior; bad variant → FAIL row. 100% coverage.

---

## 3. Cross-cutting: the NHWC / no-surgery change for the AI Hub YOLO variant

This is the subtle one. `YOLOModel` currently assumes NCHW + 3-output surgery. The
AI Hub variant needs NHWC + raw 3 outputs.

**Recommended approach — make layout a model attribute, not a hardcode:**
- Add `input_layout: Literal["NCHW","NHWC"]` to `YOLOModel.__init__` (default
  `"NCHW"`), threaded from the registry/source for the AI Hub `qcs6490` variant.
- `prepare()` branches: build `(1,3,640,640)` for NCHW, `(1,640,640,3)` for NHWC,
  both float32 `[0,1]` RGB. (Resize 640, BGR→RGB, /255, then transpose per layout.)
- For the AI Hub DLC variant, `run()` returns `[boxes, scores, class_idx]` straight
  from `infer_dlc` (no `cls.max`, no surgery). The existing surgery methods
  (`_strip_qdq`/`_split_yolo_concat`/`_expose_cls_for_reducemax`) stay for the local
  `convert` path / other models but are **not used** by the AI Hub variant.
- `_decode` is unchanged: it already takes `boxes[N,4]` (x1y1x2y2, 640-space),
  `scores[N]`, `class_ids[N]`, applies threshold + NMS + scales to original size.

**Verify end-to-end after wiring:**
```
m2a model run yolo_v8 --variant qcs6490 images/pedestrian.jpg --backend cpu \
  --format json --threshold 0.25
# expect: a person detection ~0.89, not empty
```

---

## 4. Resolved decisions (were open questions)

1. **HF ↔ git (task 3):** RESOLVED — `../m2a-models` git **commit + push is the
   publish**; it mirrors the HF repo the registry reads. No separate `hf upload`.
2. **Context bins (task 5a):** RESOLVED — produce them via **AI Hub**. No stub
   function; the non-AI-Hub fallback is the existing `m2a model convert` (gets a
   warning about its limits). Local context-binary generation is a future evolution
   of `convert`; recipe documented in §5a-fallback. Pure-Python `qairt.compile`
   rejected (same failing quant/compile stack + still needs a cross-compiled `.so`).
3. **NPU producer (task 5a):** RESOLVED — **AI Hub whenever possible.** At minimum
   `model.npu.bin` from AI Hub; cpu/gpu fall back to `.dlc`.
4. **`qai_hub_models` packaging (task 2):** RESOLVED — goes in the **`[host]` extra**
   with an explanatory comment; lazy-imported in the command.

### Still to verify during implementation (not blocking)

- **Ultralytics ONNX output (task 1):** fresh `yolov8n` export likely yields a single
  `output0 (1,84,8400)`. Confirm `YOLOModel.run` (ONNX branch) + `_decode` handle it,
  or add a minimal adapter. User wants the AI Hub quantize path to work **without our
  split surgery** (it does — AI Hub emits boxes/scores/class_idx itself); `run()` and
  `_decode` get adapted for the NHWC/3-output variant regardless (§3).
- **AI Hub per-compute-unit context binaries (task 5a):** confirm AI Hub can emit
  CPU/GPU on-device context binaries, or only HTP. If only HTP, cpu/gpu use the
  `.dlc` fallback (fine).

## 5. Suggested implementation order

1. Task 1 (ultralytics source) + verify default `run` still works. — isolated.
2. Task 6 (`--variant` on verify). — tiny, unblocks testing variants.
3. Task 5b (loader helper) + `CompiledModel` handling in `load_model_dlc`. — infra.
4. Task 2 (AI Hub command, `[host]` extra) + task 5a flags (context-binary targets).
5. NHWC/no-surgery YOLO variant wiring (§3) — needed to run/verify the AI Hub DLC.
6. Task 3 (produce + publish artifact to `../m2a-models`, bump registry). —
   outward-facing, confirm before push.
7. Task 4 (hash check). — independent, can slot anywhere.

Every step: `just lint && just test` green, 100% `src/` coverage, follow
`.github/pull_request_template.md` exactly, no attribution lines.
```
