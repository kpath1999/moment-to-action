#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = [
#     "moment-to-action",
#     "Pillow",
# ]
#
# [tool.uv.sources]
# moment-to-action = { path = "..", editable = true }
# ///
"""Iteratively tune the VLM prompt against the labelled benchmark scenes.

This is the runnable driver for :mod:`moment_to_action.prompt_tuning`.  It closes
the loop the project needs: **prompt → run the VLM on labelled scenes → score →
iterate the prompt**, and it lets that last step be driven by a human or an LLM.

The evaluation set is the same application-specific scene set used by
``benchmark_vlms.py`` (violence / fall / animal-threat / eating / PPE), so a
tuned prompt can be compared directly against the generic benchmark prompt.

Proposer modes (``--mode``):

* ``human`` — you edit the system prompt + task template each round in the
  terminal, seeing the current best prompt and its failing cases.
* ``llm`` — an LLM iterates through a **file bridge**: the meta-prompt is written
  to ``<bridge-dir>/request_NN.md``; paste it into any chat model and save the
  JSON reply to ``<bridge-dir>/response_NN.txt``.  No API is required today.  To
  drive an online API instead, implement ``ChatClient.complete`` and return it
  from :func:`_build_chat_client` (see the commented example there).

Usage:
    uv run python scripts/tune_vlm_prompt.py --mode human --max-iterations 8
    uv run python scripts/tune_vlm_prompt.py --mode llm --bridge-dir ./bridge
    uv run python scripts/tune_vlm_prompt.py --apps violence_detection --scorer label

Requires ``llama_server_path`` in the M2A config (or pass ``--server-path``).

Some ``llama-server`` builds accept but silently ignore the legacy
``/completion`` ``image_data`` field that :class:`LlamaVLModel` sends (mtmd
multimodal support moved image input to ``/v1/chat/completions``), so the
model never actually sees the frames. Pass ``--use-chat-endpoint`` to route
the eval traffic through the chat-completions endpoint instead as a
prompt-tuning-local workaround; it does not change ``LlamaVLModel`` itself.
See "Manual (human) tuning walkthrough" in ``docs/prompt_tuning.md``.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import attrs
import httpx
from rich.console import Console
from rich.table import Table

from moment_to_action.config import AppConfig, load_config
from moment_to_action.hardware import ComputeUnit, Platform
from moment_to_action.metrics import MetricsCollector
from moment_to_action.models import ModelID, ModelManager
from moment_to_action.paths import PathManager
from moment_to_action.prompt_tuning import (
    QUIT_TOKEN,
    ChatClient,
    EvalCase,
    EvalDataset,
    FileBridgeChatClient,
    HumanProposer,
    KeywordRecallScorer,
    LabelMatchScorer,
    LLMProposer,
    PromptCandidate,
    PromptProposer,
    PromptRunner,
    PromptTuner,
    Scorer,
    TrajectoryStore,
    VLMResponseTarget,
)
from moment_to_action.utils.web import pick_free_port

if TYPE_CHECKING:
    from moment_to_action.prompt_tuning import ScoredCandidate

console = Console()

# VLM models eligible for tuning (id value -> display).
_VLM_MODELS: dict[str, ModelID] = {
    "moondream2": ModelID.MOONDREAM2,
    "qwen25_vl_3b": ModelID.QWEN25_VL_3B_INSTRUCT,
    "qwen3_vl_2b": ModelID.QWEN3_VL_2B_INSTRUCT,
    "qwen3_vl_4b": ModelID.QWEN3_VL_4B_INSTRUCT,
}

_DEFAULT_SEED_SYSTEM = (
    "You are a scene analysis AI. Answer the user's question directly and concisely. "
    "Lead with your direct answer, then give one sentence of reasoning."
)
_DEFAULT_SEED_TEMPLATE = "{question}"

_MAX_TOKENS = 128
_N_FRAMES = 4


# ---------------------------------------------------------------------------
# Dataset — reuse the benchmark scenes as a labelled eval set
# ---------------------------------------------------------------------------


def _load_benchmark_module() -> Any:  # a dynamically loaded sibling script module
    """Import ``benchmark_vlms.py`` from the scripts directory.

    Returns:
        The loaded module, exposing ``_SCENES`` and ``_get_frames``.

    Raises:
        ImportError: If the module cannot be located or executed.
    """
    path = Path(__file__).parent / "benchmark_vlms.py"
    spec = importlib.util.spec_from_file_location("benchmark_vlms", path)
    if spec is None or spec.loader is None:
        msg = f"could not load {path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the module's @dataclass field-type resolution
    # (which looks the module up in sys.modules) succeeds.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_dataset(video_dir: Path | None, n_frames: int, apps: set[str] | None) -> EvalDataset:
    """Build an :class:`EvalDataset` from the benchmark scenes.

    Args:
        video_dir: Optional directory of ``<scene>.mp4`` clips; synthetic frames
            are rendered when absent.
        n_frames: Number of frames to include per scene.
        apps: If given, keep only scenes whose ``app`` is in this set.

    Returns:
        The dataset of labelled cases.
    """
    bench = _load_benchmark_module()
    cases: list[EvalCase] = []
    for scene in bench._SCENES:  # noqa: SLF001 — reusing the benchmark's scene list
        if apps is not None and scene.app not in apps:
            continue
        frames, _is_real = bench._get_frames(scene, video_dir, n_frames)  # noqa: SLF001
        cases.append(
            EvalCase(
                case_id=scene.name,
                question=scene.task,
                images_b64=tuple(frames),
                expected_label=scene.expected_label,
                recall_keywords=tuple(scene.recall_keywords),
                app=scene.app,
            )
        )
    return EvalDataset(cases)


# ---------------------------------------------------------------------------
# Vision workaround — some llama-server builds ignore the legacy image_data
# field that LlamaVLModel sends over /completion; route through the modern
# chat-completions endpoint instead. Scoped to this driver, not LlamaVLModel.
# ---------------------------------------------------------------------------


@attrs.define
class _ChatEndpointTarget:
    """A :class:`ResponseTarget` that talks to ``/v1/chat/completions`` directly.

    Bypasses ``LlamaVLModel.prepare``/``run``, which sends images via the
    legacy ``/completion`` ``image_data`` field. On llama-server builds whose
    multimodal support has moved to ``mtmd``, that field is silently ignored
    and the model never sees the frames; this target sends images as
    ``image_url`` data URIs instead, which is what those builds expect.

    Attributes:
        client: An ``httpx.Client`` pointed at the running llama-server.
        max_tokens: Maximum tokens to request per completion.
    """

    client: httpx.Client
    max_tokens: int

    def generate(self, candidate: PromptCandidate, case: EvalCase) -> str:
        """Compose the candidate prompt and run it via chat-completions.

        Args:
            candidate: The prompt candidate to apply.
            case: The evaluation case supplying the question and images.

        Returns:
            The model's text response (empty string if none was returned).

        Raises:
            httpx.HTTPStatusError: If the server returns a non-2xx response.
        """
        prompt = candidate.compose(case.question)
        content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
        content.extend(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
            for img in case.images_b64
        )
        resp = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "m2a-prompt-tuning",
                "messages": [{"role": "user", "content": content}],
                "max_tokens": self.max_tokens,
                "temperature": 0,
            },
        )
        resp.raise_for_status()
        choices = resp.json()["choices"]
        return str(choices[0]["message"]["content"]) if choices else ""


# ---------------------------------------------------------------------------
# Proposer wiring (the human / LLM port)
# ---------------------------------------------------------------------------


def _read_field(label: str) -> str:
    """Read a possibly multi-line prompt field from the terminal.

    Args:
        label: Name of the field being edited (shown in the prompt).

    Returns:
        The entered text (blank to reuse the best value; ``/stop`` to finish).
    """
    console.print(
        f"[cyan]New {label}[/cyan] "
        f"(blank keeps the best value; finish with a single '.'; '{QUIT_TOKEN}' to stop):"
    )
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == ".":
            break
        lines.append(line)
    return "\n".join(lines)


def _await_llm_reply(request_path: Path, response_path: Path) -> None:
    """Block until the user has saved the LLM reply next to the request.

    Args:
        request_path: File the meta-prompt was written to.
        response_path: File the user must create with the LLM's JSON reply.
    """
    console.print(f"[bold]LLM step[/bold] — paste the contents of [green]{request_path}[/green]")
    console.print("into any chat model, then save its JSON reply to")
    console.print(f"[green]{response_path}[/green].")
    while not response_path.exists():
        input("Press Enter once the reply file exists (Ctrl-C to abort)... ")


def _build_chat_client(bridge_dir: Path) -> ChatClient:
    """Return the ChatClient used by the LLM proposer.

    Today this returns a :class:`FileBridgeChatClient` (offline, no API).  To use
    an online model, implement ``complete(system, user) -> str`` (e.g. an HTTP
    POST to your endpoint) and return it here instead::

        return MyHttpChatClient(base_url=..., api_key=...)

    Args:
        bridge_dir: Directory used to exchange request/response files.

    Returns:
        A configured chat client.
    """
    return FileBridgeChatClient(bridge_dir=bridge_dir, await_response=_await_llm_reply)


def _build_proposer(mode: str, bridge_dir: Path) -> PromptProposer:
    """Construct the proposer for the requested mode.

    Args:
        mode: ``"human"`` or ``"llm"``.
        bridge_dir: Directory for the LLM file bridge (used only for ``"llm"``).

    Returns:
        A proposer implementing the :class:`PromptProposer` protocol.
    """
    if mode == "human":
        return HumanProposer(read=_read_field, write=console.print)
    return LLMProposer(client=_build_chat_client(bridge_dir))


def _build_scorer(name: str) -> Scorer:
    """Return the scorer for the given name.

    Args:
        name: ``"recall"`` or ``"label"``.

    Returns:
        The corresponding scorer instance.
    """
    return LabelMatchScorer() if name == "label" else KeywordRecallScorer()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_report(scored: ScoredCandidate) -> None:
    """Print a per-round summary table and the worst failing cases.

    Args:
        scored: The scored candidate produced this round.
    """
    report = scored.report
    candidate = scored.candidate
    table = Table(title=f"{candidate.label}  (parent: {candidate.parent_id or 'seed'})")
    table.add_column("metric", style="bold cyan")
    table.add_column("value", justify="right")
    table.add_row("mean score", f"{report.mean_score:.3f}")
    table.add_row("pass rate", f"{report.pass_rate:.2f}")
    for app, score in sorted(report.per_app().items()):
        table.add_row(f"  {app}", f"{score:.3f}")
    console.print(table)
    if candidate.rationale:
        console.print(f"  [dim]rationale: {candidate.rationale}[/dim]")
    for failure in report.failures[:5]:
        detail = failure.error or failure.response.strip() or "(empty)"
        console.print(
            f"  [red]✗ {failure.case_id}[/red] (score {failure.score:.2f}): {detail[:120]}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Tune a VLM prompt against the benchmark scenes.")
    parser.add_argument("--mode", choices=["human", "llm"], default="human")
    parser.add_argument(
        "--model", choices=sorted(_VLM_MODELS), default="qwen25_vl_3b", help="VLM to tune."
    )
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--target-score", type=float, default=None, help="Stop early at this mean.")
    parser.add_argument("--pass-threshold", type=float, default=0.5)
    parser.add_argument("--scorer", choices=["recall", "label"], default="recall")
    parser.add_argument("--n-frames", type=int, default=_N_FRAMES)
    parser.add_argument("--max-tokens", type=int, default=_MAX_TOKENS)
    parser.add_argument(
        "--apps", default=None, help="Comma-separated app names to keep (default: all)."
    )
    parser.add_argument("--video-dir", default=None, help="Directory of <scene>.mp4 clips.")
    parser.add_argument("--out-dir", default=None, help="Run output dir (default: ./prompt_runs).")
    parser.add_argument("--bridge-dir", default=None, help="LLM file-bridge dir (--mode llm).")
    parser.add_argument("--server-path", default=None, help="Override llama_server_path.")
    parser.add_argument("--port", type=int, default=None, help="Override llama_server_port.")
    parser.add_argument(
        "--use-chat-endpoint",
        action="store_true",
        help=(
            "Route eval traffic through /v1/chat/completions instead of "
            "LlamaVLModel's own prepare/run path. Workaround for llama-server "
            "builds that silently ignore the legacy /completion image_data "
            "field (see module docstring)."
        ),
    )
    parser.add_argument("--seed-system", default=_DEFAULT_SEED_SYSTEM)
    parser.add_argument("--seed-template", default=_DEFAULT_SEED_TEMPLATE)
    return parser.parse_args()


def _resolve_config(args: argparse.Namespace, path_manager: PathManager) -> AppConfig:
    """Load the app config and apply llama-server overrides from CLI args.

    Args:
        args: Parsed CLI arguments.
        path_manager: The application path manager.

    Returns:
        The effective :class:`AppConfig`.

    Raises:
        SystemExit: If no llama-server path is configured.
    """
    config = load_config(path_manager.app_config_file)
    server_path = Path(args.server_path) if args.server_path else config.llama_server_path
    port = args.port if args.port is not None else config.llama_server_port
    if server_path is None:
        console.print(
            "[red]llama_server_path not set. Use --server-path or set it in config.[/red]"
        )
        raise SystemExit(1)
    return AppConfig(
        **{**config.model_dump(), "llama_server_path": server_path, "llama_server_port": port}
    )


def main() -> None:
    """Entry point: load the model and run the tuning loop."""
    args = _parse_args()
    if args.use_chat_endpoint and args.port is None:
        # Pin a concrete port so this process's own httpx.Client can reach the
        # llama-server instance that model.load() starts.
        args.port = pick_free_port()
    apps = set(args.apps.split(",")) if args.apps else None
    video_dir = Path(args.video_dir) if args.video_dir else None
    out_dir = (
        Path(args.out_dir) if args.out_dir else Path("prompt_runs") / time.strftime("%Y%m%d_%H%M%S")
    )
    bridge_dir = Path(args.bridge_dir) if args.bridge_dir else out_dir / "bridge"

    path_manager = PathManager()
    config = _resolve_config(args, path_manager)

    dataset = _build_dataset(video_dir, args.n_frames, apps)
    if len(dataset) == 0:
        console.print("[red]No scenes selected. Check --apps.[/red]")
        raise SystemExit(1)

    scorer = _build_scorer(args.scorer)
    proposer = _build_proposer(args.mode, bridge_dir)
    seed = PromptCandidate(system_prompt=args.seed_system, task_template=args.seed_template)
    task_description = (
        "Drive a small on-device VLM to answer, per scene, the binary/multi-label "
        "classification question shown to it. Applications: "
        f"{', '.join(sorted({c.app for c in dataset}))}. "
        f"Frames are provided as images. Scored by the '{scorer.name}' metric."
    )

    console.rule("[bold]M2A VLM Prompt Tuning[/bold]")
    console.print(f"  model      : {args.model}")
    console.print(f"  cases      : {len(dataset)}")
    console.print(f"  scorer     : {scorer.name} (pass >= {args.pass_threshold})")
    console.print(f"  mode       : {args.mode}")
    console.print(f"  iterations : {args.max_iterations}")
    console.print(f"  out dir    : {out_dir}")
    if args.mode == "llm":
        console.print(f"  bridge dir : {bridge_dir}")
    console.print()

    manager = ModelManager(path_manager)
    # Empty system prompt: the tuned prompt is composed into the user prompt so
    # candidates can be swapped without restarting llama-server.
    model = manager.get_model(_VLM_MODELS[args.model], system_prompt="", max_tokens=args.max_tokens)

    metrics = MetricsCollector()
    chat_client = (
        httpx.Client(base_url=f"http://127.0.0.1:{config.llama_server_port}", timeout=120.0)
        if args.use_chat_endpoint
        else None
    )
    with metrics.start_trace():
        model.load(Platform(config), ComputeUnit.GPU, metrics=metrics)
        try:
            target = (
                _ChatEndpointTarget(client=chat_client, max_tokens=args.max_tokens)
                if chat_client is not None
                else VLMResponseTarget(model, metrics=metrics)
            )
            runner = PromptRunner(
                target=target,
                scorer=scorer,
                pass_threshold=args.pass_threshold,
            )
            tuner = PromptTuner(
                runner=runner,
                proposer=proposer,
                dataset=dataset,
                task_description=task_description,
                store=TrajectoryStore(run_dir=out_dir),
            )
            final = tuner.run(
                seed,
                max_iterations=args.max_iterations,
                target_score=args.target_score,
                on_report=_print_report,
            )
        finally:
            if chat_client is not None:
                chat_client.close()
            model.unload(metrics=metrics)

    best = final.best
    console.rule("[bold green]Best prompt[/bold green]")
    console.print(f"  {best.candidate.label}  mean_score={best.report.mean_score:.3f}")
    console.print(f"  system_prompt: {best.candidate.system_prompt!r}")
    console.print(f"  task_template: {best.candidate.task_template!r}")
    console.print(f"\n[green]Full trajectory and best prompt written to {out_dir}[/green]")


if __name__ == "__main__":
    main()
