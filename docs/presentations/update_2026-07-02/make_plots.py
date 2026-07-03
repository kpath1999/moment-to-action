#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
#     "seaborn",
#     "numpy",
# ]
# ///
"""Analyse real-data LLM + VLM benchmark results and generate figures for update_2026-07-02.

Unlike prior benchmarks (synthetic frames / bounding boxes), this run uses real annotated
video clips (violence, eating, animals) fed through the actual pipeline: 1-FPS frame
extraction -> Detectron2 (NPU) detection -> LLM text prompt for the LLM path, or frames
straight into the VLM for the VLM path.

Reads llm_benchmark_results.json.gz and vlm_benchmark_results.json.gz from the same
directory. Prints a verification report to stdout and writes PNG figures to ./plots/.

Run: uv run docs/presentations/update_2026-07-02/make_plots.py
"""
# ruff: noqa: T201, ICN001, PLR2004, E501

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

HERE = Path(__file__).resolve().parent
LLM_DATA = HERE / "llm_benchmark_results.json.gz"
VLM_DATA = HERE / "vlm_benchmark_results.json.gz"
PLOTS = HERE / "plots"

MODEL_ORDER = [
    "qwen2_1_5b",
    "gemma3_1b",
    "gemma3_270m",
    "qwen3_0_6b",
    "qwen3_1_7b",
    "qwen3_4b",
    "phi35_mini",
]
MODEL_LABELS = {
    "qwen2_1_5b": "Qwen2-1.5B",
    "gemma3_1b": "Gemma3-1B",
    "gemma3_270m": "Gemma3-270M",
    "qwen3_0_6b": "Qwen3-0.6B",
    "qwen3_1_7b": "Qwen3-1.7B",
    "qwen3_4b": "Qwen3-4B",
    "phi35_mini": "Phi-3.5-Mini",
}
MODEL_COLORS = {
    "qwen2_1_5b": "#4C72B0",
    "gemma3_1b": "#55A868",
    "gemma3_270m": "#8ED08E",
    "qwen3_0_6b": "#DD8452",
    "qwen3_1_7b": "#E8A76F",
    "qwen3_4b": "#C44E52",
    "phi35_mini": "#8172B3",
}
# Models whose chat template wraps reasoning in <think>...</think> before the answer.
REASONING_MODELS = {"qwen3_0_6b", "qwen3_1_7b", "qwen3_4b"}

APP_ORDER = ["violence_detection", "eating", "animals"]
APP_LABELS = {
    "violence_detection": "Violence",
    "eating": "Eating",
    "animals": "Animal threat",
}

VLM_MODEL_ORDER = ["smolvlm2_500m", "smolvlm2_256m", "qwen25_vl_3b", "internvl3_1b", "moondream2"]
VLM_MODEL_LABELS = {
    "smolvlm2_500m": "SmolVLM2-500M",
    "smolvlm2_256m": "SmolVLM2-256M",
    "qwen25_vl_3b": "Qwen2.5-VL-3B",
    "internvl3_1b": "InternVL3-1B",
    "moondream2": "Moondream2",
}
VLM_MODEL_COLORS = {
    "smolvlm2_500m": "#4C72B0",
    "smolvlm2_256m": "#64B5CD",
    "qwen25_vl_3b": "#DD8452",
    "internvl3_1b": "#55A868",
    "moondream2": "#8172B3",
}


# ---------------------------------------------------------------------------
# Data loading + helpers
# ---------------------------------------------------------------------------


def strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning blocks emitted by Qwen3 models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def detect_yn(text: str) -> str | None:
    """Return 'yes' or 'no' from text, or None if no clear answer is present.

    Strips <think> blocks first, then looks for a leading YES/NO word or an
    'Answer: YES/NO' pattern anywhere in the (post-think) text.
    """
    cleaned = strip_think(text).strip().lower()
    words = cleaned.split()
    if words and words[0].rstrip(".,!?;:") in {"yes", "no"}:
        return words[0].rstrip(".,!?;:")
    m = re.search(r"\banswer\s*:\s*(yes|no)\b", cleaned)
    return m.group(1) if m else None


def load_data(path: Path) -> dict:
    """Load and index benchmark JSON (gzip'd) by model name."""
    with gzip.open(path, "rt") as f:
        raw = json.load(f)
    return {m["model"]: m for m in raw.get("models", [])}


def runs_for(models: dict, name: str) -> list[dict]:
    """Return all run dicts for a model, or empty list."""
    return models.get(name, {}).get("runs", [])


def infer_span_ids_for(models: dict, name: str) -> set[int]:
    """Return span IDs corresponding to streaming inference for a model."""
    report = models.get(name, {}).get("metrics_report", {})
    spans = report.get("traces", [{}])[0].get("spans", []) if report.get("traces") else []
    return {s["id"] for s in spans if "stream" in s.get("name", "")}


def hw_samples(models: dict, name: str, span_ids: set[int]) -> list[dict]:
    """Return resource usage samples taken during inference spans."""
    report = models.get(name, {}).get("metrics_report", {})
    if not report.get("traces"):
        return []
    samples = report["traces"][0].get("resource_usage_samples", [])
    return [s for s in samples if s.get("running_span_id") in span_ids]


# ---------------------------------------------------------------------------
# LLM summary stats
# ---------------------------------------------------------------------------


def build_model_stats(models: dict) -> dict[str, dict]:
    """Compute per-model aggregates used across all LLM figures."""
    stats: dict[str, dict] = {}
    for name in MODEL_ORDER:
        runs = runs_for(models, name)
        if not runs:
            stats[name] = {}
            continue

        answered = 0
        correct = 0
        yn_by_app: dict[str, list[bool]] = {}
        for r in runs:
            yn = detect_yn(r.get("response", ""))
            if yn is not None:
                answered += 1
                is_correct = yn == r["expected"].lower()
                if is_correct:
                    correct += 1
                yn_by_app.setdefault(r.get("application", ""), []).append(is_correct)

        infer = [r["infer_ms"] for r in runs if r.get("infer_ms")]
        ttfyd = [r["ttfyd_ms"] for r in runs if r.get("ttfyd_ms")]
        recall_by_app: dict[str, list[float]] = {}
        for r in runs:
            app = r.get("application", "")
            recall_by_app.setdefault(app, []).append(r.get("recall", 0.0))

        span_ids = infer_span_ids_for(models, name)
        hw = hw_samples(models, name, span_ids)
        cpu_mem_mb = [s["mem_usage"]["rss_bytes"] / 1e6 for s in hw if s.get("mem_usage")]

        m = models[name]
        stats[name] = {
            "n_runs": len(runs),
            "n_answered": answered,
            "answer_rate": answered / len(runs),
            "yn_acc": correct / answered if answered else None,
            "overall_acc": correct / len(runs),
            "yn_acc_by_app": {app: (sum(v) / len(v), len(v)) for app, v in yn_by_app.items()},
            "mean_infer": sum(infer) / len(infer) if infer else None,
            "mean_ttfyd": sum(ttfyd) / len(ttfyd) if ttfyd else None,
            "n_ttfyd": len(ttfyd),
            "mean_recall": sum(r.get("recall", 0) for r in runs) / len(runs),
            "recall_by_app": {app: sum(v) / len(v) for app, v in recall_by_app.items()},
            "load_ms": m.get("load_ms", 0.0),
            "unload_ms": m.get("unload_ms", 0.0),
            "peak_rss_mb": max(cpu_mem_mb) if cpu_mem_mb else None,
        }
    return stats


def print_report(stats: dict[str, dict]) -> None:
    """Print LLM verification report."""
    print("=" * 88)
    print("REAL-DATA LLM BENCHMARK 2026-07-02 — PER-MODEL SUMMARY (detector: Detectron2 NPU)")
    print("=" * 88)
    for name in MODEL_ORDER:
        s = stats.get(name, {})
        if not s:
            print(f"  {name}: NO DATA")
            continue
        ans = f"{s['n_answered']}/{s['n_runs']} ({s['answer_rate']:.0%})"
        acc = f"{s['yn_acc']:.0%}" if s["yn_acc"] is not None else "n/a"
        infer = f"{s['mean_infer'] / 1000:.1f}s" if s["mean_infer"] else "n/a"
        rss = f"{s['peak_rss_mb']:.0f}MB" if s["peak_rss_mb"] else "n/a"
        print(
            f"  {MODEL_LABELS[name]:14}  answered={ans:14}  acc_of_answered={acc:5}"
            f"  recall={s['mean_recall']:.3f}  load={s['load_ms'] / 1000:5.1f}s"
            f"  infer={infer:7}  unload={s['unload_ms'] / 1000:.2f}s  peak_rss={rss}"
        )
    print()


# ---------------------------------------------------------------------------
# LLM figures
# ---------------------------------------------------------------------------


def fig_answer_rate(stats: dict[str, dict]) -> None:
    """Horizontal bar: fraction of runs where a YES/NO answer was detected."""
    names = [n for n in MODEL_ORDER if stats.get(n)]
    labels = [MODEL_LABELS[n] for n in names]
    rates = [stats[n]["answer_rate"] for n in names]
    colors = ["#C44E52" if n in REASONING_MODELS else MODEL_COLORS[n] for n in names]

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    bars = ax.barh(labels, [r * 100 for r in rates], color=colors)
    for bar, rate in zip(bars, rates, strict=True):
        ax.text(
            max(rate * 100 + 1.5, 2.0),
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.0%}",
            va="center",
            fontsize=10,
        )
    ax.set_xlabel("runs with a detectable YES/NO answer (%)")
    ax.set_xlim(0, 108)
    ax.set_title("Answer rate — 128-token budget, reasoning models in red")
    ax.grid(axis="x", alpha=0.3)
    fig.text(
        0.5,
        0.01,
        "Qwen3 models emit <think>...</think> before answering — most exhaust the token "
        "budget mid-thought and never answer.",
        ha="center",
        fontsize=8,
        style="italic",
        color="#555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(PLOTS / "answer_rate.png", dpi=150)
    plt.close(fig)


def fig_yn_accuracy(stats: dict[str, dict]) -> None:
    """Horizontal bar: Y/N accuracy conditioned on the model having answered."""
    names = [n for n in MODEL_ORDER if stats.get(n) and stats[n]["yn_acc"] is not None]
    labels = [MODEL_LABELS[n] for n in names]
    accs = [stats[n]["yn_acc"] for n in names]
    colors = [MODEL_COLORS[n] for n in names]

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    bars = ax.barh(labels, [a * 100 for a in accs], color=colors)
    for bar, acc, name in zip(bars, accs, names, strict=True):
        n_ans = stats[name]["n_answered"]
        ax.text(
            max(acc * 100 + 1.5, 2.0),
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.0%} (n={n_ans})",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("Y/N accuracy among answered runs (%)")
    ax.set_xlim(0, 115)
    ax.set_title("Y/N accuracy — conditioned on answering (real video, GPU)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "yn_accuracy.png", dpi=150)
    plt.close(fig)


def _fig_accuracy_overall(
    stats: dict[str, dict],
    order: list[str],
    labels_map: dict[str, str],
    title: str,
    out_name: str,
) -> None:
    """Grouped horizontal bar: accuracy conditioned on answering vs. overall.

    "Overall" counts every non-answered run as incorrect (0), so it penalizes
    models that dodge the question by running out of token budget or emitting
    unparseable text — the fairer number for a real deployment decision.
    """
    names = [n for n in order if stats.get(n) and stats[n].get("overall_acc") is not None]
    labels = [labels_map[n] for n in names]
    conditioned = [(stats[n]["yn_acc"] or 0.0) * 100 for n in names]
    overall = [stats[n]["overall_acc"] * 100 for n in names]

    y = np.arange(len(names))
    height = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    bars_conditioned = ax.barh(
        y + height / 2, conditioned, height, label="answered runs only", color="#8CA5C0"
    )
    bars_overall = ax.barh(
        y - height / 2, overall, height, label="overall (non-answer = 0)", color="#C44E52"
    )
    for bars, vals in [(bars_conditioned, conditioned), (bars_overall, overall)]:
        for bar, val in zip(bars, vals, strict=True):
            ax.text(
                max(val + 1.5, 2.0),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%",
                va="center",
                fontsize=9,
            )
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Y/N accuracy (%)")
    ax.set_xlim(0, 115)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / out_name, dpi=150)
    plt.close(fig)


def fig_yn_accuracy_overall(stats: dict[str, dict]) -> None:
    """LLM: accuracy conditioned on answering vs. overall (non-answer = 0)."""
    _fig_accuracy_overall(
        stats,
        MODEL_ORDER,
        MODEL_LABELS,
        "LLM Y/N accuracy: answered-only vs. overall (real video, GPU)",
        "yn_accuracy_overall.png",
    )


def fig_latency_breakdown(stats: dict[str, dict]) -> None:
    """Stacked horizontal bar: load / mean_infer / unload per model (log x)."""
    names = [n for n in MODEL_ORDER if stats.get(n)]
    labels = [MODEL_LABELS[n] for n in names]
    loads = [stats[n]["load_ms"] for n in names]
    infers = [stats[n]["mean_infer"] or 0.0 for n in names]
    unloads = [stats[n]["unload_ms"] for n in names]
    stage_colors = {"load": "#8172B3", "infer (mean/clip)": "#C44E52", "unload": "#CCB974"}

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    lefts = [0.0] * len(names)
    for vals, (stage, color) in zip([loads, infers, unloads], stage_colors.items(), strict=True):
        ax.barh(labels, vals, left=lefts, label=stage, color=color)
        lefts = [left + v for left, v in zip(lefts, vals, strict=True)]
    for i, total in enumerate(lefts):
        ax.text(total * 1.02, i, f"{total / 1000:.0f}s", va="center", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("latency (ms, log scale)")
    ax.set_title("LLM latency breakdown (load + mean infer/clip + unload, Detectron2 NPU + GPU)")
    ax.legend(ncol=3, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "latency_breakdown.png", dpi=150)
    plt.close(fig)


def _fig_ttfyd_speedup(
    stats: dict[str, dict],
    order: list[str],
    labels_map: dict[str, str],
    colors_map: dict[str, str],
    subtitle: str,
    out_name: str,
) -> None:
    """Two-panel: full response latency vs. streamed time-to-YES/NO-decision.

    Only covers models whose streaming YES/NO detector actually fired — the
    benchmark's real-time detector looks for a leading YES/NO token and
    doesn't strip <think> blocks, so it can under-fire relative to the
    post-hoc (think-aware) detector used elsewhere in this script. Bars are
    annotated with n so low-sample-size speedups aren't read as solid as the
    high-n ones.
    """
    names = [n for n in order if stats.get(n) and stats[n].get("mean_ttfyd")]
    labels = [labels_map[n] for n in names]
    infers = [stats[n]["mean_infer"] or 0.0 for n in names]
    ttfyds = [stats[n]["mean_ttfyd"] for n in names]
    ns = [stats[n]["n_ttfyd"] for n in names]
    speedups = [inf / tfy for inf, tfy in zip(infers, ttfyds, strict=True)]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4.5))

    x = range(len(names))
    width = 0.38
    b1 = ax_left.bar(
        [i - width / 2 for i in x], infers, width, label="full response", color="#C44E52"
    )
    b2 = ax_left.bar(
        [i + width / 2 for i in x],
        ttfyds,
        width,
        label="TTFYD (streamed decision)",
        color="#4C72B0",
    )
    for bar, val in zip(list(b1) + list(b2), infers + ttfyds, strict=True):
        ax_left.text(
            bar.get_x() + bar.get_width() / 2,
            val * 1.06,
            f"{val / 1000:.1f}s" if val >= 1000 else f"{val:.0f}ms",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax_left.set_yscale("log")
    ymin, ymax = min(ttfyds + infers) * 0.8, max(ttfyds + infers) * 4.0
    ax_left.set_ylim(ymin, ymax)
    ax_left.set_xticks(list(x))
    ax_left.set_xticklabels(labels, fontsize=9)
    ax_left.set_ylabel("ms (log)")
    ax_left.set_title("Full response vs. streamed YES/NO decision time")
    ax_left.legend(fontsize=9, loc="upper center", ncol=2)
    ax_left.grid(axis="y", alpha=0.3)

    colors = [colors_map[n] for n in names]
    bars = ax_right.bar(labels, speedups, color=colors)
    for bar, sp, n in zip(bars, speedups, ns, strict=True):
        ax_right.text(
            bar.get_x() + bar.get_width() / 2,
            sp + 0.15,
            f"{sp:.1f}x (n={n})",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax_right.set_xticks(range(len(labels)))
    ax_right.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax_right.set_ylabel("speedup (full response / TTFYD)")
    ax_right.set_title("Early-decision speedup")
    ax_right.grid(axis="y", alpha=0.3)

    fig.suptitle(subtitle, fontsize=9, style="italic", color="#555")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(PLOTS / out_name, dpi=150)
    plt.close(fig)


def fig_ttfyd_speedup(stats: dict[str, dict]) -> None:
    """LLM: full response vs. streamed YES/NO decision time."""
    _fig_ttfyd_speedup(
        stats,
        MODEL_ORDER,
        MODEL_LABELS,
        MODEL_COLORS,
        "TTFYD: time until model commits YES/NO under streaming vs. waiting for the full response\n"
        "(Qwen3 family + Gemma3-270M excluded — real-time detector doesn't strip <think>, never fires)",
        "ttfyd_speedup.png",
    )


def fig_recall_heatmap(stats: dict[str, dict]) -> None:
    """Seaborn heatmap: rows=model, cols=app, values=mean recall."""
    model_label_order = [MODEL_LABELS[n] for n in MODEL_ORDER if stats.get(n)]
    app_label_order = [APP_LABELS[a] for a in APP_ORDER]
    active_names = [n for n in MODEL_ORDER if stats.get(n)]

    data_matrix = [
        [stats[n]["recall_by_app"].get(app, 0.0) for app in APP_ORDER] for n in active_names
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    sns.heatmap(
        np.array(data_matrix),
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=0.4,
        linewidths=0.5,
        xticklabels=app_label_order,
        yticklabels=model_label_order,
        ax=ax,
        cbar_kws={"label": "recall"},
    )
    ax.set_title("LLM keyword recall by model x application (real video)")
    ax.set_ylabel("")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(PLOTS / "recall_heatmap.png", dpi=150)
    plt.close(fig)


def _fig_yn_accuracy_by_app_heatmap(
    stats: dict[str, dict],
    order: list[str],
    labels_map: dict[str, str],
    title: str,
    out_name: str,
    figsize: tuple[float, float],
) -> None:
    """Seaborn heatmap: rows=model, cols=app, values=Y/N accuracy among answered runs.

    Cells with zero answered runs for that model x app pair are left blank
    (NaN) rather than shown as 0% — no data isn't the same as "always wrong".
    Each cell is annotated with n so a 100% cell built on n=1 doesn't read
    the same as one built on n=10.
    """
    app_label_order = [APP_LABELS[a] for a in APP_ORDER]
    active_names = [n for n in order if stats.get(n) and stats[n].get("yn_acc_by_app")]
    model_label_order = [labels_map[n] for n in active_names]

    data_matrix = np.full((len(active_names), len(APP_ORDER)), np.nan)
    annot = np.full(data_matrix.shape, "", dtype=object)
    for i, name in enumerate(active_names):
        by_app = stats[name]["yn_acc_by_app"]
        for j, app in enumerate(APP_ORDER):
            if app in by_app:
                acc, n = by_app[app]
                data_matrix[i, j] = acc
                annot[i, j] = f"{acc:.0%}\n(n={n})"
            else:
                annot[i, j] = "n/a"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        data_matrix,
        annot=annot,
        fmt="",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        xticklabels=app_label_order,
        yticklabels=model_label_order,
        ax=ax,
        cbar_kws={"label": "Y/N accuracy"},
        annot_kws={"fontsize": 8.5},
    )
    ax.set_title(title)
    ax.set_ylabel("")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(PLOTS / out_name, dpi=150)
    plt.close(fig)


def fig_yn_accuracy_by_app(stats: dict[str, dict]) -> None:
    """LLM: Y/N accuracy among answered runs, by model x app."""
    _fig_yn_accuracy_by_app_heatmap(
        stats,
        MODEL_ORDER,
        MODEL_LABELS,
        "LLM Y/N accuracy by model x application (answered runs only, real video)",
        "yn_accuracy_by_app.png",
        (8.0, 4.6),
    )


# ---------------------------------------------------------------------------
# VLM stats + figures
# ---------------------------------------------------------------------------


def build_vlm_stats(models: dict) -> dict[str, dict]:
    """Compute per-model aggregates for VLM results."""
    stats: dict[str, dict] = {}
    for name in VLM_MODEL_ORDER:
        runs = runs_for(models, name)
        if not runs:
            stats[name] = {}
            continue

        empty = sum(1 for r in runs if not r.get("response", "").strip())
        answered = 0
        correct = 0
        yn_by_app: dict[str, list[bool]] = {}
        for r in runs:
            yn = detect_yn(r.get("response", ""))
            if yn is not None:
                answered += 1
                is_correct = yn == r["expected"].lower()
                if is_correct:
                    correct += 1
                yn_by_app.setdefault(r.get("application", ""), []).append(is_correct)

        infer = [r["infer_ms"] for r in runs if r.get("infer_ms")]
        ttfyd = [r["ttfyd_ms"] for r in runs if r.get("ttfyd_ms")]
        recall_by_app: dict[str, list[float]] = {}
        for r in runs:
            app = r.get("application", "")
            recall_by_app.setdefault(app, []).append(r.get("recall", 0.0))

        m = models[name]
        stats[name] = {
            "n_runs": len(runs),
            "empty_rate": empty / len(runs),
            "n_answered": answered,
            "answer_rate": answered / len(runs),
            "yn_acc": correct / answered if answered else None,
            "overall_acc": correct / len(runs),
            "yn_acc_by_app": {app: (sum(v) / len(v), len(v)) for app, v in yn_by_app.items()},
            "mean_infer": sum(infer) / len(infer) if infer else None,
            "mean_ttfyd": sum(ttfyd) / len(ttfyd) if ttfyd else None,
            "n_ttfyd": len(ttfyd),
            "mean_recall": sum(r.get("recall", 0) for r in runs) / len(runs),
            "recall_by_app": {app: sum(v) / len(v) for app, v in recall_by_app.items()},
            "load_ms": m.get("load_ms", 0.0),
            "unload_ms": m.get("unload_ms", 0.0),
        }
    return stats


def print_vlm_report(stats: dict[str, dict]) -> None:
    """Print VLM verification report."""
    print("=" * 88)
    print("REAL-DATA VLM BENCHMARK 2026-07-01 — PER-MODEL SUMMARY")
    print("=" * 88)
    for name in VLM_MODEL_ORDER:
        s = stats.get(name, {})
        if not s:
            print(f"  {name}: NO DATA")
            continue
        ans = f"{s['n_answered']}/{s['n_runs']} ({s['answer_rate']:.0%})"
        acc = f"{s['yn_acc']:.0%}" if s["yn_acc"] is not None else "n/a"
        infer = f"{s['mean_infer'] / 1000:.1f}s" if s["mean_infer"] else "n/a"
        print(
            f"  {VLM_MODEL_LABELS[name]:15}  empty={s['empty_rate']:.0%}  answered={ans:14}"
            f"  acc_of_answered={acc:5}  recall={s['mean_recall']:.3f}"
            f"  load={s['load_ms'] / 1000:5.1f}s  infer={infer}"
        )
    print()


def fig_vlm_quality(stats: dict[str, dict]) -> None:
    """Grouped horizontal bar: empty-response rate vs answer rate per VLM."""
    names = [n for n in VLM_MODEL_ORDER if stats.get(n)]
    labels = [VLM_MODEL_LABELS[n] for n in names]
    empty = [stats[n]["empty_rate"] * 100 for n in names]
    answered = [stats[n]["answer_rate"] * 100 for n in names]

    y = np.arange(len(names))
    height = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.barh(y + height / 2, empty, height, label="empty response", color="#C44E52")
    ax.barh(y - height / 2, answered, height, label="detectable YES/NO", color="#4C72B0")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlabel("% of runs")
    ax.set_xlim(0, 108)
    ax.set_title("VLM response quality on real video (4-15 frames/clip)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "vlm_quality.png", dpi=150)
    plt.close(fig)


def fig_vlm_yn_accuracy(stats: dict[str, dict]) -> None:
    """Horizontal bar: VLM Y/N accuracy conditioned on the model having answered."""
    names = [n for n in VLM_MODEL_ORDER if stats.get(n) and stats[n]["yn_acc"] is not None]
    labels = [VLM_MODEL_LABELS[n] for n in names]
    accs = [stats[n]["yn_acc"] for n in names]
    colors = [VLM_MODEL_COLORS[n] for n in names]

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    bars = ax.barh(labels, [a * 100 for a in accs], color=colors)
    for bar, acc, name in zip(bars, accs, names, strict=True):
        n_ans = stats[name]["n_answered"]
        ax.text(
            max(acc * 100 + 1.5, 2.0),
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.0%} (n={n_ans})",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("Y/N accuracy among answered runs (%)")
    ax.set_xlim(0, 115)
    ax.set_title("VLM Y/N accuracy — conditioned on answering (real video, GPU)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "vlm_yn_accuracy.png", dpi=150)
    plt.close(fig)


def fig_vlm_yn_accuracy_overall(stats: dict[str, dict]) -> None:
    """VLM: accuracy conditioned on answering vs. overall (non-answer = 0)."""
    _fig_accuracy_overall(
        stats,
        VLM_MODEL_ORDER,
        VLM_MODEL_LABELS,
        "VLM Y/N accuracy: answered-only vs. overall (real video, GPU)",
        "vlm_yn_accuracy_overall.png",
    )


def fig_vlm_yn_accuracy_by_app(stats: dict[str, dict]) -> None:
    """VLM: Y/N accuracy among answered runs, by model x app."""
    _fig_yn_accuracy_by_app_heatmap(
        stats,
        VLM_MODEL_ORDER,
        VLM_MODEL_LABELS,
        "VLM Y/N accuracy by model x application (answered runs only, real video)",
        "vlm_yn_accuracy_by_app.png",
        (8.0, 3.8),
    )


def fig_vlm_ttfyd_speedup(stats: dict[str, dict]) -> None:
    """VLM: full response vs. streamed YES/NO decision time."""
    _fig_ttfyd_speedup(
        stats,
        VLM_MODEL_ORDER,
        VLM_MODEL_LABELS,
        VLM_MODEL_COLORS,
        "TTFYD: time until model commits YES/NO under streaming vs. waiting for the full response\n"
        "(Moondream2 excluded — never answers, no early signal to measure)",
        "vlm_ttfyd_speedup.png",
    )


def fig_vlm_latency(stats: dict[str, dict]) -> None:
    """Stacked horizontal bar: VLM load / infer / unload per model."""
    names = [n for n in VLM_MODEL_ORDER if stats.get(n)]
    labels = [VLM_MODEL_LABELS[n] for n in names]
    loads = [stats[n]["load_ms"] / 1000 for n in names]
    infers = [(stats[n]["mean_infer"] or 0.0) / 1000 for n in names]
    unloads = [stats[n]["unload_ms"] / 1000 for n in names]
    stage_colors = {"load": "#8172B3", "infer (mean/clip)": "#C44E52", "unload": "#CCB974"}

    fig, ax = plt.subplots(figsize=(9, 4.0))
    lefts = [0.0] * len(names)
    for vals, (stage, color) in zip([loads, infers, unloads], stage_colors.items(), strict=True):
        ax.barh(labels, vals, left=lefts, label=stage, color=color)
        lefts = [left + v for left, v in zip(lefts, vals, strict=True)]
    for i, total in enumerate(lefts):
        ax.text(total * 1.01, i, f"{total:.0f}s", va="center", fontsize=9)
    ax.set_xlabel("time (seconds)")
    ax.set_title("VLM latency breakdown per model (mean over all clips, GPU, real video)")
    ax.legend(ncol=3, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "vlm_latency.png", dpi=150)
    plt.close(fig)


def fig_vlm_recall_heatmap(stats: dict[str, dict]) -> None:
    """Seaborn heatmap: VLM models x apps, values = mean recall."""
    model_label_order = [VLM_MODEL_LABELS[n] for n in VLM_MODEL_ORDER if stats.get(n)]
    app_label_order = [APP_LABELS[a] for a in APP_ORDER]
    active_names = [n for n in VLM_MODEL_ORDER if stats.get(n)]

    data_matrix = [
        [stats[n]["recall_by_app"].get(app, 0.0) for app in APP_ORDER] for n in active_names
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    sns.heatmap(
        np.array(data_matrix),
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=0.4,
        linewidths=0.5,
        xticklabels=app_label_order,
        yticklabels=model_label_order,
        ax=ax,
        cbar_kws={"label": "recall"},
    )
    ax.set_title("VLM keyword recall by model x application (real video)")
    ax.set_ylabel("")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(PLOTS / "vlm_recall_heatmap.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate the report + all figures."""
    sns.set_theme(style="whitegrid")
    PLOTS.mkdir(exist_ok=True)

    llm_models = load_data(LLM_DATA)
    llm_stats = build_model_stats(llm_models)
    print_report(llm_stats)

    vlm_models = load_data(VLM_DATA)
    vlm_stats = build_vlm_stats(vlm_models)
    print_vlm_report(vlm_stats)

    fig_answer_rate(llm_stats)
    fig_yn_accuracy(llm_stats)
    fig_yn_accuracy_overall(llm_stats)
    fig_yn_accuracy_by_app(llm_stats)
    fig_latency_breakdown(llm_stats)
    fig_ttfyd_speedup(llm_stats)
    fig_recall_heatmap(llm_stats)
    fig_vlm_quality(vlm_stats)
    fig_vlm_yn_accuracy(vlm_stats)
    fig_vlm_yn_accuracy_overall(vlm_stats)
    fig_vlm_yn_accuracy_by_app(vlm_stats)
    fig_vlm_ttfyd_speedup(vlm_stats)
    fig_vlm_latency(vlm_stats)
    fig_vlm_recall_heatmap(vlm_stats)

    print(f"Wrote 14 figures to {PLOTS}")


if __name__ == "__main__":
    main()
