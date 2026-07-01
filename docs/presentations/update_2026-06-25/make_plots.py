#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
#     "seaborn",
# ]
# ///
"""Analyse LLM + VLM benchmark results and generate presentation figures for update_2026-06-25.

Reads llm_benchmark_results.json and vlm_benchmark_results.csv from the same directory.
Prints a verification report to stdout and writes PNG figures to ./plots/.

Run: uv run docs/presentations/update_2026-06-25/make_plots.py
"""
# ruff: noqa: T201, ICN001

from __future__ import annotations

import csv
import gzip
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

HERE = Path(__file__).resolve().parent
DATA = HERE / "llm_benchmark_results.json.gz"
VLM_CSV = HERE / "vlm_benchmark_results.csv"
PLOTS = HERE / "plots"

MODEL_ORDER = ["qwen2_1_5b", "qwen2_7b", "qwen3_4b", "phi35_mini", "moondream2"]
MODEL_LABELS = {
    "qwen2_1_5b": "Qwen2-1.5B",
    "qwen2_7b": "Qwen2-7B",
    "qwen3_4b": "Qwen3-4B",
    "phi35_mini": "Phi-3.5-Mini",
    "moondream2": "Moondream2",
}
MODEL_COLORS = {
    "qwen2_1_5b": "#4C72B0",
    "qwen2_7b": "#DD8452",
    "qwen3_4b": "#55A868",
    "phi35_mini": "#C44E52",
    "moondream2": "#8172B3",
}

APP_ORDER = [
    "violence_detection",
    "fall_detection",
    "animal_threat_detection",
    "eating_detection",
    "ppe_compliance",
]
APP_LABELS = {
    "violence_detection": "Violence",
    "fall_detection": "Fall",
    "animal_threat_detection": "Animal threat",
    "eating_detection": "Eating",
    "ppe_compliance": "PPE",
}

VLM_MODEL_ORDER = ["qwen25_vl_3b", "qwen3_vl_2b", "qwen3_vl_4b", "moondream2"]
VLM_MODEL_LABELS = {
    "qwen25_vl_3b": "Qwen2.5-VL-3B",
    "qwen3_vl_2b": "Qwen3-VL-2B",
    "qwen3_vl_4b": "Qwen3-VL-4B",
    "moondream2": "Moondream2",
}
VLM_MODEL_COLORS = {
    "qwen25_vl_3b": "#4C72B0",
    "qwen3_vl_2b": "#55A868",
    "qwen3_vl_4b": "#DD8452",
    "moondream2": "#8172B3",
}


# ---------------------------------------------------------------------------
# Data loading + helpers
# ---------------------------------------------------------------------------


def detect_yn(text: str) -> str | None:
    """Return 'yes' or 'no' from text, handling leading word or 'Answer: YES' format."""
    cleaned = text.strip().lower()
    words = cleaned.split()
    if words and words[0].rstrip(".,!?;:") in {"yes", "no"}:
        return words[0].rstrip(".,!?;:")
    m = re.search(r"\banswer\s*:\s*(yes|no)\b", cleaned)
    return m.group(1) if m else None


def load_data() -> dict:
    """Load and index benchmark JSON by model name."""
    with gzip.open(DATA, "rt") as f:
        raw = json.load(f)
    models = {m["model"]: m for m in raw.get("models", [])}
    return models


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
# Summary stats
# ---------------------------------------------------------------------------


def build_model_stats(models: dict) -> dict[str, dict]:
    """Compute per-model aggregates used across all figures."""
    stats: dict[str, dict] = {}
    for name in MODEL_ORDER:
        runs = runs_for(models, name)
        if not runs:
            stats[name] = {}
            continue

        yn_correct: list[bool] = []
        for r in runs:
            yn = detect_yn(r.get("response", ""))
            if yn is not None:
                yn_correct.append(yn == r["expected"].lower())

        ttft = [r["ttft_ms"] for r in runs if r.get("ttft_ms")]
        ttfyd = [r["ttfyd_ms"] for r in runs if r.get("ttfyd_ms")]
        itl = [r["mean_itl_ms"] for r in runs if r.get("mean_itl_ms")]
        infer = [r["infer_ms"] for r in runs if r.get("infer_ms")]
        recall_by_app: dict[str, list[float]] = {}
        for r in runs:
            app = r.get("app", "")
            recall_by_app.setdefault(app, []).append(r.get("recall", 0.0))

        span_ids = infer_span_ids_for(models, name)
        hw = hw_samples(models, name, span_ids)
        gpu_mem = [s["gpu_usage"]["memory_mb"] for s in hw if s.get("gpu_usage")]
        cpu_util = [s["cpu_usage"]["usage_pct"] for s in hw if s.get("cpu_usage")]
        cpu_mem_mb = [s["cpu_usage"]["memory_mb"] for s in hw if s.get("cpu_usage")]

        m = models[name]
        stats[name] = {
            "yn_acc": sum(yn_correct) / len(yn_correct) if yn_correct else None,
            "yn_n": len(yn_correct),
            "yn_total": len(runs),
            "mean_ttft": sum(ttft) / len(ttft) if ttft else None,
            "mean_ttfyd": sum(ttfyd) / len(ttfyd) if ttfyd else None,
            "mean_itl": sum(itl) / len(itl) if itl else None,
            "mean_infer": sum(infer) / len(infer) if infer else None,
            "mean_recall": sum(r.get("recall", 0) for r in runs) / len(runs),
            "recall_by_app": {app: sum(v) / len(v) for app, v in recall_by_app.items()},
            "load_ms": m.get("load_ms", 0.0),
            "unload_ms": m.get("unload_ms", 0.0),
            "peak_gpu_mem": max(gpu_mem) if gpu_mem else None,
            "peak_cpu_mem_mb": max(cpu_mem_mb) if cpu_mem_mb else None,
            "mean_cpu_util": sum(cpu_util) / len(cpu_util) if cpu_util else None,
        }
    return stats


def print_report(stats: dict[str, dict]) -> None:
    """Print verification report."""
    print("=" * 78)
    print("LLM BENCHMARK 2026-06-25 — PER-MODEL SUMMARY")
    print("=" * 78)
    for name in MODEL_ORDER:
        s = stats.get(name, {})
        if not s:
            print(f"  {name}: NO DATA")
            continue
        yn_pct = s["yn_acc"] if s["yn_acc"] is not None else 0.0
        yn = f"{yn_pct:.0%} ({s['yn_n']}/{s['yn_total']})"
        ttft = f"{s['mean_ttft']:.0f}ms" if s["mean_ttft"] else "n/a"
        ttfyd = f"{s['mean_ttfyd']:.0f}ms" if s["mean_ttfyd"] else "n/a"
        itl = f"{s['mean_itl']:.0f}ms" if s["mean_itl"] else "n/a"
        gpu = f"{s['peak_gpu_mem']:.0f}MB" if s["peak_gpu_mem"] else "n/a"
        rss = f"{s['peak_cpu_mem_mb']:.0f}MB" if s["peak_cpu_mem_mb"] else "n/a"
        cpu = f"{s['mean_cpu_util']:.0f}%" if s["mean_cpu_util"] is not None else "n/a"
        print(
            f"  {MODEL_LABELS[name]:16}  yn={yn:18}  recall={s['mean_recall']:.3f}"
            f"  ttft={ttft:8}  ttfyd={ttfyd:8}  itl={itl:6}"
            f"  gpu_mem={gpu:8}  cpu_mem={rss:8}  cpu_util={cpu}"
        )
    print()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_yn_accuracy(stats: dict[str, dict]) -> None:
    """Horizontal bar: Y/N accuracy per model."""
    names = [n for n in MODEL_ORDER if stats.get(n)]
    labels = [MODEL_LABELS[n] for n in names]
    accs = [stats[n]["yn_acc"] or 0.0 for n in names]
    colors = [MODEL_COLORS[n] if (stats[n]["yn_acc"] or 0) > 0 else "#BBBBBB" for n in names]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.barh(labels, [a * 100 for a in accs], color=colors)
    for bar, acc, name in zip(bars, accs, names):
        val_s = f"{acc:.0%}"
        suffix = " *" if name == "phi35_mini" else ""
        ax.text(
            max(acc * 100 + 0.5, 1.0),
            bar.get_y() + bar.get_height() / 2,
            val_s + suffix,
            va="center",
            fontsize=10,
        )
    ax.set_xlabel("Y/N accuracy (%)")
    ax.set_xlim(0, 105)
    ax.set_title("Y/N accuracy by model (GPU)")
    ax.grid(axis="x", alpha=0.3)
    fig.text(
        0.5,
        0.01,
        "* Phi-3.5-Mini uses 'Answer: YES/NO' format — relaxed detector applied.",
        ha="center",
        fontsize=8,
        style="italic",
        color="#555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(PLOTS / "yn_accuracy.png", dpi=150)
    plt.close(fig)


def fig_latency_breakdown(stats: dict[str, dict]) -> None:
    """Stacked horizontal bar: load / mean_infer / unload per model (log x)."""
    names = [n for n in MODEL_ORDER if stats.get(n)]
    labels = [MODEL_LABELS[n] for n in names]
    loads = [stats[n]["load_ms"] for n in names]
    infers = [stats[n]["mean_infer"] or 0.0 for n in names]
    unloads = [stats[n]["unload_ms"] for n in names]
    stage_colors = {"load": "#8172B3", "infer": "#C44E52", "unload": "#CCB974"}

    fig, ax = plt.subplots(figsize=(9, 4.2))
    lefts = [0.0] * len(names)
    for vals, (stage, color) in zip([loads, infers, unloads], stage_colors.items(), strict=True):
        ax.barh(labels, vals, left=lefts, label=stage, color=color)
        lefts = [l + v for l, v in zip(lefts, vals, strict=True)]
    for i, total in enumerate(lefts):
        ax.text(total * 1.02, i, f"{total / 1000:.0f}s", va="center", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("latency (ms, log scale) — load + infer + unload")
    ax.set_title("Latency breakdown by model (mean over all scenes, GPU)")
    ax.legend(ncol=3, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "latency_breakdown.png", dpi=150)
    plt.close(fig)


def fig_ttft_itl(stats: dict[str, dict]) -> None:
    """Grouped bar (log y): TTFT and ITL per model."""
    names = [n for n in MODEL_ORDER if stats.get(n) and stats[n].get("mean_ttft")]
    labels = [MODEL_LABELS[n] for n in names]
    ttfts = [stats[n]["mean_ttft"] for n in names]
    itls = [stats[n]["mean_itl"] or 0.0 for n in names]

    x = range(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.0))
    b1 = ax.bar([i - width / 2 for i in x], ttfts, width, label="TTFT", color="#4C72B0")
    b2 = ax.bar([i + width / 2 for i in x], itls, width, label="Mean ITL", color="#DD8452")
    for bar, val in zip(list(b1) + list(b2), ttfts + itls):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val * 1.08,
            f"{val / 1000:.1f}s" if val >= 1000 else f"{val:.0f}ms",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("latency (ms, log scale)")
    ax.set_title("Time-to-first-token (TTFT) and inter-token latency (ITL) by model")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(PLOTS / "ttft_itl.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_recall_heatmap(models: dict, stats: dict[str, dict]) -> None:
    """Seaborn heatmap: rows=model, cols=app, values=mean recall."""
    rows = []
    for name in MODEL_ORDER:
        if not stats.get(name):
            continue
        for app in APP_ORDER:
            rows.append(
                {
                    "model": MODEL_LABELS[name],
                    "app": APP_LABELS.get(app, app),
                    "recall": stats[name]["recall_by_app"].get(app, 0.0),
                }
            )

    pivot: dict[str, dict[str, float]] = {}
    for row in rows:
        pivot.setdefault(row["model"], {})[row["app"]] = row["recall"]

    model_label_order = [MODEL_LABELS[n] for n in MODEL_ORDER if stats.get(n)]
    app_label_order = [APP_LABELS[a] for a in APP_ORDER]

    import numpy as np

    data_matrix = [
        [pivot.get(m, {}).get(a, 0.0) for a in app_label_order] for m in model_label_order
    ]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    sns.heatmap(
        np.array(data_matrix),
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        xticklabels=app_label_order,
        yticklabels=model_label_order,
        ax=ax,
        cbar_kws={"label": "recall"},
    )
    ax.set_title("Recall by model × application (GPU)")
    ax.set_ylabel("")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(PLOTS / "recall_heatmap.png", dpi=150)
    plt.close(fig)


def fig_yn_per_app(models: dict, stats: dict[str, dict]) -> None:
    """Grouped bar: Y/N accuracy per app × model (skip PPE, all n/a)."""
    yn_apps = [a for a in APP_ORDER if a != "ppe_compliance"]

    yn_by_model_app: dict[str, dict[str, float | None]] = {}
    for name in MODEL_ORDER:
        if not stats.get(name):
            continue
        runs = runs_for(models, name)
        by_app: dict[str, list[bool]] = {}
        for r in runs:
            app = r.get("app", "")
            if app not in yn_apps:
                continue
            yn = detect_yn(r.get("response", ""))
            if yn is not None:
                by_app.setdefault(app, []).append(yn == r["expected"].lower())
        yn_by_model_app[name] = {
            app: (sum(v) / len(v) if v else None) for app in yn_apps for v in [by_app.get(app, [])]
        }

    active_names = [n for n in MODEL_ORDER if n in yn_by_model_app]
    n_models = len(active_names)
    n_apps = len(yn_apps)
    width = 0.7 / n_models
    x = range(n_apps)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    for mi, name in enumerate(active_names):
        offset = (mi - n_models / 2 + 0.5) * width
        vals = [yn_by_model_app[name].get(app) for app in yn_apps]
        bar_vals = [v * 100 if v is not None else 0.0 for v in vals]
        bars = ax.bar(
            [xi + offset for xi in x],
            bar_vals,
            width * 0.9,
            label=MODEL_LABELS[name],
            color=MODEL_COLORS[name],
            alpha=0.85,
        )
        for bar, val in zip(bars, vals):
            if val is None:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    1.5,
                    "n/a",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#888",
                )
    ax.set_xticks(list(x))
    ax.set_xticklabels([APP_LABELS[a] for a in yn_apps])
    ax.set_ylabel("Y/N accuracy (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Y/N accuracy by application × model (GPU, PPE excluded — not binary)")
    ax.legend(ncol=n_models, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "yn_per_app.png", dpi=150)
    plt.close(fig)


def fig_ttfyd_speedup(stats: dict[str, dict]) -> None:
    """Two-panel: raw infer_ms vs ttfyd_ms (left) and speedup ratio (right)."""
    names = [n for n in MODEL_ORDER if stats.get(n) and stats[n].get("mean_ttfyd")]
    labels = [MODEL_LABELS[n] for n in names]
    infers = [stats[n]["mean_infer"] or 0.0 for n in names]
    ttfyds = [stats[n]["mean_ttfyd"] for n in names]
    speedups = [inf / tfy for inf, tfy in zip(infers, ttfyds, strict=True)]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: raw ms comparison
    x = range(len(names))
    width = 0.38
    b1 = ax_left.bar(
        [i - width / 2 for i in x], infers, width, label="Full response", color="#C44E52"
    )
    b2 = ax_left.bar(
        [i + width / 2 for i in x], ttfyds, width, label="TTFYD (decision)", color="#4C72B0"
    )
    for bar, val in zip(list(b1) + list(b2), infers + ttfyds):
        ax_left.text(
            bar.get_x() + bar.get_width() / 2,
            val * 1.06,
            f"{val / 1000:.0f}s",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax_left.set_yscale("log")
    ax_left.set_xticks(list(x))
    ax_left.set_xticklabels(labels, fontsize=9)
    ax_left.set_ylabel("ms (log)")
    ax_left.set_title("Full response vs decision time")
    ax_left.legend(fontsize=9)
    ax_left.grid(axis="y", alpha=0.3)

    # Right: speedup ratio
    colors = [MODEL_COLORS[n] for n in names]
    bars = ax_right.bar(labels, speedups, color=colors)
    for bar, sp in zip(bars, speedups):
        ax_right.text(
            bar.get_x() + bar.get_width() / 2,
            sp + 0.02,
            f"{sp:.1f}×",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax_right.set_ylabel("speedup (full response / TTFYD)")
    ax_right.set_title("Early-decision speedup")
    ax_right.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "TTFYD: time until model commits YES/NO vs waiting for full response\n"
        "(Moondream2, Phi-3.5-Mini excluded — no Y/N decision detected)",
        fontsize=9,
        style="italic",
        color="#555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(PLOTS / "ttfyd_speedup.png", dpi=150)
    plt.close(fig)


def fig_hw_profile(stats: dict[str, dict]) -> None:
    """Three-panel: peak GPU mem, peak CPU memory, avg CPU utilization per model."""
    names = [n for n in MODEL_ORDER if stats.get(n) and stats[n].get("peak_gpu_mem")]
    labels = [MODEL_LABELS[n] for n in names]
    gpu_mem = [stats[n]["peak_gpu_mem"] for n in names]
    cpu_mem_mb = [stats[n]["peak_cpu_mem_mb"] or 0.0 for n in names]
    cpu_util = [stats[n]["mean_cpu_util"] or 0.0 for n in names]
    colors = [MODEL_COLORS[n] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    for ax, vals, title, xlabel in [
        (axes[0], gpu_mem, "Peak GPU memory (MB)", "MB"),
        (axes[1], cpu_mem_mb, "Peak CPU memory (MB)", "MB"),
        (axes[2], cpu_util, "Avg CPU utilization (%)", "%"),
    ]:
        bars = ax.barh(labels, vals, color=colors)
        for bar, val in zip(bars, vals):
            ax.text(
                val + max(vals) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}{xlabel}",
                va="center",
                fontsize=9,
            )
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlim(0, max(vals) * 1.18)

    fig.suptitle(
        "Hardware profile during inference (GPU run)\n"
        "Note: GPU utilization counter reads 0% for Vulkan compute on QCS6490 — not shown.",
        fontsize=8,
        style="italic",
        color="#555",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(PLOTS / "hw_profile.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# VLM helpers + figures
# ---------------------------------------------------------------------------


def load_vlm() -> list[dict]:
    """Load VLM benchmark CSV as list of row dicts."""
    with VLM_CSV.open() as f:
        return list(csv.DictReader(f))


def build_vlm_stats(rows: list[dict]) -> dict[str, dict]:
    """Compute per-model aggregates for VLM results."""
    stats: dict[str, dict] = {}
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    for name in VLM_MODEL_ORDER:
        rs = by_model.get(name, [])
        if not rs:
            stats[name] = {}
            continue
        recall = [float(r["recall"]) for r in rs]
        infer = [float(r["infer_ms"]) for r in rs]
        recall_by_app: dict[str, list[float]] = {}
        for r in rs:
            recall_by_app.setdefault(r["app"], []).append(float(r["recall"]))
        stats[name] = {
            "load_ms": float(rs[0]["load_ms"]),
            "unload_ms": float(rs[0]["unload_ms"]),
            "mean_infer": sum(infer) / len(infer),
            "mean_recall": sum(recall) / len(recall),
            "recall_by_app": {app: sum(v) / len(v) for app, v in recall_by_app.items()},
        }
    return stats


def print_vlm_report(stats: dict[str, dict]) -> None:
    """Print VLM verification report."""
    print("\n" + "=" * 78)
    print("VLM BENCHMARK 2026-06-25 — PER-MODEL SUMMARY")
    print("=" * 78)
    for name in VLM_MODEL_ORDER:
        s = stats.get(name, {})
        if not s:
            print(f"  {name}: NO DATA")
            continue
        print(
            f"  {VLM_MODEL_LABELS[name]:18}  recall={s['mean_recall']:.3f}"
            f"  load={s['load_ms'] / 1000:.0f}s  infer={s['mean_infer'] / 1000:.0f}s"
            f"  unload={s['unload_ms'] / 1000:.1f}s"
        )
    print()


def fig_vlm_latency(stats: dict[str, dict]) -> None:
    """Stacked horizontal bar: VLM load / infer / unload per model."""
    names = [n for n in VLM_MODEL_ORDER if stats.get(n)]
    labels = [VLM_MODEL_LABELS[n] for n in names]
    loads = [stats[n]["load_ms"] / 1000 for n in names]
    infers = [stats[n]["mean_infer"] / 1000 for n in names]
    unloads = [stats[n]["unload_ms"] / 1000 for n in names]
    stage_colors = {"load": "#8172B3", "infer": "#C44E52", "unload": "#CCB974"}

    fig, ax = plt.subplots(figsize=(9, 4.0))
    lefts = [0.0] * len(names)
    for vals, (stage, color) in zip([loads, infers, unloads], stage_colors.items(), strict=True):
        ax.barh(labels, vals, left=lefts, label=stage, color=color)
        lefts = [l + v for l, v in zip(lefts, vals, strict=True)]
    for i, total in enumerate(lefts):
        ax.text(total * 1.01, i, f"{total / 60:.0f} min", va="center", fontsize=9)
    ax.set_xlabel("time (seconds)")
    ax.set_title("VLM latency breakdown per model (mean over all scenes, GPU, 4 frames)")
    ax.legend(ncol=3, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "vlm_latency.png", dpi=150)
    plt.close(fig)


def fig_vlm_recall_heatmap(stats: dict[str, dict]) -> None:
    """Seaborn heatmap: VLM models × apps, values = mean recall."""
    import numpy as np

    model_label_order = [VLM_MODEL_LABELS[n] for n in VLM_MODEL_ORDER if stats.get(n)]
    app_label_order = [APP_LABELS[a] for a in APP_ORDER]
    active_names = [n for n in VLM_MODEL_ORDER if stats.get(n)]

    data_matrix = [
        [stats[n]["recall_by_app"].get(app, 0.0) for app in APP_ORDER] for n in active_names
    ]

    fig, ax = plt.subplots(figsize=(9, 3.8))
    sns.heatmap(
        np.array(data_matrix),
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        xticklabels=app_label_order,
        yticklabels=model_label_order,
        ax=ax,
        cbar_kws={"label": "recall"},
    )
    ax.set_title("VLM recall by model × application (GPU, 4 synthetic frames)")
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

    models = load_data()
    stats = build_model_stats(models)
    print_report(stats)

    vlm_rows = load_vlm()
    vlm_stats = build_vlm_stats(vlm_rows)
    print_vlm_report(vlm_stats)

    fig_yn_accuracy(stats)
    fig_latency_breakdown(stats)
    fig_ttft_itl(stats)
    fig_recall_heatmap(models, stats)
    fig_yn_per_app(models, stats)
    fig_ttfyd_speedup(stats)
    fig_hw_profile(stats)
    fig_vlm_latency(vlm_stats)
    fig_vlm_recall_heatmap(vlm_stats)

    print(f"Wrote 9 figures to {PLOTS}")


if __name__ == "__main__":
    main()
