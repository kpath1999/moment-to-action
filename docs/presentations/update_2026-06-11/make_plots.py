#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas",
#     "matplotlib",
#     "seaborn",
# ]
# ///
"""Analyse benchmark_results.csv and generate presentation figures.

Reads the QCS6490 detector benchmark CSV, prints a verification report (means,
std, FPS, CPU->NPU/GPU speedups) to stdout, and writes PNG figures into ./plots.

Run: uv run docs/presentations/update_2026-06-11/make_plots.py
"""
# Standalone report/figure script (run via uv): print is the output; the multiplication
# sign in plot titles is intentional; matplotlib.use() must precede the pyplot import.
# ruff: noqa: T201, ICN001, RUF001

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

HERE = Path(__file__).resolve().parent
CSV = HERE / "benchmark_results.csv"
PLOTS = HERE / "plots"

# Display order + colours.
MODEL_ORDER = ["yolo_v8", "rf_detr", "rtm_det", "detectron2_w8a8", "detectron2_w8a16"]
BACKEND_ORDER = ["cpu", "gpu", "npu"]
BACKEND_COLORS = {"cpu": "#4C72B0", "gpu": "#DD8452", "npu": "#55A868"}
STAGES = ["load_ms", "preproc_ms", "infer_ms", "postproc_ms", "unload_ms"]
STAGE_LABELS = ["load", "preproc", "infer", "postproc", "unload"]
STAGE_COLORS = ["#8172B3", "#937860", "#C44E52", "#DA8BC3", "#CCB974"]

# Why a (model, backend) cell is empty / degraded — for the support matrix + slides.
NOTES = {
    ("rf_detr", "gpu"): "GPU OpPackage rejects Tile",
    ("rf_detr", "npu"): "float-only; v68 rejects float I/O",
    ("rtm_det", "gpu"): "GPU OpPackage rejects Cast",
    ("rtm_det", "npu"): "float decode head; not quantizable",
    ("detectron2_w8a8", "gpu"): "GPU OpPackage rejects Exp",
    ("detectron2_w8a16", "gpu"): "GPU OpPackage rejects Exp",
}


def load() -> pd.DataFrame:
    """Load the benchmark CSV into a DataFrame."""
    df = pd.read_csv(CSV)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    df["backend"] = pd.Categorical(df["backend"], categories=BACKEND_ORDER, ordered=True)
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Return per (model, backend) means/std + FPS, sorted by display order."""
    agg = (
        df.groupby(["model", "backend"], observed=True)
        .agg(
            n=("infer_ms", "size"),
            load=("load_ms", "mean"),
            preproc=("preproc_ms", "mean"),
            infer=("infer_ms", "mean"),
            infer_std=("infer_ms", "std"),
            post=("postproc_ms", "mean"),
            unload=("unload_ms", "mean"),
            ap50=("ap50", "mean"),
        )
        .reset_index()
    )
    agg["fps"] = 1000.0 / agg["infer"]
    return agg.sort_values(["model", "backend"])


def print_report(agg: pd.DataFrame) -> None:
    """Print the verification report the user can check against the plots."""
    print("=" * 78)
    print("PER (MODEL, BACKEND) MEANS  [ms unless noted]")
    print("=" * 78)
    cols = ("model", "be", "n", "load", "pre", "infer", "post", "unload", "fps", "ap50")
    widths = (-17, -4, 4, 8, 6, 9, 6, 7, 7, 6)
    print(
        "".join(f"{c:>{w}} " if w > 0 else f"{c:<{-w}} " for c, w in zip(cols, widths, strict=True))
    )
    print("-" * 78)
    for _, r in agg.iterrows():
        print(
            f"{r['model']:17} {r['backend']:4} {int(r['n']):>4} {r['load']:>8.1f} "
            f"{r['preproc']:>6.2f} {r['infer']:>9.1f} {r['post']:>6.3f} {r['unload']:>7.1f} "
            f"{r['fps']:>7.2f} {r['ap50']:>6.4f}"
        )

    print("\n" + "=" * 78)
    print("INFERENCE SPEEDUPS  (mean infer_ms ratio)")
    print("=" * 78)
    infer = {(r["model"], r["backend"]): r["infer"] for _, r in agg.iterrows()}
    for model in MODEL_ORDER:
        cpu = infer.get((model, "cpu"))
        if cpu is None:
            continue
        parts = [f"{model:17} cpu={cpu:8.1f}ms"]
        if (model, "npu") in infer:
            parts.append(f"CPU->NPU={cpu / infer[(model, 'npu')]:5.1f}x")
        if (model, "gpu") in infer:
            parts.append(f"CPU->GPU={cpu / infer[(model, 'gpu')]:5.1f}x")
        if (model, "npu") in infer and (model, "gpu") in infer:
            parts.append(f"GPU->NPU={infer[(model, 'gpu')] / infer[(model, 'npu')]:5.1f}x")
        print("  ".join(parts))

    print("\n" + "=" * 78)
    print("MISSING / DEGRADED BACKENDS")
    print("=" * 78)
    present = {(r["model"], r["backend"]) for _, r in agg.iterrows()}
    for model in MODEL_ORDER:
        miss = [be for be in BACKEND_ORDER if (model, be) not in present]
        if miss:
            for be in miss:
                why = NOTES.get((model, be), "not run")
                print(f"{model:17} {be:4} MISSING  ({why})")
    print(f"{'yolo_v8':17} gpu  RUNS but AP50 collapses (numeric) — see accuracy plot")


def fig_support_matrix(agg: pd.DataFrame) -> None:
    """Grid of which (model, backend) combos ran."""
    present = {(r["model"], r["backend"]) for _, r in agg.iterrows()}
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for yi, model in enumerate(MODEL_ORDER):
        for xi, be in enumerate(BACKEND_ORDER):
            ran = (model, be) in present
            ax.add_patch(
                plt.Rectangle(
                    (xi, yi), 1, 1, facecolor="#2E7D32" if ran else "#C62828", edgecolor="white"
                )
            )
            label = "RAN" if ran else "✗"
            ax.text(
                xi + 0.5,
                yi + 0.5,
                label,
                ha="center",
                va="center",
                color="white",
                fontsize=15,
                fontweight="bold",
            )
    ax.set_xlim(0, len(BACKEND_ORDER))
    ax.set_ylim(0, len(MODEL_ORDER))
    ax.set_xticks([i + 0.5 for i in range(len(BACKEND_ORDER))])
    ax.set_xticklabels([b.upper() for b in BACKEND_ORDER])
    ax.set_yticks([i + 0.5 for i in range(len(MODEL_ORDER))])
    ax.set_yticklabels(MODEL_ORDER)
    ax.set_title("Backend support on QCS6490 (Hexagon v68)")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(PLOTS / "support_matrix.png", dpi=150)
    plt.close(fig)


def fig_infer_latency(agg: pd.DataFrame) -> None:
    """Grouped bar of mean infer_ms (log y)."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = range(len(MODEL_ORDER))
    width = 0.26
    for bi, be in enumerate(BACKEND_ORDER):
        vals, xs = [], []
        for mi, model in enumerate(MODEL_ORDER):
            row = agg[(agg["model"] == model) & (agg["backend"] == be)]
            if not row.empty:
                vals.append(row["infer"].to_numpy()[0])
                xs.append(mi + (bi - 1) * width)
        bars = ax.bar(xs, vals, width, label=be.upper(), color=BACKEND_COLORS[be])
        for b, v in zip(bars, vals, strict=True):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v * 1.05,
                f"{v:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(MODEL_ORDER, rotation=15, ha="right")
    ax.set_ylabel("mean inference (ms, log)")
    ax.set_title("Inference latency per model × backend")
    ax.legend(title="backend")
    ax.grid(axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "infer_latency.png", dpi=150)
    plt.close(fig)


def fig_npu_speedup(agg: pd.DataFrame) -> None:
    """CPU->NPU (and yolo CPU->GPU) inference speedups."""
    infer = {(r["model"], r["backend"]): r["infer"] for _, r in agg.iterrows()}
    labels, vals, colors = [], [], []
    for model in MODEL_ORDER:
        cpu = infer.get((model, "cpu"))
        if cpu and (model, "npu") in infer:
            labels.append(f"{model}\nCPU→NPU")
            vals.append(cpu / infer[(model, "npu")])
            colors.append(BACKEND_COLORS["npu"])
    cpu = infer.get(("yolo_v8", "cpu"))
    if cpu and ("yolo_v8", "gpu") in infer:
        labels.append("yolo_v8\nCPU→GPU")
        vals.append(cpu / infer[("yolo_v8", "gpu")])
        colors.append(BACKEND_COLORS["gpu"])
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals, strict=True):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.6,
            f"{v:.1f}×",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_ylabel("speedup (×)")
    ax.set_title("Inference speedup vs CPU")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "npu_speedup.png", dpi=150)
    plt.close(fig)


def fig_latency_breakdown(agg: pd.DataFrame) -> None:
    """Stacked load/preproc/infer/post/unload per (model, backend), log-ish via two panels."""
    rows = [(r["model"], r["backend"], r) for _, r in agg.iterrows()]
    labels = [f"{m}\n{b}" for m, b, _ in rows]
    fig, ax = plt.subplots(figsize=(11, 5))
    bottoms = [0.0] * len(rows)
    cols = ["load", "preproc", "infer", "post", "unload"]
    for col, lab, color in zip(cols, STAGE_LABELS, STAGE_COLORS, strict=True):
        vals = [float(r[col]) for _, _, r in rows]
        ax.bar(labels, vals, bottom=bottoms, label=lab, color=color)
        bottoms = [b + v for b, v in zip(bottoms, vals, strict=True)]
    for i, total in enumerate(bottoms):
        ax.text(i, total * 1.02, f"{total:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel("latency (ms, log) — load+preproc+infer+post+unload")
    ax.set_title("End-to-end latency breakdown per cycle")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "latency_breakdown.png", dpi=150)
    plt.close(fig)


def fig_throughput(agg: pd.DataFrame) -> None:
    """FPS (1000/infer_ms) per working pair."""
    labels = [f"{r['model']}\n{r['backend']}" for _, r in agg.iterrows()]
    vals = agg["fps"].tolist()
    colors = [BACKEND_COLORS[r["backend"]] for _, r in agg.iterrows()]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals, strict=True):
        ax.text(
            b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=8
        )
    ax.axhline(30, color="gray", ls="--", lw=1)
    ax.text(len(vals) - 0.5, 31, "30 fps (real-time)", ha="right", color="gray", fontsize=8)
    ax.set_ylabel("throughput (FPS = 1000 / infer_ms)")
    ax.set_title("Inference throughput per model × backend")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "throughput_fps.png", dpi=150)
    plt.close(fig)


def fig_accuracy(agg: pd.DataFrame) -> None:
    """AP50 (crude proxy) per pair, with caveat caption."""
    labels = [f"{r['model']}\n{r['backend']}" for _, r in agg.iterrows()]
    vals = agg["ap50"].tolist()
    colors = [BACKEND_COLORS[r["backend"]] for _, r in agg.iterrows()]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals, strict=True):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.005,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylabel("AP50 (class-agnostic IoU proxy)")
    ax.set_title("Accuracy proxy — NOT real COCO mAP (labels ignored)")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.text(
        0.5,
        0.005,
        "Caveat: benchmark _ap50 matches boxes by IoU only (ignores class) "
        "— a rough trend signal, not validation mAP.",
        ha="center",
        fontsize=8,
        style="italic",
        color="#555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(PLOTS / "accuracy_ap50.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Generate the report + all figures."""
    sns.set_theme(style="whitegrid")
    PLOTS.mkdir(exist_ok=True)
    df = load()
    agg = summarise(df)
    print_report(agg)
    fig_support_matrix(agg)
    fig_infer_latency(agg)
    fig_npu_speedup(agg)
    fig_latency_breakdown(agg)
    fig_throughput(agg)
    fig_accuracy(agg)
    print(f"\nWrote 6 figures to {PLOTS}")


if __name__ == "__main__":
    main()
