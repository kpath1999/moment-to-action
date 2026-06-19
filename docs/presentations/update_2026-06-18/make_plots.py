#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas",
#     "matplotlib",
#     "seaborn",
# ]
# ///
"""Analyse benchmark CSVs and generate presentation figures for update_2026-06-18.

Reads benchmark_results.csv (detection) and llm_benchmark_results.csv from HERE.
Prints a verification report to stdout and writes PNG figures to ./plots/.

Run: uv run docs/presentations/update_2026-06-18/make_plots.py
"""
# Standalone report/figure script: print is the output; the multiplication sign in plot
# titles is intentional; matplotlib.use() must precede the pyplot import.
# ruff: noqa: T201, ICN001, RUF001

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

HERE = Path(__file__).resolve().parent
DET_CSV = HERE / "benchmark_results.csv"
LLM_CSV = HERE / "llm_benchmark_results.csv"
PLOTS = HERE / "plots"

MODEL_ORDER = ["yolo_v8", "rf_detr", "rtm_det", "detectron2_w8a8", "detectron2_w8a16"]
BACKEND_ORDER = ["cpu", "gpu", "npu"]
BACKEND_COLORS = {"cpu": "#4C72B0", "gpu": "#DD8452", "npu": "#55A868"}

LLM_MODEL_ORDER = ["qwen2_1_5b", "qwen2_7b", "qwen3_4b", "phi35_mini"]
LLM_MODEL_LABELS = {
    "qwen2_1_5b": "Qwen2-1.5B",
    "qwen2_7b": "Qwen2-7B",
    "qwen3_4b": "Qwen3-4B",
    "phi35_mini": "Phi-3.5-mini",
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


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def load_det() -> pd.DataFrame:
    """Load the detection benchmark CSV."""
    df = pd.read_csv(DET_CSV)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    df["backend"] = pd.Categorical(df["backend"], categories=BACKEND_ORDER, ordered=True)
    return df


def summarise_det(df: pd.DataFrame) -> pd.DataFrame:
    """Return per (model, backend) means."""
    agg = (
        df.groupby(["model", "backend"], observed=True)
        .agg(
            n=("infer_ms", "size"),
            load=("load_ms", "mean"),
            preproc=("preproc_ms", "mean"),
            infer=("infer_ms", "mean"),
            post=("postproc_ms", "mean"),
            unload=("unload_ms", "mean"),
            ap50=("ap50", "mean"),
        )
        .reset_index()
    )
    agg["fps"] = 1000.0 / agg["infer"]
    return agg.sort_values(["model", "backend"])


def print_det_report(agg: pd.DataFrame) -> None:
    """Print the detection verification report."""
    print("=" * 78)
    print("DETECTION — PER (MODEL, BACKEND) MEANS  [ms unless noted]")
    print("=" * 78)
    for _, r in agg.iterrows():
        print(
            f"  {r['model']:17} {r['backend']:4}  infer={r['infer']:8.1f}ms  "
            f"fps={r['fps']:6.2f}  ap50={r['ap50']:.4f}"
        )


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


def load_llm() -> pd.DataFrame:
    """Load the LLM benchmark CSV."""
    df = pd.read_csv(LLM_CSV)
    df["model"] = pd.Categorical(df["model"], categories=LLM_MODEL_ORDER, ordered=True)
    df["app"] = pd.Categorical(df["app"], categories=APP_ORDER, ordered=True)
    return df


def summarise_llm_model(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-model mean recall and infer_ms."""
    return (
        df.groupby("model", observed=True)
        .agg(recall=("recall", "mean"), infer_ms=("infer_ms", "mean"))
        .reset_index()
        .sort_values("model")
    )


def summarise_llm_app(df: pd.DataFrame) -> pd.DataFrame:
    """Return per (model, app) mean recall."""
    return df.groupby(["model", "app"], observed=True).agg(recall=("recall", "mean")).reset_index()


def print_llm_report(model_agg: pd.DataFrame, app_agg: pd.DataFrame) -> None:
    """Print the LLM verification report."""
    print("\n" + "=" * 78)
    print("LLM — PER MODEL MEANS")
    print("=" * 78)
    for _, r in model_agg.iterrows():
        print(f"  {r['model']:20}  recall={r['recall']:.4f}  infer={r['infer_ms']:.0f}ms")

    print("\n" + "=" * 78)
    print("LLM — PER (MODEL, APP) RECALL")
    print("=" * 78)
    for _, r in app_agg.iterrows():
        print(f"  {r['model']:20}  {r['app']:30}  recall={r['recall']:.3f}")


# ---------------------------------------------------------------------------
# Detection figures
# ---------------------------------------------------------------------------


def fig_accuracy_ap50(agg: pd.DataFrame) -> None:
    """Bar chart of AP50 by (model, backend), GPU bar hatched to mark collapse."""
    labels = [f"{r['model']}\n{r['backend']}" for _, r in agg.iterrows()]
    vals = agg["ap50"].tolist()
    backends = agg["backend"].tolist()
    colors = [BACKEND_COLORS[b] for b in backends]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    bars = ax.bar(labels, vals, color=colors)
    for bar, val, be in zip(bars, vals, backends, strict=True):
        if be == "gpu":
            bar.set_hatch("///")
            bar.set_edgecolor("#C62828")
            bar.set_linewidth(1.5)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=BACKEND_COLORS["cpu"], label="CPU"),
        Patch(facecolor=BACKEND_COLORS["gpu"], label="GPU"),
        Patch(facecolor=BACKEND_COLORS["npu"], label="NPU"),
    ]
    ax.legend(handles=legend_elements, title="backend")
    ax.set_ylabel("AP50 (class-agnostic IoU proxy)")
    ax.set_title("Accuracy proxy — NOT real COCO mAP")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.text(
        0.5,
        0.005,
        "Caveat: class-agnostic IoU proxy (labels ignored). Hatched GPU bar = FP16 accuracy collapse.",
        ha="center",
        fontsize=8,
        style="italic",
        color="#555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(PLOTS / "accuracy_ap50.png", dpi=150)
    plt.close(fig)


def fig_yolo_gpu_collapse(agg: pd.DataFrame) -> None:
    """YOLO-only grouped bar: AP50 by backend, annotated with infer_ms."""
    yolo = agg[agg["model"] == "yolo_v8"].copy()
    fig, ax = plt.subplots(figsize=(7, 4.6))
    x = range(len(yolo))
    bars = ax.bar(
        x,
        yolo["ap50"].tolist(),
        color=[BACKEND_COLORS[b] for b in yolo["backend"].tolist()],
    )
    for bar, (_, r) in zip(bars, yolo.iterrows(), strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            r["ap50"] + 0.005,
            f"AP50={r['ap50']:.3f}\n{r['infer']:.0f}ms",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    if "gpu" in yolo["backend"].tolist():
        gpu_idx = list(yolo["backend"]).index("gpu")
        bars[gpu_idx].set_hatch("///")
        bars[gpu_idx].set_edgecolor("#C62828")
        bars[gpu_idx].set_linewidth(1.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels([b.upper() for b in yolo["backend"].tolist()], fontsize=13)
    ax.set_ylabel("AP50")
    ax.set_title("YOLOv8: accuracy vs backend (same DLC, different execution precision)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(yolo["ap50"]) * 1.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "yolo_gpu_collapse.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# LLM figures
# ---------------------------------------------------------------------------


def fig_llm_latency_breakdown(df: pd.DataFrame) -> None:
    """Stacked bar of mean load/infer/unload per LLM model (log y)."""
    agg = (
        df.groupby("model", observed=True)
        .agg(load=("load_ms", "mean"), infer=("infer_ms", "mean"), unload=("unload_ms", "mean"))
        .reset_index()
    )
    labels = [LLM_MODEL_LABELS.get(m, m) for m in agg["model"].tolist()]
    stage_colors = {"load": "#8172B3", "infer": "#C44E52", "unload": "#CCB974"}
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bottoms = [0.0] * len(agg)
    for stage, color in stage_colors.items():
        vals = agg[stage].tolist()
        ax.bar(labels, vals, bottom=bottoms, label=stage, color=color)
        bottoms = [b + v for b, v in zip(bottoms, vals, strict=True)]
    for i, (total, (_, r)) in enumerate(zip(bottoms, agg.iterrows(), strict=True)):
        ax.text(i, total * 1.05, f"{total / 1000:.0f}s", ha="center", va="bottom", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("latency (ms, log) — load + infer + unload")
    ax.set_title("LLM latency breakdown per model (mean over all scenes)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(PLOTS / "llm_latency_breakdown.png", dpi=150)
    plt.close(fig)


def fig_llm_recall_model(model_agg: pd.DataFrame) -> None:
    """Horizontal bar chart of mean recall per model, annotated with infer_ms."""
    labels = [LLM_MODEL_LABELS.get(m, m) for m in model_agg["model"].tolist()]
    recalls = model_agg["recall"].tolist()
    infer_ms = model_agg["infer_ms"].tolist()
    colors = ["#4C72B0", "#55A868", "#DD8452", "#C44E52"]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels, recalls, color=colors)
    for bar, rec, ms in zip(bars, recalls, infer_ms, strict=True):
        ax.text(
            rec + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{rec:.3f}  ({ms / 1000:.0f}s infer)",
            va="center",
            fontsize=10,
        )
    ax.set_xlabel("mean recall (keyword match fraction)")
    ax.set_title("LLM scene classification — recall by model\n(YOLO detections → text → LLM)")
    ax.set_xlim(0, 0.72)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "llm_recall_model.png", dpi=150)
    plt.close(fig)


def fig_llm_recall_heatmap(app_agg: pd.DataFrame) -> None:
    """Seaborn heatmap: rows=model, cols=app, values=mean recall."""
    pivot = app_agg.pivot(index="model", columns="app", values="recall")
    pivot.index = [LLM_MODEL_LABELS.get(m, m) for m in pivot.index]
    pivot.columns = [APP_LABELS.get(c, c) for c in pivot.columns]
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "recall"},
    )
    ax.set_title("LLM recall by model × application")
    ax.set_ylabel("")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(PLOTS / "llm_recall_heatmap.png", dpi=150)
    plt.close(fig)


def fig_llm_latency_recall(model_agg: pd.DataFrame) -> None:
    """Scatter: x=infer_ms (log), y=recall, labeled by model."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = ["#4C72B0", "#55A868", "#DD8452", "#C44E52"]
    for (_, r), color in zip(model_agg.iterrows(), colors, strict=True):
        label = LLM_MODEL_LABELS.get(r["model"], r["model"])
        ax.scatter(r["infer_ms"], r["recall"], s=120, color=color, zorder=3)
        ax.annotate(
            label,
            (r["infer_ms"], r["recall"]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=10,
        )
    ax.set_xscale("log")
    ax.set_xlabel("mean inference time (ms, log scale)")
    ax.set_ylabel("mean recall")
    ax.set_title("LLM: latency vs recall tradeoff")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "llm_latency_recall.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate the report + all figures."""
    sns.set_theme(style="whitegrid")
    PLOTS.mkdir(exist_ok=True)

    det_df = load_det()
    det_agg = summarise_det(det_df)
    print_det_report(det_agg)

    llm_df = load_llm()
    llm_model_agg = summarise_llm_model(llm_df)
    llm_app_agg = summarise_llm_app(llm_df)
    print_llm_report(llm_model_agg, llm_app_agg)

    fig_accuracy_ap50(det_agg)
    fig_yolo_gpu_collapse(det_agg)
    fig_llm_latency_breakdown(llm_df)
    fig_llm_recall_model(llm_model_agg)
    fig_llm_recall_heatmap(llm_app_agg)
    fig_llm_latency_recall(llm_model_agg)

    print(f"\nWrote 6 figures to {PLOTS}")


if __name__ == "__main__":
    main()
