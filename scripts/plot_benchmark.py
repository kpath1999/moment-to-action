"""Bar-chart visualizations for benchmark results.

Generates PNG plots from ``VariantProfile`` data collected by
``benchmark_model.py``.  Can be invoked directly:

    uv run python scripts/plot_benchmark.py --results-dir scripts/tmp_results_<ts>

or used programmatically::

    from plot_benchmark import plot_profiles
    plot_profiles(profiles, model_name="yolo", output_dir=Path("scripts/my_run"))

Plots produced (all saved as PNG into *output_dir*):
  - ``<model>_latency.png``    — grouped bars: mean / p50 / p95 / p99 per compute unit
  - ``<model>_load.png``       — model load latency per compute unit
  - ``<model>_memory.png``     — peak RSS memory per compute unit
  - ``<model>_accuracy.png``   — accuracy per compute unit (only when data available)
  - ``<model>_power.png``      — power draw and energy/inference (only when data available)
  - ``<model>_summary.png``    — combined 2x3 dashboard of all subplots
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from moment_to_action.benchmark import VariantProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette — one colour per compute unit
# ---------------------------------------------------------------------------

_UNIT_COLOURS: dict[str, str] = {
    "cpu": "#4C72B0",
    "gpu": "#DD8452",
    "npu": "#55A868",
}
_DEFAULT_COLOUR = "#8172B2"


def _unit_colour(unit_str: str) -> str:
    return _UNIT_COLOURS.get(unit_str.lower(), _DEFAULT_COLOUR)


# ---------------------------------------------------------------------------
# Core plotting function
# ---------------------------------------------------------------------------


def plot_profiles(  # noqa: C901, PLR0915
    profiles: list[VariantProfile],
    model_name: str,
    output_dir: Path,
) -> None:
    """Generate bar-chart PNGs for *profiles* and write them to *output_dir*.

    Args:
        profiles:   List of ``VariantProfile`` objects (one per compute unit).
        model_name: Short model name used as title and file-name prefix.
        output_dir: Directory where PNGs are written (must already exist).
    """
    try:
        import matplotlib as mpl

        mpl.use("Agg")  # non-interactive backend — safe in headless envs
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError as exc:
        logger.warning("matplotlib not installed — skipping plots (%s)", exc)
        return

    if not profiles:
        logger.warning("No profiles to plot.")
        return

    units = [p.variant_id.compute_unit.value for p in profiles]
    colours = [_unit_colour(u) for u in units]
    display_names: dict[str, str] = {
        "yolo": "YOLOv8",
        "mobileclip": "MobileCLIP-S2",
        "smolvlm2": "SmolVLM2",
        "qwen3": "Qwen3",
    }
    title_prefix = display_names.get(model_name.lower(), model_name.upper())

    # ── helpers ──────────────────────────────────────────────────────────────

    def _bar(  # type: ignore[name-defined]
        ax: plt.Axes,
        labels: list[str],
        values: list[float],
        ylabel: str,
        title: str,
        clrs: list[str],
    ) -> None:
        bars = ax.bar(labels, values, color=clrs, edgecolor="white", linewidth=0.5, zorder=3)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.grid(axis="y", alpha=0.35, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, val in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{val:.3g}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    def _grouped_bar(
        ax: plt.Axes,  # type: ignore[name-defined]
        labels: list[str],
        series: dict[str, list[float]],
        ylabel: str,
        title: str,
        series_colours: list[str],
    ) -> None:
        import numpy as np

        n_groups = len(labels)
        n_series = len(series)
        width = 0.7 / n_series
        x = np.arange(n_groups)
        offsets = [(i - (n_series - 1) / 2) * width for i in range(n_series)]

        for (sname, vals), offset, colour in zip(
            series.items(), offsets, series_colours, strict=False
        ):
            bars = ax.bar(
                x + offset,
                vals,
                width=width * 0.9,
                label=sname,
                color=colour,
                zorder=3,
                edgecolor="white",
                linewidth=0.5,
            )
            for bar, val in zip(bars, vals, strict=True):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{val:.3g}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=30,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.35, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ── figure factory ────────────────────────────────────────────────────────

    def _save(fig: plt.Figure, name: str) -> Path:  # type: ignore[name-defined]
        path = output_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Saved %s", path)
        return path

    # ── 1. Latency grouped bar chart ─────────────────────────────────────────

    latency_series = {
        "mean": [p.inference_mean_ms for p in profiles],
        "p50": [p.inference_p50_ms for p in profiles],
        "p95": [p.inference_p95_ms for p in profiles],
        "p99": [p.inference_p99_ms for p in profiles],
    }
    latency_colours = ["#4C72B0", "#55A868", "#DD8452", "#C44E52"]

    fig_lat, ax_lat = plt.subplots(figsize=(max(5, len(units) * 2.5), 4))
    _grouped_bar(
        ax_lat,
        units,
        latency_series,
        "Latency (ms)",
        f"{title_prefix} — Inference Latency",
        latency_colours,
    )
    _save(fig_lat, f"{model_name}_latency.png")

    # ── 2. Load latency ──────────────────────────────────────────────────────

    fig_load, ax_load = plt.subplots(figsize=(max(4, len(units) * 1.5), 4))
    _bar(
        ax_load,
        units,
        [p.load_latency_ms for p in profiles],
        "Load latency (ms)",
        f"{title_prefix} — Model Load Latency",
        colours,
    )
    _save(fig_load, f"{model_name}_load.png")

    # ── 3. Peak memory ───────────────────────────────────────────────────────

    fig_mem, ax_mem = plt.subplots(figsize=(max(4, len(units) * 1.5), 4))
    _bar(
        ax_mem,
        units,
        [p.peak_memory_mb for p in profiles],
        "Peak RSS (MB)",
        f"{title_prefix} — Peak Memory",
        colours,
    )
    _save(fig_mem, f"{model_name}_memory.png")

    # ── 4. Accuracy (optional) ───────────────────────────────────────────────

    acc_values = [p.accuracy for p in profiles if p.accuracy is not None]
    acc_units = [p.variant_id.compute_unit.value for p in profiles if p.accuracy is not None]
    acc_colours = [_unit_colour(u) for u in acc_units]

    if acc_values:
        fig_acc, ax_acc = plt.subplots(figsize=(max(4, len(acc_units) * 1.5), 4))
        _bar(
            ax_acc,
            acc_units,
            acc_values,
            "Score [0-1]",
            f"{title_prefix} — Accuracy vs CPU Oracle",
            acc_colours,
        )
        ax_acc.set_ylim(0, 1.1)
        _save(fig_acc, f"{model_name}_accuracy.png")

    # ── 5. Power / energy (optional) ─────────────────────────────────────────

    power_profiles = [p for p in profiles if p.cost.power_mw is not None]
    if power_profiles:
        pw_units = [p.variant_id.compute_unit.value for p in power_profiles]
        pw_colours = [_unit_colour(u) for u in pw_units]
        pw_power = [p.cost.power_mw for p in power_profiles]  # type: ignore[misc]
        pw_energy = [p.cost.energy_per_inference_mj or 0.0 for p in power_profiles]

        fig_pw, (ax_pw, ax_en) = plt.subplots(1, 2, figsize=(max(7, len(pw_units) * 2.5), 4))
        _bar(ax_pw, pw_units, pw_power, "Power (mW)", f"{title_prefix} — Power Draw", pw_colours)  # type: ignore[arg-type]
        _bar(
            ax_en,
            pw_units,
            pw_energy,
            "Energy/inf (mJ)",
            f"{title_prefix} — Energy per Inference",
            pw_colours,
        )
        _save(fig_pw, f"{model_name}_power.png")

    # ── 6. Summary dashboard ─────────────────────────────────────────────────

    n_rows, n_cols = 2, 3
    fig_sum, axes = plt.subplots(n_rows, n_cols, figsize=(14, 8))
    fig_sum.suptitle(f"{title_prefix} Benchmark Summary", fontsize=14, fontweight="bold", y=1.01)

    _grouped_bar(
        axes[0, 0],
        units,
        latency_series,
        "Latency (ms)",
        "Inference Latency",
        latency_colours,
    )
    _bar(axes[0, 1], units, [p.load_latency_ms for p in profiles], "ms", "Load Latency", colours)
    _bar(axes[0, 2], units, [p.peak_memory_mb for p in profiles], "MB", "Peak Memory", colours)

    if acc_values:
        _bar(axes[1, 0], acc_units, acc_values, "Score [0-1]", "Accuracy vs Oracle", acc_colours)
        axes[1, 0].set_ylim(0, 1.1)
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            "No accuracy data\n(pass --eval-images)",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
            fontsize=9,
            color="grey",
        )
        axes[1, 0].set_title("Accuracy vs Oracle", fontsize=10, fontweight="bold", pad=6)
        axes[1, 0].axis("off")

    if power_profiles:
        _bar(axes[1, 1], pw_units, pw_power, "mW", "Power Draw", pw_colours)  # type: ignore[arg-type]
        _bar(axes[1, 2], pw_units, pw_energy, "mJ", "Energy/Inference", pw_colours)
    else:
        for ax in (axes[1, 1], axes[1, 2]):
            ax.text(
                0.5,
                0.5,
                "No power data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
                color="grey",
            )
            ax.axis("off")

    _save(fig_sum, f"{model_name}_summary.png")


# ---------------------------------------------------------------------------
# CLI entry point — load profiles from a results dir JSON and re-plot
# ---------------------------------------------------------------------------


def _load_profiles_from_json(json_path: Path) -> tuple[str, list[dict[str, object]]]:
    """Reconstruct minimal profile objects from a saved JSON results file.

    Returns ``(model_name, profiles)`` where each profile is a plain dict
    with the same shape as ``VariantProfile.json()``.
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return data["model"], data["profiles"]


class _DictProfile:
    """Thin wrapper that provides attribute access over a ``VariantProfile.json()`` dict."""

    class _Cost:
        def __init__(self, d: dict) -> None:
            self.power_mw = d.get("power_mw")
            self.energy_per_inference_mj = d.get("energy_per_inference_mj")

    class _VariantID:
        def __init__(self, d: dict) -> None:
            self.compute_unit = _UnitValue(d["compute_unit"])

    def __init__(self, d: dict) -> None:
        self.variant_id = _DictProfile._VariantID(d["variant_id"])
        self.accuracy = d.get("accuracy")
        self.load_latency_ms = d["load_latency_ms"]
        self.inference_mean_ms = d["inference_mean_ms"]
        self.inference_p50_ms = d["inference_p50_ms"]
        self.inference_p95_ms = d["inference_p95_ms"]
        self.inference_p99_ms = d["inference_p99_ms"]
        self.peak_memory_mb = d["peak_memory_mb"]
        self.max_batch_size = d["max_batch_size"]
        self.cost = _DictProfile._Cost(d["cost"])


class _UnitValue:
    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:
        return self.value


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    p = argparse.ArgumentParser(description="Re-plot benchmark results from a results directory.")
    p.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directory containing *_profiles.json files produced by benchmark_model.py.",
    )
    cli = p.parse_args()

    results_dir: Path = cli.results_dir
    if not results_dir.exists():
        logger.error("Results dir %s does not exist", results_dir)
        raise SystemExit(1)

    found_any = False
    for json_file in sorted(results_dir.glob("*_profiles.json")):
        model_name, raw_profiles = _load_profiles_from_json(json_file)
        profiles_wrapped = [_DictProfile(r) for r in raw_profiles]
        logger.info("Plotting %s …", json_file.name)
        plot_profiles(profiles_wrapped, model_name=model_name, output_dir=results_dir)  # type: ignore[arg-type]
        found_any = True

    if not found_any:
        logger.warning("No *_profiles.json files found in %s", results_dir)
