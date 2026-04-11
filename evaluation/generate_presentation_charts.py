#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT_DIR = ROOT / "presentation_assets"

MODELS = ["Base", "SFT", "GRPO"]
COLORS = {
    "Base": "#4C78A8",
    "SFT": "#F58518",
    "GRPO": "#54A24B",
}
FIG_BG = "#F6F8FB"
AX_BG = "#FFFFFF"
GRID = "#D8E0EA"
TEXT = "#2D2D2D"
TITLE = "#203040"
SPINE = "#C9D3DF"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct_axis(ax) -> None:
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 1.05)


def pct_axis_x(ax) -> None:
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0, 1.05)


def style_axes(ax, title: str, grid_axis: str = "y") -> None:
    ax.set_facecolor(AX_BG)
    ax.set_title(title, fontsize=14, weight="bold", color=TITLE, pad=12)
    ax.tick_params(labelsize=10, colors=TEXT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE)
    ax.spines["bottom"].set_color(SPINE)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1, alpha=0.7, zorder=0)


def annotate_bars(ax, bars, fmt: str = "{:.1%}") -> None:
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.015,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=9,
            color=TEXT,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="none", alpha=0.92),
        )


def annotate_bars_h(ax, bars, fmt: str = "{:.1%}") -> None:
    for bar in bars:
        w = bar.get_width()
        ax.text(
            min(w + 0.015, 1.02),
            bar.get_y() + bar.get_height() / 2,
            fmt.format(w),
            ha="left",
            va="center",
            fontsize=9,
            color=TEXT,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="none", alpha=0.92),
        )


def grouped_bars(ax, categories: list[str], series: dict[str, list[float]], title: str) -> None:
    x = np.arange(len(categories))
    width = 0.22
    offsets = np.linspace(-width, width, len(series))
    for offset, (label, values) in zip(offsets, series.items(), strict=True):
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=label,
            color=COLORS[label],
            edgecolor="#FFFFFF",
            linewidth=1.2,
            zorder=3,
        )
        annotate_bars(ax, bars)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    style_axes(ax, title, grid_axis="y")
    ax.legend(
        frameon=False,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(0.01, 1.0),
        fontsize=10,
    )
    pct_axis(ax)


def grouped_bars_horizontal(ax, categories: list[str], series: dict[str, list[float]], title: str) -> None:
    y = np.arange(len(categories))
    height = 0.22
    offsets = np.linspace(-height, height, len(series))
    for offset, (label, values) in zip(offsets, series.items(), strict=True):
        bars = ax.barh(
            y + offset,
            values,
            height=height,
            label=label,
            color=COLORS[label],
            edgecolor="#FFFFFF",
            linewidth=1.2,
            zorder=3,
        )
        annotate_bars_h(ax, bars)
    ax.set_yticks(y)
    ax.set_yticklabels(categories)
    ax.invert_yaxis()
    style_axes(ax, title, grid_axis="x")
    ax.legend(
        frameon=False,
        ncol=1,
        loc="lower right",
        bbox_to_anchor=(0.99, 1.0),
        fontsize=10,
    )
    pct_axis_x(ax)


def save_fig(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(FIG_BG)
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=240, bbox_inches="tight", facecolor=FIG_BG)
    plt.close(fig)


def load_all_data() -> dict:
    base_gsm = load_json(RESULTS / "base_gsm8k_xml_1024_300.json")["summary"]
    sft_gsm = load_json(RESULTS / "sft_gsm8k_xml_1024_300.json")["summary"]
    grpo_gsm = load_json(RESULTS / "grpo_gsm8k_xml_1024_300_greedy.json")["summary"]

    base_arc = load_json(RESULTS / "lm_eval_general" / "base_arc_easy_300" / "base_arc_easy_300_summary.json")["tasks"]["arc_easy"]
    sft_arc = load_json(RESULTS / "lm_eval_general" / "sft_arc_easy_300_adapter" / "sft_arc_easy_300_adapter_summary.json")["tasks"]["arc_easy"]
    grpo_arc = load_json(RESULTS / "lm_eval_general" / "grpo_arc_easy_300" / "grpo_arc_easy_300_summary.json")["tasks"]["arc_easy"]

    return {
        "gsm": {"Base": base_gsm, "SFT": sft_gsm, "GRPO": grpo_gsm},
        "arc": {"Base": base_arc, "SFT": sft_arc, "GRPO": grpo_arc},
    }


def chart_gsm8k_core_metrics(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    categories = ["Accuracy", "Strict\naccuracy", "Content\naccuracy"]
    series = {
        label: [
            payload["accuracy"],
            payload["strict_accuracy"],
            payload["content_accuracy"],
        ]
        for label, payload in data["gsm"].items()
    }
    grouped_bars(ax, categories, series, "GSM8K XML Core Metrics")
    save_fig(fig, "chart_gsm8k_core_metrics.png")


def chart_slide1_benchmark_snapshot(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    categories = [
        "GSM8K XML\naccuracy",
        "GSM8K XML\nstrict acc.",
        "GSM8K XML\ncontent acc.",
        "ARC-Easy\nacc_norm",
    ]
    series = {
        "Base": [
            data["gsm"]["Base"]["accuracy"],
            data["gsm"]["Base"]["strict_accuracy"],
            data["gsm"]["Base"]["content_accuracy"],
            data["arc"]["Base"]["acc_norm,none"],
        ],
        "SFT": [
            data["gsm"]["SFT"]["accuracy"],
            data["gsm"]["SFT"]["strict_accuracy"],
            data["gsm"]["SFT"]["content_accuracy"],
            data["arc"]["SFT"]["acc_norm,none"],
        ],
        "GRPO": [
            data["gsm"]["GRPO"]["accuracy"],
            data["gsm"]["GRPO"]["strict_accuracy"],
            data["gsm"]["GRPO"]["content_accuracy"],
            data["arc"]["GRPO"]["acc_norm,none"],
        ],
    }
    grouped_bars_horizontal(ax, categories, series, "Benchmark Snapshot: post-training changed different metrics")
    fig.text(
        0.5,
        0.02,
        "GRPO leads on answer landing, SFT leads on content accuracy, and ARC-Easy does not show a general-capability gain.",
        ha="center",
        fontsize=11,
        color=TEXT,
    )
    save_fig(fig, "chart_slide1_benchmark_snapshot_v2.png")


def chart_format_vs_content(data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))

    categories_left = ["Strict\naccuracy", "Content\naccuracy"]
    series_left = {
        label: [
            payload["strict_accuracy"],
            payload["content_accuracy"],
        ]
        for label, payload in data["gsm"].items()
    }
    grouped_bars(axes[0], categories_left, series_left, "Format vs Content Outcomes")

    categories_right = ["Explicit final\nanswer rate", "Has <think>\nrate", "Has <answer>\nrate"]
    series_right = {
        label: [
            payload["explicit_final_answer_rate"],
            payload["has_think_rate"],
            payload["has_answer_rate"],
        ]
        for label, payload in data["gsm"].items()
    }
    grouped_bars(axes[1], categories_right, series_right, "Protocol Adherence Rates")

    save_fig(fig, "chart_format_vs_content.png")




def chart_arc_easy_general_capability(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    categories = ["Accuracy", "Normalized\naccuracy"]
    series = {
        label: [
            payload["acc,none"],
            payload["acc_norm,none"],
        ]
        for label, payload in data["arc"].items()
    }
    grouped_bars(ax, categories, series, "ARC-Easy General Capability Check")
    save_fig(fig, "chart_arc_easy_general_capability.png")

def chart_sft_vs_base_why_better(data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4))

    uplift_metrics = [
        ("Content\naccuracy", data["gsm"]["Base"]["content_accuracy"], data["gsm"]["SFT"]["content_accuracy"]),
        ("Strict\naccuracy", data["gsm"]["Base"]["strict_accuracy"], data["gsm"]["SFT"]["strict_accuracy"]),
        ("Format-cond.\naccuracy", data["gsm"]["Base"]["format_conditioned_accuracy"], data["gsm"]["SFT"]["format_conditioned_accuracy"]),
        ("Explicit final\nanswer rate", data["gsm"]["Base"]["explicit_final_answer_rate"], data["gsm"]["SFT"]["explicit_final_answer_rate"]),
    ]
    categories_left = [m[0] for m in uplift_metrics]
    series_left = {
        "Base": [m[1] for m in uplift_metrics],
        "SFT": [m[2] for m in uplift_metrics],
    }
    grouped_bars(axes[0], categories_left, series_left, "Why SFT Beats Base: More Content + Better Landing")

    stability_metrics = [
        ("Degeneration\nrate", data["gsm"]["Base"]["reasoning_quality"]["degeneration_rate"], data["gsm"]["SFT"]["reasoning_quality"]["degeneration_rate"]),
        ("Repetition\nrate", data["gsm"]["Base"]["reasoning_quality"]["repetition_rate"], data["gsm"]["SFT"]["reasoning_quality"]["repetition_rate"]),
        ("Computation\nerror rate", data["gsm"]["Base"]["error_classification"]["rates"]["computation_error"], data["gsm"]["SFT"]["error_classification"]["rates"]["computation_error"]),
    ]
    categories_right = [m[0] for m in stability_metrics]
    series_right = {
        "Base": [m[1] for m in stability_metrics],
        "SFT": [m[2] for m in stability_metrics],
    }
    grouped_bars(axes[1], categories_right, series_right, "Why SFT Beats Base: Less Severe Degeneration")
    save_fig(fig, "chart_sft_vs_base_why_better.png")


def chart_format_coverage_diagnostic(data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3))
    x = np.arange(len(MODELS))

    explicit_counts = []
    strict_counts = []
    fmt_acc = []
    for label in MODELS:
        payload = data["gsm"][label]
        explicit_counts.append(round(payload["explicit_final_answer_rate"] * payload["limit"]))
        strict_counts.append(payload["strict_num_correct"])
        fmt_acc.append(payload["format_conditioned_accuracy"])

    width = 0.36
    bars1 = axes[0].bar(
        x - width / 2,
        explicit_counts,
        width=width,
        color="#9C755F",
        label="explicit answer count",
        edgecolor="#FFFFFF",
        linewidth=1.2,
        zorder=3,
    )
    bars2 = axes[0].bar(
        x + width / 2,
        strict_counts,
        width=width,
        color="#4C78A8",
        label="strict correct",
        edgecolor="#FFFFFF",
        linewidth=1.2,
        zorder=3,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(MODELS)
    style_axes(axes[0], "Format Coverage vs Strict Correct Count", grid_axis="y")
    axes[0].set_ylim(0, 320)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                h + 4,
                f"{int(h)}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=TEXT,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="none", alpha=0.92),
            )

    bars3 = axes[1].bar(MODELS, fmt_acc, color=[COLORS[m] for m in MODELS], edgecolor="#FFFFFF", linewidth=1.2, zorder=3)
    pct_axis(axes[1])
    style_axes(axes[1], "Accuracy Within Well-Formatted Answers", grid_axis="y")
    annotate_bars(axes[1], bars3)

    save_fig(fig, "chart_format_coverage_diagnostic.png")


def chart_failure_modes(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.8))
    keys = [
        ("correct", "#54A24B"),
        ("degeneration", "#E45756"),
        ("truncation", "#F2CF5B"),
        ("computation_error", "#4C78A8"),
        ("format_only_error", "#B279A2"),
        ("reasoning_error", "#9D755D"),
    ]

    y = np.arange(len(MODELS))
    left = np.zeros(len(MODELS))
    for key, color in keys:
        values = []
        for label in MODELS:
            rates = data["gsm"][label]["error_classification"]["rates"]
            values.append(rates.get(key, 0.0))
        ax.barh(y, values, left=left, color=color, label=key.replace("_", " "), height=0.58, edgecolor="#FFFFFF", linewidth=1.0)
        left += np.array(values)

    ax.set_yticks(y)
    ax.set_yticklabels(MODELS)
    style_axes(ax, "GSM8K XML Failure Mode Composition", grid_axis="x")
    pct_axis_x(ax)
    ax.legend(frameon=False, ncol=3, bbox_to_anchor=(0.5, -0.12), loc="upper center", fontsize=9)
    save_fig(fig, "chart_failure_modes.png")


def main() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": FIG_BG,
            "axes.facecolor": AX_BG,
            "axes.edgecolor": SPINE,
            "axes.labelcolor": TEXT,
            "text.color": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "font.family": "DejaVu Sans",
        }
    )
    data = load_all_data()
    chart_slide1_benchmark_snapshot(data)
    chart_gsm8k_core_metrics(data)
    chart_format_vs_content(data)
    chart_arc_easy_general_capability(data)
    chart_sft_vs_base_why_better(data)
    chart_format_coverage_diagnostic(data)
    chart_failure_modes(data)
    print(f"Saved charts to: {OUT_DIR}")


if __name__ == "__main__":
    main()
