#!/usr/bin/env python3
"""Compare multiple evaluation JSON files and export summary tables.

Supports the enhanced evaluation metrics including:
  - Mathematical capability (accuracy, strict, pass@k)
  - Format adherence (XML structure, boxed answers)
  - Reasoning quality (repetition, degeneration, step count)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MATH_METRIC_COLUMNS = [
    "accuracy",
    "strict_accuracy",
    "content_accuracy",
    "format_conditioned_accuracy",
]

PASS_K_PREFIX = "pass@"

FORMAT_METRIC_COLUMNS = [
    "explicit_final_answer_rate",
    "truncated_rate",
    "has_think_rate",
    "has_answer_rate",
    "has_single_boxed_rate",
    "exact_xml_boxed_rate",
]

REASONING_QUALITY_COLUMNS = [
    "avg_word_count",
    "avg_reasoning_steps",
    "truncation_rate",
    "repetition_rate",
    "degeneration_rate",
]

ERROR_TYPES = [
    "correct",
    "truncation",
    "degeneration",
    "format_only_error",
    "computation_error",
    "reasoning_error",
    "comprehension_error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        type=Path,
        help="Evaluation JSON files from evaluate_math_model.py",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels matching --inputs order, e.g. base sft rl",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path for merged comparison JSON.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Optional output path for Markdown table.",
    )
    parser.add_argument(
        "--show-by-difficulty",
        action="store_true",
        help="Include difficulty-bucket comparison in Markdown output.",
    )
    parser.add_argument(
        "--show-by-topic",
        action="store_true",
        help="Include topic comparison in Markdown output.",
    )
    parser.add_argument(
        "--top-topics",
        type=int,
        default=10,
        help="Max number of topics to show in Markdown output.",
    )
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return str(value)


def build_row(label: str, payload: dict[str, Any], source_path: Path) -> dict[str, Any]:
    summary = payload.get("summary", {})
    row = {
        "label": label,
        "source_file": str(source_path),
        "model": summary.get("model"),
        "adapter": summary.get("adapter"),
        "dataset": summary.get("dataset"),
        "split": summary.get("split"),
        "limit": summary.get("limit"),
        "num_correct": summary.get("num_correct"),
        "content_correct_count": summary.get("content_correct_count"),
        **{metric: summary.get(metric) for metric in MATH_METRIC_COLUMNS + FORMAT_METRIC_COLUMNS},
    }

    pass_at_k = summary.get("pass_at_k", {})
    for key, value in pass_at_k.items():
        row[key] = value

    rq = summary.get("reasoning_quality", {})
    for col in REASONING_QUALITY_COLUMNS:
        row[f"rq_{col}"] = rq.get(col)

    ec = summary.get("error_classification", {})
    ec_rates = ec.get("rates", {})
    for et in ERROR_TYPES:
        row[f"err_{et}"] = ec_rates.get(et)

    return row


def math_markdown_table(rows: list[dict[str, Any]]) -> str:
    pass_k_keys = sorted({
        k for row in rows for k in row
        if k.startswith(PASS_K_PREFIX)
    })

    headers = [
        "label",
        "dataset",
        "limit",
        "num_correct",
        "accuracy",
        "strict_acc",
        "content_acc",
        "fmt_cond_acc",
    ] + pass_k_keys

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            str(row.get("label", "-")),
            str(row.get("dataset", "-")),
            str(row.get("limit", "-")),
            str(row.get("num_correct", "-")),
            pct(row.get("accuracy")),
            pct(row.get("strict_accuracy")),
            pct(row.get("content_accuracy")),
            pct(row.get("format_conditioned_accuracy")),
        ] + [pct(row.get(k)) for k in pass_k_keys]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def format_markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "label",
        "dataset",
        "limit",
        "fmt_rate",
        "truncated",
        "has_think",
        "has_answer",
        "single_boxed",
        "exact_xml",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            str(row.get("label", "-")),
            str(row.get("dataset", "-")),
            str(row.get("limit", "-")),
            pct(row.get("explicit_final_answer_rate")),
            pct(row.get("truncated_rate")),
            pct(row.get("has_think_rate")),
            pct(row.get("has_answer_rate")),
            pct(row.get("has_single_boxed_rate")),
            pct(row.get("exact_xml_boxed_rate")),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def reasoning_quality_markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "label",
        "dataset",
        "avg_words",
        "avg_steps",
        "truncation",
        "repetition",
        "degeneration",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            str(row.get("label", "-")),
            str(row.get("dataset", "-")),
            str(row.get("rq_avg_word_count", "-")),
            str(row.get("rq_avg_reasoning_steps", "-")),
            pct(row.get("rq_truncation_rate")),
            pct(row.get("rq_repetition_rate")),
            pct(row.get("rq_degeneration_rate")),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def compare_group_metric(
    labeled_payloads: list[tuple[str, dict[str, Any]]],
    group_key: str,
    metric_key: str,
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    all_groups: set[str] = set()
    for _, payload in labeled_payloads:
        all_groups.update(payload.get(group_key, {}).keys())

    for group_name in sorted(all_groups):
        merged[group_name] = {}
        for label, payload in labeled_payloads:
            value = payload.get(group_key, {}).get(group_name, {}).get(metric_key)
            merged[group_name][label] = value
    return merged


def difficulty_markdown(labeled_payloads: list[tuple[str, dict[str, Any]]]) -> str:
    merged = compare_group_metric(labeled_payloads, "by_difficulty_bucket", "accuracy")
    if not merged:
        return ""

    labels = [label for label, _ in labeled_payloads]
    lines = [
        "## By Difficulty",
        "",
        "| difficulty_bucket | " + " | ".join(labels) + " |",
        "| " + " | ".join(["---"] * (len(labels) + 1)) + " |",
    ]
    for bucket, values in merged.items():
        row = [bucket] + [pct(values.get(label)) for label in labels]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def error_classification_markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "label",
        "correct",
        "truncation",
        "degeneration",
        "fmt_only",
        "computation",
        "reasoning",
        "comprehension",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            str(row.get("label", "-")),
            pct(row.get("err_correct")),
            pct(row.get("err_truncation")),
            pct(row.get("err_degeneration")),
            pct(row.get("err_format_only_error")),
            pct(row.get("err_computation_error")),
            pct(row.get("err_reasoning_error")),
            pct(row.get("err_comprehension_error")),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def topic_markdown(labeled_payloads: list[tuple[str, dict[str, Any]]], top_topics: int) -> str:
    merged = compare_group_metric(labeled_payloads, "by_topic", "accuracy")
    if not merged:
        return ""

    topic_strength: list[tuple[str, float]] = []
    for topic, values in merged.items():
        numeric_values = [float(v) for v in values.values() if v is not None]
        if numeric_values:
            topic_strength.append((topic, max(numeric_values)))
    topic_strength.sort(key=lambda item: item[1], reverse=True)
    selected_topics = [topic for topic, _ in topic_strength[:top_topics]]

    labels = [label for label, _ in labeled_payloads]
    lines = [
        "## By Topic",
        "",
        "| topic | " + " | ".join(labels) + " |",
        "| " + " | ".join(["---"] * (len(labels) + 1)) + " |",
    ]
    for topic in selected_topics:
        values = merged[topic]
        row = [topic] + [pct(values.get(label)) for label in labels]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.labels is not None and args.labels and len(args.labels) != len(args.inputs):
        raise ValueError("--labels count must match --inputs count")

    labels = args.labels or [path.stem for path in args.inputs]
    payloads = [load_payload(path) for path in args.inputs]
    labeled_payloads = list(zip(labels, payloads))

    rows = [
        build_row(label=label, payload=payload, source_path=path)
        for (label, payload), path in zip(labeled_payloads, args.inputs)
    ]

    pass_k_keys = sorted({
        k for row in rows for k in row
        if k.startswith(PASS_K_PREFIX)
    })

    comparison = {
        "runs": rows,
        "math_capability": [
            {
                "label": row["label"],
                "dataset": row.get("dataset"),
                "limit": row.get("limit"),
                "num_correct": row.get("num_correct"),
                "accuracy": row.get("accuracy"),
                "content_accuracy": row.get("content_accuracy"),
                **{k: row.get(k) for k in pass_k_keys},
            }
            for row in rows
        ],
        "format_adherence": [
            {
                "label": row["label"],
                "dataset": row.get("dataset"),
                "limit": row.get("limit"),
                "has_think_rate": row.get("has_think_rate"),
                "has_answer_rate": row.get("has_answer_rate"),
                "has_single_boxed_rate": row.get("has_single_boxed_rate"),
                "exact_xml_boxed_rate": row.get("exact_xml_boxed_rate"),
            }
            for row in rows
        ],
        "error_classification": [
            {
                "label": row["label"],
                "dataset": row.get("dataset"),
                **{et: row.get(f"err_{et}") for et in ERROR_TYPES},
            }
            for row in rows
        ],
        "reasoning_quality": [
            {
                "label": row["label"],
                "dataset": row.get("dataset"),
                **{col: row.get(f"rq_{col}") for col in REASONING_QUALITY_COLUMNS},
            }
            for row in rows
        ],
        "by_difficulty_bucket_accuracy": compare_group_metric(
            labeled_payloads, "by_difficulty_bucket", "accuracy"
        ),
        "by_topic_accuracy": compare_group_metric(labeled_payloads, "by_topic", "accuracy"),
    }

    md_sections = [
        "# Evaluation Comparison",
        "",
        "## Mathematical Capability",
        "",
        "- **accuracy**: loose extraction (last number fallback)",
        "- **strict_acc**: only counts answers in explicit format (boxed/Final Answer/answer tag)",
        "- **content_acc**: correct answer appears anywhere in output (measures true math ability)",
        "- **fmt_cond_acc**: accuracy among samples with explicit format (format-independent correctness)",
        "",
        math_markdown_table(rows),
        "",
        "## Error Classification",
        "",
        "Breakdown of failure modes (rates sum to 1.0):",
        "- **truncation**: output cut off before final answer",
        "- **degeneration**: repetitive/degenerate output",
        "- **fmt_only**: correct answer in output but extraction failed",
        "- **computation**: numeric answer but wrong value",
        "- **reasoning**: structured reasoning but wrong conclusion",
        "- **comprehension**: misunderstood the problem",
        "",
        error_classification_markdown_table(rows),
        "",
        "## Format Adherence",
        "",
        format_markdown_table(rows),
        "",
        "## Reasoning Quality",
        "",
        reasoning_quality_markdown_table(rows),
    ]
    if args.show_by_difficulty:
        section = difficulty_markdown(labeled_payloads)
        if section:
            md_sections.extend(["", section])
    if args.show_by_topic:
        section = topic_markdown(labeled_payloads, args.top_topics)
        if section:
            md_sections.extend(["", section])
    md_text = "\n".join(md_sections).strip() + "\n"

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(md_text, encoding="utf-8")

    print(md_text)


if __name__ == "__main__":
    main()
