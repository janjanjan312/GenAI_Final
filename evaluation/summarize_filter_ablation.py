#!/usr/bin/env python3
"""Summarize data filtering ablation metadata and evaluation outputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METADATA_DIR = REPO_ROOT / "ablation" / "metadata"
DEFAULT_RESULTS_DIR = REPO_ROOT / "evaluation" / "results" / "filter_ablation"


@dataclass
class VariantRecord:
    name: str
    safe_name: str
    kept_examples: int
    variant_overrides: dict[str, Any]
    removed_rule: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_RESULTS_DIR / "filter_ablation_summary.json")
    parser.add_argument("--output-md", type=Path, default=DEFAULT_RESULTS_DIR / "filter_ablation_summary.md")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def format_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.2f}%"


def format_delta(value: int | float | None, pct: bool = False) -> str:
    if value is None:
        return "-"
    if pct:
        return f"{value * 100:+.2f} pp"
    return f"{value:+}"


def describe_removed_rule(name: str, overrides: dict[str, Any]) -> str:
    if name == "00_full_all_rules":
        return "baseline"
    if "ambiguity_regexes" in overrides:
        return "remove ambiguity regex filter"
    if overrides.get("require_any_r1_solution") is False:
        return "remove r1-solution requirement"
    if overrides.get("min_difficulty") == -1000000000.0 and overrides.get("max_difficulty") == 1000000000.0:
        return "remove difficulty band"
    if overrides.get("min_question_chars") == 0:
        return "remove min question length"
    if overrides.get("max_question_chars") == 10000000:
        return "remove max question length"
    if overrides.get("min_final_answer_chars") == 0:
        return "remove min answer length"
    if overrides.get("max_final_answer_chars") == 10000000:
        return "remove max answer length"
    return "custom override"


def load_variants(metadata_dir: Path) -> list[VariantRecord]:
    records: list[VariantRecord] = []
    for path in sorted(metadata_dir.glob("*.json")):
        payload = load_json(path)
        overrides = payload.get("variant_overrides", {})
        name = payload["name"]
        records.append(
            VariantRecord(
                name=name,
                safe_name=payload["safe_name"],
                kept_examples=int(payload["kept_examples"]),
                variant_overrides=overrides,
                removed_rule=describe_removed_rule(name, overrides),
            )
        )
    return records


def load_eval_summary(results_dir: Path, dataset_name: str, variant_name: str) -> dict[str, Any] | None:
    path = results_dir / dataset_name / f"{variant_name}.json"
    if not path.exists():
        return None
    payload = load_json(path)
    return payload.get("summary", {})


def build_payload(variants: list[VariantRecord], results_dir: Path) -> dict[str, Any]:
    baseline = next(v for v in variants if v.name == "00_full_all_rules")
    payload: dict[str, Any] = {
        "baseline": {
            "name": baseline.name,
            "kept_examples": baseline.kept_examples,
        },
        "variants": [],
    }

    for variant in variants:
        gsm = load_eval_summary(results_dir, "gsm8k", variant.name)
        deepmath = load_eval_summary(results_dir, "deepmath_smoke", variant.name)
        record = {
            "name": variant.name,
            "removed_rule": variant.removed_rule,
            "variant_overrides": variant.variant_overrides,
            "kept_examples": variant.kept_examples,
            "kept_delta": variant.kept_examples - baseline.kept_examples,
            "gsm8k": gsm,
            "deepmath_smoke": deepmath,
        }
        for dataset_key in ("gsm8k", "deepmath_smoke"):
            dataset_summary = record[dataset_key]
            baseline_summary = load_eval_summary(results_dir, dataset_key, baseline.name)
            if dataset_summary and baseline_summary:
                record[f"{dataset_key}_delta"] = {
                    "accuracy": dataset_summary.get("accuracy", 0.0) - baseline_summary.get("accuracy", 0.0),
                    "strict_accuracy": dataset_summary.get("strict_accuracy", 0.0) - baseline_summary.get("strict_accuracy", 0.0),
                    "content_accuracy": dataset_summary.get("content_accuracy", 0.0) - baseline_summary.get("content_accuracy", 0.0),
                    "exact_xml_boxed_rate": dataset_summary.get("exact_xml_boxed_rate", 0.0) - baseline_summary.get("exact_xml_boxed_rate", 0.0),
                    "degeneration_rate": dataset_summary.get("reasoning_quality", {}).get("degeneration_rate", 0.0)
                    - baseline_summary.get("reasoning_quality", {}).get("degeneration_rate", 0.0),
                    "truncation_rate": dataset_summary.get("reasoning_quality", {}).get("truncation_rate", 0.0)
                    - baseline_summary.get("reasoning_quality", {}).get("truncation_rate", 0.0),
                }
        payload["variants"].append(record)
    return payload


def make_dataset_table(variants: list[dict[str, Any]], dataset_key: str) -> str:
    headers = [
        "variant",
        "removed_rule",
        "kept",
        "delta_kept",
        "accuracy",
        "strict_acc",
        "content_acc",
        "exact_xml",
        "degeneration",
        "truncation",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for variant in variants:
        summary = variant.get(dataset_key) or {}
        rq = summary.get("reasoning_quality", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    variant["name"],
                    variant["removed_rule"],
                    str(variant["kept_examples"]),
                    format_delta(variant["kept_delta"]),
                    format_pct(summary.get("accuracy")),
                    format_pct(summary.get("strict_accuracy")),
                    format_pct(summary.get("content_accuracy")),
                    format_pct(summary.get("exact_xml_boxed_rate")),
                    format_pct(rq.get("degeneration_rate")),
                    format_pct(rq.get("truncation_rate")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def make_delta_table(variants: list[dict[str, Any]], dataset_key: str) -> str:
    headers = [
        "variant",
        "acc_delta",
        "strict_delta",
        "content_delta",
        "xml_delta",
        "degeneration_delta",
        "truncation_delta",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for variant in variants:
        delta = variant.get(f"{dataset_key}_delta") or {}
        if not delta:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    variant["name"],
                    format_delta(delta.get("accuracy"), pct=True),
                    format_delta(delta.get("strict_accuracy"), pct=True),
                    format_delta(delta.get("content_accuracy"), pct=True),
                    format_delta(delta.get("exact_xml_boxed_rate"), pct=True),
                    format_delta(delta.get("degeneration_rate"), pct=True),
                    format_delta(delta.get("truncation_rate"), pct=True),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def select_key_findings(variants: list[dict[str, Any]]) -> list[str]:
    findings: list[str] = []
    baseline = next(v for v in variants if v["name"] == "00_full_all_rules")
    baseline_gsm = baseline.get("gsm8k") or {}
    baseline_deep = baseline.get("deepmath_smoke") or {}

    by_kept = sorted(variants, key=lambda item: item["kept_examples"], reverse=True)
    if by_kept:
        top_kept = by_kept[0]
        if top_kept["name"] != baseline["name"]:
            findings.append(
                f"`{top_kept['name']}` has the largest retained dataset ({top_kept['kept_examples']}, "
                f"{top_kept['kept_examples'] - baseline['kept_examples']:+} vs baseline)."
            )

    def best_variant(dataset_key: str, metric_key: str) -> tuple[str, float] | None:
        candidates = []
        for variant in variants:
            summary = variant.get(dataset_key) or {}
            if metric_key in summary:
                candidates.append((variant["name"], float(summary[metric_key])))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[1])

    for dataset_key, baseline_summary in (("gsm8k", baseline_gsm), ("deepmath_smoke", baseline_deep)):
        best_acc = best_variant(dataset_key, "accuracy")
        best_strict = best_variant(dataset_key, "strict_accuracy")
        if best_acc:
            findings.append(
                f"On `{dataset_key}`, best `accuracy` is `{best_acc[0]}` at {best_acc[1] * 100:.2f}% "
                f"(baseline {baseline_summary.get('accuracy', 0.0) * 100:.2f}%)."
            )
        if best_strict:
            findings.append(
                f"On `{dataset_key}`, best `strict_accuracy` is `{best_strict[0]}` at {best_strict[1] * 100:.2f}% "
                f"(baseline {baseline_summary.get('strict_accuracy', 0.0) * 100:.2f}%)."
            )
    return findings


def render_markdown(payload: dict[str, Any]) -> str:
    variants = payload["variants"]
    findings = select_key_findings(variants)
    lines = [
        "# Data Filtering Sensitivity Summary",
        "",
        "## Scope",
        "",
        "This summary reuses the existing rule-ablation LoRA adapters in `ablation/runs` and combines:",
        "- data-side changes from `ablation/metadata/*.json`",
        "- downstream evaluation outputs under `evaluation/results/filter_ablation`",
        "",
        "## Key Findings",
        "",
    ]
    if findings:
        lines.extend([f"- {item}" for item in findings])
    else:
        lines.append("- Evaluation outputs are not complete yet.")

    lines.extend(
        [
            "",
            "## Data-Side Summary",
            "",
            "| variant | removed_rule | kept_examples | kept_delta | overrides |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for variant in variants:
        lines.append(
            "| "
            + " | ".join(
                [
                    variant["name"],
                    variant["removed_rule"],
                    str(variant["kept_examples"]),
                    format_delta(variant["kept_delta"]),
                    json.dumps(variant["variant_overrides"], ensure_ascii=False),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## DeepMath Smoke Metrics",
            "",
            make_dataset_table(variants, "deepmath_smoke"),
            "",
            "### DeepMath Smoke Deltas vs Baseline",
            "",
            make_delta_table(variants, "deepmath_smoke"),
            "",
            "## GSM8K Metrics",
            "",
            make_dataset_table(variants, "gsm8k"),
            "",
            "### GSM8K Deltas vs Baseline",
            "",
            make_delta_table(variants, "gsm8k"),
            "",
            "## Presentation Framing",
            "",
            "1. Start with how much each removed rule changes `kept_examples`.",
            "2. Then compare whether the same rule changes mostly affect `accuracy`, `strict_accuracy`, or degeneration/truncation.",
            "3. Highlight trade-offs: larger dataset vs cleaner supervision vs better protocol adherence.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    variants = load_variants(args.metadata_dir)
    payload = build_payload(variants, args.results_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Wrote JSON summary to: {args.output_json}")
    print(f"Wrote Markdown summary to: {args.output_md}")


if __name__ == "__main__":
    main()
