#!/usr/bin/env python3
"""Wrapper around lm-eval-harness for standardized benchmark evaluation.

Supports evaluating base, SFT, and RL models on:
  - GSM8K (math reasoning)
  - MATH / MATH-500 (competition math)
  - MMLU (general knowledge - for detecting RL capability degradation)
  - HellaSwag (commonsense reasoning - for detecting RL capability degradation)
  - ARC-Easy / ARC-Challenge (science reasoning)

Usage:
    # Evaluate base model on GSM8K + MMLU (detect RL degradation)
    python evaluation/run_lm_eval.py \
        --model models/Qwen2.5-0.5B \
        --tasks gsm8k mmlu \
        --output-dir evaluation/results/lm_eval

    # Evaluate SFT LoRA adapter
    python evaluation/run_lm_eval.py \
        --model models/Qwen2.5-0.5B \
        --adapter sft/qwen25_0p5b_math_lora \
        --tasks gsm8k \
        --output-dir evaluation/results/lm_eval

    # Evaluate GRPO model on math + general QA (full Topic 1 eval)
    python evaluation/run_lm_eval.py \
        --model grpo_trl_model \
        --tasks gsm8k mmlu hellaswag \
        --output-dir evaluation/results/lm_eval \
        --label grpo

    # Quick smoke test
    python evaluation/run_lm_eval.py \
        --model models/Qwen2.5-0.5B \
        --tasks gsm8k \
        --limit 50 \
        --output-dir evaluation/results/lm_eval
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


TASK_CONFIGS = {
    "gsm8k": {
        "task": "gsm8k",
        "description": "Grade-school math (8-shot CoT by default)",
        "num_fewshot": 8,
        "metric_key": "exact_match,strict-match",
    },
    "gsm8k_cot": {
        "task": "gsm8k_cot",
        "description": "GSM8K with chain-of-thought prompting",
        "num_fewshot": 8,
        "metric_key": "exact_match,strict-match",
    },
    "minerva_math": {
        "task": "minerva_math",
        "description": "MATH benchmark (competition-level math)",
        "num_fewshot": 4,
        "metric_key": "exact_match",
    },
    "mmlu": {
        "task": "mmlu",
        "description": "Massive Multitask Language Understanding (general QA)",
        "num_fewshot": 5,
        "metric_key": "acc",
    },
    "hellaswag": {
        "task": "hellaswag",
        "description": "HellaSwag commonsense reasoning",
        "num_fewshot": 10,
        "metric_key": "acc_norm",
    },
    "arc_easy": {
        "task": "arc_easy",
        "description": "ARC Easy (science reasoning, lighter)",
        "num_fewshot": 0,
        "metric_key": "acc_norm",
    },
    "arc_challenge": {
        "task": "arc_challenge",
        "description": "ARC Challenge (harder science reasoning)",
        "num_fewshot": 25,
        "metric_key": "acc_norm",
    },
    "winogrande": {
        "task": "winogrande",
        "description": "Winogrande coreference resolution",
        "num_fewshot": 5,
        "metric_key": "acc",
    },
}

TASK_PRESETS = {
    "math_only": ["gsm8k", "minerva_math"],
    "math_and_general": ["gsm8k", "minerva_math", "mmlu", "hellaswag"],
    "full": ["gsm8k", "minerva_math", "mmlu", "hellaswag", "arc_easy", "winogrande"],
    "quick": ["gsm8k", "arc_easy"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Model path or HF model ID.")
    parser.add_argument("--adapter", default=None, help="Optional LoRA adapter path (merged on-the-fly via PEFT).")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["gsm8k"],
        help=(
            "Tasks to evaluate. Can be task names (gsm8k, mmlu, etc.) "
            "or presets (math_only, math_and_general, full, quick). "
            f"Available tasks: {', '.join(TASK_CONFIGS.keys())}. "
            f"Available presets: {', '.join(TASK_PRESETS.keys())}."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results/lm_eval"))
    parser.add_argument("--label", default=None, help="Label for this run (used in output filenames).")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples per task.")
    parser.add_argument("--batch-size", default="auto", help="Batch size for lm-eval (default: auto).")
    parser.add_argument("--device", default=None, help="Device override (cuda, mps, cpu).")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--num-fewshot-override", type=int, default=None, help="Override default few-shot count for all tasks.")
    parser.add_argument(
        "--gen-kwargs",
        nargs="+",
        default=None,
        help="Optional lm-eval generation kwargs, e.g. max_gen_toks=1024 temperature=0.0.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing.")
    return parser.parse_args()


def resolve_tasks(task_specs: list[str]) -> list[str]:
    resolved = []
    for spec in task_specs:
        if spec in TASK_PRESETS:
            resolved.extend(TASK_PRESETS[spec])
        elif spec in TASK_CONFIGS:
            resolved.append(spec)
        else:
            resolved.append(spec)
    return list(dict.fromkeys(resolved))


def build_model_args(args: argparse.Namespace) -> str:
    parts = [f"pretrained={args.model}"]
    if args.adapter:
        parts.append(f"peft={args.adapter}")
    if args.dtype != "auto":
        parts.append(f"dtype={args.dtype}")
    return ",".join(parts)


def build_lm_eval_command(args: argparse.Namespace, tasks: list[str]) -> list[str]:
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", build_model_args(args),
        "--tasks", ",".join(tasks),
        "--batch_size", str(args.batch_size),
        "--output_path", str(args.output_dir),
    ]

    if args.device is not None:
        cmd.extend(["--device", args.device])

    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])

    if args.num_fewshot_override is not None:
        cmd.extend(["--num_fewshot", str(args.num_fewshot_override)])

    if args.gen_kwargs:
        cmd.extend(["--gen_kwargs", *args.gen_kwargs])

    cmd.append("--log_samples")

    return cmd


def run_evaluation(args: argparse.Namespace) -> dict[str, Any] | None:
    tasks = resolve_tasks(args.tasks)
    print(f"Tasks to evaluate: {tasks}")
    for t in tasks:
        if t in TASK_CONFIGS:
            cfg = TASK_CONFIGS[t]
            print(f"  - {t}: {cfg['description']} (default {cfg['num_fewshot']}-shot)")
        else:
            print(f"  - {t}: (custom task, will use lm-eval defaults)")

    cmd = build_lm_eval_command(args, tasks)
    print(f"\nCommand:\n  {' '.join(cmd)}\n")

    if args.dry_run:
        print("(dry run, not executing)")
        return None

    args.output_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\nlm-eval exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    return collect_results(args, tasks)


def collect_results(args: argparse.Namespace, tasks: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "model": args.model,
        "adapter": args.adapter,
        "label": args.label,
        "tasks": {},
    }

    for result_file in args.output_dir.rglob("results*.json"):
        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
            results = data.get("results", {})
            for task_name, metrics in results.items():
                summary["tasks"][task_name] = metrics
        except Exception as exc:
            print(f"Warning: could not parse {result_file}: {exc}", file=sys.stderr)

    label = args.label or Path(args.model).stem
    summary_path = args.output_dir / f"{label}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSummary saved to: {summary_path}")

    print_summary_table(summary)
    return summary


def print_summary_table(summary: dict[str, Any]) -> None:
    tasks = summary.get("tasks", {})
    if not tasks:
        print("No results found in output directory.")
        return

    print("\n" + "=" * 60)
    print(f"  Model: {summary.get('model', '?')}")
    if summary.get("adapter"):
        print(f"  Adapter: {summary['adapter']}")
    print("=" * 60)
    print(f"  {'Task':<25} {'Metric':<25} {'Value':<10}")
    print("  " + "-" * 56)

    for task_name, metrics in sorted(tasks.items()):
        for metric_name, value in sorted(metrics.items()):
            if metric_name.startswith("_") or "stderr" in metric_name:
                continue
            if isinstance(value, (int, float)):
                print(f"  {task_name:<25} {metric_name:<25} {value:.4f}")
    print("=" * 60)


def main() -> None:
    args = parse_args()

    try:
        import lm_eval  # noqa: F401
    except ImportError:
        print(
            "Error: lm-eval-harness is not installed.\n"
            "Install it with:\n"
            "  pip install lm-eval\n"
            "Or:\n"
            "  pip install lm_eval[math,multilingual]\n",
            file=sys.stderr,
        )
        sys.exit(1)

    run_evaluation(args)


if __name__ == "__main__":
    main()
