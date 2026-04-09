#!/usr/bin/env python3
"""Prepare a TRL-compatible GRPO dataset from DeepMath Phase A parquet."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset


SYSTEM_PROMPT = """You are a careful math solver.

Respond in exactly this XML format:
<think>
brief reasoning
</think>
<answer>
\\boxed{final answer}
</answer>

Rules:
1. Keep the reasoning concise.
2. Put exactly one final answer inside \\boxed{...}.
3. Do not add any text before <think> or after </answer>.
"""

RESPONSE_PREFIX = "<think>\n"


def build_prompt(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
        {"role": "assistant", "content": RESPONSE_PREFIX},
    ]


def convert_row(row_id: int, row: dict) -> dict:
    question = (row.get("question") or "").strip()
    ground_truth = (row.get("final_answer_norm") or row.get("final_answer") or "").strip()

    return {
        "prompt": build_prompt(question),
        "ground_truth": ground_truth,
        "question": question,
        "topic": row.get("topic"),
        "difficulty": row.get("difficulty"),
        "answer_format": "xml_boxed",
        "response_prefix": RESPONSE_PREFIX,
        "data_source": "deepmath_phase_a",
        "row_id": row_id,
    }


def load_examples(input_path: Path, max_samples: int | None) -> list[dict]:
    table = pq.read_table(
        input_path,
        columns=["question", "final_answer", "final_answer_norm", "difficulty", "topic"],
    )
    rows = table.to_pylist()
    if max_samples is not None:
        rows = rows[:max_samples]
    examples = [convert_row(i, row) for i, row in enumerate(rows)]
    return [example for example in examples if example["ground_truth"]]


def save_split(examples: list[dict], output_path: Path) -> None:
    dataset = Dataset.from_list(examples)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Source parquet path.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--val-size", type=int, default=1000, help="Validation sample count.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick experiments.")
    args = parser.parse_args()

    examples = load_examples(args.input, args.max_samples)
    rng = random.Random(args.seed)
    rng.shuffle(examples)

    if len(examples) < 2:
        raise ValueError("Need at least 2 usable examples to build train/val splits.")

    val_size = args.val_size if args.val_size > 0 else max(1, round(len(examples) * 0.01))
    val_size = min(val_size, len(examples) - 1)
    train_examples = examples[val_size:]
    val_examples = examples[:val_size]

    if not train_examples or not val_examples:
        raise ValueError("Train/val split is empty. Reduce --val-size or increase --max-samples.")

    train_path = args.output_dir / "train.parquet"
    val_path = args.output_dir / "val.parquet"
    save_split(train_examples, train_path)
    save_split(val_examples, val_path)

    print(f"Loaded {len(examples)} usable examples from {args.input}")
    print(f"Train set: {len(train_examples)} -> {train_path}")
    print(f"Val set: {len(val_examples)} -> {val_path}")
    print(
        "Dataset fields:",
        [
            "prompt",
            "ground_truth",
            "question",
            "topic",
            "difficulty",
            "answer_format",
            "response_prefix",
            "data_source",
            "row_id",
        ],
    )


if __name__ == "__main__":
    main()
