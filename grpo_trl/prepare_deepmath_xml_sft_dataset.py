#!/usr/bin/env python3
"""Convert DeepMath alpaca JSONL into XML-format SFT train/val parquet files."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

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
1. Keep the reasoning concise and directly relevant.
2. Put the final answer only inside the <answer> block.
3. Do not add any text before <think> or after </answer>.
"""


def strip_final_answer_section(text: str) -> str:
    text = (text or "").strip()
    split_patterns = [
        r"\*\*Final Answer\*\*",
        r"Final Answer\s*:?",
        r"</think>",
        r"<answer>",
    ]
    for pattern in split_patterns:
        parts = re.split(pattern, text, flags=re.IGNORECASE, maxsplit=1)
        if len(parts) > 1:
            text = parts[0].strip()
    boxed_match = list(re.finditer(r"\\boxed\s*\{", text))
    if boxed_match:
        text = text[: boxed_match[-1].start()].strip()
    return text


def cleanup_reasoning(text: str) -> str:
    text = strip_final_answer_section(text)
    text = text.replace("\r\n", "\n")
    text = re.sub(r"^\s*Okay,\s*so\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*Hmm,\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def shorten_reasoning(text: str, max_chars: int) -> str:
    text = cleanup_reasoning(text)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rstrip()
    split_idx = max(cut.rfind("\n\n"), cut.rfind(". "), cut.rfind("。"))
    if split_idx >= max_chars // 2:
        cut = cut[:split_idx].rstrip()
    return cut.strip()


def build_assistant_output(reasoning: str, final_answer_norm: str) -> str:
    return (
        "<think>\n"
        f"{reasoning.strip()}\n"
        "</think>\n"
        "<answer>\n"
        f"\\boxed{{{final_answer_norm.strip()}}}\n"
        "</answer>"
    )


def build_messages(question: str, assistant_output: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
        {"role": "assistant", "content": assistant_output},
    ]


def convert_example(row: dict, max_think_chars: int) -> dict | None:
    question = (row.get("instruction") or "").strip()
    reasoning = shorten_reasoning(row.get("output") or "", max_think_chars)
    final_answer_norm = (row.get("final_answer_norm") or "").strip()

    if not question or not reasoning or not final_answer_norm:
        return None

    assistant_output = build_assistant_output(reasoning, final_answer_norm)
    return {
        "messages": build_messages(question, assistant_output),
        "question": question,
        "topic": row.get("topic"),
        "difficulty": row.get("difficulty"),
        "final_answer_norm": final_answer_norm,
        "answer_format": "xml_boxed",
        "source": "deepmath_phase_a_alpaca",
    }


def load_jsonl(path: Path, max_samples: int | None, max_think_chars: int) -> list[dict]:
    examples: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            example = convert_example(row, max_think_chars)
            if example is not None:
                examples.append(example)
            if max_samples is not None and len(examples) >= max_samples:
                break
    return examples


def save_split(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(examples).to_parquet(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Source alpaca JSONL path.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-think-chars", type=int, default=1200)
    args = parser.parse_args()

    examples = load_jsonl(args.input, args.max_samples, args.max_think_chars)
    if len(examples) < 2:
        raise ValueError("Need at least 2 usable examples to build train/val splits.")

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    val_size = min(args.val_size, len(examples) - 1)
    train_examples = examples[val_size:]
    val_examples = examples[:val_size]

    if not train_examples or not val_examples:
        raise ValueError("Train/val split is empty. Reduce --val-size or increase available samples.")

    train_path = args.output_dir / "train.parquet"
    val_path = args.output_dir / "val.parquet"
    save_split(train_examples, train_path)
    save_split(val_examples, val_path)

    print(f"Loaded {len(examples)} XML SFT examples from {args.input}")
    print(f"Train set: {len(train_examples)} -> {train_path}")
    print(f"Val set: {len(val_examples)} -> {val_path}")
    print("Dataset fields:", ["messages", "question", "topic", "difficulty", "final_answer_norm", "answer_format", "source"])


if __name__ == "__main__":
    main()
