#!/usr/bin/env python3
"""Evaluate a local causal LM on a subset of GSM8K."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


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

BOXED_RE = re.compile(r"\\boxed\s*\{")
NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-path", type=Path, default=None, help="Optional local GSM8K jsonl path.")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--split", default="test")
    return parser.parse_args()


def extract_balanced_braces(text: str, start_idx: int) -> str | None:
    depth = 0
    buf: list[str] = []
    for idx in range(start_idx, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
            if depth > 1:
                buf.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(buf).strip()
            if depth < 0:
                return None
            buf.append(ch)
        elif depth >= 1:
            buf.append(ch)
    return None


def extract_boxed_answers(text: str) -> list[str]:
    answers: list[str] = []
    for match in BOXED_RE.finditer(text):
        brace_start = text.find("{", match.start())
        if brace_start < 0:
            continue
        boxed = extract_balanced_braces(text, brace_start)
        if boxed is not None:
            answers.append(boxed)
    return answers


def normalize_answer(text: str) -> str:
    text = text.strip()
    text = text.replace("$", "")
    text = text.replace(",", "")
    text = text.strip()
    return text


def extract_pred_answer(text: str) -> str:
    boxed_answers = extract_boxed_answers(text)
    if boxed_answers:
        # Prefer the last boxed answer in case the model rambles and revises itself.
        return normalize_answer(boxed_answers[-1])

    matches = NUMBER_RE.findall(text)
    if matches:
        return normalize_answer(matches[-1])
    return normalize_answer(text)


def extract_gold_answer(answer: str) -> str:
    if "####" in answer:
        answer = answer.split("####", 1)[1]
    return normalize_answer(answer)


def build_prompts(tokenizer, questions: list[str]) -> list[str]:
    prompts: list[str] = []
    for question in questions:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return prompts


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.dataset_path is not None:
        dataset = load_dataset("json", data_files=str(args.dataset_path), split="train")
    else:
        dataset = load_dataset("gsm8k", "main", split=args.split)
    dataset = dataset.select(range(min(args.limit, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    num_correct = 0
    results: list[dict[str, object]] = []

    for start in range(0, len(dataset), args.batch_size):
        batch = dataset[start : start + args.batch_size]
        questions = batch["question"]
        gold_answers = [extract_gold_answer(item) for item in batch["answer"]]
        prompts = build_prompts(tokenizer, questions)
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        new_tokens = generated[:, input_len:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        for question, gold_answer, output in zip(questions, gold_answers, decoded):
            pred_answer = extract_pred_answer(output)
            correct = pred_answer == gold_answer
            num_correct += int(correct)
            results.append(
                {
                    "question": question,
                    "gold_answer": gold_answer,
                    "pred_answer": pred_answer,
                    "correct": correct,
                    "output": output.strip(),
                }
            )

    summary = {
        "model": args.model,
        "limit": len(dataset),
        "num_correct": num_correct,
        "accuracy": num_correct / len(dataset) if len(dataset) else 0.0,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
    }

    payload = {"summary": summary, "results": results}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
