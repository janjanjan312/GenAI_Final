#!/usr/bin/env python3
"""Run a few XML-format instruction-following tests against a local model."""

from __future__ import annotations

import argparse
import json
import re

import torch
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


TEST_QUESTIONS = [
    "Compute 2+3.",
    "Solve x+2=5.",
    "What is the derivative of x^2?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def check_xml(text: str) -> dict[str, object]:
    think = bool(re.search(r"<think>\s*.*?\s*</think>", text, re.DOTALL))
    answer = bool(re.search(r"<answer>\s*.*?\s*</answer>", text, re.DOTALL))
    boxed = len(re.findall(r"\\boxed\s*\{", text))
    exact = bool(
        re.match(
            r"^\s*<think>\s*.*?\s*</think>\s*<answer>\s*\\boxed\s*\{.*?\}\s*</answer>\s*$",
            text,
            re.DOTALL,
        )
    )
    return {
        "has_think": think,
        "has_answer": answer,
        "boxed_count": boxed,
        "exact_xml_boxed": exact,
    }


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    outputs = []
    for question in TEST_QUESTIONS:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=max(args.temperature, 1e-5),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[0][inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        outputs.append(
            {
                "question": question,
                "output": text,
                "checks": check_xml(text),
            }
        )

    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
