#!/usr/bin/env python3
"""Evaluate math reasoning models with XML-format and answer metrics.

Enhanced evaluation supporting:
  - Greedy accuracy (strict + loose extraction)
  - Pass@k via temperature sampling (--pass-k / --num-samples)
  - Reasoning quality analysis (repetition detection, step counting)
  - Output degeneration detection
  - Format adherence (XML structure, boxed answers)
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except Exception:  # pragma: no cover
    PeftModel = None

try:
    import sympy as sp
    from sympy.parsing.latex import parse_latex
except Exception:  # pragma: no cover
    sp = None
    parse_latex = None


SYSTEM_PROMPT_XML = """You are a careful math solver.

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

SYSTEM_PROMPT_MATH_PLAIN = """You are a careful math solver.

Solve the user's math problem clearly and concisely.
Show the key reasoning steps if needed, but avoid unnecessary filler language.
End your response with a final line in this exact format:
Final Answer: <answer>
"""

THINK_RE = re.compile(r"<think>\s*.*?\s*</think>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>\s*.*?\s*</answer>", re.DOTALL | re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\s*\{")
NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Merged/local model path or HF model id.")
    parser.add_argument("--adapter", default=None, help="Optional LoRA adapter path.")
    parser.add_argument("--output", required=True, type=Path, help="JSON output path.")
    parser.add_argument("--dataset", choices=["gsm8k", "jsonl", "parquet"], default="gsm8k")
    parser.add_argument("--dataset-path", type=Path, default=None, help="Required when --dataset=jsonl or --dataset=parquet.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N examples before evaluating.")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--question-field", default="instruction")
    parser.add_argument("--answer-field", default="final_answer_norm")
    parser.add_argument("--topic-field", default="topic")
    parser.add_argument("--difficulty-field", default="difficulty")
    parser.add_argument("--prompt-style", choices=["xml", "plain", "math_plain"], default="xml")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=20, help="Save intermediate results every N examples (0 to disable).")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file, skipping already evaluated examples.")
    parser.add_argument("--save-outputs", action="store_true", help="Keep raw generations in the results JSON.")

    passk = parser.add_argument_group("Pass@k evaluation")
    passk.add_argument("--pass-k", type=int, nargs="+", default=None,
                       help="Compute pass@k for these k values (e.g. --pass-k 1 5). Requires --num-samples >= max(k).")
    passk.add_argument("--num-samples", type=int, default=1,
                       help="Number of samples per question for pass@k (requires temperature > 0).")
    passk.add_argument("--sampling-temperature", type=float, default=0.7,
                       help="Temperature used for pass@k sampling (only when --num-samples > 1).")
    return parser.parse_args()


def resolve_dtype(dtype_name: str, device_name: str) -> torch.dtype | None:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    if device_name == "mps":
        return torch.float16
    return None


def resolve_device(device_name: str) -> str:
    if device_name != "auto":
        return device_name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


def extract_boxed_answer(text: str) -> str | None:
    matches = list(BOXED_RE.finditer(text))
    if not matches:
        return None
    last_match = matches[-1]
    brace_start = text.find("{", last_match.start())
    if brace_start < 0:
        return None
    return extract_balanced_braces(text, brace_start)


def normalize_text(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("$", "")
    text = text.replace(",", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("\\!", "")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = text.strip(" $。，,.;")
    text = re.sub(r"\s+", "", text)
    return text



def safe_parse_expr(expr: str):
    if sp is None:
        return None
    expr = expr.strip()
    if not expr:
        return None
    if parse_latex is not None and ("\\" in expr or "{" in expr or "^" in expr):
        try:
            return parse_latex(expr)
        except Exception:
            pass
    try:
        return sp.sympify(expr.replace("^", "**"))
    except Exception:
        return None


def math_equiv(pred: str, gold: str) -> bool:
    pred_norm = normalize_text(pred)
    gold_norm = normalize_text(gold)
    if pred_norm == gold_norm:
        return True

    if sp is None:
        return False

    if "=" in pred and "=" in gold:
        pred_lhs, pred_rhs = pred.split("=", 1)
        gold_lhs, gold_rhs = gold.split("=", 1)
        pred_lhs_expr = safe_parse_expr(pred_lhs)
        pred_rhs_expr = safe_parse_expr(pred_rhs)
        gold_lhs_expr = safe_parse_expr(gold_lhs)
        gold_rhs_expr = safe_parse_expr(gold_rhs)
        if None not in (pred_lhs_expr, pred_rhs_expr, gold_lhs_expr, gold_rhs_expr):
            try:
                pred_expr = pred_lhs_expr - pred_rhs_expr
                gold_expr = gold_lhs_expr - gold_rhs_expr
                return sp.simplify(pred_expr - gold_expr) == 0 or sp.simplify(pred_expr + gold_expr) == 0
            except Exception:
                pass

    pred_expr = safe_parse_expr(pred)
    gold_expr = safe_parse_expr(gold)
    if pred_expr is not None and gold_expr is not None:
        try:
            return sp.simplify(pred_expr - gold_expr) == 0
        except Exception:
            return False
    return False


_SENTENCE_BOUNDARY_RE = re.compile(r"[.?!]\s+[A-Z]")


def clean_answer_candidate(text: str) -> str:
    candidate = (text or "").strip()
    boxed = extract_boxed_answer(candidate)
    if boxed is not None:
        return boxed.strip()
    boundary = _SENTENCE_BOUNDARY_RE.search(candidate)
    if boundary:
        candidate = candidate[: boundary.start()]
    return candidate.strip(" \t\n\r$.,;:，。")


def _find_last_match(pattern: str, text: str, flags: int = 0) -> re.Match | None:
    matches = list(re.finditer(pattern, text, flags))
    return matches[-1] if matches else None


_TRAILING_TEXT_THRESHOLD = 80


def _match_is_premature(match: re.Match, text: str) -> bool:
    """Return True if the match is followed by enough text with numbers,
    indicating the model continued reasoning after claiming an answer."""
    trailing = text[match.end():]
    return len(trailing) >= _TRAILING_TEXT_THRESHOLD and bool(NUMBER_RE.search(trailing))


def extract_explicit_pred_answer(text: str) -> tuple[str | None, str | None]:
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed.strip(), "boxed"

    answer_match = _find_last_match(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if answer_match:
        return clean_answer_candidate(answer_match.group(1)), "answer_tag"

    explicit_patterns = [
        ("final_answer", r"Final\s*Answer\s*[:：]\s*(.+?)(?:\n|$)"),
        ("answer_line", r"(?m)^Answer\s*[:：]\s*(.+?)\s*$"),
        ("the_answer_is", r"The\s+answer\s+is\s+(.+?)(?:\n|$)"),
        ("so_answer_is", r"So\s+the\s+answer\s+is\s+(.+?)(?:\n|$)"),
        ("therefore_answer_is", r"Therefore,?\s+the\s+answer\s+is\s+(.+?)(?:\n|$)"),
        ("thus_answer_is", r"Thus,?\s+the\s+answer\s+is\s+(.+?)(?:\n|$)"),
        ("hence_answer_is", r"Hence,?\s+the\s+answer\s+is\s+(.+?)(?:\n|$)"),
    ]
    for source, pattern in explicit_patterns:
        match = _find_last_match(pattern, text, re.IGNORECASE | re.DOTALL)
        if match and not _match_is_premature(match, text):
            return clean_answer_candidate(match.group(1)), source

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        last_line = clean_answer_candidate(lines[-1])
        if extract_boxed_answer(last_line) is not None or re.fullmatch(r"-?\d[\d,]*\.?\d*", last_line):
            return clean_answer_candidate(last_line), "last_line"

    return None, None


def extract_loose_pred_answer(text: str) -> tuple[str, str]:
    explicit_answer, explicit_source = extract_explicit_pred_answer(text)
    if explicit_answer is not None:
        return explicit_answer, explicit_source or "explicit"

    matches = NUMBER_RE.findall(text)
    if matches:
        return matches[-1].strip(), "last_number"
    return text.strip(), "full_text"


def detect_truncation(output: str) -> bool:
    """Heuristic: output appears to have been cut off mid-generation."""
    stripped = output.rstrip()
    if not stripped:
        return False
    if any(stripped.endswith(end) for end in (
        "</answer>", "</think>", ".", "!", "?", "。", "！", "？",
        "\"", "'", ")", "}", ">", "]",
    )):
        return False
    return True


# ---------------------------------------------------------------------------
# Reasoning quality analysis
# ---------------------------------------------------------------------------

_STEP_MARKERS = re.compile(
    r"(?:^|\n)\s*(?:"
    r"(?:Step|步骤)\s*\d+"
    r"|(?:\d+)\.\s"
    r"|(?:First|Second|Third|Next|Then|Finally|Now|So|Therefore|Thus|Hence)[\s,]"
    r")",
    re.IGNORECASE,
)


def count_reasoning_steps(text: str) -> int:
    """Count approximate number of reasoning steps in the output."""
    return max(len(_STEP_MARKERS.findall(text)), 1)


_NGRAM_WINDOW = 4


def detect_repetition(text: str) -> dict[str, Any]:
    """Detect output degeneration via n-gram repetition analysis.

    Returns:
        has_repetition: True if severe repetition detected
        repetition_ratio: fraction of 4-gram tokens that are repeated
        longest_repeated_span: length of longest verbatim repeated substring
    """
    words = text.split()
    if len(words) < _NGRAM_WINDOW * 2:
        return {"has_repetition": False, "repetition_ratio": 0.0, "longest_repeated_span": 0}

    ngrams = [
        " ".join(words[i : i + _NGRAM_WINDOW])
        for i in range(len(words) - _NGRAM_WINDOW + 1)
    ]
    counts = Counter(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    ratio = repeated / len(ngrams) if ngrams else 0.0

    longest_span = 0
    seen_spans: set[str] = set()
    for length in range(min(30, len(words) // 2), 2, -1):
        for i in range(len(words) - length + 1):
            span = " ".join(words[i : i + length])
            if span in seen_spans:
                longest_span = max(longest_span, length)
                break
            seen_spans.add(span)
        if longest_span > 0:
            break

    has_rep = ratio > 0.3 or longest_span > 15
    return {
        "has_repetition": has_rep,
        "repetition_ratio": round(ratio, 4),
        "longest_repeated_span": longest_span,
    }


def analyze_reasoning_quality(output: str) -> dict[str, Any]:
    """Comprehensive reasoning quality analysis for a single output."""
    rep = detect_repetition(output)
    word_count = len(output.split())
    return {
        "word_count": word_count,
        "reasoning_steps": count_reasoning_steps(output),
        "is_truncated": detect_truncation(output),
        "has_repetition": rep["has_repetition"],
        "repetition_ratio": rep["repetition_ratio"],
        "longest_repeated_span": rep["longest_repeated_span"],
        "is_degenerate": rep["has_repetition"] or word_count < 5,
    }


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

def classify_error(
    output: str,
    pred_answer: str | None,
    gold_answer: str,
    correct: bool,
    rq: dict[str, Any],
) -> str:
    """Classify a wrong answer into an error category.

    Categories:
        correct           – answer is correct (not an error)
        truncation        – output was cut off before reaching a final answer
        degeneration      – repetitive / degenerate output
        format_only_error – model got the right number but format extraction failed
        computation_error – final answer is a number but wrong value
        reasoning_error   – structured reasoning present but wrong conclusion
        comprehension_error – misunderstood the problem (very short / no reasoning)
    """
    if correct:
        return "correct"

    if rq.get("is_degenerate"):
        return "degeneration"
    if rq.get("is_truncated"):
        return "truncation"

    if pred_answer is None or pred_answer.strip() == "":
        return "truncation"

    gold_norm = normalize_text(gold_answer)
    all_numbers = NUMBER_RE.findall(output)
    if gold_norm and any(normalize_text(n) == gold_norm for n in all_numbers):
        return "format_only_error"

    pred_norm = normalize_text(pred_answer)
    pred_is_number = bool(re.fullmatch(r"-?\d[\d,]*\.?\d*", pred_norm))
    gold_is_number = bool(re.fullmatch(r"-?\d[\d,]*\.?\d*", gold_norm))
    if pred_is_number and gold_is_number:
        return "computation_error"

    steps = rq.get("reasoning_steps", 0)
    word_count = rq.get("word_count", 0)
    if steps >= 2 and word_count >= 30:
        return "reasoning_error"

    return "comprehension_error"


# ---------------------------------------------------------------------------
# Pass@k computation
# ---------------------------------------------------------------------------

def _estimator_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k (from Chen et al., Codex paper)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def compute_pass_at_k(
    per_question_results: list[list[bool]],
    k_values: list[int],
) -> dict[str, float]:
    """Compute pass@k for multiple k values.

    Args:
        per_question_results: list of lists, each inner list contains
            True/False for each sample of a question.
        k_values: list of k values to compute.

    Returns:
        dict mapping "pass@{k}" to the average pass@k score.
    """
    results = {}
    for k in k_values:
        scores = []
        for sample_results in per_question_results:
            n = len(sample_results)
            c = sum(sample_results)
            if n < k:
                scores.append(float(c > 0))
            else:
                scores.append(_estimator_pass_at_k(n, c, k))
        results[f"pass@{k}"] = sum(scores) / len(scores) if scores else 0.0
    return results


def extract_gold_answer(item: dict[str, Any], args: argparse.Namespace) -> str:
    if args.dataset == "gsm8k":
        answer = str(item["answer"])
    else:
        answer = str(item.get(args.answer_field, ""))

    # Local parquet exports of GSM8K keep the original rationale plus
    # a trailing "#### final_answer" marker, so strip it even outside the
    # direct datasets.load_dataset("gsm8k") path.
    if "####" in answer:
        answer = answer.split("####", 1)[1]
    return answer.strip()



def get_question(item: dict[str, Any], args: argparse.Namespace) -> str:
    if args.dataset == "gsm8k":
        return str(item["question"])
    return str(item.get(args.question_field, "")).strip()


def get_topic(item: dict[str, Any], args: argparse.Namespace) -> str | None:
    if args.dataset == "gsm8k":
        return None
    value = item.get(args.topic_field)
    return None if value is None else str(value)


def get_difficulty(item: dict[str, Any], args: argparse.Namespace) -> float | None:
    if args.dataset == "gsm8k":
        return None
    value = item.get(args.difficulty_field)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def difficulty_bucket(value: float | None) -> str | None:
    if value is None:
        return None
    if value < 3:
        return "[1,3)"
    if value < 5:
        return "[3,5)"
    if value < 7:
        return "[5,7)"
    return "[7,10]"


def check_xml_metrics(text: str) -> dict[str, bool | int]:
    return {
        "has_think": bool(THINK_RE.search(text)),
        "has_answer": bool(ANSWER_RE.search(text)),
        "boxed_count": len(BOXED_RE.findall(text)),
        "has_single_boxed": len(BOXED_RE.findall(text)) == 1 and extract_boxed_answer(text) is not None,
        "exact_xml_boxed": bool(
            re.match(
                r"^\s*<think>\s*.*?\s*</think>\s*<answer>\s*\\boxed\s*\{.*?\}\s*</answer>\s*$",
                text,
                re.DOTALL | re.IGNORECASE,
            )
        ),
    }


def build_prompt(tokenizer, question: str, prompt_style: str) -> str:
    if prompt_style == "plain":
        messages = [{"role": "user", "content": question}]
    elif prompt_style == "math_plain":
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_MATH_PLAIN},
            {"role": "user", "content": question},
        ]
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_XML},
            {"role": "user", "content": question},
        ]

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    joined = "\n\n".join(msg["content"] for msg in messages)
    return f"{joined}\n"


def load_examples(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.dataset == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split=args.split)
    elif args.dataset == "jsonl":
        if args.dataset_path is None:
            raise ValueError("--dataset-path is required when --dataset=jsonl")
        dataset = load_dataset("json", data_files=str(args.dataset_path), split="train")
    else:
        if args.dataset_path is None:
            raise ValueError("--dataset-path is required when --dataset=parquet")
        dataset = load_dataset("parquet", data_files=str(args.dataset_path), split="train")

    start = min(args.offset, len(dataset))
    end = min(args.offset + args.limit, len(dataset))
    return [dataset[i] for i in range(start, end)]


def load_model_and_tokenizer(args: argparse.Namespace):
    runtime_device = resolve_device(args.device)
    torch_dtype = resolve_dtype(args.dtype, runtime_device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if runtime_device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
    if args.adapter:
        if PeftModel is None:
            raise ImportError("peft is required when using --adapter")
        model = PeftModel.from_pretrained(model, args.adapter)
    if runtime_device in {"cpu", "mps"}:
        model = model.to(runtime_device)
    model.eval()
    return model, tokenizer, runtime_device


def aggregate_group_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)

    payload: dict[str, dict[str, float]] = {}
    for group, items in sorted(grouped.items()):
        total = len(items)
        correct = sum(int(item["correct"]) for item in items)
        exact_xml = sum(int(item["exact_xml_boxed"]) for item in items)
        payload[group] = {
            "count": total,
            "accuracy": correct / total if total else 0.0,
            "exact_xml_rate": exact_xml / total if total else 0.0,
        }
    return payload


def _build_summary(
    args: argparse.Namespace,
    runtime_device: str,
    total: int,
    num_correct: int,
    strict_num_correct: int,
    explicit_final_answer_count: int,
    truncated_count: int,
    extraction_source_counts: dict[str, int],
    has_think: int,
    has_answer: int,
    has_single_boxed: int,
    exact_xml_boxed: int,
    reasoning_quality_stats: dict[str, Any] | None = None,
    pass_at_k_results: dict[str, float] | None = None,
    error_type_counts: dict[str, int] | None = None,
    content_correct_count: int = 0,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "model": args.model,
        "adapter": args.adapter,
        "dataset": args.dataset,
        "dataset_path": str(args.dataset_path) if args.dataset_path else None,
        "split": args.split,
        "limit": total,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "prompt_style": args.prompt_style,
        "device": runtime_device,
        "accuracy": num_correct / total if total else 0.0,
        "num_correct": num_correct,
        "strict_accuracy": strict_num_correct / total if total else 0.0,
        "strict_num_correct": strict_num_correct,
        "content_accuracy": content_correct_count / total if total else 0.0,
        "content_correct_count": content_correct_count,
        "format_conditioned_accuracy": (
            strict_num_correct / explicit_final_answer_count
            if explicit_final_answer_count > 0 else None
        ),
        "explicit_final_answer_rate": explicit_final_answer_count / total if total else 0.0,
        "truncated_rate": truncated_count / total if total else 0.0,
        "has_think_rate": has_think / total if total else 0.0,
        "has_answer_rate": has_answer / total if total else 0.0,
        "has_single_boxed_rate": has_single_boxed / total if total else 0.0,
        "exact_xml_boxed_rate": exact_xml_boxed / total if total else 0.0,
        "extraction_source_distribution": dict(extraction_source_counts),
    }

    if pass_at_k_results:
        summary["pass_at_k"] = pass_at_k_results

    if reasoning_quality_stats:
        summary["reasoning_quality"] = reasoning_quality_stats

    if error_type_counts:
        error_total = sum(error_type_counts.values())
        summary["error_classification"] = {
            "counts": dict(error_type_counts),
            "rates": {
                k: round(v / total, 4) if total else 0.0
                for k, v in error_type_counts.items()
            },
        }

    return summary


def _save_payload(
    args: argparse.Namespace,
    summary: dict[str, Any],
    detailed_results: list[dict[str, Any]],
    difficulty_rows: list[dict[str, Any]],
    topic_rows: list[dict[str, Any]],
) -> None:
    payload = {
        "summary": summary,
        "by_difficulty_bucket": aggregate_group_metrics(difficulty_rows) if difficulty_rows else {},
        "by_topic": aggregate_group_metrics(topic_rows) if topic_rows else {},
        "results": detailed_results,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    runtime_device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    """Generate outputs for a batch of prompts."""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    if runtime_device in {"cpu", "mps", "cuda"}:
        inputs = {k: v.to(runtime_device) for k, v in inputs.items()}

    with torch.no_grad():
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
        generated = model.generate(**inputs, **generation_kwargs)

    input_len = inputs["input_ids"].shape[1]
    new_tokens = generated[:, input_len:]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def _aggregate_reasoning_quality(detailed_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate reasoning quality metrics across all results."""
    rq_list = [r.get("reasoning_quality", {}) for r in detailed_results if r.get("reasoning_quality")]
    if not rq_list:
        return {}
    total = len(rq_list)
    return {
        "avg_word_count": round(sum(r.get("word_count", 0) for r in rq_list) / total, 1),
        "avg_reasoning_steps": round(sum(r.get("reasoning_steps", 0) for r in rq_list) / total, 2),
        "truncation_rate": round(sum(int(r.get("is_truncated", False)) for r in rq_list) / total, 4),
        "repetition_rate": round(sum(int(r.get("has_repetition", False)) for r in rq_list) / total, 4),
        "avg_repetition_ratio": round(sum(r.get("repetition_ratio", 0.0) for r in rq_list) / total, 4),
        "degeneration_rate": round(sum(int(r.get("is_degenerate", False)) for r in rq_list) / total, 4),
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    do_pass_k = args.num_samples > 1 and args.pass_k is not None
    if do_pass_k:
        max_k = max(args.pass_k)
        if args.num_samples < max_k:
            print(f"Warning: --num-samples ({args.num_samples}) < max(--pass-k) ({max_k}). "
                  f"Setting --num-samples to {max_k}.", flush=True)
            args.num_samples = max_k

    examples = load_examples(args)

    num_correct = 0
    strict_num_correct = 0
    content_correct_count = 0
    explicit_final_answer_count = 0
    truncated_count = 0
    extraction_source_counts: dict[str, int] = defaultdict(int)
    error_type_counts: dict[str, int] = defaultdict(int)
    has_think = 0
    has_answer = 0
    has_single_boxed = 0
    exact_xml_boxed = 0
    detailed_results: list[dict[str, Any]] = []
    difficulty_rows: list[dict[str, Any]] = []
    topic_rows: list[dict[str, Any]] = []
    pass_k_per_question: list[list[bool]] = []
    processed = 0
    skip_count = 0

    if args.resume and args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        existing_results = existing.get("results", [])
        skip_count = len(existing_results)
        if skip_count > 0:
            print(f"Resuming: found {skip_count} existing results in {args.output}, skipping them.", flush=True)
            for r in existing_results:
                detailed_results.append(r)
                num_correct += int(r.get("correct", False))
                strict_num_correct += int(r.get("strict_correct", False))
                content_correct_count += int(r.get("content_correct", False))
                explicit_final_answer_count += int(r.get("has_explicit_final_answer", False))
                truncated_count += int(r.get("is_truncated", False))
                extraction_source_counts[r.get("extraction_source", "unknown")] += 1
                error_type_counts[r.get("error_type", "unknown")] += 1
                xm = r.get("xml_metrics", {})
                has_think += int(xm.get("has_think", False))
                has_answer += int(xm.get("has_answer", False))
                has_single_boxed += int(xm.get("has_single_boxed", False))
                exact_xml_boxed += int(xm.get("exact_xml_boxed", False))
                d = r.get("difficulty")
                if d is not None:
                    difficulty_rows.append({"group": difficulty_bucket(d), "correct": r.get("correct", False), "exact_xml_boxed": xm.get("exact_xml_boxed", False)})
                t = r.get("topic")
                if t:
                    topic_rows.append({"group": t, "correct": r.get("correct", False), "exact_xml_boxed": xm.get("exact_xml_boxed", False)})
                if do_pass_k and "pass_k_samples" in r:
                    pass_k_per_question.append(r["pass_k_samples"])
                processed += 1

    remaining_examples = examples[skip_count:]
    if not remaining_examples:
        print(f"All {len(examples)} examples already evaluated. Nothing to do.", flush=True)
        return

    model, tokenizer, runtime_device = load_model_and_tokenizer(args)

    for start in range(0, len(remaining_examples), args.batch_size):
        batch = remaining_examples[start : start + args.batch_size]
        questions = [get_question(item, args) for item in batch]
        gold_answers = [extract_gold_answer(item, args) for item in batch]
        prompts = [build_prompt(tokenizer, question, args.prompt_style) for question in questions]

        decoded = _generate_batch(
            model, tokenizer, prompts, runtime_device,
            args.max_new_tokens, args.temperature, args.top_p,
        )

        # --- Pass@k: generate additional samples with temperature ---
        pass_k_sample_outputs: list[list[str]] | None = None
        if do_pass_k:
            pass_k_sample_outputs = [[] for _ in batch]
            for _ in range(args.num_samples):
                sample_decoded = _generate_batch(
                    model, tokenizer, prompts, runtime_device,
                    args.max_new_tokens, args.sampling_temperature, args.top_p,
                )
                for idx, sample_out in enumerate(sample_decoded):
                    pass_k_sample_outputs[idx].append(sample_out)

        for batch_idx, (item, question, gold_answer, output) in enumerate(
            zip(batch, questions, gold_answers, decoded)
        ):
            strict_pred_answer, strict_source = extract_explicit_pred_answer(output)
            loose_pred_answer, loose_source = extract_loose_pred_answer(output)
            strict_correct = strict_pred_answer is not None and math_equiv(strict_pred_answer, gold_answer)
            correct = math_equiv(loose_pred_answer, gold_answer)
            xml_metrics = check_xml_metrics(output)
            is_truncated = detect_truncation(output)
            rq = analyze_reasoning_quality(output)
            topic = get_topic(item, args)
            difficulty = get_difficulty(item, args)

            error_type = classify_error(output, loose_pred_answer, gold_answer, correct, rq)

            gold_norm = normalize_text(gold_answer)
            all_numbers = NUMBER_RE.findall(output)
            content_correct = correct or (
                bool(gold_norm) and any(normalize_text(n) == gold_norm for n in all_numbers)
            )

            num_correct += int(correct)
            strict_num_correct += int(strict_correct)
            content_correct_count += int(content_correct)
            explicit_final_answer_count += int(strict_pred_answer is not None)
            truncated_count += int(is_truncated)
            extraction_source_counts[loose_source] += 1
            error_type_counts[error_type] += 1
            has_think += int(xml_metrics["has_think"])
            has_answer += int(xml_metrics["has_answer"])
            has_single_boxed += int(xml_metrics["has_single_boxed"])
            exact_xml_boxed += int(xml_metrics["exact_xml_boxed"])

            result: dict[str, Any] = {
                "question": question,
                "gold_answer": gold_answer,
                "pred_answer": loose_pred_answer,
                "correct": correct,
                "content_correct": content_correct,
                "error_type": error_type,
                "extraction_source": loose_source,
                "strict_pred_answer": strict_pred_answer,
                "strict_correct": strict_correct,
                "strict_extraction_source": strict_source,
                "has_explicit_final_answer": strict_pred_answer is not None,
                "is_truncated": is_truncated,
                "topic": topic,
                "difficulty": difficulty,
                "xml_metrics": xml_metrics,
                "reasoning_quality": rq,
            }

            if do_pass_k and pass_k_sample_outputs is not None:
                sample_correctness = []
                for sample_out in pass_k_sample_outputs[batch_idx]:
                    s_pred, _ = extract_loose_pred_answer(sample_out)
                    sample_correctness.append(math_equiv(s_pred, gold_answer))
                result["pass_k_samples"] = sample_correctness
                pass_k_per_question.append(sample_correctness)

            if args.save_outputs:
                result["output"] = output.strip()
            detailed_results.append(result)

            if difficulty is not None:
                difficulty_rows.append(
                    {
                        "group": difficulty_bucket(difficulty),
                        "correct": correct,
                        "exact_xml_boxed": xml_metrics["exact_xml_boxed"],
                    }
                )
            if topic:
                topic_rows.append(
                    {
                        "group": topic,
                        "correct": correct,
                        "exact_xml_boxed": xml_metrics["exact_xml_boxed"],
                    }
                )
            processed += 1
            if args.progress_every > 0 and processed % args.progress_every == 0:
                print(
                    f"Processed {processed}/{len(examples)} examples; "
                    f"running accuracy={num_correct / processed:.4f}",
                    flush=True,
                )

        if args.checkpoint_every > 0 and processed > 0 and processed % args.checkpoint_every == 0:
            ckpt_summary = _build_summary(
                args, runtime_device, processed,
                num_correct, strict_num_correct,
                explicit_final_answer_count, truncated_count, extraction_source_counts,
                has_think, has_answer, has_single_boxed, exact_xml_boxed,
                error_type_counts=dict(error_type_counts),
                content_correct_count=content_correct_count,
            )
            _save_payload(args, ckpt_summary, detailed_results, difficulty_rows, topic_rows)
            print(f"Checkpoint saved ({processed}/{len(examples)} examples) -> {args.output}", flush=True)

    total = len(examples)

    pass_at_k_results = None
    if do_pass_k and pass_k_per_question:
        pass_at_k_results = compute_pass_at_k(pass_k_per_question, args.pass_k)
        print(f"\nPass@k results (n={args.num_samples} samples):")
        for key, val in pass_at_k_results.items():
            print(f"  {key}: {val:.4f}")

    reasoning_quality_stats = _aggregate_reasoning_quality(detailed_results)

    summary = _build_summary(
        args, runtime_device, total,
        num_correct, strict_num_correct,
        explicit_final_answer_count, truncated_count, extraction_source_counts,
        has_think, has_answer, has_single_boxed, exact_xml_boxed,
        reasoning_quality_stats=reasoning_quality_stats,
        pass_at_k_results=pass_at_k_results,
        error_type_counts=dict(error_type_counts),
        content_correct_count=content_correct_count,
    )
    _save_payload(args, summary, detailed_results, difficulty_rows, topic_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
