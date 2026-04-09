#!/usr/bin/env python3
"""Reward helpers for TRL GRPO on DeepMath using XML + boxed answers."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

import sympy as sp

try:
    from sympy.parsing.latex import parse_latex
except Exception:  # pragma: no cover
    parse_latex = None


THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\s*\{")


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return text


def _extract_balanced_braces(text: str, start_idx: int) -> str | None:
    depth = 0
    buf: list[str] = []
    for i in range(start_idx, len(text)):
        ch = text[i]
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
        else:
            if depth >= 1:
                buf.append(ch)
    return None


def extract_boxed_answer(text: str) -> str | None:
    text = _strip_code_fence(text)
    match = BOXED_RE.search(text)
    if not match:
        return None
    brace_start = text.find("{", match.start())
    if brace_start < 0:
        return None
    return _extract_balanced_braces(text, brace_start)


def extract_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
        return str(first)
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def merge_prefix(prefix: str | None, completion_text: str) -> str:
    return f"{prefix or ''}{completion_text}"


def extract_answer_text(solution_str: str) -> str:
    solution_str = _strip_code_fence(solution_str)
    answer_match = ANSWER_RE.search(solution_str)
    if answer_match:
        answer_body = answer_match.group(1).strip()
        boxed = extract_boxed_answer(answer_body)
        return boxed if boxed is not None else answer_body
    boxed = extract_boxed_answer(solution_str)
    if boxed is not None:
        return boxed
    return solution_str.strip()


def normalize_text(text: str) -> str:
    text = _strip_code_fence(text).strip()
    text = text.replace("\n", " ")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("\\!", "")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = text.strip(" $。，,.;")
    text = re.sub(r"\s+", "", text)
    return text


def _safe_parse_expr(expr: str):
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

    if "=" in pred and "=" in gold:
        pred_lhs, pred_rhs = pred.split("=", 1)
        gold_lhs, gold_rhs = gold.split("=", 1)
        pred_lhs_expr = _safe_parse_expr(pred_lhs)
        pred_rhs_expr = _safe_parse_expr(pred_rhs)
        gold_lhs_expr = _safe_parse_expr(gold_lhs)
        gold_rhs_expr = _safe_parse_expr(gold_rhs)
        if None not in (pred_lhs_expr, pred_rhs_expr, gold_lhs_expr, gold_rhs_expr):
            try:
                pred_expr = pred_lhs_expr - pred_rhs_expr
                gold_expr = gold_lhs_expr - gold_rhs_expr
                return sp.simplify(pred_expr - gold_expr) == 0 or sp.simplify(pred_expr + gold_expr) == 0
            except Exception:
                pass

    pred_expr = _safe_parse_expr(pred)
    gold_expr = _safe_parse_expr(gold)
    if pred_expr is not None and gold_expr is not None:
        try:
            return sp.simplify(pred_expr - gold_expr) == 0
        except Exception:
            return False
    return False


def _parse_xml(solution_str: str) -> dict[str, Any]:
    text = _strip_code_fence(solution_str)
    think_match = THINK_RE.search(text)
    answer_match = ANSWER_RE.search(text)
    boxed = extract_boxed_answer(answer_match.group(1)) if answer_match else None
    exact_pattern = re.compile(
        r"^\s*<think>\s*.*?\s*</think>\s*<answer>\s*\\boxed\s*\{.*?\}\s*</answer>\s*$",
        re.DOTALL | re.IGNORECASE,
    )
    return {
        "text": text,
        "think_match": think_match,
        "answer_match": answer_match,
        "boxed": boxed,
        "boxed_count": len(BOXED_RE.findall(text)),
        "exact_xml": bool(exact_pattern.match(text)),
    }


def score_format(solution_str: str) -> tuple[float, dict[str, Any]]:
    parsed = _parse_xml(solution_str)
    has_think = parsed["think_match"] is not None
    has_answer = parsed["answer_match"] is not None
    has_single_boxed = parsed["boxed_count"] == 1 and parsed["boxed"] is not None
    format_ok = parsed["exact_xml"] and has_think and has_answer and has_single_boxed
    reward = 0.2 if format_ok else 0.0
    return reward, {
        "format_ok": float(format_ok),
        "has_think": float(has_think),
        "has_answer": float(has_answer),
        "has_single_boxed": float(has_single_boxed),
    }


def score_answer(solution_str: str, ground_truth: str) -> tuple[float, dict[str, Any]]:
    pred_answer = extract_answer_text(solution_str)
    is_correct = math_equiv(pred_answer, ground_truth)
    reward = 0.8 if is_correct else 0.0
    return reward, {
        "answer_ok": float(is_correct),
        "pred_answer": pred_answer,
        "ground_truth": ground_truth,
    }


def score_format_progress(solution_str: str) -> tuple[float, dict[str, Any]]:
    parsed = _parse_xml(solution_str)
    has_think_open = "<think>" in parsed["text"]
    has_think = parsed["think_match"] is not None
    has_answer_open = "<answer>" in parsed["text"]
    has_answer = parsed["answer_match"] is not None
    has_boxed = parsed["boxed"] is not None
    has_single_boxed = parsed["boxed_count"] == 1

    reward = 0.0
    if has_think_open:
        reward += 0.03
    if has_think:
        reward += 0.05
    if has_answer_open:
        reward += 0.04
    if has_answer:
        reward += 0.05
    if has_boxed:
        reward += 0.06
    if has_single_boxed:
        reward += 0.05

    return reward, {
        "format_progress": reward,
        "progress_has_think_open": float(has_think_open),
        "progress_has_think": float(has_think),
        "progress_has_answer_open": float(has_answer_open),
        "progress_has_answer": float(has_answer),
        "progress_has_boxed": float(has_boxed),
        "progress_has_single_boxed": float(has_single_boxed),
    }


def score_xml_tag_shape(solution_str: str) -> tuple[float, dict[str, Any]]:
    parsed = _parse_xml(solution_str)
    reward = 0.0
    if "<think>" in parsed["text"] and "</think>" in parsed["text"]:
        reward += 0.05
    if "<answer>" in parsed["text"] and "</answer>" in parsed["text"]:
        reward += 0.05
    if parsed["boxed"] is not None:
        reward += 0.05
    return reward, {"xml_tag_shape_reward": reward}


def score_brevity(solution_str: str) -> tuple[float, dict[str, Any]]:
    parsed = _parse_xml(solution_str)
    think_text = parsed["think_match"].group(1).strip() if parsed["think_match"] else ""
    char_len = len(parsed["text"])

    reward = 0.0
    if parsed["exact_xml"] and char_len <= 900:
        reward = 0.03
    if parsed["exact_xml"] and char_len <= 600:
        reward = 0.06
    if parsed["exact_xml"] and len(think_text) <= 280:
        reward = max(reward, 0.1)

    return reward, {
        "brevity_reward": reward,
        "char_len": char_len,
        "think_len": len(think_text),
    }


def _to_float(expr: str) -> float | None:
    parsed = _safe_parse_expr(expr)
    if parsed is None:
        return None
    try:
        value = complex(parsed.evalf())
    except Exception:
        return None
    if abs(value.imag) > 1e-8:
        return None
    return float(value.real)


def score_answer_similarity(solution_str: str, ground_truth: str) -> tuple[float, dict[str, Any]]:
    pred_answer = extract_answer_text(solution_str)
    pred_norm = normalize_text(pred_answer)
    gold_norm = normalize_text(ground_truth)

    if not pred_norm or not gold_norm:
        return 0.0, {"answer_similarity_reward": 0.0}

    if math_equiv(pred_answer, ground_truth):
        return 0.45, {"answer_similarity_reward": 0.45}

    reward = 0.0
    text_ratio = SequenceMatcher(None, pred_norm, gold_norm).ratio()
    reward = max(reward, 0.18 * text_ratio)

    pred_float = _to_float(pred_answer)
    gold_float = _to_float(ground_truth)
    if pred_float is not None and gold_float is not None:
        scale = max(1.0, abs(gold_float))
        rel_err = abs(pred_float - gold_float) / scale
        reward = max(reward, 0.2 * (1.0 - min(rel_err, 1.0)))

    if pred_norm in gold_norm or gold_norm in pred_norm:
        reward = max(reward, 0.12)

    return reward, {"answer_similarity_reward": reward}


def score_repetition_penalty(solution_str: str) -> tuple[float, dict[str, Any]]:
    text = _strip_code_fence(solution_str).lower()
    markers = [
        "okay, so",
        "let me",
        "hmm,",
        "i need to",
        "to solve this",
        "first,",
        "next,",
    ]
    hits = sum(text.count(marker) for marker in markers)
    penalty = 0.0
    if hits >= 2:
        penalty = -0.08
    elif hits == 1:
        penalty = -0.03
    return penalty, {"repetition_penalty": penalty}


def format_progress_reward_func(completions, response_prefix=None, **kwargs) -> list[float]:
    rewards = []
    prefixes = response_prefix or [None] * len(completions)
    for completion, prefix in zip(completions, prefixes):
        reward, _ = score_format_progress(merge_prefix(prefix, extract_completion_text(completion)))
        rewards.append(reward)
    return rewards


def strict_format_reward_func(completions, response_prefix=None, **kwargs) -> list[float]:
    rewards = []
    prefixes = response_prefix or [None] * len(completions)
    for completion, prefix in zip(completions, prefixes):
        reward, _ = score_format(merge_prefix(prefix, extract_completion_text(completion)))
        rewards.append(reward)
    return rewards


def brevity_reward_func(completions, response_prefix=None, **kwargs) -> list[float]:
    rewards = []
    prefixes = response_prefix or [None] * len(completions)
    for completion, prefix in zip(completions, prefixes):
        reward, _ = score_brevity(merge_prefix(prefix, extract_completion_text(completion)))
        rewards.append(reward)
    return rewards


def xml_tag_shape_reward_func(completions, response_prefix=None, **kwargs) -> list[float]:
    rewards = []
    prefixes = response_prefix or [None] * len(completions)
    for completion, prefix in zip(completions, prefixes):
        reward, _ = score_xml_tag_shape(merge_prefix(prefix, extract_completion_text(completion)))
        rewards.append(reward)
    return rewards


def answer_similarity_reward_func(completions, ground_truth, response_prefix=None, **kwargs) -> list[float]:
    rewards = []
    prefixes = response_prefix or [None] * len(completions)
    for completion, gold, prefix in zip(completions, ground_truth, prefixes):
        reward, _ = score_answer_similarity(merge_prefix(prefix, extract_completion_text(completion)), gold)
        rewards.append(reward)
    return rewards


def answer_reward_func(completions, ground_truth, response_prefix=None, **kwargs) -> list[float]:
    rewards = []
    prefixes = response_prefix or [None] * len(completions)
    for completion, gold, prefix in zip(completions, ground_truth, prefixes):
        reward, _ = score_answer(merge_prefix(prefix, extract_completion_text(completion)), gold)
        rewards.append(reward)
    return rewards


def repetition_penalty_func(completions, response_prefix=None, **kwargs) -> list[float]:
    rewards = []
    prefixes = response_prefix or [None] * len(completions)
    for completion, prefix in zip(completions, prefixes):
        reward, _ = score_repetition_penalty(merge_prefix(prefix, extract_completion_text(completion)))
        rewards.append(reward)
    return rewards
