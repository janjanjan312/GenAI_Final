# Evaluation Scripts

Comprehensive evaluation pipeline for **Topic 1: Data Selection + RL for LLMs (Math / STEM)**.

## Overview

The evaluation is organized into **three layers**, following the AIMS5740 project requirements:

| Layer | Purpose | Tools |
|-------|---------|-------|
| **Layer 1: Math Capability** | Does SFT/RL improve math correctness? | `evaluate_math_model.py`, `run_lm_eval.py` |
| **Layer 2: Format Adherence** | Does the model follow the intended output format? | `evaluate_math_model.py` |
| **Layer 3: Reasoning Quality** | Repetition, degeneration, truncation, step quality | `evaluate_math_model.py` |

## Quick Start

### One-click full evaluation

```bash
# Full evaluation (base + SFT + GRPO on GSM8K + MATH-500)
bash evaluation/run_full_eval.sh

# Quick smoke test (20 examples)
QUICK=1 bash evaluation/run_full_eval.sh

# Skip lm-eval-harness if not installed
SKIP_LM_EVAL=1 bash evaluation/run_full_eval.sh
```

### Install dependencies

```bash
pip install lm-eval          # for standardized benchmarks
pip install sympy             # for mathematical equivalence checking
pip install peft              # for LoRA adapter loading
```

## Evaluation Tools

### 1. `evaluate_math_model.py` — Custom Math Evaluator

Primary evaluator with fine-grained metrics. Supports greedy decoding + pass@k sampling.

**Key metrics produced:**

- `accuracy` — loose extraction (tries structured answer first, falls back to last number)
- `strict_accuracy` — only counts when model outputs explicit answer format (boxed, tags, etc.)
- `format_conditioned_accuracy` — accuracy among samples with explicit answer format
- `pass@k` — pass@1, pass@3, pass@5 via temperature sampling (for RL models)
- `reasoning_quality` — word count, step count, repetition rate, degeneration rate

**Greedy evaluation:**

```bash
# Base model
python evaluation/evaluate_math_model.py \
  --model models/Qwen2.5-0.5B \
  --dataset parquet \
  --dataset-path datasets/gsm8k/main/test-00000-of-00001.parquet \
  --question-field question \
  --answer-field answer \
  --limit 200 \
  --max-new-tokens 1024 \
  --prompt-style math_plain \
  --output evaluation/results/base_gsm8k.json \
  --save-outputs

# SFT LoRA adapter
python evaluation/evaluate_math_model.py \
  --model models/Qwen2.5-0.5B \
  --adapter sft/qwen25_0p5b_math_lora \
  --dataset parquet \
  --dataset-path datasets/gsm8k/main/test-00000-of-00001.parquet \
  --question-field question \
  --answer-field answer \
  --limit 200 \
  --max-new-tokens 1024 \
  --prompt-style math_plain \
  --output evaluation/results/sft_gsm8k.json \
  --save-outputs

# GRPO/RL model (XML prompt style)
python evaluation/evaluate_math_model.py \
  --model grpo_trl_model \
  --dataset parquet \
  --dataset-path datasets/gsm8k/main/test-00000-of-00001.parquet \
  --question-field question \
  --answer-field answer \
  --limit 200 \
  --max-new-tokens 1024 \
  --prompt-style xml \
  --output evaluation/results/grpo_gsm8k.json \
  --save-outputs
```

**Pass@k evaluation (for RL models):**

```bash
python evaluation/evaluate_math_model.py \
  --model grpo_trl_model \
  --dataset parquet \
  --dataset-path datasets/gsm8k/main/test-00000-of-00001.parquet \
  --question-field question \
  --answer-field answer \
  --limit 200 \
  --max-new-tokens 1024 \
  --prompt-style xml \
  --num-samples 10 \
  --pass-k 1 3 5 10 \
  --sampling-temperature 0.7 \
  --output evaluation/results/grpo_gsm8k_passk.json \
  --save-outputs
```

**MATH-500 evaluation:**

```bash
python evaluation/evaluate_math_model.py \
  --model grpo_trl_model \
  --dataset jsonl \
  --dataset-path datasets/math500/test.jsonl \
  --question-field problem \
  --answer-field answer \
  --limit 200 \
  --max-new-tokens 1024 \
  --prompt-style xml \
  --output evaluation/results/grpo_math500.json \
  --save-outputs
```

### 2. `run_lm_eval.py` — lm-eval-harness Wrapper

Standardized evaluation using [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness).
This is the recommended approach for comparable benchmark numbers.

**Available tasks:**

| Task | Description | Default few-shot |
|------|-------------|------------------|
| `gsm8k` | Grade-school math | 8-shot |
| `minerva_math` | Competition math (MATH) | 4-shot |
| `mmlu` | General knowledge QA | 5-shot |
| `hellaswag` | Commonsense reasoning | 10-shot |
| `arc_easy` | Science reasoning (easy) | 0-shot |
| `arc_challenge` | Science reasoning (hard) | 25-shot |
| `winogrande` | Coreference resolution | 5-shot |

**Presets:**

| Preset | Tasks |
|--------|-------|
| `quick` | gsm8k, arc_easy |
| `math_only` | gsm8k, minerva_math |
| `math_and_general` | gsm8k, minerva_math, mmlu, hellaswag |
| `full` | gsm8k, minerva_math, mmlu, hellaswag, arc_easy, winogrande |

**Usage:**

```bash
# Base model on math-only benchmarks
python evaluation/run_lm_eval.py \
  --model models/Qwen2.5-0.5B \
  --tasks math_only \
  --label base \
  --output-dir evaluation/results/lm_eval

# GRPO model on math + general QA (detect capability degradation)
python evaluation/run_lm_eval.py \
  --model grpo_trl_model \
  --tasks math_and_general \
  --label grpo \
  --output-dir evaluation/results/lm_eval

# SFT LoRA adapter
python evaluation/run_lm_eval.py \
  --model models/Qwen2.5-0.5B \
  --adapter sft/qwen25_0p5b_math_lora \
  --tasks gsm8k \
  --label sft \
  --output-dir evaluation/results/lm_eval

# Quick test with limited examples
python evaluation/run_lm_eval.py \
  --model models/Qwen2.5-0.5B \
  --tasks gsm8k \
  --limit 50 \
  --label base_quick

# Dry run (print command only)
python evaluation/run_lm_eval.py \
  --model models/Qwen2.5-0.5B \
  --tasks full \
  --dry-run
```

### 3. `compare_results.py` — Multi-model Comparison

Compare evaluation results across base/SFT/GRPO models with three-layer tables.

```bash
python evaluation/compare_results.py \
  --inputs \
    evaluation/results/base_gsm8k_full.json \
    evaluation/results/sft_gsm8k_full.json \
    evaluation/results/grpo_gsm8k_full.json \
  --labels base sft grpo \
  --output-json evaluation/results/full_comparison.json \
  --output-md evaluation/results/full_comparison.md \
  --show-by-difficulty \
  --show-by-topic
```

Output includes:

- **Mathematical Capability** table (accuracy, strict_accuracy, pass@k)
- **Format Adherence** table (XML structure, boxed answers, truncation)
- **Reasoning Quality** table (word count, steps, repetition, degeneration)
- By-difficulty and by-topic breakdowns (when available)

## Evaluation Metrics Reference

### Layer 1: Mathematical Capability

| Metric | Description |
|--------|-------------|
| `accuracy` | Answer correctness with loose extraction (tries structured answer markers first, falls back to last number) |
| `strict_accuracy` | Answer correctness only when model produces explicit answer format (`\boxed{}`, `<answer>`, `Final Answer:`) |
| `format_conditioned_accuracy` | Accuracy among the subset that produced explicit answer format |
| `pass@k` | Probability that at least 1 of k random samples is correct (unbiased estimator) |

### Layer 2: Format Adherence

| Metric | Description |
|--------|-------------|
| `explicit_final_answer_rate` | Fraction of outputs with structured answer markers |
| `has_think_rate` | Fraction with `<think>...</think>` tags |
| `has_answer_rate` | Fraction with `<answer>...</answer>` tags |
| `has_single_boxed_rate` | Fraction with exactly one `\boxed{}` |
| `exact_xml_boxed_rate` | Fraction with perfect `<think>...</think><answer>\boxed{...}</answer>` format |
| `extraction_source_distribution` | Where the answer was extracted from (boxed, answer_tag, final_answer, last_number, etc.) |

### Layer 3: Reasoning Quality

| Metric | Description |
|--------|-------------|
| `avg_word_count` | Average output length in words |
| `avg_reasoning_steps` | Average number of reasoning steps detected |
| `truncation_rate` | Fraction of outputs cut off mid-generation |
| `repetition_rate` | Fraction with severe n-gram repetition |
| `degeneration_rate` | Fraction with degenerate output (repetition or very short) |

## Topic 1 Analysis Questions

The evaluation pipeline is designed to answer these key questions from the project requirements:

1. **Does RL improve correctness but hurt general QA?**
   - Compare `accuracy` / `pass@k` on GSM8K/MATH-500 across base → SFT → GRPO
   - Use `run_lm_eval.py --tasks math_and_general` to check MMLU/HellaSwag degradation

2. **Sensitivity to data filtering?**
   - Evaluate on DeepMath smoke test vs. clean GSM8K
   - Compare `by_difficulty_bucket` and `by_topic` breakdowns

3. **Typical failure modes:**
   - **Format drift**: Check `exact_xml_boxed_rate` and `extraction_source_distribution`
   - **Reward hacking**: Compare `accuracy` vs `strict_accuracy` (models gaming the format)
   - **Degeneration**: Check `repetition_rate` and `degeneration_rate` in reasoning quality

## Notes

- Default `--max-new-tokens` is 1024 (up from 256/512) to avoid premature truncation.
- Default `--limit` is 200 for statistical significance. Use 20-50 for quick debugging.
- `--save-outputs` stores raw model outputs for qualitative analysis.
- `--resume` allows resuming interrupted evaluation runs.
- For pass@k, use `--num-samples >= max(k)` with `--sampling-temperature 0.7`.
