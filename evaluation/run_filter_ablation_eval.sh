#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results/filter_ablation}"
ABLATION_DIR="${ABLATION_DIR:-$PROJECT_DIR/ablation/runs}"
BASE_MODEL="${BASE_MODEL:-$PROJECT_DIR/models/Qwen2.5-0.5B}"
GSM8K_DATASET="${GSM8K_DATASET:-$PROJECT_DIR/datasets/gsm8k/main/test-00000-of-00001.parquet}"
DEEPMATH_SMOKE_DATASET="${DEEPMATH_SMOKE_DATASET:-$PROJECT_DIR/GenAI_final_project_data/outputs/phase_a_deepmath/deepmath_phase_a_alpaca.jsonl}"
DEVICE="${DEVICE:-auto}"
GSM8K_LIMIT="${GSM8K_LIMIT:-100}"
DEEPMATH_LIMIT="${DEEPMATH_LIMIT:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
RUNS="${RUNS:-00_full_all_rules 01_wo_difficulty_band 02_wo_question_min_len 03_wo_question_max_len 04_wo_answer_min_len 05_wo_answer_max_len 06_wo_ambiguity_regex 07_wo_r1_requirement}"

mkdir -p "$RESULTS_DIR/gsm8k" "$RESULTS_DIR/deepmath_smoke"

run_eval() {
  local dataset_name="$1"; shift
  local run_name="$1"; shift
  echo ">>> [$dataset_name] $run_name"
  "$PYTHON_BIN" "$SCRIPT_DIR/evaluate_math_model.py" "$@"
}

for run_name in $RUNS; do
  adapter_path="$ABLATION_DIR/$run_name"
  if [[ ! -f "$adapter_path/adapter_config.json" ]]; then
    echo "Missing adapter in $adapter_path" >&2
    exit 1
  fi

  run_eval "deepmath_smoke" "$run_name" \
    --model "$BASE_MODEL" \
    --adapter "$adapter_path" \
    --dataset jsonl \
    --dataset-path "$DEEPMATH_SMOKE_DATASET" \
    --question-field instruction \
    --answer-field final_answer_norm \
    --topic-field topic \
    --difficulty-field difficulty \
    --limit "$DEEPMATH_LIMIT" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --prompt-style xml \
    --device "$DEVICE" \
    --output "$RESULTS_DIR/deepmath_smoke/${run_name}.json" \
    --resume

  run_eval "gsm8k" "$run_name" \
    --model "$BASE_MODEL" \
    --adapter "$adapter_path" \
    --dataset parquet \
    --dataset-path "$GSM8K_DATASET" \
    --question-field question \
    --answer-field answer \
    --limit "$GSM8K_LIMIT" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --prompt-style xml \
    --device "$DEVICE" \
    --output "$RESULTS_DIR/gsm8k/${run_name}.json" \
    --resume
done

"$PYTHON_BIN" "$SCRIPT_DIR/compare_results.py" \
  --inputs "$RESULTS_DIR/deepmath_smoke/"*.json \
  --labels $RUNS \
  --output-json "$RESULTS_DIR/deepmath_smoke_comparison.json" \
  --output-md "$RESULTS_DIR/deepmath_smoke_comparison.md" \
  --show-by-difficulty \
  --show-by-topic

"$PYTHON_BIN" "$SCRIPT_DIR/compare_results.py" \
  --inputs "$RESULTS_DIR/gsm8k/"*.json \
  --labels $RUNS \
  --output-json "$RESULTS_DIR/gsm8k_comparison.json" \
  --output-md "$RESULTS_DIR/gsm8k_comparison.md" \
  --show-by-difficulty \
  --show-by-topic

"$PYTHON_BIN" "$SCRIPT_DIR/summarize_filter_ablation.py" \
  --results-dir "$RESULTS_DIR" \
  --output-json "$RESULTS_DIR/filter_ablation_summary.json" \
  --output-md "$RESULTS_DIR/filter_ablation_summary.md"

echo "Saved filter ablation results to: $RESULTS_DIR"
