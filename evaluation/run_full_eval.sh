#!/usr/bin/env bash
# =============================================================================
# Topic 1 Evaluation (AIMS5740 Final Project)
#
# Two-part evaluation:
#   Part 1: Custom evaluator on GSM8K — deep analysis (failure modes, format
#           drift, error classification, pass@k)
#   Part 2: lm-eval-harness — standardized benchmarks
#           - minerva_math  (competition math, replaces custom MATH-500)
#           - mmlu           (general QA — detect RL capability degradation)
#           - arc_easy       (science reasoning)
#
# Usage:
#   bash evaluation/run_full_eval.sh
#   QUICK=1 bash evaluation/run_full_eval.sh       # smoke test
#   SKIP_LM_EVAL=1 bash evaluation/run_full_eval.sh
#   STAGES="base grpo" bash evaluation/run_full_eval.sh
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
FAIL_COUNT=0

run_eval() {
    local desc="$1"; shift
    echo ">>> $desc"
    if "$@"; then
        echo ">>> DONE: $desc"
    else
        echo ">>> FAILED: $desc (exit code $?)" >&2
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
BASE_MODEL="${BASE_MODEL:-$PROJECT_DIR/models/Qwen2.5-0.5B}"
SFT_MODEL="${SFT_MODEL:-$BASE_MODEL}"
SFT_ADAPTER="${SFT_ADAPTER:-$PROJECT_DIR/sft/qwen25_0p5b_math_lora}"
SFT_MERGED="${SFT_MERGED:-$PROJECT_DIR/sft/qwen25_0p5b_math_merged}"
GRPO_MODEL="${GRPO_MODEL:-$PROJECT_DIR/grpo_trl_model}"

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
GSM8K_PARQUET="$PROJECT_DIR/datasets/gsm8k/main/test-00000-of-00001.parquet"

# ---------------------------------------------------------------------------
# Eval parameters
# ---------------------------------------------------------------------------
DEVICE="${DEVICE:-auto}"
QUICK="${QUICK:-0}"
SKIP_LM_EVAL="${SKIP_LM_EVAL:-0}"
STAGES="${STAGES:-base sft grpo}"
LM_EVAL_LIMIT="${LM_EVAL_LIMIT:-200}"

if [ "$QUICK" = "1" ]; then
    LIMIT=30
    MAX_TOKENS=512
    PASS_K_SAMPLES=2
    LM_EVAL_LIMIT=30
    echo "=== QUICK MODE: limit=$LIMIT, max_tokens=$MAX_TOKENS ==="
else
    LIMIT=100
    MAX_TOKENS=256
    PASS_K_SAMPLES=2
fi

mkdir -p "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Download GSM8K if needed
# ---------------------------------------------------------------------------
if [ ! -f "$GSM8K_PARQUET" ]; then
    echo ">>> Downloading GSM8K dataset..."
    python "$SCRIPT_DIR/download_gsm8k.py" --output-dir "$PROJECT_DIR/datasets/gsm8k"
fi

echo ""
echo "============================================================"
echo " Topic 1 Evaluation"
echo "============================================================"
echo " Base model:        $BASE_MODEL"
echo " SFT merged:        $SFT_MERGED"
echo " GRPO model:        $GRPO_MODEL"
echo " Custom eval limit: $LIMIT examples (GSM8K)"
echo " lm-eval limit:     $LM_EVAL_LIMIT examples per task"
echo " Max tokens:        $MAX_TOKENS"
echo " Stages:            $STAGES"
echo "============================================================"
echo ""

should_run() { echo "$STAGES" | grep -qw "$1"; }

# =============================================================================
# Part 1: Custom evaluator on GSM8K — deep analysis
#
# This is the ONLY benchmark using the custom evaluator. It provides:
#   - Error classification (computation / reasoning / comprehension / ...)
#   - Format drift detection (XML structure adherence across stages)
#   - Reasoning quality (repetition, degeneration, truncation)
#   - Pass@k for GRPO
# =============================================================================
echo "============================================================"
echo " Part 1: GSM8K Deep Analysis (custom evaluator)"
echo "============================================================"

if should_run base; then
    run_eval "[1a] BASE on GSM8K" \
        python "$SCRIPT_DIR/evaluate_math_model.py" \
        --model "$BASE_MODEL" \
        --dataset parquet \
        --dataset-path "$GSM8K_PARQUET" \
        --question-field question \
        --answer-field answer \
        --limit "$LIMIT" \
        --max-new-tokens "$MAX_TOKENS" \
        --prompt-style xml \
        --device "$DEVICE" \
        --output "$RESULTS_DIR/base_gsm8k_xml.json" \
        --save-outputs \
        --resume
fi

if should_run sft; then
    SFT_CMD_ARGS=()
    if [ -d "$SFT_MERGED" ]; then
        SFT_CMD_ARGS+=(--model "$SFT_MERGED")
    else
        SFT_CMD_ARGS+=(--model "$SFT_MODEL" --adapter "$SFT_ADAPTER")
    fi
    echo ""
    run_eval "[1b] SFT on GSM8K" \
        python "$SCRIPT_DIR/evaluate_math_model.py" \
        "${SFT_CMD_ARGS[@]}" \
        --dataset parquet \
        --dataset-path "$GSM8K_PARQUET" \
        --question-field question \
        --answer-field answer \
        --limit "$LIMIT" \
        --max-new-tokens "$MAX_TOKENS" \
        --prompt-style xml \
        --device "$DEVICE" \
        --output "$RESULTS_DIR/sft_gsm8k_xml.json" \
        --save-outputs \
        --resume
fi

if should_run grpo; then
    echo ""
    run_eval "[1c] GRPO on GSM8K (+ pass@k)" \
        python "$SCRIPT_DIR/evaluate_math_model.py" \
        --model "$GRPO_MODEL" \
        --dataset parquet \
        --dataset-path "$GSM8K_PARQUET" \
        --question-field question \
        --answer-field answer \
        --limit "$LIMIT" \
        --max-new-tokens "$MAX_TOKENS" \
        --prompt-style xml \
        --device "$DEVICE" \
        --num-samples "$PASS_K_SAMPLES" \
        --pass-k 1 \
        --sampling-temperature 0.7 \
        --output "$RESULTS_DIR/grpo_gsm8k_xml.json" \
        --save-outputs \
        --resume
fi

# =============================================================================
# Part 1b: GSM8K comparison table
# =============================================================================
echo ""
echo "============================================================"
echo " Part 1b: GSM8K comparison table"
echo "============================================================"

COMPARE_INPUTS=()
COMPARE_LABELS=()
for stage in base sft grpo; do
    f="$RESULTS_DIR/${stage}_gsm8k_xml.json"
    if [ -f "$f" ]; then
        COMPARE_INPUTS+=("$f")
        COMPARE_LABELS+=("$stage")
    fi
done

if [ "${#COMPARE_INPUTS[@]}" -ge 2 ]; then
    python "$SCRIPT_DIR/compare_results.py" \
        --inputs "${COMPARE_INPUTS[@]}" \
        --labels "${COMPARE_LABELS[@]}" \
        --output-json "$RESULTS_DIR/gsm8k_comparison.json" \
        --output-md "$RESULTS_DIR/gsm8k_comparison.md" \
        --show-by-difficulty \
        --show-by-topic
    echo ">>> GSM8K comparison saved"
fi

# =============================================================================
# Part 2: lm-eval-harness — standardized benchmarks
#
# minerva_math : competition-level math (replaces custom MATH-500 eval)
# mmlu         : general QA — "Does RL hurt general QA?"
# arc_easy     : science reasoning
# =============================================================================
if [ "$SKIP_LM_EVAL" != "1" ]; then
    if python -c "import lm_eval" 2>/dev/null; then
        echo ""
        echo "============================================================"
        echo " Part 2: lm-eval-harness (standardized benchmarks)"
        echo "============================================================"

        if [ "$QUICK" = "1" ]; then
            LM_EVAL_TASK_LIST="arc_easy"
        else
            LM_EVAL_TASK_LIST="minerva_math mmlu arc_easy"
        fi
        LM_EVAL_LIMIT_FLAG="--limit $LM_EVAL_LIMIT"

        for lm_task in $LM_EVAL_TASK_LIST; do
            if should_run base; then
                run_eval "[2] lm-eval BASE on $lm_task" \
                    python "$SCRIPT_DIR/run_lm_eval.py" \
                    --model "$BASE_MODEL" \
                    --tasks "$lm_task" \
                    --label "base_${lm_task}" \
                    --device "$DEVICE" \
                    $LM_EVAL_LIMIT_FLAG \
                    --output-dir "$RESULTS_DIR/lm_eval"
            fi

            if should_run sft; then
                SFT_LM_EVAL_ARGS=""
                if [ -d "$SFT_MERGED" ]; then
                    SFT_LM_EVAL_MODEL="$SFT_MERGED"
                else
                    SFT_LM_EVAL_MODEL="$SFT_MODEL"
                    SFT_LM_EVAL_ARGS="--adapter $SFT_ADAPTER"
                fi
                run_eval "[2] lm-eval SFT on $lm_task" \
                    python "$SCRIPT_DIR/run_lm_eval.py" \
                    --model "$SFT_LM_EVAL_MODEL" \
                    $SFT_LM_EVAL_ARGS \
                    --tasks "$lm_task" \
                    --label "sft_${lm_task}" \
                    --device "$DEVICE" \
                    $LM_EVAL_LIMIT_FLAG \
                    --output-dir "$RESULTS_DIR/lm_eval"
            fi

            if should_run grpo; then
                run_eval "[2] lm-eval GRPO on $lm_task" \
                    python "$SCRIPT_DIR/run_lm_eval.py" \
                    --model "$GRPO_MODEL" \
                    --tasks "$lm_task" \
                    --label "grpo_${lm_task}" \
                    --device "$DEVICE" \
                    $LM_EVAL_LIMIT_FLAG \
                    --output-dir "$RESULTS_DIR/lm_eval"
            fi
            echo ""
        done
    else
        echo ""
        echo ">>> Skipping lm-eval-harness (not installed). Install with:"
        echo "    pip install lm-eval"
    fi
else
    echo ""
    echo ">>> Skipping lm-eval-harness (SKIP_LM_EVAL=1)"
fi

echo ""
echo "============================================================"
echo " Evaluation complete!"
echo " Results in: $RESULTS_DIR/"
echo "============================================================"
echo ""
echo " Key output files:"
echo "   Custom evaluator (GSM8K deep analysis):"
echo "     $RESULTS_DIR/{base,sft,grpo}_gsm8k_xml.json"
echo "     $RESULTS_DIR/gsm8k_comparison.md"
echo "   lm-eval-harness (standardized benchmarks):"
echo "     $RESULTS_DIR/lm_eval/*_summary.json"
echo "============================================================"

if [ "$FAIL_COUNT" -gt 0 ]; then
    echo ""
    echo "WARNING: $FAIL_COUNT evaluation(s) failed. Check logs above."
    echo "Re-run this script to retry (--resume skips completed custom evals)."
    exit 1
fi
