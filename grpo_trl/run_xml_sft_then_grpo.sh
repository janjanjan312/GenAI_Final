#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PARENT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR_DEFAULT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -d "${SCRIPT_PARENT}/qwen25_0p5b_math_merged" ]]; then
  ROOT_DIR="${ROOT_DIR:-${SCRIPT_PARENT}}"
  ALPACA_DATA_DEFAULT="${SCRIPT_PARENT}/deepmath_phase_a_alpaca.jsonl"
  GRPO_RAW_DATA_DEFAULT="${SCRIPT_PARENT}/deepmath_phase_a.parquet"
  BASE_MODEL_DEFAULT="${SCRIPT_PARENT}/qwen25_0p5b_math_merged"
else
  ROOT_DIR="${ROOT_DIR:-${ROOT_DIR_DEFAULT}}"
  ALPACA_DATA_DEFAULT="${ROOT_DIR}/final_project/outputs/phase_a_deepmath/deepmath_phase_a_alpaca.jsonl"
  GRPO_RAW_DATA_DEFAULT="${ROOT_DIR}/final_project/outputs/phase_a_deepmath/deepmath_phase_a.parquet"
  BASE_MODEL_DEFAULT="${ROOT_DIR}/upload/qwen25_0p5b_math_merged"
fi

ALPACA_DATA="${ALPACA_DATA:-${ALPACA_DATA_DEFAULT}}"
GRPO_RAW_DATA="${GRPO_RAW_DATA:-${GRPO_RAW_DATA_DEFAULT}}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${BASE_MODEL_DEFAULT}}"
WORK_DIR="${WORK_DIR:-${SCRIPT_DIR}/artifacts/xml_pipeline}"
SFT_DATA_DIR="${SFT_DATA_DIR:-${WORK_DIR}/sft_data}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-${WORK_DIR}/sft_lora}"
MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-${WORK_DIR}/sft_merged_model}"
GRPO_WORK_DIR="${GRPO_WORK_DIR:-${WORK_DIR}/grpo}"

VAL_SIZE="${VAL_SIZE:-1000}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_THINK_CHARS="${MAX_THINK_CHARS:-1200}"
FORCE_PREPARE_SFT="${FORCE_PREPARE_SFT:-0}"

SFT_NUM_PROCESSES="${SFT_NUM_PROCESSES:-1}"
SFT_MIXED_PRECISION="${SFT_MIXED_PRECISION:-bf16}"
SFT_LEARNING_RATE="${SFT_LEARNING_RATE:-2e-4}"
SFT_PER_DEVICE_TRAIN_BATCH_SIZE="${SFT_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
SFT_PER_DEVICE_EVAL_BATCH_SIZE="${SFT_PER_DEVICE_EVAL_BATCH_SIZE:-1}"
SFT_GRADIENT_ACCUMULATION_STEPS="${SFT_GRADIENT_ACCUMULATION_STEPS:-16}"
SFT_NUM_TRAIN_EPOCHS="${SFT_NUM_TRAIN_EPOCHS:-1}"
SFT_LOGGING_STEPS="${SFT_LOGGING_STEPS:-10}"
SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-200}"
SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-200}"
SFT_MAX_SEQ_LENGTH="${SFT_MAX_SEQ_LENGTH:-2048}"

mkdir -p "${WORK_DIR}" "${SFT_DATA_DIR}" "${SFT_OUTPUT_DIR}" "${GRPO_WORK_DIR}"

SFT_TRAIN_PARQUET="${SFT_DATA_DIR}/train.parquet"
SFT_VAL_PARQUET="${SFT_DATA_DIR}/val.parquet"

if [[ "${FORCE_PREPARE_SFT}" == "1" || ! -f "${SFT_TRAIN_PARQUET}" || ! -f "${SFT_VAL_PARQUET}" || "${SCRIPT_DIR}/prepare_deepmath_xml_sft_dataset.py" -nt "${SFT_TRAIN_PARQUET}" ]]; then
  DATA_CMD=(
    python3 "${SCRIPT_DIR}/prepare_deepmath_xml_sft_dataset.py"
    --input "${ALPACA_DATA}"
    --output-dir "${SFT_DATA_DIR}"
    --val-size "${VAL_SIZE}"
    --max-think-chars "${MAX_THINK_CHARS}"
  )
  if [[ -n "${MAX_SAMPLES}" ]]; then
    DATA_CMD+=(--max-samples "${MAX_SAMPLES}")
  fi
  "${DATA_CMD[@]}"
fi

accelerate launch \
  --num_processes "${SFT_NUM_PROCESSES}" \
  --mixed_precision "${SFT_MIXED_PRECISION}" \
  "${SCRIPT_DIR}/train_sft_lora.py" \
  --model "${BASE_MODEL_PATH}" \
  --train-data "${SFT_TRAIN_PARQUET}" \
  --eval-data "${SFT_VAL_PARQUET}" \
  --output-dir "${SFT_OUTPUT_DIR}" \
  --learning-rate "${SFT_LEARNING_RATE}" \
  --per-device-train-batch-size "${SFT_PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per-device-eval-batch-size "${SFT_PER_DEVICE_EVAL_BATCH_SIZE}" \
  --gradient-accumulation-steps "${SFT_GRADIENT_ACCUMULATION_STEPS}" \
  --num-train-epochs "${SFT_NUM_TRAIN_EPOCHS}" \
  --logging-steps "${SFT_LOGGING_STEPS}" \
  --save-steps "${SFT_SAVE_STEPS}" \
  --eval-steps "${SFT_EVAL_STEPS}" \
  --max-seq-length "${SFT_MAX_SEQ_LENGTH}" \
  --bf16

python3 "${SCRIPT_DIR}/merge_lora_adapter.py" \
  --base-model "${BASE_MODEL_PATH}" \
  --adapter "${SFT_OUTPUT_DIR}/final_adapter" \
  --output-dir "${MERGED_MODEL_DIR}"

RAW_DATA="${GRPO_RAW_DATA}" \
MODEL_PATH="${MERGED_MODEL_DIR}" \
WORK_DIR="${GRPO_WORK_DIR}" \
FORCE_PREPARE=1 \
bash "${SCRIPT_DIR}/run_grpo_qwen25_math.sh"
