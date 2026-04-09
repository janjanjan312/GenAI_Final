#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PARENT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR_DEFAULT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -d "${SCRIPT_PARENT}/qwen25_0p5b_math_merged" ]]; then
  ROOT_DIR="${ROOT_DIR:-${SCRIPT_PARENT}}"
  RAW_DATA_DEFAULT="${SCRIPT_PARENT}/deepmath_phase_a.parquet"
  MODEL_PATH_DEFAULT="${SCRIPT_PARENT}/qwen25_0p5b_math_merged"
else
  ROOT_DIR="${ROOT_DIR:-${ROOT_DIR_DEFAULT}}"
  RAW_DATA_DEFAULT="${ROOT_DIR}/final_project/outputs/phase_a_deepmath/deepmath_phase_a.parquet"
  MODEL_PATH_DEFAULT="${ROOT_DIR}/upload/qwen25_0p5b_math_merged"
fi

RAW_DATA="${RAW_DATA:-${RAW_DATA_DEFAULT}}"
MODEL_PATH="${MODEL_PATH:-${MODEL_PATH_DEFAULT}}"
WORK_DIR="${WORK_DIR:-${SCRIPT_DIR}/artifacts}"
DATA_DIR="${DATA_DIR:-${WORK_DIR}/data}"
TRAIN_PARQUET="${TRAIN_PARQUET:-${DATA_DIR}/train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-${DATA_DIR}/val.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_DIR}/checkpoints}"

VAL_SIZE="${VAL_SIZE:-1000}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
FORCE_PREPARE="${FORCE_PREPARE:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
SAVE_STEPS="${SAVE_STEPS:-20}"
EVAL_STEPS="${EVAL_STEPS:-20}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
BETA="${BETA:-0.0}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"
RUN_NAME="${RUN_NAME:-qwen25_0p5b_deepmath_grpo_trl}"
REPORT_TO="${REPORT_TO:-none}"
SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-AIMS5740-GRPO}"
SWANLAB_EXPERIMENT_NAME="${SWANLAB_EXPERIMENT_NAME:-${RUN_NAME}}"
USE_VLLM="${USE_VLLM:-0}"
VLLM_MODE="${VLLM_MODE:-colocate}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.55}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

mkdir -p "${WORK_DIR}" "${DATA_DIR}" "${OUTPUT_DIR}"

if [[ -n "${SWANLAB_API_KEY}" ]]; then
  export SWANLAB_API_KEY
  export SWANLAB_PROJECT
  export SWANLAB_EXPERIMENT_NAME
fi

if [[ "${FORCE_PREPARE}" == "1" || ! -f "${TRAIN_PARQUET}" || ! -f "${VAL_PARQUET}" || "${SCRIPT_DIR}/prepare_deepmath_grpo_dataset.py" -nt "${TRAIN_PARQUET}" ]]; then
  DATA_CMD=(
    python3 "${SCRIPT_DIR}/prepare_deepmath_grpo_dataset.py"
    --input "${RAW_DATA}"
    --output-dir "${DATA_DIR}"
    --val-size "${VAL_SIZE}"
  )
  if [[ -n "${MAX_SAMPLES}" ]]; then
    DATA_CMD+=(--max-samples "${MAX_SAMPLES}")
  fi
  "${DATA_CMD[@]}"
fi

TRAIN_CMD=(
  accelerate launch
  --num_processes "${NUM_PROCESSES}"
  --mixed_precision "${MIXED_PRECISION}"
  "${SCRIPT_DIR}/train_grpo_trl.py"
  --model "${MODEL_PATH}"
  --train-data "${TRAIN_PARQUET}"
  --eval-data "${VAL_PARQUET}"
  --output-dir "${OUTPUT_DIR}"
  --run-name "${RUN_NAME}"
  --learning-rate "${LEARNING_RATE}"
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --per-device-eval-batch-size "${PER_DEVICE_EVAL_BATCH_SIZE}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  --num-generations "${NUM_GENERATIONS}"
  --max-completion-length "${MAX_COMPLETION_LENGTH}"
  --num-train-epochs "${NUM_TRAIN_EPOCHS}"
  --save-steps "${SAVE_STEPS}"
  --eval-steps "${EVAL_STEPS}"
  --logging-steps "${LOGGING_STEPS}"
  --beta "${BETA}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --report-to "${REPORT_TO}"
)

if [[ -n "${MAX_STEPS}" ]]; then
  TRAIN_CMD+=(--max-steps "${MAX_STEPS}")
fi

if [[ "${MIXED_PRECISION}" == "fp16" ]]; then
  TRAIN_CMD+=(--fp16)
else
  TRAIN_CMD+=(--bf16)
fi

if [[ "${USE_VLLM}" == "1" ]]; then
  TRAIN_CMD+=(
    --use-vllm
    --vllm-mode "${VLLM_MODE}"
    --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
    --vllm-tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}"
  )
fi

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  TRAIN_CMD+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

"${TRAIN_CMD[@]}"
