#!/usr/bin/env python3
"""Train GRPO with TRL on the DeepMath Phase A dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from reward_math_format_accuracy import (
    answer_reward_func,
    answer_similarity_reward_func,
    brevity_reward_func,
    format_progress_reward_func,
    repetition_penalty_func,
    strict_format_reward_func,
    xml_tag_shape_reward_func,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local model path or HF model id.")
    parser.add_argument("--train-data", required=True, type=Path, help="Training parquet path.")
    parser.add_argument("--eval-data", type=Path, default=None, help="Validation parquet path.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Checkpoint output directory.")
    parser.add_argument("--run-name", default="qwen25_0p5b_deepmath_grpo_trl", help="Experiment name.")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--save-steps", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--report-to", default="none", help="Trainer reporting backend, e.g. none or swanlab.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", help="Force bf16 training.")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 instead of bf16.")
    parser.add_argument("--use-vllm", action="store_true", help="Enable vLLM generation.")
    parser.add_argument("--vllm-mode", choices=["colocate", "server"], default="colocate")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


def load_parquet_dataset(path: Path):
    dataset_dict = load_dataset("parquet", data_files=str(path))
    return dataset_dict["train"]


def main() -> None:
    args = parse_args()

    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "TRL is not installed. Install dependencies first, e.g. "
            "`pip install trl accelerate peft` or `pip install 'trl[vllm]' accelerate peft`."
        ) from exc

    train_dataset = load_parquet_dataset(args.train_data)
    eval_dataset = load_parquet_dataset(args.eval_data) if args.eval_data else None

    effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
    if effective_batch % args.num_generations != 0:
        raise ValueError(
            "TRL requires effective batch size "
            "(per_device_train_batch_size * gradient_accumulation_steps * num_processes) "
            f"to be divisible by num_generations. Current per-process value is {effective_batch}, "
            f"num_generations is {args.num_generations}."
        )

    eval_batch = args.per_device_eval_batch_size
    if eval_dataset is not None and eval_batch % args.num_generations != 0:
        raise ValueError(
            "TRL requires eval batch size "
            "(per_device_eval_batch_size * num_processes) "
            f"to be divisible by num_generations. Current per-process eval batch is {eval_batch}, "
            f"num_generations is {args.num_generations}."
        )

    bf16 = args.bf16 or (not args.fp16)
    training_args = GRPOConfig(
        output_dir=str(args.output_dir),
        run_name=args.run_name,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        warmup_steps=args.warmup_steps,
        beta=args.beta,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        bf16=bf16,
        fp16=args.fp16,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to=args.report_to,
        log_completions=True,
        mask_truncated_completions=True,
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
    )

    trainer = GRPOTrainer(
        model=args.model,
        args=training_args,
        reward_funcs=[
            format_progress_reward_func,
            xml_tag_shape_reward_func,
            strict_format_reward_func,
            brevity_reward_func,
            answer_similarity_reward_func,
            answer_reward_func,
            repetition_penalty_func,
        ],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir / "final_model"))


if __name__ == "__main__":
    main()
