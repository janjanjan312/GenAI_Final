#!/usr/bin/env python

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
TOKENIZATION_CACHE_VERSION = 1
DEFAULT_SYSTEM_PROMPT = (
    "You are a careful math reasoning assistant. Solve the problem step by step. "
    "End every response with a single line in the format `Final Answer: <answer>`."
)
IGNORE_INDEX = -100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Qwen2.5-0.5B LoRA adapter with Transformers + PEFT."
    )
    parser.add_argument(
        "--dataset-dir",
        default="artifacts/datasets/math_sft_qwen25",
        help="Directory containing train.json and optional validation.json.",
    )
    parser.add_argument("--train-file", help="Override train file path.")
    parser.add_argument("--validation-file", help="Override validation file path.")
    parser.add_argument("--model-name-or-path", default=DEFAULT_MODEL)
    parser.add_argument(
        "--output-dir",
        default="outputs/sft/qwen25_0p5b_math_lora",
        help="Directory for adapter checkpoints and logs.",
    )
    parser.add_argument("--cutoff-len", type=int, default=1024)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated target modules for LoRA.",
    )
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preprocessing-num-workers", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument(
        "--tokenized-cache-dir",
        help="Directory used to persist tokenized datasets. Defaults to <dataset-dir>/tokenized_cache.",
    )
    parser.add_argument(
        "--overwrite-tokenized-cache",
        action="store_true",
        help="Force rebuilding tokenized datasets instead of reusing a matching cache.",
    )
    parser.add_argument(
        "--tokenize-only",
        action="store_true",
        help="Prepare and cache tokenized datasets, then exit before loading the model.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable BF16. Prefer this on Ampere/Hopper GPUs when supported.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable FP16 when BF16 is unavailable.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to tokenizer and model loaders.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Path to an existing Trainer checkpoint to resume from.",
    )
    parser.add_argument(
        "--overwrite-output-dir",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the config snapshot and exit before loading model/training.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, Path | None, Path]:
    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    train_file = Path(args.train_file).resolve() if args.train_file else dataset_dir / "train.json"
    validation_file = (
        Path(args.validation_file).resolve()
        if args.validation_file
        else dataset_dir / "validation.json"
    )

    if not train_file.exists():
        raise FileNotFoundError(f"Train file does not exist: {train_file}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_markers = [output_dir / "adapter_config.json", output_dir / "checkpoint-1"]
    if any(path.exists() for path in checkpoint_markers) and not args.overwrite_output_dir:
        raise FileExistsError(
            f"Output directory already contains training artifacts: {output_dir}. "
            "Re-run with --overwrite-output-dir if this is intentional."
        )

    tokenized_cache_dir = (
        Path(args.tokenized_cache_dir).resolve()
        if args.tokenized_cache_dir
        else (dataset_dir / "tokenized_cache").resolve()
    )
    tokenized_cache_dir.mkdir(parents=True, exist_ok=True)

    return output_dir, train_file, validation_file if validation_file.exists() else None, tokenized_cache_dir


def parse_target_modules(raw_value: str) -> list[str]:
    modules = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not modules:
        raise ValueError("--lora-target-modules must contain at least one module name.")
    return modules


def select_dtype(args: argparse.Namespace) -> Any:
    import torch

    if args.bf16 and args.fp16:
        raise ValueError("Choose only one of --bf16 or --fp16.")
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return None


def write_config_snapshot(args: argparse.Namespace, output_dir: Path, train_file: Path, validation_file: Path | None) -> Path:
    payload = vars(args).copy()
    payload["train_file"] = str(train_file)
    payload["validation_file"] = str(validation_file) if validation_file else None
    payload["output_dir"] = str(output_dir)
    snapshot_path = output_dir / "train_config.json"
    snapshot_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return snapshot_path


def file_signature(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    stats = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stats.st_size,
        "mtime_ns": stats.st_mtime_ns,
    }


def build_tokenized_cache_paths(
    args: argparse.Namespace,
    train_file: Path,
    validation_file: Path | None,
    tokenizer: Any,
    cache_root: Path,
) -> tuple[Path, dict[str, Any]]:
    payload = {
        "version": TOKENIZATION_CACHE_VERSION,
        "model_name_or_path": args.model_name_or_path,
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", args.model_name_or_path),
        "tokenizer_class": tokenizer.__class__.__name__,
        "has_chat_template": bool(getattr(tokenizer, "chat_template", None)),
        "cutoff_len": args.cutoff_len,
        "train_file": file_signature(train_file),
        "validation_file": file_signature(validation_file),
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "trust_remote_code": args.trust_remote_code,
        "default_system_prompt": DEFAULT_SYSTEM_PROMPT,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    return cache_root / f"cache_{digest}", payload


def build_user_text(example: dict[str, Any]) -> str:
    instruction = str(example.get("instruction", "")).strip()
    input_text = str(example.get("input", "")).strip()
    if instruction and input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction or input_text


def encode_example(
    example: dict[str, Any],
    tokenizer: Any,
    cutoff_len: int,
) -> dict[str, Any]:
    system_text = str(example.get("system") or DEFAULT_SYSTEM_PROMPT).strip()
    user_text = build_user_text(example)
    assistant_text = str(example.get("output", "")).strip()
    if not user_text or not assistant_text:
        return {"input_ids": [], "attention_mask": [], "labels": [], "is_valid": False}

    if getattr(tokenizer, "chat_template", None):
        prompt_messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        full_messages = prompt_messages + [{"role": "assistant", "content": assistant_text}]
        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        full_ids = tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
        )
    else:
        prompt_text = (
            f"<|system|>\n{system_text}\n"
            f"<|user|>\n{user_text}\n"
            f"<|assistant|>\n"
        )
        full_text = prompt_text + assistant_text
        prompt_ids = tokenizer(prompt_text, add_special_tokens=True).input_ids
        full_ids = tokenizer(full_text, add_special_tokens=True).input_ids

    if tokenizer.eos_token_id is not None and full_ids and full_ids[-1] != tokenizer.eos_token_id:
        full_ids = full_ids + [tokenizer.eos_token_id]

    full_ids = full_ids[:cutoff_len]
    prompt_len = min(len(prompt_ids), len(full_ids))
    labels = [IGNORE_INDEX] * prompt_len + full_ids[prompt_len:]
    labels = labels[:cutoff_len]
    if not any(label != IGNORE_INDEX for label in labels):
        return {"input_ids": [], "attention_mask": [], "labels": [], "is_valid": False}

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "is_valid": True,
    }


@dataclass
class SupervisedCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must define pad_token_id before collation.")

        max_len = max(len(feature["input_ids"]) for feature in features)
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        for feature in features:
            seq_len = len(feature["input_ids"])
            pad_len = max_len - seq_len
            batch_input_ids.append(feature["input_ids"] + [pad_id] * pad_len)
            batch_attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            batch_labels.append(feature["labels"] + [IGNORE_INDEX] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def set_random_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize_split(
    raw_split: Any,
    split_name: str,
    preprocess_fn: Any,
    num_proc: int,
) -> Any:
    tokenized = raw_split.map(
        preprocess_fn,
        remove_columns=raw_split.column_names,
        num_proc=num_proc,
        desc=f"Tokenizing {split_name} split",
    )
    before_filter = len(tokenized)
    tokenized = tokenized.filter(
        lambda example: example["is_valid"],
        desc=f"Filtering {split_name} split",
    )
    tokenized = tokenized.remove_columns(["is_valid"])
    print(
        f"[INFO] {split_name.capitalize()} examples kept after tokenization: "
        f"{len(tokenized)} / {before_filter}"
    )
    return tokenized


def load_or_build_tokenized_datasets(
    raw_datasets: Any,
    preprocess_fn: Any,
    args: argparse.Namespace,
    cache_root: Path,
    cache_manifest: dict[str, Any],
) -> tuple[Any, Any]:
    from datasets import load_from_disk

    train_cache_dir = cache_root / "train"
    validation_cache_dir = cache_root / "validation"
    manifest_path = cache_root / "cache_manifest.json"

    if (
        not args.overwrite_tokenized_cache
        and manifest_path.exists()
        and train_cache_dir.exists()
        and ("validation" not in raw_datasets or validation_cache_dir.exists())
    ):
        print(f"[INFO] Reusing tokenized train cache from {train_cache_dir}")
        train_dataset = load_from_disk(str(train_cache_dir))
        eval_dataset = None
        if "validation" in raw_datasets and validation_cache_dir.exists():
            print(f"[INFO] Reusing tokenized validation cache from {validation_cache_dir}")
            eval_dataset = load_from_disk(str(validation_cache_dir))
        return train_dataset, eval_dataset

    print(f"[INFO] Building tokenized cache under {cache_root}")
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(cache_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    train_dataset = tokenize_split(
        raw_split=raw_datasets["train"],
        split_name="train",
        preprocess_fn=preprocess_fn,
        num_proc=args.preprocessing_num_workers,
    )
    train_dataset.save_to_disk(str(train_cache_dir))

    eval_dataset = None
    if "validation" in raw_datasets:
        eval_dataset = tokenize_split(
            raw_split=raw_datasets["validation"],
            split_name="validation",
            preprocess_fn=preprocess_fn,
            num_proc=args.preprocessing_num_workers,
        )
        eval_dataset.save_to_disk(str(validation_cache_dir))

    return train_dataset, eval_dataset


def build_training_arguments(TrainingArguments: Any, args: argparse.Namespace, output_dir: Path, use_eval: bool) -> Any:
    signature = inspect.signature(TrainingArguments.__init__)
    kwargs = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": args.overwrite_output_dir,
        "do_train": True,
        "do_eval": use_eval,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "dataloader_num_workers": 0,
        "gradient_checkpointing": True,
        "report_to": [],
        "remove_unused_columns": False,
        "seed": args.seed,
        "data_seed": args.seed,
        "label_names": ["labels"],
    }

    if use_eval:
        if "evaluation_strategy" in signature.parameters:
            kwargs["evaluation_strategy"] = "steps"
        elif "eval_strategy" in signature.parameters:
            kwargs["eval_strategy"] = "steps"
        kwargs["eval_steps"] = args.eval_steps
    else:
        if "evaluation_strategy" in signature.parameters:
            kwargs["evaluation_strategy"] = "no"
        elif "eval_strategy" in signature.parameters:
            kwargs["eval_strategy"] = "no"

    return TrainingArguments(**kwargs)


def main() -> None:
    args = parse_args()
    output_dir, train_file, validation_file, tokenized_cache_dir = validate_args(args)
    snapshot_path = write_config_snapshot(args, output_dir=output_dir, train_file=train_file, validation_file=validation_file)
    print(f"[INFO] Training config written to {snapshot_path}")

    if args.dry_run:
        print("[INFO] Dry run completed. Training was not launched.")
        return

    set_random_seed(args.seed)

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"

    cache_dir, cache_manifest = build_tokenized_cache_paths(
        args=args,
        train_file=train_file,
        validation_file=validation_file,
        tokenizer=tokenizer,
        cache_root=tokenized_cache_dir,
    )

    data_files: dict[str, str] = {"train": str(train_file)}
    if validation_file:
        data_files["validation"] = str(validation_file)

    raw_datasets = load_dataset("json", data_files=data_files)
    if args.max_train_samples is not None:
        raw_datasets["train"] = raw_datasets["train"].select(
            range(min(args.max_train_samples, len(raw_datasets["train"])))
        )
    if validation_file and args.max_eval_samples is not None:
        raw_datasets["validation"] = raw_datasets["validation"].select(
            range(min(args.max_eval_samples, len(raw_datasets["validation"])))
        )

    preprocess_fn = lambda example: encode_example(example, tokenizer=tokenizer, cutoff_len=args.cutoff_len)

    train_dataset, eval_dataset = load_or_build_tokenized_datasets(
        raw_datasets=raw_datasets,
        preprocess_fn=preprocess_fn,
        args=args,
        cache_root=cache_dir,
        cache_manifest=cache_manifest,
    )

    if args.tokenize_only:
        print("[INFO] Tokenize-only mode completed. Training was not launched.")
        return

    torch_dtype = select_dtype(args)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=parse_target_modules(args.lora_target_modules),
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    collator = SupervisedCollator(tokenizer=tokenizer)
    use_eval = eval_dataset is not None and len(eval_dataset) > 0

    training_args = build_training_arguments(
        TrainingArguments=TrainingArguments,
        args=args,
        output_dir=output_dir,
        use_eval=use_eval,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if use_eval else None,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    if use_eval:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    print(f"[INFO] Training completed. Adapter saved to {output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
