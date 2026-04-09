#!/usr/bin/env python3
"""Supervised LoRA fine-tuning for XML-formatted DeepMath data."""

from __future__ import annotations

import argparse
import inspect
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Base model path or HF model id.")
    parser.add_argument("--train-data", required=True, type=Path)
    parser.add_argument("--eval-data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-name", default="qwen25_0p5b_deepmath_xml_sft_lora")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def load_parquet_dataset(path: Path):
    return load_dataset("parquet", data_files=str(path))["train"]


def build_tokenize_fn(tokenizer, max_seq_length: int):
    def tokenize_row(row: dict) -> dict:
        messages = row["messages"]
        prompt_messages = messages[:-1]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = full_ids[:max_seq_length]
        attention_mask = [1] * len(full_ids)

        labels = full_ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return tokenize_row


@dataclass
class SupervisedDataCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features):
        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(feature["input_ids"]) for feature in features)

        input_ids = []
        attention_masks = []
        labels = []
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_id] * pad_len)
            attention_masks.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.float16 if args.fp16 else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)

    train_dataset = load_parquet_dataset(args.train_data)
    eval_dataset = load_parquet_dataset(args.eval_data)
    tokenize_fn = build_tokenize_fn(tokenizer, args.max_seq_length)
    train_dataset = train_dataset.map(tokenize_fn, remove_columns=train_dataset.column_names, desc="Tokenizing train")
    eval_dataset = eval_dataset.map(tokenize_fn, remove_columns=eval_dataset.column_names, desc="Tokenizing eval")

    training_kwargs = dict(
        output_dir=str(args.output_dir),
        run_name=args.run_name,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=args.bf16 or not args.fp16,
        fp16=args.fp16,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        max_steps=args.max_steps,
    )
    strategy_arg = "evaluation_strategy" if "evaluation_strategy" in inspect.signature(TrainingArguments).parameters else "eval_strategy"
    training_kwargs[strategy_arg] = "steps"
    training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SupervisedDataCollator(tokenizer),
    )
    trainer_signature = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(str(args.output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(args.output_dir / "final_adapter"))


if __name__ == "__main__":
    main()
