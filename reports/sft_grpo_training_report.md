# DeepMath XML SFT + GRPO Training Report

## 1. Goal

This run trained a math model through a full three-stage pipeline:

1. Build XML-format SFT data from DeepMath Phase A.
2. Perform LoRA SFT cold-start from an existing math-tuned checkpoint.
3. Continue optimization with TRL `GRPOTrainer`.

The main objective was to improve answer accuracy while pushing the model toward the following output format:

```xml
<think>
brief reasoning
</think>
<answer>
\boxed{final answer}
</answer>
```

## 2. Data Sources

### 2.1 SFT source

- Raw file: `final_project/outputs/phase_a_deepmath/deepmath_phase_a_alpaca.jsonl`
- Each row contains Alpaca-style fields plus metadata from DeepMath.

### 2.2 GRPO source

- Raw file: `final_project/outputs/phase_a_deepmath/deepmath_phase_a.parquet`
- Key fields used:
  - `question`
  - `final_answer`
  - `final_answer_norm`
  - `difficulty`
  - `topic`

## 3. SFT Data Construction

SFT data was rebuilt into a stricter XML format rather than directly using the original long-form Alpaca outputs.

### 3.1 Conversion logic

Implemented in `grpo_trl/prepare_deepmath_xml_sft_dataset.py`.

Key steps:

1. Read the Alpaca JSONL file line by line.
2. Extract the question from `instruction`.
3. Clean the original long reasoning text:
   - strip repeated `Final Answer` sections
   - strip trailing boxed answers from the reasoning body
   - remove conversational fillers such as `Okay, so ...`
4. Truncate reasoning to a bounded length (`max_think_chars`, default `1200`).
5. Rebuild the assistant response into strict XML:

```xml
<think>
...
</think>
<answer>
\boxed{final_answer_norm}
</answer>
```

6. Save the output as chat `messages` in parquet format.

### 3.2 SFT output schema

Each SFT example contains:

- `messages`
- `question`
- `topic`
- `difficulty`
- `final_answer_norm`
- `answer_format`
- `source`

### 3.3 Split sizes

Full XML SFT data:

- Total usable examples: `100,891`
- Train: `99,891`
- Val: `1,000`

Final SFT run used a reduced cold-start subset:

- Train: `30,000`
- Val: `1,000`

## 4. SFT Cold Start

### 4.1 Initialization

The SFT run did **not** start from `qwen2.5-base`.

It started from the existing local model:

- Remote training model: `/root/autodl-tmp/5740/qwen25_0p5b_math_merged`

This choice was made because the existing math-tuned checkpoint already had better math behavior than the untouched base model.

### 4.2 Training method

Implemented in `grpo_trl/train_sft_lora.py`.

Method:

- `transformers.Trainer`
- LoRA fine-tuning with PEFT
- causal LM objective on chat-formatted XML supervision

LoRA targets:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

### 4.3 Final SFT configuration used

The final accepted SFT run used:

- model: `qwen25_0p5b_math_merged`
- train size: `30k`
- val size: `1k`
- learning rate: `2e-4`
- `per_device_train_batch_size = 8`
- `per_device_eval_batch_size = 2`
- `gradient_accumulation_steps = 4`
- `max_seq_length = 1024`
- epochs: `1`
- save/eval steps: `250`

### 4.4 SFT artifacts

Main outputs:

- LoRA checkpoints
- `final_adapter`
- merged model after adapter merge

Relevant remote paths:

- SFT LoRA output:
  `/root/autodl-tmp/5740/grpo_trl/artifacts/xml_sft_lora_30k_b8_g4`
- merged model:
  `/root/autodl-tmp/5740/grpo_trl/artifacts/xml_sft_merged_30k_b8_g4`

### 4.5 SFT outcome

Observed final SFT metrics:

- `train_loss = 0.5701`
- `eval_loss = 0.6167`

Qualitative result:

- The merged SFT model learned partial XML behavior.
- Under larger generation budgets it could produce XML segments and correct boxed answers.
- However, it often failed to stop cleanly after a single valid XML answer block.

This led to the next stage: GRPO.

## 5. GRPO Training

### 5.1 GRPO data construction

Implemented in `grpo_trl/prepare_deepmath_grpo_dataset.py`.

Each GRPO example contains:

- `prompt`
- `ground_truth`
- `question`
- `topic`
- `difficulty`
- `answer_format`
- `response_prefix`
- `data_source`
- `row_id`

Prompt construction used:

- a system prompt specifying strict XML format
- the user question
- an assistant prefix of:

```text
<think>
```

This was designed to bias generation toward structured XML output.

### 5.2 Reward design

Implemented in `grpo_trl/reward_math_format_accuracy.py`.

The final reward stack combined several terms:

- `format_progress_reward_func`
- `xml_tag_shape_reward_func`
- `strict_format_reward_func`
- `brevity_reward_func`
- `answer_similarity_reward_func`
- `answer_reward_func`
- `repetition_penalty_func`

This reward design was necessary because the original strict-only reward was too sparse and often yielded all-zero rewards.

### 5.3 Final GRPO configuration used

Implemented in `grpo_trl/train_grpo_trl.py` and `grpo_trl/run_grpo_qwen25_math.sh`.

Final monitored run:

- initialization model:
  `/root/autodl-tmp/5740/grpo_trl/artifacts/xml_sft_merged_30k_b8_g4`
- trainer: `trl.GRPOTrainer`
- `max_completion_length = 1024`
- `max_steps = 100`
- `per_device_train_batch_size = 1`
- `per_device_eval_batch_size = 4`
- `gradient_accumulation_steps = 4`
- `num_generations = 4`
- `learning_rate = 1e-6`
- `temperature = 0.8`
- `top_p = 0.95`
- validation size reduced to `64`
- `eval_steps = 50`
- `save_steps = 50`
- logging backend: `SwanLab`

### 5.4 GRPO artifacts

Relevant remote output directory:

- `/root/autodl-tmp/5740/grpo_trl/artifacts/grpo_swanlab_v2`

Key checkpoints:

- `checkpoint-50`
- `checkpoint-100`
- `final_model`

### 5.5 GRPO outcome

Final training metrics:

- `loss = 0.06538`
- `reward = 0.925`
- `entropy = 0.1644`

Final evaluation metrics on the 64-sample validation set:

- `eval_reward = 0.6139`
- `eval_entropy = 0.106`
- `eval_reward_std = 0.1864`
- `eval_rewards/answer_reward_func/mean = 0.175`
- `eval_rewards/strict_format_reward_func/mean = 0`
- `eval_completions/clipped_ratio = 0.8828`

Interpretation:

- GRPO improved answer behavior and reward quality.
- The model still did not learn perfectly clean XML stopping behavior.
- Strict XML formatting remained the weakest part of the pipeline.

## 6. GSM8K Evaluation

Evaluation script:

- `grpo_trl/evaluate_gsm8k.py`

Protocol:

- Dataset: GSM8K test split, first `1000` samples
- `max_new_tokens = 512`
- `batch_size = 4`

Compared models:

1. Base math model:
   `/root/autodl-tmp/5740/qwen25_0p5b_math_merged`
2. Final GRPO model:
   `/root/autodl-tmp/5740/grpo_trl/artifacts/grpo_swanlab_v2/checkpoints/final_model`

Results:

- Base:
  - `47 / 1000`
  - `accuracy = 0.047`
- Final GRPO model:
  - `220 / 1000`
  - `accuracy = 0.220`

Improvement:

- absolute gain: `+17.3` points
- relative gain: about `4.7x`

## 7. Local Downloaded Weights

The final model weights were downloaded from the remote server to the local workspace for archival and further testing.

Remote source:

- `/root/autodl-tmp/5740/grpo_trl/artifacts/grpo_swanlab_v2/checkpoints/final_model`

Expected local destination:

- `downloads/final_model`

The downloaded directory contains:

- `model.safetensors`
- `config.json`
- tokenizer files
- generation config
- `training_args.bin`

These weights are not suitable for direct GitHub versioning because the directory is about `1.9G`.

## 8. Uploaded Code

The relevant code used in this run has been copied into:

- `grpo_trl/`

Included files:

- `prepare_deepmath_xml_sft_dataset.py`
- `train_sft_lora.py`
- `merge_lora_adapter.py`
- `prepare_deepmath_grpo_dataset.py`
- `reward_math_format_accuracy.py`
- `train_grpo_trl.py`
- `run_grpo_qwen25_math.sh`
- `run_xml_sft_then_grpo.sh`
- `evaluate_gsm8k.py`
- `test_xml_following.py`
- `README_zh.md`

## 9. Summary

This training run established a full `SFT -> merge -> GRPO -> GSM8K evaluation` pipeline on top of TRL.

Main conclusions:

1. Rebuilding the SFT data into XML format was necessary to steer the model toward structured output.
2. LoRA SFT on top of the existing math-tuned checkpoint provided a workable initialization.
3. GRPO improved math answer accuracy substantially, even though strict XML termination remained imperfect.
4. On GSM8K 1k samples, the final GRPO model improved from `4.7%` to `22.0%` accuracy over the starting math model.
