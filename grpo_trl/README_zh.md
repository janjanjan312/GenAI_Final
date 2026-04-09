# DeepMath 上的 SFT + GRPO 训练流程说明

本目录保存了这次实验中真正用到的训练代码。整个流程不是单独讲 `GRPO`，而是一个完整的三阶段训练链路：

1. 构造 XML 格式的 SFT 数据
2. 基于已有数学模型做 LoRA SFT 冷启动
3. 在 SFT 模型上继续做 GRPO 强化训练

最终还对模型做了 GSM8K 评测。

## 1. 训练目标

目标是让模型在数学题上同时具备两种能力：

- 答案正确率提升
- 输出尽量满足统一 XML 格式

目标输出格式为：

```xml
<think>
brief reasoning
</think>
<answer>
\boxed{final answer}
</answer>
```

## 2. 数据与代码分工

### 2.1 SFT 数据构造

使用文件：

- `prepare_deepmath_xml_sft_dataset.py`

输入数据：

- `deepmath_phase_a_alpaca.jsonl`

处理逻辑：

1. 从 Alpaca 数据里读取题目和原始长推理。
2. 删除原始回答中重复的 `Final Answer` 段落。
3. 去掉多余的口语化前缀，例如 `Okay, so ...`。
4. 截断过长推理，只保留相对精炼的 reasoning。
5. 用 `final_answer_norm` 重建最终答案。
6. 统一重写为 `<think>...</think><answer>\boxed{...}</answer>`。

输出数据格式：

- `messages`
- `question`
- `topic`
- `difficulty`
- `final_answer_norm`
- `answer_format`
- `source`

### 2.2 SFT 训练

使用文件：

- `train_sft_lora.py`
- `merge_lora_adapter.py`

训练方式：

- `transformers.Trainer`
- `PEFT LoRA`
- 监督微调 chat-format XML 数据

LoRA 目标模块：

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

本次最终采用的冷启动模型不是 `qwen2.5-base`，而是已经存在的数学模型：

- `qwen25_0p5b_math_merged`

这样做的原因很直接：它已经有基础数学能力，用它做 XML 格式冷启动比从纯 base 重新学更稳。

### 2.3 GRPO 数据构造

使用文件：

- `prepare_deepmath_grpo_dataset.py`

输入数据：

- `deepmath_phase_a.parquet`

输出字段：

- `prompt`
- `ground_truth`
- `question`
- `topic`
- `difficulty`
- `answer_format`
- `response_prefix`
- `data_source`
- `row_id`

这里的 `prompt` 采用 chat 形式，里面包含：

- system：要求模型输出严格 XML
- user：题目
- assistant prefix：`<think>\n`

`ground_truth` 用于 reward 计算。

### 2.4 GRPO 训练

使用文件：

- `reward_math_format_accuracy.py`
- `train_grpo_trl.py`
- `run_grpo_qwen25_math.sh`

训练框架：

- `trl.GRPOTrainer`

最终奖励函数不是只有一个“答对/答错”硬奖励，而是组合式设计：

- `format_progress_reward_func`
- `xml_tag_shape_reward_func`
- `strict_format_reward_func`
- `brevity_reward_func`
- `answer_similarity_reward_func`
- `answer_reward_func`
- `repetition_penalty_func`

这样做的原因是：如果只用严格格式奖励和最终答案奖励，早期训练经常全部是 `0 reward`，GRPO 很难学动。

## 3. 实际训练流程

### 3.1 第一步：构造 XML SFT 数据

从原始 `deepmath_phase_a_alpaca.jsonl` 生成 XML 格式监督数据。

完整 XML SFT 数据规模：

- 总样本：`100,891`
- train：`99,891`
- val：`1,000`

### 3.2 第二步：SFT 冷启动

最终采用的是 30k 子集做冷启动：

- train：`30,000`
- val：`1,000`

最终采用的 SFT 配置：

- base model：`qwen25_0p5b_math_merged`
- learning rate：`2e-4`
- `per_device_train_batch_size = 8`
- `per_device_eval_batch_size = 2`
- `gradient_accumulation_steps = 4`
- `max_seq_length = 1024`
- epoch：`1`

SFT 结果：

- `train_loss = 0.5701`
- `eval_loss = 0.6167`

SFT 后得到两个关键产物：

- `final_adapter`
- merge 后的 `xml_sft_merged_30k_b8_g4`

### 3.3 第三步：GRPO 训练

GRPO 以上一步 merge 后的 SFT 模型作为初始化模型。

最终采用的 GRPO 配置：

- init model：`xml_sft_merged_30k_b8_g4`
- `max_completion_length = 1024`
- `max_steps = 100`
- `per_device_train_batch_size = 1`
- `per_device_eval_batch_size = 4`
- `gradient_accumulation_steps = 4`
- `num_generations = 4`
- `learning_rate = 1e-6`
- `temperature = 0.8`
- `top_p = 0.95`
- val size：`64`
- `eval_steps = 50`
- `save_steps = 50`

训练日志通过 `SwanLab` 记录，重点看：

- `loss`
- `reward`
- `entropy`
- 各个 reward 子项

最终 GRPO 训练指标：

- `loss = 0.06538`
- `reward = 0.925`
- `entropy = 0.1644`

最终 eval 指标：

- `eval_reward = 0.6139`
- `eval_entropy = 0.106`
- `eval_reward_std = 0.1864`
- `eval_rewards/answer_reward_func/mean = 0.175`
- `eval_rewards/strict_format_reward_func/mean = 0`

这说明：

- 数学答案能力有明显提升
- 但“严格只输出一次完整 XML 并停止”还没有完全学稳

## 4. 评测

使用文件：

- `evaluate_gsm8k.py`
- `test_xml_following.py`

### 4.1 XML 跟随测试

观察到的现象是：

- SFT 和 GRPO 都能让模型更容易生成出 XML 片段
- 但模型仍然可能在输出一段正确 XML 之后继续重复生成

所以当前模型的主要问题不是完全不会 XML，而是停止行为还不够稳定。

### 4.2 GSM8K 1k 对比结果

测试设置：

- GSM8K 前 `1000` 条
- `max_new_tokens = 512`
- `batch_size = 4`

对比模型：

1. 初始数学模型 `qwen25_0p5b_math_merged`
2. 最终 GRPO 模型 `final_model`

结果：

- 初始模型：
  - `47 / 1000`
  - `accuracy = 4.7%`
- 最终模型：
  - `220 / 1000`
  - `accuracy = 22.0%`

结论：

- 经过 `SFT + GRPO` 之后，GSM8K 1k 准确率从 `4.7%` 提升到 `22.0%`
- 说明整条训练链路对数学求解能力是有效的

## 5. 目录说明

- `prepare_deepmath_xml_sft_dataset.py`
  - 构造 XML SFT 数据
- `train_sft_lora.py`
  - LoRA SFT 训练
- `merge_lora_adapter.py`
  - 合并 SFT adapter
- `prepare_deepmath_grpo_dataset.py`
  - 构造 GRPO 数据
- `reward_math_format_accuracy.py`
  - GRPO 奖励函数
- `train_grpo_trl.py`
  - GRPO 训练入口
- `run_grpo_qwen25_math.sh`
  - 单独运行 GRPO
- `run_xml_sft_then_grpo.sh`
  - 串联 SFT -> merge -> GRPO
- `test_xml_following.py`
  - XML 跟随测试
- `evaluate_gsm8k.py`
  - GSM8K 批量评测

## 6. 总结

这次实验最重要的不是“从哪个框架切到哪个框架”，而是完整训练链路已经打通：

1. 从 DeepMath 原始长回答中构造出 XML SFT 数据
2. 在已有数学模型上做 LoRA SFT 冷启动
3. 在 SFT 模型上继续做 GRPO 强化训练
4. 最终在 GSM8K 上得到明显提升

当前最主要的剩余问题是：

- 模型答案能力已经提升
- 但 XML 输出后的停止行为仍需进一步优化
