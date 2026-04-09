# DeepMath Phase A 上的 TRL GRPO 训练说明

这套代码是把原来依赖 `verl` 的 GRPO 训练流程，改写成 Hugging Face `trl` 版本。

本目录包含 4 个核心文件：

- `prepare_deepmath_grpo_dataset.py`：把原始 parquet 转成 `trl.GRPOTrainer` 可直接读取的 parquet。
- `reward_math_format_accuracy.py`：格式奖励 + 答案奖励。
- `train_grpo_trl.py`：`trl.GRPOTrainer` 训练入口。
- `run_grpo_qwen25_math.sh`：一键准备数据并通过 `accelerate launch` 启动训练。

## 1. 这次改写的重点

原来的 `verl` 版本有两个实际问题：

- 框架本身在你环境里部署不顺。
- 脚本里路径是硬编码的，而且和当前项目真实目录不一致，直接运行就容易失败。

这版 `trl` 改写做了两件事：

- 训练框架切到 `trl.GRPOTrainer`
- 所有路径都改成基于脚本位置自动推导

所以你现在不需要再维护 `verl` 的 `reward_model` / `custom_reward_function` 那套配置结构。

## 2. 数据格式

`trl` 的 `GRPOTrainer` 只要求数据里至少有 `prompt` 列。  
如果奖励函数还需要标准答案等额外列，也可以直接保留在 parquet 里传进 reward function。

这版输出字段为：

- `prompt`
- `ground_truth`
- `question`
- `topic`
- `difficulty`
- `answer_format`
- `data_source`
- `row_id`

其中：

- `prompt` 是 chat messages 列表
- `ground_truth` 是标准答案
- 其余字段主要用于 reward function 和日志

## 3. 奖励设计

总奖励仍然保持原逻辑：

```text
total_reward = format_reward + answer_reward
```

权重：

- `format_reward = 0.2`
- `answer_reward = 0.8`

### 3.1 格式奖励

要求输出满足：

```text
<think>
...
</think>
<answer>
\boxed{...}
</answer>
```

具体检查：

- 有 `<think>...</think>`
- 有 `<answer>...</answer>`
- 只出现一个 `\boxed{...}`
- `</answer>` 后没有额外文本

### 3.2 答案奖励

逻辑：

1. 优先提取 `<answer>` 里的内容
2. 如果有 `\boxed{...}`，只取 boxed 里面的文本
3. 与 `ground_truth` 做归一化比较
4. 能被 `sympy` 解析时，额外做数学等价判断

## 4. 依赖

最少需要这些包：

```bash
pip install trl accelerate peft datasets transformers pyarrow sympy
```

如果你想用 vLLM 加速生成：

```bash
pip install "trl[vllm]" accelerate peft datasets transformers pyarrow sympy
```

我本地检查过当前环境，现状是：

- 已有：`transformers`、`datasets`、`sympy`
- 缺少：`trl`、`accelerate`、`peft`

所以在当前机器上，这几项还得先装。

## 5. 启动方式

直接运行：

```bash
cd /Users/obb/Desktop/study/CHUKstudy/AIMS5740/HW/final
bash upload/grpo_trl/run_grpo_qwen25_math.sh
```

脚本默认会：

1. 读取 `final_project/outputs/phase_a_deepmath/deepmath_phase_a.parquet`
2. 自动生成训练/验证 parquet
3. 使用 `upload/qwen25_0p5b_math_merged`
4. 用 `accelerate launch` 启动 `trl` 训练

## 6. 默认参数

默认值基本沿用了你原来 `verl` 脚本的意图：

- `learning_rate=1e-6`
- `num_generations=8`
- `max_completion_length=1536`
- `num_train_epochs=1`
- `beta=0.001`
- `temperature=0.8`
- `top_p=0.95`

另外按 TRL 官方 GRPO 文档，这版训练脚本直接基于：

- `GRPOTrainer` 接收 `prompt` 列和自定义 reward function
- `num_generations`
- `max_completion_length`
- `use_vllm`
- `vllm_mode`

参考文档：

- https://huggingface.co/docs/trl/en/grpo_trainer

## 7. 可覆盖环境变量

例如单卡先跑一个小实验：

```bash
export CUDA_VISIBLE_DEVICES=0
export NUM_PROCESSES=1
export PER_DEVICE_TRAIN_BATCH_SIZE=2
export GRADIENT_ACCUMULATION_STEPS=4
export NUM_GENERATIONS=4
export MAX_SAMPLES=2000
bash upload/grpo_trl/run_grpo_qwen25_math.sh
```

如果想启用 vLLM：

```bash
export USE_VLLM=1
export VLLM_MODE=colocate
export VLLM_GPU_MEMORY_UTILIZATION=0.55
bash upload/grpo_trl/run_grpo_qwen25_math.sh
```

## 8. 一个重要约束

TRL 官方文档说明：`num_generations` 必须整除有效 batch size。  
这里至少要满足每个进程上：

```text
per_device_train_batch_size * gradient_accumulation_steps
```

能被 `num_generations` 整除。

默认值下：

```text
4 * 4 = 16
16 % 8 = 0
```

所以默认配置是成立的。

## 9. 我这次没有做的事

我没有在当前环境把训练真正跑起来，因为本机还没装：

- `trl`
- `accelerate`
- `peft`

代码已经按 TRL 官方当前文档接口重写，但真正开训前，你还需要先把依赖补齐。
