# SFT 阶段说明文档

## 1. 文档范围

本文档只描述本项目的 SFT阶段，包括：

1. SFT 的目标
2. SFT 的输入与输出
3. 模型与训练方法
4. 训练参数设置
5. 训练产物说明
6. 训练结果与解释


## 2. SFT 的目标

本阶段的目标是基于 `Qwen/Qwen2.5-0.5B` 构建一个面向数学推理任务的监督微调模型，使模型具备以下能力：

1. 更稳定地遵循数学解题型指令
2. 更一致地输出带有推理过程的答案
3. 更稳定地给出最终答案
4. 为后续 RL 或其他后训练阶段提供一个较好的 cold-start 模型


## 3. SFT 的基本方法

本项目的 SFT 采用了 PEFT 的 LoRA 微调路线，而不是全参数微调。

### 3.1 训练框架

训练框架如下：

1. `transformers`
2. `peft`

### 3.2 微调方式

采用 `LoRA`（Low-Rank Adaptation）进行参数高效微调。

这样做的原因是：

1. 显著降低显存占用
2. 训练成本更低
3. 更适合课程项目中的多轮实验
4. 便于后续保存 adapter 和导出 merged model

### 3.3 基础模型

本阶段使用的基础模型为：

1. `Qwen/Qwen2.5-0.5B`

选择该模型的原因主要是：

1. 模型规模小，训练速度快
2. 适合单卡 GPU 环境
3. 便于后续开展 SFT 与 RL 的对比实验

## 4. SFT 输入形式

SFT 阶段直接使用已经处理完成的训练集与验证集，不在本阶段重新讨论数据构建过程。

训练脚本使用的是对话式监督样本，其核心字段包括：

1. `instruction`
2. `input`
3. `output`
4. 可选 `system`

在训练时，脚本会将样本组织为如下结构：

1. system message
2. user message
3. assistant message

其中：

1. `instruction + input` 构成用户输入
2. `output` 构成监督目标
3. 如果样本中没有 `system`，则使用默认 system prompt

默认 system prompt 为：

1. 要求模型认真进行数学推理
2. 要求逐步作答
3. 要求在最后使用统一的最终答案格式

## 5. Tokenization 与监督信号设计

### 5.1 Tokenization 方式

训练脚本优先使用 Qwen tokenizer 自带的 chat template，将样本编码为标准对话格式。

如果 tokenizer 不提供 chat template，则退化为显式拼接：

1. system 段
2. user 段
3. assistant 段

### 5.2 截断长度

本次训练的最大序列长度为：

1. `cutoff_len = 2048`

这意味着单条样本在 tokenization 后最多保留 2048 个 token。

### 5.3 Loss 计算方式

具体做法是：

1. 对 system + user 部分对应的 label 全部置为 `-100`
2. 仅对 assistant 响应部分保留真实 token label

这样可以确保训练目标是：

1. 学会生成期望的回答
2. 而不是“复述提示词”

## 6. LoRA 训练方法

本阶段采用的是 `PEFT LoRA`。

### 6.1 LoRA 配置

本次训练实际使用的 LoRA 参数如下：

1. `r = 32`
2. `lora_alpha = 64`
3. `lora_dropout = 0.05`
4. `bias = none`
5. `task_type = CAUSAL_LM`

### 6.2 LoRA 注入模块

本次训练对以下模块注入 LoRA：

1. `q_proj`
2. `k_proj`
3. `v_proj`
4. `o_proj`
5. `gate_proj`
6. `up_proj`
7. `down_proj`

这表示 LoRA 不仅作用于注意力投影层，也作用于 MLP 相关投影层，从而提高模型对任务的适应能力。

## 7. 实际训练参数

根据训练输出目录中的 `train_config.json`，本次 SFT 训练的关键参数如下。

### 7.1 模型与输出路径

1. 基础模型：`Qwen/Qwen2.5-0.5B`
2. LoRA 输出目录：`outputs/sft/qwen25_0p5b_math_lora`
3. 合并后模型目录：`outputs/sft/qwen25_0p5b_math_merged`

### 7.2 训练超参数

1. `num_train_epochs = 3.0`
2. `learning_rate = 2e-4`
3. `per_device_train_batch_size = 4`
4. `gradient_accumulation_steps = 4`
5. `per_device_eval_batch_size = 2`
6. `warmup_ratio = 0.03`
7. `weight_decay = 0.0`
8. `max_grad_norm = 1.0`
9. `save_steps = 200`
10. `eval_steps = 200`
11. `save_total_limit = 2`
12. `seed = 42`

### 7.3 精度设置

1. `fp16 = true`
2. `bf16 = false`

因此本次训练是以 FP16 混合精度完成的。

### 7.4 有效批大小

1. `per_device_train_batch_size = 4`
2. `gradient_accumulation_steps = 4`

所以单卡下的有效 batch size 为：`4 x 4 = 16`

## 8. 训练过程与检查点

根据 `trainer_state.json`，本次训练的整体信息如下：

1. 总训练步数：`18540`
2. 最终 global step：`18540`
3. 训练轮数：`3.0`
4. checkpoint 保存间隔：`200 steps`

## 9. 训练结果

根据训练输出文件：

1. `train_results.json`
2. `eval_results.json`

本次 SFT 的主要结果如下。

### 9.1 训练指标

1. `train_loss = 0.6993799155584045`
2. `train_runtime = 66507.5612 s`
3. `train_samples_per_second = 4.46`
4. `train_steps_per_second = 0.279`
5. `total_flos = 1.0264775075427779e+18`

### 9.2 验证指标

1. `eval_loss = 0.6896898746490479`
2. `eval_runtime = 124.0416 s`
3. `eval_samples_per_second = 16.261`
4. `eval_steps_per_second = 8.134`

## 10. 结果解释

从当前结果看，可以得到以下几点结论。

### 10.1 训练已经完成到预设轮数

训练已经完整运行到：

1. `3.0 epochs`
2. `18540 steps`

因此这不是中途终止或仅完成部分训练的结果。

### 10.2 训练损失与验证损失接近

当前：

1. `train_loss ≈ 0.6994`
2. `eval_loss ≈ 0.6897`

这说明：

1. 训练集与验证集上的损失水平接近
2. 当前没有从这两个指标上看到非常明显的过拟合信号

### 10.3 当前 SFT 输出的直接意义

本阶段的产物主要是：

1. 一个数学推理风格更稳定的 LoRA adapter
2. 一个可直接加载的 merged model
3. 一套可重复复现实验的训练配置与结果记录

这为后续工作提供了：

1. 基线 SFT 模型
2. 后续 RL 训练的初始模型
3. 与 base model 做对比评估的实验对象


## 11. 本阶段总结

本项目的 SFT 阶段已经完成了以下工作：

1. 基于 `Qwen/Qwen2.5-0.5B` 完成数学推理 LoRA 微调
2. 采用 `PEFT` 的参数高效训练方式
3. 使用对话模板和 assistant-only loss 进行监督微调
4. 完成 3 个 epoch、18540 步训练
5. 得到最终 LoRA adapter 与 merged model

