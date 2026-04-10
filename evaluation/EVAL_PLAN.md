# Evaluation Plan — AIMS5740 Final Project Topic 1

## Overview

评估 Data Selection + SFT + GRPO-RL 对 Qwen2.5-0.5B 数学推理能力的影响。
对比三个阶段：**Base → SFT → GRPO**。

当前评估定位更新如下：

- **主 benchmark**：`lm-eval` 的 `minerva_math`
- **补充分析**：`GSM8K xml + 自定义评估器`

原因：

- `minerva_math` 是更标准的数学 benchmark，结论更适合用于报告中的主结果。
- `GSM8K xml` 评测混合了数学能力、格式遵循、截断和自定义答案提取，更适合做 error analysis，而不是单独作为“真实数学能力”的主分数。

## 1. Models

| 阶段 | 路径 |
|------|------|
| Base | `models/Qwen2.5-0.5B` |
| SFT | `sft/qwen25_0p5b_math_merged` |
| GRPO | `grpo_trl_model` |

## 2. Evaluation Structure

两部分评估，分工明确，但主次有区别：

| | Part 1: 自定义评估器 | Part 2: lm-eval-harness |
|--|---------------------|------------------------|
| 评估方式 | 生成式（模型实际输出） | 标准化（logprob / few-shot） |
| 速度 | 慢 | 快 |
| 用途 | **补充诊断**：failure modes、format drift、error 分类 | **主结果**：标准化 benchmark 对比 |
| Benchmark | GSM8K (xml prompt) | minerva_math + MMLU + ARC-Easy |

### Part 1: GSM8K 补充分析（自定义评估器）

100 条 × 3 模型 = **3 次评估**。GRPO 额外做 pass@k（3 次采样）。

定位：

- 用于分析格式漂移、截断、退化、答案提取失败等现象
- 不作为数学能力主 benchmark
- 其中 `accuracy / strict_accuracy / content_accuracy` 主要用于辅助解释模型行为
- 尤其是 `content_accuracy - accuracy` 的差距，可用来量化“会做但没被成功抽取/成功落地”的部分

采集指标：

```
正确性: accuracy / strict_accuracy / content_accuracy / pass@1 / pass@3
格式:   has_think / has_answer / exact_xml_boxed (检测 format drift)
质量:   avg_word_count / reasoning_steps / repetition / degeneration
错误:   truncation / format_only / computation / reasoning / comprehension
```

产出：`gsm8k_comparison.md` — 三阶段 side-by-side 对比表。

### Part 2: lm-eval 标准化评测（主 benchmark）

200 条/task × 3 模型 × 3 tasks = **9 次评估**。

定位：

- `minerva_math` 是本项目**主数学 benchmark**
- `MMLU` / `ARC-Easy` 作为补充，用于回答 RL 是否伤害 general QA / science reasoning
- 若时间有限，优先顺序为：`minerva_math` > `MMLU` > `ARC-Easy`

| Task | 评估方式 | 目的 |
|------|---------|------|
| `minerva_math` | 4-shot 生成 | **主结果**：竞赛数学能力（最接近真实数学解题能力） |
| `mmlu` | 5-shot 选择题 | 通用 QA，回答"RL 是否伤害 general QA" |
| `arc_easy` | 0-shot 选择题 | 科学推理，第二参照 |

## 3. How to Run

```bash
cd /Users/chenshiliang/Desktop/GenAI_Final

# 完整评估
bash evaluation/run_full_eval.sh

# 优先跑主 benchmark（推荐）
source .venv/bin/activate
python evaluation/run_lm_eval.py \
  --model models/Qwen2.5-0.5B \
  --tasks minerva_math \
  --num-fewshot-override 4 \
  --limit 50 \
  --device mps \
  --dtype float16 \
  --batch-size 1 \
  --gen-kwargs max_gen_toks=1024 \
  --output-dir evaluation/results/lm_eval/base_minerva_math \
  --label base_minerva_math

# 快速验证（30 条/task）
QUICK=1 bash evaluation/run_full_eval.sh

# 只跑自定义评估器，不跑 lm-eval
SKIP_LM_EVAL=1 bash evaluation/run_full_eval.sh

# 只跑特定阶段
STAGES="base grpo" bash evaluation/run_full_eval.sh
```

## 4. Crash Recovery

| 组件 | 保护机制 |
|------|---------|
| 自定义评估器 | 每 20 条 checkpoint + `--resume` 自动续跑 |
| lm-eval | 每个 task 独立运行，崩了只丢当前 task |
| 整体脚本 | 单个失败不阻塞后续，重跑自动跳过已完成的 |

## 5. Output Files

```
evaluation/results/
├── {base,sft,grpo}_gsm8k_xml.json    Part 1: 自定义评估器
├── gsm8k_comparison.{json,md}         Part 1: 对比表
└── lm_eval/                           Part 2: lm-eval
    ├── base_minerva_math/
    │   ├── results_*.json
    │   ├── samples_*.jsonl
    │   └── base_minerva_math_summary.json
    ├── sft_minerva_math/
    │   └── sft_minerva_math_summary.json
    ├── grpo_minerva_math/
    │   └── grpo_minerva_math_summary.json
    ├── base_mmlu/
    │   └── base_mmlu_summary.json
    ├── sft_mmlu/
    │   └── sft_mmlu_summary.json
    ├── grpo_mmlu/
    │   └── grpo_mmlu_summary.json
    ├── base_arc_easy/
    │   └── base_arc_easy_summary.json
    ├── sft_arc_easy/
    │   └── sft_arc_easy_summary.json
    └── grpo_arc_easy/
        └── grpo_arc_easy_summary.json
```

说明：

- `results_*.json` 是 `lm-eval` 原始结果文件
- `samples_*.jsonl` 是开启 `--log_samples` 后保存的逐题输出，可用于后续检查 truncation / answer format / failure cases
- `*_summary.json` 是 `evaluation/run_lm_eval.py` 汇总后的项目侧结果文件，更适合直接用于报告整理

## 6. Mapping to PDF Requirements

| PDF 要求 | 对应评估 |
|---------|---------|
| Benchmark: MATH / 数学主结果 | **Part 2 lm-eval `minerva_math`** |
| Benchmark: GSM8K / 补充解释 | Part 1 自定义评估器 (100 条, xml prompt) |
| RL 是否伤害 general QA | Part 2 lm-eval `mmlu` + `arc_easy` |
| Format drift | Part 1 `exact_xml_boxed_rate` 三阶段对比 |
| Reward hacking | Part 1 `repetition_rate` / `degeneration_rate` |
| Failure modes | Part 1 `error_classification` + `content_accuracy` vs `accuracy` |

## 7. Estimated Runtime (MPS, Qwen2.5-0.5B)

| Part | 内容 | 估计时间 |
|------|------|---------|
| 1 | GSM8K × 3 models (100 条, max_tokens=256 + pass@k) | ~30-50min |
| 1b | 对比表生成 | <1min |
| 2a | `minerva_math` 主 benchmark（推荐设置：50 条/subtask, 7 subtasks, 4-shot, 1024 gen tokens） | ~25-50min / 模型 |
| 2b | `MMLU`（若跑全量 grouped task，成本很高） | 可能数小时 / 模型 |
| 2c | `ARC-Easy`（补充参照） | ~10-20min / 模型 |
| **Recommended total** | `minerva_math` × 3 models + GSM8K 分析 | **~2-4h** |
| **Full total** | 再加 `MMLU` + `ARC-Easy` | **显著更久，取决于 MMLU 配置** |

> 注：
> - `GSM8K xml` 仍然保留，但其结果应解释为“生成行为诊断”而非主 benchmark。
> - `minerva_math` 是当前版本中更应优先汇报的数学能力结果。
> - 若 `minerva_math` 使用生成式评测时仍观察到明显 truncation，可辅以 `log_samples` 抽样检查，而不应直接用 `MMLU` 数学子集替代其主 benchmark 地位。
> - 当前更推荐的小规模主 benchmark 配置是：`limit=50` / subtask、`4-shot`、`max_gen_toks=1024`、`batch_size=1`。
