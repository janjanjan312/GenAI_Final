# GSM8K XML 1024 Tokens 实验分析

## 目标

这份文档总结当前这轮 `GSM8K xml + 自定义评估器` 实验。配置如下：

- `max_new_tokens=1024`
- `limit=300`
- `Base` 与 `SFT` 使用 greedy decoding
- XML-style prompting + 项目自定义答案提取与错误归因

本文仅基于这轮 `1024` tokens 实验的结果展开分析。

## 结果范围

已完成的运行：

- `Base`：已完成
- `SFT`：已完成
- `GRPO`：已完成

## 实验设置

- 数据集：`gsm8k` test parquet
- Prompt 风格：`xml`
- 样本数：`300`
- 生成长度：`1024`
- 设备：`mps`
- 本文比较口径：greedy decoding

对应结果文件：

- `evaluation/results/base_gsm8k_xml_1024_300.json`
- `evaluation/results/sft_gsm8k_xml_1024_300.json`
- `evaluation/results/grpo_gsm8k_xml_1024_300_greedy.json`

## 结果

### 指标定义

- `accuracy`：宽松正确率。
  评估器会先尝试提取显式最终答案。
  如果没有，再回退到较宽松的启发式提取，例如 `boxed`、`Final answer`、最后一行或最后一个数字，然后与 gold answer 做等价比较。
- `strict_accuracy`：严格正确率。
  只有当模型明确给出可识别的最终答案时才计为有效预测，再与 gold answer 做等价比较。
  它更强调答案是否被清晰、规范地落出来。
- `content_accuracy`：内容正确率。
  若 `accuracy` 已经判对，则记为正确。
  否则只要输出文本中出现了与 gold answer 一致的数字内容，也记为正确。
  它更适合衡量“内容上已经做出来了，但最终答案没有稳定落好”的情况。

### 核心指标

| Model | accuracy | strict_acc | content_acc | format_cond_acc | trunc_rate |
|---|---:|---:|---:|---:|---:|
| Base | 14.00% | 1.67% | 26.67% | 9.80% | 81.00% |
| SFT | 14.67% | 8.00% | 47.67% | 32.43% | 90.00% |
| GRPO | 19.33% | 19.33% | 31.33% | 19.59% | 91.00% |

### 输出质量指标

说明：这组表头较长。
若导出到 PDF，可将其理解为：

- `avg_word_count`：平均词数
- `avg_reasoning_steps`：平均推理步数
- `repetition_rate`：重复率
- `avg_repetition_ratio`：平均重复占比
- `degeneration_rate`：退化率

| Model | avg_word_count | avg_reasoning_steps | repetition_rate | avg_repetition_ratio | degeneration_rate |
|---|---:|---:|---:|---:|---:|
| Base | 513.9 | 4.41 | 83.33% | 0.7384 | 83.67% |
| SFT | 633.0 | 4.80 | 70.67% | 0.2335 | 70.67% |
| GRPO | 452.7 | 2.49 | 85.00% | 0.4395 | 85.00% |

### 错误分类

说明：错误类型包括：

- `degeneration`
- `truncation`
- `computation_error`
- `format_only_error`
- `reasoning_error`

| Model | correct | degeneration | truncation | computation_error | format_only_error | reasoning_error |
|---|---:|---:|---:|---:|---:|---:|
| Base | 42 | 218 | 9 | 27 | 2 | 2 |
| SFT | 44 | 185 | 67 | 3 | 1 | 0 |
| GRPO | 58 | 206 | 36 | 0 | 0 | 0 |

## ARC-Easy: general capability check

为了补充 `Topic 1` 里“RL 是否提升数学能力但伤害 general capability”这个问题，这里再加入一个轻量的通用 benchmark 检查。

`ARC-Easy` 使用 `lm-eval-harness`，`limit=300`。其中：

- `Base` 与 `GRPO` 使用完整模型路径
- `SFT` 使用 `Base + LoRA adapter` 口径跑通

| Model | acc | acc_norm |
|---|---:|---:|
| Base | 59.00% | 56.67% |
| SFT | 56.00% | 56.00% |
| GRPO | 55.67% | 55.67% |

这组结果说明：

- `Base` 仍然最好
- `SFT` 与 `GRPO` 都略低于 `Base`
- 但三者差距不大，更像是轻微回落，而不是明显的 general capability 崩塌

因此，当前更稳妥的结论是：

- `SFT` 和 `GRPO` 没有在 `ARC-Easy` 上带来通用能力提升
- 它们的增益更集中在数学或协议相关行为上
- 目前也没有证据表明它们造成了特别严重的通用能力损伤

## 主要发现

### 1. `GRPO` 在 top-line `accuracy` 和 `strict_accuracy` 上最好

如果先看最直接的结果排序：

- `accuracy`: `GRPO (19.33%) > SFT (14.67%) > Base (14.00%)`
- `strict_accuracy`: `GRPO (19.33%) > SFT (8.00%) > Base (1.67%)`

这说明在当前这套 `GSM8K XML 1024` 评测下，`GRPO` 最擅长把答案落成评估器可以直接识别的形式。

### 2. `SFT` 的 `content_accuracy` 最高，说明内容层面仍然最强

如果看 `content_accuracy`，排序会变成：

- `SFT`: `47.67%`
- `GRPO`: `31.33%`
- `Base`: `26.67%`

这说明 `SFT` 仍然最容易在输出中生成正确答案内容，即使这些内容并不总能稳定地落成最终得分。

因此，`GRPO` 的领先主要体现在最终答案的可提取性和协议化表达上，而 `SFT` 的优势更多体现在内容本身。

这里需要补充一点：`content_accuracy` 更偏向“正文里有没有出现正确答案”，而不是“最终答案格式是否规范”。

评估器会先看 `accuracy` 是否正确；若没有判对，再检查整段输出中是否出现过与 gold answer 一致的数字内容。

这正好解释了为什么 `GRPO` 的 `strict_accuracy` 很高，但 `content_accuracy` 仍低于 `SFT`：

- `GRPO` 更擅长把答案按 `<answer>` / `\boxed{...}` 的结构化形式交出来
- `SFT` 更擅长在长推理正文里写出正确答案内容，即使最终落点不够稳定

因此，更准确的理解是：`GRPO` 更像是在提升“按协议交卷”的能力，而 `SFT` 更像是在提升“把题目内容做出来”的能力。

### 3. `GRPO` 几乎完全学会了这套 XML / boxed 答案协议

`GRPO` 的格式相关指标非常高：

- `strict_accuracy = 19.33%`
- `explicit_final_answer_rate = 98.67%`
- `has_think_rate = 98.67%`
- `has_answer_rate = 98.67%`
- `has_single_boxed_rate = 51.00%`

同时，`GRPO` 的答案提取来源几乎完全来自结构化答案：

- `boxed`: `263`
- `answer_tag`: `33`
- `last_number`: `4`

这说明 RL 训练显著强化了模型对 `<think> / <answer> / \boxed{...}` 这一类协议的遵循。

### 4. `SFT` 和 `GRPO` 分别代表两种不同的优势

从三模型对比来看：

- `SFT`：内容更强，`content_accuracy` 最高，重复率和退化率也最低
- `GRPO`：答案落地更强，`accuracy` 与 `strict_accuracy` 最高
- `Base`：在这轮实验里整体最弱，尤其在严格答案落地上明显落后

这意味着 `SFT` 与 `GRPO` 的改进方向并不相同：

- `SFT` 更像是在提升“题做出来”的概率
- `GRPO` 更像是在提升“把答案按协议交出来”的概率

### 4.1 一个轻量验证：`GRPO` 的提升主要来自格式覆盖率，而不是更高的格式内正确率

如果把“会不会按协议交卷”和“交卷后答得准不准”拆开看，这个结论会更清楚。

| Model | explicit answer count | strict correct | format_conditioned_accuracy |
|---|---:|---:|---:|
| Base | 51 | 5 | 9.80% |
| SFT | 74 | 24 | 32.43% |
| GRPO | 296 | 58 | 19.59% |

这张表说明：

- `GRPO` 几乎把所有样本都变成了“有显式终答案”的样本
- 但在这些已经按格式交卷的样本里，`SFT` 的正确率其实更高

因此，一个更直接的解释是：

- `GRPO` 的 `strict_accuracy` 提升，主要来自格式覆盖率被大幅拉高
- 它并不意味着 `GRPO` 在“已经形成结构化终答案之后”的内容正确率也全面超过了 `SFT`

### 5. 三个模型都仍然受到严重 truncation 影响

即使 `max_new_tokens=1024`，三个模型的截断率仍然都很高：

- `Base`: `81%`
- `SFT`: `90%`
- `GRPO`: `91%`

这说明当前评测仍然同时受到以下因素影响：

- 数学内容是否正确
- 最终答案是否及时落地
- 输出是否被截断
- 格式是否可提取

因此，这个 benchmark 不能被简单理解为“纯数学能力排行”。

### 6. `GRPO` 并没有在推理质量上全面胜过 `SFT`

虽然 `GRPO` 的 top-line 得分最好，但它并不在所有行为指标上最强。

例如：

- `GRPO` 的 `content_accuracy` 低于 `SFT`
- `GRPO` 的 `repetition_rate` 与 `degeneration_rate` 都高于 `SFT`
- `GRPO` 的平均推理步数更少，输出更短

这更像是在说明：

- `GRPO` 学会了更高强度的协议化收尾
- 但 `SFT` 仍然保留了更丰富的解题内容

## 解释

基于这轮 `1024`-token `GSM8K xml` 实验，可以得到如下总结：

- `GRPO` 在 `accuracy` 与 `strict_accuracy` 上最好
- `SFT` 在 `content_accuracy` 上最好
- `GRPO` 的优势主要来自更强的答案协议遵循与可提取性
- `SFT` 的优势主要来自更强的内容生成能力

从这轮实验本身出发，更自然的结论是：

`GRPO` 最擅长“按当前协议把答案交出来”，`SFT` 最擅长“把题目内容做出来”。

因此，这套结果不应被读成单一维度上的绝对强弱，而更适合被解释为三者在：

- 内容生成
- 答案落地
- 格式遵循

这三个维度上的不同取舍。

## 这个 Benchmark 实际在测什么

这个 benchmark 是有价值的，但它的含义需要说清楚。

这里的分数混合了多种因素：

- 数学正确性
- XML 风格答案格式
- 最终答案是否足够早地落下来
- 固定 token budget 下的 truncation
- 自定义答案提取逻辑

因此，更准确的理解应该是：

- 它是一个针对项目 XML 协议下生成行为的诊断型 benchmark
- 它不是一个纯粹、独立的数学能力分数

## 训练数据生成方式可能带来的影响

从 `GenAI_final_project_data` 里的可见数据生成逻辑看，SFT 训练数据的构造方式很可能会让模型在当前这类 `GSM8K` 自定义评测里表现更好。

这里的关键不是 XML 标签本身，而是训练数据被系统性地推向了以下风格：

- 长推理
- tutor-style 解题过程
- 明确的最终答案落地
- `\boxed{...}` 形式的终答案表达

`phase_a_manifest.json` 里明确要求：

- `require_any_r1_solution: true`
- `min_r1_solution_chars: 200`

这说明进入下游训练管道的样本，至少需要带有一条足够长的 `r1_solution`。这种过滤会天然偏向：

- 更完整的逐步推理
- 更长的数学讲解
- 较少的短答案、短收尾样本

此外，`StageA.ipynb` 里导出训练样本时，会对解答做额外的终答案补全：

- 如果解答里没有 `\boxed{...}`，就在末尾追加 `\boxed{final_answer_norm}`
- `alpaca output` 还会额外保留一行 `Final answer: {final_answer_norm}`

这意味着训练目标会反复强化两类行为：

- 把较长的数学推理完整写出来
- 在结尾显式落一个可提取的标准答案

这和当前 `SFT` 在 `GSM8K XML 1024` 里的表现是一致的：

- `content_accuracy` 明显高于 `Base`
- `strict_accuracy` 明显高于 `Base`
- 输出更长
- 但也更容易被截断

因此，一个合理的解释是：

SFT 数据生成方式确实可能帮助模型在 `GSM8K` 这类较基础的数学题上写出更多正确内容，并更频繁地显式给出终答案。

这会直接抬高：

- `content_accuracy`
- `strict_accuracy`

不过，这个现象不应被解释成“模型学会了 XML 协议本身”。

原因是当前结果里：

- `has_think_rate` 仍然极低
- `has_answer_rate` 基本为 `0`
- `exact_xml_boxed_rate` 仍然为 `0`

这说明 SFT 学到的更像是：

- 数学讲解式回答
- `boxed` / `Final answer` 风格的终答案表达

而不是严格遵循 `<think> ... </think><answer> ... </answer>` 的 XML 模板。

所以，更准确的说法是：

训练数据生成方式很可能提升了 SFT 在当前自定义 `GSM8K` 评测中的内容表现和答案落地表现，但这种提升主要来自长推理与终答案风格，而不是来自对 XML 标签协议的直接对齐。

## Failure Modes

结合三模型输出和错误归因，这轮实验里最主要的 failure modes 可以概括为四类：

1. `degeneration` 仍然是最大头。
   三个模型的退化率都很高，说明很多错误并不是单纯“不会做”，而是输出后段进入重复、崩坏或无效续写。
2. `truncation` 依然严重。
   即使 `max_new_tokens=1024`，三个模型的截断率仍在 `81%` 到 `91%` 之间，说明 token budget 仍在显著影响最终得分。
3. `protocol adherence` 与 `content correctness` 明显分离。
   `GRPO` 更强的是 `<answer>` / `\boxed{...}` 这类协议化收尾，`SFT` 更强的是正文里写出正确内容。
4. `format drift` 主要发生在 `Base` 和 `SFT` 上。
   它们常常能写出某种终答案风格，但并不稳定遵循当前 XML 协议；`GRPO` 则更容易出现“格式对了、内容未必对”的现象。

## Limitations

这轮结果已经能较好说明 `Base / SFT / GRPO` 在数学生成行为上的差异，但仍有几项限制需要明确：

- 当前虽然补充了 `ARC-Easy` 这一项 general benchmark，但 general capability 部分仍只基于单个、相对轻量的任务，证据还不算很宽。
- `GSM8K XML` 分数混合了内容正确性、答案落地、格式遵循和 truncation 等多个因素，因此不应被直接解释为纯数学能力排行。
- 目前还没有完成 `data filtering sensitivity` 的对照实验，因此关于 SFT 数据筛选策略的判断仍主要是机制性解释，而不是严格 ablation 结论。

## 结论

对于这轮 `1024` 实验，最简洁的结论是：

1. `GRPO` 在 `accuracy` 和 `strict_accuracy` 上最好，说明它最擅长按当前协议稳定交出可提取答案。
2. `SFT` 在 `content_accuracy` 上最好，说明它在内容层面仍然最容易生成正确答案。
3. 在 `ARC-Easy` 上，`Base` 仍然略好于 `SFT` 和 `GRPO`，说明 post-training 没有带来通用能力提升，但目前也只看到轻微回落。
4. 三个模型都仍然受到严重 truncation 影响，因此当前结果更适合解释为“生成行为与答案落地”的对比，而不是纯数学能力排行。
