# Evaluation Report: Compact Version

## Scope and Setup

This section gives a brief overview of three pieces of evidence from the current evaluation pipeline:
- A comparison of `Base`, `SFT`, and `GRPO` in `GSM8K XML 1024`
- a examination of `ARC-Easy`'s general capabilities
- an ablation of Stage A data filtering on `gsm8k` and `deepmath_smoke`

The main GSM8K settings are greedy decoding, `300` examples, `max_new_tokens=1024`, XML-style prompting, and a custom evaluator that examines for math correctness, format adherence, and reasoning quality.

For GSM8K, We mainly look at three metrics: Accuracy, which allows fallback extraction, Strict Accuracy, which only counts answers with an explicit structured final answer, and Content Accuracy, which checks when the correct answer appears anywhere in the output.


## Main Results on GSM8K

| Model | accuracy | strict_acc | content_acc | fmt_cond_acc | trunc_rate |
|---|---:|---:|---:|---:|---:|
| Base | 14.00% | 1.67% | 26.67% | 9.80% | 81.00% |
| SFT | 14.67% | 8.00% | 47.67% | 32.43% | 90.00% |
| GRPO | 19.33% | 19.33% | 31.33% | 19.59% | 91.00% |

The main pattern is a difference between the quality of the content and the answer landing:
- `GRPO` is best for "accuracy" and "strict_accuracy" on the top line.
- `SFT` is best at "content_accuracy".
- Overall, `Base` is the weakest, especially when it comes to giving explicit final answers.

This means that `GRPO` is best at turning generations into final answers that can be extracted, while `SFT` is best at making sure that the output inludes the correct math content.


## Why GRPO Wins the Headline Metric

The primary benefit of `GRPO` seems to be protocol adherence rather than a distinct improvement in reasoning content. In the analysis that follows:
- `strict_accuracy = 19.33%`
- `explicit_final_answer_rate = 98.67%`
- `has_think_rate = 98.67%`
- `has_answer_rate = 98.67%`
- `has_single_boxed_rate = 51.00%`

Its sources for extraction are mostly organized:

- `boxed = 263`
- `answer_tag = 33`
- `last_number = 4`

This shows that RL training makes it much easier to follow the `<think> / <answer> / boxed` protocol.

When we look at answer coverage and accuracy in formatted outputs separately , the same point becomes clearer:

| Model | explicit answer count | strict correct | fmt_cond_acc |
|---|---:|---:|---:|
| Base | 51 | 5 | 9.80% |
| SFT | 74 | 24 | 32.43% |
| GRPO | 296 | 58 | 19.59% |

`GRPO` wins because it generates clear answers in almost every case. But when it comes to outputs that already have clear answers, `SFT` is actually more accurate. So, the `GRPO` gain could be better understood as a gain in the number of answer formats, not as a general improvement in the math content.

## Reasoning Quality and Failure Modes

| Model | avg_words | avg_steps | repetition | degeneration |
|---|---:|---:|---:|---:|
| Base | 513.9 | 4.41 | 83.33% | 83.67% |
| SFT | 633.0 | 4.80 | 70.67% | 70.67% |
| GRPO | 452.7 | 2.49 | 85.00% | 85.00% |

Behaviorally, `SFT` is better than `GRPO`: it gives longer outputs, more steps of reasoning, and less repetition and degeneration. On the other hand, `GRPO` gets the best headline metric even though it doesn't do better on these quality indicators.

The most common failure modes are still:

- degeneration
- truncation
- distinction between content accuracy and protocol compliance

Even with `1024` output tokens, truncation is still very high, so this benchmark mixes mathematical correctness with timing, formatting, and extractability of the answer.


## General Capability Check

| Model | acc | acc_norm |
|---|---:|---:|
| Base | 59.00% | 56.67% |
| SFT | 56.00% | 56.00% |
| GRPO | 55.67% | 55.67% |

On `ARC-Easy`, `Base` is still the strongest, and `SFT` and `GRPO` are a little weaker. The gap is small, though. The careful reading is that post-training doesn't make things better in general, but it also doesn't cause a catastrophic forgetting. The gains that were seen are mostly in math-task behavior and answer-protocol behavior.

## Data Filtering Ablation

### Data-side effect

| variant | kept_examples | delta |
|---|---:|---:|
| baseline | 100891 | +0 |
| wo difficulty band | 100895 | +4 |
| wo question min length | 100964 | +73 |
| wo question max length | 100893 | +2 |
| wo answer min length | 100893 | +2 |
| wo answer max length | 100891 | +0 |
| wo ambiguity regex | 102941 | +2050 |
| wo r1 requirement | 100891 | +0 |

Removing the ambiguity regex keeps more data than any other ablation, but this extra volume doesn't make the quality downstream better.

### Downstream effect

| Variant | GSM8K acc | GSM8K strict | DeepMath acc | Main takeaway |
|---|---:|---:|---:|---|
| baseline | 16.67% | 10.00% | 6.67% | reference |
| wo question max length | 23.33% | 16.67% | 10.00% | best overall ablation |
| wo ambiguity regex | 16.67% | 6.67% | 3.33% | more data, not better |
| wo answer max length | 6.67% | 3.33% | 6.67% | harmful |
| wo r1 requirement | 0.00% | 0.00% | 3.33% | worst case |


The most solid conclusions are four:

1. The need for `r1` supervision is important. Removing it doesn't change the size of the dataset as much, but it does drop the accuracy of `GSM8K` from `16.67%` to `0.00%`.
2. The quality of the data is more important than the number of raw data points. Regex adds `2050` examples to remove the ambiguity, but it doesn't improve the performance.
3. The answer max-length filter should be kept. Filtering it out lowers the accuracy of `GSM8K` from `16.67%` to `6.67%` and raises the degeneration from `66.67%` to `86.67%`.
4. The question max-length filter is the clearest choise for revision. The best results on both `GSM8K` and `DeepMath` come from taking it out.


## Interpretation and Conclusion

The current benchmark should also be seen as a way to test how well math-oriented generation follow an XML-style answer protocol, not as a pure measure of mathmatical capbility. The score is made up of:

- content correctness
- answer format
- final-answer landing
- truncation
- custom extraction behavior

The main points are:

1. The fact that `GRPO` is best at answer landing and following protocols explains why it is ahead in `accuracy` and `strict_accuracy`.
2. `SFT` excels at genrating content, which explains why it is the best at `content_accuracy` and looks better on reasoning-quality metrics.
3. General ability does not get better on `ARC-Easy`, but the drop is mild rather than catastrophic.
4. In the data ablation, filtering quality is more important than data volume. The `r1` requirement and answer max-length filter seem to be useful, the ambiguity regex serves a quality-control role, and the question max-length cap is the clearest rule for revision.

In conclusion: `SFT` improves content, `GRPO` improves answer formatting, and better filtering is more important than just getting more training data.
