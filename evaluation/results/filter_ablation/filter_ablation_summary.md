# Data Filtering Sensitivity Summary

## Scope

This summary reuses the existing rule-ablation LoRA adapters in `ablation/runs` and combines:
- data-side changes from `ablation/metadata/*.json`
- downstream evaluation outputs under `evaluation/results/filter_ablation`

## Key Findings

- `06_wo_ambiguity_regex` has the largest retained dataset (102941, +2050 vs baseline).
- On `gsm8k`, best `accuracy` is `03_wo_question_max_len` at 23.33% (baseline 16.67%).
- On `gsm8k`, best `strict_accuracy` is `03_wo_question_max_len` at 16.67% (baseline 10.00%).
- On `deepmath_smoke`, best `accuracy` is `03_wo_question_max_len` at 10.00% (baseline 6.67%).
- On `deepmath_smoke`, best `strict_accuracy` is `00_full_all_rules` at 3.33% (baseline 3.33%).

## Data-Side Summary

| variant | removed_rule | kept_examples | kept_delta | overrides |
| --- | --- | --- | --- | --- |
| 00_full_all_rules | baseline | 100891 | +0 | {} |
| 01_wo_difficulty_band | remove difficulty band | 100895 | +4 | {"min_difficulty": -1000000000.0, "max_difficulty": 1000000000.0} |
| 02_wo_question_min_len | remove min question length | 100964 | +73 | {"min_question_chars": 0} |
| 03_wo_question_max_len | remove max question length | 100893 | +2 | {"max_question_chars": 10000000} |
| 04_wo_answer_min_len | remove min answer length | 100893 | +2 | {"min_final_answer_chars": 0} |
| 05_wo_answer_max_len | remove max answer length | 100891 | +0 | {"max_final_answer_chars": 10000000} |
| 06_wo_ambiguity_regex | remove ambiguity regex filter | 102941 | +2050 | {"ambiguity_regexes": []} |
| 07_wo_r1_requirement | remove r1-solution requirement | 100891 | +0 | {"require_any_r1_solution": false} |

## DeepMath Smoke Metrics

| variant | removed_rule | kept | delta_kept | accuracy | strict_acc | content_acc | exact_xml | degeneration | truncation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 00_full_all_rules | baseline | 100891 | +0 | 6.67% | 3.33% | 26.67% | 0.00% | 53.33% | 93.33% |
| 01_wo_difficulty_band | remove difficulty band | 100895 | +4 | 6.67% | 0.00% | 26.67% | 0.00% | 50.00% | 93.33% |
| 02_wo_question_min_len | remove min question length | 100964 | +73 | 3.33% | 0.00% | 20.00% | 0.00% | 56.67% | 93.33% |
| 03_wo_question_max_len | remove max question length | 100893 | +2 | 10.00% | 0.00% | 23.33% | 0.00% | 50.00% | 90.00% |
| 04_wo_answer_min_len | remove min answer length | 100893 | +2 | 6.67% | 0.00% | 23.33% | 0.00% | 76.67% | 93.33% |
| 05_wo_answer_max_len | remove max answer length | 100891 | +0 | 6.67% | 0.00% | 23.33% | 0.00% | 60.00% | 93.33% |
| 06_wo_ambiguity_regex | remove ambiguity regex filter | 102941 | +2050 | 3.33% | 0.00% | 20.00% | 0.00% | 63.33% | 100.00% |
| 07_wo_r1_requirement | remove r1-solution requirement | 100891 | +0 | 3.33% | 0.00% | 23.33% | 0.00% | 66.67% | 86.67% |

### DeepMath Smoke Deltas vs Baseline

| variant | acc_delta | strict_delta | content_delta | xml_delta | degeneration_delta | truncation_delta |
| --- | --- | --- | --- | --- | --- | --- |
| 00_full_all_rules | +0.00 pp | +0.00 pp | +0.00 pp | +0.00 pp | +0.00 pp | +0.00 pp |
| 01_wo_difficulty_band | +0.00 pp | -3.33 pp | +0.00 pp | +0.00 pp | -3.33 pp | +0.00 pp |
| 02_wo_question_min_len | -3.33 pp | -3.33 pp | -6.67 pp | +0.00 pp | +3.34 pp | +0.00 pp |
| 03_wo_question_max_len | +3.33 pp | -3.33 pp | -3.33 pp | +0.00 pp | -3.33 pp | -3.33 pp |
| 04_wo_answer_min_len | +0.00 pp | -3.33 pp | -3.33 pp | +0.00 pp | +23.34 pp | +0.00 pp |
| 05_wo_answer_max_len | +0.00 pp | -3.33 pp | -3.33 pp | +0.00 pp | +6.67 pp | +0.00 pp |
| 06_wo_ambiguity_regex | -3.33 pp | -3.33 pp | -6.67 pp | +0.00 pp | +10.00 pp | +6.67 pp |
| 07_wo_r1_requirement | -3.33 pp | -3.33 pp | -3.33 pp | +0.00 pp | +13.34 pp | -6.66 pp |

## GSM8K Metrics

| variant | removed_rule | kept | delta_kept | accuracy | strict_acc | content_acc | exact_xml | degeneration | truncation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 00_full_all_rules | baseline | 100891 | +0 | 16.67% | 10.00% | 40.00% | 0.00% | 66.67% | 86.67% |
| 01_wo_difficulty_band | remove difficulty band | 100895 | +4 | 13.33% | 3.33% | 36.67% | 0.00% | 70.00% | 83.33% |
| 02_wo_question_min_len | remove min question length | 100964 | +73 | 10.00% | 6.67% | 46.67% | 0.00% | 60.00% | 96.67% |
| 03_wo_question_max_len | remove max question length | 100893 | +2 | 23.33% | 16.67% | 46.67% | 0.00% | 63.33% | 93.33% |
| 04_wo_answer_min_len | remove min answer length | 100893 | +2 | 16.67% | 6.67% | 50.00% | 0.00% | 76.67% | 86.67% |
| 05_wo_answer_max_len | remove max answer length | 100891 | +0 | 6.67% | 3.33% | 30.00% | 0.00% | 86.67% | 90.00% |
| 06_wo_ambiguity_regex | remove ambiguity regex filter | 102941 | +2050 | 16.67% | 6.67% | 40.00% | 0.00% | 83.33% | 86.67% |
| 07_wo_r1_requirement | remove r1-solution requirement | 100891 | +0 | 0.00% | 0.00% | 30.00% | 0.00% | 70.00% | 83.33% |

### GSM8K Deltas vs Baseline

| variant | acc_delta | strict_delta | content_delta | xml_delta | degeneration_delta | truncation_delta |
| --- | --- | --- | --- | --- | --- | --- |
| 00_full_all_rules | +0.00 pp | +0.00 pp | +0.00 pp | +0.00 pp | +0.00 pp | +0.00 pp |
| 01_wo_difficulty_band | -3.33 pp | -6.67 pp | -3.33 pp | +0.00 pp | +3.33 pp | -3.34 pp |
| 02_wo_question_min_len | -6.67 pp | -3.33 pp | +6.67 pp | +0.00 pp | -6.67 pp | +10.00 pp |
| 03_wo_question_max_len | +6.67 pp | +6.67 pp | +6.67 pp | +0.00 pp | -3.34 pp | +6.66 pp |
| 04_wo_answer_min_len | +0.00 pp | -3.33 pp | +10.00 pp | +0.00 pp | +10.00 pp | +0.00 pp |
| 05_wo_answer_max_len | -10.00 pp | -6.67 pp | -10.00 pp | +0.00 pp | +20.00 pp | +3.33 pp |
| 06_wo_ambiguity_regex | +0.00 pp | -3.33 pp | +0.00 pp | +0.00 pp | +16.66 pp | +0.00 pp |
| 07_wo_r1_requirement | -16.67 pp | -10.00 pp | -10.00 pp | +0.00 pp | +3.33 pp | -3.34 pp |

## Presentation Framing

1. Start with how much each removed rule changes `kept_examples`.
2. Then compare whether the same rule changes mostly affect `accuracy`, `strict_accuracy`, or degeneration/truncation.
3. Highlight trade-offs: larger dataset vs cleaner supervision vs better protocol adherence.
