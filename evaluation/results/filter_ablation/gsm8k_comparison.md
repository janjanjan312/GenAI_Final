# Evaluation Comparison

## Mathematical Capability

- **accuracy**: loose extraction (last number fallback)
- **strict_acc**: only counts answers in explicit format (boxed/Final Answer/answer tag)
- **content_acc**: correct answer appears anywhere in output (measures true math ability)
- **fmt_cond_acc**: accuracy among samples with explicit format (format-independent correctness)

| label | dataset | limit | num_correct | accuracy | strict_acc | content_acc | fmt_cond_acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 00_full_all_rules | parquet | 30 | 5 | 16.67% | 10.00% | 40.00% | 42.86% |
| 01_wo_difficulty_band | parquet | 30 | 4 | 13.33% | 3.33% | 36.67% | 16.67% |
| 02_wo_question_min_len | parquet | 30 | 3 | 10.00% | 6.67% | 46.67% | 28.57% |
| 03_wo_question_max_len | parquet | 30 | 7 | 23.33% | 16.67% | 46.67% | 62.50% |
| 04_wo_answer_min_len | parquet | 30 | 5 | 16.67% | 6.67% | 50.00% | 22.22% |
| 05_wo_answer_max_len | parquet | 30 | 2 | 6.67% | 3.33% | 30.00% | 16.67% |
| 06_wo_ambiguity_regex | parquet | 30 | 5 | 16.67% | 6.67% | 40.00% | 40.00% |
| 07_wo_r1_requirement | parquet | 30 | 0 | 0.00% | 0.00% | 30.00% | 0.00% |

## Error Classification

Breakdown of failure modes (rates sum to 1.0):
- **truncation**: output cut off before final answer
- **degeneration**: repetitive/degenerate output
- **fmt_only**: correct answer in output but extraction failed
- **computation**: numeric answer but wrong value
- **reasoning**: structured reasoning but wrong conclusion
- **comprehension**: misunderstood the problem

| label | correct | truncation | degeneration | fmt_only | computation | reasoning | comprehension |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 00_full_all_rules | 16.67% | 23.33% | 56.67% | - | 3.33% | - | - |
| 01_wo_difficulty_band | 13.33% | 13.33% | 66.67% | 3.33% | 3.33% | - | - |
| 02_wo_question_min_len | 10.00% | 40.00% | 50.00% | - | - | - | - |
| 03_wo_question_max_len | 23.33% | 23.33% | 46.67% | - | 6.67% | - | - |
| 04_wo_answer_min_len | 16.67% | 20.00% | 60.00% | - | 3.33% | - | - |
| 05_wo_answer_max_len | 6.67% | 10.00% | 83.33% | - | - | - | - |
| 06_wo_ambiguity_regex | 16.67% | 16.67% | 66.67% | - | - | - | - |
| 07_wo_r1_requirement | - | 26.67% | 70.00% | - | 3.33% | - | - |

## Format Adherence

| label | dataset | limit | fmt_rate | truncated | has_think | has_answer | single_boxed | exact_xml |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 00_full_all_rules | parquet | 30 | 23.33% | 86.67% | 0.00% | 0.00% | 6.67% | 0.00% |
| 01_wo_difficulty_band | parquet | 30 | 20.00% | 83.33% | 0.00% | 0.00% | 6.67% | 0.00% |
| 02_wo_question_min_len | parquet | 30 | 23.33% | 96.67% | 0.00% | 0.00% | 10.00% | 0.00% |
| 03_wo_question_max_len | parquet | 30 | 26.67% | 93.33% | 0.00% | 0.00% | 13.33% | 0.00% |
| 04_wo_answer_min_len | parquet | 30 | 30.00% | 86.67% | 0.00% | 0.00% | 6.67% | 0.00% |
| 05_wo_answer_max_len | parquet | 30 | 20.00% | 90.00% | 0.00% | 0.00% | 10.00% | 0.00% |
| 06_wo_ambiguity_regex | parquet | 30 | 16.67% | 86.67% | 0.00% | 0.00% | 3.33% | 0.00% |
| 07_wo_r1_requirement | parquet | 30 | 20.00% | 83.33% | 0.00% | 0.00% | 6.67% | 0.00% |

## Reasoning Quality

| label | dataset | avg_words | avg_steps | truncation | repetition | degeneration |
| --- | --- | --- | --- | --- | --- | --- |
| 00_full_all_rules | parquet | 630.2 | 5.83 | 86.67% | 66.67% | 66.67% |
| 01_wo_difficulty_band | parquet | 605.6 | 4.27 | 83.33% | 70.00% | 70.00% |
| 02_wo_question_min_len | parquet | 631.3 | 5.33 | 96.67% | 60.00% | 60.00% |
| 03_wo_question_max_len | parquet | 614.2 | 5.13 | 93.33% | 63.33% | 63.33% |
| 04_wo_answer_min_len | parquet | 641.2 | 4.23 | 86.67% | 76.67% | 76.67% |
| 05_wo_answer_max_len | parquet | 633.8 | 4.23 | 90.00% | 86.67% | 86.67% |
| 06_wo_ambiguity_regex | parquet | 642.2 | 3.9 | 86.67% | 83.33% | 83.33% |
| 07_wo_r1_requirement | parquet | 628.8 | 4.43 | 83.33% | 70.00% | 70.00% |
