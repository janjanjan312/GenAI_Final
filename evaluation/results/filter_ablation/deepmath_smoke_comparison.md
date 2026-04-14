# Evaluation Comparison

## Mathematical Capability

- **accuracy**: loose extraction (last number fallback)
- **strict_acc**: only counts answers in explicit format (boxed/Final Answer/answer tag)
- **content_acc**: correct answer appears anywhere in output (measures true math ability)
- **fmt_cond_acc**: accuracy among samples with explicit format (format-independent correctness)

| label | dataset | limit | num_correct | accuracy | strict_acc | content_acc | fmt_cond_acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 00_full_all_rules | jsonl | 30 | 2 | 6.67% | 3.33% | 26.67% | 25.00% |
| 01_wo_difficulty_band | jsonl | 30 | 2 | 6.67% | 0.00% | 26.67% | 0.00% |
| 02_wo_question_min_len | jsonl | 30 | 1 | 3.33% | 0.00% | 20.00% | 0.00% |
| 03_wo_question_max_len | jsonl | 30 | 3 | 10.00% | 0.00% | 23.33% | 0.00% |
| 04_wo_answer_min_len | jsonl | 30 | 2 | 6.67% | 0.00% | 23.33% | - |
| 05_wo_answer_max_len | jsonl | 30 | 2 | 6.67% | 0.00% | 23.33% | 0.00% |
| 06_wo_ambiguity_regex | jsonl | 30 | 1 | 3.33% | 0.00% | 20.00% | 0.00% |
| 07_wo_r1_requirement | jsonl | 30 | 1 | 3.33% | 0.00% | 23.33% | 0.00% |

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
| 00_full_all_rules | 6.67% | 40.00% | 50.00% | - | - | 3.33% | - |
| 01_wo_difficulty_band | 6.67% | 50.00% | 43.33% | - | - | - | - |
| 02_wo_question_min_len | 3.33% | 33.33% | 56.67% | - | - | 6.67% | - |
| 03_wo_question_max_len | 10.00% | 40.00% | 46.67% | - | - | 3.33% | - |
| 04_wo_answer_min_len | 6.67% | 16.67% | 73.33% | 3.33% | - | - | - |
| 05_wo_answer_max_len | 6.67% | 36.67% | 53.33% | - | 3.33% | - | - |
| 06_wo_ambiguity_regex | 3.33% | 36.67% | 60.00% | - | - | - | - |
| 07_wo_r1_requirement | 3.33% | 30.00% | 63.33% | - | - | - | 3.33% |

## Format Adherence

| label | dataset | limit | fmt_rate | truncated | has_think | has_answer | single_boxed | exact_xml |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 00_full_all_rules | jsonl | 30 | 13.33% | 93.33% | 0.00% | 0.00% | 0.00% | 0.00% |
| 01_wo_difficulty_band | jsonl | 30 | 6.67% | 93.33% | 0.00% | 0.00% | 0.00% | 0.00% |
| 02_wo_question_min_len | jsonl | 30 | 10.00% | 93.33% | 0.00% | 0.00% | 0.00% | 0.00% |
| 03_wo_question_max_len | jsonl | 30 | 6.67% | 90.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 04_wo_answer_min_len | jsonl | 30 | 0.00% | 93.33% | 0.00% | 0.00% | 0.00% | 0.00% |
| 05_wo_answer_max_len | jsonl | 30 | 6.67% | 93.33% | 0.00% | 0.00% | 0.00% | 0.00% |
| 06_wo_ambiguity_regex | jsonl | 30 | 3.33% | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| 07_wo_r1_requirement | jsonl | 30 | 3.33% | 86.67% | 0.00% | 0.00% | 0.00% | 0.00% |

## Reasoning Quality

| label | dataset | avg_words | avg_steps | truncation | repetition | degeneration |
| --- | --- | --- | --- | --- | --- | --- |
| 00_full_all_rules | jsonl | 590.8 | 3.83 | 93.33% | 53.33% | 53.33% |
| 01_wo_difficulty_band | jsonl | 618.7 | 4.63 | 93.33% | 50.00% | 50.00% |
| 02_wo_question_min_len | jsonl | 617.6 | 5.47 | 93.33% | 56.67% | 56.67% |
| 03_wo_question_max_len | jsonl | 604.4 | 4.53 | 90.00% | 50.00% | 50.00% |
| 04_wo_answer_min_len | jsonl | 612.7 | 4.5 | 93.33% | 76.67% | 76.67% |
| 05_wo_answer_max_len | jsonl | 612.1 | 4.53 | 93.33% | 60.00% | 60.00% |
| 06_wo_ambiguity_regex | jsonl | 614.9 | 4.1 | 100.00% | 63.33% | 63.33% |
| 07_wo_r1_requirement | jsonl | 613.0 | 5.0 | 86.67% | 66.67% | 66.67% |

## By Difficulty

| difficulty_bucket | 00_full_all_rules | 01_wo_difficulty_band | 02_wo_question_min_len | 03_wo_question_max_len | 04_wo_answer_min_len | 05_wo_answer_max_len | 06_wo_ambiguity_regex | 07_wo_r1_requirement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [3,5) | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 16.67% |
| [5,7) | 14.29% | 0.00% | 5.26% | 15.79% | 0.00% | 0.00% | 5.26% | 0.00% |
| [7,10] | 0.00% | 40.00% | 0.00% | 0.00% | 40.00% | 40.00% | 0.00% | 0.00% |

## By Topic

| topic | 00_full_all_rules | 01_wo_difficulty_band | 02_wo_question_min_len | 03_wo_question_max_len | 04_wo_answer_min_len | 05_wo_answer_max_len | 06_wo_ambiguity_regex | 07_wo_r1_requirement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mathematics -> Algebra -> Algebra -> Equations and Inequalities | - | 0.00% | 100.00% | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Mathematics -> Calculus -> Integral Calculus -> Techniques of Integration -> Multi-variable | - | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Mathematics -> Calculus -> Integral Calculus -> Techniques of Integration -> Single-variable | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Mathematics -> Geometry -> Solid Geometry -> 3D Shapes | - | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 100.00% |
| Mathematics -> Calculus -> Integral Calculus -> Applications of Integrals | 50.00% | 33.33% | 0.00% | 0.00% | 33.33% | 33.33% | 0.00% | 0.00% |
| Mathematics -> Applied Mathematics -> Statistics -> Probability -> Other | 0.00% | 33.33% | 0.00% | 33.33% | 33.33% | 33.33% | 33.33% | 0.00% |
| Mathematics -> Algebra -> Abstract Algebra -> Field Theory | 0.00% | - | - | - | - | - | - | - |
| Mathematics -> Algebra -> Algebra -> Polynomial Operations | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Mathematics -> Algebra -> Intermediate Algebra -> Complex Numbers | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Mathematics -> Algebra -> Intermediate Algebra -> Exponential Functions | 0.00% | - | - | - | - | - | - | - |
