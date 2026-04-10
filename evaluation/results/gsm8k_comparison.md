# Evaluation Comparison

## Mathematical Capability

- **accuracy**: loose extraction (last number fallback)
- **strict_acc**: only counts answers in explicit format (boxed/Final Answer/answer tag)
- **content_acc**: correct answer appears anywhere in output (measures true math ability)
- **fmt_cond_acc**: accuracy among samples with explicit format (format-independent correctness)

| label | dataset | limit | num_correct | accuracy | strict_acc | content_acc | fmt_cond_acc | pass@1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | parquet | 100 | 10 | 10.00% | 0.00% | 25.00% | 0.00% | - |
| sft | parquet | 100 | 6 | 6.00% | 0.00% | 18.00% | 0.00% | - |
| grpo | parquet | 100 | 1 | 1.00% | 0.00% | 15.00% | 0.00% | 3.00% |

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
| base | 10.00% | 13.00% | 68.00% | - | 9.00% | - | - |
| sft | 6.00% | 79.00% | 4.00% | 1.00% | 10.00% | - | - |
| grpo | 1.00% | 85.00% | 5.00% | - | 9.00% | - | - |

## Format Adherence

| label | dataset | limit | fmt_rate | truncated | has_think | has_answer | single_boxed | exact_xml |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | parquet | 100 | 9.00% | 89.00% | 5.00% | 5.00% | 2.00% | 0.00% |
| sft | parquet | 100 | 3.00% | 89.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| grpo | parquet | 100 | 4.00% | 90.00% | 2.00% | 2.00% | 1.00% | 0.00% |

## Reasoning Quality

| label | dataset | avg_words | avg_steps | truncation | repetition | degeneration |
| --- | --- | --- | --- | --- | --- | --- |
| base | parquet | 145.2 | 2.03 | 89.00% | 75.00% | 75.00% |
| sft | parquet | 177.5 | 2.0 | 89.00% | 4.00% | 4.00% |
| grpo | parquet | 180.6 | 2.0 | 90.00% | 5.00% | 5.00% |
