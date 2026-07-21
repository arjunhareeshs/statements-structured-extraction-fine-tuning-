# Results

> **Status: template, not yet run.** This sandbox has no GPU and no network
> access (see README.md), so `train.py` / `eval.py` could not actually be
> executed here. What follows is the exact structure to fill in — run the
> two commands below on Colab/Kaggle T4 and drop the numbers in.
>
> ```bash
> python eval.py --mode baseline                        > baseline.json
> python train.py                                       # writes lora-adapter/
> python eval.py --mode finetuned --adapter lora-adapter > finetuned.json
> ```

## Metrics (test split, n=12 rows / 3 unique stories)

| Metric | Zero-shot baseline (Qwen2.5-0.5B-Instruct) | LoRA fine-tuned |
|---|---|---|
| JSON parse rate | TODO | TODO |
| ROUGE-L — `problem` field (avg, parsed rows only) | TODO | TODO |
| ROUGE-L — `solution` field (avg, parsed rows only) | TODO | TODO |
| ROUGE-L — overall avg | TODO | TODO |
| % outputs that needed JSON repair (of parsed) | TODO | TODO |

Fill each `TODO` from the `summary` dict that `eval.py` prints to stdout.

## Why this metric (ROUGE-L + parse rate, not exact match)

See the docstring in `eval.py` for the full reasoning. Short version:
problem/solution extraction has many valid phrasings, so ROUGE-L (longest
common subsequence overlap) rewards genuine content overlap without
requiring word-for-word match, is free and deterministic (unlike an
LLM-judge, which costs money and adds variance), and is standard for
summarization-style tasks. JSON parse rate is reported *separately* rather
than folded into the text score, because a small base model's most common
failure mode is producing invalid JSON — that's a distinct, important
failure to see on its own, not one to average away.

## Qualitative examples

Below are 3 real test-set inputs. After running eval, paste the baseline
and fine-tuned raw outputs (from `predictions_baseline.jsonl` /
`predictions_finetuned.jsonl`) into the blanks, and write one sentence on
whether fine-tuning helped, hurt, or made no difference for that row.

### Example 1

**Input paragraph:**
> _(copy `text` field of the first row of `data/test.jsonl` here)_

**Gold label:**
```json
_(copy `label` field of that row here)_
```

**Baseline output:** `TODO — paste raw_output from predictions_baseline.jsonl`

**Fine-tuned output:** `TODO — paste raw_output from predictions_finetuned.jsonl`

**Verdict:** TODO — e.g. "Baseline invented an extra key and wrapped the JSON
in markdown fences (parse failure). Fine-tuned matched the schema exactly
and paraphrased the solution correctly — clear win."

### Example 2

**Input paragraph:** TODO
**Gold label:** TODO
**Baseline output:** TODO
**Fine-tuned output:** TODO
**Verdict:** TODO — e.g. "Both parsed fine. Fine-tuned's `problem` field
was shorter and dropped a location detail the gold label had — a case
where fine-tuning traded completeness for a more 'templated' brevity it
overfit to from the training paraphrases."

### Example 3

**Input paragraph:** TODO
**Gold label:** TODO
**Baseline output:** TODO
**Fine-tuned output:** TODO
**Verdict:** TODO — pick a row where fine-tuning *hurt*, if one exists
(e.g. hallucinated a fact not in the paragraph, or overfit to a template
phrase from training data that doesn't fit this input) — an honest report
should show at least one such case if it occurred, not only wins.

## Known limitation of this specific run

The test split has only 3 unique underlying stories (12 rows, 4 paraphrases
each) because the source dataset has only 37 independently-sourced cases
total. A metric computed on 3 stories has high variance — treat the numbers
above as a directional signal for this prototype, not a statistically
robust benchmark. Scaling `CASES` in `build_dataset.py` to 100+
independently-sourced stories (e.g. from the NIF award-book PDFs in
SOURCES.md) before re-running would give a much more trustworthy result.
