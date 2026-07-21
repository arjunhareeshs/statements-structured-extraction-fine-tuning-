# Structured Extraction Fine-Tuning — Qwen2.5-0.5B-Instruct

Fine-tunes a small instruction-tuned LLM to extract a compact
`{"problem": ..., "solution": ...}` JSON pair from a paragraph describing
one grassroots innovation (in the style of Honey Bee Network / NIF stories).

> **Status:** scripts are complete and ready to run, but have **not yet been
> executed** (built in a sandboxed, GPU-less, network-less environment).
> `results.md` is an honest template — see [Honest Disclosures](#honest-disclosures).

---

## Table of Contents

- [Repository Layout](#repository-layout)
- [Quick Start](#quick-start)
- [Model Choice](#model-choice)
- [LoRA Configuration](#lora-configuration)
- [Cost & Time Estimate](#cost--time-estimate)
- [JSON Parsing & Failure Handling](#json-parsing--failure-handling)
- [Honest Disclosures](#honest-disclosures)
- [Stretch Ideas](#stretch-ideas)

---

## Repository Layout

| Path | Description |
|---|---|
| `data/train.jsonl` | 414 rows / 103 unique stories |
| `data/val.jsonl` | 52 rows / 13 unique stories |
| `data/test.jsonl` | 52 rows / 13 unique stories |
| `build_dataset.py` | Generates the splits above from 129 sourced cases (rerun-able, seeded) |
| `prompts.py` | Shared prompt template used by `train.py` and `eval.py` |
| `train.py` | LoRA fine-tuning script (HF Trainer + PEFT) |
| `eval.py` | Baseline vs. fine-tuned evaluation on the test split |
| `results.md` | Metrics table template + qualitative examples (fill in after a real run) |
| `SOURCES.md` | Per-case source URLs and honesty flags |

---

## Quick Start

```bash
pip install -U transformers peft accelerate bitsandbytes datasets rouge-score --break-system-packages
```

```bash
# 1. (optional) regenerate the dataset — already committed, but this is
#    how it was built, and it's fully reproducible (seeded)
python build_dataset.py

# 2. Zero-shot baseline
python eval.py --mode baseline

# 3. Fine-tune (writes lora-adapter/)
python train.py

# 4. Evaluate the fine-tuned model
python eval.py --mode finetuned --adapter lora-adapter

# 5. Fill in results.md with the two summaries + 3 qualitative rows
```

---

## Model Choice

**Qwen2.5-0.5B-Instruct**

| Option | Verdict |
|---|---|
| GPT-4o / Claude API | Great quality, but defeats the point of avoiding per-token API cost at thousands-of-records scale |
| **Qwen2.5-0.5B-Instruct** ✅ | Cheap, fast inference (the actual production goal); its instruct checkpoint follows a "reply with only this JSON" instruction more reliably zero-shot than Llama-3.2-1B in informal reports |
| Llama-3.2-1B-Instruct | Valid alternative — `train.py`/`eval.py` work unchanged if you swap `MODEL_NAME` |
| Qwen2.5-1.5B-Instruct | Better raw quality, but ~3x inference cost for a narrow 2-field extraction task — only worth it if the 0.5B fine-tune underperforms |

At 0.5B parameters, a full LoRA fine-tune fits on a free T4 without 4-bit
quantization, and inference is cheap enough to comfortably support
thousands of records on a single CPU or small GPU instance.

---

## LoRA Configuration

```python
LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    bias="none", task_type=TaskType.CAUSAL_LM,
)
```

**Why r=16, not r=8:** the model isn't just adapting to a topic — it's
learning a strict output grammar (single-line JSON, exactly two keys) it
doesn't reliably produce zero-shot, plus new domain vocabulary (e.g.
*groundnut decorticator*, *biomass gasifier*). That's more to encode than
typical style-adaptation LoRA use cases where r=8 is standard.

**Why not r=32:** with 414 training rows still a modest dataset, a larger
adapter raises overfitting risk with no expected gain — r=16 is the
smaller of the two "safe" options for a model this size.

**Why all 7 linear projections, not just q_proj/v_proj:** restricting LoRA
to attention-only (the classic default for 7B models) leaves the MLP
layers (`gate_proj`/`up_proj`/`down_proj`) unadapted — but MLP blocks carry
much of a transformer's format and vocabulary knowledge. Since the target
behavior change here includes format compliance, not just phrasing style,
all 7 projection types are targeted. This is still a small adapter overall
— well under 1% of the base model's parameters.

**Alpha = 2× rank** is the standard rule of thumb, keeping the effective
LoRA scaling factor (α/r = 2) constant regardless of rank.

---

## Cost & Time Estimate

> Estimated, not measured — no GPU was available in this sandbox.

- 414 training rows, 6 epochs, effective batch size 8 → ~310 optimizer steps total.
- A 0.5B-parameter LoRA fine-tune at this data size typically takes **10–20 minutes** on a free T4, based on published throughput figures for similarly-sized runs.
- **$0 marginal cost** on free-tier Colab/Kaggle; on a paid on-demand T4 (~$0.35–0.60/hr on common clouds), a 15-minute run is well under $0.20.

Replace this section with the real wall-clock time and billed cost once `train.py` has actually been run.

---

## JSON Parsing & Failure Handling

Small base models frequently wrap JSON in markdown fences, prepend a
leading sentence, use smart quotes, or leave a trailing comma. `eval.py`'s
`extract_json()` handles this in order:

1. Direct `json.loads`
2. Regex pull of the first `{...}` span
3. A small set of common repairs (strip code fences, normalize smart quotes, drop trailing commas)

**Parse rate is reported as its own metric**, separate from ROUGE-L — see
`results.md` and the `eval.py` docstring for why parse failures aren't
folded into the text-similarity score.

---

## Honest Disclosures

- `problem`/`solution` labels were written by Claude from sourced facts,
  not hand-verified by a second human reviewer — see `SOURCES.md` for the
  full disclosure and per-case source list.
- 518 total rows come from only **129 independently-sourced real stories**
  (each expanded into 4–5 paraphrased variants, split by story so no story
  crosses train/val/test). This is disclosed, not hidden — see
  `SOURCES.md` for the rationale and how to extend toward more independent
  stories for a larger, more robust benchmark.
- **No metrics in this repo are fabricated.** `results.md` is a template
  precisely because this sandbox can't run a GPU model — filling it with
  invented numbers would be worse than leaving it blank.

---

## Stretch Ideas

*(not implemented here)*

- **Cost-vs-quality curve:** compare against few-shot prompting of a
  larger API model (e.g. Claude Haiku) — `prompts.py`'s template can be
  reused directly, just add 2–3 worked examples before the paragraph.
- **Calibration analysis:** log per-token logprobs for the JSON values and
  check whether confidently-produced fields are more or less likely to be
  wrong — `model.generate(..., output_scores=True)` provides what's
  needed; not built here to keep this deliverable focused.
