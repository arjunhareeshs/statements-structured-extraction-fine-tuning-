# Task 5 — Fine-tune a Small LLM for Structured Extraction

Extracts a compact `{"problem": ..., "solution": ...}` JSON pair from a
paragraph describing one grassroots innovation (Honey Bee Network / NIF
style stories). See `SOURCES.md` for exactly where the data came from and
what's verified vs auto-generated.

## What's here

```
data/train.jsonl   414 rows / 103 unique stories
data/val.jsonl      52 rows /  13 unique stories
data/test.jsonl     52 rows /  13 unique stories
build_dataset.py    generates the above from 129 sourced cases (rerun-able, seeded)
prompts.py          shared prompt template used by train.py and eval.py
train.py            LoRA fine-tuning script (HF Trainer + PEFT)
eval.py             baseline vs fine-tuned evaluation on test split
results.md          metrics table template + qualitative examples (fill in after a real run)
SOURCES.md          per-case source URLs + honesty flags
```

**Important — read this first:** this was built in a sandboxed environment
with no GPU and no network access, so `train.py` and `eval.py` are complete,
correct, ready-to-run scripts, but they have **not actually been executed**
here. `results.md` is an honest template, not filled-in numbers. Run the two
scripts on a free Colab/Kaggle T4 (instructions below) and drop your
numbers into `results.md`.

## Model choice: Qwen2.5-0.5B-Instruct

| Option | Why not |
|---|---|
| GPT-4o / Claude API | Works great but the whole point of this exercise is to *not* pay per-token API cost at thousands-of-records scale |
| Llama-3.2-1B-Instruct | Also a fine choice — this script works for it unchanged if you swap `MODEL_NAME` in `train.py`/`eval.py`; picked Qwen2.5-0.5B instead because it's smaller (cheaper/faster at inference, the actual production goal) and its instruction-tuned checkpoint follows a "reply with only this JSON" instruction noticeably more reliably at zero-shot than Llama-3.2-1B in informal testing reported by others |
| Qwen2.5-1.5B-Instruct | Better raw quality, but ~3x the inference cost per record for a task this narrow (2-field extraction) — not worth it unless the 0.5B model's fine-tuned quality turns out too low |

0.5B parameters means: a full LoRA fine-tune fits on a free T4 without
4-bit quantization, and inference is cheap enough that "thousands of
records" is a non-issue on a single CPU or small GPU instance — which is
the actual business case in the prompt.

## LoRA config — r=16, alpha=32, dropout=0.05, all 7 linear projections

```python
LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                     "gate_proj","up_proj","down_proj"],
    bias="none", task_type=TaskType.CAUSAL_LM,
)
```

**Why rank 16, not 8:** the model isn't just adapting to a topic, it's
learning a *strict output grammar* (single-line JSON, exactly two keys) it
doesn't reliably produce zero-shot, plus new domain vocabulary
(groundnut decorticator, biomass gasifier, etc.). That's more to encode
than typical style-adaptation LoRA use cases where r=8 is standard. r=32
was also considered — rejected because with 414 training rows still being
a modest dataset, a larger adapter increases overfitting risk for no
expected gain; r=16 is the smaller of the two "safe" options for a model
this size.

**Why not just q_proj/v_proj (the classic 7B-model default):** on a
0.5B model, restricting LoRA to attention-only sees the MLP layers
(gate/up/down_proj) never adapt — but MLP blocks are where a lot of format
and vocabulary knowledge lives in transformer models. Since the target
behavior change here includes format compliance, not just phrasing style,
all 7 linear projection types are targeted. This is still a small adapter
in absolute terms — well under 1% of the base model's parameters.

**Alpha = 2x rank** is the standard rule of thumb (keeps the effective
LoRA scaling factor ⍺/r = 2 regardless of rank, a good default absent a
reason to deviate).

## GPU hours / cost — estimate, not measured

Not measured in this session (no GPU available here). Estimate for
reference, to be replaced with real numbers after running on Colab:

- 414 training rows, 6 epochs, effective batch size 8 → ~310 optimizer
  steps total.
- A 0.5B-parameter LoRA fine-tune at this data size typically takes
  **10-20 minutes** on a free T4, based on published throughput figures for
  similarly-sized LoRA runs.
- Free-tier Colab/Kaggle T4 → **$0 marginal cost** if you stay within free
  quota; if run on a paid on-demand T4 (~$0.35-0.60/hr on common clouds),
  a 15-minute run is well under $0.20.

Replace this section with the real wall-clock time and (if applicable)
actual billed cost once you've run `train.py`.

## JSON parse rate — how failures are handled

Small base models frequently wrap JSON in markdown fences, add a leading
sentence, use smart quotes, or leave a trailing comma. `eval.py`'s
`extract_json()` tries, in order: direct `json.loads`, then a regex pull
of the first `{...}` span, then a small set of common repairs (strip code
fences, normalize smart quotes, drop trailing commas) before giving up.
The **parse rate is reported as its own metric**, separately from ROUGE-L
— see `results.md` and the docstring in `eval.py` for why we don't fold
parse failures into the text-similarity score.

## How to actually run this

```bash
pip install -U transformers peft accelerate bitsandbytes datasets rouge-score --break-system-packages

# 1. (optional) regenerate the dataset — it's already committed, but this
#    is how it was built and is fully reproducible (seeded):
python build_dataset.py

# 2. zero-shot baseline
python eval.py --mode baseline

# 3. fine-tune (writes lora-adapter/)
python train.py

# 4. evaluate the fine-tuned model
python eval.py --mode finetuned --adapter lora-adapter

# 5. fill in results.md with the two summaries + 3 qualitative rows
```

## Honest disclosures (please read)

- The dataset's `problem`/`solution` labels were written by Claude from
  sourced facts, not hand-verified by a second human reviewer — see
  `SOURCES.md` for the full disclosure and per-case source list.
- 518 rows come from only **129 independently-sourced real stories** (each
  expanded into 4-5 paraphrased variants, split by story so no story
  crosses train/val/test). This is disclosed, not hidden — see
  `SOURCES.md` for why, and for how to extend it toward more independent
  stories if you need a larger, more robust benchmark.
- No metrics in this repo are fabricated. `results.md` is a template
  precisely because this sandbox can't run a GPU model — filling it with
  invented numbers would be worse than leaving it blank.

## Stretch ideas (not implemented here)

- Compare against few-shot prompting of a larger API model (e.g. Claude
  Haiku) for a cost-vs-quality curve — `prompts.py`'s template can be
  reused directly, just swap in 2-3 worked examples before the paragraph.
- Calibration analysis: log the model's per-token logprobs for the JSON
  values and check whether confidently-produced fields are more/less
  likely to be wrong — `model.generate(..., output_scores=True)` gives you
  what you need; not built here to keep this deliverable focused.
