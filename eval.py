"""
eval.py — evaluate zero-shot baseline vs LoRA fine-tuned model on data/test.jsonl.

    pip install -U transformers peft accelerate rouge-score datasets --break-system-packages

Usage:
    python eval.py --mode baseline
    python eval.py --mode finetuned --adapter lora-adapter

Outputs a metrics summary to stdout and a per-row JSON file
(predictions_<mode>.jsonl) for building the qualitative examples in
results.md.

METRIC CHOICE (see README.md for the full write-up):
We score with ROUGE-L (per field: problem, solution; then averaged) computed
only over rows where the JSON parsed successfully, and we report JSON parse
rate as a first-class separate metric rather than folding parse failures
into the text metric. Rationale:
  - This is a *summarization/rewriting* task, not an exact-match slot-filling
    task -- there are many acceptable phrasings of the same problem/solution
    -- so ROUGE-L (longest common subsequence overlap) is more forgiving and
    informative than exact string match, and it is deterministic/free, unlike
    LLM-as-judge.
  - JSON parse rate is reported separately and explicitly because it is the
    dominant failure mode for small base models on structured output, and a
    parse failure is a total miss regardless of what the text metric would
    say about the (non-existent) parsed fields. Averaging a "0" score for
    unparsable rows into ROUGE-L would hide *why* the model failed.
"""
import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rouge_score import rouge_scorer

from prompts import build_prompt

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TEST_PATH = "data/test.jsonl"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def extract_json(raw_text):
    """
    Small models often wrap JSON in prose or code fences, or leave a
    trailing comma/quote. Try, in order:
      1. json.loads on the raw text directly.
      2. Find the first {...} span via regex and parse that.
      3. Common repairs: strip markdown fences, smart quotes -> straight
         quotes, trailing commas removed.
    Returns (parsed_dict_or_None, repaired_bool).
    """
    text = raw_text.strip()

    # 1. direct
    try:
        return json.loads(text), False
    except Exception:
        pass

    # 2. first {...} span
    match = re.search(r"\{.*\}", text, re.DOTALL)
    candidate = match.group(0) if match else text

    # 3. common repairs
    repaired = candidate
    repaired = repaired.replace("```json", "").replace("```", "")
    repaired = repaired.replace("\u201c", '"').replace("\u201d", '"')
    repaired = re.sub(r",\s*}", "}", repaired)
    repaired = re.sub(r",\s*]", "]", repaired)

    try:
        return json.loads(repaired), (repaired != candidate or match is not None)
    except Exception:
        return None, False


def generate(model, tokenizer, paragraph, max_new_tokens=150):
    prompt = build_prompt(paragraph)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run_eval(mode, adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if mode == "finetuned":
        assert adapter_path, "--adapter required for --mode finetuned"
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    test_rows = load_jsonl(TEST_PATH)

    n_parsed = 0
    n_repaired = 0
    rouge_problem, rouge_solution = [], []
    results = []

    for row in test_rows:
        raw = generate(model, tokenizer, row["text"])
        parsed, was_repaired = extract_json(raw)

        record = {
            "row_id": row["row_id"],
            "text": row["text"],
            "gold": row["label"],
            "raw_output": raw,
            "parsed": parsed,
        }

        if parsed and "problem" in parsed and "solution" in parsed:
            n_parsed += 1
            if was_repaired:
                n_repaired += 1
            r_p = scorer.score(row["label"]["problem"], str(parsed["problem"]))["rougeL"].fmeasure
            r_s = scorer.score(row["label"]["solution"], str(parsed["solution"]))["rougeL"].fmeasure
            rouge_problem.append(r_p)
            rouge_solution.append(r_s)
            record["rougeL_problem"] = r_p
            record["rougeL_solution"] = r_s
        else:
            record["rougeL_problem"] = None
            record["rougeL_solution"] = None

        results.append(record)

    n = len(test_rows)
    summary = {
        "mode": mode,
        "n_test_rows": n,
        "json_parse_rate": n_parsed / n,
        "json_repaired_rate_of_parsed": (n_repaired / n_parsed) if n_parsed else 0.0,
        "avg_rougeL_problem": sum(rouge_problem) / len(rouge_problem) if rouge_problem else 0.0,
        "avg_rougeL_solution": sum(rouge_solution) / len(rouge_solution) if rouge_solution else 0.0,
    }
    summary["avg_rougeL_overall"] = (
        (summary["avg_rougeL_problem"] + summary["avg_rougeL_solution"]) / 2
    )

    out_path = f"predictions_{mode}.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"Per-row predictions written to {out_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "finetuned"], required=True)
    parser.add_argument("--adapter", default=None, help="path to LoRA adapter dir")
    args = parser.parse_args()
    run_eval(args.mode, args.adapter)
