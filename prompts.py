"""
Shared prompt template for both the zero-shot baseline and the fine-tuned
model, so the comparison is fair (same instruction, same input format).
"""

SYSTEM_INSTRUCTION = (
    "You are an information-extraction assistant. Read the paragraph about a "
    "grassroots innovation and output ONLY a single-line JSON object with "
    "exactly two keys: \"problem\" and \"solution\". Each value must be 1-2 "
    "plain-English sentences. Do not add any text before or after the JSON."
)


def build_prompt(paragraph: str) -> str:
    """Chat-style prompt used at both training and inference time."""
    return (
        f"<|system|>\n{SYSTEM_INSTRUCTION}\n"
        f"<|user|>\nParagraph: {paragraph}\n"
        f"<|assistant|>\n"
    )


def build_target(label: dict) -> str:
    """Canonical target string the model is trained to produce."""
    import json
    return json.dumps(
        {"problem": label["problem"], "solution": label["solution"]},
        ensure_ascii=False,
    )
