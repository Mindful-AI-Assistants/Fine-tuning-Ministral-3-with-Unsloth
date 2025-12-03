
"""
Data utilities and Unsloth integration examples.
This module provides reproducible, auditable steps for dataset curation.
Adapt and extend rules to your data sources and Unsloth recipes.
"""

import json
import re
from pathlib import Path
from typing import Iterable, Dict


def load_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


CLEAN_REPLACEMENTS = [
    (re.compile(r"\s+"), " "),  # collapse whitespace
    (re.compile(r"\s([?.!,;:])"), r"\1"),  # remove space before punctuation
]


def clean_text(text: str) -> str:
    t = text.strip()
    for pattern, repl in CLEAN_REPLACEMENTS:
        t = pattern.sub(repl, t)
    # Basic PII removal examples — extend as needed
    t = re.sub(r"\b\d{9,}\b", "[REDACTED_NUMBER]", t)
    return t


def process_jsonl_to_jsonl(in_path: Path, out_path: Path, text_field: str = "text"):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        for item in load_jsonl(in_path):
            if text_field not in item:
                continue
            text = clean_text(item[text_field])
            if len(text) < 10:
                continue
            fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


def sample_jsonl(in_path: Path, out_path: Path, n: int = 100):
    import random

    lines = list(load_jsonl(in_path))
    sampled = random.sample(lines, min(n, len(lines)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        for s in sampled:
            fout.write(json.dumps(s, ensure_ascii=False) + "\n")
