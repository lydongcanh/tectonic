"""Assemble the unified document-type dataset from all sources.

Steps:
  1. pull Examples from every source loader,
  2. keep only the labels we currently model (dataset.LABELS),
  3. drop exact-duplicate documents (same text) anywhere,
  4. write the result to data/ and print a summary.

Run from the repo root:
    poetry run python document_type_classification/training/build_dataset.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from dataset import LABELS, Example, write_jsonl
from sources.contract_nli import load_contract_nli
from sources.cuad import load_cuad

OUT_PATH = Path("data/document_type/dataset.jsonl")


def _normalise(text: str) -> str:
    """Collapse whitespace and lowercase, so trivially-different copies of the
    same document produce the same key and get deduped."""
    return " ".join(text.split()).lower()


def build() -> list[Example]:
    examples: list[Example] = []
    seen_text: set[str] = set()

    for load in (load_cuad, load_contract_nli):
        for ex in load():
            if ex.type not in LABELS:
                continue  # a type we are not modelling yet (CUAD's ip / other)

            key = _normalise(ex.text)
            if key in seen_text:
                continue  # exact-duplicate document, skip

            seen_text.add(key)
            examples.append(ex)
            
    return examples


def main() -> None:
    examples = build()
    write_jsonl(examples, OUT_PATH)

    print(f"{len(examples)} documents written to {OUT_PATH}\n")
    print("by type:")
    for t, n in Counter(ex.type for ex in examples).most_common():
        print(f"  {n:4}  {t}")
    print("by source:")
    for s, n in Counter(ex.source for ex in examples).most_common():
        print(f"  {n:4}  {s}")


if __name__ == "__main__":
    main()
