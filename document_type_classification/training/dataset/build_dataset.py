"""Assemble the unified document-type dataset from all sources.

Steps:
  1. pull Examples from every source loader,
  2. keep only the labels we currently model (dataset.LABELS),
  3. drop exact-duplicate documents (same normalised text),
  4. drop near-duplicate documents (keep one representative per near-dup group),
  5. write the result to data/ and print a summary.

Step 4 exists because the split audit found near-identical NDA templates leaking
across train and test. Removing near-dups here fixes it at the source and keeps
the split logic simple.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from dataset import LABELS, Example, write_jsonl
from fingerprint import NEAR_DUP_THRESHOLD, sketch, similarity
from sources.contract_nli import load_contract_nli
from sources.cuad import load_cuad

OUT_PATH = Path("data/document_type/dataset.jsonl")


def _normalise(text: str) -> str:
    """Collapse whitespace and lowercase, so trivially-different copies of the
    same document produce the same key and get deduped."""
    return " ".join(text.split()).lower()


def _load_exact_deduped() -> tuple[list[Example], int]:
    """All in-scope Examples with byte-identical (normalised) duplicates removed."""
    examples: list[Example] = []
    seen_text: set[str] = set()
    exact_dropped = 0
    for load in (load_cuad, load_contract_nli):
        for ex in load():
            if ex.type not in LABELS:
                continue  # a type we are not modelling yet (CUAD's ip / other)
            key = _normalise(ex.text)
            if key in seen_text:
                exact_dropped += 1
                continue
            seen_text.add(key)
            examples.append(ex)
    return examples, exact_dropped


def _drop_near_duplicates(examples: list[Example]) -> tuple[list[Example], int]:
    """Greedily keep a document only if it is not a near-duplicate of one already
    kept. The first occurrence wins and stands in for its near-dup group."""
    kept: list[Example] = []
    kept_sketches: list[frozenset[int]] = []
    near_dropped = 0
    for ex in examples:
        sk = sketch(ex.text)
        if any(similarity(sk, other) >= NEAR_DUP_THRESHOLD for other in kept_sketches):
            near_dropped += 1
            continue
        kept.append(ex)
        kept_sketches.append(sk)
    return kept, near_dropped


def build() -> list[Example]:
    examples, exact_dropped = _load_exact_deduped()
    kept, near_dropped = _drop_near_duplicates(examples)
    print(f"dropped {exact_dropped} exact + {near_dropped} near duplicates")
    return kept


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
