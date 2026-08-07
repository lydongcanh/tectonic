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

import re
from collections import Counter
from dataclasses import replace
from pathlib import Path

from dataset import LABELS, Example, write_jsonl
from fingerprint import NEAR_DUP_THRESHOLD, sketch, similarity
from sources.contract_nli import load_contract_nli
from sources.cuad import load_cuad
from sources.edgar_acquisition import load_edgar_acquisition
from sources.edgar_constitutional import load_edgar_constitutional
from sources.edgar_employment import load_edgar_employment
from sources.edgar_financial_statements import load_edgar_financial_statements
from sources.edgar_financing import load_edgar_financing
from sources.edgar_ip import load_edgar_ip
from sources.edgar_lease import load_edgar_lease

# Anchor data to the repo root so the location never depends on the current
# working directory. Relative paths resolve against wherever you launch from, so
# running this from a different folder would silently build a second dataset tree.
_REPO_ROOT = Path(__file__).resolve().parents[3]  # .../tectonic/
OUT_PATH = _REPO_ROOT / "data/document_type/dataset.jsonl"


def _normalise(text: str) -> str:
    """Collapse whitespace and lowercase, so trivially-different copies of the
    same document produce the same key and get deduped."""
    return " ".join(text.split()).lower()


# SEC filing wrappers that say WHERE a document was filed, not WHAT it is:
# exhibit labels ("Exhibit 10.16", "EX-3.1") and source filenames ("dex31.htm").
_EXHIBIT_LABEL = re.compile(r"\b(?:ex|exhibit)[\s-]*\d+(?:\.\d+)*(?:\([a-z0-9]+\))?\b", re.I)
_SOURCE_FILENAME = re.compile(r"\b\S+\.(?:htm|html|txt)\b", re.I)


def _strip_filing_boilerplate(text: str) -> str:
    """Remove SEC filing wrappers so the model keys on content, not on artifacts of
    our sources. Real dataroom documents won't carry "Exhibit 10", so letting the
    model learn it would hurt real-world generalisation. We remove only these
    wrappers, never real content like dates or "30 days"."""
    text = _EXHIBIT_LABEL.sub(" ", text)
    text = _SOURCE_FILENAME.sub(" ", text)
    return " ".join(text.split())


def _load_exact_deduped() -> tuple[list[Example], int]:
    """All in-scope Examples with byte-identical (normalised) duplicates removed."""
    examples: list[Example] = []
    seen_text: set[str] = set()
    exact_dropped = 0
    for load in (load_cuad, load_contract_nli, load_edgar_constitutional,
                 load_edgar_financial_statements, load_edgar_ip,
                 load_edgar_employment, load_edgar_lease, load_edgar_acquisition,
                 load_edgar_financing):
        for ex in load():
            if ex.type not in LABELS:
                continue  # a type we are not modelling yet (CUAD's ip / other)
            ex = replace(ex, text=_strip_filing_boilerplate(ex.text))
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


MIN_PER_LABEL = 10  # a class below this almost certainly means a source failed


def _require_all_labels(examples: list[Example]) -> None:
    """Fail loudly if any modelled label is missing or barely present.

    Without this, a source that fails to load (e.g. a transient network error)
    silently drops its whole class and the pipeline still reports success.
    """
    counts = Counter(ex.type for ex in examples)
    bad = [f"{lbl}={counts.get(lbl, 0)}" for lbl in LABELS if counts.get(lbl, 0) < MIN_PER_LABEL]
    if bad:
        raise SystemExit(
            f"FAILED: labels missing or under-populated (< {MIN_PER_LABEL}): "
            + ", ".join(bad)
            + ".\nA source probably failed to load. Fix it before continuing."
        )


def build() -> list[Example]:
    examples, exact_dropped = _load_exact_deduped()
    kept, near_dropped = _drop_near_duplicates(examples)
    print(f"dropped {exact_dropped} exact + {near_dropped} near duplicates")
    _require_all_labels(kept)
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
