"""Audit the train/test split for near-duplicate leakage.

The split is stratified but does not by itself guarantee that a near-duplicate of
a training document is absent from the test set. A leaked near-dup means the
model effectively "sees" a test document while training, inflating the score.

We fingerprint each document (see fingerprint.py) and compare every train x test
pair. Pairs at/above the near-duplicate threshold are leaks; we also surface
near-misses for visibility. This audit only reports; it deletes nothing.

After build_dataset.py drops near-duplicates at the source, this should report
zero leaks.
"""

from __future__ import annotations

from pathlib import Path

from dataset import Example, read_jsonl
from fingerprint import NEAR_DUP_THRESHOLD, sketch, similarity

# Anchor to the repo root so the audit checks the same tree regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[3]  # .../tectonic/
DATA_DIR = _REPO_ROOT / "data/document_type"
REPORT_FLOOR = 0.3  # also surface near-misses above this, for visibility


def main() -> None:
    train = read_jsonl(DATA_DIR / "train.jsonl")
    test = read_jsonl(DATA_DIR / "test.jsonl")

    train_sk = [(ex, sketch(ex.text)) for ex in train]
    test_sk = [(ex, sketch(ex.text)) for ex in test]

    notable: list[tuple[float, Example, Example]] = []
    for te, ts in test_sk:
        for tr, trs in train_sk:
            sim = similarity(ts, trs)
            if sim >= REPORT_FLOOR:
                notable.append((sim, te, tr))

    notable.sort(key=lambda x: x[0], reverse=True)
    leaks = [row for row in notable if row[0] >= NEAR_DUP_THRESHOLD]

    print(f"checked {len(test)} test x {len(train)} train pairs")
    print(f"pairs >= {NEAR_DUP_THRESHOLD} (likely leaks): {len(leaks)}")
    print(f"pairs >= {REPORT_FLOOR} (incl. near-misses): {len(notable)}\n")
    for sim, te, tr in notable[:25]:
        print(f"  sim={sim:.2f}  TEST {te.type}:{te.doc_id[:45]}  ~  TRAIN {tr.type}:{tr.doc_id[:45]}")

    # Fail loudly: a leak must stop the pipeline, not be a line someone scrolls past.
    if leaks:
        raise SystemExit(
            f"\nFAILED: {len(leaks)} near-duplicate leak(s) across train/test. "
            "Fix the dedup in build_dataset.py before training."
        )


if __name__ == "__main__":
    main()
