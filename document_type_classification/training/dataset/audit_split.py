"""Audit the train/test split for near-duplicate leakage.

The split is stratified but does not by itself guarantee that a near-duplicate of
a training document is absent from the test set. A leaked near-dup means the model
effectively "sees" a test document while training, inflating the score.

This audit is deliberately NOT a re-run of the dedup step. build_dataset.py drops
near-duplicates using the fast SAMPLED sketch (bottom-k MinHash), whose Jaccard is
only an estimate; a genuinely near-identical pair can be under-estimated below the
0.8 threshold and survive. If this audit used that same estimate at that same
threshold it could only ever confirm what dedup already enforced (a tautology).
Instead it re-checks with the EXACT Jaccard over full shingle sets, so it can catch
true near-dups that dedup's sampling missed, an independent verification.

To stay cheap it prefilters all train x test pairs with the fast sketch and only
computes the exact Jaccard for pairs whose estimate clears REPORT_FLOOR. That floor
sits far below the 0.8 threshold on purpose: with a 128-hash sketch the estimate's
error is a few hundredths, so a truly >=0.8 pair cannot be estimated anywhere near
0.3, and the exact recheck can never be skipped for a real leak. This audit only
reports and fails; it deletes nothing.
"""

from __future__ import annotations

from pathlib import Path

from dataset import Example, read_jsonl
from fingerprint import NEAR_DUP_THRESHOLD, full_shingles, similarity, sketch

# Anchor to the repo root so the audit checks the same tree regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[3]  # .../tectonic/
DATA_DIR = _REPO_ROOT / "data/document_type"
REPORT_FLOOR = 0.3  # prefilter: compute the exact Jaccard for estimates above this
WARN_FLOOR = 0.6    # exact similarity in [WARN_FLOOR, threshold) = review by hand


def main() -> None:
    train = read_jsonl(DATA_DIR / "train.jsonl")
    test = read_jsonl(DATA_DIR / "test.jsonl")

    train_sk = [(ex, sketch(ex.text)) for ex in train]
    test_sk = [(ex, sketch(ex.text)) for ex in test]

    # Stage 1 (fast, approximate): find candidate close pairs by the sampled sketch.
    candidates: list[tuple[Example, Example]] = []
    for te, ts in test_sk:
        for tr, trs in train_sk:
            if similarity(ts, trs) >= REPORT_FLOOR:
                candidates.append((te, tr))

    # Stage 2 (exact, independent): recompute the true Jaccard for each candidate on
    # full shingle sets. Cache the full sets so each involved doc is shingled once.
    full: dict[str, frozenset[int]] = {}

    def exact(ex: Example) -> frozenset[int]:
        if ex.doc_id not in full:
            full[ex.doc_id] = full_shingles(ex.text)
        return full[ex.doc_id]

    scored = [(similarity(exact(te), exact(tr)), te, tr) for te, tr in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    leaks = [row for row in scored if row[0] >= NEAR_DUP_THRESHOLD]
    warn = [row for row in scored if WARN_FLOOR <= row[0] < NEAR_DUP_THRESHOLD]

    print(f"checked {len(test)} test x {len(train)} train pairs "
          f"({len(candidates)} candidates rechecked with exact Jaccard)")
    print(f"exact >= {NEAR_DUP_THRESHOLD} (leaks): {len(leaks)}")
    print(f"exact in [{WARN_FLOOR}, {NEAR_DUP_THRESHOLD}) (review by hand): {len(warn)}\n")
    for sim, te, tr in scored[:25]:
        print(f"  exact={sim:.2f}  TEST {te.type}:{te.doc_id[:45]}  ~  TRAIN {tr.type}:{tr.doc_id[:45]}")

    # Fail loudly: a leak must stop the pipeline, not be a line someone scrolls past.
    if leaks:
        raise SystemExit(
            f"\nFAILED: {len(leaks)} near-duplicate leak(s) across train/test (exact "
            "Jaccard). dedup's sampled sketch under-estimated these; lower the dedup "
            "threshold or drop the offending docs in build_dataset.py before training."
        )


if __name__ == "__main__":
    main()
