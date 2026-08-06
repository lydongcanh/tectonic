"""Split the unified dataset into train and test, without fooling ourselves.

Two properties we want from the split:

  * stratified: train and test keep the same class mix as the whole dataset, so
    the test set is representative and not accidentally lopsided toward one class.
  * no leakage: a document (or a near-duplicate of it) must not sit on both
    sides, or the test score is inflated by copies the model already saw.

Exact-duplicate documents were already removed in build_dataset.py. Near-duplicate
leakage is *verified separately* by the audit step (audit_split.py); we do not
assume it away here. This file just does the stratified split, with a fixed seed
so the result is reproducible run to run.

We stratify by hand (split each class, then combine) rather than call a library,
because the mechanic is the whole lesson.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path

from dataset import Example, read_jsonl, write_jsonl

DATA_DIR = Path("data/document_type")
DATASET = DATA_DIR / "dataset.jsonl"
TRAIN_OUT = DATA_DIR / "train.jsonl"
TEST_OUT = DATA_DIR / "test.jsonl"

TEST_FRACTION = 0.2
SEED = 20260806  # any fixed number; makes the split identical every run


def stratified_split(
    examples: list[Example], test_fraction: float, seed: int
) -> tuple[list[Example], list[Example]]:
    """Split each class separately by `test_fraction`, then combine.

    Splitting per class is exactly what "stratified" means: it guarantees both
    sides carry the same proportion of each class as the input.
    """
    by_type: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        by_type[ex.type].append(ex)

    rng = random.Random(seed)
    train: list[Example] = []
    test: list[Example] = []
    for rows in by_type.values():
        shuffled = rows[:]         # copy, so we never reorder the caller's list
        rng.shuffle(shuffled)      # deterministic because the rng is seeded
        n_test = round(len(shuffled) * test_fraction)
        test.extend(shuffled[:n_test])
        train.extend(shuffled[n_test:])

    rng.shuffle(train)             # mix classes back together within each side
    rng.shuffle(test)
    return train, test


def _summary(name: str, rows: list[Example]) -> None:
    counts = Counter(ex.type for ex in rows)
    parts = ", ".join(f"{t}={n}" for t, n in sorted(counts.items()))
    print(f"{name:5} {len(rows):4}  ({parts})")


def main() -> None:
    examples = read_jsonl(DATASET)
    train, test = stratified_split(examples, TEST_FRACTION, SEED)

    write_jsonl(train, TRAIN_OUT)
    write_jsonl(test, TEST_OUT)

    _summary("all", examples)
    _summary("train", train)
    _summary("test", test)


if __name__ == "__main__":
    main()
