"""Split the unified dataset into train and test, without fooling ourselves.

Two properties we want from the split:

  * stratified: train and test keep the same class mix as the whole dataset, so
    the test set is representative and not accidentally lopsided toward one class.
  * no leakage: a document, OR A NEAR-DUPLICATE of it, must not sit on both sides,
    or the test score is inflated by a near-twin the model already saw in training.

Exact-duplicate documents were removed in build_dataset.py. Near-duplicates were
KEPT there (they are often distinct documents sharing a boilerplate template), so
this file must prevent them from leaking across the split. It does so with a
GROUP-AWARE split: near-dup documents are first clustered (fingerprint.cluster,
exact-Jaccard), then whole clusters are assigned to one side. A singleton document
is its own group, so unique docs split normally; a near-dup family moves together.
The audit step (audit_split.py) independently re-checks that no near-twin straddles.

We stratify by hand (split each class, then combine) rather than call a library,
because the mechanic is the whole lesson.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path

from dataset import Example, read_jsonl, write_jsonl
from fingerprint import cluster

# Anchor to the repo root so the split reads/writes the same tree regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[3]  # .../tectonic/
DATA_DIR = _REPO_ROOT / "data/document_type"
DATASET = DATA_DIR / "dataset.jsonl"
TRAIN_OUT = DATA_DIR / "train.jsonl"
TEST_OUT = DATA_DIR / "test.jsonl"

TEST_FRACTION = 0.2
SEED = 20260806  # any fixed number; makes the split identical every run
# Cluster documents whose exact-Jaccard overlap is at least this. Set at the audit's
# review floor (not its 0.8 leak threshold) so no near-twin above 0.6 can straddle,
# which removes template overlap from the eval, not just literal duplicates.
CLUSTER_THRESHOLD = 0.6


def stratified_split(
    examples: list[Example], test_fraction: float, seed: int
) -> tuple[list[Example], list[Example]]:
    """Group-aware stratified split: keep near-dup clusters together, on one side.

    Documents are grouped into near-dup clusters, then each class is split at the
    CLUSTER level (whole clusters go to test or train). Splitting per class keeps
    the stratification; splitting by cluster keeps near-twins off opposite sides.
    Most clusters are singletons, so this behaves like an ordinary stratified split
    apart from the handful of near-dup families, which move as a unit.
    """
    groups = cluster([ex.text for ex in examples], CLUSTER_THRESHOLD)

    # Gather each cluster's members, then bucket clusters by their (dominant) class.
    members: dict[int, list[Example]] = defaultdict(list)
    for ex, gid in zip(examples, groups):
        members[gid].append(ex)
    by_type: dict[str, list[list[Example]]] = defaultdict(list)
    for group in members.values():
        cls = Counter(m.type for m in group).most_common(1)[0][0]
        by_type[cls].append(group)

    rng = random.Random(seed)
    train: list[Example] = []
    test: list[Example] = []
    for clusters in by_type.values():
        rng.shuffle(clusters)  # deterministic because the rng is seeded
        n_target = round(sum(len(g) for g in clusters) * test_fraction)
        n_test = 0
        for group in clusters:
            if n_test < n_target:
                test.extend(group)
                n_test += len(group)
            else:
                train.extend(group)

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
