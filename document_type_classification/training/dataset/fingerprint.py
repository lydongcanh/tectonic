"""Cheap near-duplicate fingerprints for documents.

A document is summarised by a MinHash "bottom-k" sketch: the k smallest hashes of
its word-shingles. Two documents that share most of their smallest hashes are
near-identical, so we can estimate text overlap without comparing full texts:

    similarity(a, b) = |a & b| / |a | b|     (a Jaccard estimate on the sketches)

This lives in one place because it is used both to drop near-duplicates when
building the dataset and to audit a split for leakage, and those two must agree.
"""

from __future__ import annotations

import hashlib

SKETCH_SIZE = 128  # how many hashes summarise each document
SHINGLE_WORDS = 5  # a "shingle" is this many consecutive words
NEAR_DUP_THRESHOLD = 0.8  # sketch similarity at/above this = near-duplicate


def _shingle_hashes(text: str) -> set[int]:
    """Every distinct word-shingle hash of a document (the full set, no sampling)."""
    words = text.lower().split()
    return {
        int.from_bytes(
            hashlib.blake2b(" ".join(words[i : i + SHINGLE_WORDS]).encode(), digest_size=8).digest()
        )
        for i in range(len(words) - SHINGLE_WORDS + 1)
    }


def sketch(text: str) -> frozenset[int]:
    """Summarise a document as the SKETCH_SIZE smallest shingle hashes.

    This is a SAMPLED estimate (bottom-k MinHash): fast, but its Jaccard is only
    approximate. Used to drop near-duplicates when building the dataset, where speed
    over all pairs matters. The audit re-checks survivors with `full_shingles` +
    `similarity` (exact Jaccard) so it does not merely re-run this same estimate.
    """
    # bottom-k: the smallest k distinct hashes stand in for the whole document
    return frozenset(sorted(_shingle_hashes(text))[:SKETCH_SIZE])


def full_shingles(text: str) -> frozenset[int]:
    """The COMPLETE shingle-hash set for an EXACT Jaccard (no bottom-k sampling).

    `sketch` samples 128 hashes, so its similarity is an estimate that can miss a
    true near-duplicate by chance (a genuinely 0.85-overlap pair can be estimated
    below the 0.8 dedup threshold and survive). Comparing full shingle sets removes
    that sampling variance, which is what makes the split audit an INDEPENDENT check
    on dedup rather than a re-run of the same approximation.
    """
    return frozenset(_shingle_hashes(text))


def similarity(a: frozenset[int], b: frozenset[int]) -> float:
    """Jaccard overlap of two shingle sets (0.0 disjoint, 1.0 identical).

    On two `sketch`es this is the sampled MinHash ESTIMATE; on two `full_shingles`
    sets it is the EXACT Jaccard. Same formula, different inputs.
    """
    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


# Prefilter floor for the exact recheck in `cluster`: a pair whose true Jaccard is at
# or above any realistic clustering threshold cannot have a 128-hash sketch estimate
# anywhere near this low, so no genuine near-dup is skipped by prefiltering here.
_CANDIDATE_FLOOR = 0.3


def cluster(texts: list[str], threshold: float) -> list[int]:
    """Group near-duplicate documents so a group-aware split can keep each group on
    one side of a train/test split (no near-twin straddling the split).

    Two texts join the same group when their EXACT shingle Jaccard (`full_shingles`)
    is at least `threshold`. The fast sampled `sketch` prefilters candidate pairs so
    we only pay for the exact comparison on plausibly-close ones. Grouping is
    transitive via union-find (A~B and B~C put A, B, C in one group). Returns one
    group id per input text; near-dup families share an id, unique docs get their own.
    """
    n = len(texts)
    sketches = [sketch(t) for t in texts]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    full: dict[int, frozenset[int]] = {}

    def full_of(i: int) -> frozenset[int]:
        if i not in full:
            full[i] = full_shingles(texts[i])
        return full[i]

    def is_near_dup(i: int, j: int) -> bool:
        if similarity(sketches[i], sketches[j]) < _CANDIDATE_FLOOR:
            return False  # not even plausibly close; skip the exact comparison
        return similarity(full_of(i), full_of(j)) >= threshold

    for i in range(n):
        for j in range(i + 1, n):
            if is_near_dup(i, j):
                union(i, j)

    return [find(i) for i in range(n)]
