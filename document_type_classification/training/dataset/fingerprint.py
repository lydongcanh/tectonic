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


def sketch(text: str) -> frozenset[int]:
    """Summarise a document as the SKETCH_SIZE smallest shingle hashes."""
    words = text.lower().split()
    hashes: set[int] = set()

    for i in range(len(words) - SHINGLE_WORDS + 1):
        shingle = " ".join(words[i : i + SHINGLE_WORDS])
        digest = hashlib.blake2b(shingle.encode(), digest_size=8).digest()
        hashes.add(int.from_bytes(digest))

    # bottom-k: the smallest k distinct hashes stand in for the whole document
    return frozenset(sorted(hashes)[:SKETCH_SIZE])


def similarity(a: frozenset[int], b: frozenset[int]) -> float:
    """Estimated Jaccard overlap of two sketches (0.0 disjoint, 1.0 identical)."""
    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)
