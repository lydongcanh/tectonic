"""Internal: turn documents into one dense vector each (chunk + mean-pool).

Private module (leading underscore): not part of the public API. It reproduces the exact
preprocessing used at training, so predictions match how the model was trained. A neural
encoder's token window is far shorter than a contract, so each document is split into word
windows, every window is embedded, and the windows are mean-pooled into one vector.
"""

from __future__ import annotations

import numpy as np


def _chunks(text: str, words_per_chunk: int, cap: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    windows = [
        " ".join(words[i : i + words_per_chunk])
        for i in range(0, len(words), words_per_chunk)
    ]
    return windows[:cap]


def embed_documents(encoder, texts: list[str], words_per_chunk: int, cap: int) -> np.ndarray:
    """One L2-normalized document vector per text. All windows across all documents are
    encoded in a single batched pass, then regrouped and mean-pooled per document."""
    per_doc = [_chunks(t, words_per_chunk, cap) for t in texts]
    flat = [c for doc in per_doc for c in doc]
    vecs = encoder.encode(flat, convert_to_numpy=True, normalize_embeddings=False)

    out = np.empty((len(texts), vecs.shape[1]), dtype=np.float32)
    cursor = 0
    for i, doc in enumerate(per_doc):
        pooled = vecs[cursor : cursor + len(doc)].mean(axis=0)
        cursor += len(doc)
        norm = np.linalg.norm(pooled)
        out[i] = pooled / norm if norm > 0 else pooled
    return out
