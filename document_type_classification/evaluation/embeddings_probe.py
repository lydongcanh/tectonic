"""Embeddings probe: does a SEMANTIC representation beat TF-IDF for our classifier?

This is a DIAGNOSTIC experiment, not a new maintained baseline. baseline.py turns a
document into a sparse bag of weighted word counts (TF-IDF); it can only "see" the
exact surface tokens present. This probe keeps the classifier (LogReg) and the splits
identical and swaps ONLY the representation: a frozen neural sentence encoder turns
each document into a dense 768-dim vector that captures meaning, so "licence" and
"grants a right to use" land near each other even with no shared words.

Isolating the representation is the whole point. If the score moves, it is the
embeddings talking, not a different classifier or test set. Two honesty guards:

  * Fair length. TF-IDF reads the WHOLE document. A neural encoder has a small token
    window (~384 tokens for mpnet, ~300 words), far shorter than a contract. Truncating
    to the first window would handicap embeddings and make any loss meaningless. So we
    CHUNK each document into word windows, embed each chunk, and MEAN-POOL into one
    document vector, so both representations see roughly the whole document. We cap the
    number of chunks (CHUNK_CAP) to bound cost; that cap is logged, not silent.

  * Right question. In-distribution parity is NOT the point (see the caveat below): our
    EDGAR test set cannot tell semantic from surface learning, because both live in the
    same house style. The point is GENERALIZATION, measured by the proxies in a
    follow-up (ip cross-source recall, which was 0.488 for TF-IDF, and the OOS set).
    This file (step 2a) builds the embedding engine and reports the in-distribution
    number as a sanity check; the proxies (step 2b) import embed_texts() from here.

Embedding is the slow part, so we compute each split's vectors ONCE and cache them to
artifacts/document_type/embeddings/. Read-only on the data; saves only the cache.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import certifi

# Cloudflare WARP installs SSL_CERT_FILE pointing at its own single-cert bundle and
# leaves it set even when WARP is turned OFF. OpenSSL/httpx honor that env var, so any
# TLS call (here: the HuggingFace model download via httpx) trusts ONLY the Cloudflare
# root and rejects legitimate public certs ("unable to get local issuer certificate").
# We point the process at certifi's complete public-root bundle instead. This is FULL
# verification against the standard Mozilla roots, not a weakening of it, and it must
# run before any SSL context (httpx client) is created, so it lives at import time.
os.environ["SSL_CERT_FILE"] = certifi.where()

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

_REPO_ROOT = Path(__file__).resolve().parents[2]  # .../tectonic/
DATA_DIR = _REPO_ROOT / "data/document_type"
CACHE_DIR = _REPO_ROOT / "artifacts/document_type/embeddings"

MODEL_NAME = "all-mpnet-base-v2"  # general high-quality sentence encoder, 768-dim
WORDS_PER_CHUNK = 250   # ~325 tokens, safely under mpnet's 384-token window
CHUNK_CAP = 12          # at most 12 chunks/doc (~3000 words); bounds embedding cost
ENCODE_BATCH = 64       # chunks per forward pass; throughput knob, not a result knob

# Encoders are loaded lazily and cached by name (each is hundreds of MB), so the
# bake-off can hold several at once and re-runs pay the load cost only once.
_ENCODERS: dict[str, object] = {}


def _safe_name(model_name: str) -> str:
    """Make a model id safe to use in a filename (e.g. 'BAAI/bge-large' has a slash)."""
    return model_name.replace("/", "__")


def _encoder(model_name: str = MODEL_NAME):
    """Load a frozen sentence encoder once per name and reuse it.

    A raw masked-LM checkpoint (e.g. LegalBERT) is not a sentence-transformer; passing
    its id here makes sentence-transformers wrap it with a MEAN-pooling head, which is
    exactly the "frozen LegalBERT as features" setup we want to test. Imported inside
    the function so importing embed_texts() elsewhere does not force a model load.
    """
    if model_name not in _ENCODERS:
        from sentence_transformers import SentenceTransformer

        enc = SentenceTransformer(model_name)
        print(f"loaded encoder {model_name} on device={enc.device}")
        _ENCODERS[model_name] = enc
    return _ENCODERS[model_name]


def _chunks(text: str) -> list[str]:
    """Split into up to CHUNK_CAP word-windows of WORDS_PER_CHUNK words.

    Word windows (not token windows) keep this simple and readable; WORDS_PER_CHUNK is
    chosen with margin so a full window stays under the encoder's token limit and is not
    silently truncated. An empty document yields one empty chunk so its row still maps
    to a vector (it will pool to a near-zero vector, which is the honest representation
    of "no text").
    """
    words = text.split()
    if not words:
        return [""]
    windows = [
        " ".join(words[i : i + WORDS_PER_CHUNK])
        for i in range(0, len(words), WORDS_PER_CHUNK)
    ]
    return windows[:CHUNK_CAP]


def embed_texts(texts: list[str], model_name: str = MODEL_NAME) -> np.ndarray:
    """Turn documents into one L2-normalized vector each (chunk + mean-pool).

    All chunks across all documents are encoded in ONE batched pass (throughput), then
    regrouped and mean-pooled per document. Pooling before normalizing, then normalizing
    the pooled vector, gives a unit-length document embedding, the natural input for a
    linear classifier. The output dimensionality follows the chosen encoder (768 for
    mpnet, 1024 for bge-large, etc.).
    """
    encoder = _encoder(model_name)

    chunks_per_doc = [_chunks(t) for t in texts]
    flat = [c for doc in chunks_per_doc for c in doc]
    total_chunks = len(flat)
    print(f"embedding {len(texts)} docs -> {total_chunks} chunks "
          f"(cap {CHUNK_CAP}/doc, {WORDS_PER_CHUNK} words/chunk)")

    vecs = encoder.encode(
        flat, batch_size=ENCODE_BATCH, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=False,
    )

    out = np.empty((len(texts), vecs.shape[1]), dtype=np.float32)
    cursor = 0
    for i, doc_chunks in enumerate(chunks_per_doc):
        n = len(doc_chunks)
        pooled = vecs[cursor : cursor + n].mean(axis=0)
        cursor += n
        norm = np.linalg.norm(pooled)
        out[i] = pooled / norm if norm > 0 else pooled
    return out


def embed_rows_cached(rows: list[dict], cache_key: str,
                      model_name: str = MODEL_NAME) -> np.ndarray:
    """Embed rows, caching per (split, encoder, chunking) so we pay the cost only once.

    The cache stores the vectors alongside the doc_ids they were built from, and the
    filename carries the encoder name AND the chunking parameters (words/chunk, cap).
    A doc_id mismatch (data rebuilt, rows reordered) recomputes; and because the
    chunking is in the KEY, changing WORDS_PER_CHUNK / CHUNK_CAP writes to a different
    file rather than silently returning vectors built under the old chunking. (An
    earlier version keyed only on split+encoder and claimed a stale cache "can never
    silently feed wrong vectors", which was false precisely for a chunking change.)
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{cache_key}.{_safe_name(model_name)}.w{WORDS_PER_CHUNK}.c{CHUNK_CAP}.npz"
    ids = [r["doc_id"] for r in rows]

    if path.exists():
        cached = np.load(path, allow_pickle=True)
        if list(cached["doc_ids"]) == ids:
            print(f"cache hit: {path.name} ({len(ids)} docs)")
            return cached["vectors"]
        print(f"cache stale ({path.name}), recomputing")

    vectors = embed_texts([r["text"] for r in rows], model_name)
    np.savez(path, vectors=vectors, doc_ids=np.array(ids, dtype=object))
    print(f"cached: {path.name}")
    return vectors


def _load(split: str) -> list[dict]:
    path = DATA_DIR / f"{split}.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _classifier() -> LogisticRegression:
    """Same classifier config as baseline.py, on dense features instead of sparse."""
    return LogisticRegression(max_iter=1000, class_weight="balanced")


def main() -> None:
    train = _load("train")
    test = _load("test")
    labels = sorted({r["type"] for r in train})
    print(f"train={len(train)}  test={len(test)}  classes={len(labels)}")

    x_train = embed_rows_cached(train, "train")
    x_test = embed_rows_cached(test, "test")
    y_train = [r["type"] for r in train]
    y_test = [r["type"] for r in test]

    clf = _classifier()
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)

    macro = float(f1_score(y_test, preds, labels=labels, average="macro"))
    print(f"\n===== embeddings probe ({MODEL_NAME}) =====")
    print(f"in-distribution macro-F1: {macro:.3f}   (TF-IDF baseline ~0.977 as of "
          "2026-08-24; run baseline.py for the current number)")
    print(classification_report(y_test, preds, labels=labels, digits=3))
    print("NOTE: in-distribution parity is expected and is NOT the question. The real")
    print("test is generalization (ip cross-source, OOS), measured next in step 2b.")


if __name__ == "__main__":
    main()
