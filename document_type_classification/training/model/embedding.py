"""Production embedding engine: turn a document into one dense semantic vector.

This is the representation our v1 production classifier uses instead of TF-IDF. The
choice is evidence-based: the embeddings probe (evaluation/) showed a frozen sentence
encoder generalizes markedly better across corpora than TF-IDF (ip cross-source recall
0.63 vs 0.49, OOS confidence far higher), and the encoder bake-off picked THIS encoder:

  * all-mpnet-base-v2 tied the strongest modern general encoder (bge-large) on every
    generalization metric within noise, while being more confident on out-of-source
    docs (better for the cascade's confidence gate), 3x smaller, and lower-dimensional.
  * Frozen LegalBERT lost on every metric (a raw masked-LM makes poor pooled document
    vectors), so domain pre-training without sentence-training was not worth it.

Two design choices keep this honest and match the probe exactly:

  * Fair length. A neural encoder's token window (~384 tokens for mpnet) is far shorter
    than a contract, so we CHUNK each document into word windows, embed each, and
    MEAN-POOL into one document vector (capped at CHUNK_CAP windows to bound cost). This
    lets the encoder see roughly the whole document, the way TF-IDF does.
  * Reproducibility. Vectors are cached per (split, encoder) under artifacts/, keyed by
    doc_id, so training is instant on re-runs and a stale cache can never feed wrong
    vectors into a fit (a doc_id mismatch forces recompute).

The trade-off we accept: unlike TF-IDF, embeddings are not glass-box (we cannot read a
coefficient as a word). The TF-IDF baseline (model/baseline.py) stays as the
interpretability tool and the reference bar.
"""

from __future__ import annotations

import os
from pathlib import Path

import certifi

# Cloudflare WARP sets SSL_CERT_FILE to its own single-cert bundle and leaves it set even
# when WARP is off, which makes httpx (the HuggingFace downloader) reject legitimate public
# certs. Point the process at certifi's full public-root bundle: FULL verification against
# the standard roots, not a weakening of it. Must run before any SSL context is created.
os.environ["SSL_CERT_FILE"] = certifi.where()

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]  # .../tectonic/
CACHE_DIR = _REPO_ROOT / "artifacts/document_type/embeddings"

MODEL_NAME = "all-mpnet-base-v2"  # the bake-off winner; see module docstring
WORDS_PER_CHUNK = 250   # ~325 tokens, safely under mpnet's 384-token window
CHUNK_CAP = 12          # at most 12 chunks/doc (~3000 words); bounds embedding cost
ENCODE_BATCH = 64       # chunks per forward pass; throughput only, not a result knob

_ENCODER = None


def _encoder():
    """Load the frozen encoder once and reuse it (imported lazily so importing this
    module stays cheap until embedding actually happens)."""
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer

        _ENCODER = SentenceTransformer(MODEL_NAME)
        print(f"loaded encoder {MODEL_NAME} on device={_ENCODER.device}")
    return _ENCODER


def _chunks(text: str) -> list[str]:
    """Up to CHUNK_CAP word-windows of WORDS_PER_CHUNK words. An empty document yields
    one empty chunk so its row still maps to a (near-zero) vector."""
    words = text.split()
    if not words:
        return [""]
    windows = [
        " ".join(words[i : i + WORDS_PER_CHUNK])
        for i in range(0, len(words), WORDS_PER_CHUNK)
    ]
    return windows[:CHUNK_CAP]


def embed_texts(texts: list[str]) -> np.ndarray:
    """Turn documents into one L2-normalized vector each (chunk + mean-pool).

    All chunks across all documents are encoded in one batched pass, then regrouped and
    mean-pooled per document; the pooled vector is L2-normalized so it is unit length,
    the natural input for a linear classifier.
    """
    encoder = _encoder()

    chunks_per_doc = [_chunks(t) for t in texts]
    flat = [c for doc in chunks_per_doc for c in doc]
    print(f"embedding {len(texts)} docs -> {len(flat)} chunks "
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


def embed_rows_cached(rows: list[dict], cache_key: str) -> np.ndarray:
    """Embed rows, caching per split so the encoder cost is paid only once.

    The cache stores vectors alongside the doc_ids they were built from; if the saved
    doc_ids do not match the current rows exactly, we recompute, so a stale cache can
    never silently feed wrong vectors into a result.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{cache_key}.{MODEL_NAME}.npz"
    ids = [r["doc_id"] for r in rows]

    if path.exists():
        cached = np.load(path, allow_pickle=True)
        if list(cached["doc_ids"]) == ids:
            print(f"cache hit: {path.name} ({len(ids)} docs)")
            return cached["vectors"]
        print(f"cache stale ({path.name}), recomputing")

    vectors = embed_texts([r["text"] for r in rows])
    np.savez(path, vectors=vectors, doc_ids=np.array(ids, dtype=object))
    print(f"cached: {path.name}")
    return vectors
