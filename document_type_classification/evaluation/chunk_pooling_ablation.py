"""Chunk-pooling ablation for the bge-m3 document-type model.

Question: is the production config (first 6 chunks of 2000 words, mean-pooled) the right
way to turn a long document into one vector, or would fewer/more chunks, the whole doc, or
a head+tail selection do as well or better? Document TYPE is usually front-loaded (title,
parties, recitals on page 1), so we expect little difference, but we measure instead of
guessing.

Design: pay the expensive embedding ONCE. We embed EVERY 2000-word chunk of every document
(no cap) with bge-m3 and cache the per-chunk vectors. Then each pooling strategy is just a
cheap function of those cached chunk vectors (a subset + mean + L2-normalise), so we compare
many strategies without re-embedding. NOTE this ablates the POOLING/COVERAGE only; the chunk
SIZE (2000 words vs a larger chunk that fills bge-m3's 8192-token context) is a different
embedding and would need its own pass, added only if coverage turns out to matter here.

Evaluation is a GROUP-AWARE 5-fold CV: near-duplicate documents are clustered (same as the
train/test split) and kept within one fold, so a near-twin never leaks across the CV boundary
and inflates a strategy's score. LogReg is the classifier (kept for its calibrated
probabilities, which the cascade needs; a separate check showed SVM is within noise).

Run (embedding is a one-time ~90+ min MPS cost, cached afterwards; bge-m3 must be cached, we
run offline):

    HF_HUB_OFFLINE=1 .venv/bin/python document_type_classification/evaluation/chunk_pooling_ablation.py

Smoke-test the wiring cheaply first (embed only N docs, skip CV):

    SMOKE=15 HF_HUB_OFFLINE=1 .venv/bin/python .../chunk_pooling_ablation.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()  # WARP leftover-env-var guard

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

_REPO_ROOT = Path(__file__).resolve().parents[2]  # .../tectonic/
DATA = _REPO_ROOT / "data/document_type/dataset.jsonl"
CACHE = _REPO_ROOT / "artifacts/document_type/embeddings"
# The cluster helper lives with the dataset code; reuse it so CV grouping matches the split.
import sys
sys.path.insert(0, str(_REPO_ROOT / "document_type_classification/training/dataset"))
from fingerprint import cluster  # noqa: E402

ENCODER = "BAAI/bge-m3"
MAX_SEQ = 8192
WORDS_PER_CHUNK = 2000   # so "first6" reproduces the current production config exactly
BATCH = 8
CLUSTER_THRESHOLD = 0.6  # identical to split.py, so CV folds respect the same near-dup groups
CV_FOLDS = 5
CV_SEED = 20260806


def _rows() -> list[dict]:
    return [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]


def _chunks(text: str) -> list[str]:
    """Every 2000-word window (NO cap), so the cache holds the whole document."""
    words = text.split()
    if not words:
        return [""]
    return [" ".join(words[i:i + WORDS_PER_CHUNK]) for i in range(0, len(words), WORDS_PER_CHUNK)]


def embed_all_chunks(rows: list[dict], smoke: bool) -> dict[str, np.ndarray]:
    """doc_id -> (n_chunks x dim) matrix of raw (un-pooled) chunk vectors, cached flat.

    The cache is keyed by encoder + chunk size + "allchunks". A smoke run embeds a few docs
    and never touches the real cache, so a partial run cannot masquerade as the full one.
    """
    from sentence_transformers import SentenceTransformer

    path = CACHE / f"allchunks.{ENCODER.replace('/', '__')}.w{WORDS_PER_CHUNK}.npz"
    ids = [r["doc_id"] for r in rows]
    if not smoke and path.exists():
        z = np.load(path, allow_pickle=True)
        if list(z["doc_ids"]) == ids:
            print(f"cache hit: {path.name}")
            return {d: v for d, v in zip(z["doc_ids"], z["per_doc"])}
        print("cache present but doc set changed; re-embedding")

    encoder = SentenceTransformer(ENCODER)
    encoder.max_seq_length = MAX_SEQ
    print(f"loaded {ENCODER} on {encoder.device}; embedding ALL chunks of {len(rows)} docs")

    per_doc_chunks = [_chunks(r["text"]) for r in rows]
    flat = [c for doc in per_doc_chunks for c in doc]
    print(f"{len(flat)} chunks total ({len(flat) / max(1, len(rows)):.1f}/doc avg)")
    vecs = encoder.encode(flat, batch_size=BATCH, show_progress_bar=True,
                          convert_to_numpy=True, normalize_embeddings=False)

    out: dict[str, np.ndarray] = {}
    cur = 0
    for r, doc_chunks in zip(rows, per_doc_chunks):
        n = len(doc_chunks)
        out[r["doc_id"]] = vecs[cur:cur + n].astype(np.float32)
        cur += n

    if not smoke:
        CACHE.mkdir(parents=True, exist_ok=True)
        per_doc = np.empty(len(rows), dtype=object)
        for i, r in enumerate(rows):
            per_doc[i] = out[r["doc_id"]]
        np.savez(path, per_doc=per_doc, doc_ids=np.array(ids, dtype=object))
        print(f"cached: {path.name}")
    return out


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# Each strategy maps a doc's (n_chunks x dim) matrix to ONE L2-normalised document vector.
STRATEGIES = {
    "first1 (page 1 only)":  lambda m: _norm(m[0]),
    "first3":                lambda m: _norm(m[:3].mean(0)),
    "first6 (PRODUCTION)":   lambda m: _norm(m[:6].mean(0)),
    "all (whole doc)":       lambda m: _norm(m.mean(0)),
    "head+tail (1+1)":       lambda m: _norm(np.vstack([m[:1], m[-1:]]).mean(0)),
    "head+tail (2+2)":       lambda m: _norm(np.vstack([m[:2], m[-2:]]).mean(0)),
}


def main() -> None:
    smoke = int(os.environ.get("SMOKE", "0"))
    rows = _rows()
    if smoke:
        rows = rows[:smoke]
        print(f"SMOKE MODE: {len(rows)} docs, will verify wiring only (no CV)")

    chunks = embed_all_chunks(rows, smoke=bool(smoke))
    labels = np.array([r["type"] for r in rows])

    if smoke:
        for name, fn in STRATEGIES.items():
            X = np.array([fn(chunks[r["doc_id"]]) for r in rows])
            print(f"  {name:22} -> X {X.shape}")
        print("wiring OK")
        return

    print("\nclustering for group-aware CV (near-dup families kept in one fold)...")
    groups = np.array(cluster([r["text"] for r in rows], CLUSTER_THRESHOLD))
    print(f"{len(set(groups))} groups over {len(rows)} docs\n")

    cv = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    print(f"group-aware {CV_FOLDS}-fold CV macro-F1 (LogReg) by pooling strategy:")
    for name, fn in STRATEGIES.items():
        X = np.array([fn(chunks[r["doc_id"]]) for r in rows])
        s = cross_val_score(clf, X, labels, groups=groups, cv=cv, scoring="f1_macro")
        print(f"  {name:22} {s.mean():.4f} ± {s.std():.4f}")


if __name__ == "__main__":
    main()
