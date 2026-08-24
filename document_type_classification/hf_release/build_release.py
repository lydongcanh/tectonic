"""Build (and validate) the production document-type model + its Hugging Face bundle.

The production model is multilingual: a logistic-regression head on frozen BAAI/bge-m3
embeddings (100+ languages, 8192-token context). It replaces the earlier English-only mpnet
model and adds cross-lingual support. It also scored higher on English (0.957 vs mpnet's
0.940), but treat that as suggestive, not proof of a better encoder: the two runs used
DIFFERENT chunking (bge-m3 embeds 2000-word chunks, mpnet used 250-word chunks), so the gain
is confounded with chunk size. Both numbers also predate the 2026-08-24 leak-free dataset
rebuild, so re-embed before quoting them. Run this only when the model changes:

    poetry run python document_type_classification/hf_release/build_release.py

It embeds the corpus with bge-m3 (large coherent chunks that use the long context), retrains
the head, writes the bundle to ./tectonic-doctype/ (config, classifier.skops, metrics, a
confusion-matrix chart, README), and self-checks. Embedding is a one-time cost, cached under
artifacts/; publishing is separate (the GitHub Action / hf upload). bge-m3 is downloaded on
first run (WARP OFF); afterwards this runs fully offline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()  # WARP leftover-env-var guard (full verification)

import matplotlib
import numpy as np
import skops.io as sio
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
DATA = _REPO_ROOT / "data/document_type"
CACHE = _REPO_ROOT / "artifacts/document_type/embeddings"
STAGE = _HERE / "tectonic-doctype"

ENCODER = "BAAI/bge-m3"
# Use bge-m3's long context: embed large, coherent spans so we mean-pool as few, as-complete
# chunks as possible (a document is usually 1-3 chunks). One-time ~90-110 min on MPS, cached.
MAX_SEQ = 8192
WORDS_PER_CHUNK = 2000
CHUNK_CAP = 6
BATCH = 8


def _safe(name: str) -> str:
    return name.replace("/", "__")


def _rows(name: str) -> list[dict]:
    return [json.loads(l) for l in (DATA / name).read_text().splitlines() if l.strip()]


def _chunks(text: str) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    return [" ".join(words[i:i + WORDS_PER_CHUNK])
            for i in range(0, len(words), WORDS_PER_CHUNK)][:CHUNK_CAP]


_ENCODER = None  # loaded lazily, only if we actually have to embed (see _get_encoder)


def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer

        _ENCODER = SentenceTransformer(ENCODER)
        _ENCODER.max_seq_length = MAX_SEQ
        print(f"loaded {ENCODER} on device={_ENCODER.device}, max_seq_length={_ENCODER.max_seq_length}")
    return _ENCODER


def _pool(chunk_vecs: np.ndarray) -> np.ndarray:
    """Mean-pool the first CHUNK_CAP chunk vectors, then L2-normalise (the production
    document representation). Identical whether the chunks come from a fresh encode or
    from the all-chunks cache, so the two paths produce the same vectors."""
    pooled = chunk_vecs[:CHUNK_CAP].mean(axis=0)
    n = np.linalg.norm(pooled)
    return (pooled / n if n > 0 else pooled).astype(np.float32)


def _from_allchunks(rows: list[dict]) -> np.ndarray | None:
    """Reuse the all-chunks cache the pooling ablation built (every chunk of every doc,
    same encoder + 2000-word chunking) by pooling the first CHUNK_CAP chunks, instead of
    re-embedding. Returns None if that cache is absent or does not cover these docs, in
    which case the caller falls back to embedding.
    """
    path = CACHE / f"allchunks.{_safe(ENCODER)}.w{WORDS_PER_CHUNK}.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    by_id = dict(zip(z["doc_ids"], z["per_doc"]))
    if any(r["doc_id"] not in by_id for r in rows):
        return None
    print(f"pooling first-{CHUNK_CAP} from {path.name} (no re-embed)")
    return np.array([_pool(by_id[r["doc_id"]]) for r in rows], dtype=np.float32)


def _embed_cached(rows: list[dict], split: str) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    # Key on encoder AND chunking (words/chunk, cap): changing the chunking must write a
    # new cache file, not silently reuse vectors pooled under the old chunking. The doc_id
    # check below additionally recomputes when the split's documents change.
    path = CACHE / f"{split}.{_safe(ENCODER)}.w{WORDS_PER_CHUNK}.c{CHUNK_CAP}.npz"
    ids = [r["doc_id"] for r in rows]
    if path.exists():
        z = np.load(path, allow_pickle=True)
        if list(z["doc_ids"]) == ids:
            print(f"cache hit: {path.name}")
            return z["vectors"]

    out = _from_allchunks(rows)
    if out is None:
        encoder = _get_encoder()
        per_doc = [_chunks(r["text"]) for r in rows]
        flat = [c for doc in per_doc for c in doc]
        print(f"embedding {len(rows)} docs -> {len(flat)} chunks with {ENCODER}")
        vecs = encoder.encode(flat, batch_size=BATCH, show_progress_bar=True,
                              convert_to_numpy=True, normalize_embeddings=False)
        out = np.empty((len(rows), vecs.shape[1]), dtype=np.float32)
        cur = 0
        for i, doc in enumerate(per_doc):
            out[i] = _pool(vecs[cur:cur + len(doc)])
            cur += len(doc)

    np.savez(path, vectors=out, doc_ids=np.array(ids, dtype=object))
    print(f"cached: {path.name}")
    return out


def _confusion_chart(labels: list[str], y_true: list[str], preds) -> None:
    m = confusion_matrix(y_true, preds, labels=labels).astype(float)
    short = [l.replace("_agreement", "").replace("_", " ") for l in labels]
    fig, ax = plt.subplots(figsize=(6.8, 6))
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(labels)), short, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), short, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix (English held-out test)")
    thresh = m.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            if m[i, j]:
                ax.text(j, i, str(int(m[i, j])), ha="center", va="center", fontsize=8,
                        color="white" if m[i, j] > thresh else "#111")
    fig.tight_layout()
    fig.savefig(STAGE / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def _card(macro: float, per_class: dict) -> str:
    rows = "\n".join(f"- `{k}`: {v:.3f}" for k, v in per_class.items())
    return f"""---
license: cc-by-4.0
pipeline_tag: text-classification
tags:
  - text-classification
  - legal
  - contracts
  - multilingual
base_model: BAAI/bge-m3
language:
  - en
library_name: sklearn
---

# Document Type Classifier

Classifies a legal / deal document into one of nine types from its text. A
logistic-regression head on frozen [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3)
embeddings, so it is **multilingual** (100+ languages, 8192-token context) and embeds whole
documents.

**Labels:** `acquisition_agreement`, `commercial_agreement`, `constitutional`,
`employment_agreement`, `financial_statements`, `financing_agreement`, `ip_agreement`,
`lease_agreement`, `nda` (`commercial_agreement` is the catch-all for "some other contract").

## Evaluation

**Measured (English held-out test): macro-F1 {macro:.3f}.** This is the only *labelled* test
set that exists (all training data is English: EDGAR / CUAD / ContractNLI), so it is the only
computed benchmark. Per-class F1:

{rows}

![Confusion matrix](confusion_matrix.png)

**Other languages: a capability, not a measured result.** The head is trained only on
English, and there is no labelled non-English test set, so a score for other languages cannot
be reported honestly. Other languages work *zero-shot* through bge-m3's shared multilingual
space, which performs well on public multilingual benchmarks but is **unvalidated for this
task**. Treat non-English predictions as usable but unverified. Confidence is not calibrated,
so set any accept/escalate threshold empirically, and note training documents are
US-filing-style, so non-US document structures may differ.

## Usage

```python
import numpy as np, skops.io as sio
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

REPO = "lydongcanh/tectonic-doctype"
enc = SentenceTransformer("BAAI/bge-m3")
enc.max_seq_length = {MAX_SEQ}
head = sio.load(hf_hub_download(REPO, "classifier.skops"), trusted=[])

def classify(text: str):
    words = text.split()
    chunks = [" ".join(words[i:i+{WORDS_PER_CHUNK}]) for i in range(0, len(words), {WORDS_PER_CHUNK})][:{CHUNK_CAP}] or [""]
    v = enc.encode(chunks).mean(0); v = v / np.linalg.norm(v)
    p = head.predict_proba([v])[0]; i = int(p.argmax())
    return {{"label": head.classes_[i], "confidence": float(p[i])}}
```

## Data & license

Built from CUAD (© The Atticus Project, CC BY 4.0), ContractNLI (CC BY 4.0), and SEC EDGAR
(public). Released under CC BY 4.0.
"""


def main() -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    train, test = _rows("train.jsonl"), _rows("test.jsonl")
    labels = sorted({r["type"] for r in train})
    # Reuses the all-chunks cache when present (pools first-CHUNK_CAP, no re-embed) and
    # only loads bge-m3 to embed if that cache is missing.
    x_train = _embed_cached(train, "train")
    x_test = _embed_cached(test, "test")
    y_train, y_test = [r["type"] for r in train], [r["type"] for r in test]

    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(x_train, y_train)
    preds = clf.predict(x_test)
    macro = float(f1_score(y_test, preds, labels=labels, average="macro"))
    report = classification_report(y_test, preds, labels=labels, digits=3, output_dict=True)
    per_class = {c: report[c]["f1-score"] for c in labels}

    print(f"\n===== production model ({ENCODER}) =====")
    print(f"English macro-F1: {macro:.3f}")
    print(classification_report(y_test, preds, labels=labels, digits=3))

    head_path = STAGE / "classifier.skops"
    sio.dump(clf, head_path)
    trusted = sorted(sio.get_untrusted_types(file=head_path))
    config = {"model_type": "sentence_embedding_mean_pooled + logistic_regression",
              "encoder": ENCODER, "words_per_chunk": WORDS_PER_CHUNK,
              "chunk_cap": CHUNK_CAP, "max_seq_length": MAX_SEQ,
              "labels": labels, "trusted_types": trusted}
    (STAGE / "config.json").write_text(json.dumps(config, indent=2))
    (STAGE / "metrics.json").write_text(json.dumps(
        {"encoder": ENCODER, "in_distribution_english": {
            "macro_f1": round(macro, 3),
            "per_class_f1": {c: round(v, 3) for c, v in per_class.items()},
            "n_test": len(test)}}, indent=2))
    _confusion_chart(labels, y_test, preds)
    (STAGE / "README.md").write_text(_card(macro, per_class))

    reloaded = sio.load(head_path, trusted=trusted)
    macro2 = float(f1_score(y_test, reloaded.predict(x_test), labels=labels, average="macro"))
    assert abs(macro - macro2) < 1e-9, "reloaded head disagrees"
    print(f"\nbuilt bundle at {STAGE}/ ; reloaded-head macro-F1 {macro2:.3f} (matches)")


if __name__ == "__main__":
    main()
