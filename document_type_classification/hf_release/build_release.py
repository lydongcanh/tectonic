"""Build (and validate) the Hugging Face model bundle from the trained artifact.

Run this ONLY when the model changes. It bakes the trained logistic-regression head into
the safe `skops` format, writes the config + metrics that ship with it, and renders the
result charts for the model card. Publishing is separate (the GitHub Action / `hf upload`);
this just refreshes the bundle, which you then commit.

    poetry run python document_type_classification/hf_release/build_release.py

The bundle it writes to ./tectonic-doctype/ contains NO code: config.json, classifier.skops,
metrics.json, two PNG charts, README.md, requirements.txt. The usage snippet lives in the
README. To guard against that snippet drifting from how the model was trained, `_validate`
runs the SAME embedding recipe the snippet documents and checks it reproduces the trained
model's predictions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import certifi

# WARP leaves SSL_CERT_FILE pointing at its own single-cert bundle even when off, breaking
# the httpx-based Hub client; point at certifi's full public roots (full verification).
os.environ["SSL_CERT_FILE"] = certifi.where()
# The encoder was already downloaded during training, so validation needs no network.
# Force cache-only loads so a WARP re-enable (genuine TLS interception) can't break the build.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import joblib
import matplotlib
import numpy as np
import skops.io as sio

matplotlib.use("Agg")  # headless: render to file, never open a window
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]  # .../tectonic/
STAGE = _HERE / "tectonic-doctype"

TRAINED_MODEL = _REPO_ROOT / "artifacts/document_type/embedding_classifier.model.joblib"
TRAINED_META = _REPO_ROOT / "artifacts/document_type/embedding_classifier.json"
TEST_JSONL = _REPO_ROOT / "data/document_type/test.jsonl"
TEST_VECS = _REPO_ROOT / "artifacts/document_type/embeddings/test.all-mpnet-base-v2.npz"

ENCODER_ID = "sentence-transformers/all-mpnet-base-v2"
WORDS_PER_CHUNK = 250
CHUNK_CAP = 12

# The generalization + bake-off numbers (from evaluation/); kept here so the bundle is
# self-documenting and the charts and metrics.json agree.
GEN = {"macro_f1_in_dist": (0.940, 0.968), "ip_x_source": (0.628, 0.488),
       "ip_control": (0.837, 0.628), "oos_confidence": (0.85, 0.48)}  # (embeddings, tfidf)


def _embed(texts: list[str], encoder) -> np.ndarray:
    """The published embedding recipe: chunk into word windows, embed, mean-pool, L2-norm.
    Kept identical to the README usage snippet so `_validate` guards against drift."""
    def chunks(t):
        w = t.split()
        return ([" ".join(w[i:i + WORDS_PER_CHUNK]) for i in range(0, len(w), WORDS_PER_CHUNK)]
                or [""])[:CHUNK_CAP]

    per_doc = [chunks(t) for t in texts]
    flat = [c for doc in per_doc for c in doc]
    vecs = encoder.encode(flat, convert_to_numpy=True, normalize_embeddings=False)
    out = np.empty((len(texts), vecs.shape[1]), dtype=np.float32)
    cur = 0
    for i, doc in enumerate(per_doc):
        pooled = vecs[cur:cur + len(doc)].mean(axis=0)
        cur += len(doc)
        n = np.linalg.norm(pooled)
        out[i] = pooled / n if n > 0 else pooled
    return out


def _chart_vs_baseline() -> None:
    labels = ["macro-F1\n(in-dist)", "IP cross-\nsource recall",
              "IP control\nrecall", "out-of-source\nconfidence"]
    emb = [GEN["macro_f1_in_dist"][0], GEN["ip_x_source"][0], GEN["ip_control"][0], GEN["oos_confidence"][0]]
    tfidf = [GEN["macro_f1_in_dist"][1], GEN["ip_x_source"][1], GEN["ip_control"][1], GEN["oos_confidence"][1]]

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.2))
    b1 = ax.bar(x - w / 2, emb, w, label="embeddings (this model)", color="#2563eb")
    b2 = ax.bar(x + w / 2, tfidf, w, label="TF-IDF baseline", color="#9ca3af")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Embeddings vs TF-IDF baseline\n(TF-IDF wins in-distribution; embeddings generalize better)")
    ax.set_xticks(x, labels, fontsize=9)
    ax.legend(loc="lower right", fontsize=9)
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    fig.tight_layout()
    fig.savefig(STAGE / "results_vs_baseline.png", dpi=120)
    plt.close(fig)


def _chart_confusion(meta: dict) -> None:
    labels = meta["labels"]
    m = np.array(meta["confusion_matrix"], dtype=float)
    short = [l.replace("_agreement", "").replace("_", " ") for l in labels]

    fig, ax = plt.subplots(figsize=(6.8, 6))
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(labels)), short, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), short, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix (held-out test)")
    thresh = m.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            if m[i, j]:
                ax.text(j, i, str(int(m[i, j])), ha="center", va="center", fontsize=8,
                        color="white" if m[i, j] > thresh else "#111")
    fig.tight_layout()
    fig.savefig(STAGE / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def _build() -> dict:
    STAGE.mkdir(parents=True, exist_ok=True)
    clf = joblib.load(TRAINED_MODEL)
    meta = json.loads(TRAINED_META.read_text())

    head_path = STAGE / "classifier.skops"
    sio.dump(clf, head_path)
    trusted = sorted(sio.get_untrusted_types(file=head_path))

    config = {
        "model_type": "sentence_embedding_mean_pooled + logistic_regression",
        "encoder": ENCODER_ID, "words_per_chunk": WORDS_PER_CHUNK,
        "chunk_cap": CHUNK_CAP, "labels": meta["labels"], "trusted_types": trusted,
    }
    (STAGE / "config.json").write_text(json.dumps(config, indent=2))

    metrics = {
        "model": "embeddings (all-mpnet-base-v2) + logistic regression",
        "in_distribution": {
            "macro_f1": round(meta["macro_f1"], 3),
            "macro_f1_ci95": [round(x, 3) for x in meta["macro_f1_ci"]],
            "per_class_f1": {c: round(meta["report"][c]["f1-score"], 3) for c in meta["labels"]},
            "n_test": meta["n_test"],
        },
        "generalization_vs_tfidf": {
            "macro_f1_in_distribution": {"embeddings": 0.940, "tfidf": 0.968},
            "ip_cross_source_recall": {"embeddings": 0.628, "tfidf": 0.488},
            "ip_control_recall_both_sources": {"embeddings": 0.837, "tfidf": 0.628},
            "out_of_source_confidence": {"embeddings": 0.85, "tfidf": 0.48},
        },
        "encoder_bakeoff": {
            "columns": ["in_dist_macro_f1", "ip_x_source_recall", "oos_mean_conf"],
            "all-mpnet-base-v2 (chosen)": [0.940, 0.628, 0.85],
            "bge-large-en-v1.5": [0.940, 0.651, 0.60],
            "legal-bert-base-uncased (frozen)": [0.880, 0.581, 0.37],
        },
    }
    (STAGE / "metrics.json").write_text(json.dumps(metrics, indent=2))

    _chart_vs_baseline()
    _chart_confusion(meta)
    print(f"built bundle at {STAGE}/ (config, classifier.skops, metrics, 2 charts)")
    return meta


def _validate(n: int = 25) -> None:
    from sentence_transformers import SentenceTransformer

    rows = [json.loads(l) for l in TEST_JSONL.read_text().splitlines() if l.strip()][:n]
    cached = np.load(TEST_VECS, allow_pickle=True)
    ids = list(cached["doc_ids"])
    clf = joblib.load(TRAINED_MODEL)

    ref = [clf.predict(cached["vectors"][ids.index(r["doc_id"]):ids.index(r["doc_id"]) + 1])[0]
           for r in rows]
    encoder = SentenceTransformer(ENCODER_ID)
    cand = list(clf.predict(_embed([r["text"] for r in rows], encoder)))

    agree = sum(a == b for a, b in zip(ref, cand))
    print(f"validation: {agree}/{len(rows)} recipe predictions match the trained model")
    if agree != len(rows):
        raise SystemExit("MISMATCH: the published recipe differs from training")
    print("OK: the documented usage recipe reproduces the trained model.")


def main() -> None:
    _build()
    _validate()


if __name__ == "__main__":
    main()
