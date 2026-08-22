"""v1 production classifier: frozen sentence embeddings + Logistic Regression.

This is the model we ship. It mirrors baseline.py's classifier and evaluation exactly
(same LogisticRegression config, same bootstrap confidence intervals, same held-out test
split) and changes ONE thing: the representation. Instead of TF-IDF's sparse word counts
it uses a dense semantic vector per document from a frozen sentence encoder (see
embedding.py for the encoder choice and the chunk/mean-pool method).

Why this and not TF-IDF: the evaluation probes showed embeddings generalize better across
corpora (the property that matters, because we deploy on non-EDGAR documents), while
TF-IDF wins only in-distribution, the metric we least trust. TF-IDF stays as the
baseline/interpretability reference in baseline.py; this is the production model.

What we save (all under artifacts/, gitignored):
  * embedding_classifier.model.joblib  - the trained LogReg (the classifier head only).
  * embedding_classifier.json          - metrics AND the metadata an inference wrapper
                                         needs to rebuild the full pipeline: the encoder
                                         name and the chunk/pool parameters. The 420MB
                                         encoder is referenced by NAME, not pickled.
  * runs.jsonl                         - one summary line appended per run.

Because the encoder is not pickled, loading the model for inference means: load this
LogReg, load the named encoder, embed with the same chunk/pool params. Those params live
in the metadata here so inference cannot silently drift from training.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from embedding import CHUNK_CAP, MODEL_NAME, WORDS_PER_CHUNK, embed_rows_cached

_REPO_ROOT = Path(__file__).resolve().parents[3]  # .../tectonic/
DATA_DIR = _REPO_ROOT / "data/document_type"
ARTIFACT_DIR = _REPO_ROOT / "artifacts/document_type"
RUN_NAME = "embedding_classifier"

BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 20260806
CI_ALPHA = 0.05

CLF_CONFIG = {"max_iter": 1000, "class_weight": "balanced"}


def _classifier() -> LogisticRegression:
    """Identical config to baseline.py, so only the representation differs."""
    return LogisticRegression(**CLF_CONFIG)


def _load(split: str) -> list[dict]:
    path = DATA_DIR / f"{split}.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _bootstrap_f1(
    y_true: list[str], y_pred: list[str], labels: list[str],
    n: int, seed: int, alpha: float,
) -> tuple[tuple[float, float], dict[str, tuple[float, float]]]:
    """Bootstrap CIs for macro-F1 AND each class's F1 from one set of resamples.

    Predictions are fixed; we resample the test rows with replacement and recompute F1
    each time, so the spread reflects how much the score would wobble on a slightly
    different sample from the SAME distribution. It says nothing about other sources.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(yt))
    macro = np.empty(n)
    per_class = np.empty((n, len(labels)))
    for i in range(n):
        pick = rng.choice(idx, size=len(idx), replace=True)
        f = f1_score(yt[pick], yp[pick], labels=labels, average=None, zero_division=0)
        per_class[i] = f
        macro[i] = f.mean()
    q = [alpha / 2, 1 - alpha / 2]
    macro_ci = tuple(float(v) for v in np.quantile(macro, q))
    class_ci = {
        lbl: tuple(float(v) for v in np.quantile(per_class[:, j], q))
        for j, lbl in enumerate(labels)
    }
    return macro_ci, class_ci


def _save(metrics: dict, clf: LogisticRegression) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, ARTIFACT_DIR / f"{RUN_NAME}.model.joblib")
    (ARTIFACT_DIR / f"{RUN_NAME}.json").write_text(json.dumps(metrics, indent=2))
    summary = {
        "run": RUN_NAME,
        "timestamp": metrics["timestamp"],
        "macro_f1": metrics["macro_f1"],
        "macro_f1_ci": metrics["macro_f1_ci"],
        "n_test": metrics["n_test"],
    }
    with (ARTIFACT_DIR / "runs.jsonl").open("a") as f:
        f.write(json.dumps(summary) + "\n")


def main() -> None:
    train = _load("train")
    test = _load("test")
    labels = sorted({r["type"] for r in train})
    print(f"train={len(train)}  test={len(test)}  classes={len(labels)}")

    x_train = embed_rows_cached(train, "train")   # cache hits from the probe runs
    x_test = embed_rows_cached(test, "test")
    y_train = [r["type"] for r in train]
    y_test = [r["type"] for r in test]

    clf = _classifier()
    clf.fit(x_train, y_train)
    preds = list(clf.predict(x_test))

    macro = float(f1_score(y_test, preds, labels=labels, average="macro"))
    (ci_lo, ci_hi), class_ci = _bootstrap_f1(
        y_test, preds, labels, BOOTSTRAP_N, BOOTSTRAP_SEED, CI_ALPHA
    )
    report = classification_report(y_test, preds, labels=labels, digits=3, output_dict=True)
    matrix = confusion_matrix(y_test, preds, labels=labels).tolist()

    print(f"\n===== run: {RUN_NAME} ({MODEL_NAME}) =====")
    print(f"macro-F1: {macro:.3f}   (95% CI {ci_lo:.3f}-{ci_hi:.3f})   baseline TF-IDF: 0.968")
    print(classification_report(y_test, preds, labels=labels, digits=3))
    print("per-class F1 95% CI (bootstrap; sampling wobble within THIS test set only):")
    for cls in labels:
        lo, hi = class_ci[cls]
        print(f"  {cls:22} [{lo:.3f}, {hi:.3f}]")
    print("confusion matrix (rows = true, cols = predicted)")
    print("labels:", labels)
    print(np.array(matrix))

    metrics = {
        "run": RUN_NAME,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "representation": {
            "kind": "sentence_embedding_mean_pooled",
            "encoder": MODEL_NAME,
            "words_per_chunk": WORDS_PER_CHUNK,
            "chunk_cap": CHUNK_CAP,
        },
        "clf": CLF_CONFIG,
        "n_train": len(train),
        "n_test": len(test),
        "labels": labels,
        "macro_f1": macro,
        "macro_f1_ci": [ci_lo, ci_hi],
        "per_class_f1_ci": {cls: list(class_ci[cls]) for cls in labels},
        "report": report,
        "confusion_matrix": matrix,
    }
    _save(metrics, clf)
    print(f"\nsaved model + metrics to {ARTIFACT_DIR}/")


if __name__ == "__main__":
    main()
