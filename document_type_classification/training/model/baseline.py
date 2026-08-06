"""Baseline document-type classifier: TF-IDF + Logistic Regression.

This is the number any fancier model must beat. It is deliberately simple and
INSPECTABLE:

  * TF-IDF turns each document into a vector of weighted word counts. "TF" is how
    often a word appears in this document; "IDF" downweights words common across
    all documents (like "the"), so distinctive words dominate.
  * Logistic Regression then learns one weight per word per class. To predict, it
    adds the weights of the words present and picks the highest-scoring class.

Because it is linear, we can read off which words drive each prediction (the
top-words readout), our check that it learned real document type rather than
dataset artefacts.

This lives in `model/` and depends on the data phase only through its OUTPUT
FILES (`data/document_type/train.jsonl` and `test.jsonl`), not its code. The two
phases share a file format, not imports, which keeps them cleanly separate.

Outputs, all under artifacts/ so they survive the console:
  - the trained pipeline, saved as <run>.model.joblib (the actual usable model),
  - full metrics as <run>.json, and a one-line summary appended to runs.jsonl.

Run from the repo root:
    poetry run python document_type_classification/training/model/baseline.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

DATA_DIR = Path("data/document_type")
ARTIFACT_DIR = Path("artifacts/document_type")
TOP_K = 15

# Model config, kept as plain data for the saved report. The pipeline below is
# built with these same values (mirrored, not unpacked, so the types stay clear).
CONFIG = {
    "vectorizer": {"sublinear_tf": True, "ngram_range": [1, 2], "min_df": 2},
    "clf": {"max_iter": 1000, "class_weight": "balanced"},
}


def _build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def _load(split: str) -> tuple[list[str], list[str]]:
    """Read the data phase's JSONL output into (texts, labels).

    We read the file directly rather than import the data-prep code: the model
    depends on the dataset FORMAT, not on how it was built.
    """
    path = DATA_DIR / f"{split}.jsonl"
    texts: list[str] = []
    labels: list[str] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        texts.append(row["text"])
        labels.append(row["type"])
    return texts, labels


def _top_features(pipe: Pipeline, k: int) -> dict[str, list[str]]:
    """The k words pushing hardest toward each class.

    For a 2-class linear model there is a single coefficient vector: the most
    positive weights favour classes_[1], the most negative favour classes_[0].
    """
    vec: TfidfVectorizer = pipe.named_steps["tfidf"]
    clf: LogisticRegression = pipe.named_steps["clf"]
    names = np.asarray(vec.get_feature_names_out())
    coef = clf.coef_[0]
    order = np.argsort(coef)
    return {
        clf.classes_[0]: names[order[:k]].tolist(),         # most negative weights
        clf.classes_[1]: names[order[-k:][::-1]].tolist(),  # most positive weights
    }


def evaluate(
    name: str,
    x_train: list[str], y_train: list[str],
    x_test: list[str], y_test: list[str],
    labels: list[str],
) -> dict:
    """Train, evaluate, print, and save one run (model + metrics). Returns metrics."""
    pipe = _build_pipeline()
    pipe.fit(x_train, y_train)
    preds = pipe.predict(x_test)

    macro = float(f1_score(y_test, preds, labels=labels, average="macro"))
    report = classification_report(y_test, preds, labels=labels, digits=3, output_dict=True)
    matrix = confusion_matrix(y_test, preds, labels=labels).tolist()
    features = _top_features(pipe, TOP_K)

    print(f"\n===== run: {name} =====")
    print(f"macro-F1: {macro:.3f}")
    print(classification_report(y_test, preds, labels=labels, digits=3))
    print("confusion matrix (rows = true, cols = predicted)")
    print("labels:", labels)
    print(np.array(matrix))
    print("top words per class:")
    for cls, feats in features.items():
        print(f"  {cls:22} {', '.join(feats)}")

    metrics = {
        "run": name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {**CONFIG, "preprocessing": name},
        "n_train": len(x_train),
        "n_test": len(x_test),
        "labels": labels,
        "macro_f1": macro,
        "report": report,
        "confusion_matrix": matrix,
        "top_features": features,
    }
    _save(name, metrics, pipe)
    return metrics


def _save(name: str, metrics: dict, pipe: Pipeline) -> None:
    """Save the trained model, the full metrics, and a summary line."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, ARTIFACT_DIR / f"{name}.model.joblib")
    (ARTIFACT_DIR / f"{name}.json").write_text(json.dumps(metrics, indent=2))
    summary = {"run": name, "timestamp": metrics["timestamp"],
               "macro_f1": metrics["macro_f1"], "n_test": metrics["n_test"]}
    with (ARTIFACT_DIR / "runs.jsonl").open("a") as f:
        f.write(json.dumps(summary) + "\n")


def main() -> None:
    x_train, y_train = _load("train")
    x_test, y_test = _load("test")
    labels = sorted(set(y_train))
    print(f"train={len(x_train)}  test={len(x_test)}")

    evaluate("baseline", x_train, y_train, x_test, y_test, labels)
    print(f"\nsaved model + metrics to {ARTIFACT_DIR}/")


if __name__ == "__main__":
    main()
