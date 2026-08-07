"""On-demand deep inspection of the baseline's learned features, for one class.

`baseline.py` logs only the top 15 words per class on every run. That is the right
altitude for a routine glass-box check: a leaked artifact strong enough to hurt
generalisation almost always ranks near the very top, so 15 reliably surfaces it.

But the model actually weighs ~290k features, and their coefficients trail off
gently rather than falling off a cliff, so 15 is a thin slice when you want to
AUDIT one class for a *distributed* bias. A class built mostly from one filer shows
a whole cluster of tell-tale words (for our ip_agreement: beijing, sina, weibo, ...)
and only the first of them may sit in the top 15.

This script loads the saved model and lets you look as deep as you want, on demand,
without making every training run verbose:

    # the top 40 words driving one class (rank 1 = strongest push toward it)
    python inspect_features.py ip_agreement --top 40

    # hunt a suspected bias cluster: every feature containing "beijing", each with
    # its coefficient and its rank among the class's features
    python inspect_features.py ip_agreement --grep beijing

Coefficient = the LogReg weight: how strongly a word pushes a document toward this
class. A positive weight pushes toward it, negative pushes away. Rank orders words
by coefficient (highest first), so rank 1 is the single strongest word for the class.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]  # .../tectonic/
MODEL_PATH = _REPO_ROOT / "artifacts/document_type/baseline.model.joblib"


def _load() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (feature_names, class_names, coefficient_matrix) from the saved model.

    The coefficient matrix has one row per class (one-vs-rest) and one column per
    feature, so `coef[c][f]` is the weight of feature f for class c.
    """
    pipe = joblib.load(MODEL_PATH)
    tfidf = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    return tfidf.get_feature_names_out(), clf.classes_, clf.coef_


def _class_row(classes: np.ndarray, coef: np.ndarray, cls: str) -> np.ndarray:
    """The coefficient row for one class, or a helpful exit if the name is unknown.

    Guard: this helper assumes the multiclass model, where there is exactly one
    coefficient row per class. If that ever stops holding (e.g. a 2-class model,
    where sklearn stores a single shared row), fail loudly rather than silently
    reading the wrong row.
    """
    if coef.shape[0] != len(classes):
        raise SystemExit(
            "this helper assumes the multiclass model (one coef row per class); "
            f"got {coef.shape[0]} rows for {len(classes)} classes."
        )
    names = list(classes)
    if cls not in names:
        raise SystemExit(f"unknown class {cls!r}; choose one of: {', '.join(names)}")
    return coef[names.index(cls)]


def _ranks(row: np.ndarray) -> dict[int, int]:
    """Map each feature index to its rank (1 = highest coefficient) within the class."""
    order = np.argsort(row)[::-1]
    return {int(i): rank for rank, i in enumerate(order, 1)}


def show_top(names: np.ndarray, row: np.ndarray, cls: str, k: int) -> None:
    """Print the k highest-coefficient features for the class."""
    order = np.argsort(row)[::-1][:k]
    print(f"top {k} features for {cls}  (rank   coef   word):")
    for rank, i in enumerate(order, 1):
        print(f"  {rank:4}  {row[i]:+.3f}  {names[i]}")


def show_grep(names: np.ndarray, row: np.ndarray, cls: str, needle: str) -> None:
    """Print every feature whose text contains `needle`, highest coefficient first.

    This is the bias-cluster hunter: give it a suspected leak word and it shows the
    whole family (unigrams and bigrams) with each one's weight and rank, so you can
    see how much of the class's signal is really that one confound.
    """
    rank_of = _ranks(row)
    needle = needle.lower()
    hits = [i for i in range(len(names)) if needle in names[i].lower()]
    hits.sort(key=lambda i: row[i], reverse=True)
    print(f"features containing {needle!r} for {cls}  (rank   coef   word):")
    if not hits:
        print("  (none)")
        return
    for i in hits:
        print(f"  {rank_of[i]:4}  {row[i]:+.3f}  {names[i]}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Inspect the baseline's learned features for one class."
    )
    ap.add_argument("cls", help="class name, e.g. ip_agreement")
    ap.add_argument("--top", type=int, metavar="N", default=40,
                    help="show the top N features (default 40); ignored if --grep is used")
    ap.add_argument("--grep", metavar="WORD",
                    help="instead, show every feature containing WORD")
    args = ap.parse_args()

    names, classes, coef = _load()
    row = _class_row(classes, coef, args.cls)

    if args.grep:
        show_grep(names, row, args.cls, args.grep)
    else:
        show_top(names, row, args.cls, args.top)


if __name__ == "__main__":
    main()
