"""Open-set probe: can an embedding-distance novelty signal flag "not one of my types"?

Our classifier is CLOSED-set: it always returns one of the nine labels, so an off-taxonomy
document gets a confident wrong answer. Production needs an "unknown" path. The idea here
does NOT require collecting "Other" data (that space is unbounded and unsamplable). Instead
we model what KNOWN looks like in embedding space and flag whatever falls far outside it.

Detector: the standard Mahalanobis out-of-distribution score. On the KNOWN classes' training
embeddings we estimate one mean per class and a single shared covariance (shrunk with
Ledoit-Wolf, since 768 dims dwarf the per-class counts). A document's novelty is its
Mahalanobis distance to the NEAREST known-class mean: small = looks like a known type, large
= unlike anything known. No "unknown" data is used to fit it.

Honest evaluation with NO new data: leave-one-class-out. Hold one class entirely out of
training and treat its documents as the "unknown" at test time; check the detector flags them
while still accepting the eight known classes' held-out test docs. Rotate over all nine.
We report, per held-out class:
  * unknown-detect — fraction of the held-out (unknown) docs flagged, AT the threshold that
                     keeps 95% of KNOWN test docs (a "detection @ 95% known-retention" point;
                     the threshold is calibrated on the known TEST docs, which the detector
                     never saw, NOT on training scores, which are optimistically low)
  * AUROC          — threshold-free separation of known vs unknown (1.0 = perfect, 0.5 = none)

Read-only on the cached embeddings; trains nothing that is saved.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

_REPO_ROOT = Path(__file__).resolve().parents[2]  # .../tectonic/
DATA = _REPO_ROOT / "data/document_type"
EMB = _REPO_ROOT / "artifacts/document_type/embeddings"
DATASET_VECS = EMB / "dataset.all-mpnet-base-v2.npz"
OOS_VECS = EMB / "oos.all-mpnet-base-v2.npz"

KEEP_KNOWN = 0.95  # fix the threshold to retain this fraction of known docs, then measure


def _rows(name: str) -> list[dict]:
    return [json.loads(l) for l in (DATA / name).read_text().splitlines() if l.strip()]


def _id2vec(npz_path: Path) -> dict[str, np.ndarray]:
    z = np.load(npz_path, allow_pickle=True)
    return {doc_id: z["vectors"][i] for i, doc_id in enumerate(z["doc_ids"])}


def _fit_novelty(vecs: np.ndarray, labels: list[str]):
    """Return novelty(X): squared Mahalanobis distance to the nearest known-class mean,
    using per-class means and one shared Ledoit-Wolf-shrunk precision matrix."""
    labels = np.asarray(labels)
    classes = sorted(set(labels.tolist()))
    means, centered = [], []
    for c in classes:
        Xc = vecs[labels == c]
        mu = Xc.mean(axis=0)
        means.append(mu)
        centered.append(Xc - mu)  # subtract each class's own mean -> pooled within-class scatter
    precision = LedoitWolf().fit(np.vstack(centered)).precision_
    mu_mat = np.stack(means)  # (C, d)

    def novelty(X: np.ndarray) -> np.ndarray:
        deltas = np.asarray(X)[:, None, :] - mu_mat[None, :, :]   # (n, C, d)
        # squared Mahalanobis to each class mean with the shared precision P (d,d):
        # d2[n,c] = delta[n,c] @ P @ delta[n,c]
        d2 = np.einsum("ncd,de,nce->nc", deltas, precision, deltas)  # (n, C)
        return d2.min(axis=1)

    return novelty


def _standardize(known: np.ndarray, unknown: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Z-score both arrays using the KNOWN docs' mean/std, so two signals on different
    scales can be summed fairly."""
    mu, sd = known.mean(), known.std() + 1e-9
    return (known - mu) / sd, (unknown - mu) / sd


def _operating_point(known: np.ndarray, unknown: np.ndarray) -> tuple[float, float]:
    """Detection rate at the threshold that keeps KEEP_KNOWN of the known docs, plus the
    threshold-free AUROC (known=0, unknown=1)."""
    thr = np.quantile(known, KEEP_KNOWN)
    detect = float((unknown > thr).mean())
    auroc = roc_auc_score([0] * len(known) + [1] * len(unknown),
                          np.concatenate([known, unknown]))
    return detect, auroc


def _leave_one_class_out(id2vec, train, test, labels) -> None:
    # Three "unknown" signals, all scored so that HIGHER = more novel:
    #   distance   — Mahalanobis distance to the nearest known-class mean
    #   confidence — 1 - max softmax of a classifier trained on the known classes
    #   combined   — the two, each standardized on the known docs, summed
    signals = ("distance", "confidence", "combined")
    detects = {s: [] for s in signals}
    aurocs = {s: [] for s in signals}

    print(f"{'held-out (as unknown)':24} {'n_unk':>6}   detect@95 (dist / conf / comb)")
    for held in labels:
        ktrain = [(id2vec[r["doc_id"]], r["type"]) for r in train if r["type"] != held]
        x_tr = np.array([v for v, _ in ktrain])
        y_tr = [t for _, t in ktrain]
        ktest = np.array([id2vec[r["doc_id"]] for r in test if r["type"] != held])
        # the held-out class is unseen in training, so ALL its docs are valid "unknowns"
        unknown = np.array([id2vec[r["doc_id"]] for r in (train + test) if r["type"] == held])

        novelty = _fit_novelty(x_tr, y_tr)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(x_tr, y_tr)

        scores = {
            "distance": (novelty(ktest), novelty(unknown)),
            "confidence": (1 - clf.predict_proba(ktest).max(axis=1),
                           1 - clf.predict_proba(unknown).max(axis=1)),
        }
        # combined = sum of the two, each standardized by the KNOWN-doc mean/std so they
        # share a scale before adding.
        dk, du = _standardize(*scores["distance"])
        ck, cu = _standardize(*scores["confidence"])
        scores["combined"] = (dk + ck, du + cu)

        row = []
        for s in signals:
            known_s, unknown_s = scores[s]
            detect, auroc = _operating_point(known_s, unknown_s)
            detects[s].append(detect)
            aurocs[s].append(auroc)
            row.append(f"{detect:5.1%}")
        print(f"{held:24} {len(unknown):6}   {' / '.join(row)}")

    print("\nmean over classes (detect@95% known-kept  |  AUROC):")
    for s in signals:
        print(f"  {s:11} {np.mean(detects[s]):6.1%}   |   {np.mean(aurocs[s]):.3f}")
    print("(higher = better. If 'combined' does not clearly beat 'distance', the cheap leg "
          "cannot do open-set on its own and the LLM leg must own it.)")


def _oos_check(id2vec, train) -> None:
    """Bonus: the 3 non-EDGAR OOS docs are KNOWN types (nda/commercial) from a different
    source. A good detector should NOT flag them as unknown; we print where their novelty
    sits versus the known threshold, to see if source-shift alone trips a false 'unknown'."""
    if not OOS_VECS.exists():
        return
    oos_rows = _rows("oos.jsonl")
    oos_v = _id2vec(OOS_VECS)
    novelty = _fit_novelty(np.array([id2vec[r["doc_id"]] for r in train]),
                           [r["type"] for r in train])
    thr = np.quantile(novelty(np.array([id2vec[r["doc_id"]] for r in train])), KEEP_KNOWN)
    print(f"\nOOS (known types, different source) vs known threshold {thr:.1f}:")
    for r in oos_rows:
        s = float(novelty(oos_v[r["doc_id"]][None])[0])
        flag = "flagged UNKNOWN (false positive)" if s > thr else "accepted as known (good)"
        print(f"  {r['doc_id']:45} type={r['type']:20} novelty={s:8.1f}  {flag}")


def main() -> None:
    id2vec = _id2vec(DATASET_VECS)
    train, test = _rows("train.jsonl"), _rows("test.jsonl")
    labels = sorted({r["type"] for r in train})
    print(f"known classes: {len(labels)}   train={len(train)}  test={len(test)}\n")
    _leave_one_class_out(id2vec, train, test, labels)
    _oos_check(id2vec, train)


if __name__ == "__main__":
    main()
