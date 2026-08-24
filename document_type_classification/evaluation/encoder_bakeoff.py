"""Encoder bake-off: pick the embedding encoder on EVIDENCE, not reputation.

We proved (steps 2a/2b) that a semantic representation generalizes better than TF-IDF.
This asks the follow-up: WHICH encoder? Same splits, same classifier, same chunk/pool
method as the rest of the probe; the ONLY thing that varies is the encoder, so the
numbers are directly comparable. Contenders:

  * all-mpnet-base-v2            - our current probe encoder (strong 2021 general model)
  * BAAI/bge-large-en-v1.5       - a strong MODERN general embedder (1024-dim, MTEB top-tier)
  * nlpaueb/legal-bert-base-uncased - frozen legal masked-LM + mean pooling. This settles
        the "LegalBERT without fine-tuning" question empirically. Raw BERT is not a
        sentence encoder, so mean-pooled vectors are expected to be WEAK despite the
        legal vocabulary; we measure rather than assume.

For each encoder we report the metrics that matter, with in-distribution kept only as a
reference (it is the metric we least trust):

  in-dist macro-F1     - EDGAR test set (reference only)
  ip x-source recall   - train EDGAR ip, test UNSEEN CUAD ip  <-- the sharp generalization test
  ip in-source recall  - held-out EDGAR ip (the ceiling)
  control recall       - CUAD ip, 3-fold, trained on BOTH sources (how recognizable it is)
  OOS                  - the 3 non-EDGAR docs: how many correct, and mean confidence

Each encoder embeds the corpus ONCE (cached per encoder in artifacts/), then every metric
is derived from those vectors by doc_id lookup, so we never re-embed. Read-only on data;
trains throwaway classifiers; saves only the embedding caches.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import f1_score

from embeddings_probe import embed_rows_cached, _classifier
from embeddings_generalization import HELDOUT_FRACTION, IP, _rows, _split

# TF-IDF baseline, carried inline as the reference row of the final table. RECORDED
# SNAPSHOT from baseline.py (in_dist) + ip_source_transfer.py (transfer) on the 2188-doc
# leak-free dataset (2026-08-24). These DRIFT on any dataset rebuild; re-run those scripts
# to refresh rather than trusting the numbers here as live.
TFIDF = {"in_dist": 0.977, "x_source": 0.500, "in_source": 1.000,
         "control": 0.659, "oos_correct": 3, "oos_total": 3, "oos_conf": 0.48}

ENCODERS = [
    "all-mpnet-base-v2",
    "BAAI/bge-large-en-v1.5",
    "nlpaueb/legal-bert-base-uncased",
]


def _in_distribution(train, test, id2vec, labels) -> float:
    """Macro-F1 on the EDGAR test split, using vectors looked up from the dataset embed."""
    clf = _classifier()
    clf.fit(np.array([id2vec[r["doc_id"]] for r in train]), [r["type"] for r in train])
    preds = clf.predict(np.array([id2vec[r["doc_id"]] for r in test]))
    return float(f1_score([r["type"] for r in test], preds, labels=labels, average="macro"))


def _ip_transfer(dataset, id2vec) -> tuple[float, float, Counter]:
    """Train ip from EDGAR only, measure recall on held-out EDGAR ip (ceiling) and on
    the fully unseen CUAD ip (the transfer). Mirror of embeddings_generalization."""
    non_ip = [r for r in dataset if r["type"] != IP]
    ip_edgar = [r for r in dataset if r["type"] == IP and r["source"] == "edgar"]
    ip_cuad = [r for r in dataset if r["type"] == IP and r["source"] == "cuad"]
    ip_tr, ip_heldout = _split(ip_edgar, HELDOUT_FRACTION)

    clf = _classifier()
    train = non_ip + ip_tr
    clf.fit(np.array([id2vec[r["doc_id"]] for r in train]), [r["type"] for r in train])

    def recall(docs):
        preds = clf.predict(np.array([id2vec[r["doc_id"]] for r in docs]))
        return sum(p == IP for p in preds) / len(docs), Counter(preds)

    in_rec, _ = recall(ip_heldout)
    cross_rec, landing = recall(ip_cuad)
    return in_rec, cross_rec, landing


def _control(dataset, id2vec, folds: int = 3) -> float:
    """CUAD ip recall, 3-fold, trained on BOTH sources (mirror of the TF-IDF control)."""
    non_ip = [r for r in dataset if r["type"] != IP]
    other_ip = [r for r in dataset if r["type"] == IP and r["source"] != "cuad"]
    target = [r for r in dataset if r["type"] == IP and r["source"] == "cuad"]

    rng = np.random.default_rng(20260808)  # same seed family as the split
    order = rng.permutation(len(target))
    fold_of = {int(order[i]): i % folds for i in range(len(target))}

    preds: list[str] = []
    for f in range(folds):
        test = [target[i] for i in range(len(target)) if fold_of[i] == f]
        held = [target[i] for i in range(len(target)) if fold_of[i] != f]
        train = non_ip + other_ip + held
        clf = _classifier()
        clf.fit(np.array([id2vec[r["doc_id"]] for r in train]), [r["type"] for r in train])
        preds.extend(list(clf.predict(np.array([id2vec[r["doc_id"]] for r in test]))))
    return sum(p == IP for p in preds) / len(preds)


def _oos(train, id2vec, model_name) -> tuple[int, int, float, list]:
    """Fit on the train split, predict the non-EDGAR OOS docs; return hits, total, and
    mean confidence in the predicted class (TF-IDF's mean was ~0.48). Train vectors come
    from the dataset embed via id2vec, so only the 3 OOS docs are embedded here."""
    oos = _rows("oos.jsonl")
    x_train = np.array([id2vec[r["doc_id"]] for r in train])
    x_oos = embed_rows_cached(oos, "oos", model_name)
    clf = _classifier()
    clf.fit(x_train, [r["type"] for r in train])
    classes = list(clf.classes_)
    preds = clf.predict(x_oos)
    proba = clf.predict_proba(x_oos)
    detail = []
    hits = 0
    confs = []
    for row, pred, p in zip(oos, preds, proba):
        ok = pred == row["type"]
        hits += ok
        confs.append(float(p.max()))
        detail.append((row["doc_id"], row["type"], pred, float(p[classes.index(row["type"])]), ok))
    return hits, len(oos), float(np.mean(confs)), detail


def _evaluate(model_name: str) -> dict:
    print(f"\n########## encoder: {model_name} ##########")
    dataset = _rows("dataset.jsonl")
    train = _rows("train.jsonl")
    test = _rows("test.jsonl")
    labels = sorted({r["type"] for r in train})

    dataset_vecs = embed_rows_cached(dataset, "dataset", model_name)
    id2vec = {r["doc_id"]: dataset_vecs[i] for i, r in enumerate(dataset)}
    # fail loudly rather than KeyError deep in a fit if the split and dataset disagree
    missing = [r["doc_id"] for r in train + test if r["doc_id"] not in id2vec]
    if missing:
        raise SystemExit(f"{len(missing)} split doc_ids absent from dataset.jsonl embed; "
                         f"first: {missing[0]}")

    in_dist = _in_distribution(train, test, id2vec, labels)
    in_rec, x_rec, landing = _ip_transfer(dataset, id2vec)
    control = _control(dataset, id2vec)
    oos_hits, oos_tot, oos_conf, oos_detail = _oos(train, id2vec, model_name)

    print(f"  in-dist macro-F1: {in_dist:.3f}")
    print(f"  ip in-source recall: {in_rec:.3f}   x-source recall: {x_rec:.3f}   control: {control:.3f}")
    print(f"  CUAD ip landed: {dict(landing)}")
    print(f"  OOS: {oos_hits}/{oos_tot} correct, mean conf {oos_conf:.2f}")
    for doc_id, true, pred, p_true, ok in oos_detail:
        print(f"     {'OK ' if ok else 'XX '}{doc_id}: {true} -> {pred} (p_true={p_true:.2f})")

    return {"encoder": model_name, "in_dist": in_dist, "in_source": in_rec,
            "x_source": x_rec, "control": control,
            "oos_correct": oos_hits, "oos_total": oos_tot, "oos_conf": oos_conf}


def _table(results: list[dict]) -> None:
    print("\n" + "=" * 92)
    print("BAKE-OFF SUMMARY (x-source recall is the metric that matters; in-dist is reference)")
    print(f"{'encoder':34} {'in-dist':>8} {'x-src':>7} {'in-src':>7} {'ctrl':>6} {'OOS':>10}")
    ref = {**TFIDF, "encoder": "TF-IDF (baseline)"}
    for r in [ref] + results:
        oos = f"{r['oos_correct']}/{r['oos_total']}@{r['oos_conf']:.2f}"
        print(f"{r['encoder']:34} {r['in_dist']:8.3f} {r['x_source']:7.3f} "
              f"{r['in_source']:7.3f} {r['control']:6.3f} {oos:>10}")
    best = max(results, key=lambda r: r["x_source"])
    print(f"\nbest generalizer (x-source recall): {best['encoder']}  ({best['x_source']:.3f})")


def main() -> None:
    results = [_evaluate(name) for name in ENCODERS]
    _table(results)


if __name__ == "__main__":
    main()
