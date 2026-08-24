"""Embeddings probe, step 2b: does a SEMANTIC representation GENERALIZE better?

Step 2a showed embeddings and TF-IDF are within a few points IN-DISTRIBUTION, which
was expected and is not the question. This step runs the two places TF-IDF is known to
struggle, and asks whether embeddings hold up better. It is a deliberate MIRROR of the
TF-IDF probes (same seed, same splits, same classifier config, same rows); the ONLY
thing that changes is the representation, so any difference is the embeddings talking.

  Proxy 1 - ip cross-source (the sharp test). ip_agreement is our one two-corpus class.
    We train a full 9-class model whose ip examples come from ONE source (EDGAR), then
    test recall on ip from the OTHER source (CUAD), which the model never saw. This is
    exactly ip_source_transfer.py. TF-IDF's numbers to beat:
        in-source recall (held-out EDGAR ip):        1.000   (the ceiling)
        cross-source recall (unseen CUAD ip):        0.500   <-- the number that matters
        control (CUAD ip 3-fold, trained on BOTH):   0.659   (how recognizable CUAD is)
    The gap between 0.500 and 0.659 is the transfer gap TF-IDF pays for leaning on house
    style. If embeddings shrink that gap, semantics generalize better.

  Proxy 2 - OOS spot-check (anecdote, not statistic). Predict the 3 genuinely non-EDGAR
    Common Paper docs. TF-IDF got all 3 right but at LOW confidence (0.40-0.56). We look
    at whether embeddings are right and whether they are more confident. Three docs is
    too few to be a statistic; read a confident hit as mild positive, a confident miss
    as a red flag.

Reuses the embedding engine (and its cert fix) from embeddings_probe.py. The "train"
cache from step 2a is reused for the OOS classifier; "dataset" and "oos" are embedded
and cached here. Read-only on the data; trains throwaway classifiers, saves nothing.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from embeddings_probe import MODEL_NAME, embed_rows_cached, _classifier

_REPO_ROOT = Path(__file__).resolve().parents[2]  # .../tectonic/
DATA_DIR = _REPO_ROOT / "data/document_type"

IP = "ip_agreement"
SEED = 20260808            # identical to ip_source_transfer.py so the split matches
HELDOUT_FRACTION = 0.2

# TF-IDF baseline numbers, printed inline so the comparison is on-screen. These are a
# RECORDED SNAPSHOT from ip_source_transfer.py on the 2188-doc leak-free dataset
# (2026-08-24); they DRIFT whenever the dataset is rebuilt (class counts change), so
# treat them as a reference, not a live truth, and re-run ip_source_transfer.py to refresh.
TFIDF_IN_SOURCE = 1.000
TFIDF_CROSS_SOURCE = 0.500
TFIDF_CONTROL = 0.659


def _rows(name: str) -> list[dict]:
    return [json.loads(l) for l in (DATA_DIR / name).read_text().splitlines() if l.strip()]


def _split(items: list, frac: float) -> tuple[list, list]:
    """Deterministic shuffle then (train, heldout); identical to ip_source_transfer."""
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(items))
    cut = int(round(len(items) * (1 - frac)))
    return [items[i] for i in order[:cut]], [items[i] for i in order[cut:]]


def _run_direction(train_src: str, test_src: str, rows: list[dict], vec_of) -> None:
    """Train ip from ONE source, test recall on ip from the OTHER (mirror of TF-IDF)."""
    non_ip = [r for r in rows if r["type"] != IP]
    ip_train_src = [r for r in rows if r["type"] == IP and r["source"] == train_src]
    ip_test_src = [r for r in rows if r["type"] == IP and r["source"] == test_src]

    ip_tr, ip_heldout = _split(ip_train_src, HELDOUT_FRACTION)
    train = non_ip + ip_tr

    clf = _classifier()
    clf.fit(np.array([vec_of(r) for r in train]), [r["type"] for r in train])

    def recall(docs: list[dict]) -> tuple[float, Counter]:
        if not docs:
            return float("nan"), Counter()
        preds = clf.predict(np.array([vec_of(r) for r in docs]))
        return sum(p == IP for p in preds) / len(docs), Counter(preds)

    in_rec, _ = recall(ip_heldout)
    cross_rec, cross_preds = recall(ip_test_src)

    print(f"\n=== train ip = {train_src} ({len(ip_tr)} docs)  ->  test ip = {test_src} ===")
    print(f"  in-source recall  (held-out {train_src} ip, n={len(ip_heldout)}): {in_rec:.3f}")
    print(f"  cross-source recall (all {test_src} ip, n={len(ip_test_src)}): {cross_rec:.3f}")
    if train_src == "edgar":
        print(f"    TF-IDF was: in-source {TFIDF_IN_SOURCE:.3f}, cross-source {TFIDF_CROSS_SOURCE:.3f}")
    print(f"  where the {test_src} ip docs actually landed:")
    for label, n in cross_preds.most_common():
        print(f"     {n:3}  {label}")


def _control_recognizable(test_src: str, rows: list[dict], vec_of, folds: int = 3) -> None:
    """Is CUAD ip recognizable when the model HAS seen its style? Mirror of the TF-IDF
    control: cross-validate CUAD ip while training on BOTH sources. High here means the
    earlier cross-source drop was a genuine transfer gap, not CUAD being inherently
    commercial-like."""
    non_ip = [r for r in rows if r["type"] != IP]
    other_ip = [r for r in rows if r["type"] == IP and r["source"] != test_src]
    target_ip = [r for r in rows if r["type"] == IP and r["source"] == test_src]

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(target_ip))
    fold_of = {int(order[i]): i % folds for i in range(len(target_ip))}

    preds: list[str] = []
    for f in range(folds):
        test = [target_ip[i] for i in range(len(target_ip)) if fold_of[i] == f]
        held = [target_ip[i] for i in range(len(target_ip)) if fold_of[i] != f]
        train = non_ip + other_ip + held
        clf = _classifier()
        clf.fit(np.array([vec_of(r) for r in train]), [r["type"] for r in train])
        preds.extend(list(clf.predict(np.array([vec_of(r) for r in test]))))

    recall = sum(p == IP for p in preds) / len(preds)
    print(f"\n=== CONTROL: trained on BOTH sources, {folds}-fold recall on {test_src} ip "
          f"(n={len(target_ip)}) ===")
    print(f"  recall: {recall:.3f}   (TF-IDF control was {TFIDF_CONTROL:.3f})")
    for label, n in Counter(preds).most_common():
        print(f"     {n:3}  {label}")


def _oos_spotcheck() -> None:
    """Predict the non-EDGAR OOS docs with an embeddings classifier trained on train.jsonl
    (mirror of oos_eval.py, which uses the TF-IDF baseline model)."""
    train = _rows("train.jsonl")
    oos = _rows("oos.jsonl")
    x_train = embed_rows_cached(train, "train")   # reuses the step-2a cache
    x_oos = embed_rows_cached(oos, "oos")

    clf = _classifier()
    clf.fit(x_train, [r["type"] for r in train])
    classes = list(clf.classes_)
    preds = clf.predict(x_oos)
    proba = clf.predict_proba(x_oos)

    print(f"\n=== OOS spot-check ({len(oos)} non-EDGAR docs; TF-IDF got all right at p=0.40-0.56) ===")
    correct = 0
    for row, pred, p in zip(oos, preds, proba):
        true = row["type"]
        ok = pred == true
        correct += ok
        conf_true = p[classes.index(true)]
        conf_pred = p[classes.index(pred)]
        mark = "OK " if ok else "XX "
        print(f"  {mark}{row['doc_id']}: true={true} (p={conf_true:.2f}) -> pred={pred} (p={conf_pred:.2f})")
    print(f"  overall: {correct}/{len(oos)} correct")


def main() -> None:
    print(f"embeddings generalization probe ({MODEL_NAME})")

    # Embed the full dataset once (cached), then look vectors up by doc_id so every
    # direction/control below reuses the same vectors instead of re-embedding.
    dataset = _rows("dataset.jsonl")
    dataset_vecs = embed_rows_cached(dataset, "dataset")
    id2vec = {r["doc_id"]: dataset_vecs[i] for i, r in enumerate(dataset)}
    vec_of = lambda r: id2vec[r["doc_id"]]

    by_src = Counter(r["source"] for r in dataset if r["type"] == IP)
    print(f"ip_agreement by source: {dict(by_src)}")

    _run_direction("edgar", "cuad", dataset, vec_of)   # the sharp test
    _run_direction("cuad", "edgar", dataset, vec_of)   # thinner train; read with caveat
    _control_recognizable("cuad", dataset, vec_of)
    _oos_spotcheck()


if __name__ == "__main__":
    main()
