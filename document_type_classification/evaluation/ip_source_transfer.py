"""Phase 0 out-of-source probe: does ip_agreement knowledge transfer across corpora?

ip_agreement is the one class we built from TWO corpora: CUAD (contracts, ~43) and
EDGAR EX-10 licences (~114). Every other class comes from a single corpus, so this
is the only FREE, data-internal test of the worry behind our perfect in-distribution
scores: is the model learning LICENSING CONTENT, or each corpus's house style?

Method: train a full 9-class model whose ip examples come from ONE source, then test
on ip docs from the OTHER source (which the model has never seen). We compare:

  * in-source recall  — held-out ip from the SAME source  (the ceiling)
  * cross-source recall — all ip from the OTHER source     (the transfer)

If cross-source recall is close to in-source, ip generalises across corpora (good).
If it collapses (docs land in commercial etc.), the model leaned on source style.

We run both directions. Caveat: CUAD->EDGAR trains on only ~34 ip docs, so a lower
number there is partly thinness, not only source mismatch; EDGAR->CUAD (train ~91)
is the cleaner read. The near-duplicate filter in build_dataset already guarantees
no CUAD-ip doc is a near-copy of an EDGAR-ip doc, so the two sides are independent.

Model mirrors baseline.py exactly (same TF-IDF + LogReg config). Read-only on the
dataset; trains throwaway models, saves nothing.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

_REPO_ROOT = Path(__file__).resolve().parents[2]  # .../tectonic/
DATASET = _REPO_ROOT / "data/document_type/dataset.jsonl"
IP = "ip_agreement"
SEED = 20260808
HELDOUT_FRACTION = 0.2


def _pipeline() -> Pipeline:
    """Same model as baseline.py, so this probe measures the real classifier."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def _rows() -> list[dict]:
    return [json.loads(l) for l in DATASET.read_text().splitlines() if l.strip()]


def _split(items: list, frac: float) -> tuple[list, list]:
    """Deterministic shuffle, then (train, heldout) with `frac` held out."""
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(items))
    cut = int(round(len(items) * (1 - frac)))
    return [items[i] for i in order[:cut]], [items[i] for i in order[cut:]]


def _run_direction(train_src: str, test_src: str, rows: list[dict]) -> None:
    non_ip = [r for r in rows if r["type"] != IP]
    ip_train_src = [r for r in rows if r["type"] == IP and r["source"] == train_src]
    ip_test_src = [r for r in rows if r["type"] == IP and r["source"] == test_src]

    # Hold out part of the TRAIN source's ip as the in-source ceiling control.
    ip_tr, ip_heldout = _split(ip_train_src, HELDOUT_FRACTION)

    train = non_ip + ip_tr
    pipe = _pipeline()
    pipe.fit([r["text"] for r in train], [r["type"] for r in train])

    def recall(docs: list[dict]) -> tuple[float, Counter]:
        if not docs:
            return float("nan"), Counter()
        preds = pipe.predict([r["text"] for r in docs])
        hits = sum(p == IP for p in preds)
        return hits / len(docs), Counter(preds)

    in_rec, _ = recall(ip_heldout)
    cross_rec, cross_preds = recall(ip_test_src)

    print(f"\n=== train ip = {train_src} ({len(ip_tr)} docs)  ->  test ip = {test_src} ===")
    print(f"  in-source recall  (held-out {train_src} ip, n={len(ip_heldout)}): {in_rec:.3f}")
    print(f"  cross-source recall (all {test_src} ip, n={len(ip_test_src)}): {cross_rec:.3f}")
    print(f"  where the {test_src} ip docs actually landed:")
    for label, n in cross_preds.most_common():
        print(f"     {n:3}  {label}")


def _control_recognizable(test_src: str, rows: list[dict], folds: int = 3) -> None:
    """Disentangle 'transfer gap' from 'this corpus is intrinsically hard'.

    Cross-validate over the TEST source's ip while training on BOTH sources, so the
    model HAS seen this corpus's style. If recall here is high, the corpus IS
    recognizable when represented in training, so the earlier cross-source drop was a
    genuine transfer gap, not the corpus being inherently commercial-like. If it is
    also low, the corpus's licences are just borderline regardless of training.
    """
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
        pipe = _pipeline()
        pipe.fit([r["text"] for r in train], [r["type"] for r in train])
        preds.extend(list(pipe.predict([r["text"] for r in test])))

    recall = sum(p == IP for p in preds) / len(preds)
    print(f"\n=== CONTROL: trained on BOTH sources, {folds}-fold recall on {test_src} ip "
          f"(n={len(target_ip)}) ===")
    print(f"  recall: {recall:.3f}   (vs cross-source {0.488 if test_src=='cuad' else '?'} when {test_src} unseen)")
    for label, n in Counter(preds).most_common():
        print(f"     {n:3}  {label}")


def main() -> None:
    rows = _rows()
    by_src = Counter(r["source"] for r in rows if r["type"] == IP)
    print(f"ip_agreement by source: {dict(by_src)}")
    _run_direction("edgar", "cuad", rows)   # cleaner read (more training ip)
    _run_direction("cuad", "edgar", rows)   # thinner train; read with the caveat
    _control_recognizable("cuad", rows)     # is CUAD ip recognizable when SEEN in training?


if __name__ == "__main__":
    main()
