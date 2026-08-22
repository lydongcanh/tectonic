"""Phase 1 out-of-source evaluation: run the trained model on GENUINELY non-EDGAR
documents and see whether the in-distribution scores hold up.

Every training class comes from EDGAR / EDGAR-derived corpora, so our held-out
test is in-distribution. This script scores the saved model on documents from a
different origin entirely (e.g. Common Paper CC-BY standard agreements), the real
test of whether the model learned document-type CONTENT or corpus house style.

Reads an eval-only `data/document_type/oos.jsonl` (same row shape as the training
data: doc_id, source, type, text) that is NEVER used for training. For each doc it
prints the true label, the predicted label, whether it was right, and the model's
confidence in both, so a wrong-but-borderline call reads differently from a
confident miss. The set is deliberately small and hand-verified, so read it as
anecdotes, not statistics: a correct call on a non-EDGAR doc is mild positive
evidence; a confident miss is a real red flag.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]  # .../tectonic/
OOS_PATH = _REPO_ROOT / "data/document_type/oos.jsonl"
MODEL_PATH = _REPO_ROOT / "artifacts/document_type/baseline.model.joblib"


def main() -> None:
    rows = [json.loads(l) for l in OOS_PATH.read_text().splitlines() if l.strip()]
    pipe = joblib.load(MODEL_PATH)
    classes = list(pipe.named_steps["clf"].classes_)

    texts = [r["text"] for r in rows]
    preds = pipe.predict(texts)
    proba = pipe.predict_proba(texts)

    print(f"OOS documents: {len(rows)} (from {Counter(r['source'] for r in rows)})\n")
    correct = 0
    per_class = {}
    for row, pred, p in zip(rows, preds, proba):
        true = row["type"]
        ok = pred == true
        correct += ok
        per_class.setdefault(true, [0, 0])
        per_class[true][0] += ok
        per_class[true][1] += 1
        conf_true = p[classes.index(true)]
        conf_pred = p[classes.index(pred)]
        mark = "OK " if ok else "XX "
        print(f"{mark}{row['doc_id']}")
        print(f"     true={true} (model p={conf_true:.2f})  ->  predicted={pred} (p={conf_pred:.2f})")

    print(f"\noverall: {correct}/{len(rows)} correct")
    print("by class (recall on this tiny OOS set):")
    for cls, (hit, tot) in sorted(per_class.items()):
        print(f"  {cls:22} {hit}/{tot}")


if __name__ == "__main__":
    main()
