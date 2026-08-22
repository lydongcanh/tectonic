"""Error analysis: look at every test document the model got wrong.

The point is to decide, honestly, WHY the model errs, because the fix differs:
  * genuine ambiguity (a licence that really is commercial) -> ceiling, not fixable
    by a bigger bag-of-words model; only a semantic model or a different label might
    help.
  * mislabel / data bug (the text plainly is type X but we labelled it Y) -> fix the
    data, a real and immediate win.
  * systematic weakness (errors cluster on one source or one pattern) -> targetable.

For each miss we print the true and predicted labels with the model's probability for
each (a confident miss reads differently from a near-tie), the source corpus, and an
excerpt so a human can judge the document. We also summarise the error patterns:
which true->predicted pairs dominate, and whether errors cluster by source.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]  # .../tectonic/
DATA = _REPO_ROOT / "data/document_type/test.jsonl"
MODEL = _REPO_ROOT / "artifacts/document_type/baseline.model.joblib"
EXCERPT = 320


def main() -> None:
    rows = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]
    pipe = joblib.load(MODEL)
    classes = list(pipe.named_steps["clf"].classes_)
    proba = pipe.predict_proba([r["text"] for r in rows])
    pred_idx = proba.argmax(axis=1)

    errors = []
    for i, row in enumerate(rows):
        pred = classes[pred_idx[i]]
        if pred != row["type"]:
            errors.append((row, pred, proba[i]))

    print(f"{len(errors)} errors out of {len(rows)} test docs\n")

    print("error patterns (true -> predicted):")
    for (t, p), n in Counter((r["type"], pred) for r, pred, _ in errors).most_common():
        print(f"  {n:2}  {t}  ->  {p}")
    print("\nerrors by source corpus:")
    for src, n in Counter(r["source"] for r, _, _ in errors).most_common():
        total = sum(1 for r in rows if r["source"] == src)
        print(f"  {n:2}/{total:<4} {src}")

    print("\n" + "=" * 78)
    for row, pred, p in errors:
        true = row["type"]
        p_true = p[classes.index(true)]
        p_pred = p[classes.index(pred)]
        text = " ".join(row["text"].split())
        print(f"\n[{row['source']}] {row['doc_id']}")
        print(f"  true = {true} (p={p_true:.2f})   predicted = {pred} (p={p_pred:.2f})")
        print(f"  excerpt: {text[:EXCERPT]}")


if __name__ == "__main__":
    main()
