"""Risk-coverage analysis for the confidence-gated cascade (ML first, LLM on doubt).

The plan: auto-accept the ML label when the model is confident, escalate the rest to
an LLM. This script quantifies, on the held-out test set, how good that gate is:

  * coverage           = fraction of docs auto-accepted at a confidence threshold
  * selective accuracy = accuracy AMONG the auto-accepted (the precision we ship)
  * escalation rate    = 1 - coverage = fraction sent to the LLM (the cost)
  * error-catch rate   = fraction of the model's mistakes that fall BELOW the
                         threshold, i.e. get escalated instead of shipped wrong

We also print a calibration check: LogReg with class_weight="balanced" is not
guaranteed to produce probabilities that mean what they say, so we bin by confidence
and compare mean confidence to actual accuracy in each bin. If they track, the
threshold is trustworthy; if not, the numbers still order right-from-wrong but the
absolute values are only a guide.

IMPORTANT: this is measured IN-DISTRIBUTION (EDGAR test set). It shows how well the
gate separates correct from incorrect for the KNOWN types; it cannot speak to unseen
types or other sources, which is exactly why the LLM leg (not the threshold alone)
must be the authority on "is this even one of our types".
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]  # .../tectonic/
DATA = _REPO_ROOT / "data/document_type/test.jsonl"
MODEL = _REPO_ROOT / "artifacts/document_type/baseline.model.joblib"


def _load() -> tuple[np.ndarray, np.ndarray]:
    """Return (confidence, correct) per test doc: max predicted proba, and hit/miss."""
    rows = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]
    pipe = joblib.load(MODEL)
    proba = pipe.predict_proba([r["text"] for r in rows])
    classes = list(pipe.named_steps["clf"].classes_)
    pred_idx = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    correct = np.array([classes[pred_idx[i]] == rows[i]["type"] for i in range(len(rows))])
    return conf, correct


def _curve(conf: np.ndarray, correct: np.ndarray) -> None:
    n = len(conf)
    print(f"test docs: {n}   overall accuracy: {correct.mean():.3f}   errors: {(~correct).sum()}\n")

    print("threshold  coverage  accepted  selective-acc  errors-shipped  escalated")
    for t in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        acc_mask = conf >= t
        k = int(acc_mask.sum())
        if k == 0:
            print(f"  {t:.2f}       0.000        0            n/a              0          100%")
            continue
        sel = correct[acc_mask].mean()
        shipped_err = int((~correct[acc_mask]).sum())
        print(f"  {t:.2f}      {k/n:.3f}     {k:4}       {sel:.3f}          {shipped_err:4}         {100*(n-k)/n:4.0f}%")


def _operating_points(conf: np.ndarray, correct: np.ndarray) -> None:
    n = len(conf)
    print("\noperating points (max coverage while auto-accept precision >= target):")
    cand = np.unique(conf)
    for target in [0.95, 0.98, 0.99, 1.00]:
        best = None
        for t in cand:
            mask = conf >= t
            if mask.sum() == 0:
                continue
            sel = correct[mask].mean()
            if sel >= target:
                cov = mask.mean()
                if best is None or cov > best[1]:
                    best = (float(t), float(cov), int(mask.sum()), int((~correct[mask]).sum()))
        if best:
            t, cov, k, err = best
            print(f"  precision >= {target:.2f}: threshold {t:.3f} -> coverage {cov:.1%} "
                  f"({k} auto-accepted, {err} wrong), escalate {1-cov:.1%} to LLM")
        else:
            print(f"  precision >= {target:.2f}: not achievable at any threshold")


def _error_catch(conf: np.ndarray, correct: np.ndarray) -> None:
    print("\nerror-catch (of the model's mistakes, how many the gate escalates):")
    errs = conf[~correct]
    if len(errs) == 0:
        print("  no errors to catch")
        return
    for t in [0.50, 0.60, 0.70, 0.80]:
        caught = (errs < t).mean()
        print(f"  threshold {t:.2f}: escalates {caught:.0%} of errors "
              f"({int((errs < t).sum())}/{len(errs)})")


def _calibration(conf: np.ndarray, correct: np.ndarray) -> None:
    print("\ncalibration (does confidence match actual accuracy?):")
    print("  conf-bin      n   mean-conf   actual-acc")
    for lo in [0.2, 0.4, 0.6, 0.8]:
        hi = lo + 0.2
        m = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= 1.0)
        if m.sum() == 0:
            continue
        print(f"  {lo:.1f}-{hi:.1f}   {int(m.sum()):4}     {conf[m].mean():.3f}       {correct[m].mean():.3f}")


def main() -> None:
    conf, correct = _load()
    _curve(conf, correct)
    _operating_points(conf, correct)
    _error_catch(conf, correct)
    _calibration(conf, correct)


if __name__ == "__main__":
    main()
