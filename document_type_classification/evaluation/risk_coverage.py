"""Risk-coverage analysis for the confidence-gated cascade (ML first, LLM on doubt).

The plan: auto-accept the ML label when the model is confident, escalate the rest to
an LLM. This script quantifies how good that gate is:

  * coverage           = fraction of docs auto-accepted at a confidence threshold
  * selective accuracy = accuracy AMONG the auto-accepted (the precision we ship)
  * escalation rate    = 1 - coverage = fraction sent to the LLM (the cost)
  * error-catch rate    = fraction of the model's mistakes that fall BELOW the
                         threshold, i.e. get escalated instead of shipped wrong

HONEST OPERATING POINTS: a threshold that is BOTH chosen and scored on the same set
is an in-sample optimum; it flatters the gate and will not hold on fresh data. So the
prescriptive operating points here are CALIBRATED on the training set's out-of-fold
cross-validated confidences (each fold's confidence comes from a model that did not
train on that fold) and then REPORTED on the untouched test set. The descriptive
curve and the calibration check below are still read off the test set directly, which
is fine, they describe, they do not pick a threshold to then grade themselves on.
(Caveat: the fold models train on 80% of the data, so they are marginally less
confident than the deployed full-data model; this biases the calibrated threshold
slightly LOW, i.e. conservative, the achieved test precision tends to meet or beat
the target.)

IMPORTANT: everything here is IN-DISTRIBUTION (EDGAR). It shows how well the gate
separates correct from incorrect for the KNOWN types; it cannot speak to unseen types
or other sources, which is exactly why the LLM leg (not the threshold alone) must be
the authority on "is this even one of our types".
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

_REPO_ROOT = Path(__file__).resolve().parents[2]  # .../tectonic/
DATA_DIR = _REPO_ROOT / "data/document_type"
MODEL = _REPO_ROOT / "artifacts/document_type/baseline.model.joblib"
CV_FOLDS = 5
CV_SEED = 20260806  # fixed so the calibration is reproducible


def _rows(split: str) -> tuple[list[str], list[str]]:
    lines = (DATA_DIR / f"{split}.jsonl").read_text().splitlines()
    rows = [json.loads(l) for l in lines if l.strip()]
    return [r["text"] for r in rows], [r["type"] for r in rows]


def _scores_from_proba(proba: np.ndarray, classes: list[str], y_true: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(confidence, correct) per doc from a proba matrix: max proba, and hit/miss."""
    pred_idx = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    correct = np.array([classes[pred_idx[i]] == y_true[i] for i in range(len(y_true))])
    return conf, correct


def _test_scores(pipe) -> tuple[np.ndarray, np.ndarray]:
    """Deployed model (trained on all of train) scored on the held-out test set."""
    x_test, y_test = _rows("test")
    proba = pipe.predict_proba(x_test)
    classes = list(pipe.named_steps["clf"].classes_)
    return _scores_from_proba(proba, classes, y_test)


def _calibration_scores(pipe) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold confidences on the TRAINING set: each row scored by a model that
    did not see it. An honest held-out signal for CHOOSING a threshold, independent
    of the test set we then report on."""
    x_train, y_train = _rows("train")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED)
    proba = cross_val_predict(clone(pipe), x_train, y_train, cv=cv, method="predict_proba")
    classes = sorted(set(y_train))  # cross_val_predict columns follow sorted(unique(y))
    return _scores_from_proba(proba, classes, y_train)


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


def _operating_points(calib_conf: np.ndarray, calib_correct: np.ndarray,
                      test_conf: np.ndarray, test_correct: np.ndarray) -> None:
    """Choose each threshold on the CALIBRATION set (train OOF), report it on TEST."""
    n_te = len(test_conf)
    print(f"\noperating points (threshold chosen on {len(calib_conf)} train OOF docs, "
          f"reported on {n_te} test docs):")
    for target in [0.95, 0.98, 0.99, 1.00]:
        best = None  # pick max calibration coverage while calibration precision >= target
        for t in np.unique(calib_conf):
            mask = calib_conf >= t
            if mask.sum() == 0 or calib_correct[mask].mean() < target:
                continue
            cov = mask.mean()
            if best is None or cov > best[1]:
                best = (float(t), float(cov))
        if best is None:
            print(f"  precision >= {target:.2f}: not achievable at any threshold (calibration)")
            continue
        t, calib_cov = best
        te_mask = test_conf >= t
        if te_mask.sum() == 0:
            print(f"  precision >= {target:.2f}: threshold {t:.3f} -> accepts 0 test docs")
            continue
        te_cov = te_mask.mean()
        te_sel = test_correct[te_mask].mean()
        te_err = int((~test_correct[te_mask]).sum())
        print(f"  target >= {target:.2f}: threshold {t:.3f} (calib coverage {calib_cov:.1%}) "
              f"-> TEST coverage {te_cov:.1%}, achieved precision {te_sel:.3f} "
              f"({te_err} wrong of {int(te_mask.sum())}), escalate {1 - te_cov:.1%}")


def _error_catch(conf: np.ndarray, correct: np.ndarray) -> None:
    print("\nerror-catch on test (of the model's mistakes, how many the gate escalates):")
    errs = conf[~correct]
    if len(errs) == 0:
        print("  no errors to catch")
        return
    for t in [0.50, 0.60, 0.70, 0.80]:
        caught = (errs < t).mean()
        print(f"  threshold {t:.2f}: escalates {caught:.0%} of errors "
              f"({int((errs < t).sum())}/{len(errs)})")


def _calibration(conf: np.ndarray, correct: np.ndarray) -> None:
    print("\ncalibration on test (does confidence match actual accuracy?):")
    print("  conf-bin      n   mean-conf   actual-acc")
    for lo in [0.2, 0.4, 0.6, 0.8]:
        hi = lo + 0.2
        m = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= 1.0)
        if m.sum() == 0:
            continue
        print(f"  {lo:.1f}-{hi:.1f}   {int(m.sum()):4}     {conf[m].mean():.3f}       {correct[m].mean():.3f}")


def main() -> None:
    pipe = joblib.load(MODEL)
    test_conf, test_correct = _test_scores(pipe)
    calib_conf, calib_correct = _calibration_scores(pipe)

    _curve(test_conf, test_correct)
    _operating_points(calib_conf, calib_correct, test_conf, test_correct)
    _error_catch(test_conf, test_correct)
    _calibration(test_conf, test_correct)


if __name__ == "__main__":
    main()
