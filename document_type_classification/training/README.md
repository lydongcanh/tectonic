# training

The real, kept code for the document-type classifier, in two phases:

```
training/
  dataset/  prepare a clean dataset (sources -> dedupe -> split -> audit)
  model/    train and evaluate a model on that dataset
```

The two phases are separated on purpose. `dataset/` writes data files; `model/`
reads those files. They share a **file format** (`train.jsonl` / `test.jsonl`),
not code, so neither imports the other.

## dataset/ — prepare the data

Three ordered stages. Run them all with one command:

```bash
poetry run python document_type_classification/training/dataset/prepare.py
```

That runs, in order:

1. **build_dataset.py** — pull each source, keep only modelled labels, drop exact
   and near-duplicate documents, write `data/document_type/dataset.jsonl`.
2. **split.py** — stratified train/test split (same class mix on both sides),
   writes `train.jsonl` and `test.jsonl`.
3. **audit_split.py** — check no near-duplicate leaked across the split. **Fails
   the run** if it finds one, so a clean finish means the data is safe to train on.

You can also run any stage on its own while iterating.

Supporting modules (not run directly):

- **dataset.py** — the `Example` row shape (`doc_id`, `source`, `type`, `text`),
  the `LABELS` we currently model, and JSONL read/write.
- **fingerprint.py** — the near-duplicate sketch, shared by build (to remove
  near-dups) and audit (to check they are gone), so both agree on "near-duplicate".
- **sources/** — one loader per data source (`cuad.py`, `contract_nli.py`). Each
  maps its raw dataset onto `Example`s. Adding a new type = add a file here.

## model/ — train and evaluate

```bash
poetry run python document_type_classification/training/model/baseline.py
```

- **baseline.py** — TF-IDF + Logistic Regression. Reads the data files, prints
  macro-F1 / per-class / confusion matrix / top words, and saves the trained model
  plus metrics.

## Outputs

Everything generated lands in gitignored folders at the repo root, never committed
(this is why the source folder is `dataset/`, not `data/`: a `data/` folder would
be caught by the `data/` ignore rule):

- `data/document_type/` — the data files (`dataset.jsonl`, `train.jsonl`, `test.jsonl`).
- `artifacts/document_type/` — the trained model (`*.model.joblib`), per-run metrics
  (`*.json`), and a `runs.jsonl` history for comparing runs over time.

All commands are run from the repo root.
