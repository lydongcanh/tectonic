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
- **sources/** — one loader per data source, each mapping its raw data onto
  `Example`s:
  - `cuad.py` — CUAD contracts (commercial_agreement, and some ip_agreement).
  - `contract_nli.py` — ContractNLI NDAs.
  - `edgar.py` — shared SEC EDGAR helper: search, per-type caching, a frozen
    manifest, and the junk-page guards. Two ways to label an EDGAR exhibit:
    by its **exhibit type** when a dedicated one exists (EX-3 = constitutional,
    EX-13 = financials), or by the filer's own **`file_description` title** when
    it does not (licences and employment agreements are both EX-10 "material
    contracts", so `title_says("...")` keeps only full agreements of that name).
  - `edgar_constitutional.py`, `edgar_financial_statements.py`,
    `edgar_acquisition.py` (EX-2, labelled by exhibit type), `edgar_ip.py`,
    `edgar_employment.py`, `edgar_lease.py` (EX-10, labelled by title) — thin
    wrappers, one per EDGAR type.

  Adding a type = add a loader here, add the label to `LABELS`, and register the
  loader in `build_dataset.py`.

## model/ — train and evaluate

```bash
poetry run python document_type_classification/training/model/baseline.py
```

- **baseline.py** — TF-IDF + Logistic Regression. Reads the data files, prints
  macro-F1 / per-class report / **per-class bootstrap confidence intervals** /
  confusion matrix / top words, and saves the trained model plus metrics. The
  per-class CI is the honest reading of a small-sample score: a class the bootstrap
  never misclassifies stays at [1.000, 1.000]; a "1.0" on ~25 docs shows an interval
  reaching well below 1.0. (This measures sampling wobble within the test set only,
  not generalisation to other document sources.)
- **inspect_features.py** — on-demand deep look at a trained class's learned
  words, for auditing bias the routine top-15 log is too shallow to show. E.g.
  `inspect_features.py ip_agreement --top 40` or `--grep beijing` (shows every
  feature containing a suspected word, with its weight and rank).

## Current status

Eight classes, macro-F1 ≈ 0.96 (held-out test, with a bootstrap confidence
interval): `commercial_agreement`, `nda`, `constitutional`, `financial_statements`,
`ip_agreement`, `employment_agreement`, `lease_agreement`, `acquisition_agreement`.
The remaining confusion is concentrated at the edges of `commercial_agreement`, the
residual "contract that is not one of the specific ones" bucket, which genuinely
overlaps with ip / lease / acquisition. Not a data artefact.

## Outputs

Everything generated lands in gitignored folders at the repo root, never committed
(this is why the source folder is `dataset/`, not `data/`: a `data/` folder would
be caught by the `data/` ignore rule):

- `data/document_type/` — the data files (`dataset.jsonl`, `train.jsonl`, `test.jsonl`).
- `artifacts/document_type/` — the trained model (`*.model.joblib`), per-run metrics
  (`*.json`), and a `runs.jsonl` history for comparing runs over time.

All commands are run from the repo root.
