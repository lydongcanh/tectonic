# training

The real, kept code for the document-type classifier. Two phases live here: first
preparing a clean dataset, then (next) training and measuring a model on it.

## Data preparation

The pipeline is three ordered stages. Run them all with one command:

```bash
poetry run python document_type_classification/training/prepare.py
```

That runs, in order:

1. **build_dataset.py** — pull each source, keep only modelled labels, drop exact
   and near-duplicate documents, write `data/document_type/dataset.jsonl`.
2. **split.py** — stratified train/test split (same class mix on both sides),
   writes `train.jsonl` and `test.jsonl`.
3. **audit_split.py** — check no near-duplicate leaked across the split. **Fails
   the run** if it finds one, so a clean finish means the data is safe to train on.

You can also run any stage on its own while iterating.

### Supporting modules (not run directly)

- **dataset.py** — the `Example` row shape (`doc_id`, `source`, `type`, `text`),
  the `LABELS` we currently model, and JSONL read/write.
- **fingerprint.py** — the near-duplicate sketch, shared by build (to remove
  near-dups) and audit (to check they are gone), so both agree on "near-duplicate".
- **sources/** — one loader per data source (`cuad.py`, `contract_nli.py`). Each
  maps its raw dataset onto `Example`s. Adding a new type = add a file here.

Output lands in `data/` (gitignored): we never commit the documents, only the
code that regenerates them.

## Model (coming next)

Training and evaluation code. When these files arrive we will split this folder
into `data/` and `model/` subfolders; today it is small enough to stay flat.
