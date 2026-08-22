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
    `edgar_acquisition.py` and `edgar_financing.py` (EX-2 / EX-4, labelled by
    exhibit type), `edgar_ip.py`, `edgar_employment.py`, `edgar_lease.py` (EX-10,
    labelled by the filer's title) — thin wrappers, one per EDGAR type.

  Adding a type = add a loader here, add the label to `LABELS`, and register the
  loader in `build_dataset.py`.

## model/ — train and evaluate

Two models are trained on the same data. TF-IDF is the **baseline** (the permanent
reference bar and our glass-box interpretability tool); embeddings + LogReg is the
**v1 production model**, chosen on evidence from the `evaluation/` probes.

```bash
# baseline (reference / glass-box)
poetry run python document_type_classification/training/model/baseline.py
# v1 production model
poetry run python document_type_classification/training/model/embedding_classifier.py
```

- **baseline.py** — TF-IDF + Logistic Regression. Reads the data files, prints
  macro-F1 / per-class report / **per-class bootstrap confidence intervals** /
  confusion matrix / top words, and saves the trained model plus metrics. The
  per-class CI is the honest reading of a small-sample score: a class the bootstrap
  never misclassifies stays at [1.000, 1.000]; a "1.0" on ~25 docs shows an interval
  reaching well below 1.0. (This measures sampling wobble within the test set only,
  not generalisation to other document sources.) Because it is linear over words it
  stays useful even now: it is how we read *which words* drive a class, so it remains
  the interpretability tool and the bar the production model is measured against.
- **embedding_classifier.py** — the v1 **production** model: the same
  LogisticRegression, on a dense semantic vector per document instead of word counts.
  Same evaluation (macro-F1, per-class bootstrap CI, confusion matrix). macro-F1 ≈ 0.940
  in-distribution, slightly under TF-IDF's 0.968, but it generalises better across
  sources, which is the property that matters for deployment (see status below and
  `evaluation/`). The saved metadata records the encoder and chunk parameters, so
  inference cannot silently drift from how the model was trained.
- **embedding.py** — the embedding engine behind it: loads the frozen encoder
  (`all-mpnet-base-v2`), splits each document into word windows, embeds and mean-pools
  them into one vector (so the short-context encoder still sees roughly the whole
  document, the way TF-IDF does), and caches vectors per split so re-training is instant.
- **inspect_features.py** — on-demand deep look at a trained class's learned
  words, for auditing bias the routine top-15 log is too shallow to show. E.g.
  `inspect_features.py ip_agreement --top 40` or `--grep beijing` (shows every
  feature containing a suspected word, with its weight and rank).

## evaluation/ — should we trust the scores? (read-only probes)

Sibling of `training/`, at the workstream root. These scripts do not build the shipped
model; they interrogate it, so they answer "should we trust the scores?" rather than
"how do we train it?". This is where the embeddings-vs-TF-IDF decision was made.

- **embeddings_probe.py** — embeddings vs TF-IDF in-distribution (the sanity check).
- **embeddings_generalization.py** — the generalisation proxies: ip cross-source recall
  and the out-of-source spot-check.
- **encoder_bakeoff.py** — mpnet vs bge-large vs frozen LegalBERT on the same proxies;
  this is what selected the production encoder.
- **ip_source_transfer.py**, **oos_eval.py**, **risk_coverage.py**, **error_analysis.py**
  — the TF-IDF-era probes the above build on: cross-source transfer, out-of-source
  scoring, the confidence-gate risk-coverage curve, and per-error inspection.

## Current status

All nine v1 classes are built (`commercial_agreement`, `nda`, `constitutional`,
`financial_statements`, `ip_agreement`, `employment_agreement`, `lease_agreement`,
`acquisition_agreement`, `financing_agreement`), and two models are trained on the same
data:

- **Baseline (TF-IDF + LogReg):** macro-F1 ≈ 0.968 in-distribution. Kept as the
  reference bar and interpretability tool.
- **v1 production (embeddings + LogReg):** macro-F1 ≈ 0.940 in-distribution.

The production model being the *lower* in-distribution scorer is deliberate.
In-distribution scores are the ones we trust least: every class is sourced from
essentially one origin (EDGAR / EDGAR-derived), so a bag-of-words model can score well
by keying on corpus house style rather than document meaning. The `evaluation/` probes
tested the property that actually matters for deployment on non-EDGAR documents,
generalisation across sources, and embeddings won clearly:

- ip cross-source recall (train on EDGAR ip, test on unseen CUAD ip): **0.63** vs
  TF-IDF's **0.49**.
- out-of-source docs (genuinely non-EDGAR): both models correct, but embeddings far more
  confident (mean 0.85 vs 0.48).

An encoder bake-off (`all-mpnet-base-v2` vs `bge-large` vs frozen LegalBERT) picked
mpnet: it tied the strongest modern encoder on generalisation within noise while being
more confident, smaller, and faster; frozen LegalBERT lost on every metric (a raw
masked-LM makes poor pooled document vectors). All encoders miss the *same* handful of
CUAD content-licences that genuinely read as commercial, a data/label ceiling, not an
encoder problem.

The remaining confusion is concentrated at the edges of `commercial_agreement`, the
residual "contract that is not one of the specific ones" bucket, which genuinely overlaps
the specific types. The honest next milestones are outside this repo: validation on real
target-domain (non-EDGAR) documents, which needs data governance, and the
confidence-gated cascade (accept confident ML predictions, escalate the rest to an LLM).

## Outputs

Everything generated lands in gitignored folders at the repo root, never committed
(this is why the source folder is `dataset/`, not `data/`: a `data/` folder would
be caught by the `data/` ignore rule):

- `data/document_type/` — the data files (`dataset.jsonl`, `train.jsonl`, `test.jsonl`).
- `artifacts/document_type/` — the trained model (`*.model.joblib`), per-run metrics
  (`*.json`), and a `runs.jsonl` history for comparing runs over time.

All commands are run from the repo root.
