# Document-type taxonomy: rationale

This is the reference for **which document types we classify and why**. Its job is
to stop us re-litigating the taxonomy every few days. If a proposed type cannot
pass the test below, it does not become a class.

Context: this is an M&A / due-diligence document-intelligence system. A document's
type is detected early and then used to route everything after it.

## The test: a type must earn its place

A type is worth classifying only if a distinction genuinely **forks something
downstream**. We judge each candidate on four lenses:

1. **Data** — can we get real, openly-licensed documents to *train* on, and a
   separate real set to *evaluate* on? No data, no honest model.
2. **Text** — is it distinguishable from the document's own text/markdown? (Table
   structure survives in markdown, so "mostly tables" is a real textual signal.)
3. **Human fork** — does a human workflow behave differently? Reviewer/workstream
   routing, due-diligence checklist, cross-document analytics, or an operational
   step (e.g. an NDA gates access to the room).
4. **AI fork** — does a *different AI treatment* apply? A different extraction
   schema/prompt, a different clause set for segmentation, different risk rules,
   or a different playbook to compare against.

Rule of thumb: **if nothing downstream forks on the distinction, do not classify
it.** Detect only at the coarsest level something actually consumes.

## Two resolutions: type and form (they are not parallel)

- **DocumentType** = *what the document is*. This is the single label the model
  predicts.
- **DocumentForm** = *how it is processed*. It is **derived** from the type by a
  fixed lookup, never predicted separately, so it cannot drift from the type.

There are 5 forms (the full set of processing pipelines):

| Form | Processing |
|---|---|
| contract | clause segmentation + clause classification |
| statement | table / number extraction |
| record | events / schema fields |
| report | summarisation |
| correspondence | metadata |

Every type rolls up into exactly one form. So "9 types, 5 forms" does not mean
they line up. The 9 types below live in only **2** of the 5 forms.

## The v1 tier: 9 types

These 9 are the types that pass the test convincingly *and* have real data today.

```
FORM: contract               FORM: statement
  acquisition_agreement         financial_statements
  commercial_agreement
  ip_agreement               FORM: record          -> (none yet)
  employment_agreement       FORM: report          -> (none yet)
  financing_agreement        FORM: correspondence  -> (none yet)
  lease_agreement
  nda
  constitutional
```

Four-lens assessment (honest, including the weak spots):

| Type | Data | Text | Human fork | AI fork | Note |
|---|---|---|---|---|---|
| acquisition_agreement | strong (EDGAR EX-2, MAUD) | strong | strong (the deal doc) | strong (price, indemnity, MAC, earn-out) | built; crisp (plan of merger/merger sub/surviving corp), F1 0.96 |
| financial_statements | strong (EDGAR 10-K) | strong *if tables survive to markdown* | strong (finance) | strong (numbers, not clauses) | solid; depends on table parsing |
| nda | good (ContractNLI) | strong | strong (gates room access) | strong (term, purpose, carve-outs) | solid |
| employment_agreement | strong (EDGAR EX-10) | strong | strong (comp exposure) | strong (comp, change-of-control) | solid |
| financing_agreement | strong (EDGAR EX-10/EX-4) | strong | strong (debt workstream) | strong (covenants, events of default) | solid; internally broad |
| constitutional | strong (EDGAR EX-3) | strong | strong (approval rights) | strong (its OWN clause taxonomy) | solid |
| lease_agreement | good (EDGAR EX-10, ~120) | strong | strong (lease-expiry analytics) | strong (rent, term, break) | built; crisp (lessor/lessee/rent/premises), F1 0.96 |
| commercial_agreement | strong (CUAD ~454) | moderate (umbrella) | moderate (catch-all contract) | moderate (one loose schema) | the residual contract bucket |
| ip_agreement | good (CUAD + EDGAR EX-10 licences) | moderate (overlaps commercial) | good (IP counsel) | strong (licence scope, royalties, ownership) | fuzzy vs commercial (the one real confusion); data no longer thin |

## Known limits (write them down, don't be surprised later)

- **Form coverage is narrow.** 8 of 9 are contracts; only financial_statements is
  a statement. The record / report / correspondence pipelines get no coverage from
  this tier. Building these 9 validates the contract pipeline, little else.
- **`commercial_agreement` is a super-bucket**, not a crisp type. Its real role is
  "a contract that is not one of the specific ones."
- **Expected confusions** (the pairs the classifier will most likely mix up):
  commercial ↔ ip; employment ↔ consulting (consulting is commercial);
  constitutional ↔ shareholders_agreement; a standalone NDA ↔ a confidentiality
  clause embedded in a larger contract.

## Deferred (and why)

The other 14 original types are not dropped forever, they just do not pass the
test *yet* or need a different mechanism.

- **Later tier — real reason, data is the blocker:** shareholders_agreement,
  litigation, insurance_policy, minutes, correspondence, information_memorandum,
  diligence_qa, tax_document. Add once we find/curate real data and a concrete
  downstream fork.
- **Not a text problem — detect by file structure/format instead:**
  financial_model, cap_table. These are spreadsheets; detect them from the file
  being tabular (mime type + markdown table density), not from prose.
- **Folded into `other` for now — no clean downstream fork:** certificate,
  regulatory, report (as a distinct type), disclosure_schedule.

## Adding types later (versioning)

- The **data** is the durable asset. The **model** is a cheap, reproducible
  artifact rebuilt from the data.
- To add a type: add its data (synthetic for training where needed, real for
  evaluation), then **re-run the whole training pipeline from scratch** on all
  types together. Produce a new model version with a new label set.
- We do **not** incrementally fine-tune a new class onto the old model
  (catastrophic forgetting + broken calibration).
- Label sets are **versioned**: v1 = a small set, later versions add types. The
  meaning of `other` shifts each version as former-`other` documents become real
  classes.

## Current status

Eight of the nine v1 types are built and trained (held-out macro-F1 ≈ 0.96):

```
BUILT:   commercial_agreement, nda, constitutional, financial_statements,
         ip_agreement, employment_agreement, lease_agreement, acquisition_agreement
PENDING: financing_agreement
```

We started from the *fuzziest* boundary on purpose (commercial vs ip): if that is
learnable from real data, the crisper ones are easier. It was, and the crisper
classes (constitutional, financial_statements, employment) score ~1.0. The only
meaningful confusion left is ip ↔ commercial, which is genuine, not an artefact.

How each is labelled honestly: CUAD/ContractNLI carry their own type; EDGAR types
use the exhibit type where one is authoritative (EX-3 constitutional, EX-13
financials) and otherwise the filer's own exhibit title (EX-10 licences and
employment agreements). The three pending types are all EDGAR EX-2 / EX-4 / EX-10
and slot into the same machinery.
