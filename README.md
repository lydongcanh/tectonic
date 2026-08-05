# tectonic

Deal-document intelligence, rebuilt from scratch to be understood end to end.

This is a learning-first rewrite of an earlier prototype. The rule here is
simple: no code lands unless it is understood. We add complexity only when we
hit the problem it solves.

## Workstreams

- `document_type_classification/` — given a document's text, predict what kind
  of document it is (an NDA, a lease, a shareholders' agreement, ...).

## Layout convention

Inside each workstream:

- `exploration/` — throwaway scripts to look at data and try things. Messy is
  fine here. Nothing in the product depends on it.
- `training/` — the real, kept code that builds and evaluates the model.

## Setup

```bash
poetry install
```
