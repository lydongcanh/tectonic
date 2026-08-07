"""The one row shape every source produces, and how we read/write a dataset.

A "dataset" here is just a list of `Example`s: one document, its type label, and
where it came from. Keeping this shape tiny and explicit means every source
loader (CUAD, ContractNLI, later EDGAR) aims at the same target, and the trainer
never needs to know which source a given row originated from.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

# The document types this dataset currently covers. It grows as we add sources.
LABELS = ("commercial_agreement", "nda", "constitutional", "financial_statements",
          "ip_agreement", "employment_agreement", "lease_agreement",
          "acquisition_agreement")


@dataclass(frozen=True)
class Example:
    """One labelled document.

    `frozen=True` makes it immutable: once a row is built it cannot be changed by
    accident later in the pipeline, which removes a whole class of bugs.
    """

    doc_id: str  # stable id within its source (its file name / title)
    source: str  # which dataset it came from: "cuad", "contract_nli", ...
    type: str  # the document-type label; one of LABELS
    text: str  # the full document text (the model's input)


def write_jsonl(examples: list[Example], path: Path) -> None:
    """Write examples as JSON Lines: one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex)) + "\n")


def read_jsonl(path: Path) -> list[Example]:
    """Read a dataset written by `write_jsonl` back into Examples."""
    lines = path.read_text().splitlines()
    return [Example(**json.loads(line)) for line in lines if line.strip()]
