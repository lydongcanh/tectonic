"""Load ContractNLI as Examples: every document is an NDA.

ContractNLI pairs each NDA with many entailment hypotheses, so the document text
repeats across rows. We dedupe by document id and yield each NDA once.

Licence note: the original ContractNLI is CC BY 4.0, but this mirror does not
declare a licence. Fine for learning the pipeline; must be resolved before we
train anything we intend to publish.
"""

from __future__ import annotations

from collections.abc import Iterator

from datasets import load_dataset

from dataset import Example

CONTRACT_NLI_ID = "reuben256/contract-nli"


def load_contract_nli() -> Iterator[Example]:
    """Yield one Example per distinct NDA (deduped by document id)."""
    seen: set[str] = set()
    for split in ("train", "validation", "test"):
        try:
            rows = load_dataset(CONTRACT_NLI_ID, split=split, streaming=True)
        except Exception:
            continue  # a split this mirror does not provide
        
        for row in rows:
            doc_id = row.get("document_id") or row.get("file_name")
            if doc_id in seen:
                continue

            seen.add(doc_id)

            yield Example(
                doc_id=str(doc_id),
                source="contract_nli",
                type="nda",
                text=row["text"],
            )
