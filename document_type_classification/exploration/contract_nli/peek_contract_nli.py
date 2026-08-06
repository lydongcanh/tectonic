"""Peek at ContractNLI: confirm it gives clean, full NDA documents, and count them.

Exploration only. ContractNLI is a set of non-disclosure agreements, each paired
with many entailment hypotheses. So, exactly like CUAD repeats a contract across
its questions, ContractNLI repeats an NDA across its hypotheses. For document-type
we only care about two things: the full text, and that every document is an `nda`.
So we dedupe by document and treat each distinct one as a single NDA.

Dataset: reuben256/contract-nli (a script-free mirror of Stanford's ContractNLI).
The original ContractNLI is licensed CC BY 4.0; we must confirm this mirror carries
the same licence before using it to train anything.

Run:
    poetry run python document_type_classification/exploration/contract_nli/peek_contract_nli.py
"""

from __future__ import annotations

from datasets import load_dataset

DATASET_ID = "reuben256/contract-nli"
HOW_MANY = 4          # distinct NDAs to print
PREVIEW_CHARS = 400


def main() -> None:
    seen_docs: set[str] = set()
    shown = 0

    # ContractNLI is split by document, so counting distinct doc ids across all
    # splits gives the true number of NDAs available.
    for split in ("train", "validation", "test"):
        try:
            rows = load_dataset(DATASET_ID, split=split, streaming=True)
        except Exception:
            continue  # split not present in this mirror
        for row in rows:
            doc_id = row.get("document_id") or row.get("file_name")
            if doc_id in seen_docs:
                continue
            
            seen_docs.add(doc_id)

            if shown < HOW_MANY:
                text = row["text"]
                print("=" * 90)
                print(f"FILE : {row.get('file_name')}")
                print(f"LEN  : {len(text):,} chars")
                print(f"TEXT : {' '.join(text.split())[:PREVIEW_CHARS]}")
                print()
                shown += 1

    print(f"{len(seen_docs)} distinct NDA documents in ContractNLI")


if __name__ == "__main__":
    main()
