"""Peek at real contracts from CUAD.

This is EXPLORATION code: throwaway, meant only to let us see the raw data with
our own eyes before we build anything. It is not part of the model.

CUAD (Contract Understanding Atticus Dataset) is a public, openly-licensed
collection of ~500 real commercial contracts. Each row gives us:
  - `context`: the full text of a contract        <- this is the document
  - `title`  : the original filename, which names the contract type

We stream a few rows (so we do NOT download the whole dataset) and print them.
Network required: turn Cloudflare WARP OFF first.

Run:
    poetry run python document_type_classification/exploration/peek_cuad.py
"""

from __future__ import annotations

from datasets import load_dataset

CUAD_ID = "theatticusproject/cuad-qa"
HOW_MANY = 5  # how many distinct contracts to look at


def main() -> None:
    # streaming=True reads rows one at a time over the network instead of
    # downloading the whole dataset. trust_remote_code=True lets CUAD run the
    # small loading script it ships with.
    rows = load_dataset(CUAD_ID, split="test", streaming=True, trust_remote_code=True)

    seen_titles: set[str] = set()
    shown = 0
    for row in rows:
        title = row["title"]
        # CUAD repeats each contract across many question/answer rows, so we
        # skip titles we have already printed and show each contract once.
        if title in seen_titles:
            continue
        
        seen_titles.add(title)

        context = row["context"]
        print("=" * 80)
        print(f"TITLE : {title}")
        print(f"LENGTH: {len(context):,} characters")
        print("FIRST 400 CHARACTERS OF THE CONTRACT:")
        print(context[:400].strip())
        print()

        shown += 1
        if shown >= HOW_MANY:
            break


if __name__ == "__main__":
    main()
