"""Eyeball a sample of CUAD contracts: does the TEXT match the title's type?

Exploration only. So far we trusted the title to tell us a contract's type,
and we never read the actual document. This prints a spread of contracts (some
commercial, some IP, some "other") with the first chunk of their real text next
to the title, so a human can confirm the labels are real before we train on
them. Garbage labels in, garbage model out.

We deliberately over-sample the rarer IP and "other" buckets, because that is
where the title-based mapping is most likely to be wrong.
"""

from __future__ import annotations

from collections import defaultdict

from datasets import load_dataset

from document_type_classification.exploration.cuad.count_cuad_types import CUAD_ID, contract_type
from document_type_classification.exploration.cuad.map_cuad_to_labels import our_label

# How many examples to pull for each of our labels.
CAPS = {"commercial_agreement": 6, "ip_agreement": 4, "other": 2}
PREVIEW_CHARS = 600


def main() -> None:
    seen_titles: set[str] = set()
    collected: dict[str, list[tuple[str, str, str]]] = defaultdict(list)

    for split in ("train", "test"):
        rows = load_dataset(CUAD_ID, split=split, streaming=True, trust_remote_code=True)
        for row in rows:
            title = row["title"]
            if title in seen_titles:
                continue
            seen_titles.add(title)

            cuad_type = contract_type(title)
            label = our_label(cuad_type)
            if len(collected[label]) < CAPS[label]:
                collected[label].append((title, cuad_type, row["context"]))

            if all(len(collected[lbl]) >= cap for lbl, cap in CAPS.items()):
                break
        if all(len(collected[lbl]) >= cap for lbl, cap in CAPS.items()):
            break

    for label in CAPS:
        for title, cuad_type, context in collected[label]:
            preview = " ".join(context.split())[:PREVIEW_CHARS]
            print("=" * 90)
            print(f"OUR LABEL : {label}")
            print(f"CUAD TYPE : {cuad_type}")
            print(f"TITLE     : {title}")
            print(f"TEXT      : {preview}")
            print()


if __name__ == "__main__":
    main()
