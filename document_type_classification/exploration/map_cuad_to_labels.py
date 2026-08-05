"""Map CUAD's contract types onto OUR document-type labels, and count.

Exploration only. CUAD's ~25 fine-grained contract types (see
count_cuad_types.py) are, in our taxonomy, almost all a single label:
`commercial_agreement`. Two carve-outs matter:

  - the licensing / IP cluster        -> `ip_agreement`
  - SEC joint-filing formalities      -> `other`

Everything else defaults to `commercial_agreement`. That default is correct
here because CUAD is entirely commercial contracts: even the rows whose title
we failed to parse into a clean type are still commercial contracts underneath.

The output tells us how many REAL examples CUAD can give each of our labels,
which is what decides where real evaluation is even possible.

Run:
    poetry run python document_type_classification/exploration/map_cuad_to_labels.py
"""

from __future__ import annotations

from collections import Counter, defaultdict

from datasets import load_dataset

# Reuse the (throwaway) extractor from the sibling script instead of copying it.
from count_cuad_types import CUAD_ID, contract_type

# The only CUAD types we pull OUT of the commercial default, each keyed to the
# label it belongs to in our taxonomy. This mapping is a judgment call, so it is
# written out explicitly rather than guessed by a rule.
_IP_TYPES = {
    "CONTENT LICENSE AGREEMENT",
    "INTELLECTUAL PROPERTY AGREEMENT",
    "TRADEMARK LICENSE AGREEMENT",
    "LICENSE AGREEMENT",
}
_OTHER_TYPES = {
    "JOINT FILING AGREEMENT",
}


def our_label(cuad_type: str) -> str:
    """Map one normalized CUAD type onto one of our taxonomy labels."""
    if cuad_type in _IP_TYPES:
        return "ip_agreement"
    if cuad_type in _OTHER_TYPES:
        return "other"
    return "commercial_agreement"


def main() -> None:
    seen_titles: set[str] = set()
    per_label: Counter[str] = Counter()
    types_feeding: dict[str, Counter[str]] = defaultdict(Counter)

    for split in ("train", "test"):
        rows = load_dataset(CUAD_ID, split=split, streaming=True, trust_remote_code=True)
        for row in rows:
            title = row["title"]
            if title in seen_titles:
                continue
            seen_titles.add(title)

            cuad_type = contract_type(title)
            label = our_label(cuad_type)
            per_label[label] += 1
            types_feeding[label][cuad_type] += 1

    print(f"{len(seen_titles)} contracts mapped onto our labels:\n")
    for label, n in per_label.most_common():
        print(f"{n:4}  {label}")

    print("\nwhat feeds each label (top CUAD types):")
    for label, _ in per_label.most_common():
        print(f"\n== {label} ==")
        for cuad_type, n in types_feeding[label].most_common(8):
            print(f"   {n:4}  {cuad_type}")


if __name__ == "__main__":
    main()
