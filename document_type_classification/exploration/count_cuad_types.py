"""Count how many distinct contracts CUAD has of each type.

Exploration only. Two things make this less trivial than it sounds:

  1. Each CUAD contract appears many times (once per lawyer-review question),
     so we must dedupe by `title` to count contracts, not rows.
  2. There is no clean "type" column. The contract type is the tail end of the
     filename-style `title`, e.g. "..._EX-10.16_Supply Agreement". We extract
     it crudely and count the raw strings, to see what CUAD actually covers.

The point is to SEE the real distribution (and how messy the raw labels are)
before we decide which types we can learn.
"""

from __future__ import annotations

import re
from collections import Counter

from datasets import load_dataset

CUAD_ID = "theatticusproject/cuad-qa"

# Trailing disambiguation digits CUAD appends, e.g. "Agreement3" or "Agreement (1)".
_TRAILING_NUM = re.compile(r"\s*\(?\d+\)?\s*$")

# Titles that carry an SEC exhibit prefix start like this, e.g. "2020-EX-10.1-...".
_SEC_START = re.compile(r"^\d{4}-EX")


def _strip_sec_prefix(text: str) -> str:
    """Drop a leading SEC exhibit prefix like "2020-EX-10.1-" if present.

    The prefix has no spaces and is joined to the type by a dash, so within the
    first space-delimited chunk everything up to the LAST dash is the prefix.
    We do this without a backtracking regex: find that dash directly.
    """
    if not _SEC_START.match(text):
        return text
    head, sep, rest = text.partition(" ")
    cut = head.rfind("-")
    return head[cut + 1:] + sep + rest if cut != -1 else text


def contract_type(title: str) -> str:
    """Pull a NORMALIZED contract type out of a CUAD title.

    The raw tail is noisy in three ways that split one real type into many
    strings. We remove each so the same type counts as the same type:
        1. an SEC exhibit prefix ("2020-EX-10.1-COOPERATION AGREEMENT"),
        2. trailing disambiguation digits ("Franchise Agreement3"),
        3. mixed casing ("MARKETING AGREEMENT" vs "Marketing Agreement").
    """
    tail = title.rsplit("_", 1)[-1]
    if " - " in tail:
        tail = tail.rsplit(" - ", 1)[-1]
        
    tail = tail.strip().upper()          # casing no longer splits a type in two
    tail = _strip_sec_prefix(tail)       # drop the "2020-EX-10.1-" style prefix
    tail = _TRAILING_NUM.sub("", tail)   # drop the trailing "3", " (1)", etc.
    return tail.strip()


def main() -> None:
    seen_titles: set[str] = set()
    counts: Counter[str] = Counter()

    # CUAD ships two splits (train + test), we want the whole picture, so we
    # walk both and dedupe contracts across them by title.
    for split in ("train", "test"):
        rows = load_dataset(CUAD_ID, split=split, streaming=True, trust_remote_code=True)
        for row in rows:
            title = row["title"]
            if title in seen_titles:
                continue
            
            seen_titles.add(title)
            counts[contract_type(title)] += 1

    print(f"{len(seen_titles)} distinct contracts, {len(counts)} distinct type strings\n")

    singletons = 0
    for type_string, n in counts.most_common():
        if n == 1:
            singletons += 1
            continue
        print(f"{n:4}  {type_string}")
    print(f"\n... plus {singletons} types with a single contract each")


if __name__ == "__main__":
    main()
