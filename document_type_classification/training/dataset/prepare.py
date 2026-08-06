"""One entry point that prepares the dataset end to end, in the right order.

Data preparation is three ordered stages, and the order matters:

    1. build_dataset  -> pull sources, filter, dedupe (exact + near) -> dataset.jsonl
    2. split          -> stratified train/test split                -> train/test.jsonl
    3. audit_split    -> verify no near-duplicate leaked across the split

Stage 3 raises and stops the run if it finds leakage, so a green run of this
script means the data is clean and ready to train on. You can still run any
stage on its own while iterating; this just chains them so you do not have to
remember the order.
"""

from __future__ import annotations

import audit_split
import build_dataset
import split


def main() -> None:
    print("STEP 1/3  build dataset")
    build_dataset.main()

    print("\nSTEP 2/3  split train/test")
    split.main()

    print("\nSTEP 3/3  audit split for leakage")
    audit_split.main()  # raises SystemExit if any leak is found

    print("\nOK: dataset built, split, and verified clean.")


if __name__ == "__main__":
    main()
