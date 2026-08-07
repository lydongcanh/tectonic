"""Load `lease_agreement` from SEC EDGAR EX-10 exhibits.

Leases, like licences and employment agreements, are EX-10 "material contracts"
with no dedicated exhibit type, so we label by the filer's own `file_description`
title via `title_says("lease agreement")` (see `sources.edgar`). That keeps full
lease agreements and drops amendments/assignments of them, including a real
misspelling the scout found ("FIFTH AMENDEMNT TO LEASE AGREEMENT"), which the
preposition rule catches where a keyword check would not.

Design note: we require the title "LEASE AGREEMENT" and do NOT try to also catch
bare "LEASE" / "OFFICE LEASE" titles. Two reasons: "LEASE" is a substring of
"RELEASE"/"SUBLEASE" (a false-match trap), and the title is only our LABEL, not a
feature. The model learns from the body, and an office lease's body (lessor,
lessee, premises, rent, term) is the same as a lease agreement's, so the gate stays
clean without hurting how well the class generalises. A few translated foreign
leases ("... TRANSLATION OF LEASE AGREEMENT") are dropped conservatively.

Sourcing, caching (data/raw/edgar/lease_agreement/), and the min-length / junk-page
guards are all handled by the shared EDGAR helper.
"""

from __future__ import annotations

from collections.abc import Iterator

from dataset import Example
from sources.edgar import load_edgar_exhibits, title_says

QUERIES = ['"lease agreement"']  # relevance search; the LABEL is the title
TARGET = 150
MAX_OFFSET = 3000  # page deep enough to still reach TARGET after dropping amendments


def load_edgar_lease() -> Iterator[Example]:
    """Yield one Example per distinct-company full lease agreement (EX-10)."""
    return load_edgar_exhibits(
        "lease_agreement", "EX-10", QUERIES, TARGET,
        max_offset=MAX_OFFSET, description_ok=title_says("lease agreement"),
    )
