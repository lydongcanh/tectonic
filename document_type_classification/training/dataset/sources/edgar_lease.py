"""Load `lease_agreement` from SEC EDGAR EX-10 exhibits.

Leases, like licences and employment agreements, are EX-10 "material contracts"
with no dedicated exhibit type, so we label by the filer's own `file_description`
title via `title_says("lease agreement")` (see `sources.edgar`). That keeps full
lease agreements and drops amendments/assignments of them, including a real
misspelling the scout found ("FIFTH AMENDEMNT TO LEASE AGREEMENT"), which the
preposition rule catches where a keyword check would not.

Design note: we require the title "LEASE AGREEMENT" and do NOT try to also catch
bare "LEASE" / "OFFICE LEASE" titles, because the title is only our LABEL, not a
feature: the model learns from the body (lessor, lessee, premises, rent, term), so a
narrower title gate does not hurt how well the class generalises.

The title is matched as a substring, so "SUBLEASE AGREEMENT" is kept, correctly, a
sublease IS a lease. But that substring also matches "RELEASE AGREEMENT" (a release
of claims, not a lease), so we pass `disallow=("RELEASE",)` to reject those while
keeping subleases. (An earlier version of this note wrongly claimed requiring the
two-word phrase avoided the "RELEASE"/"SUBLEASE" trap; it does not, "LEASE AGREEMENT"
is itself a substring of both.) A few translated foreign leases
("... TRANSLATION OF LEASE AGREEMENT") are dropped conservatively.

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
        "lease_agreement", "EX-10", QUERIES, TARGET, max_offset=MAX_OFFSET,
        description_ok=title_says("lease agreement", disallow=("RELEASE",)),
    )
