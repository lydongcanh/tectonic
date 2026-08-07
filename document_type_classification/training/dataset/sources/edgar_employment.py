"""Load `employment_agreement` from SEC EDGAR EX-10 exhibits.

Like licence agreements, employment agreements have no dedicated exhibit type: they
are EX-10 "material contracts". So we label them the same independent way, by the
filer's own exhibit title in `file_description` ("EMPLOYMENT AGREEMENT"), not by the
body text the model will learn from. See `sources.edgar.title_says` for the rule
that keeps only full agreements and drops amendments/consents/assignments (including
abbreviations like "AMENDS. TO EMPLOYMENT AGREEMENT", which a naive keyword filter
would have let through).

Employment agreements are abundant on EDGAR. The taxonomy expects some confusion
with consulting/commercial agreements (a consulting agreement is commercial), so the
confusion matrix and top features are the things to check after training.

Sourcing, caching (data/raw/edgar/employment_agreement/), and the min-length /
junk-page guards are all handled by the shared EDGAR helper.
"""

from __future__ import annotations

from collections.abc import Iterator

from dataset import Example
from sources.edgar import load_edgar_exhibits, title_says

QUERIES = ['"employment agreement"']  # relevance search; the LABEL is the title
TARGET = 150
MAX_OFFSET = 3000  # page deep enough to still reach TARGET after dropping amendments


def load_edgar_employment() -> Iterator[Example]:
    """Yield one Example per distinct-company full employment agreement (EX-10)."""
    return load_edgar_exhibits(
        "employment_agreement", "EX-10", QUERIES, TARGET,
        max_offset=MAX_OFFSET, description_ok=title_says("employment agreement"),
    )
