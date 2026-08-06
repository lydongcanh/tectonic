"""Load `constitutional` documents (charters / bylaws) from SEC EDGAR EX-3 exhibits.

EX-3 is the constitutional family (certificate/articles of incorporation, bylaws),
so the exhibit type is our authoritative label. Sourcing, caching (under
data/raw/edgar/constitutional/), and labelling are all handled by the shared
EDGAR helper.
"""

from __future__ import annotations

from collections.abc import Iterator

from dataset import Example
from sources.edgar import load_edgar_exhibits

# Phrases only used to SURFACE candidate EX-3 documents, the label is the EX-3 type.
QUERIES = [
    '"certificate of incorporation"',
    '"amended and restated bylaws"',
    '"articles of incorporation"',
]
TARGET = 300  # distinct-company constitutional docs to collect


def load_edgar_constitutional() -> Iterator[Example]:
    """Yield one Example per distinct-company EX-3 (constitutional) document."""
    return load_edgar_exhibits("constitutional", "EX-3", QUERIES, TARGET)
