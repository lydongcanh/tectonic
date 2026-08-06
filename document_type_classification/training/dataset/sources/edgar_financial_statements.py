"""Load `financial_statements` from SEC EDGAR EX-13 exhibits.

EX-13 is the "annual report to security holders" exhibit: in practice the audited
consolidated financial statements (balance sheet, statements of operations / cash
flows, notes, auditor's report).

The trick is the QUERY. A first attempt with generic phrases and no form filter
surfaced mostly 8-K earnings releases and XBRL and found only ~13 EX-13 docs. But
querying phrases that occur INSIDE these exhibits and restricting to 10-K filings
(where financials-as-a-separate-exhibit live) surfaces EX-13 densely (~30 per
search page). So EX-13 was never sparse; the earlier queries were just wrong.

Sourcing, caching (data/raw/edgar/financial_statements/), and labelling are all
handled by the shared EDGAR helper.
"""

from __future__ import annotations

from collections.abc import Iterator

from dataset import Example
from sources.edgar import load_edgar_exhibits

QUERIES = [
    '"notes to consolidated financial statements"',
    '"report of independent registered public accounting firm"',
    '"consolidated statements of operations"',
]
TARGET = 200
FORMS = "10-K"  # financials filed as a separate exhibit live inside 10-K filings


def load_edgar_financial_statements() -> Iterator[Example]:
    """Yield one Example per distinct-company EX-13 (financial statements) document."""
    return load_edgar_exhibits("financial_statements", "EX-13", QUERIES, TARGET, forms=FORMS)
