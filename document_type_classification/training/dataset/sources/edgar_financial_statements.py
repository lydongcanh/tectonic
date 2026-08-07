"""Load `financial_statements` from SEC EDGAR EX-13 exhibits.

EX-13 is the "annual report to security holders" exhibit: in practice the audited
consolidated financial statements (balance sheet, statements of operations / cash
flows, notes, auditor's report).

The trick is the QUERY. A first attempt with generic phrases and no form filter
surfaced mostly 8-K earnings releases and XBRL and found only ~13 EX-13 docs.
Querying phrases that occur INSIDE these exhibits and restricting to 10-K filings
(where financials-as-a-separate-exhibit live) surfaces EX-13 densely. So EX-13 was
never sparse; the earlier queries were just wrong.

Second problem, found by inspection: 69 of the first 78 docs (88%) closed their
books on 31 December, so the model learned "december 31" as a shortcut for
"financial statements" instead of the accounting itself. We fix that here by
targeting a spread of fiscal year-ends. A year-end phrase alone does not surface
EX-13, so each query COMBINES the reliable EX-13 phrase with a balance-sheet date
("...financial statements" AND "June 30,"). `max_per_query` then caps each
year-end's contribution so the (far more common) December filers cannot refill the
set and bring the shortcut back. June and September are the dense non-December
buckets; July/August/October add a thin spread; March/April/January are absent
because those filers use moving 52/53-week dates that no fixed phrase matches.

Sourcing, caching (data/raw/edgar/financial_statements/), and labelling are all
handled by the shared EDGAR helper.
"""

from __future__ import annotations

from collections.abc import Iterator

from dataset import Example
from sources.edgar import load_edgar_exhibits

# The phrase that reliably surfaces EX-13; every query pins a year-end date onto it.
_EX13 = '"notes to consolidated financial statements"'
QUERIES = [
    f'{_EX13} "June 30,"',
    f'{_EX13} "September 30,"',
    f'{_EX13} "July 31,"',
    f'{_EX13} "August 31,"',
    f'{_EX13} "October 31,"',
    f'{_EX13} "December 31,"',  # last, and capped, so December cannot dominate
]
TARGET = 200
MAX_PER_QUERY = 40  # no single year-end may contribute more than this
FORMS = "10-K"  # financials filed as a separate exhibit live inside 10-K filings


def load_edgar_financial_statements() -> Iterator[Example]:
    """Yield one Example per distinct-company EX-13 (financial statements) document."""
    return load_edgar_exhibits(
        "financial_statements", "EX-13", QUERIES, TARGET,
        forms=FORMS, max_per_query=MAX_PER_QUERY,
    )
