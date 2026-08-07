"""Load `acquisition_agreement` from SEC EDGAR EX-2 exhibits.

EX-2 is the SEC exhibit type for a "plan of acquisition, reorganization,
arrangement, liquidation or succession", in practice merger agreements and asset /
stock purchase agreements. So unlike licences and employment agreements (EX-10
material contracts, where the exhibit type says nothing about the kind of
contract), here the EXHIBIT TYPE is itself the authoritative label, like EX-3
(constitutional) and EX-13 (financials).

Two consequences shape this loader:

  * The prefix must be "EX-2." WITH THE DOT. Plain "EX-2" also matches EX-21
    (subsidiaries), EX-23 (auditor consent), EX-24 (power of attorney), and so on,
    none of which are acquisition agreements. "EX-2." matches EX-2.1 ... EX-2.10.
  * We label by exhibit type, not by title: most full merger agreements have no
    descriptive `file_description` (just "EX-2.1" or blank). So we default-accept
    and use `title_not_modification()` only to drop the sub-documents that DO carry
    titles ("AMENDMENT NO. 1 TO ...", "ADDENDUM ... TO ...").

The queries just surface EX-2-bearing filings densely across deal types (mergers,
asset and stock purchases); the EX-2. prefix is what actually selects the label.

Sourcing, caching (data/raw/edgar/acquisition_agreement/), and the min-length /
junk-page guards are all handled by the shared EDGAR helper.
"""

from __future__ import annotations

from collections.abc import Iterator

from dataset import Example
from sources.edgar import load_edgar_exhibits, title_not_modification

QUERIES = [
    '"agreement and plan of merger"',
    '"asset purchase agreement"',
    '"stock purchase agreement"',
]
TARGET = 150
MAX_OFFSET = 3000


def load_edgar_acquisition() -> Iterator[Example]:
    """Yield one Example per distinct-company acquisition agreement (EX-2.x)."""
    return load_edgar_exhibits(
        "acquisition_agreement", "EX-2.", QUERIES, TARGET,
        max_offset=MAX_OFFSET, description_ok=title_not_modification(),
    )
