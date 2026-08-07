"""Load `ip_agreement` (licence agreements) from SEC EDGAR EX-10 exhibits.

There is NO authoritative "IP agreement" exhibit type on EDGAR: licences are filed
as EX-10 "material contracts", the same bucket as supply, employment, and lease
agreements. So the exhibit type alone cannot label them. What CAN label them,
independently of the document body, is the filer's own exhibit title in
`file_description` (e.g. "PATENT LICENSE AGREEMENT"). Selecting on that title is the
same idea as EX-3/EX-13: a human categorised the document; we did not infer it from
the words the classifier keys on. That distinction is the whole point here, because
selecting IP docs by body words like "licensor"/"royalty" would be circular (we
would be hand-picking the very evidence the model then "discovers") and would
inflate the score dishonestly.

We keep only FULL licence agreements. The title must say "LICENSE AGREEMENT" and
must not be an amendment / consent / assignment / termination of one: those bodies
are short and procedural ("Section 3.2 is deleted and replaced..."), unrepresentative
of a licence agreement, and read like a generic commercial amendment, which would
worsen the ip <-> commercial confusion rather than fix it.

Why add these at all: CUAD gave only 43 IP contracts, dominated by a few filers
(SINA/Leju/Fox content licences), so the model learned company names as IP tells
(beijing, sina, leju...). These EDGAR agreements span pharma, biotech, software,
chemicals, medical devices and more, drowning out that bias with genuine, varied
licensing language.

Sourcing, caching (data/raw/edgar/ip_agreement/), and the min-length / junk-page
guards are all handled by the shared EDGAR helper.
"""

from __future__ import annotations

from collections.abc import Iterator

from dataset import Example
from sources.edgar import load_edgar_exhibits, title_says

QUERIES = ['"license agreement"']  # relevance search; the LABEL is the title, below
TARGET = 120
MAX_OFFSET = 5000  # page deep enough to still reach TARGET after dropping amendments


def load_edgar_ip() -> Iterator[Example]:
    """Yield one Example per distinct-company full licence agreement (EX-10)."""
    return load_edgar_exhibits(
        "ip_agreement", "EX-10", QUERIES, TARGET,
        max_offset=MAX_OFFSET, description_ok=title_says("license agreement"),
    )
