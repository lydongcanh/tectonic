"""Load `financing_agreement` from SEC EDGAR EX-4 debt instruments.

Scope decision (see the taxonomy doc): financing_agreement is built from DEBT
instruments, indentures, notes, debentures, not from bank credit agreements.
Credit/loan agreements are filed as EX-10 (the generic "material contracts" bucket)
with mostly blank titles, so the only way to label them would be to match "credit
agreement" in the BODY, the circular body-word selection we avoid. EX-4 instead is
a DEDICATED exhibit type ("instruments defining the rights of security holders"),
so the exhibit type is an authoritative label, like EX-2 / EX-3 / EX-13.

We therefore label by exhibit type (prefix "EX-4." with the dot, so we do not catch
stray "EX-40"-style types) and default-accept, since full indentures often have no
descriptive title. Two tweaks specific to debt:

  * The QUERIES are debt terms (indenture / senior notes / debenture) so the EX-4
    documents we pull are debt instruments, not the warrants and rights agreements
    that EX-4 also covers (those are equity, and would make the class incoherent).
  * We keep SUPPLEMENTAL indentures (a supplemental indenture is a full financing
    document with its own covenants), dropping only genuine amendments, so the
    modification filter here excludes "SUPPLEMENT" from the drop list.

Sourcing, caching (data/raw/edgar/financing_agreement/), and the min-length /
junk-page guards are all handled by the shared EDGAR helper.
"""

from __future__ import annotations

from collections.abc import Iterator

from dataset import Example
from sources.edgar import load_edgar_exhibits, title_not_modification

QUERIES = ['"indenture"', '"senior notes"', '"debenture"']
TARGET = 150
MAX_OFFSET = 3000


def load_edgar_financing() -> Iterator[Example]:
    """Yield one Example per distinct-company EX-4 debt instrument.

    `keep_supplements=True`: a supplemental indenture is a full, substantive debt
    document (a new note series with its own covenants), so unlike other types we do
    not treat "supplement" as a modification; we still drop true amendments.
    """
    return load_edgar_exhibits(
        "financing_agreement", "EX-4.", QUERIES, TARGET,
        max_offset=MAX_OFFSET, description_ok=title_not_modification(keep_supplements=True),
    )
