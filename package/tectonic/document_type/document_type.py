"""The document types this feature can predict."""

from __future__ import annotations

from enum import StrEnum


class DocumentType(StrEnum):
    """One of the nine document types the classifier recognises.

    A `StrEnum`, so a member compares and serialises as its plain string value
    (e.g. `DocumentType.NDA == "nda"`), which keeps the public API friendly to callers
    that just want the label as text while still giving type-safety to those that want it.
    """

    ACQUISITION_AGREEMENT = "acquisition_agreement"
    COMMERCIAL_AGREEMENT = "commercial_agreement"  # catch-all for "some other contract"
    CONSTITUTIONAL = "constitutional"
    EMPLOYMENT_AGREEMENT = "employment_agreement"
    FINANCIAL_STATEMENTS = "financial_statements"
    FINANCING_AGREEMENT = "financing_agreement"
    IP_AGREEMENT = "ip_agreement"
    LEASE_AGREEMENT = "lease_agreement"
    NDA = "nda"
