"""The result of classifying one document."""

from __future__ import annotations

from dataclasses import dataclass

from .document_type import DocumentType


@dataclass(frozen=True)
class Prediction:
    """A single document's predicted type.

    `confidence` is the model's probability for `label`. It is deliberately NOT calibrated
    (the model is under-confident, see the model card), so choose any accept/escalate
    threshold empirically for your data. `scores` holds the full per-type distribution,
    ordered most- to least-likely, for callers that want to inspect runners-up or apply
    their own gate.
    """

    label: DocumentType
    confidence: float
    scores: dict[DocumentType, float]
