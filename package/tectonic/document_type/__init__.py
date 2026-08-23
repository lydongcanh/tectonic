"""Document-type classification: predict what kind of legal/deal document some text is.

Public API (re-exported here so callers import from the subpackage, not internal modules):

    from tectonic.document_type import DocumentTypeClassifier, DocumentType, Prediction

    clf = DocumentTypeClassifier.from_pretrained()   # loads the published model
    clf.classify("This Mutual Non-Disclosure Agreement ...")

Requires the `document-type` extra: `pip install tectonic[document-type]`.
"""

from .classifier import DocumentTypeClassifier
from .document_type import DocumentType
from .prediction import Prediction

__all__ = ["DocumentTypeClassifier", "DocumentType", "Prediction"]
