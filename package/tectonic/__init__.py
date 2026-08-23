"""tectonic: deal-document intelligence.

Each feature is a subpackage (the first is `tectonic.document_type`). Subpackages are
imported explicitly, e.g. `from tectonic.document_type import DocumentTypeClassifier`, so
that importing `tectonic` itself stays cheap and never pulls a feature's heavy optional
dependencies (installed via extras, e.g. `pip install tectonic[document-type]`).
"""

__version__ = "0.1.0"
