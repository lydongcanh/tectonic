"""The document-type classifier: load a trained model and classify document text."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ._embedding import embed_documents
from .document_type import DocumentType
from .prediction import Prediction

DEFAULT_MODEL = "lydongcanh/tectonic-doctype"  # the published model on the Hugging Face Hub
_CONFIG_FILE = "config.json"
_HEAD_FILE = "classifier.skops"


class DocumentTypeClassifier:
    """Predict a document's type from its text.

    The model is a logistic-regression head on top of frozen sentence embeddings. Load it
    with `from_pretrained` (from the Hub or a local directory), then call `classify` /
    `classify_batch`. The encoder is loaded lazily on the first prediction and reused.
    """

    def __init__(self, head, encoder_name: str, words_per_chunk: int, chunk_cap: int):
        self._head = head
        self._encoder_name = encoder_name
        self._words_per_chunk = words_per_chunk
        self._chunk_cap = chunk_cap
        self._encoder = None

    @classmethod
    def from_pretrained(cls, model: str = DEFAULT_MODEL) -> "DocumentTypeClassifier":
        """Load from a Hugging Face repo id or a local directory.

        A path that exists on disk is used directly; otherwise the two model files are
        fetched from the Hub (and cached) by repo id. All preprocessing parameters come
        from the model's own config, so inference cannot drift from training.
        """
        local = Path(model)
        if local.is_dir():
            config_path, head_path = local / _CONFIG_FILE, local / _HEAD_FILE
        else:
            from huggingface_hub import hf_hub_download

            config_path = Path(hf_hub_download(model, _CONFIG_FILE))
            head_path = Path(hf_hub_download(model, _HEAD_FILE))

        config = json.loads(config_path.read_text())

        import skops.io as sio

        # skops loads only the types the model author vetted at build time (recorded in the
        # config), which is why it is safer than pickle.
        head = sio.load(head_path, trusted=config.get("trusted_types", []))
        return cls(head, config["encoder"], config["words_per_chunk"], config["chunk_cap"])

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self._encoder_name)
        return self._encoder

    def classify(self, text: str) -> Prediction:
        """Classify one document."""
        return self.classify_batch([text])[0]

    def classify_batch(self, texts: list[str]) -> list[Prediction]:
        """Classify many documents in a single embedding pass (much faster than one call
        each when you have several)."""
        x = embed_documents(self._get_encoder(), texts, self._words_per_chunk, self._chunk_cap)
        proba = self._head.predict_proba(x)
        classes = [DocumentType(c) for c in self._head.classes_]

        predictions = []
        for row in proba:
            order = np.argsort(row)[::-1]
            top = int(order[0])
            scores = {classes[j]: float(row[j]) for j in order}
            predictions.append(
                Prediction(label=classes[top], confidence=float(row[top]), scores=scores)
            )
        return predictions
