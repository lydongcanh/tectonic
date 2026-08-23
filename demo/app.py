"""Gradio demo for the tectonic document-type classifier.

Paste a legal/deal document and see the predicted type with the full per-type confidence.

Run it:
    poetry install --all-extras --with demo
    poetry run python demo/app.py

By default it loads the published model from the Hugging Face Hub. To run offline against
a local model bundle instead, set TECTONIC_MODEL to that directory, e.g.:
    TECTONIC_MODEL=document_type_classification/hf_release/tectonic-doctype poetry run python demo/app.py
"""

from __future__ import annotations

import os

import gradio as gr

from tectonic.document_type import DocumentTypeClassifier

MODEL = os.environ.get("TECTONIC_MODEL", "lydongcanh/tectonic-doctype")
classifier = DocumentTypeClassifier.from_pretrained(MODEL)

EXAMPLES = [
    ["This Mutual Non-Disclosure Agreement is entered into by and between the parties to "
     "protect confidential information disclosed in connection with evaluating a potential "
     "business relationship. Each party agrees to hold the other's Confidential Information "
     "in strict confidence and not to disclose it to any third party."],
    ["This Lease Agreement is made between the Landlord and the Tenant for the premises "
     "located at 100 Main Street. The Tenant shall pay monthly rent in advance, maintain "
     "the leased premises, and surrender possession at the end of the term."],
    ["This Employment Agreement sets forth the terms of the Executive's employment with the "
     "Company, including base salary, annual bonus, benefits, vacation, and obligations upon "
     "termination of employment."],
]


def classify(text: str) -> dict[str, float]:
    """Return {type: confidence} for gr.Label to render as a ranked bar chart."""
    if not text or not text.strip():
        return {}
    prediction = classifier.classify(text)
    return {str(label): score for label, score in prediction.scores.items()}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=14, label="Document text",
                      placeholder="Paste a legal or deal document here..."),
    outputs=gr.Label(num_top_classes=9, label="Predicted document type"),
    title="Tectonic — Document Type Classifier",
    description=("Classifies an English legal / deal document into one of nine types. "
                 "Confidence is not calibrated, so treat borderline scores with care. "
                 "Trained on public filings (EDGAR / CUAD / ContractNLI)."),
    examples=EXAMPLES,
    flagging_mode="never",
)


if __name__ == "__main__":
    demo.launch()
