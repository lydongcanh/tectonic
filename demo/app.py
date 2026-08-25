"""Gradio demo for the tectonic document-type classifier.

Upload one or more documents (PDF, DOCX, or TXT) and see the predicted type for each, with
its confidence. Text is extracted from each file locally, then classified in one batch.

Run it:
    poetry install --all-extras --with demo
    poetry run python demo/app.py

By default it loads the published model from the Hugging Face Hub. To run offline against
a local model bundle instead, set TECTONIC_MODEL to that directory, e.g.:
    TECTONIC_MODEL=document_type_classification/hf_release/tectonic-doctype poetry run python demo/app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from tectonic.document_type import DocumentTypeClassifier

MODEL = os.environ.get("TECTONIC_MODEL", "lydongcanh/tectonic-doctype")
classifier = DocumentTypeClassifier.from_pretrained(MODEL)

SUPPORTED = [".pdf", ".docx", ".txt", ".md"]


def _extract_text(path: str) -> str:
    """Pull plain text out of one uploaded file. Raises ValueError with a readable message
    if the type is unsupported or no text could be extracted (e.g. a scanned-image PDF)."""
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    elif suffix == ".docx":
        import docx

        text = "\n".join(p.text for p in docx.Document(path).paragraphs)
    elif suffix in (".txt", ".md"):
        text = Path(path).read_text(errors="ignore")
    else:
        raise ValueError(f"unsupported file type '{suffix}' (use {', '.join(SUPPORTED)})")

    if not text.strip():
        raise ValueError("no extractable text (a scanned-image PDF needs OCR first)")
    return text


def classify_files(paths: list[str] | None) -> list[list[str]]:
    """Extract + classify each uploaded file; return rows for the results table.

    Files that parse are classified together in one batch (faster); files that fail
    extraction get a row explaining why, so one bad file never hides the good results.
    """
    if not paths:
        return []

    names, texts, rows = [], [], []
    for path in paths:
        name = Path(path).name
        try:
            texts.append(_extract_text(path))
            names.append(name)
        except Exception as exc:  # any parse error becomes a table row, never a crash
            rows.append([name, "—", "—", f"could not read: {exc}"])

    for name, prediction in zip(names, classifier.classify_batch(texts)):
        ranked = sorted(prediction.scores.items(), key=lambda kv: kv[1], reverse=True)
        top_label, top_score = ranked[0]
        runner_label, runner_score = ranked[1]
        rows.append([
            name,
            str(top_label),
            f"{top_score:.0%}",
            f"{runner_label} ({runner_score:.0%})",
        ])
    return rows


with gr.Blocks(title="Document Type Classifier") as demo:
    gr.Markdown(
        "# Document Type Classifier\n"
        "Upload one or more legal / deal documents (**PDF, DOCX, TXT**) to classify each into "
        "one of nine types. Text is extracted locally, then classified. Confidence is not "
        "calibrated, so treat borderline scores with care. Trained on public filings "
        "(EDGAR / CUAD / ContractNLI)."
    )
    files = gr.Files(
        file_count="multiple",
        file_types=SUPPORTED,
        label="Documents (PDF, DOCX, TXT)",
    )
    run = gr.Button("Classify", variant="primary")
    results = gr.Dataframe(
        headers=["File", "Predicted type", "Confidence", "Runner-up"],
        datatype=["str", "str", "str", "str"],
        label="Results",
        wrap=True,
    )
    run.click(classify_files, inputs=files, outputs=results)
    files.change(classify_files, inputs=files, outputs=results)  # classify as soon as files land


if __name__ == "__main__":
    demo.launch()
