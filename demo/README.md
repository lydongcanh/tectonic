# demo

A small [Gradio](https://www.gradio.app/) web UI for the document-type classifier: upload one
or more documents (**PDF, DOCX, TXT**), and it extracts the text locally and shows the
predicted type and confidence for each file in a results table.

## Run it

```bash
poetry install --all-extras --with demo
poetry run python demo/app.py
```

Then open the local URL it prints (default http://127.0.0.1:7860).

By default it loads the published model from the Hugging Face Hub
(`lydongcanh/tectonic-doctype`). To run **offline** against a local model bundle instead:

```bash
TECTONIC_MODEL=document_type_classification/hf_release/tectonic-doctype \
  poetry run python demo/app.py
```

## Notes

- The demo depends on `gradio`, `pypdf`, and `python-docx` (the `demo` dependency group) and
  on the `document-type` extra of the package. It is not part of the shipped `tectonic`
  package.
- Text extraction is best-effort: a scanned-image PDF (no embedded text) can't be read
  without OCR, and unsupported file types are reported per-file in the table rather than
  failing the whole batch.
- This same `app.py` can be dropped into a Hugging Face Space to host the demo publicly:
  add a `requirements.txt` with `tectonic[document-type]`, `gradio`, `pypdf`, and
  `python-docx`.
