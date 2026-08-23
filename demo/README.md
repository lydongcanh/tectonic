# demo

A small [Gradio](https://www.gradio.app/) web UI for the document-type classifier: paste a
document, see the predicted type with the full per-type confidence as a ranked bar chart.

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

- The demo depends on `gradio` (the `demo` dependency group) and on the `document-type`
  extra of the package. It is not part of the shipped `tectonic` package.
- This same `app.py` can be dropped into a Hugging Face Space to host the demo publicly:
  add a `requirements.txt` with `tectonic[document-type]` and `gradio`.
