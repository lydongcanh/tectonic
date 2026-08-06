"""Shared EDGAR sourcing: fetch exhibit documents of a given type, cached by type.

EDGAR labels each filing document with an exhibit type, so an exhibit-type prefix
is an authoritative document-type label: "EX-3" = charters/bylaws (constitutional),
"EX-13" = the annual-report financials (financial_statements), and so on. This
module turns such a prefix into Examples, with the same guarantees for every type:

  * a frozen MANIFEST per type (the chosen documents) so the set does not drift;
  * raw documents cached on disk per type, so rebuilds are offline and fast;
  * a descriptive User-Agent + a small delay, to respect SEC's request policy.

Everything for a type lives under gitignored data/raw/edgar/<type>/, so it is
obvious which cached documents belong to which document type.

First run needs WARP off (SEC uses normal TLS).
"""

from __future__ import annotations

import html
import json
import re
import time
from collections.abc import Iterator
from pathlib import Path

import requests

from dataset import Example

EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
UA = {"User-Agent": "tectonic-research ted.ly@ansarada.com"}
CACHE_ROOT = Path("data/raw/edgar")
DEFAULT_MAX_OFFSET = 1000  # how deep to page each query (100 hits per page)
DELAY_SECONDS = 0.15

# A fetched "document" that is too short, or carries a known SEC error/maintenance
# signature, is not a real exhibit. SEC returns these with HTTP 200, so a status
# check alone does not catch them (a maintenance page cost us junk once).
MIN_DOC_CHARS = 600
JUNK_SIGNATURES = (
    "sec.gov | file unavailable",
    "undergoing maintenance",
    "request rate threshold",
    "nosuchkey",
)


def html_to_text(doc: str) -> str:
    """Strip an EDGAR HTML exhibit down to readable plain text."""
    doc = re.sub(r"(?is)<(script|style).*?</\1>", " ", doc)  # drop script/style
    doc = re.sub(r"<[^>]+>", " ", doc)  # remove all tags
    doc = html.unescape(doc)  # &nbsp; &#160; -> chars
    return " ".join(doc.split())  # collapse whitespace


def _search(query: str, offset: int, forms: str | None) -> list[dict]:
    params: dict = {"q": query, "from": offset}
    if forms:
        params["forms"] = forms  # restrict to a filing form, e.g. "10-K"
    try:
        resp = requests.get(EFTS_URL, params=params, headers=UA, timeout=30)
        return resp.json().get("hits", {}).get("hits", [])
    except requests.RequestException:
        return []


def _gather_manifest(
    type_dir: Path,
    exhibit_prefix: str,
    queries: list[str],
    target: int,
    max_offset: int,
    forms: str | None,
) -> list[dict]:
    """Freeze the list of documents for this type: one exhibit per company, up to
    target. Built once from EDGAR search, then cached, so the dataset is stable."""
    manifest_path = type_dir / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())

    entries: list[dict] = []
    seen_company: set[str] = set()
    for query in queries:
        for offset in range(0, max_offset, 100):
            if len(entries) >= target:
                break

            for hit in _search(query, offset, forms):
                source = hit["_source"]
                if not str(source.get("file_type", "")).startswith(exhibit_prefix):
                    continue  # not the exhibit type we want

                company = source["ciks"][0]
                if company in seen_company:
                    continue  # one document per company, for variety

                seen_company.add(company)
                accession, filename = hit["_id"].split(":", 1)
                entries.append(
                    {
                        "accession": accession,
                        "filename": filename,
                        "cik": company,
                        "file_type": source.get("file_type"),
                        "filer": source.get("display_names", ["?"])[0],
                    }
                )
                if len(entries) >= target:
                    break

            time.sleep(DELAY_SECONDS)

        if len(entries) >= target:
            break

    type_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(entries, indent=2))
    return entries


def _is_real_document(text: str) -> bool:
    """Reject SEC error / maintenance / rate-limit pages (all served as HTTP 200)
    and anything implausibly short to be a real exhibit."""
    if len(text) < MIN_DOC_CHARS:
        return False
    low = text.lower()
    return not any(sig in low for sig in JUNK_SIGNATURES)


def _fetch_doc(type_dir: Path, entry: dict) -> str:
    """Return the cleaned text of one exhibit, fetching + caching the raw HTML once.

    We only cache CONTENT we have validated as a real document. SEC serves error
    and maintenance pages with HTTP 200, so a status check is not enough; we also
    check length + known junk signatures (see `_is_real_document`).
    """
    cache = type_dir / f"{entry['accession']}_{entry['filename']}".replace("/", "_")
    if cache.exists():
        return html_to_text(cache.read_text())

    url = f"{ARCHIVES}/{int(entry['cik'])}/{entry['accession'].replace('-', '')}/{entry['filename']}"
    try:
        resp = requests.get(url, headers=UA, timeout=30)
    except requests.RequestException:
        return ""
    time.sleep(DELAY_SECONDS)
    if resp.status_code != 200:
        return ""

    text = html_to_text(resp.text)
    if not _is_real_document(text):
        return ""  # error / maintenance / rate-limit page, not a real document

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(resp.text)
    return text


def load_edgar_exhibits(
    doc_type: str,
    exhibit_prefix: str,
    queries: list[str],
    target: int,
    max_offset: int = DEFAULT_MAX_OFFSET,
    forms: str | None = None,
) -> Iterator[Example]:
    """Yield one Example per distinct-company exhibit of the given type."""
    type_dir = CACHE_ROOT / doc_type
    for entry in _gather_manifest(
        type_dir, exhibit_prefix, queries, target, max_offset, forms
    ):
        text = _fetch_doc(type_dir, entry)
        if not text.strip():
            continue  # fetch failed or empty; skip

        yield Example(
            doc_id=f"{entry['accession']}:{entry['filename']}",
            source="edgar",
            type=doc_type,
            text=text,
        )
