"""Load `constitutional` documents (charters / bylaws) from SEC EDGAR.

There is no ready dataset for these, so we fetch them ourselves. EDGAR labels
each filing document with an exhibit type, and EX-3 is exactly the constitutional
family, so that type is our authoritative label (see exploration/edgar).

Unlike the CUAD / ContractNLI loaders, this one hits the network, so it is built
to be polite and reproducible:

  * a MANIFEST (the frozen list of chosen documents) is written on the first run
    and reused after, so the dataset does not drift as new filings appear;
  * each raw document is CACHED on disk, so rebuilds are offline and fast and we
    never re-download from SEC;
  * a descriptive User-Agent and a small delay respect SEC's request policy.

Everything cached lives under gitignored data/raw/edgar/. First run needs WARP
off (SEC uses normal TLS).
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

# Phrases only used to SURFACE candidate EX-3 documents; the label is the EX-3
# exhibit type, not the phrase.
QUERIES = [
    '"certificate of incorporation"',
    '"amended and restated bylaws"',
    '"articles of incorporation"',
]
TARGET = 300          # how many distinct-company constitutional docs to collect
MAX_OFFSET = 1000     # how deep to page each query (100 hits per page)
DELAY_SECONDS = 0.15  # be polite to SEC

CACHE_DIR = Path("data/raw/edgar")
MANIFEST_PATH = CACHE_DIR / "constitutional_manifest.json"


def html_to_text(doc: str) -> str:
    """Strip an EDGAR HTML exhibit down to readable plain text."""
    doc = re.sub(r"(?is)<(script|style).*?</\1>", " ", doc)  # drop script/style
    doc = re.sub(r"<[^>]+>", " ", doc)                        # remove all tags
    doc = html.unescape(doc)                                  # &nbsp; &#160; -> real chars
    return " ".join(doc.split())                              # collapse whitespace


def _search(query: str, offset: int) -> list[dict]:
    try:
        resp = requests.get(EFTS_URL, params={"q": query, "from": offset},
                            headers=UA, timeout=30)
        return resp.json().get("hits", {}).get("hits", [])
    except requests.RequestException:
        return []


def _gather_manifest(target: int) -> list[dict]:
    """Freeze the list of documents to use: one EX-3 per company, up to target.

    Built once from EDGAR search, then cached, so the dataset is stable.
    """
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())

    entries: list[dict] = []
    seen_company: set[str] = set()
    for query in QUERIES:
        for offset in range(0, MAX_OFFSET, 100):
            if len(entries) >= target:
                break
            for hit in _search(query, offset):
                source = hit["_source"]
                if not str(source.get("file_type", "")).startswith("EX-3"):
                    continue  # not a constitutional exhibit
                company = source["ciks"][0]
                if company in seen_company:
                    continue  # one document per company, for variety
                seen_company.add(company)
                accession, filename = hit["_id"].split(":", 1)
                entries.append({
                    "accession": accession,
                    "filename": filename,
                    "cik": company,
                    "file_type": source.get("file_type"),
                    "filer": source.get("display_names", ["?"])[0],
                })
                if len(entries) >= target:
                    break
            time.sleep(DELAY_SECONDS)
        if len(entries) >= target:
            break

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(entries, indent=2))
    return entries


def _fetch_doc(entry: dict) -> str:
    """Return the cleaned text of one exhibit, fetching + caching the raw HTML once."""
    cache = CACHE_DIR / f"{entry['accession']}_{entry['filename']}".replace("/", "_")
    if cache.exists():
        return html_to_text(cache.read_text())

    url = f"{ARCHIVES}/{int(entry['cik'])}/{entry['accession'].replace('-', '')}/{entry['filename']}"
    try:
        raw = requests.get(url, headers=UA, timeout=30).text
    except requests.RequestException:
        return ""
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(raw)
    time.sleep(DELAY_SECONDS)
    return html_to_text(raw)


def load_edgar_constitutional() -> Iterator[Example]:
    """Yield one Example per distinct-company EX-3 (constitutional) document."""
    for entry in _gather_manifest(TARGET):
        text = _fetch_doc(entry)
        if not text.strip():
            continue  # fetch failed or empty; skip
        yield Example(
            doc_id=f"{entry['accession']}:{entry['filename']}",
            source="edgar",
            type="constitutional",
            text=text,
        )
