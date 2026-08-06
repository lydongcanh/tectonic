"""Peek at EDGAR EX-3 exhibits: confirm they give clean constitutional documents.

Exploration only. There is no ready-made dataset of corporate charters/bylaws, so
we source them from SEC EDGAR. Every EDGAR filing labels each document with an
exhibit type, and EX-3 is exactly the constitutional family (certificate/articles
of incorporation, bylaws). That exhibit type is our ground-truth label, the same
idea as CUAD's titles, but authoritative because SEC assigns it.

How it works:
  1. hit EDGAR full-text search for a charter-ish phrase (this just surfaces
     candidate documents; the phrase is not the label),
  2. keep only hits whose file_type starts with "EX-3" (that IS the label),
  3. fetch each exhibit and clean the HTML to plain text.

Network required: WARP off (SEC uses normal TLS; WARP's interception breaks it).
SEC asks for a descriptive User-Agent and <= 10 requests/second.

Run:
    poetry run python document_type_classification/exploration/edgar/peek_edgar_ex3.py
"""

from __future__ import annotations

import html
import re
import time

import requests

EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
UA = {"User-Agent": "tectonic-research ted.ly@ansarada.com"}
QUERY = '"certificate of incorporation"'
HOW_MANY = 6


def html_to_text(doc: str) -> str:
    """Strip an EDGAR HTML exhibit down to readable plain text."""
    doc = re.sub(r"(?is)<(script|style).*?</\1>", " ", doc)  # drop script/style blocks
    doc = re.sub(r"<[^>]+>", " ", doc)                        # remove all tags
    doc = html.unescape(doc)                                  # &nbsp; &#160; -> real chars
    return " ".join(doc.split())                              # collapse whitespace


def _doc_url(source: dict, doc_id: str) -> str:
    accession, filename = doc_id.split(":", 1)
    cik = int(source["ciks"][0])
    adsh = source["adsh"].replace("-", "")
    return f"{ARCHIVES}/{cik}/{adsh}/{filename}"


def main() -> None:
    resp = requests.get(EFTS_URL, params={"q": QUERY}, headers=UA, timeout=30)
    hits = resp.json().get("hits", {}).get("hits", [])
    ex3 = [h for h in hits if str(h["_source"].get("file_type", "")).startswith("EX-3")]
    print(f"{len(hits)} hits, {len(ex3)} are EX-3 (constitutional)\n")

    seen_companies: set[str] = set()
    shown = 0
    for h in ex3:
        source = h["_source"]
        company = source["ciks"][0]
        if company in seen_companies:
            continue  # one document per company for a varied peek
        seen_companies.add(company)

        text = html_to_text(requests.get(_doc_url(source, h["_id"]), headers=UA, timeout=30).text)
        print("=" * 90)
        print(f"TYPE : {source.get('file_type')}   ({source.get('file_description', '')[:50]})")
        print(f"FILER: {source.get('display_names', ['?'])[0]}")
        print(f"LEN  : {len(text):,} chars")
        print(f"TEXT : {text[:400]}")
        print()

        shown += 1
        if shown >= HOW_MANY:
            break
        time.sleep(0.2)  # be polite to SEC


if __name__ == "__main__":
    main()
