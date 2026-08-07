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
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import requests

from dataset import Example

EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
UA = {"User-Agent": "tectonic-research ted.ly@ansarada.com"}
# Anchor the cache to the repo root (this file is two levels deeper than the
# dataset scripts) so raw documents are cached in one place regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[4]  # .../tectonic/
CACHE_ROOT = _REPO_ROOT / "data/raw/edgar"
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


# Titles that describe a document ABOUT an agreement rather than the agreement
# itself, for cases the preposition rule in `title_says` does not reach.
_NOT_FULL_KEYWORDS = (
    "AMENDMENT", "AMENDS", "ADDENDUM", "CONSENT", "ASSIGNMENT",
    "TERMINATION", "WAIVER", "SUPPLEMENT", "NOTICE",
)


def title_says(agreement: str) -> Callable[[str], bool]:
    """Build a `description_ok` predicate that keeps only FULL agreements of a kind.

    EDGAR has no dedicated exhibit type for licence / employment / lease agreements
    (all are EX-10 material contracts), so we label by the filer's own exhibit title
    in `file_description`. We keep a document only if its title names the agreement
    (e.g. "EMPLOYMENT AGREEMENT") and is not a sub-document that merely modifies one.

    A sub-document names the agreement AFTER a preposition: "AMENDMENT TO <A>",
    "ASSIGNMENT OF <A>", "CONSENT TO THE <A>". Keying on that structure catches
    modifier words we did not enumerate and their abbreviations (a real example the
    scout caught: "AMENDS. TO EMPLOYMENT AGREEMENT ..."). "AMENDED AND RESTATED <A>"
    is kept: it is the full, current text, not a modifying document.
    """
    agreement = agreement.upper()
    modifies = re.compile(r"\b(?:TO|OF)\s+(?:THE\s+)?" + re.escape(agreement))

    def ok(description: str) -> bool:
        d = description.upper()
        if agreement not in d:
            return False
        if modifies.search(d):
            return False
        return not any(k in d for k in _NOT_FULL_KEYWORDS)

    return ok


def title_not_modification(keywords: tuple[str, ...] = _NOT_FULL_KEYWORDS) -> Callable[[str], bool]:
    """A `description_ok` predicate for sources whose EXHIBIT TYPE is already the
    authoritative label (e.g. EX-2 = acquisition / merger agreements), used only to
    drop sub-documents.

    Unlike `title_says`, this DEFAULT-ACCEPTS. Many full EX-2 agreements carry no
    descriptive title at all (just "EX-2.1" or blank), so requiring a title phrase
    would throw most of them away. We keep everything except descriptions that name
    a modification (amendment, addendum, ...). "AMENDED AND RESTATED ..." is still
    kept: that is the adjective, the full current text, not a modifying document.

    `keywords` lets a caller override which titles count as a modification. Financing
    passes a set WITHOUT "SUPPLEMENT", because a supplemental indenture is a full,
    substantive debt document (it establishes a new note series with its own
    covenants), not a throwaway amendment.
    """
    def ok(description: str) -> bool:
        d = description.upper()
        return not any(k in d for k in keywords)

    return ok


def _hit_to_entry(
    hit: dict,
    exhibit_prefix: str,
    description_ok: Callable[[str], bool] | None,
    seen_company: set[str],
) -> dict | None:
    """Turn one search hit into a manifest entry, or return None to skip it.

    A hit is skipped when it is the wrong exhibit type, when its filer already
    contributed a document (one per company, for variety), or when a
    `description_ok` predicate is given and the filer's own exhibit title
    (`file_description`) does not pass it. The description gate is how we label IP
    agreements: there is no authoritative exhibit type for them, so we keep only
    EX-10 exhibits the filer titled a licence agreement, an independent label that
    does not peek at the body text the model will learn from.
    """
    source = hit["_source"]
    if not str(source.get("file_type", "")).startswith(exhibit_prefix):
        return None  # not the exhibit type we want
    if description_ok is not None and not description_ok(source.get("file_description") or ""):
        return None  # filer's own title does not qualify (e.g. not a full licence)

    company = source["ciks"][0]
    if company in seen_company:
        return None
    seen_company.add(company)

    accession, filename = hit["_id"].split(":", 1)
    return {
        "accession": accession,
        "filename": filename,
        "cik": company,
        "file_type": source.get("file_type"),
        "file_description": source.get("file_description"),  # kept so labels stay auditable
        "filer": source.get("display_names", ["?"])[0],
    }


@dataclass(frozen=True)
class _Search:
    """The fixed parameters of a manifest-gathering search: everything except the
    query string and the running result. Bundled so they need not be threaded
    through each helper one by one.

    * `max_per_query` caps how many documents a single query may contribute, to stop
      one query dominating. Financials use it: each query targets a fiscal year-end,
      and without a cap the (far more common) December filers would refill the set
      and the model would key on the date instead of the accounting. `None` = no cap.
    * `description_ok` optionally gates on the filer's exhibit title (see
      `_hit_to_entry`); `None` accepts any title.
    """

    exhibit_prefix: str
    target: int
    max_offset: int
    forms: str | None
    max_per_query: int | None
    description_ok: Callable[[str], bool] | None


def _page_one_query(
    spec: _Search, query: str, entries: list[dict], seen_company: set[str]
) -> None:
    """Page a single query, appending accepted entries in place, and stop as soon as
    the overall target or this query's per-query cap is reached. All stop conditions
    live here, in one place."""
    added = 0
    for offset in range(0, spec.max_offset, 100):
        for hit in _search(query, offset, spec.forms):
            entry = _hit_to_entry(hit, spec.exhibit_prefix, spec.description_ok, seen_company)
            if entry is None:
                continue
            entries.append(entry)
            added += 1
            if len(entries) >= spec.target:
                return
            if spec.max_per_query is not None and added >= spec.max_per_query:
                return
        time.sleep(DELAY_SECONDS)


def _gather_manifest(type_dir: Path, spec: _Search, queries: list[str]) -> list[dict]:
    """Freeze the list of documents for this type: one exhibit per company, up to
    target. Built once from EDGAR search, then cached, so the dataset is stable."""
    manifest_path = type_dir / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())

    entries: list[dict] = []
    seen_company: set[str] = set()
    for query in queries:
        if len(entries) >= spec.target:
            break
        _page_one_query(spec, query, entries, seen_company)

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
    max_per_query: int | None = None,
    description_ok: Callable[[str], bool] | None = None,
) -> Iterator[Example]:
    """Yield one Example per distinct-company exhibit of the given type."""
    type_dir = CACHE_ROOT / doc_type
    spec = _Search(
        exhibit_prefix=exhibit_prefix,
        target=target,
        max_offset=max_offset,
        forms=forms,
        max_per_query=max_per_query,
        description_ok=description_ok,
    )
    for entry in _gather_manifest(type_dir, spec, queries):
        text = _fetch_doc(type_dir, entry)
        if not text.strip():
            continue  # fetch failed or empty; skip

        yield Example(
            doc_id=f"{entry['accession']}:{entry['filename']}",
            source="edgar",
            type=doc_type,
            text=text,
        )
