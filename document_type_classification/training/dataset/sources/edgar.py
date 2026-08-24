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
SEARCH_RETRIES = 4  # transient search failures are retried, then raised (never swallowed)
# SEC throttles with an HTTP-200 HTML page (not a JSON error), and also returns 429/5xx
# under load. Any of these must NOT be read as "no more hits" (that silently truncates a
# frozen manifest); we retry, then fail loudly if they persist.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

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
    """One page of EDGAR full-text search hits.

    A returned list is an AUTHORITATIVE result: empty means genuinely no hits, so the
    pager can stop. A transient failure (network error, throttling HTML served as HTTP
    200, or a retryable 5xx/429) must never be mistaken for "no hits", or a frozen
    manifest silently ends up short. We retry those, then raise if they persist; a
    non-retryable client error (e.g. 400) raises immediately.
    """
    params: dict = {"q": query, "from": offset}
    if forms:
        params["forms"] = forms  # restrict to a filing form, e.g. "10-K"

    last_error = "unknown"
    for attempt in range(SEARCH_RETRIES):
        try:
            resp = requests.get(EFTS_URL, params=params, headers=UA, timeout=30)
        except requests.RequestException as exc:
            last_error = f"request error: {exc}"
        else:
            if resp.status_code == 200:
                try:
                    return resp.json().get("hits", {}).get("hits", [])
                except ValueError:  # JSONDecodeError: HTTP 200 but not JSON (throttle HTML)
                    last_error = "HTTP 200 but response was not JSON (likely throttling)"
            elif resp.status_code in _RETRYABLE_STATUS:
                last_error = f"HTTP {resp.status_code}"
            else:
                raise RuntimeError(f"EDGAR search failed: HTTP {resp.status_code} for q={query!r}")
        time.sleep(DELAY_SECONDS * 4 * (attempt + 1))  # linear backoff

    raise RuntimeError(
        f"EDGAR search failed after {SEARCH_RETRIES} attempts (q={query!r}, offset={offset}): "
        f"{last_error}. Not treating this as 'no results', which would silently truncate "
        "the manifest; fix connectivity / back off and rebuild."
    )


# "amendment" as a NOUN (a modifying document), tolerant of the misspellings and
# abbreviations filers really use ("amendement", "amendemnt", "amends", "amend.",
# "amdt"). It must NOT match the adjective "amended" ("amended and restated ..." is
# the full current text, which we keep), so we enumerate the noun forms rather than
# use the shared stem "amend". A real leak this caught: "FIRST AMENDEMENT TO
# DATABASE LICENSE AGREEMENT" slipped past a plain "AMENDMENT" substring check.
_AMENDMENT_RE = re.compile(
    r"\b(?:AMENDMENTS?|AMENDEMENTS?|AMENDEMNTS?|AMENDS|AMEND\.|AMENDMT|AMDTS?)\b"
)

# Other words that mark a document as modifying an agreement rather than being one.
# Amendment is handled separately (above), spelling-tolerant.
_MODIFIER_WORDS = (
    "ADDENDUM", "CONSENT", "ASSIGNMENT", "TERMINATION", "WAIVER", "SUPPLEMENT", "NOTICE",
)


def _names_a_modification(upper_desc: str, modifier_words: tuple[str, ...]) -> bool:
    """True if the (upper-cased) title names a document that MODIFIES an agreement
    rather than being one: an amendment (spelling-tolerant) or a modifier word."""
    return bool(_AMENDMENT_RE.search(upper_desc)) or any(w in upper_desc for w in modifier_words)


def title_says(agreement: str, disallow: tuple[str, ...] = ()) -> Callable[[str], bool]:
    """Build a `description_ok` predicate that keeps only FULL agreements of a kind.

    EDGAR has no dedicated exhibit type for licence / employment / lease agreements
    (all are EX-10 material contracts), so we label by the filer's own exhibit title
    in `file_description`. We keep a document only if its title names the agreement
    (e.g. "EMPLOYMENT AGREEMENT") and is not a sub-document that modifies one.

    The agreement name is matched as a SUBSTRING, which is deliberate for prefixes
    that are still the same kind of document ("SUBLEASE AGREEMENT" is a lease,
    "SUBLICENSE AGREEMENT" is a licence). But a substring also matches confusable
    DIFFERENT documents ("RELEASE AGREEMENT" contains "LEASE AGREEMENT" yet is a
    release of claims, not a lease). `disallow` names words whose presence (as whole
    words) disqualifies the title despite the substring match, e.g. lease passes
    `disallow=("RELEASE",)` to reject releases while still keeping subleases.

    Two independent guards catch modifying sub-documents:
      * the agreement named after "TO" ("AMENDMENT TO <A>", "EXTENSION TO <A>"),
        which catches modifier words we did not enumerate; and
      * an explicit modification check (`_names_a_modification`), which catches an
        amendment even when a qualifier sits between "TO" and the agreement name
        ("... TO DATABASE LICENSE AGREEMENT", which the first guard alone misses) or
        the word is misspelled, plus the named modifier words (assignment, consent,
        addendum, ...).
    We deliberately do NOT treat "OF <A>" as modifying: its real modifiers
    (assignment/termination OF) are already in the word list, whereas "FORM OF <A>"
    and "TRANSLATION OF <A>" are full instances we want to keep. "AMENDED AND
    RESTATED <A>" is kept too: adjective, the full current text.
    """
    agreement = agreement.upper()
    modifies = re.compile(r"\bTO\s+(?:THE\s+)?" + re.escape(agreement))
    disallowed = [re.compile(rf"\b{re.escape(w.upper())}\b") for w in disallow]

    def ok(description: str) -> bool:
        d = description.upper()
        if agreement not in d:
            return False
        if any(bad.search(d) for bad in disallowed):
            return False  # a confusable different document (e.g. RELEASE, not a lease)
        if modifies.search(d):
            return False
        return not _names_a_modification(d, _MODIFIER_WORDS)

    return ok


def title_not_modification(keep_supplements: bool = False) -> Callable[[str], bool]:
    """A `description_ok` predicate for sources whose EXHIBIT TYPE is already the
    authoritative label (e.g. EX-2 = acquisition / merger agreements), used only to
    drop sub-documents.

    Unlike `title_says`, this DEFAULT-ACCEPTS. Many full EX-2 agreements carry no
    descriptive title at all (just "EX-2.1" or blank), so requiring a title phrase
    would throw most of them away. We keep everything except titles that name a
    modification. "AMENDED AND RESTATED ..." is still kept (adjective, full text).

    `keep_supplements=True` (used by financing) does NOT treat "supplement" as a
    modification, because a supplemental indenture is a full, substantive debt
    document (it establishes a new note series with its own covenants).
    """
    words = tuple(w for w in _MODIFIER_WORDS if not (keep_supplements and w == "SUPPLEMENT"))

    def ok(description: str) -> bool:
        return not _names_a_modification(description.upper(), words)

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

    # Landing well under target can be genuine (the queries exhausted matching filings)
    # OR a sign of dropped pages. We cannot tell the two apart here, so we surface it
    # loudly rather than freeze a possibly-degraded manifest without a word. (`_search`
    # already turns transient failures into hard errors; this catches quieter shortfalls.)
    if len(entries) < 0.75 * spec.target:
        print(f"WARNING: {type_dir.name} gathered {len(entries)} docs, well under target "
              f"{spec.target}. Verify this is real exhaustion, not dropped search pages, "
              "before trusting the frozen manifest.")

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
        # Re-apply the label filter to the stored title on every load, not just when
        # the manifest is first built. This makes the predicate the single source of
        # truth: an improved filter (e.g. one that now catches a misspelled amendment)
        # retroactively drops entries a frozen manifest still lists, with no re-query.
        desc = entry.get("file_description")
        if description_ok is not None and desc is not None and not description_ok(desc):
            continue

        text = _fetch_doc(type_dir, entry)
        if not text.strip():
            continue  # fetch failed or empty; skip

        yield Example(
            doc_id=f"{entry['accession']}:{entry['filename']}",
            source="edgar",
            type=doc_type,
            text=text,
        )
