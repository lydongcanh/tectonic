"""Load ContractNLI as Examples: every document is an NDA.

We pull ContractNLI from its CANONICAL Stanford source: a single CC BY 4.0 zip
containing train/dev/test JSON, each with a `documents` list of full NDA texts.
We had been using a HuggingFace mirror, but it was script-based, flaky, and did
not declare a licence; once a transient failure there silently wiped the whole
NDA class. The canonical zip fixes all three: stable, properly licensed, and
cached locally so builds are reproducible and offline after the first download.

Licence: CC BY 4.0 (the zip bundles its own LICENSE / TERMS files).
Cached under gitignored data/raw/contract_nli/. First download needs WARP off.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterator
from pathlib import Path

import requests

from dataset import Example

CONTRACT_NLI_URL = "https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip"
UA = {"User-Agent": "tectonic-research ted.ly@ansarada.com"}
# Anchor the cache to the repo root (this file is two levels deeper than the
# dataset scripts) so the zip is cached in one place regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[4]  # .../tectonic/
CACHE_DIR = _REPO_ROOT / "data/raw/contract_nli"
CACHE_ZIP = CACHE_DIR / "contract-nli.zip"
SPLITS = ("contract-nli/train.json", "contract-nli/dev.json", "contract-nli/test.json")


def _ensure_zip() -> Path:
    """Download the ContractNLI zip once, then reuse the cached copy."""
    if not CACHE_ZIP.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        resp = requests.get(CONTRACT_NLI_URL, headers=UA, timeout=120)
        resp.raise_for_status()
        CACHE_ZIP.write_bytes(resp.content)
    return CACHE_ZIP


def load_contract_nli() -> Iterator[Example]:
    """Yield one Example per distinct NDA across all splits."""
    archive = zipfile.ZipFile(_ensure_zip())
    seen: set[str] = set()
    for split in SPLITS:
        data = json.loads(archive.read(split))
        for doc in data["documents"]:
            key = doc.get("id") or doc["file_name"]
            if key in seen:
                continue
            seen.add(key)
            text = doc["text"]
            if not text.strip():
                continue
            yield Example(
                doc_id=str(doc.get("file_name") or key),
                source="contract_nli",
                type="nda",
                text=text,
            )
