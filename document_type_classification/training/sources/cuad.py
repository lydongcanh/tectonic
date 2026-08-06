"""Load CUAD as Examples: real commercial contracts, labelled by our taxonomy.

CUAD has no type field; the type is the tail of the filename-style title. We
extract and normalise it (the approach we validated in exploration/cuad), then
map it onto our labels. Almost everything is commercial; a few licensing/IP
contracts map to `ip_agreement` and the SEC joint-filing formalities to `other`.

This deliberately re-implements what exploration/cuad prototyped. Exploration was
the throwaway experiment; this is the kept version the real pipeline depends on,
so it must stand on its own and not import from exploration.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from datasets import load_dataset

from dataset import Example

CUAD_ID = "theatticusproject/cuad-qa"

_TRAILING_NUM = re.compile(r"\s*\(?\d+\)?\s*$")  # "Agreement3", "Agreement (1)"
_SEC_START = re.compile(r"^\d{4}-EX")            # "2020-EX-10.1-..."
_IP_TYPES = {
    "CONTENT LICENSE AGREEMENT",
    "INTELLECTUAL PROPERTY AGREEMENT",
    "TRADEMARK LICENSE AGREEMENT",
    "LICENSE AGREEMENT",
}
_OTHER_TYPES = {"JOINT FILING AGREEMENT"}


def _strip_sec_prefix(text: str) -> str:
    """Drop a leading SEC exhibit prefix like "2020-EX-10.1-" if present."""
    if not _SEC_START.match(text):
        return text
    head, sep, rest = text.partition(" ")
    cut = head.rfind("-")
    return head[cut + 1:] + sep + rest if cut != -1 else text


def _contract_type(title: str) -> str:
    """The normalised CUAD contract type sitting at the end of the title."""
    tail = title.rsplit("_", 1)[-1]
    if " - " in tail:
        tail = tail.rsplit(" - ", 1)[-1]
    tail = tail.strip().upper()
    tail = _strip_sec_prefix(tail)
    return _TRAILING_NUM.sub("", tail).strip()


def _label(cuad_type: str) -> str:
    """Map one CUAD type onto one of our taxonomy labels."""
    if cuad_type in _IP_TYPES:
        return "ip_agreement"
    if cuad_type in _OTHER_TYPES:
        return "other"
    return "commercial_agreement"


def load_cuad() -> Iterator[Example]:
    """Yield one Example per distinct CUAD contract (deduped by title)."""
    seen: set[str] = set()
    for split in ("train", "test"):
        rows = load_dataset(CUAD_ID, split=split, streaming=True, trust_remote_code=True)
        for row in rows:
            title = row["title"]
            if title in seen:
                continue
            seen.add(title)
            yield Example(
                doc_id=title,
                source="cuad",
                type=_label(_contract_type(title)),
                text=row["context"],
            )
