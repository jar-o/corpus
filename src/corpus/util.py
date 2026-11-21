"""Small shared helpers: hashing, normalization, atomic writes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

_WS = re.compile(r"\s+")


def collapse(text: str) -> str:
    return _WS.sub(" ", text).strip()


def entity_key(text: str) -> str:
    """Conservative entity grouping key: case, whitespace and a leading
    "the" are ignored; nothing else is merged."""
    key = collapse(unicodedata.normalize("NFC", text)).casefold()
    return key[4:] if key.startswith("the ") else key


def doc_id(relpath: str) -> str:
    return hashlib.sha1(relpath.encode("utf-8")).hexdigest()[:16]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def fingerprint(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def est_tokens(chars: int) -> int:
    """Rough token estimate (~4 characters per token for English prose)."""
    return round(chars / 4)


def _file_mode() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return 0o666 & ~mask


_MODE = _file_mode()


def atomic_write(path: Path, data: bytes | str) -> None:
    """Write via temp file + rename so readers never see a partial file."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.chmod(tmp, _MODE)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
