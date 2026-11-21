"""Format parsers: file -> ParsedDocument.

``ParsedDocument.text`` is the *reference text*; every line number in the
output refers to it.

* Markdown / plain text: the reference text is the source file itself
  (decoded, newlines normalized - line numbers match any editor). Nothing
  is cached; LLMs read the original file.
* PDF / DOCX: the reference text is the extracted text, cached at
  ``.corpus-out/text/<path>.txt`` because it exists nowhere else.

``analysis_text`` has the same length as ``text`` with markup (Markdown
syntax, page markers) replaced by spaces, so NLP ignores markup while
character offsets stay aligned.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .util import collapse


@dataclass
class ParsedDocument:
    text: str
    analysis_text: str
    title: str | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.text) != len(self.analysis_text):
            raise ValueError("analysis_text must have the same length as text")


class EmptyDocument(ValueError):
    """Raised when a document has no extractable text."""


def normalize_text(text: str) -> str:
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text)


def read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def paragraph_title(text: str) -> str | None:
    """Title heuristic: a short first paragraph, plus a short second one
    when it looks like a subtitle (no sentence punctuation at the end)."""
    paras = [collapse(p) for p in re.split(r"\n[ \t]*\n", text.strip(), maxsplit=2)[:2]]
    paras = [p for p in paras if p]
    if not paras or len(paras[0]) > 150:
        return None
    title = paras[0]
    if len(paras) > 1 and len(paras[1]) <= 150:
        sub = paras[1]
        # A short label ("Chapter 3", "FEDERALIST No. 26") followed by a single
        # sentence is a title + subtitle even if the subtitle ends with a period.
        one_sentence = not re.search(r"[.!?:;]\s", sub)
        long_words = [w for w in re.findall(r"[^\W\d_]+", sub) if len(w) > 3]
        title_case = bool(long_words) and sum(w[0].isupper() for w in long_words) >= 0.6 * len(long_words)
        if not sub.endswith((".", ":", ";", "?", "!", ",")) or (len(title) <= 60 and one_sentence and title_case):
            title += " — " + sub.rstrip(".")
    return title


# ------------------------------------------------------------------ text


def parse_plain(path: Path) -> ParsedDocument:
    text = normalize_text(read_text_file(path))
    return ParsedDocument(text, text, title=paragraph_title(text))


_FRONT_MATTER = re.compile(r"\A---\n.*?\n(?:---|\.\.\.)\n", re.S)
_FENCE = re.compile(r"^(```|~~~)[^\n]*\n.*?^\1[^\n]*$", re.S | re.M)
_HTML_TAG = re.compile(r"<!--.*?-->|</?[A-Za-z][^>\n]*>", re.S)
_INLINE_CODE = re.compile(r"`[^`\n]+`")
_LINK_TARGET = re.compile(r"\]\([^)\n]*\)")
_LINK_OPEN = re.compile(r"!?\[(?=[^\]\n]*\]\()")
_HEADING = re.compile(r"^ {0,3}#{1,6}(?=\s)", re.M)
_HEADING_CLOSE = re.compile(r"\s#+\s*$", re.M)
_LIST_MARKER = re.compile(r"^\s*(?:[-*+]|\d+[.)])(?=\s)", re.M)
_BLOCKQUOTE = re.compile(r"^\s*(?:>\s?)+", re.M)
_RULE_LINE = re.compile(r"^\s*(?:[-*_]\s*){3,}$", re.M)
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$", re.M)
_EMPHASIS = re.compile(r"\*+|~~|(?<!\w)_+|_+(?!\w)")
_PIPE = re.compile(r"\|")


def mask(chars: list[str], start: int, end: int) -> None:
    for i in range(start, end):
        if chars[i] != "\n":
            chars[i] = " "


def mask_markdown(text: str) -> str:
    """Return ``text`` with Markdown syntax replaced by spaces (same length)."""
    chars = list(text)

    def mask_all(pattern: re.Pattern, source: str | None = None) -> None:
        for m in pattern.finditer(source if source is not None else "".join(chars)):
            mask(chars, m.start(), m.end())

    # Regions that are not prose at all.
    mask_all(_FRONT_MATTER, text)
    mask_all(_FENCE, text)
    mask_all(_HTML_TAG)
    mask_all(_INLINE_CODE)
    mask_all(_TABLE_SEP)
    mask_all(_RULE_LINE)
    # Syntax only; the prose it wraps is kept.
    for pattern in (_LINK_OPEN, _LINK_TARGET, _HEADING, _HEADING_CLOSE, _BLOCKQUOTE,
                    _LIST_MARKER, _EMPHASIS, _PIPE):
        mask_all(pattern)
    return "".join(chars)


def parse_markdown(path: Path) -> ParsedDocument:
    text = normalize_text(read_text_file(path))
    masked = mask_markdown(text)
    title = None
    for m in re.finditer(r"(?m)^ {0,3}#{1,6}[ \t]+(.+)$", text):
        title = collapse(masked[m.start(1):m.end(1)])
        if title:
            break
    return ParsedDocument(text, masked, title=title or paragraph_title(masked))


# ------------------------------------------------------------------ extracted formats


def parse_docx(path: Path) -> ParsedDocument:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(str(path))
    blocks: list[str] = []
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = collapse(Paragraph(child, document).text)
            if para:
                blocks.append(para)
        elif tag == "tbl":
            for row in Table(child, document).rows:
                cells = [collapse(c.text) for c in row.cells]
                if any(cells):
                    blocks.append(" | ".join(cells))
    text = normalize_text("\n\n".join(blocks)) + "\n"
    if not text.strip():
        raise EmptyDocument("no text found in DOCX")
    title = collapse(document.core_properties.title or "") or paragraph_title(text)
    return ParsedDocument(text, text, title=title, meta={"paragraphs": len(blocks)})


_PAGE_MARKER = "[page {}]"


def parse_pdf(path: Path) -> ParsedDocument:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: list[str] = []
    masked: list[str] = []
    for i, page in enumerate(reader.pages, 1):
        body = normalize_text(page.extract_text() or "").strip()
        marker = _PAGE_MARKER.format(i)
        parts.append(f"{marker}\n{body}")
        masked.append(" " * len(marker) + "\n" + body)
    text = "\n\n".join(parts) + "\n"
    if not any(p.strip() for p in masked):
        raise EmptyDocument("PDF has no text layer (scanned? OCR is not supported)")
    title = None
    info = reader.metadata
    if info is not None and info.title and not str(info.title).lower().startswith("microsoft word"):
        title = collapse(str(info.title))
    if not title:
        first = next((collapse(line) for line in "\n".join(masked).splitlines() if line.strip()), "")
        title = first if 0 < len(first) <= 150 else None
    return ParsedDocument(text, "\n\n".join(masked) + "\n", title=title, meta={"pages": len(reader.pages)})


# ------------------------------------------------------------------ registry


@dataclass(frozen=True)
class Format:
    name: str
    version: str
    extracted: bool  # True: reference text is cached under .corpus-out/text/
    parse: Callable[[Path], ParsedDocument]


FORMATS: dict[str, Format] = {
    ".md": Format("markdown", "1", False, parse_markdown),
    ".markdown": Format("markdown", "1", False, parse_markdown),
    ".txt": Format("text", "1", False, parse_plain),
    ".pdf": Format("pdf", "1", True, parse_pdf),
    ".docx": Format("docx", "1", True, parse_docx),
}


def format_for(path: Path) -> Format | None:
    return FORMATS.get(path.suffix.lower())
