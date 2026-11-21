from pathlib import Path

import pytest

from corpus.parsers import EmptyDocument, format_for, mask_markdown, paragraph_title, parse_markdown, parse_plain
from samplegen import write_docx, write_pdf


def test_mask_markdown_keeps_length_and_prose_offsets():
    src = ("---\ntitle: x\n---\n# Heading\n\nSome **bold** and [a link](http://x.y) text.\n\n"
           "- item one\n\n```python\nprint('code')\n```\n| a | b |\n|---|---|\n")
    masked = mask_markdown(src)
    assert len(masked) == len(src) and masked.count("\n") == src.count("\n")
    for token in ("**", "](", "http", "```", "print", "#", "|", "title"):
        assert token not in masked
    i = src.index("bold")
    assert masked[i:i + 4] == "bold"


def test_markdown_title_and_newlines(tmp_path: Path):
    p = tmp_path / "a.md"
    p.write_bytes("﻿# The *Title*\r\n\r\nBody text.\r\n".encode())
    doc = parse_markdown(p)
    assert doc.title == "The Title"
    assert doc.text == "# The *Title*\n\nBody text.\n"


@pytest.mark.parametrize("text,title", [
    ("FEDERALIST No. 26\n\nThe Idea of Restraining the Legislative Authority\nConsidered.\n\nBody text here.",
     "FEDERALIST No. 26 — The Idea of Restraining the Legislative Authority Considered"),
    ("Quarterly Report\n\nRevenue grew. Costs fell.\n", "Quarterly Report"),
    ("Annual Review\n\nOverview of the year\n\nText.", "Annual Review — Overview of the year"),
    ("x " * 100, None),
])
def test_paragraph_title(text, title):
    assert paragraph_title(text) == title


def test_plain_text(tmp_path: Path):
    p = tmp_path / "a.txt"
    p.write_text("Memo\n\nAll staff must attend.\n")
    assert parse_plain(p).title == "Memo"


def test_docx(tmp_path: Path):
    p = tmp_path / "x.docx"
    write_docx(p, "Policy", ["Grace Hopper worked on COBOL.", "Second paragraph."])
    doc = format_for(p).parse(p)
    assert doc.title == "Policy"
    assert doc.text == "Policy\n\nGrace Hopper worked on COBOL.\n\nSecond paragraph.\n"


def test_pdf_pages_are_marked_and_masked(tmp_path: Path):
    p = tmp_path / "x.pdf"
    write_pdf(p, [["Marie Curie studied radioactivity in Paris."], ["Second page text."]])
    doc = format_for(p).parse(p)
    assert doc.meta["pages"] == 2
    assert doc.text.startswith("[page 1]\nMarie Curie")
    assert "[page 2]" in doc.text and "[page" not in doc.analysis_text


def test_pdf_without_text_is_an_error(tmp_path: Path):
    p = tmp_path / "scan.pdf"
    write_pdf(p, [[]])
    with pytest.raises(EmptyDocument):
        format_for(p).parse(p)


def test_supported_formats():
    for ext in (".md", ".markdown", ".txt", ".pdf", ".docx"):
        assert format_for(Path("a" + ext)) is not None
    assert format_for(Path("a.csv")) is None
