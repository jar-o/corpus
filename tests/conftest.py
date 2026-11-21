import re
import shutil
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "examples" / "sample-docs"
FEDERALIST = ROOT / "tests" / "integration" / "federalist"
FEDERALIST_ZIP = FEDERALIST / "papers.zip"
_PAPER = re.compile(r"federalist-\d{2}\.txt")


def extract_federalist(dest: Path) -> Path:
    """The 85 papers from papers.zip, unpacked flat into ``dest`` (nothing else)."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(FEDERALIST_ZIP) as z:
        for info in z.infolist():
            name = Path(info.filename).name
            if not info.is_dir() and _PAPER.fullmatch(name):  # flat, known names only: no path traversal
                (dest / name).write_bytes(z.read(info))
    return dest


@pytest.fixture
def federalist_dir(tmp_path: Path) -> Path:
    if not FEDERALIST_ZIP.exists():
        pytest.skip(f"missing {FEDERALIST_ZIP}")
    return extract_federalist(tmp_path / "federalist")


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    shutil.copytree(SAMPLE, root, ignore=shutil.ignore_patterns(".corpus-out"))
    return root


def alnum(text: str) -> str:
    """Letters/digits only, lowercased: compare quotes with markup-bearing source."""
    return re.sub(r"[\W_]+", "", text).lower()


def reference_lines(root: Path, path: str) -> list[str]:
    """Lines that line numbers for ``path`` refer to (source, or cached extraction)."""
    from corpus.parsers import mask_markdown

    if path.endswith((".pdf", ".docx")):
        f = root / ".corpus-out" / "text" / f"{path}.txt"
    else:
        f = root / path
    text = f.read_text(encoding="utf-8").replace("\r\n", "\n")
    if path.endswith((".md", ".markdown")):
        text = mask_markdown(text)  # quotes are shown without Markdown syntax
    return text.split("\n")
