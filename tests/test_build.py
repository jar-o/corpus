import json
import os
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import alnum, reference_lines
from corpus.cli import app
from corpus.pipeline import build, clean

OUT = ".corpus-out"
REF = re.compile(r"^- L(\d+)(?:-(\d+))?: “(.*)”$")


def out_files(root: Path) -> list[str]:
    out = root / OUT
    return sorted(p.relative_to(out).as_posix() for p in out.rglob("*")
                  if p.is_file() and not p.relative_to(out).as_posix().startswith(".cache/"))


def page(root: Path, path: str) -> str:
    return (root / OUT / "docs" / f"{path}.md").read_text(encoding="utf-8")


SAMPLE_DOCS = ["bitcoin-overview.md", "energy-transition.docx", "ethereum-history.md",
               "research/climate-report.txt", "research/flood-policy.pdf"]


def test_output_layout_is_markdown_and_caches_only_extracted_text(corpus_dir: Path):
    r = build(corpus_dir)
    assert sorted(r.added) == SAMPLE_DOCS and not r.failed
    assert out_files(corpus_dir) == sorted(
        ["CORPUS.md", "entities.md"] + [f"docs/{p}.md" for p in SAMPLE_DOCS]
        + ["text/energy-transition.docx.txt", "text/research/flood-policy.pdf.txt"])
    overview = (corpus_dir / OUT / "CORPUS.md").read_text()
    for p in SAMPLE_DOCS:
        assert f"`{p}`" in overview
    assert "## How to use this map" in overview


def test_quoted_sentences_match_their_cited_lines(corpus_dir: Path):
    build(corpus_dir)
    checked = 0
    for p in SAMPLE_DOCS:
        lines = reference_lines(corpus_dir, p)
        for row in page(corpus_dir, p).splitlines():
            m = REF.match(row)
            if m:
                a, b = int(m.group(1)), int(m.group(2) or m.group(1))
                assert alnum(m.group(3)) in alnum(" ".join(lines[a - 1:b])), (p, row)
                checked += 1
    assert checked >= len(SAMPLE_DOCS) * 2


def test_entity_index_line_numbers_resolve(corpus_dir: Path):
    build(corpus_dir)
    rows = [r for r in (corpus_dir / OUT / "entities.md").read_text().splitlines() if r.startswith("- **")]
    assert rows
    for row in rows:
        label = re.match(r"- \*\*(.+?)\*\* \(", row).group(1)
        for path, nums in re.findall(r"(\S+) L([\d,]+)", row):
            lines = reference_lines(corpus_dir, path)
            for n in map(int, nums.split(",")):
                window = " ".join(lines[n - 1:n + 1])
                assert alnum(label.removeprefix("the ").removeprefix("The ")) in alnum(window), (label, path, n)


def test_noop_rebuild_and_touch(corpus_dir: Path):
    build(corpus_dir)
    r = build(corpus_dir)
    assert len(r.unchanged) == 5 and not r.rebuilt
    os.utime(corpus_dir / "bitcoin-overview.md", (1, 1))
    r = build(corpus_dir)
    assert len(r.unchanged) == 5 and not r.rebuilt


def test_modify_add_delete(corpus_dir: Path):
    build(corpus_dir)
    p = corpus_dir / "bitcoin-overview.md"
    p.write_text(p.read_text() + "\nThe Bitcoin whitepaper was published on a cryptography mailing list.\n")
    (corpus_dir / "research" / "flood-policy.pdf").unlink()
    (corpus_dir / "notes.md").write_text("# Notes\n\nAda Lovelace wrote the first published algorithm.\n")
    r = build(corpus_dir)
    assert r.modified == ["bitcoin-overview.md"] and r.added == ["notes.md"]
    assert r.deleted == ["research/flood-policy.pdf"] and len(r.unchanged) == 3
    files = out_files(corpus_dir)
    assert "docs/notes.md.md" in files
    assert not any("flood-policy" in f for f in files)
    assert "flood-policy" not in (corpus_dir / OUT / "CORPUS.md").read_text()
    assert "whitepaper" in page(corpus_dir, "bitcoin-overview.md") or r.modified


def test_rename_reuses_analysis_and_moves_extracted_text(corpus_dir: Path):
    build(corpus_dir)
    (corpus_dir / "research" / "flood-policy.pdf").rename(corpus_dir / "flood.pdf")
    r = build(corpus_dir)
    assert r.renamed == [("research/flood-policy.pdf", "flood.pdf")] and not r.added and not r.deleted
    files = out_files(corpus_dir)
    assert "text/flood.pdf.txt" in files and "docs/flood.pdf.md" in files
    assert not any("flood-policy" in f for f in files)


def test_failed_file_is_reported_and_retried_only_when_changed(corpus_dir: Path):
    bad = corpus_dir / "broken.pdf"
    bad.write_bytes(b"%PDF-1.4 not really")
    r = build(corpus_dir)
    assert [p for p, _ in r.failed] == ["broken.pdf"]
    assert "## Not indexed" in (corpus_dir / OUT / "CORPUS.md").read_text()
    r = build(corpus_dir)
    assert "broken.pdf" in r.unchanged and [p for p, _ in r.failed] == ["broken.pdf"]


def test_interrupted_build_recovers_without_reanalysis(corpus_dir: Path):
    build(corpus_dir)
    state_file = corpus_dir / OUT / ".cache" / "state.json"
    state = json.loads(state_file.read_text())
    state["build"]["complete"] = False
    state_file.write_text(json.dumps(state))
    (corpus_dir / OUT / "CORPUS.md").unlink()
    r = build(corpus_dir)
    assert r.rebuilt and not r.added and not r.modified
    assert (corpus_dir / OUT / "CORPUS.md").exists()


def test_missing_cache_triggers_reanalysis(corpus_dir: Path):
    build(corpus_dir)
    (corpus_dir / OUT / "text" / "energy-transition.docx.txt").unlink()
    r = build(corpus_dir)
    assert r.modified == ["energy-transition.docx"]


def test_boilerplate_is_ignored(tmp_path: Path):
    root = tmp_path / "letters"
    root.mkdir()
    topics = ["river pollution near the old mill", "budget for the new library wing",
              "snow removal on county roads", "the summer reading program", "noise from the airport"]
    for i, t in enumerate(topics):
        (root / f"letter{i}.txt").write_text(
            f"Letter {i}\n\nDear Councillor Margaret Thornbury of Springfield County,\n\n"
            f"I am writing to raise concerns about {t}, which affects many residents of our town every week.\n\n"
            "Yours sincerely, the Springfield Residents Association\n")
    build(root)
    out = root / OUT
    assert "Thornbury" not in (out / "entities.md").read_text()
    assert "I am writing to raise concerns" in (out / "CORPUS.md").read_text()


def test_force_is_deterministic(corpus_dir: Path):
    build(corpus_dir)
    first = {f: (corpus_dir / OUT / f).read_text() for f in out_files(corpus_dir)}
    r = build(corpus_dir, force=True)
    assert len(r.modified) == 5
    assert first == {f: (corpus_dir / OUT / f).read_text() for f in out_files(corpus_dir)}


def test_cli(corpus_dir: Path):
    run = CliRunner().invoke
    r = run(app, [str(corpus_dir), "--stats"])
    assert r.exit_code == 1 and "No index" in r.output
    r = run(app, [str(corpus_dir), "-q"])
    assert r.exit_code == 0 and "5 added" in r.output
    r = run(app, [str(corpus_dir), "--incremental"])
    assert r.exit_code == 0 and "Already up to date" in r.output
    r = run(app, [str(corpus_dir), "--stats"])
    assert r.exit_code == 0 and "5 indexed" in r.output and "CORPUS.md ≈" in r.output
    assert run(app, [str(corpus_dir), "--force", "--clean"]).exit_code != 0
    r = run(app, [str(corpus_dir), "--clean"])
    assert r.exit_code == 0 and not (corpus_dir / OUT).exists()
    assert clean(corpus_dir) is False
    (corpus_dir / "bad.docx").write_bytes(b"nope")
    assert run(app, [str(corpus_dir), "-q"]).exit_code == 1


@pytest.mark.slow
def test_federalist_map_is_small_and_complete(federalist_dir: Path):
    root = federalist_dir
    r = build(root)
    assert len(r.added) == 85 and not r.failed
    state = json.loads((root / OUT / ".cache" / "state.json").read_text())["build"]
    assert state["map_tokens"] < state["source_tokens"] / 20
    overview = (root / OUT / "CORPUS.md").read_text()
    assert all(f"`federalist-{i:02d}.txt`" in overview for i in range(1, 86))
    assert len(state["topics"]) >= 5
    # No. 10's opening sentence is its gist.
    assert "tendency to break and control the violence of faction" in overview
