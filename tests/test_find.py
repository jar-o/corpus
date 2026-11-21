import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import ROOT
from corpus.cli import BEGIN, END, app
from corpus.embed import Embedder
from corpus.find import find_root, format_result, search, topic_query
from corpus.util import est_tokens
from corpus.pipeline import build

EMBEDDER = Embedder()


@pytest.fixture
def indexed(corpus_dir: Path) -> Path:
    build(corpus_dir)
    return corpus_dir


def paths(res):
    return [h.path for h in res.hits]


def test_finds_by_meaning_not_words(indexed: Path):
    # None of these words appear in the flood documents.
    res = search(indexed, "Which cities are threatened by the ocean?", embedder=EMBEDDER)
    assert set(paths(res)) <= {"research/climate-report.txt", "research/flood-policy.pdf"}
    assert paths(res)[0] in {"research/climate-report.txt", "research/flood-policy.pdf"}
    res = search(indexed, "digital money and its inventors", embedder=EMBEDDER)
    assert paths(res)[0] in {"bitcoin-overview.md", "ethereum-history.md"}
    assert "research/climate-report.txt" not in paths(res)


def test_evidence_lines_and_quote_are_real(indexed: Path):
    res = search(indexed, "When did Ethereum switch from mining to staking?", embedder=EMBEDDER)
    hit = res.hits[0]
    assert hit.path == "ethereum-history.md"
    lines = (indexed / hit.path).read_text().split("\n")
    a, b = hit.quote_lines
    assert re.sub(r"\W", "", hit.quote.rstrip(" …"))[:40] in re.sub(r"\W", "", " ".join(lines[a - 1:b]))
    assert all(1 <= r[0] <= r[1] <= len(lines) for r in hit.ranges)


def test_extracted_formats_point_at_cached_text(indexed: Path):
    res = search(indexed, "Dutch flood protection programme", embedder=EMBEDDER)
    hit = next(h for h in res.hits if h.path == "research/flood-policy.pdf")
    assert hit.extracted and hit.quote
    text = (indexed / ".corpus-out" / "text" / "research" / "flood-policy.pdf.txt").read_text()
    assert "Delta Programme" in text
    assert ".corpus-out/text/research/flood-policy.pdf.txt" in format_result(res, indexed)


def test_unrelated_question_is_flagged(indexed: Path):
    res = search(indexed, "a recipe for chocolate cake", embedder=EMBEDDER)
    assert res.weak and len(res.hits) <= 3
    assert "No document is clearly about this" in format_result(res, indexed)


def test_find_root_walks_up(indexed: Path):
    assert find_root(indexed / "research") == indexed.resolve()
    assert find_root(indexed.parent) is None


def test_cli_find_updates_index_first(indexed: Path, monkeypatch):
    (indexed / "volcanoes.md").write_text("# Volcanoes\n\nMount Etna in Sicily erupts frequently, "
                                          "sending lava flows toward nearby villages.\n")
    monkeypatch.chdir(indexed / "research")
    r = CliRunner().invoke(app, ["--find", "eruptions and lava in Italy"])
    assert r.exit_code == 0, r.output
    assert "1. volcanoes.md" in r.output


def test_claude_md_is_written_and_not_indexed(indexed: Path):
    run = CliRunner().invoke
    (indexed / "CLAUDE.md").write_text("# My notes\n\nKeep this.\n")
    assert run(app, [str(indexed), "--claude-md"]).exit_code == 0
    assert run(app, [str(indexed), "--claude-md"]).exit_code == 0  # idempotent
    text = (indexed / "CLAUDE.md").read_text()
    assert text.startswith("# My notes") and text.count(BEGIN) == 1 and text.count(END) == 1
    assert "--find" in text and str(indexed) in text
    r = build(indexed)
    assert "CLAUDE.md" not in r.added + r.unchanged


def test_excerpts_are_verbatim_and_within_budget(indexed: Path):
    res = search(indexed, "How did the Dutch respond to flooding?", embedder=EMBEDDER, budget=600)
    assert res.excerpt_docs and res.excerpt_docs[0].ranked
    total = 0
    for d in res.excerpt_docs:
        base = indexed / ".corpus-out" / "text" / f"{d.path}.txt" if d.extracted else indexed / d.path
        lines = base.read_text().split("\n")
        for x in d.excerpts:
            a, b = x.lines
            assert re.sub(r"\W", "", x.text)[:60] in re.sub(r"\W", "", " ".join(lines[a - 1:b]))
            total += est_tokens(len(x.text))
    assert total <= 600 + 200  # the budget may be exceeded only by each selected document's best passage
    out = format_result(res, indexed)
    assert out.startswith("Excerpts from") and "Answer from these" in out


def test_brief_output_has_no_excerpts(indexed: Path, monkeypatch):
    monkeypatch.chdir(indexed)
    r = CliRunner().invoke(app, ["--find", "Dutch flood protection", "--brief", "--no-update"])
    assert r.exit_code == 0, r.output
    assert r.output.startswith("Read ") and "Excerpts" not in r.output


def test_topic_query_drops_names_and_filler():
    t = topic_query("In the documents in this folder, tell me how do Hamilton and Madison differ on armies?",
                    {"hamilton": set(), "madison": set()})
    assert "Hamilton" not in t and "Madison" not in t and "folder" not in t and "armies" in t


def test_max_option(indexed: Path):
    res = search(indexed, "energy consumption", max_docs=1, embedder=EMBEDDER)
    assert len(res.hits) == 1


@pytest.mark.slow
def test_federalist_benchmark(federalist_dir: Path):
    """Retrieval quality on 30 questions with known answers. Prints the metrics."""
    root = federalist_dir
    build(root)
    questions = json.loads((ROOT / "tests" / "integration" / "federalist_questions.json").read_text())["questions"]
    n_docs, recall, precision, top1 = [], [], [], []
    for item in questions:
        res = search(root, item["q"], embedder=EMBEDDER)
        got = [int(re.search(r"\d+", p).group()) for p in paths(res)]
        expected = set(item["expected"])
        n_docs.append(len(got))
        recall.append(len(expected & set(got)) / len(expected))
        precision.append(len(expected & set(got)) / len(got))
        top1.append(got[0] in expected)
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    # Excerpts for a two-author comparison must include both authors' papers on the subject.
    res = search(root, "How do Hamilton and Madison differ on keeping an army during peaceful times?",
                 embedder=EMBEDDER)
    excerpted = {d.path for d in res.excerpt_docs}
    assert excerpted & {"federalist-24.txt", "federalist-25.txt", "federalist-26.txt"}
    assert excerpted & {"federalist-41.txt", "federalist-46.txt"}
    print(f"\nfind benchmark: docs/question {mean(n_docs):.1f}, recall {mean(recall):.2f}, "
          f"precision {mean(precision):.2f}, top-1 correct {mean(top1):.2f}")
    assert mean(n_docs) <= 4
    assert mean(top1) >= 0.8
    assert mean(recall) >= 0.7
    assert mean(precision) >= 0.65
