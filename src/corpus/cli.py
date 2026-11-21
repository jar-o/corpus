"""Command-line interface.

    corpus ./documents                build, or incrementally update, .corpus-out/
    corpus ./documents --incremental  same as the default, stated explicitly
    corpus ./documents --force        re-analyze every document
    corpus ./documents --stats        show what is indexed and the token savings
    corpus ./documents --clean        delete .corpus-out/
    corpus ./documents --claude-md    tell Claude Code (via CLAUDE.md) to use --find

    corpus --find "question"          from inside an indexed folder: the few
                                      documents that matter, ranked by meaning
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import typer

from . import OUTPUT_DIRNAME, __version__
from .find import EXCERPT_BUDGET, IndexMissing, find_root, format_result, search
from .nlp import ModelUnavailable
from .parsers import FORMATS
from .pipeline import build, clean, output_dir, read_state

app = typer.Typer(add_completion=False, help="Turn a folder of documents into a compact Markdown map for LLMs.")


def _version(value: bool) -> None:
    if value:
        typer.echo(f"corpus {__version__}")
        raise typer.Exit()


@app.command(help=f"Build or update the map for PATH (supported: {', '.join(sorted(FORMATS))}).")
def run(
    path: Optional[Path] = typer.Argument(None, exists=True, file_okay=False, resolve_path=True,
                                          help="Corpus root directory. Optional with --find/--stats: "
                                               "defaults to the indexed folder containing the current directory."),
    find: Optional[str] = typer.Option(None, "--find", "-f", metavar="QUESTION",
                                       help="Print the passages that answer QUESTION, from the most relevant documents."),
    max_docs: int = typer.Option(8, "--max", min=1, help="With --find: at most this many top-ranked documents."),
    budget: int = typer.Option(EXCERPT_BUDGET, "--budget", min=200, metavar="TOKENS",
                               help="With --find: about this many tokens of excerpts."),
    brief: bool = typer.Option(False, "--brief", help="With --find: list documents and line ranges only, no excerpts."),
    no_update: bool = typer.Option(False, "--no-update", help="With --find: skip the incremental index update."),
    claude_md: bool = typer.Option(False, "--claude-md", help="Add --find instructions to PATH/CLAUDE.md."),
    incremental: bool = typer.Option(False, "--incremental", help="Process only changed files (the default)."),
    force: bool = typer.Option(False, "--force", help="Re-analyze every document."),
    stats: bool = typer.Option(False, "--stats", help="Show index statistics and token estimates."),
    clean_: bool = typer.Option(False, "--clean", help=f"Delete {OUTPUT_DIRNAME}/."),
    model: str = typer.Option("en_core_web_sm", "--model", help="spaCy model for NER and sentence parsing."),
    topics: int = typer.Option(0, "--topics", min=0, help="Number of topics (0 = automatic)."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only print the summary line."),
    version: Optional[bool] = typer.Option(None, "--version", callback=_version, is_eager=True,
                                           help="Show version and exit."),
) -> None:
    chosen = [name for name, on in (("--incremental", incremental), ("--force", force), ("--stats", stats),
                                    ("--clean", clean_), ("--find", find is not None), ("--claude-md", claude_md)) if on]
    if len(chosen) > 1:
        raise typer.BadParameter(f"choose one of: {', '.join(chosen)}")

    if path is None:
        if find is None and not stats:
            raise typer.BadParameter("PATH is required (it may be omitted only with --find or --stats)")
        path = find_root(Path.cwd())
        if path is None:
            typer.secho(f"No indexed folder at or above {Path.cwd()}. Build one first: corpus <folder>",
                        fg=typer.colors.RED, err=True)
            raise typer.Exit(2)

    if find is not None:
        _find(path, find, max_docs, update=not no_update, model=model, topics=topics,
              budget=0 if brief else budget)
        return
    if claude_md:
        _claude_md(path)
        return

    if clean_:
        typer.echo(f"Removed {output_dir(path)}" if clean(path) else f"Nothing to clean at {output_dir(path)}")
        return
    if stats:
        _stats(path)
        return

    log = (lambda m: None) if quiet else (lambda m: typer.echo(f"  {m}"))
    try:
        r = build(path, model=model, num_topics=topics, force=force, log=log)
    except ModelUnavailable as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    typer.echo(f"{len(r.added)} added, {len(r.modified)} modified, {len(r.unchanged)} unchanged, "
               f"{len(r.deleted)} deleted, {len(r.renamed)} renamed ({r.seconds:.1f}s)")
    if r.failed:
        typer.secho(f"{len(r.failed)} file(s) could not be indexed:", fg=typer.colors.YELLOW)
        for p, err in r.failed:
            typer.echo(f"  {p}: {err}")
    typer.echo(f"Start here: {r.out_dir / 'CORPUS.md'}" if r.rebuilt else "Already up to date.")
    if r.failed:
        raise typer.Exit(1)


def _find(root: Path, question: str, max_docs: int, update: bool, model: str, topics: int,
          budget: int = EXCERPT_BUDGET) -> None:
    if not question.strip():
        raise typer.BadParameter("--find needs a question")
    if update:
        try:
            r = build(root, model=model, num_topics=topics)
        except ModelUnavailable as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(2)
        changes = len(r.added) + len(r.modified) + len(r.deleted) + len(r.renamed)
        if changes:
            typer.echo(f"(index updated: {changes} changed file(s))", err=True)
    try:
        res = search(root, question, max_docs=max_docs, budget=budget)
    except IndexMissing as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    typer.echo(format_result(res, root, Path.cwd()), nl=False)


BEGIN, END = "<!-- corpus:begin -->", "<!-- corpus:end -->"


def claude_md_section(root: Path) -> str:
    exe = Path(sys.executable).parent / "corpus"
    def q(x) -> str:  # quote only when needed, so permission rules can match a plain prefix
        x = str(x)
        return f'"{x}"' if re.search(r"\s", x) else x

    cmd = q(exe) if exe.exists() else "corpus"
    return f"""{BEGIN}
## Answering questions about these documents

This folder is indexed by `corpus`. For any question about its documents:

1. Always run this first, even if you think you know the answer:
   `{cmd} {q(root)} --find "<the question, in plain words>"`
   It ranks every document by meaning and prints the passages that bear on
   the question: several from the most relevant documents, then a few from
   lower-ranked ones for breadth, each with its path and line numbers.
2. Answer from those excerpts when they are enough. Do not grep, glob or
   list the folder. Open a file only for something the excerpts don't
   cover, and then read just the lines you need.
3. Cite documents by path and line numbers.
{END}
"""


def _claude_md(root: Path) -> None:
    target = root / "CLAUDE.md"
    section = claude_md_section(root)
    if target.exists():
        text = target.read_text(encoding="utf-8")
        pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n?", re.S)
        text = pattern.sub(lambda _: section, text) if pattern.search(text) else text.rstrip() + "\n\n" + section
    else:
        text = section
    target.write_text(text, encoding="utf-8")
    typer.echo(f"Wrote corpus instructions to {target}")


def _stats(root: Path) -> None:
    state = read_state(root)
    b = (state or {}).get("build", {})
    if not b.get("built_at"):
        typer.echo(f"No index at {output_dir(root)}. Run: corpus {root}")
        raise typer.Exit(1)
    out = output_dir(root)
    formats: dict[str, int] = {}
    for e in state["files"].values():
        formats[e["format"]] = formats.get(e["format"], 0) + 1
    disk = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    cache = sum(p.stat().st_size for p in (out / ".cache").rglob("*") if p.is_file())
    ratio = b["source_tokens"] / b["map_tokens"] if b.get("map_tokens") else 0
    typer.echo(f"Corpus:       {root}")
    typer.echo(f"Last build:   {b['built_at']}  ({'complete' if b.get('complete') else 'INCOMPLETE - rerun corpus'})")
    typer.echo(f"Documents:    {b['documents']} indexed, {b['failed']} failed  "
               + " ".join(f"{k}={v}" for k, v in sorted(formats.items())))
    typer.echo(f"Tokens (est): source ≈ {b['source_tokens']:,} · CORPUS.md ≈ {b['map_tokens']:,} "
               f"({ratio:.0f}x smaller) · doc page ≈ {b['doc_page_tokens_avg']:,} avg · "
               f"entities.md ≈ {b['entities_tokens']:,}")
    typer.echo(f"Output size:  {disk / 1024:,.0f} KiB (of which hidden cache {cache / 1024:,.0f} KiB)")
    typer.echo(f"Model:        {b['models']['model']}, spaCy {b['models']['spacy']}")
    if b.get("topics"):
        typer.echo("Topics:")
        for t in b["topics"]:
            typer.echo(f"  {t['id']:<4} {t['documents']:>4} docs  {t['label']}")
    failed = sorted((p, e["error"]) for p, e in state["files"].items() if e["status"] != "ok")
    if failed:
        typer.echo("Failed:")
        for p, err in failed:
            typer.echo(f"  {p}: {err}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
