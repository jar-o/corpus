"""Render the corpus model as Markdown for LLM consumption.

Layout (paths relative to the corpus root):

    .corpus-out/CORPUS.md          entry point: topics -> documents with one-line gists
    .corpus-out/docs/<path>.md     per document: summary, phrases, names, related docs
    .corpus-out/entities.md        name -> documents and lines (grep it)
    .corpus-out/topics/<Tn>.md     full member lists, only for large corpora
    .corpus-out/text/<path>.txt    extracted text of PDF/DOCX files

Every reference is a document path plus line numbers (``L12-15``), so a
reader can open exactly the lines it needs with an ordinary file read.
"""

from __future__ import annotations

from datetime import datetime, timezone

from . import OUTPUT_DIRNAME, __version__
from .aggregate import Corpus, Doc
from .util import entity_key, est_tokens

INLINE_ALL_DOCS_UP_TO = 150  # beyond this, CORPUS.md lists the top docs per topic only
INLINE_PER_TOPIC = 12
GIST_CHARS = 220
SUMMARY_SENTENCES = 3
DOC_PAGE_KEYWORDS = 10
OVERVIEW_ENTITIES = 30
OVERVIEW_PHRASES = 30
COMMON_SHARE = 0.5  # names in more than this share of documents are "common"
FREQUENT_SHARE = 0.2  # names above this share are listed by count only
CONNECTING_SHARE = 0.1  # "connecting" names appear in 2+ docs but at most this share
CONNECTING_DOCS_LISTED = 5

OUT = OUTPUT_DIRNAME


def doc_page(path: str) -> str:
    return f"{OUT}/docs/{path}.md"


def text_file(path: str) -> str:
    return f"{OUT}/text/{path}.txt"


def lref(lines: list[int]) -> str:
    a, b = lines[0], lines[-1]
    return f"L{a}" if a == b else f"L{a}-{b}"


def lines_list(lines: list[int]) -> str:
    return "L" + ",".join(str(x) for x in sorted(lines))


def clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    return text[: cut if cut > limit * 0.6 else limit].rstrip(" ,;:") + " …"


def gist(d: Doc) -> dict | None:
    """One sentence for the overview: the opening sentence, else the top
    TextRank sentence."""
    return d.lead or (d.summary[0] if d.summary else None)


def _doc_line(d: Doc) -> str:
    parts = [f"`{d.path}`"]
    if d.title:
        parts.append(clip(d.title, 140))
    line = "- " + " — ".join(parts)
    g = gist(d)
    if g:
        line += f" — “{clip(g['text'], GIST_CHARS)}” {lref(g['lines'])}"
    return line


def _is_common(e: dict, n: int) -> bool:
    return n >= 10 and len(e["docs"]) > COMMON_SHARE * n


def render(corpus: Corpus, root_name: str, failed: list[tuple[str, str]], models: dict) -> dict[str, str]:
    """Return {relative output path: content}."""
    files: dict[str, str] = {}
    docs = corpus.docs
    n = len(docs)
    by_path = {d.path: d for d in docs}
    source_chars = sum(d.result["chars"] for d in docs)
    spill = n > INLINE_ALL_DOCS_UP_TO

    # ------------------------------------------------------------ document pages
    for d in docs:
        files[f"docs/{d.path}.md"] = _render_doc(d, corpus)

    # ------------------------------------------------------------ topic pages (large corpora only)
    if spill:
        for t in corpus.topics:
            body = [f"# {t.id} · {t.label}", "",
                    f"Characteristic words: {', '.join(t.terms)}", "",
                    f"{len(t.docs)} documents whose main topic is {t.id}, strongest first:", ""]
            body += [_doc_line(by_path[p]) for p, _ in t.docs]
            files[f"topics/{t.id}.md"] = "\n".join(body) + "\n"

    # ------------------------------------------------------------ entities.md
    files["entities.md"] = _render_entities(corpus, root_name)

    # ------------------------------------------------------------ CORPUS.md
    fmt_counts: dict[str, int] = {}
    for d in docs:
        fmt_counts[d.format] = fmt_counts.get(d.format, 0) + 1
    fmts = ", ".join(f"{c} {f}" for f, c in sorted(fmt_counts.items()))
    out = [
        f"# Corpus map: {root_name}",
        "",
        f"{n} documents ({fmts}) · source text ≈ {est_tokens(source_chars):,} tokens · this map ≈ @@SELF@@ tokens",
        "",
        "## How to use this map",
        "",
        "1. Skim the topics below. Each document is listed once, under its main topic, with one representative sentence.",
        f"2. For promising documents open `{OUT}/docs/<path>.md`: summary, key phrases, names, related documents "
        "(a few hundred tokens each).",
        "3. Read only the cited lines of the source. `L12-15` means lines 12–15 of that document "
        f"(for PDF/DOCX files: of `{OUT}/text/<path>.txt`).",
        f"4. To find every document mentioning a name, search `{OUT}/entities.md`.",
        "",
        "Paths are relative to the corpus root. Quoted sentences are verbatim from the source. Topics are statistical "
        "word groupings, names and phrases are automatic extractions that can be wrong, and \"related\" means similar "
        "vocabulary — not a stated relationship.",
        "",
    ]

    assigned = set()
    if corpus.topics:
        out += ["## Documents by topic", ""]
        for t in corpus.topics:
            out += [f"### {t.id} · {t.label} ({len(t.docs)} documents)", ""]
            members = t.docs[:INLINE_PER_TOPIC] if spill else t.docs
            out += [_doc_line(by_path[p]) for p, _ in members]
            assigned.update(p for p, _ in t.docs)
            if len(t.docs) > len(members):
                out.append(f"- … {len(t.docs) - len(members)} more in `{OUT}/topics/{t.id}.md`")
            out.append("")
    rest = [d for d in docs if d.path not in assigned]
    if rest:
        out += ["## Documents" if not corpus.topics else "## Other documents", ""]
        out += [_doc_line(d) for d in rest] + [""]

    named = [e for e in corpus.entities if e["type"] != "DATE" and len(e["docs"]) >= 2]
    common = [e for e in named if _is_common(e, n)]
    frequent = [e for e in named if not _is_common(e, n) and n >= 10 and len(e["docs"]) > FREQUENT_SHARE * n]
    connecting = sorted((e for e in named if len(e["docs"]) <= max(2, CONNECTING_SHARE * n)),
                        key=lambda e: (-e["mentions"], -len(e["docs"]), e["label"].casefold()))[:OVERVIEW_ENTITIES]
    if named:
        out += ["## Names that connect documents", ""]
        if common:
            out += ["In most documents: " + ", ".join(f"{e['label']} ({len(e['docs'])})" for e in common[:20]) + ".", ""]
        if frequent:
            out += ["In many documents: " + ", ".join(f"{e['label']} ({len(e['docs'])})" for e in frequent[:30])
                    + f". Documents and lines for every name: `{OUT}/entities.md`.", ""]
        if connecting:
            out += ["Distinctive names shared by a few documents:", ""]
        for e in connecting:
            paths = list(e["docs"])
            more = len(paths) - CONNECTING_DOCS_LISTED
            listed = ", ".join(f"`{p}`" for p in paths[:CONNECTING_DOCS_LISTED]) + (f" +{more}" if more > 0 else "")
            out.append(f"- **{e['label']}** ({e['type'].lower()}): {listed}")
        out.append("")

    name_keys = {entity_key(e["label"]) for e in corpus.entities}
    recurring = [p for p in corpus.phrases
                 if len(p["docs"]) >= 2 and entity_key(p["text"]) not in name_keys][:OVERVIEW_PHRASES]
    if recurring:
        out += ["## Recurring key phrases", "",
                " · ".join(f"{p['text']} ({len(p['docs'])})" for p in recurring), ""]

    if failed:
        out += ["## Not indexed", ""] + [f"- `{p}`: {err}" for p, err in failed] + [""]

    built = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out += ["---", f"Generated by corpus {__version__} on {built} · spaCy {models.get('model')} · "
            f"{corpus.topic_model.get('library')} {corpus.topic_model.get('algorithm', '')} "
            f"({len(corpus.topics)} topics)".rstrip()]
    text = "\n".join(out) + "\n"
    files["CORPUS.md"] = text.replace("@@SELF@@", f"{est_tokens(len(text)):,}")
    return files


def _render_doc(d: Doc, corpus: Corpus) -> str:
    r = d.result
    out = [f"# {d.path}", ""]
    if d.title:
        out.append(f"Title: {d.title}")
    where = f"text extracted to `{text_file(d.path)}` (line numbers refer to it)" if d.extracted \
        else "line numbers refer to the source file"
    extra = f" · {r['meta']['pages']} pages" if "pages" in r.get("meta", {}) else ""
    out.append(f"Format: {d.format}{extra} · {r['lines']:,} lines · ≈ {est_tokens(r['chars']):,} tokens · {where}")
    if d.topics:
        labels = {t.id: t.label for t in corpus.topics}
        out.append("Topics: " + "; ".join(f"{tid} {labels[tid]} ({w:.2f})" for tid, w in d.topics))
    out.append("")

    picked = ([d.lead] if d.lead else []) + [s for s in d.summary if not d.lead or s["text"] != d.lead["text"]]
    if picked:
        out += ["## Key sentences (verbatim; opening sentence first, then TextRank picks)", ""]
        out += [f"- {lref(s['lines'])}: “{s['text']}”" for s in picked[:SUMMARY_SENTENCES + 1]] + [""]

    if d.keywords:
        out += ["## Key phrases", "",
                "; ".join(f"{k['text']} ({lines_list(k['lines'][:4])})" for k in d.keywords[:DOC_PAGE_KEYWORDS]), ""]

    names = [e for e in d.entities if e["type"] != "DATE"]
    dates = [e for e in d.entities if e["type"] == "DATE"]
    if names:
        out += ["## Names", ""]
        by_type: dict[str, list[dict]] = {}
        for e in names:
            by_type.setdefault(e["type"], []).append(e)
        for etype in sorted(by_type, key=lambda t: (-len(by_type[t]), t)):
            items = sorted(by_type[etype], key=lambda e: (-e["count"], e["lines"][0]))
            out.append(f"- {etype}: " + "; ".join(f"{e['text']} ({lines_list(e['lines'][:4])})" for e in items))
        out.append("")
    if dates:
        out += ["## Dates", "", "; ".join(f"{e['text']} ({lines_list(e['lines'][:3])})" for e in dates), ""]

    if d.related:
        out += ["## Related documents (similar vocabulary)", ""]
        out += [f"- `{rel['path']}` ({rel['score']:.2f}) — shared: {', '.join(rel['shared'])}" for rel in d.related]
        out.append("")
    return "\n".join(out)


def _render_entities(corpus: Corpus, root_name: str) -> str:
    n = len(corpus.docs)
    out = [f"# Name index: {root_name}", "",
           "Names found by automatic named-entity recognition (types can be wrong). "
           "Each entry: name (type) — documents with line numbers. Sorted by number of documents.", ""]
    for e in corpus.entities:
        if e["type"] == "DATE":
            continue
        head = f"- **{e['label']}** ({e['type']})"
        if _is_common(e, n):
            out.append(f"{head} — in {len(e['docs'])} of {n} documents (too common to list)")
        else:
            refs = " · ".join(f"{p} {lines_list(lines[:5])}" for p, lines in e["docs"].items())
            out.append(f"{head} — {refs}")
    return "\n".join(out) + "\n"
