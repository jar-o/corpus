"""``corpus --find``: pick the FEWEST, MOST RELEVANT documents for a question.

Ranking is semantic: the question is embedded with the same local model as
the passages, so documents are found by meaning ("judiciary" finds passages
about courts, judges and tribunals that never use the word).

Document score = weighted mix of
  * its best passage similarity            (one strongly matching passage)
  * the mean of its 3 best passages        (sustained discussion)
  * its document-level vector similarity   (title + opening + key sentences)
  * coverage: share of its passages close to the best passage in the corpus
plus a small keyword component (BM25 over noun/verb/adjective lemmas, at most
W_KEYWORD) that breaks ties in favor of documents using the question's own
terms. (A boost for names shared by question and document was tried and
removed: it favored documents naming generic institutions.) Meaning
dominates: the keyword part can move a
document by at most W_KEYWORD, while semantic scores span far more. Weights and the cutoff were tuned on
tests/integration/federalist_questions.json (see test_find.py).

Selection keeps documents scoring within CUTOFF of the top score, measured
as a fraction of the gap between the top and the median document, capped
at MAX_DOCS. A few runners-up are listed by path only.

Excerpts (the default output) let the LLM answer without opening files:
passages are chosen across the whole corpus by similarity, within a token
budget, so lower-ranked documents contribute their best passages too
(breadth over depth, at most EXCERPT_PER_DOC per document). Each selected
document gets its best passage first, and each name in the question (an
author, a person, a place) gets passages from up to NAME_DOCS documents
that mention it, so "how do A and B differ" sees both sides.
"""

from __future__ import annotations

import bisect
import gzip
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import math

import numpy as np

from . import OUTPUT_DIRNAME
from .embed import Embedder
from .parsers import mask_markdown, normalize_text, read_text_file
from .util import collapse, entity_key, est_tokens

W_MAX, W_TOP3, W_DOC, W_COVER = 0.25, 0.40, 0.25, 0.10
W_KEYWORD = 0.08
BM25_K1, BM25_B = 1.2, 0.75
WEAK_SHOW = 3
COVER_MARGIN = 0.06
CUTOFF = 0.30  # keep docs with score >= top - CUTOFF * (top - median)
ALSO_CUTOFF = 0.50
MAX_DOCS = 8
MAX_ALSO = 4
NAME_MAX_SHARE = 0.8  # names in more documents than this don't discriminate
RANGE_MARGIN = 0.03  # passages within this of a doc's best passage are "most relevant"
MAX_RANGES = 3
NO_MATCH_BELOW = 0.55  # best passage similarity under this: nothing is clearly about it
QUOTE_CHARS = 240
EXCERPT_BUDGET = 4000  # tokens of excerpt text, by default
EXCERPT_MARGIN = 0.10  # excerpts must be within this similarity of the best passage
NAME_MARGIN = 0.15  # looser floor for a passage that covers a name in the question
NAME_DOCS = 2  # each name in the question gets passages from up to this many documents
EXCERPT_PER_DOC = 3
BYLINE_CHARS = 160
_META = re.compile(r"\b(?:(?:in|from|across|within|of|using)\s+(?:all\s+)?(?:the|these|this|my|our)\s+"
                   r"(?:documents?|files?|folders?|papers?|corpus|texts?)(?:\s+in\s+(?:this|the)\s+folder)?"
                   r"|tell me|please|can you|could you|explain)\b", re.I)
_DOC_REF = re.compile(r"(?:\bno\.?|\bnumber|\bpaper|\bdoc(?:ument)?|#)\s*(\d{1,5})\b", re.I)


class IndexMissing(RuntimeError):
    pass


@dataclass
class Hit:
    index: int
    path: str
    title: str | None
    score: float
    tokens: int
    extracted: bool
    ranges: list[list[int]] = field(default_factory=list)
    quote: str | None = None
    quote_lines: list[int] | None = None
    reason: str | None = None


@dataclass
class Excerpt:
    doc: int
    lines: list[int]
    text: str
    sim: float


@dataclass
class ExcerptDoc:
    index: int
    path: str
    title: str | None
    byline: str
    tokens: int
    extracted: bool
    ranked: bool  # one of the selected documents (else: lower-ranked, included for breadth)
    excerpts: list[Excerpt] = field(default_factory=list)


@dataclass
class Result:
    query: str
    total_docs: int
    hits: list[Hit]
    also: list[str]
    weak: bool
    excerpt_docs: list[ExcerptDoc] = field(default_factory=list)


def find_root(start: Path) -> Path | None:
    """The nearest folder at or above ``start`` that has a corpus index."""
    start = start.resolve()
    for folder in [start, *start.parents]:
        if (folder / OUTPUT_DIRNAME / ".cache" / "state.json").exists():
            return folder
    return None


class Index:
    def __init__(self, root: Path):
        cache = root / OUTPUT_DIRNAME / ".cache"
        try:
            z = np.load(cache / "index.npz")
            self.meta = json.loads((cache / "index.json").read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise IndexMissing(f"no search index under {cache}; run: corpus {root}") from exc
        self.root = root
        self.cache = cache
        self.pvec = z["passage_vectors"].astype(np.float32)
        self.pdoc = z["passage_doc"]
        self.plines = z["passage_lines"]
        self.pspan = z["passage_span"]
        self.dvec = z["doc_vectors"].astype(np.float32)
        self.docs = self.meta["docs"]
        self.term_indptr = z["term_indptr"]
        self.term_indices = z["term_indices"]
        self.term_counts = z["term_counts"]
        self.vocab = {t: k for k, t in enumerate(self.meta.get("vocab", []))}

    def doc_result(self, i: int) -> dict:
        path = self.cache / "docs" / f"{self.docs[i]['id']}.json.gz"
        return json.loads(gzip.decompress(path.read_bytes()))

    def reference_text(self, i: int) -> str:
        d = self.docs[i]
        if d["extracted"]:
            return (self.root / OUTPUT_DIRNAME / "text" / f"{d['path']}.txt").read_text(encoding="utf-8")
        text = normalize_text(read_text_file(self.root / d["path"]))
        return mask_markdown(text) if d["format"] == "markdown" else text


_lemmatizer = None


def query_lemmas(text: str) -> list[str]:
    """Lemmas of the question's content words, in the same space as the index."""
    global _lemmatizer
    if _lemmatizer is None:
        import spacy

        try:
            _lemmatizer = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except OSError:
            _lemmatizer = False
    if not _lemmatizer:
        return [w for w in re.findall(r"[a-z]{3,}", text.lower())]
    return [t.lemma_.lower() for t in _lemmatizer(text)
            if t.is_alpha and not t.is_stop and len(t) > 2 and t.pos_ in ("NOUN", "PROPN", "ADJ", "VERB")]


def keyword_scores(index: "Index", query: str) -> np.ndarray:
    """BM25 per document, scaled so the best document scores 1."""
    n = len(index.docs)
    terms = {index.vocab[t] for t in query_lemmas(query) if t in index.vocab}
    out = np.zeros(n, dtype=np.float32)
    if not terms:
        return out
    doc_len = np.array([index.term_counts[index.term_indptr[i]:index.term_indptr[i + 1]].sum() for i in range(n)],
                       dtype=np.float32)
    avg = float(doc_len.mean()) or 1.0
    df = {t: 0 for t in terms}
    tf = [dict() for _ in range(n)]
    for i in range(n):
        a, b = index.term_indptr[i], index.term_indptr[i + 1]
        for k, c in zip(index.term_indices[a:b], index.term_counts[a:b]):
            if k in df:
                df[k] += 1
                tf[i][k] = c
    for i in range(n):
        for k, f in tf[i].items():
            idf = math.log(1 + (n - df[k] + 0.5) / (df[k] + 0.5))
            out[i] += idf * f * (BM25_K1 + 1) / (f + BM25_K1 * (1 - BM25_B + BM25_B * doc_len[i] / avg))
    top = out.max()
    return out / top if top > 0 else out


def _named_docs(index: Index, query: str) -> tuple[dict[int, list[str]], list[str]]:
    """Documents mentioning names (people, places, organizations...) that
    appear in the question."""
    q = query.casefold()
    n = len(index.docs)
    hits: dict[int, list[str]] = {}
    matched = []
    for name in index.meta["names"]:
        key = entity_key(name["label"])
        if len(key) < 3 or len(name["docs"]) > NAME_MAX_SHARE * n:
            continue
        if re.search(rf"(?<!\w){re.escape(key)}(?!\w)", q):
            matched.append(name["label"])
            for d in name["docs"]:
                hits.setdefault(d, []).append(name["label"])
    return hits, matched


def _name_coverage(index: Index, query: str, named: dict[int, list[str]], matched: list[str]) -> dict[str, set[int]]:
    """Names in the question -> documents that mention them. Uses the entity
    index, plus capitalized question words looked up in the term index (NER
    misses names in bylines and headings, e.g. an author line "MADISON")."""
    out: dict[str, set[int]] = {}
    for label in matched:
        out.setdefault(entity_key(label), set()).update(d for d, labels in named.items() if label in labels)
    n = len(index.docs)
    try:
        from spacy.lang.en.stop_words import STOP_WORDS
    except ImportError:
        STOP_WORDS = set()
    for word in re.findall(r"\b[A-Z][A-Za-z'-]{2,}\b", query):
        w = word.lower()
        k = index.vocab.get(w)
        if w in STOP_WORDS or k is None:
            continue
        docs = {i for i in range(n)
                if k in index.term_indices[index.term_indptr[i]:index.term_indptr[i + 1]]}
        if 0 < len(docs) <= NAME_MAX_SHARE * n:
            out.setdefault(w, set()).update(docs)
    return out


def _referenced_docs(index: Index, query: str) -> set[int]:
    """Documents the question points at directly: "No. 78", "paper 10", or a file name."""
    refs = set()
    numbers = {int(m.group(1)) for m in _DOC_REF.finditer(query)}
    q = query.casefold()
    for i, d in enumerate(index.docs):
        stem = Path(d["path"]).stem.casefold()
        if len(stem) >= 4 and (stem in q or Path(d["path"]).name.casefold() in q):
            refs.add(i)
            continue
        doc_numbers = {int(x) for x in re.findall(r"\d+", Path(d["path"]).stem)}
        title = d.get("title") or ""
        doc_numbers |= {int(x) for x in re.findall(r"\bNo\.?\s*(\d+)", title)}
        if numbers & doc_numbers:
            refs.add(i)
    return refs


def search(root: Path, query: str, max_docs: int = MAX_DOCS, embedder: Embedder | None = None,
           budget: int = EXCERPT_BUDGET) -> Result:
    """Rank documents for ``query``. With ``budget`` > 0, also pick excerpts
    (about ``budget`` tokens of passage text) to answer from."""
    index = Index(root)
    n = len(index.docs)
    if n == 0:
        return Result(query, 0, [], [], True)
    emb = embedder or Embedder()
    q = emb.query(query)
    psim = index.pvec @ q
    dsim = index.dvec @ q
    best_overall = float(psim.max()) if len(psim) else 0.0

    scores = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s = np.sort(psim[index.pdoc == i])[::-1]
        if len(s) == 0:
            scores[i] = dsim[i]
            continue
        cover = float((s >= best_overall - COVER_MARGIN).mean())
        scores[i] = W_MAX * s[0] + W_TOP3 * s[:3].mean() + W_DOC * dsim[i] + W_COVER * cover

    scores += W_KEYWORD * keyword_scores(index, query)
    named, matched_names = _named_docs(index, query)  # explains hits, covers names in excerpts; no score change
    refs = _referenced_docs(index, query)

    order = [int(i) for i in np.argsort(-scores, kind="stable")]
    top, med = float(scores[order[0]]), float(np.median(scores))
    spread = max(top - med, 1e-6)
    selected = [i for i in order[:max_docs] if scores[i] >= top - CUTOFF * spread]
    selected = [i for i in refs if i not in selected] + selected  # explicit references first
    selected = selected[: max(max_docs, len(refs))]
    also = [index.docs[i]["path"] for i in order if i not in selected
            and scores[i] >= top - ALSO_CUTOFF * spread][:MAX_ALSO]
    weak = best_overall < NO_MATCH_BELOW and not refs
    if weak:
        selected, also = order[:WEAK_SHOW], []

    hits = []
    for i in selected:
        d = index.docs[i]
        hit = Hit(index=i, path=d["path"], title=d.get("title"), score=float(scores[i]),
                  tokens=est_tokens(d["chars"]), extracted=d["extracted"])
        if i in refs:
            hit.reason = "named in the question"
        elif i in named:
            hit.reason = "mentions " + ", ".join(sorted(set(named[i]))[:3])
        _add_evidence(index, emb, q, psim, hit)
        hits.append(hit)
    res = Result(query, n, hits, also, weak)
    if budget > 0 and not weak:
        names = _name_coverage(index, query, named, matched_names)
        topic_sim = None
        if names:  # rank each name's documents by the topic alone: names in the query blur the embedding
            topic = topic_query(query, names)
            topic_sim = index.pvec @ emb.query(topic) if topic else None
        res.excerpt_docs = _pick_excerpts(index, psim, hits, names, budget, topic_sim)
        shown = {d.path for d in res.excerpt_docs}
        res.also = [p for p in also if p not in shown]
    return res


def topic_query(query: str, names: dict[str, set[int]]) -> str:
    """The question without its names and without "in the documents in this folder"-style filler."""
    t = _META.sub(" ", query)
    for name in names:
        t = re.sub(rf"(?<!\w){re.escape(name)}(?:'s)?(?!\w)", " ", t, flags=re.I)
    t = re.sub(r"\b(?:and|or|vs\.?|versus)\s+(?=(?:and|or|vs\.?|versus|differ|compare|agree|disagree|on|about)\b)",
               " ", collapse(t), flags=re.I)
    return collapse(t) if len(re.findall(r"\w{3,}", t)) >= 2 else ""


def _pick_excerpts(index: Index, psim: np.ndarray, hits: list[Hit], names: dict[str, set[int]],
                   budget: int, topic_sim: np.ndarray | None = None) -> list[ExcerptDoc]:
    """Passages to answer from, chosen across the corpus within a token budget."""
    best = float(psim.max())
    order = [int(p) for p in np.argsort(-psim, kind="stable")]
    texts: dict[int, str] = {}
    picked: list[int] = []
    per_doc: dict[int, int] = {}
    used = 0

    def text_of(d: int) -> str | None:
        if d not in texts:
            try:
                texts[d] = index.reference_text(d)
            except (OSError, ValueError):
                texts[d] = ""
        return texts[d] or None

    def add(p: int, floor: float) -> bool:
        nonlocal used
        d = int(index.pdoc[p])
        if p in picked or psim[p] < floor or per_doc.get(d, 0) >= EXCERPT_PER_DOC or text_of(d) is None:
            return False
        cost = est_tokens(int(index.pspan[p][1] - index.pspan[p][0])) + 8
        if used + cost > budget and picked:
            return False
        picked.append(p)
        per_doc[d] = per_doc.get(d, 0) + 1
        used += cost
        return True

    floor = best - EXCERPT_MARGIN
    for h in hits:  # every selected document: its best passage
        mine = [p for p in order if index.pdoc[p] == h.index]
        if mine:
            add(mine[0], -1.0)
    tsim = psim if topic_sim is None else topic_sim
    t_order = [int(p) for p in np.argsort(-tsim, kind="stable")]
    t_floor = float(tsim.max()) - NAME_MARGIN
    for docs in names.values():  # every name in the question: passages from documents mentioning it
        covered = {int(index.pdoc[p]) for p in picked} & docs
        for p in t_order:
            if len(covered) >= NAME_DOCS or tsim[p] < t_floor:
                break
            d = int(index.pdoc[p])
            if d in docs and d not in covered and add(p, -1.0):
                covered.add(d)
    for p in order:  # then the best passages anywhere, breadth first through the per-document cap
        if psim[p] < floor or used >= budget:
            break
        add(p, floor)

    hit_rank = {h.index: k for k, h in enumerate(hits)}
    by_doc: dict[int, list[int]] = {}
    for p in picked:
        by_doc.setdefault(int(index.pdoc[p]), []).append(p)
    doc_order = sorted(by_doc, key=lambda d: (d not in hit_rank, hit_rank.get(d, 0),
                                              -max(float(psim[p]) for p in by_doc[d])))
    out = []
    for d in doc_order:
        meta = index.docs[d]
        text = texts[d]
        ed = ExcerptDoc(index=d, path=meta["path"], title=meta.get("title"), byline=byline(text, meta.get("title")),
                        tokens=est_tokens(meta["chars"]), extracted=meta["extracted"], ranked=d in hit_rank)
        for p in sorted(by_doc[d], key=lambda p: index.plines[p][0]):
            s0, e0 = (int(x) for x in index.pspan[p])
            body = "\n".join(c for c in (collapse(par) for par in re.split(r"\n\s*\n", text[s0:e0])) if c)
            ed.excerpts.append(Excerpt(doc=d, lines=[int(x) for x in index.plines[p]], text=body, sim=float(psim[p])))
        out.append(ed)
    return out


def byline(text: str, title: str | None) -> str:
    """The short lines at the top of a document (dates, authors, addressees), minus the title."""
    title_key = re.sub(r"\W", "", (title or "").casefold())
    parts = []
    for par in re.split(r"\n\s*\n", text[:2000])[:10]:
        par = par.strip()
        if not par:
            continue
        par = collapse(par)
        key = re.sub(r"\W", "", par.casefold())
        if key and key in title_key:
            continue
        if len(par) > 100:  # body text starts
            break
        if key:
            parts.append(par)
    out = " · ".join(parts)
    return out if len(out) <= BYLINE_CHARS else out[:BYLINE_CHARS].rsplit(" ", 1)[0] + " …"


def _add_evidence(index: Index, emb: Embedder, q: np.ndarray, psim: np.ndarray, hit: Hit) -> None:
    """Most relevant line ranges, and the single sentence closest to the question."""
    mine = np.where(index.pdoc == hit.index)[0]
    if len(mine) == 0:
        return
    ranked = mine[np.argsort(-psim[mine], kind="stable")]
    best = float(psim[ranked[0]])
    chosen = sorted((int(p) for p in ranked[:MAX_RANGES] if psim[p] >= best - RANGE_MARGIN),
                    key=lambda p: index.plines[p][0])
    for p in chosen:
        a, b = (int(x) for x in index.plines[p])
        if hit.ranges and a <= hit.ranges[-1][1] + 1:
            hit.ranges[-1][1] = max(hit.ranges[-1][1], b)
        else:
            hit.ranges.append([a, b])

    try:
        text = index.reference_text(hit.index)
        result = index.doc_result(hit.index)
    except (OSError, ValueError):
        return
    s0, e0 = (int(x) for x in index.pspan[ranked[0]])
    spans = [sp for sp in result["sent_spans"] if sp[0] >= s0 and sp[1] <= e0]
    sents = [(sp, collapse(text[sp[0]:sp[1]])) for sp in spans]
    sents = [(sp, t) for sp, t in sents if len(t.split()) >= 5] or sents
    if not sents:
        return
    vecs = emb.passages([t for _, t in sents])
    k = int(np.argmax(vecs @ q))
    (s, e), quote = sents[k]
    starts = [0] + [m.end() for m in re.finditer("\n", text)]
    hit.quote_lines = [bisect.bisect_right(starts, s), bisect.bisect_right(starts, e - 1)]
    hit.quote = quote if len(quote) <= QUOTE_CHARS else quote[:QUOTE_CHARS].rsplit(" ", 1)[0] + " …"


def lines_label(r: list[int]) -> str:
    return f"L{r[0]}" if r[0] == r[1] else f"L{r[0]}-{r[1]}"


def format_result(res: Result, root: Path, cwd: Path | None = None) -> str:
    out = []
    shown_root = "" if cwd is not None and cwd.resolve() == root.resolve() else f" (paths relative to {root})"
    if not res.hits:
        return f"No documents are indexed under {root}."
    if res.excerpt_docs and not res.weak:
        return format_excerpts(res, shown_root)
    if res.weak:
        out.append(f"No document is clearly about this. Closest of {res.total_docs}{shown_root}:")
    else:
        n = len(res.hits)
        what = "this document" if n == 1 else f"these {n} documents"
        out.append(f"Read {what} (of {res.total_docs}){shown_root}, most relevant first; "
                   "ranked by meaning, not just keywords:")
    out.append("")
    for k, h in enumerate(res.hits, 1):
        head = f"{k}. {h.path}"
        if h.title:
            head += f" — {h.title}"
        out.append(head)
        where = f"lines of {OUTPUT_DIRNAME}/text/{h.path}.txt" if h.extracted else "file lines"
        detail = f"   ≈{h.tokens:,} tokens"
        if h.ranges:
            detail += f" · most relevant: {', '.join(lines_label(r) for r in h.ranges)} ({where})"
        if h.reason:
            detail += f" · {h.reason}"
        out.append(detail)
        if h.quote:
            out.append(f"   “{h.quote}” ({lines_label(h.quote_lines)})")
    if res.also:
        out += ["", "Possibly relevant, only if those don't answer it: " + ", ".join(res.also)]
    return "\n".join(out) + "\n"


def format_excerpts(res: Result, shown_root: str = "") -> str:
    docs = res.excerpt_docs
    total = sum(est_tokens(len(x.text)) for d in docs for x in d.excerpts)
    extra = sum(not d.ranked for d in docs)
    out = [f"Excerpts from {len(docs)} of {res.total_docs} documents{shown_root}, chosen by meaning "
           f"(≈{total:,} tokens; the {len(docs) - extra} most relevant first"
           + (f", then {extra} lower-ranked for breadth" if extra else "") + ").",
           "Answer from these if they are enough, citing path and lines. "
           "Open a file only for what they don't cover.", ""]
    for k, d in enumerate(docs, 1):
        head = f"## {k}. {d.path}"
        if d.title:
            head += f" — {d.title}"
        out.append(head)
        info = [d.byline] if d.byline else []
        info.append(f"≈{d.tokens:,} tokens in full")
        if d.extracted:
            info.append(f"lines of {OUTPUT_DIRNAME}/text/{d.path}.txt")
        if not d.ranked:
            info.append("lower-ranked")
        out.append(" · ".join(info))
        for x in d.excerpts:
            out += ["", f"[{lines_label(x.lines)}]", x.text]
        out.append("")
    if res.also:
        out.append("Not excerpted, possibly relevant: " + ", ".join(res.also))
    return "\n".join(out).rstrip("\n") + "\n"
