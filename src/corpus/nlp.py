"""Per-document analysis: spaCy NER, PyTextRank key phrases and extractive
summary candidates, lemma counts for topic modeling.

The result is a small JSON-able dict, cached per document, so unchanged
files are never re-analyzed. All locations are 1-based line numbers in the
document's reference text.
"""

from __future__ import annotations

import bisect
import re
from collections import Counter
from importlib import metadata

from .parsers import ParsedDocument
from .util import collapse, entity_key, short_hash

EXCLUDED_ENTITY_TYPES = {"CARDINAL", "ORDINAL", "QUANTITY", "PERCENT", "MONEY", "TIME"}
TOPIC_POS = {"NOUN", "PROPN", "ADJ"}
TERM_POS = TOPIC_POS | {"VERB"}  # keyword component of --find
MAX_KEYWORDS = 25
MAX_SUMMARY_CANDIDATES = 6
MIN_SUMMARY_WORDS = 6
MIN_LEAD_WORDS = 10
LEAD_CANDIDATES = 2
MAX_BOILERPLATE_BLOCK = 200  # chars; longer blocks are prose, never boilerplate
PASSAGE_MIN = 400  # chars; shorter paragraphs are merged with the next one
PASSAGE_TARGET = 900  # long paragraphs are split at sentence ends near this size
PASSAGE_MAX = 1600
MAX_LINES_PER_ITEM = 8
CHUNK_CHARS = 100_000

_HAS_LETTER = re.compile(r"[^\W\d_]", re.U)
_BLANK_LINE = re.compile(r"\n[ \t]*\n")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?)\]])")


def quote(text: str) -> str:
    """Display form of a sentence from the masked analysis text."""
    return _SPACE_BEFORE_PUNCT.sub(r"\1", collapse(text))


class ModelUnavailable(RuntimeError):
    pass


def pkg_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "missing"


def model_versions(model: str) -> dict:
    return {"spacy": pkg_version("spacy"), "model": f"{model}-{pkg_version(model)}",
            "pytextrank": pkg_version("pytextrank")}


def _chunks(text: str, limit: int) -> list[tuple[int, str]]:
    """Split at paragraph (else whitespace) boundaries into <= limit chars."""
    out, start, n = [], 0, len(text)
    while start < n:
        end = min(start + limit, n)
        if end < n:
            cut = text.rfind("\n\n", start, end)
            if cut <= start:
                cut = text.rfind(" ", start, end)
            if cut > start:
                end = cut
        out.append((start, text[start:end]))
        start = end
    return out or [(0, "")]


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _line_indented(text: str, pos: int) -> bool:
    start = text.rfind("\n", 0, pos) + 1
    return start < len(text) and text[start] in " \t"


def _register_block_boundaries() -> str:
    """Sentence starts at blank lines and at masked structural lines
    (headings, list items), so headings never fuse with the next sentence."""
    from spacy.language import Language

    name = "corpus_block_boundaries"
    if name not in Language.factories:

        @Language.component(name)
        def block_boundaries(doc):
            text, prev = doc.text, None
            for tok in doc:
                if tok.is_space:
                    continue
                if prev is not None:
                    gap = text[prev.idx + len(prev.text): tok.idx]
                    if "\n" in gap:
                        line_start = tok.idx - len(gap) + gap.rfind("\n") + 1
                        prev_line = text[text.rfind("\n", 0, prev.idx) + 1: line_start].strip()
                        heading_like = (len(prev_line) < 60 and tok.text[:1].isupper()
                                        and not re.search(r"[.,;:\-–—]$", prev_line))
                        if (_BLANK_LINE.search(gap) or _line_indented(text, line_start)
                                or _line_indented(text, prev.idx) or heading_like):
                            tok.is_sent_start = True
                prev = tok
            return doc

    return name


def make_passages(text: str, sent_spans: list[list[int]]) -> list[tuple[int, int]]:
    """Split text into retrieval passages (character spans).

    Paragraphs (blank-line separated) are the unit. Paragraphs shorter than
    PASSAGE_MIN are merged forward (so headings join their first paragraph);
    paragraphs longer than PASSAGE_MAX are split at sentence ends.
    """
    pieces: list[tuple[int, int]] = []
    pos = 0
    for m in list(_BLANK_LINE.finditer(text)) + [None]:
        end = m.start() if m else len(text)
        s, e = _trim(text, pos, end)
        if e > s:
            if e - s <= PASSAGE_MAX:
                pieces.append((s, e))
            else:
                start = s
                for _, se in (sp for sp in sent_spans if s < sp[1] <= e):
                    if se - start >= PASSAGE_TARGET and e - se >= PASSAGE_MIN:
                        pieces.append(_trim(text, start, se))
                        start = se
                pieces.append(_trim(text, start, e))
        pos = m.end() if m else len(text)

    out: list[tuple[int, int]] = []
    for s, e in pieces:
        if out and out[-1][1] - out[-1][0] < PASSAGE_MIN and e - out[-1][0] <= PASSAGE_MAX:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    if len(out) > 1 and out[-1][1] - out[-1][0] < PASSAGE_MIN // 2 and out[-1][1] - out[-2][0] <= PASSAGE_MAX:
        out[-2:] = [(out[-2][0], out[-1][1])]
    return out


class Analyzer:
    def __init__(self, model: str = "en_core_web_sm"):
        self.model = model
        self._nlp = None

    def _load(self):
        if self._nlp is None:
            import spacy
            import pytextrank  # noqa: F401  (registers the "textrank" factory)

            try:
                nlp = spacy.load(self.model)
            except OSError as exc:
                raise ModelUnavailable(
                    f"spaCy model '{self.model}' is not installed. "
                    f"Install it with: python -m spacy download {self.model}") from exc
            if "parser" in nlp.pipe_names:
                nlp.add_pipe(_register_block_boundaries(), before="parser")
            nlp.add_pipe("textrank", last=True)
            nlp.max_length = max(nlp.max_length, CHUNK_CHARS + 1000)
            self._nlp = nlp
        return self._nlp

    def analyze(self, parsed: ParsedDocument) -> dict:
        nlp = self._load()
        ref, ana = parsed.text, parsed.analysis_text
        starts = [0] + [m.end() for m in re.finditer("\n", ref)]

        def line_of(offset: int) -> int:
            return bisect.bisect_right(starts, offset)

        entities: dict[str, dict] = {}
        keywords: dict[str, dict] = {}
        candidates: list[list[dict]] = []
        lead: list[dict] = []
        lemmas: Counter = Counter()
        terms: Counter = Counter()
        sent_spans: list[list[int]] = []
        n_sent = n_words = 0

        chunks = _chunks(ana, CHUNK_CHARS)
        for (base, _), doc in zip(chunks, nlp.pipe(c for _, c in chunks)):
            for sent in doc.sents:
                n_sent += 1
                ss, se = _trim(ana, base + sent.start_char, base + sent.end_char)
                if se > ss:
                    sent_spans.append([ss, se])
            n_words += sum(1 for t in doc if t.is_alpha)

            for ent in doc.ents:
                if ent.label_ in EXCLUDED_ENTITY_TYPES:
                    continue
                s, e = _trim(ana, base + ent.start_char, base + ent.end_char)
                surface = collapse(ana[s:e])
                if not surface or len(surface) > 100 or not _HAS_LETTER.search(surface) \
                        or _BLANK_LINE.search(ana[s:e]):
                    continue
                rec = entities.setdefault(entity_key(surface),
                                          {"forms": Counter(), "types": Counter(), "lines": []})
                rec["forms"][surface] += 1
                rec["types"][ent.label_] += 1
                ln = line_of(s)
                if ln not in rec["lines"] and len(rec["lines"]) < MAX_LINES_PER_ITEM:
                    rec["lines"].append(ln)

            for phrase in doc._.phrases:
                if not phrase.chunks:
                    continue
                head = phrase.chunks[0]
                # Strip leading determiners/stop words ("the energy consumption").
                i = 0
                while i < len(head) and (head[i].is_stop or head[i].is_punct or head[i].is_space
                                         or head[i].pos_ in ("DET", "PRON")):
                    i += 1
                core = head[i:]
                if not len(core) or all(t.is_punct or t.like_num for t in core):
                    continue
                # A lone common noun ("work", "cities") is too generic to navigate by.
                if len(core) == 1 and core[0].pos_ != "PROPN":
                    continue
                text = collapse(core.text)
                key = text.casefold()
                if len(key) < 3 or not _HAS_LETTER.search(key):
                    continue
                rec = keywords.setdefault(key, {"text": text, "rank": 0.0, "count": 0, "lines": []})
                rec["rank"] = max(rec["rank"], float(phrase.rank))
                spans = sorted({(c.start_char, c.end_char) for c in phrase.chunks})
                rec["count"] += len(spans)
                for s, _ in spans:
                    ln = line_of(base + s)
                    if ln not in rec["lines"] and len(rec["lines"]) < MAX_LINES_PER_ITEM:
                        rec["lines"].append(ln)

            # Lead: the first real sentences (opening statements summarize well).
            for sent in doc.sents:
                if len(lead) >= LEAD_CANDIDATES:
                    break
                words = sum(1 for t in sent if t.is_alpha)
                s, e = _trim(ana, base + sent.start_char, base + sent.end_char)
                text = quote(ana[s:e])
                in_title = bool(parsed.title) and text.rstrip(".!?").casefold() in parsed.title.casefold()
                if words >= MIN_LEAD_WORDS and not in_title and re.search(r"[.!?][\"'”’)]*$", text):
                    lead.append({"text": text, "lines": [line_of(s), line_of(e - 1)]})

            ranked: list[dict] = []
            if len(doc):
                for sent in doc._.textrank.summary(limit_phrases=15, limit_sentences=MAX_SUMMARY_CANDIDATES * 2,
                                                   preserve_order=False):
                    if sum(1 for t in sent if t.is_alpha) < MIN_SUMMARY_WORDS:
                        continue
                    s, e = _trim(ana, base + sent.start_char, base + sent.end_char)
                    if e > s:
                        ranked.append({"text": quote(ana[s:e]), "lines": [line_of(s), line_of(e - 1)]})
            candidates.append(ranked)

            for t in doc:
                if t.is_alpha and not t.is_stop and len(t) > 2 and t.pos_ in TERM_POS:
                    lemma = t.lemma_.lower()
                    terms[lemma] += 1
                    if t.pos_ in TOPIC_POS:
                        lemmas[lemma] += 1

        # Round-robin over chunks so long documents are represented evenly.
        summary, seen = [], set()
        for i in range(max(len(c) for c in candidates)):
            for cand in candidates:
                if i < len(cand) and len(summary) < MAX_SUMMARY_CANDIDATES:
                    key = cand[i]["text"].casefold()
                    if key not in seen:
                        seen.add(key)
                        summary.append(cand[i])

        # Short blocks (paragraphs up to MAX_BOILERPLATE_BLOCK chars) are hashed
        # so corpus-wide analysis can spot text repeated across documents.
        short_blocks, block_start, block = [], None, []
        for i, line in enumerate(ref.split("\n") + [""], 1):
            if line.strip():
                block_start = block_start or i
                block.append(line)
                continue
            if block:
                text = collapse(" ".join(block))
                if len(text) <= MAX_BOILERPLATE_BLOCK:
                    short_blocks.append([block_start, i - 1, short_hash(text.casefold())])
            block_start, block = None, []

        kws = sorted(keywords.values(), key=lambda k: (-k["rank"], k["text"].casefold()))[:MAX_KEYWORDS]
        for k in kws:
            k["rank"] = round(k["rank"], 5)
        ents = []
        for rec in entities.values():
            label = sorted(rec["forms"].items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            etype = sorted(rec["types"].items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            ents.append({"text": label, "type": etype, "count": sum(rec["forms"].values()), "lines": rec["lines"]})
        ents.sort(key=lambda e: (e["lines"][0], e["text"]))

        return {
            "title": parsed.title,
            "meta": parsed.meta,
            "sent_spans": sent_spans,
            "passages": [{"start": a, "end": b, "lines": [line_of(a), line_of(b - 1)]}
                         for a, b in make_passages(ana, sent_spans)],
            "chars": len(ref),
            "lines": len(starts),
            "words": n_words,
            "sentences": n_sent,
            "lead": lead,
            "summary": summary,
            "keywords": kws,
            "entities": ents,
            "lemmas": dict(sorted(lemmas.items())),
            "terms": dict(sorted(terms.items())),
            "short_blocks": short_blocks,
        }
