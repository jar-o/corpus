"""Corpus-wide analysis over cached per-document results.

* Boilerplate: short paragraphs repeated verbatim in >= 80% of documents (letterheads,
  salutations, signatures) are ignored for summaries, entities and phrases.
* Topics: Gensim NMF over TF-IDF weighted noun/adjective lemmas (more
  coherent than LDA on small and mid-sized corpora); every document gets one
  primary topic, which is how the overview groups documents.
* Related documents: Gensim TF-IDF cosine similarity, with the shared terms
  that drive each match.
* Entity / phrase indexes: which documents (and lines) mention what.

Recomputed whenever any document changes; never re-runs spaCy.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .nlp import pkg_version
from .util import entity_key

BOILERPLATE_SHARE = 0.8
TOPIC_SECONDARY_MIN = 0.2
RELATED_TOP_K = 5
RELATED_MIN_SCORE = 0.1


@dataclass
class Doc:
    path: str
    format: str
    extracted: bool
    size: int
    result: dict
    # Filled in by analyze():
    lead: dict | None = None
    boiler: set[int] = field(default_factory=set)  # boilerplate line numbers
    summary: list[dict] = field(default_factory=list)
    keywords: list[dict] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)
    topics: list[tuple[str, float]] = field(default_factory=list)  # (topic id, weight), primary first
    related: list[dict] = field(default_factory=list)

    @property
    def title(self) -> str | None:
        return self.result.get("title")


@dataclass
class Topic:
    id: str
    terms: list[str]
    docs: list[tuple[str, float]]  # primary members (path, weight), strongest first

    @property
    def label(self) -> str:
        return ", ".join(self.terms[:4])


@dataclass
class Corpus:
    docs: list[Doc]
    topics: list[Topic]
    topic_model: dict
    entities: list[dict]  # {label, type, docs: {path: [lines]}}
    phrases: list[dict]  # {text, docs: {path: rank}}
    boilerplate_lines: int


def _drop_boilerplate(items: list[dict], boiler: set[int]) -> list[dict]:
    out = []
    for it in items:
        lines = [ln for ln in it["lines"] if ln not in boiler]
        if lines:
            out.append({**it, "lines": lines})
    return out


def _summary_without_boilerplate(items: list[dict], boiler: set[int]) -> list[dict]:
    return [s for s in items if not all(ln in boiler for ln in range(s["lines"][0], s["lines"][1] + 1))]


def analyze(docs: list[Doc], num_topics: int = 0, seed: int = 42) -> Corpus:
    n = len(docs)

    # ---------------------------------------------------------- boilerplate
    boiler_hashes: set[str] = set()
    if n >= 5:
        df = Counter(h for d in docs for h in {b[2] for b in d.result["short_blocks"]})
        boiler_hashes = {h for h, c in df.items() if c >= BOILERPLATE_SHARE * n}
    total_boiler = 0
    for d in docs:
        boiler = {ln for a, b, h in d.result["short_blocks"] if h in boiler_hashes for ln in range(a, b + 1)}
        total_boiler += len(boiler)
        d.boiler = boiler
        d.lead = next(iter(_summary_without_boilerplate(d.result["lead"], boiler)), None)
        d.summary = _summary_without_boilerplate(d.result["summary"], boiler)
        d.keywords = _drop_boilerplate(d.result["keywords"], boiler)
        d.entities = _drop_boilerplate(d.result["entities"], boiler)

    # ---------------------------------------------------------- entity / phrase indexes
    ents: dict[str, dict] = {}
    for d in docs:
        for e in d.entities:
            rec = ents.setdefault(entity_key(e["text"]), {"forms": Counter(), "types": Counter(), "docs": {}})
            rec["forms"][e["text"]] += e["count"]
            rec["types"][e["type"]] += e["count"]
            rec["docs"][d.path] = e["lines"]
    entities = []
    for rec in ents.values():
        entities.append({
            "label": sorted(rec["forms"].items(), key=lambda kv: (-kv[1], kv[0]))[0][0],
            "type": sorted(rec["types"].items(), key=lambda kv: (-kv[1], kv[0]))[0][0],
            "mentions": sum(rec["forms"].values()),
            "docs": dict(sorted(rec["docs"].items())),
        })
    entities.sort(key=lambda e: (-len(e["docs"]), -e["mentions"], e["label"].casefold()))

    phr: dict[str, dict] = {}
    for d in docs:
        for k in d.keywords:
            rec = phr.setdefault(k["text"].casefold(), {"forms": Counter(), "docs": {}})
            rec["forms"][k["text"]] += 1
            rec["docs"][d.path] = k["rank"]
    phrases = [{"text": sorted(r["forms"].items(), key=lambda kv: (-kv[1], kv[0]))[0][0], "docs": r["docs"]}
               for r in phr.values()]
    phrases.sort(key=lambda p: (-len(p["docs"]), -sum(p["docs"].values()), p["text"].casefold()))

    # ---------------------------------------------------------- topics + similarity
    topics, model_info = _topics_and_similarity(docs, num_topics, seed)
    return Corpus(docs=docs, topics=topics, topic_model=model_info, entities=entities,
                  phrases=phrases, boilerplate_lines=total_boiler)


def _topics_and_similarity(docs: list[Doc], num_topics: int, seed: int) -> tuple[list[Topic], dict]:
    from gensim.corpora import Dictionary
    from gensim.models import Nmf, TfidfModel
    from gensim.similarities import SparseMatrixSimilarity

    n = len(docs)
    info: dict = {"library": f"gensim {pkg_version('gensim')}"}
    if n < 2:
        info["skipped"] = "fewer than 2 documents"
        return [], info

    dictionary = Dictionary([sorted(d.result["lemmas"]) for d in docs])
    if n >= 10:
        # Drop terms in >50% of documents: they describe the corpus, not a topic.
        dictionary.filter_extremes(no_below=2, no_above=0.5, keep_n=50_000)
    t2i = dictionary.token2id
    bows = [sorted((t2i[t], c) for t, c in d.result["lemmas"].items() if t in t2i) for d in docs]

    # Related documents: TF-IDF cosine + the shared terms that drive it.
    tfidf = TfidfModel(dictionary=dictionary)
    vecs = [dict(tfidf[b]) for b in bows]
    index = SparseMatrixSimilarity([list(v.items()) for v in vecs], num_features=len(dictionary))
    for i, d in enumerate(docs):
        if not vecs[i]:
            continue
        sims = index[list(vecs[i].items())]
        ranked = sorted(((float(s), j) for j, s in enumerate(sims) if j != i and s >= RELATED_MIN_SCORE),
                        key=lambda x: (-x[0], docs[x[1]].path))[:RELATED_TOP_K]
        for score, j in ranked:
            shared = sorted(set(vecs[i]) & set(vecs[j]), key=lambda t: -min(vecs[i][t], vecs[j][t]))[:4]
            d.related.append({"path": docs[j].path, "score": round(score, 2),
                              "shared": [dictionary[t] for t in shared]})

    nonempty = [i for i, b in enumerate(bows) if b]
    if len(nonempty) < 3 or len(dictionary) < 10:
        info["skipped"] = "too few documents or terms for topic modeling"
        return [], info

    k = num_topics or max(2, min(20, round(math.sqrt(n / 2))))
    k = min(k, len(nonempty))
    passes = 20 if n < 1000 else 5
    tvecs = {i: list(vecs[i].items()) for i in nonempty}
    model = Nmf([tvecs[i] for i in nonempty], id2word=dictionary, num_topics=k, passes=passes, random_state=seed)
    info.update({"algorithm": "NMF", "topics_requested": k, "passes": passes, "seed": seed,
                 "vocabulary": len(dictionary)})

    members: dict[int, list[tuple[str, float]]] = defaultdict(list)
    doc_topics: dict[int, list[tuple[int, float]]] = {}
    for i in nonempty:
        dist = sorted(model.get_document_topics(tvecs[i], minimum_probability=0.0), key=lambda x: (-x[1], x[0]))
        if not dist:
            continue
        doc_topics[i] = dist
        members[dist[0][0]].append((docs[i].path, float(dist[0][1])))

    # Number topics T1..Tn by size; drop topics that are nobody's primary topic.
    order = sorted(members, key=lambda t: (-len(members[t]), t))
    tid = {t: f"T{rank}" for rank, t in enumerate(order, 1)}
    topics = [Topic(id=tid[t], terms=[w for w, _ in model.show_topic(t, topn=8)],
                    docs=sorted(members[t], key=lambda m: (-m[1], m[0]))) for t in order]
    for i, dist in doc_topics.items():
        docs[i].topics = [(tid[t], round(w, 2)) for rank, (t, w) in enumerate(dist)
                          if t in tid and (rank == 0 or w >= TOPIC_SECONDARY_MIN)]
    info["topics"] = len(topics)
    return topics, info
