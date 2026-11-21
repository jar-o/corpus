# How `corpus` works

`corpus` works in two phases. **Building** (`corpus <folder>`) analyzes each
document once and saves the results in `.corpus-out/`. **Finding**
(`corpus --find "…"`) embeds the question and matches it against those saved
results, so it takes 1–2 seconds instead of re-reading anything.

```
 BUILD (slow, incremental)                         FIND (fast, per question)
 ─────────────────────────                         ─────────────────────────
 files ─▶ parsers ─▶ nlp ─▶ embed ─┐               question
                                    ├─▶ .corpus-out/    │
          aggregate ─▶ render ──────┘   .cache/index    ▼
                                         ▲         embed question ─▶ score passages
                                         └──────── ─▶ rank documents ─▶ pick excerpts
```

`pipeline.py` runs the build and `cli.py` is the command-line interface. The
other modules, all in `src/corpus/`, do the actual work.

## Build phase

### 1. `parsers.py`: get clean text out of each file

- Reads `.txt`, `.md`, `.docx` (paragraphs and table rows) and `.pdf` (page by
  page, with `[page N]` markers).
- Produces **two versions of the same length** of every document:
  - the **reference text**, which line numbers and quotes point at;
  - the **analysis text**, where Markdown syntax and page markers are replaced
    with spaces so the NLP doesn't see `##` or `[link](url)`.

  Because both versions are the same length, a character position found in one
  is valid in the other. That's how excerpts get exact line numbers.
- Works out a title, for example "FEDERALIST No. 26 — The Idea of
  Restraining…", from the first short paragraph plus a subtitle.
- Text from PDF and DOCX files is saved to `.corpus-out/text/`. Text files
  aren't copied; line numbers point at the original file.

### 2. `nlp.py`: language analysis of each document (spaCy + PyTextRank)

spaCy's small English model (`en_core_web_sm`) processes each document:

- **Sentence splitting**, plus a custom rule so headings and short lines don't
  merge into the next sentence.
- **Named-entity recognition** finds people, places, organizations and dates,
  grouped by name and recorded with their line numbers.
- **Lemmatization** reduces words to their dictionary form, so "armies" becomes
  "army". It keeps nouns, adjectives and proper nouns for topics, and adds
  verbs for keyword search.
- **PyTextRank** builds a graph of words and phrases and ranks them by how
  central they are. That gives the document's **key phrases** and **summary
  sentences**.
- It also picks the **lead**: the first proper sentences after the title.
- It splits the document into **passages** of about 400–1,600 characters along
  paragraph boundaries. These are the units that search later matches and
  quotes.

The result is cached per document as a compressed JSON file.

### 3. `embed.py`: meaning as numbers (fastembed)

- Runs a small local model, `BAAI/bge-small-en-v1.5` (about 64 MB, ONNX),
  through fastembed. No server or GPU is needed.
- Turns each passage, prefixed with the document's title, into a **384-number
  vector**. Texts with similar meaning get vectors pointing in similar
  directions, so "keeping an army in peacetime" lands near "standing armies in
  time of peace" even though the words differ.
- Also makes one vector per document from its title, lead and summary.
- Vectors are stored as float16 per document, for incremental rebuilds.

### 4. `aggregate.py`: analysis across the whole corpus (Gensim)

Runs over all documents together after each change:

- **Boilerplate detection** marks short paragraphs that appear in at least 80%
  of documents, such as "To the People of the State of New York:". They're
  ignored for summaries and search.
- **Topics** are found with Gensim TF-IDF plus **NMF** (non-negative matrix
  factorization). NMF finds groups of words that tend to occur together, such
  as *army, militia, peace, establishment*, and assigns each document to its
  strongest group. Words found in more than half the documents are dropped
  first.
- **Related documents** are found by TF-IDF similarity between documents.
- It also builds the indexes of names and phrases used across documents.

### 5. `render.py` and `pipeline.py`: write the outputs

- `render.py` writes the human-readable Markdown: `CORPUS.md` with documents
  grouped by topic, one page per document, and `entities.md`.
- `pipeline.py` handles orchestration and incremental updates:
  - It detects changed files by size, modification time and SHA-256, so only
    new or changed files go through spaCy and the embedder. Renamed files keep
    their results.
  - It assembles the **search index** (`.cache/index.npz`): all passage
    vectors, the line range of each passage, document vectors, and
    per-document lemma counts for keyword search.
  - All writes are atomic, so an interrupted build can resume.

## Find phase: `find.py`

1. **Embed the question** with the same model.
2. **Score every passage** by cosine similarity, a single matrix multiplication
   over all passages.
3. **Score every document** with a weighted mix:
   - 25%: its best passage (it has one strongly matching passage);
   - 40%: the average of its 3 best passages (it discusses the topic at
     length);
   - 25%: its document-level vector (the document as a whole is about this);
   - 10%: coverage, the share of its passages that are close to the best match
     anywhere in the corpus.

   On top of that, up to 8% comes from **BM25** keyword matching over lemmas,
   which breaks ties in favor of documents that use the question's own words.
4. **Keep the fewest documents:** only those whose score is close to the top
   one, measured relative to the gap between the top and median scores, up to a
   maximum of 8. If nothing scores well, the output says that nothing is
   clearly about the question.
5. **Pick excerpts** within the token budget (about 4,000 tokens by default):
   - the best passage of each kept document;
   - for each **name in the question**, such as "Madison", passages from
     documents that mention it. These are ranked with the names removed from
     the question, so the topic decides which of that person's documents are
     used. This is what pulled in Madison's Nos. 41 and 46 when Hamilton's
     papers dominated the main ranking.
   - then the best remaining passages anywhere, at most 3 per document.
     Lower-ranked documents contribute here.
6. **Print the output:** passages grouped by document, each under a line giving
   the date and author, labeled with line numbers so the answer can cite them.

## Summary

| Library | Used for |
|---|---|
| spaCy | sentences, names, lemmas |
| PyTextRank | key phrases, summary sentences |
| fastembed (bge-small) | meaning vectors, the core of search |
| Gensim | TF-IDF, NMF topics, related documents |
| BM25 (`find.py`'s own code) | a small keyword component for ranking |
| NumPy | storing vectors and matching them quickly |

No generative model is involved. Every step is deterministic and runs locally.
`corpus` only chooses which of the documents' own words to show the LLM, and
the LLM does all of the reasoning.
