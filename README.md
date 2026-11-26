![corpus](assets/logo.png)

Front run your LLM with local NLP.

Without help, an agent such as Claude Code answers a question about a folder
by listing it, searching it and opening files until it finds the answer.
Each of those steps re-sends the whole conversation, so steps cost more than
the text read. `corpus --find` prints the passages that answer the question,
so the agent can usually answer straight from its output, in one step:

```
> corpus --find "how do Hamilton and Madison differ on keeping an army during peaceful times?"

Excerpts from 3 of 86 documents, chosen by meaning (≈953 tokens; the 1 most
relevant first, then 2 lower-ranked for breadth).

Answer from these if they are enough, citing path and lines. Open a file only
for what they don't cover.

[L73-78]
Where they differ
Emphasis: need versus danger
Hamilton concentrates on concrete needs in peacetime. He names garrisons on the
Western frontier against Indian raids, Britain and Spain building up forces
nearby, and dockyards and arsenals that need defending ( ). He openly accepts
that a small professional force amounts to "a standing army in time of peace; a
small one, indeed, but not the less real for being small." Madison treats a
standing army as a hazard to be kept as small as possible: "A standing force,
therefore, is a dangerous, at the same time that it may be a necessary,
provision… On an extensive scale its consequences may be fatal" ( ). He cites
Rome, whose liberties were "the final victim to her military triumphs." His aim
is "diminishing both the necessity and the danger" of relying on one.

[L80-83]
How much army America actually needs
... snip ...
```

The output is about 4,000 tokens by default (`--budget`), several times less
than reading the same documents whole. Each excerpt is verbatim and labeled
with its file and line numbers, so answers can cite them. `--brief` prints
only the ranked documents, line ranges and one quoted sentence each (about
400 tokens), for an agent that will open the files itself.

Everything runs locally:
- spaCy for sentences, names and lemmas.
- A small ONNX embedding model through fastembed (`BAAI/bge-small-en-v1.5`, about 64 MB) for meaning.
- Gensim for topics.

No generative model is used. After the one-time model download, nothing
leaves the machine.

Read the [Federalist Papers Case Study](./tests/integration/federalist/README.md) for more details. Here's the *tl;dr*:

| claude | claude+corpus | savings
|--------|---------------|--------
| $0.3089 | $0.1684 | 45.5%
|157,619 tokens | 52,302 tokens | 66.8%

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
```

The embedding model downloads on first use. Where it's stored:
1. `$CORPUS_MODEL_DIR`, if set.
2. Otherwise `<venv>/share/corpus-models`, inside the virtual environment.
3. If the virtual environment isn't writable, `~/.cache/corpus/models`.

## Usage

```bash
corpus ./documents                   # build, or incrementally update, ./documents/.corpus-out/
corpus ./documents --claude-md       # tell Claude Code to use --find (writes ./documents/CLAUDE.md)

cd ./documents
corpus --find "your question"        # the passages that answer it (~4k tokens)
corpus --find "…" --budget 2000      # smaller excerpt budget
corpus --find "…" --brief            # ranked documents and line ranges only
corpus --find "…" --max 3            # at most 3 top-ranked documents

corpus ./documents --incremental     # same as the default build, stated explicitly
corpus ./documents --force           # re-analyze every document
corpus ./documents --stats           # what is indexed
corpus ./documents --clean           # delete .corpus-out/
```

`--find` brings the index up to date first, analyzing only the files that
changed. It searches upward from the current folder for the index, so it
works from any subfolder.

Supported formats are `.md` / `.markdown`, `.txt`, `.docx` and `.pdf`.
Every supported file under the folder is indexed, including notes such as a
`README.md`, so keep files you don't want searched (for example, saved
answers) outside it. Hidden files and `CLAUDE.md` / `CLAUDE.local.md` /
`AGENTS.md` are not indexed.

### Using it from Claude Code

`corpus ./documents --claude-md` adds a marked section to
`./documents/CLAUDE.md`. If the file already exists, its other content is
kept. The section tells Claude Code three things:
- Always run `corpus --find "<question>"` first, even if it thinks it knows the answer.
- Answer from the excerpts when they are enough; open a file only for what they don't cover.
- Don't grep, glob or list the folder.

The command it writes uses absolute paths to the `corpus` executable and the
folder, so run `--claude-md` again if you move either. In `claude -p` runs,
Claude can't ask for permission, so allow that exact command:

```bash
claude -p "your question" \
  --allowedTools "Bash(/abs/path/to/.venv/bin/corpus:*)"
```

A rule for the bare `Bash(corpus:*)` won't match the absolute path.

## How `--find` chooses documents

1. **Passages.** Each document is split into passages of about 400–1,600
   characters along paragraph boundaries. Each passage is embedded together
   with the document's title. So is one summary text per document, made of
   the title, opening sentence and key sentences.
2. **Score per document.** The question is embedded with the same model,
   and each document's score combines:
   * its best passage match (one strongly matching passage),
   * the average of its three best passages (sustained discussion),
   * its document-level match,
   * how much of the document is close to the best match in the whole corpus.

   A small keyword component (BM25 over noun, verb and adjective lemmas, at
   most 0.08 of the score) breaks ties in favor of documents that use the
   question's own words. Meaning does most of the work: "judiciary" finds
   passages about *courts*, *judges* and *tribunals* that never use the word.
3. **Fewest documents.** A document is kept only if its score is within 30%
   of the gap between the best and the median document. At most 8 are kept.
   Up to four runners-up are listed by path only.
4. **Evidence** (the `--brief` output). For each kept document:
   * the line ranges of its best passages,
   * the one sentence closest to the question, verbatim.
5. **Direct references.** "No. 51", "paper 10" or a file name in the
   question always selects that document.
6. **No match.** If nothing is clearly about the question, it says so and
   lists only the 3 closest documents, in the `--brief` form, without excerpts.
7. **Excerpts.** Passages are then chosen across the whole corpus within the
   token budget, breadth first:
   * the best passage of each selected document;
   * for each name in the question (an author, a person, a place), passages
     from up to two documents that mention it, ranked by the question with
     the names removed. So "how do Hamilton and Madison differ" gets Madison's
     papers, not only the Hamilton papers that dominate the ranking;
   * then the best remaining passages anywhere, at most three per document,
     down to a similarity floor. Lower-ranked documents contribute here.

   The best passage of each selected document is always included, even if
   that goes over the budget.

   Excerpts are verbatim (whitespace collapsed), labeled with line numbers,
   and grouped by document under the document's byline (date, author).

### Measured on the Federalist Papers

The benchmark is 30 questions with known answer papers, in
`tests/integration/federalist_questions.json`. It runs as a test.

| | Documents returned | Expected papers found | Returned papers that are correct | Right paper first |
|---|---|---|---|---|
| `corpus --find` ranked documents | 2.6 per question (of 85) | 80% | 72% | 90% |
| with excerpts (default output) | 9.2 per question, ≈4,000 tokens | 93% | | |

During tuning, a pure keyword ranking (BM25 over words) put an expected
paper first far less often: its mean reciprocal rank was 0.67, against 0.88
for the semantic ranking.

Many questions are deliberately worded without the paper's key term, as
real questions often are.

### Measured with Claude Code

One question, asked with `claude -p` (Claude Opus 5.5) in the Federalist
folder: *"How do Hamilton and Madison differ on keeping an army during
peaceful times?"*

| | Turns | Tokens | Cost |
|---|---:|---:|---:|
| Plain (Claude greps and reads the files) | 6 | 157,619 | $0.3089 |
| With `corpus --find` (answered from the excerpts) | 2 | 52,302 | $0.1684 |
| **Saving** | | **66.8%** | **45.5%** |

Both answers covered both authors' papers and cited lines. The saving comes
mostly from fewer turns, because each turn re-sends the conversation. Most
of what remains is Claude Code's own fixed overhead (system prompt and tool
definitions). This is one run of each; results vary from run to run.

The Federalist Papers are a hard case for `corpus`. Claude already knows
them, so without help it guesses the right search terms ("standing army")
and the right papers. Documents Claude has never seen should show a larger
difference.

## Output

```
.corpus-out/
├── CORPUS.md               human/LLM overview: documents grouped by topic
├── docs/<path>.md          one page per document: key sentences, phrases, names, related docs
├── entities.md             name → documents + line numbers
├── text/<path>.txt         extracted text of PDF/DOCX files (line numbers refer to it)
└── .cache/                 incremental state, per-document analysis and vectors, search index
```

For Markdown and text files, line numbers refer to the source file itself,
so that text is never copied. For PDF and DOCX files they refer to the
extracted text in `text/`.

For the 85 Federalist Papers (1.1 MB of text), `.corpus-out/` is about 3.7 MB.
Most of that is the embedding vectors (float16), which are stored twice: per document for incremental updates, and assembled into the search index.

## Incremental updates

* Each file is tracked by size, modification time and SHA-256.
* Only new or changed files are analyzed and embedded.
* Renamed files reuse their analysis and vectors.
* Deleted files disappear from the index and the Markdown.
* Changing the spaCy model, the embedding model or the analysis code
  re-analyzes affected files.
* Writes are atomic, and an interrupted build resumes where it stopped.

On the Federalist corpus the first build takes about 75 s: about 25 s of
spaCy and 45 s of embedding. A `--find` with no changed files takes about
1–2 s.

## Development

```bash
pytest                 # everything, including the two Federalist tests (~3 min)
pytest -m "not slow"   # skip them
```

| module         | role                                                                     |
| -------------- | ------------------------------------------------------------------------ |
| `parsers.py`   | file → reference text and markup-masked analysis text; titles             |
| `nlp.py`       | per-document spaCy/PyTextRank analysis and passages, cached               |
| `embed.py`     | local embeddings (fastembed)                                             |
| `find.py`      | `--find`: scoring, document selection, excerpts, evidence               |
| `aggregate.py` | boilerplate, topics (NMF), related documents, name/phrase indexes         |
| `render.py`    | Markdown output                                                          |
| `pipeline.py`  | discovery, change detection, cache, search index, orchestration          |
| `cli.py`       | Typer CLI                                                                |

Test corpus: `tests/integration/federalist/` holds the 85 Federalist Papers
from Project Gutenberg eBook #1404 (public domain), one file per paper.
