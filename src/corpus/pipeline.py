"""Build orchestration: discover -> detect changes -> analyze changed
documents -> corpus-wide analysis -> render Markdown.

Incremental state lives in ``.corpus-out/.cache/`` (hidden; not meant for
LLMs): ``state.json`` tracks every source file (size, mtime, SHA-256,
versions) and ``docs/<id>.json.gz`` holds each document's cached analysis.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import shutil
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import NLP_VERSION, OUTPUT_DIRNAME, RENDER_VERSION, __version__
import numpy as np

from .aggregate import Doc, analyze
from .embed import Embedder, model_id
from .nlp import Analyzer, ModelUnavailable, model_versions
from .parsers import format_for
from .render import render
from .util import atomic_write, collapse, doc_id, est_tokens, fingerprint, sha256_file

# Agent instruction files live next to the documents but are not documents.
IGNORED_NAMES = {"claude.md", "claude.local.md", "agents.md"}

STATE_VERSION = 1
SAVE_EVERY_SECONDS = 10

Log = Callable[[str], None]


@dataclass
class Report:
    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    renamed: list[tuple[str, str]] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    rebuilt: bool = False
    seconds: float = 0.0
    out_dir: Path | None = None


def output_dir(root: Path) -> Path:
    return root.resolve() / OUTPUT_DIRNAME


class Store:
    def __init__(self, out: Path):
        self.out = out
        self.cache = out / ".cache"

    def load_state(self) -> dict:
        try:
            state = json.loads((self.cache / "state.json").read_text(encoding="utf-8"))
            if state.get("state_version") == STATE_VERSION:
                return state
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        return {"state_version": STATE_VERSION, "files": {}, "build": {}}

    def save_state(self, state: dict) -> None:
        atomic_write(self.cache / "state.json", json.dumps(state, indent=1, sort_keys=True))

    def doc_path(self, did: str) -> Path:
        return self.cache / "docs" / f"{did}.json.gz"

    def write_doc(self, did: str, result: dict) -> None:
        data = json.dumps(result, separators=(",", ":"), sort_keys=True).encode("utf-8")
        atomic_write(self.doc_path(did), gzip.compress(data, mtime=0))

    def read_doc(self, did: str) -> dict | None:
        try:
            return json.loads(gzip.decompress(self.doc_path(did).read_bytes()))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None

    def vec_path(self, did: str) -> Path:
        return self.cache / "vectors" / f"{did}.npy"

    def write_vec(self, did: str, vecs: np.ndarray) -> None:
        buf = io.BytesIO()
        np.save(buf, vecs.astype(np.float16))
        atomic_write(self.vec_path(did), buf.getvalue())

    def read_vec(self, did: str) -> np.ndarray | None:
        try:
            return np.load(self.vec_path(did))
        except (FileNotFoundError, OSError, ValueError):
            return None

    def text_path(self, rel: str) -> Path:
        return self.out / "text" / f"{rel}.txt"

    def cached(self, entry: dict) -> bool:
        return self.doc_path(entry["id"]).exists() and self.vec_path(entry["id"]).exists()

    def remove_doc(self, rel: str, entry: dict) -> None:
        self.doc_path(entry["id"]).unlink(missing_ok=True)
        self.vec_path(entry["id"]).unlink(missing_ok=True)
        self.text_path(rel).unlink(missing_ok=True)


def discover(root: Path) -> list[tuple[str, Path]]:
    """Supported files, sorted; hidden files/dirs (incl. .corpus-out) skipped."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if (not name.startswith(".") and name.lower() not in IGNORED_NAMES
                    and format_for(p) and p.is_file()):
                found.append((p.relative_to(root).as_posix(), p))
    return sorted(found)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build(root: Path, model: str = "en_core_web_sm", num_topics: int = 0, force: bool = False,
          log: Log = lambda m: None) -> Report:
    t0 = time.perf_counter()
    root = root.resolve()
    out = output_dir(root)
    store = Store(out)
    state = store.load_state()
    files_state: dict[str, dict] = state["files"]
    report = Report(out_dir=out)

    versions = model_versions(model)
    if versions["model"].endswith("-missing"):
        raise ModelUnavailable(f"spaCy model '{model}' is not installed. "
                               f"Install it with: python -m spacy download {model}")
    versions["embedding"] = model_id()
    doc_fp = fingerprint({"nlp": NLP_VERSION, "versions": versions})
    corpus_fp = fingerprint({"render": RENDER_VERSION, "topics": num_topics, "doc_fp": doc_fp})

    # -------------------------------------------------- change detection
    current = discover(root)
    seen = set()
    todo: list[tuple[str, Path, str, os.stat_result, str]] = []
    for rel, path in current:
        seen.add(rel)
        fmt = format_for(path)
        st = path.stat()
        prev = files_state.get(rel)
        valid = (prev is not None and prev["parser_version"] == fmt.version and prev["doc_fp"] == doc_fp
                 and (prev["status"] != "ok" or (store.cached(prev)
                                                 and (not fmt.extracted or store.text_path(rel).exists()))))
        if not force and valid and prev["size"] == st.st_size and prev["mtime_ns"] == st.st_mtime_ns:
            report.unchanged.append(rel)
            continue
        sha = sha256_file(path)
        if not force and valid and prev["sha256"] == sha:
            prev["size"], prev["mtime_ns"] = st.st_size, st.st_mtime_ns
            report.unchanged.append(rel)
            continue
        todo.append((rel, path, sha, st, "modified" if prev else "added"))

    gone = {rel: e for rel, e in files_state.items() if rel not in seen}

    # Renames: a new file with the same content as a vanished one reuses its analysis.
    if not force:
        by_sha: dict[str, list[str]] = {}
        for rel, e in gone.items():
            if e["status"] == "ok" and e["doc_fp"] == doc_fp and store.cached(e):
                by_sha.setdefault(e["sha256"], []).append(rel)
        remaining = []
        for item in todo:
            rel, path, sha, st, kind = item
            old = by_sha.get(sha, []).pop(0) if kind == "added" and by_sha.get(sha) else None
            if old is None or Path(old).suffix.lower() != path.suffix.lower():
                remaining.append(item)
                continue
            entry = gone.pop(old)
            new_id = doc_id(rel)
            os.replace(store.doc_path(entry["id"]), store.doc_path(new_id))
            os.replace(store.vec_path(entry["id"]), store.vec_path(new_id))
            if format_for(path).extracted and store.text_path(old).exists():
                store.text_path(rel).parent.mkdir(parents=True, exist_ok=True)
                os.replace(store.text_path(old), store.text_path(rel))
            files_state.pop(old)
            files_state[rel] = {**entry, "id": new_id, "size": st.st_size, "mtime_ns": st.st_mtime_ns}
            report.renamed.append((old, rel))
            log(f"renamed  {old} -> {rel}")
        todo = remaining

    for rel, entry in sorted(gone.items()):
        store.remove_doc(rel, entry)
        files_state.pop(rel)
        report.deleted.append(rel)
        log(f"deleted  {rel}")

    build_state = state["build"]
    needs_corpus = (force or bool(todo or report.deleted or report.renamed)
                    or build_state.get("corpus_fp") != corpus_fp or not build_state.get("complete")
                    or not (out / "CORPUS.md").exists() or not (store.cache / "index.npz").exists())
    if needs_corpus:
        build_state["complete"] = False
        store.save_state(state)

    # -------------------------------------------------- per-document analysis
    analyzer = Analyzer(model)
    embedder = Embedder()
    last_save = time.monotonic()
    for rel, path, sha, st, kind in todo:
        fmt = format_for(path)
        entry = {"id": doc_id(rel), "format": fmt.name, "parser_version": fmt.version, "sha256": sha,
                 "size": st.st_size, "mtime_ns": st.st_mtime_ns, "doc_fp": doc_fp, "indexed_at": _now()}
        (report.added if kind == "added" else report.modified).append(rel)
        try:
            parsed = fmt.parse(path)
            result = analyzer.analyze(parsed)
            store.write_vec(entry["id"], embedder.passages(embedding_texts(parsed.analysis_text, result)))
            if fmt.extracted:
                atomic_write(store.text_path(rel), parsed.text)
            store.write_doc(entry["id"], result)
            entry["status"] = "ok"
            log(f"{kind:<8} {rel}")
        except ModelUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - record, report, continue
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            store.remove_doc(rel, entry)
            log(f"failed   {rel}: {entry['error']}")
            if os.environ.get("CORPUS_DEBUG"):
                traceback.print_exc()
        files_state[rel] = entry
        if time.monotonic() - last_save > SAVE_EVERY_SECONDS:
            store.save_state(state)
            last_save = time.monotonic()

    report.failed = sorted((rel, e["error"]) for rel, e in files_state.items() if e["status"] != "ok")

    # -------------------------------------------------- corpus-wide analysis + render
    if needs_corpus:
        report.rebuilt = True
        docs = []
        for rel, e in sorted(files_state.items()):
            if e["status"] == "ok":
                result = store.read_doc(e["id"])
                if result is not None:
                    docs.append(Doc(path=rel, format=e["format"], extracted=format_for(Path(rel)).extracted,
                                    size=e["size"], result=result))
        corpus = analyze(docs, num_topics=num_topics)
        files = render(corpus, root.name, report.failed, versions)
        write_index(store, corpus, files_state)

        _sweep(out / "docs", {out / p for p in files if p.startswith("docs/")})
        _sweep(out / "text", {store.text_path(d.path) for d in docs if d.extracted})
        shutil.rmtree(out / "topics", ignore_errors=True)
        for rel_out, content in sorted(files.items()):
            if rel_out != "CORPUS.md":
                atomic_write(out / rel_out, content)
        atomic_write(out / "CORPUS.md", files["CORPUS.md"])  # last: marks a complete build

        page_chars = [len(c) for p, c in files.items() if p.startswith("docs/")]
        build_state.update({
            "complete": True, "corpus_fp": corpus_fp, "built_at": _now(), "generator": __version__,
            "models": versions, "documents": len(docs), "failed": len(report.failed),
            "topics": [{"id": t.id, "label": t.label, "documents": len(t.docs)} for t in corpus.topics],
            "source_tokens": est_tokens(sum(d.result["chars"] for d in docs)),
            "map_tokens": est_tokens(len(files["CORPUS.md"])),
            "doc_page_tokens_avg": est_tokens(sum(page_chars) / len(page_chars)) if page_chars else 0,
            "entities_tokens": est_tokens(len(files["entities.md"])),
        })
        log(f"wrote    {out / 'CORPUS.md'}")
    store.save_state(state)
    report.seconds = time.perf_counter() - t0
    return report


def embedding_texts(analysis_text: str, result: dict) -> list[str]:
    """One text per passage (prefixed with the document title for context),
    plus a final document-level text: title, opening and key sentences."""
    title = result.get("title") or ""
    prefix = f"{title}\n" if title else ""
    texts = [prefix + collapse(analysis_text[p["start"]:p["end"]]) for p in result["passages"]]
    gist = [s["text"] for s in result["lead"][:1] + result["summary"][:3]]
    texts.append(" ".join([title] + gist).strip() or prefix)
    return texts


def write_index(store: Store, corpus, files_state: dict) -> None:
    """Assemble the search index used by ``corpus --find``.

    index.npz: passage vectors (float16) + passage -> document / line range,
    and one document-level vector per document. index.json: document
    metadata and the name index used to recognize names in questions.
    """
    docs_meta, pvecs, pdoc, plines, pspan, dvecs = [], [], [], [], [], []
    for i, d in enumerate(corpus.docs):
        entry = files_state[d.path]
        vecs = store.read_vec(entry["id"])
        passages = d.result["passages"]
        if vecs is None or len(vecs) != len(passages) + 1:
            raise RuntimeError(f"embedding cache for {d.path} is inconsistent; rerun with --force")
        for j, p in enumerate(passages):
            if all(ln in d.boiler for ln in range(p["lines"][0], p["lines"][1] + 1)):
                continue  # passage is pure boilerplate
            pvecs.append(vecs[j])
            pdoc.append(i)
            plines.append(p["lines"])
            pspan.append([p["start"], p["end"]])
        dvecs.append(vecs[-1])
        docs_meta.append({"path": d.path, "id": entry["id"], "title": d.title, "format": d.format,
                          "extracted": d.extracted, "sha256": entry["sha256"], "chars": d.result["chars"],
                          "lines": d.result["lines"]})
    dim = len(dvecs[0]) if dvecs else 384
    # Term counts per document (noun/verb/adjective/proper-noun lemmas) for the
    # small keyword component of --find, stored as a sparse CSR matrix.
    vocab = sorted({t for d in corpus.docs for t in d.result["terms"]})
    tid = {t: k for k, t in enumerate(vocab)}
    indptr, indices, counts = [0], [], []
    for d in corpus.docs:
        items = sorted((tid[t], c) for t, c in d.result["terms"].items())
        indices += [k for k, _ in items]
        counts += [c for _, c in items]
        indptr.append(len(indices))
    buf = io.BytesIO()
    np.savez(buf,
             term_indptr=np.array(indptr, dtype=np.int64),
             term_indices=np.array(indices, dtype=np.int32),
             term_counts=np.array(counts, dtype=np.int32),
             passage_vectors=np.array(pvecs, dtype=np.float16).reshape(-1, dim),
             passage_doc=np.array(pdoc, dtype=np.int32),
             passage_lines=np.array(plines, dtype=np.int32).reshape(-1, 2),
             passage_span=np.array(pspan, dtype=np.int64).reshape(-1, 2),
             doc_vectors=np.array(dvecs, dtype=np.float16).reshape(-1, dim))
    atomic_write(store.cache / "index.npz", buf.getvalue())
    paths = {d.path: i for i, d in enumerate(corpus.docs)}
    names = [{"label": e["label"], "type": e["type"], "docs": [paths[p] for p in e["docs"]]}
             for e in corpus.entities if e["type"] != "DATE"]
    atomic_write(store.cache / "index.json",
                 json.dumps({"model": model_id(), "docs": docs_meta, "names": names, "vocab": vocab},
                            separators=(",", ":")))


def _sweep(folder: Path, keep: set[Path]) -> None:
    """Delete files under ``folder`` not in ``keep``; remove empty dirs."""
    if not folder.is_dir():
        return
    for p in sorted(folder.rglob("*"), reverse=True):
        if p.is_file() and p not in keep:
            p.unlink()
        elif p.is_dir() and not any(p.iterdir()):
            p.rmdir()


def read_state(root: Path) -> dict | None:
    out = output_dir(root)
    if not (out / ".cache" / "state.json").exists():
        return None
    return Store(out).load_state()


def clean(root: Path) -> bool:
    out = output_dir(root)
    if out.name != OUTPUT_DIRNAME or not out.is_dir():
        return False
    shutil.rmtree(out)
    return True
