"""Local sentence embeddings via fastembed (ONNX; no PyTorch, no network
after the one-time model download).

Model files are stored in ``$CORPUS_MODEL_DIR`` if set, else under the
active Python environment (``<sys.prefix>/share/corpus-models``) when it is
writable, else ``~/.cache/corpus/models``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

MODEL = "BAAI/bge-small-en-v1.5"
BATCH = 8  # small batches of length-sorted texts minimize padding on CPU


def model_dir() -> Path:
    env = os.environ.get("CORPUS_MODEL_DIR")
    if env:
        return Path(env)
    prefix = Path(sys.prefix) / "share" / "corpus-models"
    try:
        prefix.mkdir(parents=True, exist_ok=True)
        if os.access(prefix, os.W_OK):
            return prefix
    except OSError:
        pass
    return Path.home() / ".cache" / "corpus" / "models"


def model_id() -> str:
    from importlib import metadata

    try:
        version = metadata.version("fastembed")
    except metadata.PackageNotFoundError:
        version = "missing"
    return f"{MODEL}@fastembed-{version}"


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class Embedder:
    def __init__(self) -> None:
        self._model = None

    def _load(self):
        if self._model is None:
            folder = model_dir()
            folder.mkdir(parents=True, exist_ok=True)
            # Keep the Hugging Face hub's files (locks, metadata) inside the
            # model folder too, and keep it quiet.
            os.environ.setdefault("HF_HOME", str(folder / "hf"))
            os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            os.environ.setdefault("HF_HUB_VERBOSITY", "error")
            import logging
            import warnings

            warnings.filterwarnings("ignore", module="huggingface_hub")
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            from fastembed import TextEmbedding

            self._model = TextEmbedding(MODEL, cache_dir=str(folder))
        return self._model

    def passages(self, texts: list[str]) -> np.ndarray:
        """Embed passages; returns an (n, dim) float32 matrix of unit vectors."""
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        vecs = list(self._load().passage_embed([texts[i] for i in order], batch_size=BATCH))
        out = np.empty((len(texts), len(vecs[0])), dtype=np.float32)
        for pos, i in enumerate(order):
            out[i] = vecs[pos]
        return _normalize(out)

    def query(self, text: str) -> np.ndarray:
        vec = np.asarray(next(iter(self._load().query_embed([text]))), dtype=np.float32)
        return vec / (np.linalg.norm(vec) or 1)
