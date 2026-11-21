"""corpus: find the few documents that answer a question, so LLMs read less."""

__version__ = "0.4.0"

OUTPUT_DIRNAME = ".corpus-out"

# Bump to invalidate every cached per-document analysis.
NLP_VERSION = "8"
# Bump when corpus-wide analysis or Markdown rendering changes.
RENDER_VERSION = "2"
