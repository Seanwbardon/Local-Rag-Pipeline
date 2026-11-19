"""Turn text into embedding vectors with an Ollama embedding model, on the CPU only.

Used twice in the pipeline: during ingestion (thousands of chunks, sent in batches) and
for every question (one short text). Talks to Ollama's /api/embed endpoint with the Python
standard library, like the proof of concept, and always sends num_gpu=0.

Long-input check: requests are sent with truncate=false, so Ollama refuses any text longer
than the model's context (256 tokens for all-minilm by default) instead of silently cutting
it. When a batch is refused, it is split in half and each half retried, repeatedly, until
the long texts are isolated. Those are embedded with truncation and counted, so the ingest
report says exactly how many chunks were cut. Halving finds one long chunk in a batch of 32
with about 10 requests instead of 32.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when Ollama can't produce embeddings (not running, model missing, other errors)."""


class _InputTooLong(Exception):
    """Internal: Ollama refused a request sent with truncate=false."""


@dataclass
class EmbeddingRun:
    vectors: list[list[float]]
    tokens: int = 0                                          # tokens processed, as counted by Ollama
    truncated: list[int] = field(default_factory=list)      # positions of texts that had to be cut
    requests: int = 0


class OllamaEmbedder:
    def __init__(self, model: str, host: str = "http://localhost:11434", num_gpu: int = 0,
                 batch_size: int = 32, timeout: float = 300, keep_alive: str = "10m"):
        self.model = model
        self.host = host.rstrip("/")
        self.num_gpu = num_gpu
        self.batch_size = batch_size
        self.timeout = timeout
        self.keep_alive = keep_alive

    # ------------------------------------------------------------------ public API

    def embed_query(self, text: str) -> list[float]:
        """Embed one question. Long questions are truncated rather than refused."""
        vectors, _ = self._request([text], truncate=True)
        return vectors[0]

    def embed_documents(self, texts: list[str],
                        progress: Callable[[int, int], None] | None = None) -> EmbeddingRun:
        """Embed many texts in batches of `batch_size`, reporting progress as (done, total)."""
        run = EmbeddingRun(vectors=[])
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            try:
                vectors, tokens = self._request(batch, truncate=False)
                run.requests += 1
            except _InputTooLong:
                vectors, tokens = self._embed_isolating_long_texts(batch, start, run)
            run.vectors.extend(vectors)
            run.tokens += tokens
            if progress:
                progress(min(start + len(batch), len(texts)), len(texts))
        return run

    # ------------------------------------------------------------------ internals

    def _embed_isolating_long_texts(self, batch: list[str], offset: int, run: EmbeddingRun):
        """A batch was refused: split it in half and retry each half until long texts are isolated."""
        if len(batch) == 1:
            try:
                vectors, tokens = self._request(batch, truncate=False)
            except _InputTooLong:
                vectors, tokens = self._request(batch, truncate=True)
                run.truncated.append(offset)
                logger.warning("Chunk %d was longer than the embedding model's context and was truncated", offset)
            run.requests += 1
            return vectors, tokens

        middle = len(batch) // 2
        vectors, tokens = [], 0
        for part, part_offset in ((batch[:middle], offset), (batch[middle:], offset + middle)):
            try:
                vec, tok = self._request(part, truncate=False)
                run.requests += 1
            except _InputTooLong:
                vec, tok = self._embed_isolating_long_texts(part, part_offset, run)
            vectors.extend(vec)
            tokens += tok
        return vectors, tokens

    def _request(self, texts: list[str], truncate: bool) -> tuple[list[list[float]], int]:
        payload = {
            "model": self.model,
            "input": texts,
            "truncate": truncate,
            "keep_alive": self.keep_alive,
            "options": {"num_gpu": self.num_gpu},   # CPU-only, as for generation
        }
        request = urllib.request.Request(
            self.host + "/api/embed",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                body = json.load(resp)
        except urllib.error.HTTPError as exc:
            message = _error_message(exc)
            if exc.code == 404:
                raise EmbeddingError(f"Embedding model '{self.model}' not found. Run: ollama pull {self.model}") from exc
            if exc.code == 400 and not truncate:
                raise _InputTooLong(message) from exc
            raise EmbeddingError(f"Ollama returned HTTP {exc.code}: {message}") from exc
        except urllib.error.URLError as exc:
            raise EmbeddingError(f"Could not reach Ollama at {self.host}. Is it running?") from exc

        vectors = body.get("embeddings")
        if not vectors or len(vectors) != len(texts):
            raise EmbeddingError(f"Ollama returned {len(vectors or [])} embeddings for {len(texts)} texts")
        return vectors, int(body.get("prompt_eval_count") or 0)


def _error_message(exc: urllib.error.HTTPError) -> str:
    raw = exc.read().decode("utf-8", errors="replace")
    try:
        return json.loads(raw).get("error", raw)
    except ValueError:
        return raw