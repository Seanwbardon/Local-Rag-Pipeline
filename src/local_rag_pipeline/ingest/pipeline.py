"""Run the full ingestion: load documents, split them into chunks, embed and store them.

Kept separate from the command-line script so it can be tested and reused by the benchmark.
Returns an IngestSummary with every figure needed for the results log and the slides.
"""

from __future__ import annotations

import platform
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

from local_rag_pipeline.config import Config
from local_rag_pipeline.embeddings import OllamaEmbedder
from local_rag_pipeline.ingest.loader import load_documents
from local_rag_pipeline.ingest.splitter import chunk_stats, split_pages
from local_rag_pipeline.vector_store import VectorStore

TRUNCATION_WARNING_SHARE = 0.02  # warn if more than 2% of chunks had to be cut


@dataclass
class IngestSummary:
    machine: str
    config_file: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    files: int = 0
    pages: int = 0
    empty_pages: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)
    chunks: int = 0
    mean_chunk_chars: int = 0
    max_chunk_chars: int = 0
    embedding_tokens: int = 0
    truncated_chunks: int = 0
    truncated_ids: list[str] = field(default_factory=list)
    stored_total: int = 0
    load_seconds: float = 0.0
    split_seconds: float = 0.0
    embed_seconds: float = 0.0
    store_seconds: float = 0.0
    total_seconds: float = 0.0

    @property
    def truncated_share(self) -> float:
        return self.truncated_chunks / self.chunks if self.chunks else 0.0

    @property
    def chunks_per_second(self) -> float:
        return self.chunks / self.embed_seconds if self.embed_seconds else 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["truncated_share"] = round(self.truncated_share, 4)
        data["chunks_per_second"] = round(self.chunks_per_second, 2)
        return data


def run_ingest(cfg: Config, reset: bool = False,
               progress: Callable[[int, int], None] | None = None,
               embedder: OllamaEmbedder | None = None,
               store: VectorStore | None = None) -> IngestSummary:
    """Ingest everything in cfg.ingest.source_dir into the vector store.

    `embedder` and `store` can be passed in (for tests); otherwise they are built from the config.
    """
    ing = cfg.ingest
    summary = IngestSummary(
        machine=platform.node(),
        config_file=str(cfg.source_file),
        embedding_model=ing.embedding_model,
        chunk_size=ing.chunk_size,
        chunk_overlap=ing.chunk_overlap,
    )
    started = time.perf_counter()

    t = time.perf_counter()
    pages, report = load_documents(ing.source_dir)
    summary.load_seconds = time.perf_counter() - t
    summary.files, summary.pages = report.files_loaded, report.pages_loaded
    summary.empty_pages, summary.skipped = report.empty_pages, report.skipped
    if not pages:
        raise FileNotFoundError(f"No readable documents in {ing.source_dir}. Put the PDF(s) there first.")

    t = time.perf_counter()
    chunks = split_pages(pages, ing.chunk_size, ing.chunk_overlap)
    summary.split_seconds = time.perf_counter() - t
    stats = chunk_stats(chunks)
    summary.chunks, summary.mean_chunk_chars, summary.max_chunk_chars = (
        stats["chunks"], stats["mean_chars"], stats["max_chars"])

    embedder = embedder or OllamaEmbedder(ing.embedding_model, host=cfg.ollama.host,
                                          num_gpu=cfg.ollama.num_gpu, batch_size=ing.batch_size)
    store = store or VectorStore(cfg.vector_store.persist_dir, cfg.vector_store.collection, ing.embedding_model)
    if reset:
        store.reset()

    embedder.embed_query("warm up")  # load the model first so its load time isn't counted as embedding
    t = time.perf_counter()
    run = embedder.embed_documents([c.text for c in chunks], progress=progress)
    summary.embed_seconds = time.perf_counter() - t
    summary.embedding_tokens = run.tokens
    summary.truncated_chunks = len(run.truncated)
    summary.truncated_ids = [chunks[i].id for i in run.truncated]

    t = time.perf_counter()
    store.add(chunks, run.vectors)
    summary.store_seconds = time.perf_counter() - t
    summary.stored_total = store.count()

    summary.total_seconds = time.perf_counter() - started
    return summary