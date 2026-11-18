"""Local vector store: chunks, their embeddings and metadata, kept in ChromaDB on disk.

Design choices:
- Embeddings are always computed by our own OllamaEmbedder and passed in. The collection
  is created with no embedding function, so ChromaDB never downloads or runs a model of its
  own, and everything stays offline.
- ChromaDB's anonymous telemetry is switched off, so it makes no network calls either.
- Chunks are written with `upsert` using their stable IDs, so re-running ingestion
  overwrites existing chunks instead of duplicating them.
- The collection records which embedding model built it. Opening it with a different model
  is refused, because vectors from two models can't be compared.
- Cosine distance, the standard measure for sentence embeddings like MiniLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings

from local_rag_pipeline.ingest.splitter import Chunk


class VectorStoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class Hit:
    id: str
    text: str
    source: str
    page: int
    distance: float   # cosine distance: 0 = identical meaning, larger = less similar


class VectorStore:
    def __init__(self, persist_dir: Path, collection: str, embedding_model: str):
        self.persist_dir = Path(persist_dir)
        self.name = collection
        self.embedding_model = embedding_model
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = None

    # ------------------------------------------------------------------ collection

    @property
    def collection(self):
        if self._collection is None:
            col = self.client.get_or_create_collection(
                name=self.name,
                embedding_function=None,
                metadata={"hnsw:space": "cosine", "embedding_model": self.embedding_model},
            )
            built_with = (col.metadata or {}).get("embedding_model")
            if built_with and built_with != self.embedding_model:
                raise VectorStoreError(
                    f"Collection '{self.name}' was built with '{built_with}', not '{self.embedding_model}'. "
                    f"Re-run ingestion with --reset."
                )
            self._collection = col
        return self._collection

    def reset(self) -> None:
        """Delete the collection and everything in it."""
        existing = [c if isinstance(c, str) else c.name for c in self.client.list_collections()]
        if self.name in existing:
            self.client.delete_collection(self.name)
        self._collection = None

    def count(self) -> int:
        return self.collection.count()

    # ------------------------------------------------------------------ writing and searching

    def add(self, chunks: list[Chunk], vectors: list[list[float]], batch_size: int = 500) -> None:
        if len(chunks) != len(vectors):
            raise VectorStoreError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        for start in range(0, len(chunks), batch_size):
            part = chunks[start:start + batch_size]
            self.collection.upsert(
                ids=[c.id for c in part],
                embeddings=vectors[start:start + batch_size],
                documents=[c.text for c in part],
                metadatas=[c.metadata for c in part],
            )

    def search(self, query_vector: list[float], top_k: int = 4) -> list[Hit]:
        """Return the `top_k` chunks closest in meaning to the query, closest first."""
        if self.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        return [
            Hit(id=i, text=doc, source=meta["source"], page=int(meta["page"]), distance=float(dist))
            for i, doc, meta, dist in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]