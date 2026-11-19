"""Tests for the full ingestion run, using a fake embedder (no Ollama needed) and a real ChromaDB store."""

from pathlib import Path

import pytest

pytest.importorskip("chromadb")

from local_rag_pipeline.config import load_config  # noqa: E402
from local_rag_pipeline.embeddings import EmbeddingRun  # noqa: E402
from local_rag_pipeline.ingest.pipeline import run_ingest  # noqa: E402

CONFIG = """
ingest:
  source_dir: data/raw
  chunk_size: 200
  chunk_overlap: 20
  embedding_model: fake-model
vector_store:
  persist_dir: chroma_db
generation:
  model: llama3.2:3b
"""


class FakeEmbedder:
    """Returns a simple 3-number vector per text and marks texts containing 'LONG' as truncated."""

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]

    def embed_documents(self, texts, progress=None):
        if progress:
            progress(len(texts), len(texts))
        return EmbeddingRun(
            vectors=[[float(len(t)), 1.0, 0.5] for t in texts],
            tokens=10 * len(texts),
            truncated=[i for i, t in enumerate(texts) if "LONG" in t],
            requests=1,
        )


@pytest.fixture
def cfg(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "configs").mkdir()
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "configs" / "test.yaml").write_text(CONFIG, encoding="utf-8")
    raw = tmp_path / "data" / "raw"
    (raw / "a.txt").write_text("Netplan sets static addresses. " * 20, encoding="utf-8")
    (raw / "b.md").write_text("A LONG page about firewalls and ufw rules. " * 3, encoding="utf-8")
    return load_config(tmp_path / "configs" / "test.yaml")


def test_full_run_stores_every_chunk(cfg):
    s = run_ingest(cfg, embedder=FakeEmbedder())
    assert s.files == 2 and s.pages == 2
    assert s.chunks > 2
    assert s.stored_total == s.chunks
    assert s.max_chunk_chars <= 200


def test_truncated_chunks_are_reported_by_id(cfg):
    s = run_ingest(cfg, embedder=FakeEmbedder())
    assert s.truncated_chunks >= 1
    assert all(i.startswith("b.md:") for i in s.truncated_ids)


def test_running_twice_does_not_duplicate(cfg):
    first = run_ingest(cfg, embedder=FakeEmbedder())
    second = run_ingest(cfg, embedder=FakeEmbedder())
    assert second.stored_total == first.stored_total


def test_reset_removes_chunks_from_old_documents(cfg):
    run_ingest(cfg, embedder=FakeEmbedder())
    (cfg.ingest.source_dir / "a.txt").unlink()
    s = run_ingest(cfg, reset=True, embedder=FakeEmbedder())
    assert s.stored_total == s.chunks


def test_empty_source_folder_is_a_clear_error(cfg):
    for f in cfg.ingest.source_dir.iterdir():
        f.unlink()
    with pytest.raises(FileNotFoundError, match="No readable documents"):
        run_ingest(cfg, embedder=FakeEmbedder())


def test_summary_is_json_friendly(cfg):
    import json
    data = run_ingest(cfg, embedder=FakeEmbedder()).to_dict()
    json.dumps(data)
    assert "chunks_per_second" in data