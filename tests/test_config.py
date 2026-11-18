"""Unit tests for the config loader (white-box)."""

from pathlib import Path

import pytest

from local_rag_pipeline.config import ConfigError, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT = REPO_ROOT / "configs" / "default.yaml"

VALID = """
ingest:
  source_dir: data/raw
  chunk_size: 800
  chunk_overlap: 100
  embedding_model: all-minilm
vector_store:
  persist_dir: chroma_db
generation:
  model: llama3.2:3b
"""


def write(tmp_path: Path, text: str) -> Path:
    (tmp_path / "pyproject.toml").write_text("")  # marks tmp_path as the repo root
    (tmp_path / "configs").mkdir(exist_ok=True)
    path = tmp_path / "configs" / "test.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_default_config_loads():
    cfg = load_config(DEFAULT)
    assert cfg.ollama.num_gpu == 0, "the project is CPU-only"
    assert cfg.ingest.chunk_overlap < cfg.ingest.chunk_size


def test_defaults_fill_missing_optional_settings(tmp_path):
    cfg = load_config(write(tmp_path, VALID))
    assert cfg.retrieval.top_k == 4
    assert cfg.generation.temperature == 0.0
    assert cfg.ollama.num_gpu == 0


def test_relative_paths_resolve_from_repo_root(tmp_path):
    cfg = load_config(write(tmp_path, VALID))
    assert cfg.ingest.source_dir == tmp_path / "data" / "raw"
    assert cfg.benchmark.output_dir == tmp_path / "results"


def test_ollama_host_environment_variable_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:9999")
    cfg = load_config(write(tmp_path, VALID))
    assert cfg.ollama.host == "http://127.0.0.1:9999"


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_typo_in_setting_name_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="chunk_sise"):
        load_config(write(tmp_path, VALID.replace("chunk_size", "chunk_sise")))


def test_unknown_section_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="retreival"):
        load_config(write(tmp_path, VALID + "retreival:\n  top_k: 4\n"))


def test_missing_required_setting_is_reported(tmp_path):
    with pytest.raises(ConfigError, match="embedding_model"):
        load_config(write(tmp_path, VALID.replace("  embedding_model: all-minilm\n", "")))


def test_overlap_must_be_smaller_than_chunk(tmp_path):
    with pytest.raises(ConfigError, match="chunk_overlap must be smaller"):
        load_config(write(tmp_path, VALID.replace("chunk_overlap: 100", "chunk_overlap: 800")))


def test_all_problems_reported_together(tmp_path):
    bad = VALID.replace("chunk_size: 800", "chunk_size: -1") + "retrieval:\n  top_k: 0\n"
    with pytest.raises(ConfigError) as err:
        load_config(write(tmp_path, bad))
    assert "chunk_size" in str(err.value) and "top_k" in str(err.value)


def test_to_dict_is_plain_data():
    data = load_config(DEFAULT).to_dict()
    assert isinstance(data["ingest"]["source_dir"], str)