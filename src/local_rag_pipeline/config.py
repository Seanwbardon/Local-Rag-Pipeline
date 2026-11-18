"""Load and validate the experiment configuration.

Every setting the pipeline uses comes from a YAML file in configs/, so an experiment is
fully described by its config file. Settings are checked when the file is loaded, so a
typo or an invalid value fails immediately with a clear message rather than halfway
through a long benchmark run.

Usage:
    from local_rag_pipeline.config import load_config
    cfg = load_config("configs/default.yaml")
    cfg.ingest.chunk_size        # 800
    cfg.ollama.num_gpu           # 0
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a config file is missing, malformed or contains invalid values."""


# --------------------------------------------------------------------------- settings sections

@dataclass(frozen=True)
class OllamaConfig:
    host: str = "http://localhost:11434"
    num_gpu: int = 0


@dataclass(frozen=True)
class IngestConfig:
    source_dir: Path
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    batch_size: int = 32


@dataclass(frozen=True)
class VectorStoreConfig:
    persist_dir: Path
    collection: str = "documents"


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 4


@dataclass(frozen=True)
class GenerationConfig:
    model: str
    num_ctx: int = 4096
    temperature: float = 0.0
    seed: int = 42


@dataclass(frozen=True)
class BenchmarkConfig:
    repeats: int = 5
    output_dir: Path = Path("results")


@dataclass(frozen=True)
class Config:
    ollama: OllamaConfig
    ingest: IngestConfig
    vector_store: VectorStoreConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    benchmark: BenchmarkConfig
    source_file: Path

    def to_dict(self) -> dict[str, Any]:
        """Plain dictionary of all settings, e.g. to save next to benchmark results."""
        def clean(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {k: clean(v) for k, v in value.items()}
            return value
        return clean(asdict(self))


SECTIONS = {
    "ollama": OllamaConfig,
    "ingest": IngestConfig,
    "vector_store": VectorStoreConfig,
    "retrieval": RetrievalConfig,
    "generation": GenerationConfig,
    "benchmark": BenchmarkConfig,
}
PATH_FIELDS = {("ingest", "source_dir"), ("vector_store", "persist_dir"), ("benchmark", "output_dir")}


# --------------------------------------------------------------------------- loading

def find_repo_root(start: Path) -> Path:
    """Walk up from `start` to the folder holding pyproject.toml (the repository root)."""
    for folder in [start, *start.parents]:
        if (folder / "pyproject.toml").exists():
            return folder
    return start


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    """Read a YAML config file, check it, and return typed settings."""
    path = Path(path).resolve()
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path.name} must contain sections such as 'ingest:' and 'generation:'")

    unknown = set(raw) - set(SECTIONS)
    if unknown:
        raise ConfigError(f"Unknown section(s) in {path.name}: {', '.join(sorted(unknown))}")

    root = find_repo_root(path.parent)
    sections = {name: _build_section(name, cls, raw.get(name), root) for name, cls in SECTIONS.items()}

    # The OLLAMA_HOST environment variable wins, so each machine can point at its own server.
    if os.environ.get("OLLAMA_HOST"):
        host = os.environ["OLLAMA_HOST"].rstrip("/")
        if not host.startswith("http"):
            host = "http://" + host
        sections["ollama"] = OllamaConfig(host=host, num_gpu=sections["ollama"].num_gpu)

    config = Config(**sections, source_file=path)
    validate(config)
    return config


def _build_section(name: str, cls: type, values: Any, root: Path):
    values = {} if values is None else values
    if not isinstance(values, dict):
        raise ConfigError(f"Section '{name}' must be a set of key: value settings")

    allowed = {f.name for f in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ConfigError(
            f"Unknown setting(s) in '{name}': {', '.join(sorted(unknown))} "
            f"(allowed: {', '.join(sorted(allowed))})"
        )

    for key in list(values):
        if (name, key) in PATH_FIELDS:
            folder = Path(values[key])
            values[key] = folder if folder.is_absolute() else root / folder

    try:
        section = cls(**values)
    except TypeError as exc:  # a required setting with no default is missing
        missing = [f.name for f in fields(cls) if f.name not in values]
        raise ConfigError(f"Missing setting(s) in '{name}': {', '.join(missing)}") from exc

    # Default path values are relative too: anchor them to the repository root.
    for (sec, key) in PATH_FIELDS:
        if sec == name and key not in values:
            section = replace(section, **{key: root / getattr(section, key)})
    return section


# --------------------------------------------------------------------------- validation

def validate(cfg: Config) -> None:
    """Check values that are the right type but would break or distort an experiment."""
    problems: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            problems.append(message)

    def is_int(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    require(is_int(cfg.ollama.num_gpu) and cfg.ollama.num_gpu >= 0, "ollama.num_gpu must be 0 or more")
    require(str(cfg.ollama.host).startswith("http"), "ollama.host must start with http:// or https://")

    ing = cfg.ingest
    require(is_int(ing.chunk_size) and ing.chunk_size > 0, "ingest.chunk_size must be a positive whole number")
    require(is_int(ing.chunk_overlap) and ing.chunk_overlap >= 0, "ingest.chunk_overlap must be 0 or more")
    if is_int(ing.chunk_size) and is_int(ing.chunk_overlap):
        require(ing.chunk_overlap < ing.chunk_size, "ingest.chunk_overlap must be smaller than ingest.chunk_size")
    require(is_int(ing.batch_size) and ing.batch_size > 0, "ingest.batch_size must be a positive whole number")
    require(bool(str(ing.embedding_model).strip()), "ingest.embedding_model must not be empty")

    require(bool(str(cfg.vector_store.collection).strip()), "vector_store.collection must not be empty")
    require(is_int(cfg.retrieval.top_k) and cfg.retrieval.top_k > 0, "retrieval.top_k must be a positive whole number")

    gen = cfg.generation
    require(bool(str(gen.model).strip()), "generation.model must not be empty")
    require(is_int(gen.num_ctx) and gen.num_ctx >= 512, "generation.num_ctx must be at least 512")
    require(isinstance(gen.temperature, (int, float)) and not isinstance(gen.temperature, bool)
            and 0 <= gen.temperature <= 2, "generation.temperature must be between 0 and 2")
    require(is_int(gen.seed), "generation.seed must be a whole number")

    require(is_int(cfg.benchmark.repeats) and cfg.benchmark.repeats > 0, "benchmark.repeats must be a positive whole number")

    if problems:
        raise ConfigError(f"Invalid settings in {cfg.source_file.name}:\n  - " + "\n  - ".join(problems))