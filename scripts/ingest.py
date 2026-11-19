"""Build the local vector store from the documents in data/raw.

Usage (from the repository root, with the virtual environment active):
    python scripts/ingest.py                 # add or update chunks
    python scripts/ingest.py --reset         # delete the store first and rebuild it
    python scripts/ingest.py --config configs/other.yaml

Use --reset whenever the documents, chunk size or embedding model change, so no chunks
from the old settings are left behind.

A summary is printed and saved as JSON under results/ingest/ (committed as evidence).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from local_rag_pipeline.config import ConfigError, load_config
from local_rag_pipeline.embeddings import EmbeddingError
from local_rag_pipeline.ingest.pipeline import TRUNCATION_WARNING_SHARE, run_ingest


def show_progress(done: int, total: int) -> None:
    width = 30
    filled = int(width * done / total)
    print(f"\rEmbedding  [{'#' * filled}{'.' * (width - filled)}] {done}/{total}", end="", flush=True)
    if done == total:
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Load, chunk, embed and store the source documents.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--reset", action="store_true", help="delete the existing store before ingesting")
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(exc)
        return 1

    print(f"Config     {cfg.source_file.name} | chunks {cfg.ingest.chunk_size}/{cfg.ingest.chunk_overlap} chars "
          f"| embeddings {cfg.ingest.embedding_model} (num_gpu={cfg.ollama.num_gpu}) | reset={args.reset}")

    try:
        s = run_ingest(cfg, reset=args.reset, progress=show_progress)
    except (FileNotFoundError, EmbeddingError) as exc:
        print(f"\n{exc}")
        return 1

    print(f"Documents  {s.files} file(s), {s.pages} pages ({s.empty_pages} empty) in {s.load_seconds:.1f} s")
    for name, reason in s.skipped:
        print(f"  skipped  {name}: {reason}")
    print(f"Chunks     {s.chunks} (mean {s.mean_chunk_chars} chars, max {s.max_chunk_chars})")
    print(f"Embedded   {s.chunks} chunks in {s.embed_seconds:.1f} s ({s.chunks_per_second:.1f} chunks/s), "
          f"{s.embedding_tokens} tokens (~{s.embedding_tokens // max(s.chunks, 1)} per chunk)")
    print(f"Truncated  {s.truncated_chunks} chunk(s) ({s.truncated_share:.1%})")
    print(f"Stored     {s.stored_total} chunks in '{cfg.vector_store.collection}' at {cfg.vector_store.persist_dir}")
    print(f"Total      {s.total_seconds:.1f} s")

    if s.truncated_share > TRUNCATION_WARNING_SHARE:
        print(f"WARNING    more than {TRUNCATION_WARNING_SHARE:.0%} of chunks were truncated: "
              f"consider a smaller ingest.chunk_size")
    if s.stored_total > s.chunks:
        print(f"NOTE       the store holds {s.stored_total - s.chunks} chunk(s) from earlier runs: "
              f"use --reset if the documents or chunk settings changed")

    out_dir = Path(cfg.benchmark.output_dir) / "ingest"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"ingest_{datetime.now():%Y%m%d_%H%M%S}_{s.machine}.json"
    out_file.write_text(json.dumps({"summary": s.to_dict(), "config": cfg.to_dict()}, indent=2), encoding="utf-8")
    print(f"Saved      {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())