"""Search the local vector store and show the chunks a question would retrieve.

Usage (from the repository root, with the virtual environment active):
    python scripts/search.py "How do I configure a static IP address?"
    python scripts/search.py "How do I enable the ufw firewall?" --top-k 6
    python scripts/search.py "..." --full          # print whole chunks

Shows each chunk's cosine distance (lower = closer in meaning), file and page.
Nothing is written to disk. Build the store first with: python scripts/ingest.py --reset
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from local_rag_pipeline.config import ConfigError, load_config
from local_rag_pipeline.embeddings import EmbeddingError, OllamaEmbedder
from local_rag_pipeline.retrieval.search import search
from local_rag_pipeline.vector_store import VectorStore, VectorStoreError

PREVIEW_CHARS = 300


def main() -> int:
    parser = argparse.ArgumentParser(description="Show the chunks the vector store returns for a question.")
    parser.add_argument("question", help="the question, in quotes")
    parser.add_argument("--top-k", type=int, help="number of chunks to show (default: retrieval.top_k in the config)")
    parser.add_argument("--full", action="store_true", help="print whole chunks instead of a preview")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    # Chunk text can contain characters a Windows console code page can't show; never crash on them.
    sys.stdout.reconfigure(errors="replace")

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(exc)
        return 1
    top_k = args.top_k or cfg.retrieval.top_k
    if top_k < 1:
        print("--top-k must be at least 1")
        return 1

    embedder = OllamaEmbedder(cfg.ingest.embedding_model, host=cfg.ollama.host, num_gpu=cfg.ollama.num_gpu)
    try:
        store = VectorStore(cfg.vector_store.persist_dir, cfg.vector_store.collection, cfg.ingest.embedding_model)
        total = store.count()
    except VectorStoreError as exc:
        print(exc)
        return 1
    if total == 0:
        print("The vector store is empty. Build it first with:  python scripts/ingest.py --reset")
        return 1

    try:
        result = search(args.question, embedder, store, top_k=top_k)
    except (EmbeddingError, ValueError) as exc:
        print(exc)
        return 1

    print(f"Question   {result.question}")
    print(f"Store      {total:,} chunks | {cfg.ingest.embedding_model} (num_gpu={cfg.ollama.num_gpu})")
    print(f"Timing     embed {result.embed_ms:.0f} ms | search {result.search_ms:.0f} ms")

    for rank, hit in enumerate(result.hits, start=1):
        print(f"\n{rank}. [{hit.distance:.2f}] {hit.source}, p. {hit.page}")
        text = hit.text if args.full else hit.text[:PREVIEW_CHARS] + ("..." if len(hit.text) > PREVIEW_CHARS else "")
        for line in text.splitlines():
            print(textwrap.indent(line, "   "))
    return 0


if __name__ == "__main__":
    sys.exit(main())