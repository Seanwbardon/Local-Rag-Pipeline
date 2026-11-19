"""Split loaded pages into overlapping chunks for embedding.

Chunks are made per page and never cross a page boundary, so every chunk can be cited by
an exact file and page. Each chunk gets a stable ID built from where it came from, e.g.
"ubuntu-server-guide.pdf:p42:c3", so running ingestion again overwrites chunks instead
of duplicating them.

Splitting uses LangChain's RecursiveCharacterTextSplitter, which tries the separators in
order: blank lines (paragraphs), line breaks (including lines of code and config), sentence
ends, spaces, and only then single characters. Chunks therefore rarely break mid-sentence
or mid-line.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from local_rag_pipeline.ingest.loader import Page

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
MIN_CHUNK_CHARS = 20  # drop fragments such as a lone page number or heading


@dataclass(frozen=True)
class Chunk:
    id: str           # stable: "<source>:p<page>:c<index>"
    source: str
    page: int
    chunk_index: int  # position within its page, starting at 0
    text: str

    @property
    def metadata(self) -> dict:
        """Metadata stored with the chunk in the vector store."""
        return {"source": self.source, "page": self.page, "chunk_index": self.chunk_index}


def make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        length_function=len,  # sizes are in characters, matching the config
    )


def split_pages(pages: list[Page], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split every page into chunks of at most `chunk_size` characters."""
    splitter = make_splitter(chunk_size, chunk_overlap)
    chunks: list[Chunk] = []
    for page in pages:
        pieces = [p.strip() for p in splitter.split_text(page.text)]
        pieces = [p for p in pieces if len(p) >= MIN_CHUNK_CHARS]
        for index, text in enumerate(pieces):
            chunks.append(Chunk(
                id=f"{page.source}:p{page.page}:c{index}",
                source=page.source,
                page=page.page,
                chunk_index=index,
                text=text,
            ))
    return chunks


def chunk_stats(chunks: list[Chunk]) -> dict:
    """Summary figures for the ingest report and the ingestion slide."""
    if not chunks:
        return {"chunks": 0, "mean_chars": 0, "median_chars": 0, "max_chars": 0, "min_chars": 0}
    lengths = [len(c.text) for c in chunks]
    return {
        "chunks": len(chunks),
        "mean_chars": round(statistics.mean(lengths)),
        "median_chars": round(statistics.median(lengths)),
        "max_chars": max(lengths),
        "min_chars": min(lengths),
    }