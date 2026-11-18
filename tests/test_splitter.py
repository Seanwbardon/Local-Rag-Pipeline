"""Unit tests for the chunk splitter (white-box)."""

from local_rag_pipeline.ingest.loader import Page
from local_rag_pipeline.ingest.splitter import MIN_CHUNK_CHARS, chunk_stats, split_pages

# 40 different sentences (about 2,800 characters), so the text must be split and
# any repeated text between chunks can only come from the overlap.
LONG_TEXT = "".join(f"Step {i}: the netplan file sets a static IP for interface eth{i}. " for i in range(40))


def page(text: str, source: str = "guide.pdf", number: int = 1) -> Page:
    return Page(source=source, page=number, text=text)


def test_no_chunk_is_longer_than_chunk_size():
    chunks = split_pages([page(LONG_TEXT)], chunk_size=800, chunk_overlap=100)
    assert len(chunks) > 1
    assert all(len(c.text) <= 800 for c in chunks)


def test_neighbouring_chunks_overlap():
    chunks = split_pages([page(LONG_TEXT)], chunk_size=800, chunk_overlap=100)
    start_of_second = chunks[1].text[:30]
    assert start_of_second in chunks[0].text, "the second chunk should repeat the end of the first"


def test_short_page_stays_one_chunk():
    chunks = split_pages([page("A short page about SSH keys.")], chunk_size=800, chunk_overlap=100)
    assert [c.text for c in chunks] == ["A short page about SSH keys."]


def test_chunks_keep_source_and_page():
    chunks = split_pages([page(LONG_TEXT, "server/net.pdf", 42)], chunk_size=800, chunk_overlap=100)
    assert {(c.source, c.page) for c in chunks} == {("server/net.pdf", 42)}
    assert chunks[0].metadata == {"source": "server/net.pdf", "page": 42, "chunk_index": 0}


def test_ids_are_unique_and_stable_between_runs():
    pages = [page(LONG_TEXT, number=1), page(LONG_TEXT, number=2)]
    first = [c.id for c in split_pages(pages, 800, 100)]
    second = [c.id for c in split_pages(pages, 800, 100)]
    assert first == second
    assert len(set(first)) == len(first)
    assert first[0] == "guide.pdf:p1:c0"


def test_chunks_never_cross_pages():
    chunks = split_pages([page("Page one text " * 5, number=1), page("Page two text " * 5, number=2)], 800, 100)
    assert [c.page for c in chunks] == [1, 2]


def test_paragraph_boundaries_are_preferred():
    para_a = "A" * 500
    para_b = "B" * 500
    chunks = split_pages([page(f"{para_a}\n\n{para_b}")], chunk_size=800, chunk_overlap=100)
    assert [c.text for c in chunks] == [para_a, para_b], "should split at the blank line, not mid-paragraph"


def test_tiny_fragments_are_dropped():
    chunks = split_pages([page("42")], chunk_size=800, chunk_overlap=100)
    assert chunks == []
    assert MIN_CHUNK_CHARS > len("42")


def test_chunk_stats():
    chunks = split_pages([page(LONG_TEXT)], chunk_size=800, chunk_overlap=100)
    stats = chunk_stats(chunks)
    assert stats["chunks"] == len(chunks)
    assert stats["max_chars"] <= 800
    assert chunk_stats([])["chunks"] == 0