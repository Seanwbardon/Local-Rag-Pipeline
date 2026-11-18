"""Unit tests for the document loader (white-box)."""

from pathlib import Path

import pytest

from local_rag_pipeline.ingest.loader import clean_text, load_documents


def make_pdf(path: Path, pages: list[str]) -> None:
    """Write a minimal valid PDF with one line of text per page (no extra libraries needed)."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", None,
               "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream.decode('latin-1')}\nendstream")
        content_id = len(objects)
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                       f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>")
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"

    out = b"%PDF-1.4\n"
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{obj}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(out)


# --------------------------------------------------------------------------- clean_text

def test_clean_text_joins_hyphenated_line_breaks():
    assert clean_text("network config-\nuration file") == "network configuration file"


def test_clean_text_keeps_paragraphs_but_joins_lines():
    assert clean_text("line one\nline two\n\n\nnext paragraph") == "line one line two\n\nnext paragraph"


def test_clean_text_collapses_whitespace():
    assert clean_text("  too   many\t\tspaces  ") == "too many spaces"


# --------------------------------------------------------------------------- load_documents

def test_pdf_pages_keep_source_and_page_number(tmp_path):
    make_pdf(tmp_path / "guide.pdf", ["Install the server", "Configure the network"])
    pages, report = load_documents(tmp_path)
    assert [(p.source, p.page) for p in pages] == [("guide.pdf", 1), ("guide.pdf", 2)]
    assert "Configure the network" in pages[1].text
    assert report.files_loaded == 1 and report.pages_loaded == 2


def test_text_and_markdown_files_load_as_page_one(tmp_path):
    (tmp_path / "notes.md").write_text("# Title\nSome notes", encoding="utf-8")
    (tmp_path / "readme.txt").write_text("Plain text", encoding="utf-8")
    pages, _ = load_documents(tmp_path)
    assert {(p.source, p.page) for p in pages} == {("notes.md", 1), ("readme.txt", 1)}


def test_subfolders_are_included_with_relative_names(tmp_path):
    (tmp_path / "networking").mkdir()
    (tmp_path / "networking" / "dns.txt").write_text("DNS settings", encoding="utf-8")
    pages, _ = load_documents(tmp_path)
    assert pages[0].source == "networking/dns.txt"


def test_unsupported_and_broken_files_are_skipped_not_fatal(tmp_path):
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / "broken.pdf").write_bytes(b"not really a pdf")
    (tmp_path / "good.txt").write_text("Still loaded", encoding="utf-8")
    pages, report = load_documents(tmp_path)
    assert [p.source for p in pages] == ["good.txt"]
    assert {name for name, _ in report.skipped} == {"image.png", "broken.pdf"}


def test_empty_files_are_reported(tmp_path):
    (tmp_path / "empty.txt").write_text("   ", encoding="utf-8")
    pages, report = load_documents(tmp_path)
    assert pages == [] and report.empty_pages == 1
    assert report.skipped[0][0] == "empty.txt"


def test_hidden_files_like_gitkeep_are_ignored(tmp_path):
    (tmp_path / ".gitkeep").write_text("")
    _, report = load_documents(tmp_path)
    assert report.skipped == []


def test_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "missing")