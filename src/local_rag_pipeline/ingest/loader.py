"""Load source documents page by page, keeping where each piece of text came from.

Every page becomes a `Page` with its file name and page number. That metadata travels with
each chunk into the vector store, so an answer can cite "server-guide.pdf, p. 42".

Supported files: .pdf (one Page per page), .txt and .md (the whole file as page 1).
Files that can't be read are skipped and reported rather than stopping the whole run.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


@dataclass(frozen=True)
class Page:
    source: str   # file name relative to the source folder, e.g. "server-guide.pdf"
    page: int     # 1-based page number, as a reader would cite it
    text: str


@dataclass
class LoadReport:
    """What happened during loading, for the ingest summary and the results log."""
    files_loaded: int = 0
    pages_loaded: int = 0
    empty_pages: int = 0                                     # e.g. scanned images with no text layer
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (file, reason)


# --------------------------------------------------------------------------- text cleaning

def clean_text(text: str) -> str:
    """Normalise text extracted from a PDF without changing its meaning.

    - joins words split across a line break with a hyphen ("config-\\nuration" -> "configuration")
    - turns single line breaks inside a paragraph into spaces
    - keeps paragraph breaks (blank lines), which the splitter uses as natural boundaries
    - collapses runs of spaces and tabs
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)                     # runs of spaces/tabs
    text = re.sub(r" *\n *", "\n", text)                    # spaces around line breaks
    text = re.sub(r"\n{2,}", "\n\n", text)                  # paragraph breaks -> exactly one blank line
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)            # single line breaks -> spaces
    return text.strip()


# --------------------------------------------------------------------------- loading

def load_pdf(path: Path, source: str, report: LoadReport) -> list[Page]:
    reader = PdfReader(path)
    if reader.is_encrypted:
        try:
            reader.decrypt("")  # many "protected" PDFs open with an empty password
        except Exception as exc:
            raise PdfReadError("encrypted") from exc

    pages = []
    for number, pdf_page in enumerate(reader.pages, start=1):
        text = clean_text(pdf_page.extract_text() or "")
        if not text:
            report.empty_pages += 1
            continue
        pages.append(Page(source=source, page=number, text=text))
    return pages


def load_text_file(path: Path, source: str, report: LoadReport) -> list[Page]:
    text = clean_text(path.read_text(encoding="utf-8", errors="replace"))
    if not text:
        report.empty_pages += 1
        return []
    return [Page(source=source, page=1, text=text)]


def load_documents(source_dir: Path) -> tuple[list[Page], LoadReport]:
    """Load every supported file under `source_dir` (including subfolders), in a stable order."""
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source folder not found: {source_dir}")

    report = LoadReport()
    pages: list[Page] = []

    files = sorted(p for p in source_dir.rglob("*") if p.is_file() and not p.name.startswith("."))
    for path in files:
        source = path.relative_to(source_dir).as_posix()
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            report.skipped.append((source, f"unsupported file type '{suffix}'"))
            continue
        try:
            loaded = load_pdf(path, source, report) if suffix == ".pdf" else load_text_file(path, source, report)
        except (PdfReadError, OSError, ValueError) as exc:
            report.skipped.append((source, f"could not be read: {exc}"))
            logger.warning("Skipping %s: %s", source, exc)
            continue
        if not loaded:
            report.skipped.append((source, "no extractable text (scanned or empty?)"))
            continue
        pages.extend(loaded)
        report.files_loaded += 1
        report.pages_loaded += len(loaded)

    return pages, report