"""Load PDFs and extract page-level text with source metadata."""

from pathlib import Path

from pypdf import PdfReader


def load_pdfs(file_paths):
    """Return readable pages as ``text``, ``source``, and ``page`` dictionaries.

    Empty pages are skipped. A malformed file raises a descriptive error so the
    UI can tell the user which upload needs attention.
    """
    documents = []

    for raw_path in file_paths:
        path = Path(raw_path)
        try:
            reader = PdfReader(str(path))
            for page_num, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    documents.append(
                        {
                            "text": text,
                            "source": path.name,
                            "page": page_num,
                        }
                    )
        except Exception as exc:
            raise ValueError(f"Could not read '{path.name}': {exc}") from exc

    return documents
