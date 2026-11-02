from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF


@dataclass(frozen=True)
class PageChunk:
    text: str
    filename: str
    page_number: int


def iter_pdf_page_chunks(slides_dir: Path) -> Iterable[PageChunk]:
    pdf_files = sorted(slides_dir.glob("*.pdf"))
    for pdf_path in pdf_files:
        doc = fitz.open(str(pdf_path))
        try:
            for i in range(len(doc)):
                page = doc[i]
                text = (page.get_text() or "").strip()
                if not text:
                    continue
                yield PageChunk(text=text, filename=pdf_path.name, page_number=i + 1)
        finally:
            doc.close()


def ensure_slides_dir(slides_dir: Path) -> None:
    slides_dir.mkdir(parents=True, exist_ok=True)

