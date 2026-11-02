from __future__ import annotations

from functools import lru_cache
import io
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image


@lru_cache(maxsize=256)
def _render_page_png_bytes(pdf_path_str: str, page_num: int, zoom: float) -> bytes:
    pdf_path = Path(pdf_path_str)
    doc = fitz.open(str(pdf_path))
    try:
        if page_num < 1 or page_num > len(doc):
            return b""
        page = doc[page_num - 1]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        return pix.tobytes("png")
    finally:
        doc.close()


def render_page_image(pdf_path: Path, page_num: int, zoom: float = 1.2) -> Image.Image | None:
    data = _render_page_png_bytes(str(pdf_path), int(page_num), float(zoom))
    if not data:
        return None
    img = Image.open(io.BytesIO(data))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

