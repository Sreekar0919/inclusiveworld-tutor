from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from settings import DEFAULT_EMBEDDING_MODEL
from pdf_ingest import PageChunk


@dataclass(frozen=True)
class RetrievalResult:
    text: str
    filename: str
    page_number: int
    score: float


def _embedding_fn() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)


def load_or_build_chroma(
    *,
    chroma_dir: Path,
    chunks: Iterable[PageChunk],
) -> Chroma:
    """
    Loads an existing persisted Chroma DB if present; otherwise builds it once and persists.
    If no chunks are provided, returns an empty Chroma instance.
    """
    chroma_dir.mkdir(parents=True, exist_ok=True)
    embeddings = _embedding_fn()

    # If a Chroma collection already exists in the persist directory, just load it.
    # Chroma creates multiple files; we use a small heuristic: if directory is non-empty.
    if any(chroma_dir.iterdir()):
        return Chroma(persist_directory=str(chroma_dir), embedding_function=embeddings)

    texts: list[str] = []
    metadatas: list[dict] = []
    for c in chunks:
        texts.append(c.text)
        metadatas.append({"filename": c.filename, "page_number": c.page_number})

    # If no texts (no PDFs), create empty Chroma instance
    if not texts:
        return Chroma(persist_directory=str(chroma_dir), embedding_function=embeddings)

    vs = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=str(chroma_dir),
    )
    return vs


def similarity_search(
    *,
    vector_db: Chroma,
    query: str,
    k: int = 5,
) -> list[RetrievalResult]:
    """
    Uses similarity_search_with_score so the RAG pipeline can be adaptive.
    """
    docs_scores = vector_db.similarity_search_with_score(query, k=k)
    out: list[RetrievalResult] = []
    for doc, score in docs_scores:
        md = doc.metadata or {}
        out.append(
            RetrievalResult(
                text=doc.page_content,
                filename=str(md.get("filename", "")),
                page_number=int(md.get("page_number", 0) or 0),
                score=float(score),
            )
        )
    return out

