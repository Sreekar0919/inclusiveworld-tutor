from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vector_store import RetrievalResult, similarity_search
from render import render_page_image
from anthropic_client import complete_text


@dataclass(frozen=True)
class RagAnswer:
    answer_md: str
    best_pdf: str | None
    best_page: int | None
    related: list[tuple[object, str]]  # (PIL Image, label)


def _is_confident(results: list[RetrievalResult]) -> bool:
    if not results:
        return False
    if len(results) == 1:
        return True
    # Chroma scores are distances: smaller is more similar.
    top = results[0].score
    second = results[1].score
    return (top <= 0.30) and ((second - top) >= 0.06)


def _normalize_query(q: str) -> str:
    return " ".join((q or "").strip().split()).lower()


class RagTutor:
    def __init__(self, *, vector_db, slides_dir: Path, anthropic_client):
        self.vector_db = vector_db
        self.slides_dir = slides_dir
        self.client = anthropic_client
        self._retrieval_cache: dict[tuple[str, int], list[RetrievalResult]] = {}
        self._answer_cache: dict[tuple[str, str, int], str] = {}

    def chat(
        self,
        query: str,
        *,
        k: int = 5,
        show_context_pages: bool = False,
        hd: bool = False,
    ) -> RagAnswer:
        norm = _normalize_query(query)
        cache_key = (norm, int(k))
        results = self._retrieval_cache.get(cache_key)
        if results is None:
            results = similarity_search(vector_db=self.vector_db, query=query, k=k)
            self._retrieval_cache[cache_key] = results
            # Bound cache size.
            if len(self._retrieval_cache) > 128:
                self._retrieval_cache.pop(next(iter(self._retrieval_cache)))
        if not results:
            return RagAnswer(
                answer_md="I couldn't find anything relevant in the curriculum for that question.",
                best_pdf=None,
                best_page=None,
                related=[],
            )

        best = results[0]
        slide_content = best.text

        grounded = f"""You are a concise, friendly programming tutor.

Student question: \"{query}\"

Curriculum page content (ground truth):
\"\"\"{slide_content[:1500]}\"\"\"

Write the response in this structure:
1) **Answer (2–4 sentences)**: directly answer the question.
2) **From the slide**: 1–2 short bullet snippets quoted from the page (verbatim or lightly trimmed).
3) **Try this next**: 1 actionable next step (a tiny exercise or check).

Keep it calm and not overwhelming."""

        if self.client:
            answer_key = (norm, best.filename or "", int(best.page_number or 0))
            cached = self._answer_cache.get(answer_key)
            if cached is not None:
                answer_md = cached
            else:
                answer_md = complete_text(
                    client=self.client,
                    prompt=grounded,
                    max_tokens=450,
                    temperature=0.3,
                )
                self._answer_cache[answer_key] = answer_md
                if len(self._answer_cache) > 128:
                    self._answer_cache.pop(next(iter(self._answer_cache)))
        else:
            answer_md = f"**Answer:**\n\n{slide_content}\n"

        # Render: best page only by default; optionally include neighboring context pages.
        pdf_path = self.slides_dir / best.filename
        zoom = 1.6 if hd else 1.2

        related: list[tuple[object, str]] = []
        if pdf_path.exists() and best.page_number:
            page_nums = [best.page_number]
            if show_context_pages:
                page_nums = [
                    max(1, best.page_number - 1),
                    best.page_number,
                    best.page_number + 1,
                ]
            for pn in page_nums:
                img = render_page_image(pdf_path, pn, zoom=zoom)
                if img:
                    label = f"{best.filename} — Page {pn}" + (" (Most relevant)" if pn == best.page_number else "")
                    related.append((img, label))

        return RagAnswer(
            answer_md=answer_md,
            best_pdf=best.filename,
            best_page=best.page_number,
            related=related,
        )

