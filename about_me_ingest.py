from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF

from anthropic_client import complete_text
from profile_store import StudentProfile


def extract_pdf_text(pdf_path: Path, *, max_chars: int = 8000) -> str:
    doc = fitz.open(str(pdf_path))
    try:
        parts: list[str] = []
        for i in range(len(doc)):
            t = (doc[i].get_text() or "").strip()
            if t:
                parts.append(t)
        text = "\n\n".join(parts).strip()
        if len(text) > max_chars:
            return text[:max_chars]
        return text
    finally:
        doc.close()


def build_profile_from_text(*, student_id: str, about_text: str, anthropic_client) -> StudentProfile:
    if not anthropic_client:
        # Minimal fallback: no LLM structuring
        summary = about_text[:600].strip()
        return StudentProfile(
            student_id=student_id,
            raw_text=about_text,
            summary=summary,
            interests=[],
            dislikes=[],
            preferred_style=[],
        )

    prompt = f"""You are helping a teacher personalize programming practice for a neurodivergent student.

Here is the student's 1-page \"about me\" text:
\"\"\"{about_text}\"\"\"

Extract a compact profile in this exact format:

SUMMARY: <2-4 sentences>
INTERESTS: <comma-separated list, 3-10 items>
DISLIKES: <comma-separated list, 0-6 items>
PREFERRED_STYLE: <comma-separated list of learning preferences, 2-8 items>

Rules:
- Keep it concrete (topics/themes like music, sports, animals, space, art, games).
- Preferred style examples: \"step-by-step\", \"short instructions\", \"visual examples\", \"gentle tone\", \"one task at a time\".
- If a section is unknown, leave it blank after the colon."""

    text = complete_text(client=anthropic_client, prompt=prompt, max_tokens=400, temperature=0.2)

    def _grab(label: str) -> str:
        m = re.search(rf"^{label}:\s*(.*)$", text, flags=re.MULTILINE)
        return (m.group(1).strip() if m else "")

    summary = _grab("SUMMARY")
    interests = [x.strip() for x in _grab("INTERESTS").split(",") if x.strip()]
    dislikes = [x.strip() for x in _grab("DISLIKES").split(",") if x.strip()]
    style = [x.strip() for x in _grab("PREFERRED_STYLE").split(",") if x.strip()]

    return StudentProfile(
        student_id=student_id,
        raw_text=about_text,
        summary=summary,
        interests=interests,
        dislikes=dislikes,
        preferred_style=style,
    )

