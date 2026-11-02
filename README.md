# InclusiveWorld Learning Studio

An AI tutor for an introductory Python course. Students ask questions about the weekly lesson slides,
get practice problems themed around their own interests, run their code, and chat with a debugging
coach that knows what their program just did. Built with Gradio and deployable as a Hugging Face Space.

## Features

**Personal profile.** A student uploads an "About Me" PDF. Claude turns it into a small structured
profile (summary, interests, dislikes, preferred learning style), saved locally as JSON per student.
Practice problems and coaching use it.

**Chat with the curriculum (RAG).** Lesson-slide PDFs are split into page chunks, embedded with
`all-MiniLM-L6-v2`, and stored in Chroma. Claude answers from the retrieved slide
text as ground truth, and the app can show the best-matching slide page as an image. When retrieval is not confident (top match distance over 0.30, or no
clear gap between the first two hits), the tutor says so instead of guessing.

**Code practice.** Five problem types: create from scratch; debug by identifying the error type,
explaining it, or fixing it; and optimize. Each generated problem is grounded in the matching slides
and themed around the student's interests.

**Run and get coached.** Students run their code in the app. The debugging coach sees the problem,
the code, and the actual run output or error, and gives hints instead of handing over the answer.

## Run

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
mkdir Slides && cp /path/to/lesson*.pdf Slides/   # your own course PDFs
python app.py
```

The vector index (`chroma_db/`) is built on first run. Student profiles go to `data/profiles/`.
Both are git-ignored, and so is `Slides/`: the course slides belong to the course and are not included.

> **Safety:** `runner.py` executes student code in-process with `exec`. That's fine for a trusted
> classroom machine, but it is **not a sandbox**. Don't expose it publicly without containerized
> execution.

## Layout

| File | Purpose |
|---|---|
| `app.py` / `main.py` | Entry point and the Gradio UI (profile, Chat tab, Code Practice tab) |
| `pdf_ingest.py`, `vector_store.py`, `tutor_chat.py`, `render.py` | Slide ingestion, Chroma index, retrieval-and-answer flow, page rendering |
| `about_me_ingest.py`, `profile_store.py` | About-Me PDF → student profile → JSON storage |
| `problem_gen.py`, `runner.py`, `coach.py` | Practice problems, code execution, debugging coach |
| `anthropic_client.py`, `settings.py` | Claude client, paths and model settings |

## Tech

Python · Gradio · LangChain · Chroma · sentence-transformers · PyMuPDF · Anthropic Claude
