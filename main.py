from __future__ import annotations

from pathlib import Path

import gradio as gr

from settings import get_paths
from pdf_ingest import ensure_slides_dir, iter_pdf_page_chunks
from vector_store import load_or_build_chroma, similarity_search
from tutor_chat import RagTutor
from anthropic_client import get_client
from about_me_ingest import extract_pdf_text, build_profile_from_text
from profile_store import load_profile, save_profile, StudentProfile
from problem_gen import generate_problem
from runner import run_student_code, RunResult
from coach import debug_coach_reply


def build_app() -> gr.Blocks:
    paths = get_paths()
    ensure_slides_dir(paths.slides_dir)
    paths.profiles_dir.mkdir(parents=True, exist_ok=True)

    # Build/load vector DB once.
    chunks = list(iter_pdf_page_chunks(paths.slides_dir))
    vector_db = load_or_build_chroma(chroma_dir=paths.chroma_dir, chunks=chunks)

    client = get_client()
    rag = RagTutor(vector_db=vector_db, slides_dir=paths.slides_dir, anthropic_client=client)

    # --- helpers ---
    def _get_profile(student_id: str) -> StudentProfile | None:
        return load_profile(paths.profiles_dir, student_id)

    def upload_about_me(student_id: str, pdf_file):
        if not student_id.strip():
            return "Please enter a student name/id first.", ""
        if not pdf_file:
            return "Please upload a 1-page PDF.", ""
        pdf_path = Path(pdf_file.name)
        about_text = extract_pdf_text(pdf_path)
        profile = build_profile_from_text(student_id=student_id.strip(), about_text=about_text, anthropic_client=client)
        saved_to = save_profile(paths.profiles_dir, profile)
        summary_md = (
            f"**Saved profile:** `{saved_to.name}`\n\n"
            f"**Summary:** {profile.summary or '(no summary)'}\n\n"
            f"**Interests:** {', '.join(profile.interests) or '(none found)'}\n\n"
            f"**Preferred style:** {', '.join(profile.preferred_style) or '(default)'}"
        )
        return "Profile uploaded and saved.", summary_md

    def rag_chat(query: str, show_context_pages: bool, hd: bool):
        out = rag.chat(query, show_context_pages=show_context_pages, hd=hd)
        # UX: single best slide preview + optional related pages (accordion handled in UI)
        return out.answer_md, out.related

    def _curriculum_snippets_for_topic(topic: str) -> str:
        if not topic.strip():
            return "No topic provided."
        results = similarity_search(vector_db=vector_db, query=topic, k=3)
        if not results:
            return "No curriculum snippets found."
        parts = []
        for i, r in enumerate(results, start=1):
            parts.append(f"Snippet {i} ({r.filename} p{r.page_number}): {r.text[:350]}")
        return "\n\n".join(parts)

    def generate_practice(student_id: str, topic: str, problem_type: str):
        profile = _get_profile(student_id.strip()) if student_id else None
        snippets = _curriculum_snippets_for_topic(topic)
        problem_md, starter = generate_problem(
            anthropic_client=client,
            topic=topic.strip() or "programming basics",
            problem_type_label=problem_type,
            curriculum_snippets=snippets,
            profile=profile,
        )
        return problem_md, starter

    def run_code(code: str):
        rr = run_student_code(code or "")
        # store structured RunResult in state by converting to dict-like tuple
        display = rr.stdout_stderr
        return display, rr

    def coach_chat(student_id: str, problem_md: str, code: str, run_state: RunResult | None, msg: str, history):
        profile = _get_profile(student_id.strip()) if student_id else None
        reply = debug_coach_reply(
            anthropic_client=client,
            student_message=msg,
            problem_md=problem_md or "",
            student_code=code or "",
            run=run_state,
            profile=profile,
        )
        history = history or []
        history.append((msg, reply))
        return history, ""

    # --- UI ---
    css = """
    .gradio-container { font-size: 16px !important; }
    .iw_md { font-size: 18px !important; line-height: 1.6 !important; }
    .iw_mono textarea { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace !important; }
    """

    with gr.Blocks(title="InclusiveWorld Learning Studio", theme=gr.themes.Soft(), css=css) as demo:
        gr.Markdown("## InclusiveWorld Learning Studio\nCalm, step-by-step support for learning Python.")

        with gr.Row():
            with gr.Column(scale=2):
                student_id = gr.Textbox(label="Student name / id", placeholder="e.g., Jordan")
            with gr.Column(scale=3):
                profile_status = gr.Markdown("", elem_classes=["iw_md"])

        with gr.Accordion("Student profile (upload 1-page About Me PDF)", open=False):
            about_pdf = gr.File(label="About Me PDF", file_types=[".pdf"])
            upload_btn = gr.Button("Save profile")
            upload_status = gr.Textbox(label="Status", interactive=False)
            profile_summary = gr.Markdown("", elem_classes=["iw_md"])
            upload_btn.click(fn=upload_about_me, inputs=[student_id, about_pdf], outputs=[upload_status, profile_summary])

        with gr.Tabs() as tabs:
            with gr.Tab("Chat (Curriculum)"):
                with gr.Row():
                    with gr.Column(scale=2):
                        q = gr.Textbox(label="Question", placeholder="Ask about a concept from the slides...", lines=3)
                        with gr.Row():
                            show_context = gr.Checkbox(label="Show nearby pages", value=False)
                            hd = gr.Checkbox(label="HD slide render", value=False)
                        ask = gr.Button("Ask", variant="primary")
                        a = gr.Markdown(label="Answer", elem_classes=["iw_md"])
                    with gr.Column(scale=1):
                        with gr.Accordion("Most relevant slide (and optional context)", open=True):
                            gallery = gr.Gallery(columns=1, rows=3, height="520px", object_fit="contain", show_label=False)

                ask.click(fn=rag_chat, inputs=[q, show_context, hd], outputs=[a, gallery])
                q.submit(fn=rag_chat, inputs=[q, show_context, hd], outputs=[a, gallery])

            with gr.Tab("Code Practice"):
                gr.Markdown("### Practice\nGenerate a small problem, write code, run it, then ask for help with what went wrong.", elem_classes=["iw_md"])

                with gr.Row():
                    with gr.Column(scale=1):
                        topic = gr.Textbox(label="Topic", placeholder="e.g., for loops, functions, lists", lines=1)
                        problem_type = gr.Dropdown(
                            label="Problem type",
                            value="Create Practice Problems",
                            choices=[
                                "Create Practice Problems",
                                "Debug - Identify Error Type",
                                "Debug - Explain Error Reason",
                                "Debug - Fix the Error",
                                "Optimize Code Performance",
                            ],
                        )
                        gen_btn = gr.Button("Generate practice", variant="primary")
                        problem_md = gr.Markdown("", label="Problem", elem_classes=["iw_md"])
                        starter = gr.Code(label="Starter code", language="python", lines=10, value="# Starter code will appear here")

                    with gr.Column(scale=1):
                        code = gr.Code(label="Your code", language="python", lines=16, value="# Write your solution here")
                        run_btn = gr.Button("Run code", variant="secondary")
                        terminal = gr.Textbox(label="Output / Traceback", lines=10, interactive=False, elem_classes=["iw_mono"])
                        run_state = gr.State(value=None)  # stores RunResult

                gen_btn.click(fn=generate_practice, inputs=[student_id, topic, problem_type], outputs=[problem_md, starter])
                # When starter code is generated, copy it into the main editor too (UX)
                gen_btn.click(fn=lambda c: c, inputs=[starter], outputs=[code])

                run_btn.click(fn=run_code, inputs=[code], outputs=[terminal, run_state])

                with gr.Accordion("Debug Coach (ask what went wrong)", open=True):
                    coach_history = gr.Chatbot(label="Coach", height=260, type="messages")
                    coach_msg = gr.Textbox(label="Your question", placeholder="e.g., What went wrong? Why is it failing? What should I try next?")
                    coach_send = gr.Button("Ask coach")
                    coach_send.click(
                        fn=coach_chat,
                        inputs=[student_id, problem_md, code, run_state, coach_msg, coach_history],
                        outputs=[coach_history, coach_msg],
                    )
                    coach_msg.submit(
                        fn=coach_chat,
                        inputs=[student_id, problem_md, code, run_state, coach_msg, coach_history],
                        outputs=[coach_history, coach_msg],
                    )

    return demo


if __name__ == "__main__":
    build_app().launch()

