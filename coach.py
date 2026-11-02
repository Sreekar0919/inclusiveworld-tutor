from __future__ import annotations

from anthropic_client import complete_text
from runner import RunResult
from profile_store import StudentProfile


def debug_coach_reply(
    *,
    anthropic_client,
    student_message: str,
    problem_md: str,
    student_code: str,
    run: RunResult | None,
    profile: StudentProfile | None,
) -> str:
    if not anthropic_client:
        return "LLM not available. Please set `ANTHROPIC_KEY`."

    run_block = "No run yet."
    if run:
        run_block = run.stdout_stderr[-2500:]

    style_bits = ""
    if profile and profile.preferred_style:
        style_bits = f"Preferred style: {', '.join(profile.preferred_style)}"
    else:
        style_bits = "Preferred style: step-by-step, short instructions, one change at a time"

    prompt = f"""You are a debugging coach for a student learning Python.
Be calm, concrete, and brief. Do not overwhelm the student.

{style_bits}

Student asked: \"{student_message}\"

Problem:
{problem_md[:1200]}

Student code:
```python
{student_code[:2500]}
```

Most recent run output / traceback:
```
{run_block}
```

Use a hint ladder (do NOT jump to full solution):
1) **What happened**: name the error type (or the symptom).
2) **Why**: 1–2 sentences, plain English.
3) **Try this next (one step)**: one small change or one small check.
4) **Quick check**: a tiny question to confirm understanding (yes/no or one sentence).

If there is no error output, help them test their code with a simple print or example input."""

    return complete_text(client=anthropic_client, prompt=prompt, max_tokens=500, temperature=0.2)

