from __future__ import annotations

from anthropic_client import complete_text
from profile_store import StudentProfile


PROBLEM_TYPE_MAPPING: dict[str, str] = {
    "Create Practice Problems": "create",
    "Debug - Identify Error Type": "debug_error_type",
    "Debug - Explain Error Reason": "debug_error_reason",
    "Debug - Fix the Error": "debug_fix",
    "Optimize Code Performance": "optimize",
}


def generate_problem(
    *,
    anthropic_client,
    topic: str,
    problem_type_label: str,
    curriculum_snippets: str,
    profile: StudentProfile | None,
) -> tuple[str, str]:
    if not anthropic_client:
        return "LLM not available. Please set `ANTHROPIC_KEY`.", "# Write your code here..."

    internal = PROBLEM_TYPE_MAPPING.get(problem_type_label, "create")
    profile_bits = ""
    if profile:
        profile_bits = f"""
Student profile summary:
- Summary: {profile.summary}
- Interests: {", ".join(profile.interests) or "unknown"}
- Dislikes: {", ".join(profile.dislikes) or "none listed"}
- Preferred style: {", ".join(profile.preferred_style) or "step-by-step, short instructions"}
"""

    prompt = f"""Create a SIMPLE, achievable Python practice problem.

Topic: {topic}
Problem type: {problem_type_label} ({internal})

Curriculum snippets (ground truth):
{curriculum_snippets}
{profile_bits}

Constraints (important for neurodivergent learners):
- One main goal only
- Step-by-step instructions (3-6 steps)
- Minimal story, but make the theme match the student's interests when possible
- Include starter code (3-10 lines) they can edit
- Do NOT include the full solution

Return in this exact format:
PROBLEM:
<markdown problem text>

STARTER_CODE:
```python
<starter code>
```"""

    text = complete_text(client=anthropic_client, prompt=prompt, max_tokens=700, temperature=0.3)
    if "STARTER_CODE:" in text:
        parts = text.split("STARTER_CODE:", 1)
        prob = parts[0].replace("PROBLEM:", "").strip()
        starter = parts[1].strip()
        # Try to extract code fence contents; fall back to raw.
        if "```" in starter:
            chunks = starter.split("```")
            if len(chunks) >= 2:
                # chunks[1] begins with optional language tag + newline
                code_block = chunks[1]
                code_lines = code_block.splitlines()
                if code_lines and code_lines[0].strip().lower().startswith("python"):
                    code_lines = code_lines[1:]
                return prob, "\n".join(code_lines).strip()
        return prob, starter

    return text.strip(), "# Write your code here..."

