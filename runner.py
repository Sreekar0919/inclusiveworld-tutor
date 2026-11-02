from __future__ import annotations

import contextlib
import traceback
from dataclasses import dataclass
from io import StringIO


@dataclass(frozen=True)
class RunResult:
    ok: bool
    stdout_stderr: str
    exception_type: str | None
    exception_message: str | None
    traceback_text: str | None


def run_student_code(code: str) -> RunResult:
    """
    Executes code with captured stdout/stderr and returns structured error info.
    Note: this is not a sandbox. It's a teaching tool for trusted environments.
    """
    out = StringIO()
    local_vars: dict = {}
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            exec(code, {"__builtins__": __builtins__}, local_vars)
        text = out.getvalue()
        if not text.strip():
            text = "# Code executed successfully (no output)."
        return RunResult(
            ok=True,
            stdout_stderr=text,
            exception_type=None,
            exception_message=None,
            traceback_text=None,
        )
    except Exception as e:
        text = out.getvalue()
        tb = traceback.format_exc()
        if text.strip():
            combined = f"{text}\n\n{tb}"
        else:
            combined = tb
        return RunResult(
            ok=False,
            stdout_stderr=combined.strip(),
            exception_type=type(e).__name__,
            exception_message=str(e),
            traceback_text=tb.strip(),
        )

