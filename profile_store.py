from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StudentProfile:
    student_id: str
    raw_text: str
    summary: str
    interests: list[str]
    dislikes: list[str]
    preferred_style: list[str]


def _safe_id(student_id: str) -> str:
    keep = []
    for ch in (student_id or "").strip():
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")
    out = "".join(keep).strip("_")
    return out or "student"


def profile_path(profiles_dir: Path, student_id: str) -> Path:
    return profiles_dir / f"{_safe_id(student_id)}.json"


def save_profile(profiles_dir: Path, profile: StudentProfile) -> Path:
    profiles_dir.mkdir(parents=True, exist_ok=True)
    path = profile_path(profiles_dir, profile.student_id)
    payload: dict[str, Any] = {
        "student_id": profile.student_id,
        "raw_text": profile.raw_text,
        "summary": profile.summary,
        "interests": profile.interests,
        "dislikes": profile.dislikes,
        "preferred_style": profile.preferred_style,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_profile(profiles_dir: Path, student_id: str) -> StudentProfile | None:
    path = profile_path(profiles_dir, student_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return StudentProfile(
        student_id=str(data.get("student_id", student_id)),
        raw_text=str(data.get("raw_text", "")),
        summary=str(data.get("summary", "")),
        interests=list(data.get("interests", []) or []),
        dislikes=list(data.get("dislikes", []) or []),
        preferred_style=list(data.get("preferred_style", []) or []),
    )

