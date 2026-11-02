from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class AppPaths:
    root_dir: Path
    slides_dir: Path
    chroma_dir: Path
    data_dir: Path
    profiles_dir: Path


def get_paths() -> AppPaths:
    root = Path(__file__).resolve().parent  # project folder (files are flat, not in app/)
    data_dir = root / "data"
    return AppPaths(
        root_dir=root,
        slides_dir=root / "Slides",
        chroma_dir=root / "chroma_db",
        data_dir=data_dir,
        profiles_dir=data_dir / "profiles",
    )


def get_anthropic_key() -> str | None:
    # Support both names during transition; prefer ANTHROPIC_KEY.
    return os.environ.get("ANTHROPIC_KEY") or os.environ.get("ANTHROPIC_API_KEY")


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_ANTHROPIC_MODEL_FAST = "claude-3-5-sonnet-20241022"

