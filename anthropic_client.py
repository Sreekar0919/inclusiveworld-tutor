from __future__ import annotations

import anthropic

from settings import get_anthropic_key, DEFAULT_ANTHROPIC_MODEL_FAST


def get_client() -> anthropic.Anthropic | None:
    key = get_anthropic_key()
    if not key:
        return None
    try:
        return anthropic.Anthropic(api_key=key)
    except Exception:
        return None


def complete_text(
    *,
    client: anthropic.Anthropic,
    prompt: str,
    max_tokens: int = 800,
    temperature: float = 0.4,
    model: str = DEFAULT_ANTHROPIC_MODEL_FAST,
) -> str:
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()

