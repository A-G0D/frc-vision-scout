"""
claude_client.py — minimal, conventions-baked Claude API client for FRC Vision Scout.

This is the shared entry point for any Claude use in the pipeline (overlay vision OCR,
scouting-report synthesis, etc.). It is intentionally thin: it standardizes the model,
thinking mode, streaming behavior, and error handling so feature modules don't each
re-decide them.

Conventions (per the project + Anthropic guidance):
  - Model: claude-opus-4-8 (DEFAULT_MODEL). Latest, most capable Opus-tier.
  - Adaptive thinking on by default for non-trivial calls: thinking={"type": "adaptive"}.
    (budget_tokens is removed on Opus 4.8 — adaptive is the only thinking mode.)
  - Effort via output_config (default "high"); not a top-level param.
  - Streaming for anything with large input/output or high max_tokens — avoids HTTP
    timeouts. We use stream + .get_final_message() so callers get one complete Message.
  - API key from the environment (ANTHROPIC_API_KEY); never hardcode.
  - Typed exceptions, caught most-specific-first.

Vision note: Claude reads images natively, which is why this client is the basis for the
overlay reader — feed it base64 image blocks (the upscaled overlay crops from
scan_timeline.dump_roi_crops) and ask for structured JSON. See read_image_json().
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import anthropic

DEFAULT_MODEL = "claude-opus-4-8"
DEFAULT_MAX_TOKENS = 16_000          # non-streaming default (keeps under HTTP timeout)
DEFAULT_STREAM_MAX_TOKENS = 64_000   # streaming default — room to think + answer


def get_client() -> anthropic.Anthropic:
    """Construct the SDK client. Resolves ANTHROPIC_API_KEY from the environment."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it before using the Claude client:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-..."
        )
    return anthropic.Anthropic()


def complete(
    prompt: str,
    *,
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    thinking: bool = True,
    effort: str = "high",
    max_tokens: int = DEFAULT_STREAM_MAX_TOKENS,
    stream: bool = True,
) -> anthropic.types.Message:
    """
    One-shot text completion with the project's defaults. Returns the full Message.

    `thinking=True` enables adaptive thinking; `effort` is one of low|medium|high|max.
    Streaming is on by default and we return the accumulated final message, so callers
    don't have to handle individual events unless they want token-level output.
    """
    client = get_client()
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    if thinking:
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["output_config"] = {"effort": effort}

    try:
        if stream:
            with client.messages.stream(**kwargs) as s:
                return s.get_final_message()
        return client.messages.create(**kwargs)
    except anthropic.AuthenticationError:
        raise RuntimeError("Claude auth failed — check ANTHROPIC_API_KEY.")
    except anthropic.RateLimitError as e:
        retry = e.response.headers.get("retry-after", "?") if e.response else "?"
        raise RuntimeError(f"Claude rate limited (retry-after={retry}s).") from e
    except anthropic.APIStatusError as e:
        raise RuntimeError(f"Claude API error {e.status_code}: {e.message}") from e


def read_image_json(
    image_paths: list[str | Path],
    instruction: str,
    schema: dict,
    *,
    model: str = DEFAULT_MODEL,
    media_type: str = "image/png",
) -> dict:
    """
    Vision → structured JSON. Send one or more local images plus an instruction, and get
    back an object validated against `schema` (JSON Schema) via output_config.format.

    This is the backbone for the overlay reader: pass the overlay ROI crops, ask for the
    team numbers / match label / fuel counts, and receive typed JSON. Small crops keep the
    token cost low. messages.parse() would also work with a Pydantic model; we use a raw
    JSON schema here so the schema can be defined wherever the caller lives.
    """
    client = get_client()
    content: list[dict[str, Any]] = []
    for p in image_paths:
        data = base64.standard_b64encode(Path(p).read_bytes()).decode("utf-8")
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        })
    content.append({"type": "text", "text": instruction})

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=DEFAULT_MAX_TOKENS,
            messages=[{"role": "user", "content": content}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
    except anthropic.APIStatusError as e:
        raise RuntimeError(f"Claude vision call failed ({e.status_code}): {e.message}") from e

    import json
    text = next((b.text for b in resp.content if b.type == "text"), "")
    return json.loads(text)


if __name__ == "__main__":
    # Smoke test — verifies SDK import, key presence, and a live round-trip.
    print(f"anthropic SDK: {anthropic.__version__}")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set — skipping live call. Set it and re-run to verify.")
        raise SystemExit(0)
    msg = complete(
        "Reply with exactly: FRC Vision Scout Claude client OK",
        thinking=False, stream=False, max_tokens=64,
    )
    text = next((b.text for b in msg.content if b.type == "text"), "")
    print(f"model={msg.model}  stop={msg.stop_reason}")
    print(f"reply: {text}")
    print(f"tokens: in={msg.usage.input_tokens} out={msg.usage.output_tokens}")
