"""Gradio chat UI for MiniMax via NVIDIA's OpenAI-compatible API."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
_ROOT = _SCRIPT.parents[1]
_VENV = _ROOT / ".venv" / "bin" / "python"


def _maybe_reexec_with_venv() -> None:
    if not _VENV.is_file():
        return
    if Path(sys.executable).resolve() == _VENV.resolve():
        return
    if importlib.util.find_spec("openai") and importlib.util.find_spec("gradio"):
        return
    os.execv(str(_VENV), [str(_VENV), str(_SCRIPT), *sys.argv[1:]])


_maybe_reexec_with_venv()

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(_ROOT / ".env")

_BASE_URL = "https://integrate.api.nvidia.com/v1"
_MODEL = "minimaxai/minimax-m2.7"


def _openai_client() -> OpenAI:
    return OpenAI(
        base_url=_BASE_URL,
        api_key=os.environ["NVIDIA_API_KEY"],
    )


def _api_messages(history: list[dict], user_message: str) -> list[dict]:
    msgs: list[dict] = []
    for m in history:
        role, content = m.get("role"), m.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": user_message})
    return msgs


def respond(message: str, history: list[dict]):
    """Gradio 6 passes ``history`` as OpenAI-style ``{"role","content"}`` dicts."""
    if not (message and message.strip()):
        yield ""
        return

    msgs = _api_messages(history, message)
    stream = _openai_client().chat.completions.create(
        model=_MODEL,
        messages=msgs,
        temperature=1,
        top_p=0.95,
        max_tokens=4096,
        stream=True,
    )
    acc = ""
    for chunk in stream:
        ch = chunk.choices
        if not ch:
            continue
        piece = ch[0].delta.content
        if piece:
            acc += piece
            yield acc


def main() -> None:
    demo = gr.ChatInterface(
        respond,
        title="MiniMax (NVIDIA API)",
        description=(
            "Chat using `minimaxai/minimax-m2.7` via integrate.api.nvidia.com. "
            "Set `NVIDIA_API_KEY` in the repo root `.env`."
        ),
        examples=[
            "What model are you? Reply with the model name only.",
            "Explain what a tensor is in two sentences.",
        ],
        cache_examples=False,
    )
    demo.queue().launch()


if __name__ == "__main__":
    main()
