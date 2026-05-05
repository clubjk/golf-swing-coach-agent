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
    if importlib.util.find_spec("openai") is not None:
        return
    os.execv(str(_VENV), [str(_VENV), str(_SCRIPT), *sys.argv[1:]])


_maybe_reexec_with_venv()

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(_ROOT / ".env")


def main() -> None:
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ["NVIDIA_API_KEY"],
    )
    stream = client.chat.completions.create(
        model="minimaxai/minimax-m2.7",
        messages=[
            {
                "role": "user",
                "content": "what model are you using? answer only with the model name.",
            }
        ],
        temperature=1,
        top_p=0.95,
        max_tokens=256,
        stream=True,
    )
    out = sys.stdout.write
    for chunk in stream:
        choices = chunk.choices
        if not choices:
            continue
        content = choices[0].delta.content
        if content:
            out(content)


if __name__ == "__main__":
    main()
