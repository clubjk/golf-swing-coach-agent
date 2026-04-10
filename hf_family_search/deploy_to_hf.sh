#!/usr/bin/env bash
# Uses play/.venv Python (avoids broken Homebrew /usr/local/bin/python3).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$DIR/../.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "No venv at $PY" >&2
  echo "Create it from the play/ folder: cd \"$(cd "$DIR/.." && pwd)\" && python3 -m venv .venv && .venv/bin/pip install -r hf_family_search/requirements.txt huggingface_hub watchdog" >&2
  exit 1
fi
exec "$PY" "$DIR/deploy_to_hf.py" "$@"
