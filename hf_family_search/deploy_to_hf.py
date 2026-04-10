#!/usr/bin/env python3
"""
Push this folder to your Hugging Face Space (no GitHub). Uses the Hub API + token.

Default: watch + auto-deploy (debounced). Watches hf_family_search/ recursively for .py and
Space config files; also watches ../family_search_aws.py and copies it here when it changes.

  HF_TOKEN=hf_...  HF_SPACE_REPO_ID=yourname/your-space
  python deploy_to_hf.py              # deploy once at start, then on every relevant save
  python deploy_to_hf.py --once       # single push, exit
  python deploy_to_hf.py --no-initial # watch only (no deploy until first file change)

Token / repo id: ../.env or ./.env (not committed). DEPLOY_INCLUDE_INDEX=1 uploads the JSON index.
Optional HF_DEPLOY_HOT_RELOAD=1 enables commit hot_reload=1 (default off).

Hub quirk: GET /api/spaces/{id} can 400 when compute.replicaStatuses contains null. This script uses
/api/spaces/{id}/refs for existence checks and recovers from that 400 on no-op commits.

Watch mode defaults to polling (reliable when Cursor/VS Code save via rename). Override with
HF_DEPLOY_NATIVE_FS_EVENTS=1 for native FSEvents/inotify. HF_DEPLOY_POLL_INTERVAL=0.8 (seconds).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import threading
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_PLAY_ROOT = _ROOT.parent
_UPSTREAM_AWS = _PLAY_ROOT / "family_search_aws.py"
_LOCAL_AWS = _ROOT / "family_search_aws.py"

# Committed to the Space
_DEFAULT_DEPLOY_FILES = (
    "app.py",
    "family_search_aws.py",
    "requirements.txt",
    "README.md",
    ".gitignore",
)

_EXTRA_NAMES = frozenset({"requirements.txt", "README.md", ".gitignore"})


def _load_env_files() -> None:
    for env_path in (_ROOT / ".env", _ROOT.parent / ".env"):
        if not env_path.is_file():
            continue
        try:
            text = env_path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


def _repo_id() -> str:
    rid = (os.environ.get("HF_SPACE_REPO_ID") or os.environ.get("HF_SPACE_ID") or "").strip()
    if not rid:
        print(
            "Set HF_SPACE_REPO_ID (e.g. username/my-space) in the environment or .env",
            file=sys.stderr,
        )
        sys.exit(1)
    return rid


def _ensure_space_exists(api: object, repo_id: str) -> None:
    """
    Ensure the Space repo exists. Use list_repo_refs (GET .../refs) instead of repo_info
    (GET .../spaces/{id}) so we avoid Hub 400s on broken compute.replicaStatuses payloads.
    """
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        api.list_repo_refs(repo_id, repo_type="space")
        return
    except RepositoryNotFoundError:
        pass

    print(f"Space '{repo_id}' not on the Hub yet — creating (Gradio SDK)…", flush=True)
    api.create_repo(
        repo_id,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
    )


def _hub_replica_status_bug(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "replicastatuses" in msg
        or "hotreloading" in msg
        or "compute.spec" in msg
        or "compute.current.spec" in msg
    )


def _print_unchanged_ok(api: object, repo_id: str) -> None:
    """create_commit hits repo_info when all files are unchanged; that can 400 on buggy Spaces."""
    try:
        refs = api.list_repo_refs(repo_id, repo_type="space")
        sha = None
        for b in refs.branches:
            if b.name == "main":
                sha = b.target_commit
                break
        if sha is None and refs.branches:
            sha = refs.branches[0].target_commit
    except Exception:
        print(
            "Nothing to upload (local files match Hub). "
            "(Could not read refs — Space may still be fine.)",
            flush=True,
        )
        return
    print(
        f"Nothing to upload — local files already match Hub.\n"
        f"  https://huggingface.co/spaces/{repo_id}/commit/{sha}",
        flush=True,
    )


def deploy(commit_message: str | None = None) -> None:
    _load_env_files()
    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError:
        print("pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    repo_id = _repo_id()
    ops: list = []
    for name in _DEFAULT_DEPLOY_FILES:
        path = _ROOT / name
        if path.is_file():
            ops.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)))

    if os.environ.get("DEPLOY_INCLUDE_INDEX", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        idx = _ROOT / "family_search_index.json"
        if idx.is_file():
            ops.append(
                CommitOperationAdd(
                    path_in_repo="family_search_index.json",
                    path_or_fileobj=str(idx),
                )
            )

    if not ops:
        print("No files to upload.", file=sys.stderr)
        sys.exit(1)

    msg = commit_message or f"Deploy {time.strftime('%Y-%m-%d %H:%M:%S')}"
    api = HfApi()
    _ensure_space_exists(api, repo_id)

    # hot_reload=1 can trigger Hub bugs (400: replicaStatuses[..] null vs string). Full rebuild still runs on commit.
    hot = os.environ.get("HF_DEPLOY_HOT_RELOAD", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    from huggingface_hub.errors import BadRequestError

    try:
        api.create_commit(
            repo_id=repo_id,
            repo_type="space",
            operations=ops,
            commit_message=msg,
            _hot_reload=hot,
        )
    except BadRequestError as e:
        if _hub_replica_status_bug(e):
            _print_unchanged_ok(api, repo_id)
            return
        raise
    print(f"Pushed {len(ops)} file(s) to https://huggingface.co/spaces/{repo_id}")


def _path_triggers_deploy(abs_path: Path) -> bool:
    """True if this path under hf_family_search should trigger a Hub deploy."""
    try:
        rel = abs_path.resolve().relative_to(_ROOT.resolve())
    except ValueError:
        return False
    if any(p == "__pycache__" for p in rel.parts):
        return False
    name = abs_path.name
    if name == "deploy_to_hf.py":
        return False
    if name.startswith(".") and name != ".gitignore":
        return False
    if abs_path.suffix.lower() == ".py":
        return True
    if name in _EXTRA_NAMES:
        return True
    if (
        name == "family_search_index.json"
        and os.environ.get("DEPLOY_INCLUDE_INDEX", "").strip().lower()
        in ("1", "true", "yes", "on")
    ):
        return True
    return False


def _make_watchdog_observer():
    """
    Polling sees saves that use write-temp-then-rename (common in editors). Native events can miss those.
    Set HF_DEPLOY_NATIVE_FS_EVENTS=1 to use FSEvents/inotify instead.
    """
    native = os.environ.get("HF_DEPLOY_NATIVE_FS_EVENTS", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if native:
        from watchdog.observers import Observer

        return Observer(), "native FS events"
    from watchdog.observers.polling import PollingObserver

    raw = (os.environ.get("HF_DEPLOY_POLL_INTERVAL") or "0.8").strip()
    try:
        interval = max(0.25, float(raw))
    except ValueError:
        interval = 0.8
    return PollingObserver(timeout=interval), f"polling every {interval}s"


def _watch(debounce_s: float) -> None:
    try:
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        print("Watch mode needs: pip install watchdog", file=sys.stderr)
        sys.exit(1)

    def fire() -> None:
        try:
            deploy(commit_message="Auto-deploy (watch)")
        except Exception as e:
            print(f"Deploy failed: {e}", file=sys.stderr)

    timer: threading.Timer | None = None
    timer_lock = threading.Lock()

    def schedule(reason: str) -> None:
        nonlocal timer
        with timer_lock:
            if timer is not None:
                timer.cancel()
            timer = threading.Timer(debounce_s, fire)
            timer.daemon = True
            timer.start()
        print(f"{reason} → deploy in {debounce_s}s…", flush=True)

    class _SpaceDirHandler(FileSystemEventHandler):
        def on_modified(self, event):  # type: ignore[override]
            if event.is_directory:
                return
            p = Path(event.src_path)
            if _path_triggers_deploy(p):
                schedule(f"Change: {p.relative_to(_ROOT)}")

        def on_created(self, event):  # type: ignore[override]
            self.on_modified(event)

        def on_moved(self, event):  # type: ignore[override]
            if event.is_directory:
                return
            p = Path(event.dest_path)
            if _path_triggers_deploy(p):
                schedule(f"Moved into place: {p.relative_to(_ROOT)}")

    class _UpstreamAwsHandler(FileSystemEventHandler):
        def _maybe_sync_and_schedule(self, path: Path) -> None:
            if path.name != "family_search_aws.py":
                return
            try:
                shutil.copy2(_UPSTREAM_AWS, _LOCAL_AWS)
            except OSError as e:
                print(f"Sync family_search_aws.py failed: {e}", file=sys.stderr)
                return
            schedule("Synced play/family_search_aws.py → hf_family_search/")

        def on_modified(self, event):  # type: ignore[override]
            if event.is_directory:
                return
            self._maybe_sync_and_schedule(Path(event.src_path))

        def on_created(self, event):  # type: ignore[override]
            self.on_modified(event)

        def on_moved(self, event):  # type: ignore[override]
            if event.is_directory:
                return
            self._maybe_sync_and_schedule(Path(event.dest_path))

    obs, mode_note = _make_watchdog_observer()
    obs.schedule(_SpaceDirHandler(), str(_ROOT), recursive=True)
    if _UPSTREAM_AWS.is_file():
        obs.schedule(_UpstreamAwsHandler(), str(_PLAY_ROOT), recursive=False)

    obs.start()
    print(f"Watching {_ROOT} (Python + Space config files) — {mode_note}", flush=True)
    if _UPSTREAM_AWS.is_file():
        print(f"Also watching {_UPSTREAM_AWS} (syncs into hf_family_search/)")
    print("Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        obs.stop()
    obs.join()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Deploy hf_family_search to a Hugging Face Space (default: watch & auto-deploy)"
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Deploy a single time and exit",
    )
    p.add_argument(
        "--no-initial",
        action="store_true",
        help="When watching, skip the first deploy on startup",
    )
    p.add_argument(
        "--debounce",
        type=float,
        default=2.0,
        help="Seconds after last change before upload (default: 2)",
    )
    args = p.parse_args()
    _load_env_files()
    debounce = max(0.5, args.debounce)

    if args.once:
        deploy()
        return

    if not args.no_initial:
        print("Initial deploy…")
        try:
            deploy(commit_message="Auto-deploy (startup)")
        except Exception as e:
            print(f"Initial deploy failed: {e}", file=sys.stderr)

    _watch(debounce)


if __name__ == "__main__":
    main()
