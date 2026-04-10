#!/usr/bin/env python3
"""
Family Photo & Video Search
----------------------------
Indexes images and videos from an Apple Photos album using Claude's
vision AI, then lets you search them with natural language.

Usage:
  python family_search.py index                              # index everything
  python family_search.py index --album "My Album"           # different album
  python family_search.py index --type videos                # videos only
  python family_search.py index --type images                # images only
  python family_search.py index --probe                      # count local vs iCloud-only (no API)
  python family_search.py index-folder "/path/to/exports"    # index files on disk (e.g. external drive)
  python family_search.py search "boy eating pizza"            # search the index
  python family_search.py search "beach" --type videos         # only among indexed videos
  python family_search.py search "cake" --no-seek            # open videos from 0:00
  python family_search.py list                               # list all indexed items
  python family_search.py dedup                              # find & remove duplicates
  python family_search.py dedup --exact-only                 # only exact file matches
  python family_search.py dedup-folder "/path/to/folder"      # dedupe files on disk (e.g. external drive)
  python family_search.py dedup-numbered "/path/to/folder"    # .jpg/.jpeg/.mov; skips __duplicates_review__
  python family_search.py dedup-numbered "/path" --without-original  # also move clip(1).mov if clip.mov missing
  python family_search.py search "query" --top 20              # cap results; Haiku batches + Opus rerank (defaults)
  python family_search.py search "query" --no-rerank           # Haiku only (cheaper; merge scores, still --top)
  python family_search.py show "/path/to/photo.jpg"              # open in Preview / QuickTime (macOS)
  python family_search.py show --id "file:..."                 # open by index id
  python family_search.py serve --port 8765                    # browse indexed files at http://127.0.0.1:8765/

On macOS, grant Full Disk Access to your terminal app (System Settings → Privacy & Security)
so the Photos library database can be read.

The index JSON defaults to family_search_index.json next to this script. To put it on an external
volume, set FAMILY_SEARCH_INDEX_FILE in .env to a full path to the JSON file, or to a folder
(then the file used is that folder / family_search_index.json).

iCloud originals download into the Photos library bundle, not an arbitrary folder. To use an
external drive, either move the whole library in Photos → Settings → General, or export
originals into a folder and run index-folder on that path.

Search models (env, optional): FAMILY_SEARCH_SEARCH_BATCH_MODEL (default Haiku),
FAMILY_SEARCH_SEARCH_RERANK_MODEL (default Opus). Batches find candidates; one rerank
pass picks the best --top matches. serve binds 127.0.0.1 only.

Search speed (env): FAMILY_SEARCH_SEARCH_BATCH_JSON_CHARS (default 520000),
FAMILY_SEARCH_SEARCH_PARALLEL_BATCHES (default 8, 0=sequential),
FAMILY_SEARCH_SEARCH_PREFILTER_MAX (default 450; 0=off, searches whole index),
FAMILY_SEARCH_SEARCH_RERANK_THINKING=1 enables extended thinking on rerank (slower).
CLI: search --no-prefilter / --prefilter-max N.
"""

import html
import math
import os
import re
import sys
import json
import time
import shutil
import hashlib
import base64
import subprocess
import tempfile
import argparse
import mimetypes
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse
from datetime import datetime
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

_SCRIPT_DIR = Path(__file__).resolve().parent

def load_local_env() -> None:
    """
    Load KEY=VALUE pairs into os.environ (later files override earlier).

    Reads parent/.env then script_dir/.env so a copy under hf_family_search/ still picks up
    play/.env (ANTHROPIC_API_KEY, etc.). Shell exports still apply for keys not set here.
    """
    candidates = [_SCRIPT_DIR.parent / ".env", _SCRIPT_DIR / ".env"]
    seen_resolved: set[Path] = set()
    for env_path in candidates:
        try:
            if not env_path.is_file():
                continue
            resolved = env_path.resolve()
            if resolved in seen_resolved:
                continue
            seen_resolved.add(resolved)
            for raw in env_path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                if key.startswith("export "):
                    key = key[len("export "):].strip()
                value = value.split(" #", 1)[0].strip().strip('"').strip("'")
                if key:
                    os.environ[key] = value
        except OSError:
            pass

load_local_env()

try:
    import anthropic
except ModuleNotFoundError:
    _root = Path(__file__).resolve().parent
    _venv_dir = _root / ".venv"
    _venv_py = _venv_dir / "bin" / "python"
    # venv's python often resolves to the same real binary as Homebrew's; use sys.prefix.
    _in_venv = Path(sys.prefix).resolve() == _venv_dir.resolve()
    if _venv_py.is_file() and not _in_venv:
        os.execv(str(_venv_py), [str(_venv_py), *sys.argv])
    print(
        "Missing package 'anthropic'. From this directory run:\n"
        "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt",
        file=sys.stderr,
    )
    sys.exit(1)

def _resolve_index_file_path(override: str) -> Path:
    """If override is an existing directory, use family_search_index.json inside it."""
    p = Path(override).expanduser().resolve()
    if p.is_dir():
        return p / "family_search_index.json"
    return p


_index_override = os.environ.get("FAMILY_SEARCH_INDEX_FILE", "").strip()
INDEX_FILE = (
    _resolve_index_file_path(_index_override)
    if _index_override
    else _SCRIPT_DIR / "family_search_index.json"
)
DEFAULT_ALBUM = "Family Photo Grabbag"
FRAMES_PER_VIDEO = 6      # frames to sample per video
BATCH_SAVE_EVERY = 5      # save progress every N items
IMAGE_MAX_PX = 768         # resize dimension for images sent to Claude
INDEX_MODEL = "claude-haiku-4-5"   # model for indexing (cost-effective)


def _env_model(name: str, default: str) -> str:
    v = os.environ.get(name, "").strip()
    return v if v else default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


# Search: cheap model scans each batch; rerank model picks final --top (set --no-rerank to skip).
SEARCH_BATCH_MODEL = _env_model(
    "FAMILY_SEARCH_SEARCH_BATCH_MODEL", "claude-haiku-4-5"
)
SEARCH_RERANK_MODEL = _env_model(
    "FAMILY_SEARCH_SEARCH_RERANK_MODEL", "claude-opus-4-6"
)
# Larger batches ⇒ fewer API round-trips (~520k chars ≈ 130k tokens payload; under 1M context cap).
SEARCH_BATCH_JSON_CHARS = max(80_000, _env_int("FAMILY_SEARCH_SEARCH_BATCH_JSON_CHARS", 520_000))
# Parallel Haiku batch calls (0 or 1 = sequential). Default 8 for wall-clock speed.
SEARCH_PARALLEL_BATCHES = max(0, _env_int("FAMILY_SEARCH_SEARCH_PARALLEL_BATCHES", 8))
# Lexical prefilter: only send top N summaries to Claude (0 = off). Default 450 for ~20s UI latency.
SEARCH_PREFILTER_MAX = max(0, _env_int("FAMILY_SEARCH_SEARCH_PREFILTER_MAX", 450))
# Extended thinking on rerank (slower). Default off; set FAMILY_SEARCH_SEARCH_RERANK_THINKING=1 to enable.
SEARCH_RERANK_THINKING = os.environ.get(
    "FAMILY_SEARCH_SEARCH_RERANK_THINKING", ""
).strip().lower() in ("1", "true", "yes", "on")
# Max candidates passed to a single rerank call (after merging batch hits).
SEARCH_RERANK_POOL_MAX = 100

_PHOTOS_ACCESS_ERR_HELP = """\
Cannot read the Photos library — macOS denied access (Operation not permitted).

Fix:
  1. Open System Settings → Privacy & Security → Full Disk Access.
  2. Turn ON the app that runs this script (Terminal, iTerm2, Cursor, etc.).
     Adding only “python3” is not enough; grant the parent app.
  3. Quit that app completely, reopen it, and run this command again.

If Photos is open and you still see database errors, quit Photos and retry."""


def open_photos_db(osxphotos):
    """PhotosDB() with a clear message when macOS blocks library access."""
    try:
        return osxphotos.PhotosDB()
    except OSError as e:
        msg = str(e).lower()
        if (
            getattr(e, "errno", None) == 1
            or "operation not permitted" in msg
            or "error copying" in msg
        ):
            print(_PHOTOS_ACCESS_ERR_HELP, file=sys.stderr)
            sys.exit(1)
        raise


# ── Helpers ────────────────────────────────────────────────────────────────────

def local_photo_path(photo) -> str | None:
    """Path to on-disk original (or edited) if it exists; else None (e.g. iCloud placeholder)."""
    path = photo.path
    if path and os.path.exists(path):
        return path
    path = photo.path_edited or photo.path_raw
    if path and os.path.exists(path):
        return path
    return None


def get_video_duration(video_path: str) -> float:
    """Return video duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0


def extract_frames_as_b64(video_path: str, num_frames: int = FRAMES_PER_VIDEO) -> list[dict]:
    """
    Extract evenly-spaced frames from a video.
    Returns list of dicts: {'b64': ..., 'timestamp': seconds}.
    """
    duration = get_video_duration(video_path)
    if duration <= 0:
        print(f"  Warning: could not read duration for {video_path}")
        duration = 60.0

    margin = duration * 0.05
    usable = duration - 2 * margin
    if usable <= 0:
        usable = duration
        margin = 0

    timestamps = [
        margin + usable * i / max(num_frames - 1, 1)
        for i in range(num_frames)
    ]

    frames = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, ts in enumerate(timestamps):
            out_path = os.path.join(tmpdir, f"frame_{i:04d}.jpg")
            result = subprocess.run(
                [
                    "ffmpeg", "-ss", str(ts),
                    "-i", video_path,
                    "-vframes", "1",
                    "-q:v", "3",
                    "-vf", f"scale={IMAGE_MAX_PX}:-2",
                    "-y", out_path,
                ],
                capture_output=True,
            )
            if result.returncode == 0 and os.path.exists(out_path):
                with open(out_path, "rb") as f:
                    frames.append({
                        "b64": base64.standard_b64encode(f.read()).decode(),
                        "timestamp": ts,
                    })

    return frames


def image_to_b64(image_path: str) -> tuple[str, str]:
    """
    Read an image and return (base64_data, media_type).
    Resizes large images using ffmpeg for consistency.
    """
    ext = Path(image_path).suffix.lower()
    media_types = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".heic": "image/jpeg",  # convert HEIC to JPEG
        ".gif": "image/gif", ".webp": "image/webp",
        ".tiff": "image/jpeg", ".tif": "image/jpeg",
        ".bmp": "image/jpeg",
    }
    media_type = media_types.get(ext, "image/jpeg")
    needs_convert = ext in (".heic", ".tiff", ".tif", ".bmp")

    if needs_convert:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            subprocess.run(
                [
                    "ffmpeg", "-i", image_path,
                    "-vf", f"scale={IMAGE_MAX_PX}:-2",
                    "-q:v", "3", "-y", tmp_path,
                ],
                capture_output=True,
            )
            with open(tmp_path, "rb") as f:
                return base64.standard_b64encode(f.read()).decode(), "image/jpeg"
        finally:
            os.unlink(tmp_path)
    else:
        # Resize with ffmpeg to keep things consistent and small
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-i", image_path,
                    "-vf", f"scale='min({IMAGE_MAX_PX},iw)':-2",
                    "-q:v", "3", "-y", tmp_path,
                ],
                capture_output=True,
            )
            if result.returncode == 0:
                with open(tmp_path, "rb") as f:
                    return base64.standard_b64encode(f.read()).decode(), "image/jpeg"
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        # Fallback: read original file
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode(), media_type


def snap_to_nearest_sample(requested: float | None, samples: list[float]) -> float:
    """Pick the sample timestamp closest to what the model requested (seconds)."""
    if not samples:
        try:
            return float(requested or 0)
        except (TypeError, ValueError):
            return 0.0
    try:
        r = float(requested)
    except (TypeError, ValueError):
        r = float(samples[0])
    return float(min(samples, key=lambda x: abs(float(x) - r)))


def open_quicktime_at(path: str, start_sec: float) -> bool:
    """Open a video in QuickTime Player and jump to start_sec (then play)."""
    try:
        subprocess.run(["open", "-a", "QuickTime Player", path], check=False)
        time.sleep(1.0)
        script = f"""
tell application "QuickTime Player"
    activate
    if (count of documents) < 1 then return
    set d to front document
    set ts to time scale of d
    set current time of d to (round ({start_sec} * ts))
    play d
end tell
"""
        r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        return r.returncode == 0
    except OSError:
        return False


def format_duration(seconds: float) -> str:
    """Format seconds as H:MM:SS or M:SS."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


# ── Index ──────────────────────────────────────────────────────────────────────

def load_index() -> dict:
    if INDEX_FILE.exists():
        with open(INDEX_FILE) as f:
            return json.load(f)
    return {"indexed_at": None, "items": {}}


def save_index(index: dict) -> None:
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_FILE, "w") as f:
        json.dump(index, f, indent=2)


_MEDIA_IMAGE_EXT = frozenset({
    ".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".gif", ".tif", ".tiff",
})
_MEDIA_VIDEO_EXT = frozenset({".mp4", ".mov", ".m4v", ".avi", ".mkv"})

# e.g. IMG_0066(1).JPG — duplicate of IMG_0066.JPG in the same folder
_NUMBERED_COPY_NAME_RE = re.compile(
    r"^(?P<base>.+?)\s*\((?P<num>\d+)\)(?P<ext>\.[^./]+)$",
    re.IGNORECASE,
)
# dedup-numbered only considers these extensions (jpg/jpeg are interchangeable for finding the original)
_NUMBERED_DEDUP_EXTENSIONS = frozenset({".jpg", ".jpeg", ".mov"})


def _path_under_duplicates_review(path: Path) -> bool:
    return any(part == "__duplicates_review__" for part in path.parts)


def _index_item_under_duplicates_review(item: dict) -> bool:
    """True if the indexed row's path lives under a __duplicates_review__ folder."""
    p = item.get("path")
    if not p:
        return False
    try:
        return _path_under_duplicates_review(Path(str(p)))
    except (TypeError, ValueError, OSError):
        norm = str(p).replace("\\", "/").strip("/")
        return "/__duplicates_review__/" in f"/{norm}/"


def _search_skip_duplicates_review(include_duplicates_review: bool | None) -> bool:
    """When True, exclude __duplicates_review__/ paths from search (default)."""
    if include_duplicates_review is True:
        return False
    if include_duplicates_review is False:
        return True
    return os.environ.get("FAMILY_SEARCH_INCLUDE_DUPLICATES_REVIEW", "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    )


def _original_for_numbered_copy(path: Path, base: str, dup_ext: str) -> Path | None:
    """Return same-folder original if dup_ext is .jpg/.jpeg/.mov and a matching base file exists."""
    base = base.strip()
    ext_lower = dup_ext.lower()
    if ext_lower not in _NUMBERED_DEDUP_EXTENSIONS:
        return None
    parent = path.parent
    candidates: list[Path] = []
    seen: set[Path] = set()

    def add(p: Path) -> None:
        try:
            key = p.resolve()
        except OSError:
            key = p
        if key not in seen:
            seen.add(key)
            candidates.append(p)

    add(parent / f"{base}{dup_ext}")
    if ext_lower in (".jpg", ".jpeg"):
        for suf in (".jpg", ".jpeg", ".JPG", ".JPEG", ".Jpg", ".Jpeg"):
            add(parent / f"{base}{suf}")
    else:
        for suf in (".mov", ".MOV", ".Mov"):
            add(parent / f"{base}{suf}")

    for c in candidates:
        if c.is_file():
            return c
    return None


def iter_folder_media(
    root: Path, include_images: bool, include_videos: bool
) -> list[tuple[Path, str]]:
    """Recursively list image/video files under root."""
    out: list[tuple[Path, str]] = []
    root = root.resolve()
    if not root.is_dir():
        return out
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if include_images and ext in _MEDIA_IMAGE_EXT:
            out.append((p, "image"))
        elif include_videos and ext in _MEDIA_VIDEO_EXT:
            out.append((p, "video"))
    out.sort(key=lambda x: str(x[0]).lower())
    return out


def file_item_id(path: Path) -> str:
    """Stable id for a file path (separate namespace from Photos UUIDs)."""
    return "file:" + hashlib.sha256(str(path.resolve()).encode()).hexdigest()


def file_creation_date_label(path: Path) -> str:
    try:
        st = path.stat()
        ts = getattr(st, "st_birthtime", None) or st.st_mtime
        return datetime.fromtimestamp(ts).date().isoformat()
    except OSError:
        return "unknown date"


ANALYSIS_PROMPT = (
    "Please describe what is happening in this {media_type} in detail. Focus on:\n"
    "- Who appears (ages, descriptions — e.g. 'toddler boy, ~18 months')\n"
    "- What they are doing (activities, actions)\n"
    "- What they are eating or drinking, if anything\n"
    "- What is visible on any TV/screen in the background\n"
    "- The setting (room, location, outdoors/indoors)\n"
    "- Any notable clothing (e.g. 'wearing only a diaper', 'in pajamas', 'halloween costume')\n"
    "- Any toys, objects, or notable items\n"
    '- If a diaper is visible on anyone (only a diaper, or peeking from clothing), include the tag '
    '"diaper" in the "tags" array.\n'
    '- If a baby bottle or sippy cup used for feeding an infant is visible, include the tag '
    '"baby bottle" in the "tags" array.\n'
    '- If a pacifier (soother/binky) is visible in a mouth or in hand, include the tag '
    '"pacifier" in the "tags" array.\n'
    '- If a horse (or pony) appears in the scene, include the tag "horse" in the "tags" array.\n'
    "- Also add these **exact** tags when clearly visible (photos or videos):\n"
    '  • "beach" — sand and ocean/lake shore, typical beach outing\n'
    '  • "pool" — swimming pool or people clearly swimming in a pool\n'
    '  • "playground" — swings, slides, or climbing structures at a park/playground\n'
    '  • "birthday" — birthday party, cake with candles, or obvious birthday celebration\n'
    '  • "wedding" — wedding attire, ceremony, or reception\n'
    '  • "dog" / "cat" — a dog or cat pet visible\n'
    '  • "stroller" — baby stroller visible\n'
    '  • "car seat" — child in a car seat inside a vehicle\n'
    '  • "bike" — bicycle riding or bicycle prominently in frame\n'
    '  • "snow" — snowy outdoor scene or people playing in snow\n'
    "(Use these exact tag strings so they are easy to search.)\n\n"
    "Then provide a JSON block at the end in this exact format:\n"
    "```json\n"
    "{{\n"
    '  "tags": ["tag1", "diaper", "pool", "birthday", "dog", "beach"],\n'
    '  "people": ["toddler boy ~18mo", "adult woman"],\n'
    '  "activities": ["eating", "watching TV"],\n'
    '  "objects": ["pizza", "high chair", "TV"],\n'
    '  "setting": "living room",\n'
    '  "on_screen": "Barney & Friends"\n'
    "}}\n"
    "```"
)


def parse_analysis_response(response) -> dict:
    """Extract description and structured JSON from Claude's response."""
    full_text = ""
    for block in response.content:
        if block.type == "text":
            full_text = block.text
            break

    structured = {
        "tags": [], "people": [], "activities": [],
        "objects": [], "setting": "", "on_screen": "",
    }
    if "```json" in full_text:
        try:
            json_str = full_text.split("```json")[1].split("```")[0].strip()
            structured = json.loads(json_str)
        except (json.JSONDecodeError, IndexError):
            pass

    description = full_text.split("```json")[0].strip() if "```json" in full_text else full_text.strip()

    return {"description": description, **structured}


def analyze_video(client: anthropic.Anthropic, video_path: str, creation_date: str, duration: float) -> dict:
    """Extract frames and analyze a video with Claude."""
    print(f"  Extracting {FRAMES_PER_VIDEO} frames...")
    frames = extract_frames_as_b64(video_path, FRAMES_PER_VIDEO)

    if not frames:
        return {"description": "Could not extract frames.", "tags": [], "people": [],
                "activities": [], "objects": [], "setting": "", "on_screen": "",
                "sample_times_sec": []}

    content = []
    for frame in frames:
        ts = format_duration(frame["timestamp"])
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": frame["b64"]},
        })
        content.append({"type": "text", "text": f"[Frame at {ts}]"})

    content.append({
        "type": "text",
        "text": (
            f"These are frames sampled from a home video. "
            f"The video is {format_duration(duration)} long, recorded on {creation_date}.\n\n"
            + ANALYSIS_PROMPT.format(media_type="video")
        ),
    })

    print(f"  Analyzing with Claude...")
    with client.messages.stream(
        model=INDEX_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": content}],
    ) as stream:
        response = stream.get_final_message()

    out = parse_analysis_response(response)
    out["sample_times_sec"] = [round(f["timestamp"], 3) for f in frames]
    return out


def analyze_image(client: anthropic.Anthropic, image_path: str, creation_date: str) -> dict:
    """Analyze a single image with Claude."""
    print(f"  Reading image...")
    b64_data, media_type = image_to_b64(image_path)

    content = [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": b64_data},
        },
        {
            "type": "text",
            "text": (
                f"This is a home photo taken on {creation_date}.\n\n"
                + ANALYSIS_PROMPT.format(media_type="image")
            ),
        },
    ]

    print(f"  Analyzing with Claude...")
    with client.messages.stream(
        model=INDEX_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": content}],
    ) as stream:
        response = stream.get_final_message()

    return parse_analysis_response(response)


def cmd_index_folder(args):
    """Index images/videos from a folder tree (e.g. exports on an external drive)."""
    root = Path(args.path).expanduser().resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    include_videos = args.type in ("all", "videos")
    include_images = args.type in ("all", "images")
    files = iter_folder_media(root, include_images, include_videos)
    if not files:
        print("No matching image or video files found under that path.")
        sys.exit(1)

    n_img = sum(1 for _, t in files if t == "image")
    n_vid = sum(1 for _, t in files if t == "video")
    print(f"Scanning {root}\nFound {len(files)} file(s): {n_img} image(s), {n_vid} video(s)")

    index = load_index()
    index["album"] = f"folder:{root}"
    client = anthropic.Anthropic()

    newly_indexed = 0
    skipped_already = 0

    for i, (file_path, media_type) in enumerate(files, 1):
        item_id = file_item_id(file_path)
        filename = file_path.name
        label = f"[{i}/{len(files)}] ({media_type.upper()}) {filename}"
        print(f"\n{label}")

        prev = index["items"].get(item_id)
        if prev and not args.reindex and prev.get("status") == "indexed":
            print("  Already indexed, skipping. (use --reindex to force)")
            skipped_already += 1
            continue

        creation_date = file_creation_date_label(file_path)
        path_str = str(file_path)

        try:
            if media_type == "video":
                duration = get_video_duration(path_str)
                print(f"  Duration: {format_duration(duration)}  |  Date: {creation_date}")
                analysis = analyze_video(client, path_str, creation_date, duration)
                index["items"][item_id] = {
                    "filename": filename,
                    "media_type": "video",
                    "path": path_str,
                    "creation_date": creation_date,
                    "duration": duration,
                    "status": "indexed",
                    "source": "folder",
                    **analysis,
                }
            else:
                print(f"  Date: {creation_date}")
                analysis = analyze_image(client, path_str, creation_date)
                index["items"][item_id] = {
                    "filename": filename,
                    "media_type": "image",
                    "path": path_str,
                    "creation_date": creation_date,
                    "duration": None,
                    "status": "indexed",
                    "source": "folder",
                    **analysis,
                }

            newly_indexed += 1
            print(f"  Done. Tags: {', '.join(analysis.get('tags', []))}")
        except Exception as e:
            print(f"  Error: {e}")
            index["items"][item_id] = {
                "filename": filename,
                "media_type": media_type,
                "path": path_str,
                "creation_date": creation_date,
                "status": "error",
                "error": str(e),
                "source": "folder",
                "description": "",
                "tags": [],
            }

        if newly_indexed % BATCH_SAVE_EVERY == 0 and newly_indexed > 0:
            index["indexed_at"] = datetime.now().isoformat()
            save_index(index)
            print("  (Progress saved)")

    index["indexed_at"] = datetime.now().isoformat()
    save_index(index)
    total = len(index["items"])
    print(
        f"\nDone! Indexed {newly_indexed} new item(s), "
        f"{skipped_already} already in index. "
        f"Total entries in index file: {total}"
    )
    print(f"Index saved to: {INDEX_FILE}")


def cmd_index(args):
    """Index images and/or videos from the Photos album."""
    try:
        import osxphotos
    except ImportError:
        print("Error: osxphotos not installed. Run: pip install osxphotos")
        sys.exit(1)

    album_name = args.album
    index_type = args.type  # "all", "videos", "images"

    print(f"Opening Photos library... (this may take a moment)")
    photosdb = open_photos_db(osxphotos)

    # Determine what to query
    include_videos = index_type in ("all", "videos")
    include_images = index_type in ("all", "images")

    all_results = []

    if include_videos:
        print(f"Looking for videos in album: '{album_name}'")
        videos = photosdb.query(
            osxphotos.QueryOptions(album=[album_name], movies=True, photos=False)
        )
        if videos:
            all_results.extend([(p, "video") for p in videos])

    if include_images:
        print(f"Looking for images in album: '{album_name}'")
        images = photosdb.query(
            osxphotos.QueryOptions(album=[album_name], movies=False, photos=True)
        )
        if images:
            all_results.extend([(p, "image") for p in images])

    if not all_results:
        # Try case-insensitive match
        all_albums = [a.title for a in photosdb.album_info]
        matches = [a for a in all_albums if a and a.lower() == album_name.lower()]
        if matches:
            album_name = matches[0]
            print(f"  Found album with different casing: '{album_name}', retrying...")
            if include_videos:
                videos = photosdb.query(
                    osxphotos.QueryOptions(album=[album_name], movies=True, photos=False)
                )
                if videos:
                    all_results.extend([(p, "video") for p in videos])
            if include_images:
                images = photosdb.query(
                    osxphotos.QueryOptions(album=[album_name], movies=False, photos=True)
                )
                if images:
                    all_results.extend([(p, "image") for p in images])

    if not all_results:
        print(f"\nNo items found in album '{album_name}'.")
        print("Available albums:")
        for a in sorted(set(a.title for a in photosdb.album_info if a.title)):
            print(f"  - {a}")
        sys.exit(1)

    n_videos = sum(1 for _, t in all_results if t == "video")
    n_images = sum(1 for _, t in all_results if t == "image")
    print(f"Found {n_videos} video(s) and {n_images} image(s) in '{album_name}'")

    if args.probe:
        n_local = sum(1 for p, _ in all_results if local_photo_path(p) is not None)
        n_total = len(all_results)
        print(
            f"\nProbe: {n_local}/{n_total} items have a file on disk "
            f"(ready to index without iCloud download)."
        )
        if n_local == 0 and n_total:
            print(
                "None are local yet. Photos → Settings → iCloud → "
                "Download Originals to this Mac, wait for progress to finish, then run:\n"
                f"  python {Path(sys.argv[0]).name} index --probe"
            )
        elif n_local < n_total:
            print(
                f"{n_total - n_local} still missing on disk; wait for downloads or run index "
                "to update entries as files appear."
            )
        return

    index = load_index()
    index["album"] = album_name
    client = anthropic.Anthropic()

    newly_indexed = 0
    skipped_already = 0
    skipped_unavailable = 0

    for i, (photo, media_type) in enumerate(all_results, 1):
        item_id = photo.uuid
        filename = photo.original_filename or f"{media_type}_{item_id}"

        label = f"[{i}/{len(all_results)}] ({media_type.upper()}) {filename}"
        print(f"\n{label}")

        prev = index["items"].get(item_id)
        if (
            prev
            and not args.reindex
            and prev.get("status") == "indexed"
        ):
            print(f"  Already indexed, skipping. (use --reindex to force)")
            skipped_already += 1
            continue

        # Get local file path (iCloud "Optimize Storage" = placeholder only until downloaded)
        file_path = local_photo_path(photo)
        if not file_path:
            print(
                "  Skipping: not on disk yet (iCloud placeholder). "
                "Photos → Settings → iCloud → Download Originals to this Mac; "
                "when downloads finish, run index again."
            )
            skipped_unavailable += 1
            index["items"][item_id] = {
                "filename": filename, "media_type": media_type, "path": None,
                "creation_date": str(photo.date) if photo.date else None,
                "status": "unavailable", "description": "Not available locally.",
                "tags": [],
            }
            continue

        creation_date = str(photo.date.date()) if photo.date else "unknown date"

        try:
            if media_type == "video":
                duration = get_video_duration(file_path)
                print(f"  Duration: {format_duration(duration)}  |  Date: {creation_date}")
                analysis = analyze_video(client, file_path, creation_date, duration)
                index["items"][item_id] = {
                    "filename": filename, "media_type": "video",
                    "path": file_path, "creation_date": creation_date,
                    "duration": duration, "status": "indexed",
                    **analysis,
                }
            else:
                print(f"  Date: {creation_date}")
                analysis = analyze_image(client, file_path, creation_date)
                index["items"][item_id] = {
                    "filename": filename, "media_type": "image",
                    "path": file_path, "creation_date": creation_date,
                    "duration": None, "status": "indexed",
                    **analysis,
                }

            newly_indexed += 1
            print(f"  Done. Tags: {', '.join(analysis.get('tags', []))}")
        except Exception as e:
            print(f"  Error: {e}")
            index["items"][item_id] = {
                "filename": filename, "media_type": media_type,
                "path": file_path, "creation_date": creation_date,
                "status": "error", "error": str(e),
                "description": "", "tags": [],
            }

        if newly_indexed % BATCH_SAVE_EVERY == 0 and newly_indexed > 0:
            index["indexed_at"] = datetime.now().isoformat()
            save_index(index)
            print(f"  (Progress saved)")

    index["indexed_at"] = datetime.now().isoformat()
    save_index(index)

    total = len(index["items"])
    print(
        f"\nDone! Indexed {newly_indexed} new item(s), "
        f"{skipped_already} already in index, "
        f"{skipped_unavailable} not downloaded locally. "
        f"Total entries in index file: {total}"
    )
    print(f"Index saved to: {INDEX_FILE}")


# ── Search ─────────────────────────────────────────────────────────────────────

def parse_search_matches_json(result_text: str) -> list:
    """Extract JSON array of matches from Claude's reply (fenced or raw)."""
    text = result_text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif text.startswith("```"):
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            raise json.JSONDecodeError("Could not parse matches as JSON array", text, start)
        if isinstance(data, list):
            return data
    raise json.JSONDecodeError("Could not parse matches as JSON array", text, 0)


def _minimal_search_record(uid: str, v: dict) -> dict:
    """Small dict for search prompts (full index rows can be huge)."""
    tags = v.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    tags_out = [str(t)[:60] for t in tags[:24]]
    rec: dict = {
        "id": uid,
        "type": v.get("media_type", "unknown"),
        "filename": (v.get("filename") or "")[:180],
        "date": (v.get("creation_date") or "")[:32],
        "description": (v.get("description") or "")[:380],
        "tags": tags_out,
    }
    if v.get("media_type") == "video":
        rec["duration_sec"] = v.get("duration")
        sts = v.get("sample_times_sec") or []
        if sts:
            rec["sample_starts_sec"] = sts[:24]
    return rec


def _chunk_search_records(
    records: list[dict], max_json_chars: int
) -> list[list[dict]]:
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    cur_len = 0
    for rec in records:
        piece = json.dumps(rec, separators=(",", ":"))
        extra = 1 if cur else 0
        if cur and cur_len + extra + len(piece) > max_json_chars:
            chunks.append(cur)
            cur = []
            cur_len = 0
        cur.append(rec)
        cur_len += extra + len(piece)
    if cur:
        chunks.append(cur)
    return chunks


def _match_sort_key(m: dict) -> tuple:
    """Higher tuple sorts later; we sort ascending then reverse."""
    sc = m.get("score")
    if isinstance(sc, (int, float)):
        s = float(sc)
    else:
        conf = str(m.get("confidence") or "").lower()
        s = {"high": 80.0, "medium": 50.0, "low": 20.0}.get(conf, 0.0)
    return (s, str(m.get("id") or ""))


def _search_stream_text(client, model: str, prompt: str) -> str:
    """Run a single search/reasoning call; Haiku omits extended thinking (unsupported / cheaper)."""
    kwargs: dict = {
        "model": model,
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": prompt}],
    }
    if "haiku" not in model.lower():
        kwargs["thinking"] = {"type": "adaptive"}
    with client.messages.stream(**kwargs) as stream:
        response = stream.get_final_message()
    parts: list[str] = []
    for block in response.content:
        if block.type == "text":
            parts.append(block.text)
    return "".join(parts)


def _messages_complete_text(
    client,
    model: str,
    prompt: str,
    *,
    extended_thinking: bool | None = None,
) -> str:
    """
    Non-streaming completion (lower latency than stream for batch work).
    If extended_thinking is None: use thinking only for non-Haiku models.
    """
    if extended_thinking is None:
        extended_thinking = "haiku" not in model.lower()
    kwargs: dict = {
        "model": model,
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": prompt}],
    }
    if extended_thinking:
        kwargs["thinking"] = {"type": "adaptive"}
    msg = client.messages.create(**kwargs)
    return "".join(
        b.text for b in msg.content if getattr(b, "type", None) == "text"
    )


def _prefilter_search_records(
    records: list[dict], query: str, max_keep: int
) -> tuple[list[dict], str]:
    """
    Keep rows whose description/filename/tags overlap query terms (cheap local filter).
    If no term matches any row, return the full list (do not drop everything).
    """
    if max_keep <= 0 or len(records) <= max_keep:
        return records, ""
    terms = [t.lower() for t in re.findall(r"[a-zA-Z0-9']+", query) if len(t) >= 2]
    if not terms:
        return records, ""
    scored: list[tuple[int, dict]] = []
    for rec in records:
        blob = " ".join(
            [
                str(rec.get("description", "")),
                str(rec.get("filename", "")),
                " ".join(str(x) for x in (rec.get("tags") or [])),
            ]
        ).lower()
        score = sum(blob.count(t) for t in terms)
        scored.append((score, rec))
    if max(s for s, _ in scored) == 0:
        return records, "Keyword prefilter: no overlap with query; searching full index."
    scored.sort(key=lambda x: (-x[0], str(x[1].get("id", ""))))
    kept = [r for _, r in scored[:max_keep]]
    return kept, f"Keyword prefilter: {len(kept)} of {len(records)} items sent to the model."


def _effective_prefilter_max(prefilter_max: int | None) -> int:
    if prefilter_max == 0:
        return 0
    if prefilter_max is not None:
        return prefilter_max
    return SEARCH_PREFILTER_MAX


def _build_batch_search_prompt(
    batch: list[dict],
    bi: int,
    n_batches: int,
    query: str,
    per_batch_cap: int,
) -> str:
    payload = json.dumps(batch, separators=(",", ":"))
    batch_note = (
        f"This is batch {bi} of {n_batches}. Only consider items whose \"id\" "
        f"appears in this JSON array (do not invent ids).\n\n"
    )
    return (
        f"I have a collection of home photos and videos indexed below.\n"
        f"{batch_note}"
        f"Search query:\nQUERY: {query}\n\n"
        f"Item summaries (JSON array):\n{payload}\n\n"
        f"Return a JSON array of items in THIS BATCH that match the query, "
        f"best match first, **at most {per_batch_cap}** entries (the strongest only; "
        f"fewer if only a few qualify). Each object must use an \"id\" from the array above. Format:\n"
        f'[\n  {{"id": "...", "type": "video|image", "reason": "...", '
        f'"confidence": "high|medium|low", "score": 85, "start_sec": 12.3}},\n  ...\n]\n'
        f'"score" is 0–100 for how well the item matches QUERY (use for ranking).\n'
        f"For VIDEOS: include start_sec — seconds into the clip to jump to. It MUST be exactly "
        f"one of the numbers listed in that item's sample_starts_sec (the times of frames used "
        f"when indexing). Pick the sample that best matches the QUERY. If sample_starts_sec is "
        f"missing or empty, use 0.\n"
        f"For IMAGES: omit start_sec or use null.\n"
        f"If no items in this batch match, return []. Only return the JSON, no other text."
    )


def _run_claude_batch_pass(
    batches: list[list[dict]],
    query: str,
    per_batch_cap: int,
    batch_model: str,
    log_print: bool,
) -> tuple[list[dict] | None, str | None]:
    """Run all batch Haiku/Sonnet calls (parallel when SEARCH_PARALLEL_BATCHES > 1)."""
    n_batches = len(batches)
    tasks: list[tuple[int, int, str]] = []
    for bi, batch in enumerate(batches, 1):
        prompt = _build_batch_search_prompt(
            batch, bi, n_batches, query, per_batch_cap
        )
        tasks.append((bi, len(batch), prompt))

    ext_think = False if "haiku" in batch_model.lower() else None

    if SEARCH_PARALLEL_BATCHES <= 1 or len(tasks) <= 1:
        client = anthropic.Anthropic()
        matches: list[dict] = []
        for bi, n_items, prompt in tasks:
            if log_print:
                if n_batches > 1:
                    print(
                        f"Asking {batch_model} (batch {bi}/{n_batches}, {n_items} item(s))..."
                    )
                else:
                    print(f"Asking {batch_model} to find matches...")
            txt = _messages_complete_text(
                client, batch_model, prompt, extended_thinking=ext_think
            )
            try:
                matches.extend(parse_search_matches_json(txt))
            except json.JSONDecodeError:
                return None, f"Could not parse model response for batch {bi}/{n_batches}."
        return matches, None

    def worker(tup: tuple[int, str]) -> tuple[int, str]:
        bi, prompt = tup
        c = anthropic.Anthropic()
        return bi, _messages_complete_text(
            c, batch_model, prompt, extended_thinking=ext_think
        )

    to_submit = [(bi, prompt) for bi, _n, prompt in tasks]
    n_workers = min(SEARCH_PARALLEL_BATCHES, len(to_submit))
    texts: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        future_map = {ex.submit(worker, t): t[0] for t in to_submit}
        for fut in as_completed(future_map):
            bi = future_map[fut]
            try:
                bi_done, txt = fut.result()
                texts[bi_done] = txt
            except Exception as e:
                return None, f"Batch {bi} API error: {e}"
    matches = []
    for bi in sorted(texts.keys()):
        try:
            matches.extend(parse_search_matches_json(texts[bi]))
        except json.JSONDecodeError:
            return None, f"Could not parse model response for batch {bi}/{n_batches}."
    if log_print:
        print(
            f"Asking {batch_model} ({n_batches} batch(es), up to {n_workers} parallel)..."
        )
    return matches, None


def run_search_query(
    query: str,
    *,
    search_media: str = "all",
    top: int = 20,
    no_rerank: bool = False,
    batch_model: str | None = None,
    rerank_model: str | None = None,
    log_print: bool = False,
    prog_name: str = "family_search",
    prefilter_max: int | None = None,
    include_duplicates_review: bool | None = None,
) -> tuple[list[dict] | None, dict[str, dict] | None, str | None]:
    """
    Run the same search as the CLI without printing or opening files.

    By default, rows whose path is under __duplicates_review__/ are excluded. Pass
    include_duplicates_review=True or set env FAMILY_SEARCH_INCLUDE_DUPLICATES_REVIEW=1 to include them.

    Returns (matches, items, error). On success error is None; matches may be [].
    On failure (missing index, no indexed media, bad args, API parse error), matches and items are None.
    """
    if not INDEX_FILE.exists():
        return None, None, f"No index file at {INDEX_FILE}. Run index / index-folder first."

    index = load_index()
    items = index.get("items", {})
    indexed = {k: v for k, v in items.items() if v.get("status") == "indexed"}
    n_indexed_any_type = len(indexed)
    if search_media == "videos":
        indexed = {k: v for k, v in indexed.items() if v.get("media_type") == "video"}
    elif search_media == "images":
        indexed = {k: v for k, v in indexed.items() if v.get("media_type") == "image"}

    n_pre_dup = len(indexed)
    skip_dup = _search_skip_duplicates_review(include_duplicates_review)
    if skip_dup:
        _before = len(indexed)
        indexed = {
            k: v for k, v in indexed.items() if not _index_item_under_duplicates_review(v)
        }
        if log_print and _before != len(indexed):
            print(
                f"  Skipping {_before - len(indexed)} item(s) under __duplicates_review__/ "
                f"(set FAMILY_SEARCH_INCLUDE_DUPLICATES_REVIEW=1 or --include-duplicates-review to include)."
            )

    if not indexed:
        if skip_dup and n_pre_dup > 0:
            return (
                None,
                None,
                "After excluding __duplicates_review__/, no items remain to search. "
                "Set FAMILY_SEARCH_INCLUDE_DUPLICATES_REVIEW=1 or use --include-duplicates-review.",
            )
        if search_media != "all" and n_indexed_any_type > 0:
            return (
                None,
                None,
                f"No indexed {search_media} in the index ({n_indexed_any_type} other item(s)).",
            )
        if not items:
            return None, None, f"No index entries yet. Run: python {prog_name} index"
        by_status = Counter(v.get("status") or "?" for v in items.values())
        summary = ", ".join(f"{s}={n}" for s, n in sorted(by_status.items()))
        return (
            None,
            None,
            f"Index has {len(items)} entries but nothing searchable as 'indexed' ({summary}). "
            f"Index file: {INDEX_FILE}",
        )

    if top < 1 or top > 200:
        return None, None, "--top must be between 1 and 200."

    batch_model = (batch_model or SEARCH_BATCH_MODEL).strip()
    rerank_model = (rerank_model or SEARCH_RERANK_MODEL).strip()

    if log_print:
        print(f"Searching {len(indexed)} item(s) for: '{query}'")
        print(
            f"  Batch model: {batch_model}  |  Rerank: "
            f"{'(skipped)' if no_rerank else rerank_model}  |  Max results: {top}\n"
        )

    records = [_minimal_search_record(uid, v) for uid, v in indexed.items()]
    pmax = _effective_prefilter_max(prefilter_max)
    if pmax > 0:
        records, pnote = _prefilter_search_records(records, query, pmax)
        if log_print and pnote:
            print(f"  {pnote}")

    batches = _chunk_search_records(records, SEARCH_BATCH_JSON_CHARS)
    n_batches = len(batches)
    pool_target = min(SEARCH_RERANK_POOL_MAX, max(top * 5, top + 10))
    per_batch_cap = max(3, min(12, math.ceil(pool_target / max(n_batches, 1))))
    if log_print and n_batches > 1:
        print(
            f"Splitting into {n_batches} search batch(es); "
            f"up to {per_batch_cap} hit(s) per batch; "
            f"parallel workers={min(SEARCH_PARALLEL_BATCHES, n_batches) or 1}.\n"
        )

    batch_matches, berr = _run_claude_batch_pass(
        batches, query, per_batch_cap, batch_model, log_print
    )
    if berr:
        return None, None, berr
    matches = batch_matches or []

    matches.sort(key=_match_sort_key)
    matches.reverse()
    seen_ids: set[str] = set()
    deduped: list[dict] = []
    for m in matches:
        iid = m.get("id")
        if not iid or iid in seen_ids:
            continue
        seen_ids.add(str(iid))
        deduped.append(m)
    matches = deduped

    if not no_rerank and len(matches) > top:
        client = anthropic.Anthropic()
        pool = matches[: min(len(matches), SEARCH_RERANK_POOL_MAX)]
        pool_ids = [m.get("id") for m in pool if m.get("id") in indexed]
        rerank_payload = json.dumps(
            [_minimal_search_record(uid, indexed[uid]) for uid in pool_ids],
            separators=(",", ":"),
        )
        r_prompt = (
            f"Refine search results for home photos and videos.\n"
            f"QUERY: {query}\n\n"
            f"Candidate item summaries (JSON array). Each has an \"id\" field:\n{rerank_payload}\n\n"
            f"Return a JSON array of ONLY the best {top} matches for QUERY "
            f"(or fewer if there are not enough good matches). Rank best first. "
            f'Each object: "id", "type" (video|image), "reason", '
            f'"confidence" (high|medium|low), "score" (0-100), '
            f'"start_sec" for videos (must be one of that item\'s sample_starts_sec or 0), '
            f"null for images.\n"
            f"Only use ids from the candidate list. Only return the JSON array, no other text."
        )
        if log_print:
            print(
                f"Reranking {len(pool_ids)} candidate(s) with {rerank_model} "
                f"(final list up to {top})..."
            )
        r_text = _messages_complete_text(
            client,
            rerank_model,
            r_prompt,
            extended_thinking=SEARCH_RERANK_THINKING,
        )
        try:
            reranked = parse_search_matches_json(r_text)
        except json.JSONDecodeError:
            matches = matches[:top]
        else:
            seen_r: set[str] = set()
            cleaned: list[dict] = []
            for m in reranked:
                iid = m.get("id")
                if not iid or iid not in indexed or iid in seen_r:
                    continue
                seen_r.add(str(iid))
                cleaned.append(m)
            matches = cleaned[:top] if cleaned else matches[:top]
    else:
        matches = matches[:top]

    if skip_dup:
        matches = [
            m
            for m in matches
            if (iid := m.get("id"))
            and not _index_item_under_duplicates_review(items.get(str(iid), {}))
        ]

    return matches, items, None


# ── S3 (private media for Gradio / Hugging Face) ─────────────────────────────


def normalize_s3_bucket_name(raw: str) -> str:
    """
    Return a bare bucket name for boto3.

    Accepts:
      - family-search-image-video-bucket
      - s3://family-search-image-video-bucket/prefix
      - https://REGION.console.aws.amazon.com/s3/buckets/BUCKET?region=...
      - https://BUCKET.s3.REGION.amazonaws.com/...
    """
    s = (raw or "").strip()
    if not s:
        return ""
    if "/buckets/" in s:
        m = re.search(r"/buckets/([^/?#]+)", s, re.I)
        if m:
            return m.group(1).strip()
    low = s.lower()
    if "console.aws.amazon.com" in low and "/buckets/" in s:
        tail = s.split("/buckets/", 1)[1]
        return tail.split("?", 1)[0].split("/", 1)[0].strip()
    if s.startswith("s3://"):
        rest = s[5:]
        return rest.split("/", 1)[0].strip()
    if s.startswith("http://") or s.startswith("https://"):
        try:
            u = urlparse(s)
            host = (u.netloc or "").split("@")[-1]
            if ".s3." in host and ".amazonaws.com" in host:
                return host.split(".s3.", 1)[0].strip()
        except Exception:
            pass
    return s


def _norm_path_prefix(p: str) -> str:
    """Forward slashes, no trailing slash (except root '/')."""
    s = (p or "").strip().replace("\\", "/")
    while "//" in s:
        s = s.replace("//", "/")
    if s == "/":
        return s
    return s.rstrip("/")


def s3_key_for_index_item(item: dict) -> str | None:
    """
    Resolve S3 object key for an index row.

    Prefer item['s3_key'] if set. Otherwise build a key from item['path'] relative to
    FAMILY_SEARCH_S3_PATH_ROOT (or S3_INDEX_PATH_ROOT), the same root string you passed to
    index-folder, plus optional S3_MEDIA_PREFIX.

    Uses string prefix matching so it works when the UI runs on Linux (e.g. Hugging Face)
    but index paths are macOS (/Volumes/...) where resolve()/relative_to() would fail.
    """
    sk = (item.get("s3_key") or "").strip()
    if sk:
        return sk.lstrip("/")
    path = item.get("path")
    if not path:
        return None
    root = (
        os.environ.get("FAMILY_SEARCH_S3_PATH_ROOT", "").strip()
        or os.environ.get("S3_INDEX_PATH_ROOT", "").strip()
    )
    prefix = os.environ.get("S3_MEDIA_PREFIX", "").strip().strip("/")
    if not root:
        return None
    pn = _norm_path_prefix(str(path))
    rn = _norm_path_prefix(root)
    key: str | None = None
    if pn == rn:
        key = ""
    elif pn.startswith(rn + "/"):
        key = pn[len(rn) + 1 :]
    else:
        try:
            rel = Path(path).resolve().relative_to(Path(root).resolve())
            key = str(rel).replace("\\", "/")
        except (ValueError, OSError):
            return None
    key = (key or "").strip()
    if not key:
        return None
    if prefix:
        return f"{prefix}/{key}"
    return key


def s3_object_key_candidates(key: str) -> list[str]:
    """Distinct key strings to probe (APFS often uses NFD; S3/console may use NFC)."""
    import unicodedata

    out: list[str] = []
    seen: set[str] = set()
    for k in (
        key,
        unicodedata.normalize("NFC", key),
        unicodedata.normalize("NFD", key),
    ):
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def s3_resolve_existing_key(client, bucket: str, key: str) -> str | None:
    """
    Return the first candidate key that exists (head_object OK), or None.

    Tries the given key plus Unicode NFC/NFD variants. Non-404 ClientErrors propagate.
    """
    from botocore.exceptions import ClientError

    for cand in s3_object_key_candidates(key):
        try:
            client.head_object(Bucket=bucket, Key=cand)
            return cand
        except ClientError as e:
            err = (e.response or {}).get("Error", {}) or {}
            code = err.get("Code", "")
            http = (e.response or {}).get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code in ("404", "NoSuchKey", "NotFound") or http == 404:
                continue
            raise
    return None


def _s3_basename_list_cache_ttl() -> float:
    try:
        return max(5.0, float(os.environ.get("S3_FALLBACK_LISTING_CACHE_SEC", "300")))
    except ValueError:
        return 300.0


def _s3_effective_basename_scan_limit() -> int:
    """
    Max objects to walk while resolving one basename.

    S3_FALLBACK_MAX_SCAN_KEYS unset or 0: use S3_FALLBACK_HARD_MAX_SCAN (default 4_000_000).
    Positive: that cap (clamped). One cached ListObjects walk indexes all filenames in that range.
    """
    raw = os.environ.get("S3_FALLBACK_MAX_SCAN_KEYS", "").strip()
    try:
        n = int(raw) if raw else 0
    except ValueError:
        n = 0
    if n <= 0:
        try:
            return max(10_000, int(os.environ.get("S3_FALLBACK_HARD_MAX_SCAN", "4000000")))
        except ValueError:
            return 4_000_000
    return max(1000, min(n, 50_000_000))


_S3_PREFIX_BASENAME_CACHE: dict[
    tuple[str, str, int], tuple[dict[str, list[str]], float, bool]
] = {}
# (bucket, list_prefix, max_scan, schema) -> (lookup_key -> [keys], ts, listing_reached_end)
_S3_PREFIX_MM_SCHEMA = 4


def _s3_basename_lookup_key(segment: str) -> str:
    """Canonical map key: NFC + lowercase (S3 keys are case-sensitive; Photos often differs)."""
    import unicodedata

    return unicodedata.normalize("NFC", segment).lower()


def _s3_basename_lookup_variants(basename: str) -> list[str]:
    """
    Candidate lookup keys for an index filename: case/spacing/.jpg/.jpeg permutations
    that often differ between macOS and uploads.
    """
    import unicodedata
    from pathlib import PurePosixPath

    raw = unicodedata.normalize("NFC", (basename or "").strip())
    if not raw:
        return []
    seen: set[str] = set()
    out: list[str] = []

    def add(s: str) -> None:
        k = _s3_basename_lookup_key(s)
        if k not in seen:
            seen.add(k)
            out.append(k)

    add(raw)
    nfkc = unicodedata.normalize("NFKC", raw)
    if nfkc != raw:
        add(nfkc)
    if " (" in raw:
        add(raw.replace(" (", "("))
    # macOS duplicate names: "DSC05459 (3).JPG" in __duplicates_review__ vs "DSC05459.JPG" on S3.
    _dup_stripped: list[str] = []
    for _pat in (
        r" \(\d+\)(?=(\.[^./]+)$)",  # "file (2).ext"
        r"(?<!\s)\(\d+\)(?=(\.[^./]+)$)",  # "file(2).ext" only; avoid "file (2)" -> "file .ext"
    ):
        _s = re.sub(_pat, "", raw, flags=re.IGNORECASE)
        if _s != raw and _s not in _dup_stripped:
            _dup_stripped.append(_s)
    for sv in _dup_stripped:
        add(sv)
        sp = PurePosixPath(sv)
        suf2 = sp.suffix
        stem2 = sp.name[: -len(suf2)] if suf2 else sp.name
        if suf2.lower() in (".jpg", ".jpeg", ".jpe"):
            for ext in (".jpg", ".jpeg", ".JPG", ".JPEG", ".Jpg", ".Jpe", ".jpe"):
                add(stem2 + ext)
        if suf2.lower() in (".mov", ".mp4", ".m4v"):
            for ext in (".mov", ".MOV", ".mp4", ".MP4", ".m4v", ".M4V", ".M4v"):
                add(stem2 + ext)
    p = PurePosixPath(raw)
    suf = p.suffix
    stem = p.name[: -len(suf)] if suf else p.name
    if suf.lower() in (".jpg", ".jpeg", ".jpe"):
        for ext in (".jpg", ".jpeg", ".JPG", ".JPEG", ".Jpg", ".Jpe", ".jpe"):
            add(stem + ext)
    if suf.lower() in (".mov", ".mp4", ".m4v"):
        for ext in (".mov", ".MOV", ".mp4", ".MP4", ".m4v", ".M4V", ".M4v"):
            add(stem + ext)
    return out


def _build_prefix_basename_multimap(
    client,
    bucket: str,
    list_prefix: str,
    *,
    max_scan: int,
) -> tuple[dict[str, list[str]], bool]:
    """
    One ListObjects walk: map canonical basename (NFC lower) -> full S3 keys.
    Stops after max_scan objects if the listing is not exhausted.
    """
    from collections import defaultdict

    pref = list_prefix or ""
    mm: dict[str, list[str]] = defaultdict(list)
    scanned = 0
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=pref):
        for obj in page.get("Contents") or []:
            scanned += 1
            if max_scan and scanned > max_scan:
                return dict(mm), False
            k = obj.get("Key") or ""
            if not k or k.endswith("/"):
                continue
            seg = k.rsplit("/", 1)[-1]
            lk = _s3_basename_lookup_key(seg)
            mm[lk].append(k)
    return dict(mm), True


def _get_prefix_basename_multimap_cached(
    client,
    bucket: str,
    list_prefix: str,
    *,
    max_scan: int,
) -> tuple[dict[str, list[str]], bool]:
    pref = list_prefix or ""
    ck = (bucket, pref, max_scan, _S3_PREFIX_MM_SCHEMA)
    now = time.time()
    ttl = _s3_basename_list_cache_ttl()
    ent = _S3_PREFIX_BASENAME_CACHE.get(ck)
    if ent and now - ent[1] < ttl:
        return ent[0], ent[2]
    mm, complete = _build_prefix_basename_multimap(client, bucket, pref, max_scan=max_scan)
    _S3_PREFIX_BASENAME_CACHE[ck] = (mm, now, complete)
    while len(_S3_PREFIX_BASENAME_CACHE) > 16:
        oldest = min(_S3_PREFIX_BASENAME_CACHE, key=lambda x: _S3_PREFIX_BASENAME_CACHE[x][1])
        del _S3_PREFIX_BASENAME_CACHE[oldest]
    return mm, complete


def s3_find_basename_matches_cached(
    client,
    bucket: str,
    basename: str,
    list_prefix: str,
    *,
    max_scan: int,
) -> tuple[list[str], bool]:
    """
    Look up basename using a cached prefix multimap (one walk per bucket/prefix/limit until TTL).

    Second value is True only if the walk was truncated at max_scan and this basename had no
    hits (the object may still appear later in key order).
    """
    pref = list_prefix or ""
    if not basename or basename.endswith("/"):
        return [], False
    mm, complete = _get_prefix_basename_multimap_cached(
        client, bucket, pref, max_scan=max_scan
    )
    seen_k: set[str] = set()
    matches: list[str] = []
    for lk in _s3_basename_lookup_variants(basename):
        for k in mm.get(lk, []):
            if k not in seen_k:
                seen_k.add(k)
                matches.append(k)
    truncated_miss = not complete and len(matches) == 0
    return matches, truncated_miss


def s3_resolve_preview_key(
    client,
    bucket: str,
    key: str,
    *,
    index_filename: str | None = None,
) -> tuple[str | None, str | None]:
    """
    Resolve a key for preview/download: exact path, then basename search when enabled.

    Returns (resolved_key, None) on success.

    Returns (None, None) to use the generic missing-object hint.

    Returns (None, specific_message) for basename ambiguity or misconfiguration.

    Optional ``index_filename``: the row's ``filename`` field when it differs from the path
    basename (e.g. path in quarantine with `` (2)`` but filename is the camera original).

    Basename fallback is **on by default** when the key path contains ``__duplicates_review__``
    (quarantine paths often are not synced to S3). Set ``S3_PREVIEW_FALLBACK_FIND_BASENAME=0`` to
    disable. Set ``S3_PREVIEW_FALLBACK_FIND_BASENAME=1`` to also try basename match for other keys.

    Env (optional):
      S3_FALLBACK_SEARCH_PREFIX — list prefix in the bucket (no leading slash); else S3_MEDIA_PREFIX;
        for ``__duplicates_review__`` keys, if both are empty, walks from bucket root (needs
        s3:ListBucket) up to S3_FALLBACK_HARD_MAX_SCAN unless S3_FALLBACK_MAX_SCAN_KEYS is set.
      S3_FALLBACK_FULL_BUCKET_SCAN — 1/true: allow explicit basename mode without a prefix for
        non-quarantine keys (same as empty prefix + scan cap).
      S3_FALLBACK_MAX_SCAN_KEYS — max objects per cached prefix walk; 0 or unset uses
        S3_FALLBACK_HARD_MAX_SCAN (default 4_000_000).
      S3_FALLBACK_HARD_MAX_SCAN — ceiling when MAX_SCAN_KEYS is 0/unset.
      S3_FALLBACK_LISTING_CACHE_SEC — TTL for cached basename results (default 300).
    """
    hit = s3_resolve_existing_key(client, bucket, key)
    if hit:
        return hit, None

    flag = os.environ.get("S3_PREVIEW_FALLBACK_FIND_BASENAME", "").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return None, None
    dup_path = "__duplicates_review__" in key
    if flag in ("1", "true", "yes", "on"):
        do_fallback = True
    else:
        do_fallback = dup_path

    if not do_fallback:
        return None, None

    basename_from_key = key.split("/")[-1] if "/" in key else key
    basename_candidates: list[str] = []
    if basename_from_key:
        basename_candidates.append(basename_from_key)
    _ifn = (index_filename or "").strip()
    if _ifn and _ifn not in basename_candidates:
        basename_candidates.append(_ifn)
    if not basename_candidates:
        return None, None

    raw_override = os.environ.get("S3_FALLBACK_SEARCH_PREFIX", "").strip().strip("/")
    raw_media = os.environ.get("S3_MEDIA_PREFIX", "").strip().strip("/")
    p = raw_override or raw_media
    full_scan = os.environ.get("S3_FALLBACK_FULL_BUCKET_SCAN", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    if not p and not full_scan:
        if dup_path:
            p = ""
        elif flag in ("1", "true", "yes", "on"):
            return None, (
                "S3_PREVIEW_FALLBACK_FIND_BASENAME=1 requires S3_FALLBACK_SEARCH_PREFIX, "
                "S3_MEDIA_PREFIX, or S3_FALLBACK_FULL_BUCKET_SCAN=1 when the key is not under "
                "__duplicates_review__/."
            )
        else:
            return None, None

    pref = f"{p}/" if p else ""
    max_scan = _s3_effective_basename_scan_limit()

    truncated = False
    for basename in basename_candidates:
        try:
            matches, truncated = s3_find_basename_matches_cached(
                client, bucket, basename, pref, max_scan=max_scan
            )
        except Exception as e:
            from botocore.exceptions import ClientError

            if isinstance(e, ClientError):
                err = (e.response or {}).get("Error", {}) or {}
                c = err.get("Code", type(e).__name__)
                m = err.get("Message", str(e))
                return None, (
                    f"S3 list failed ({c}: {m}) while resolving basename {basename!r}. "
                    "Grant s3:ListBucket on the bucket (or on prefix "
                    f"{pref!r}) in addition to s3:GetObject, or sync __duplicates_review__/ into S3."
                )
            return None, f"Basename listing failed: {e}"

        if len(matches) == 1:
            return matches[0], None
        if len(matches) > 1:
            return None, (
                f"Basename fallback found {len(matches)} objects for {basename!r} under prefix {pref!r}; "
                "set S3_FALLBACK_SEARCH_PREFIX to a narrower folder, sync __duplicates_review__/ into S3, "
                'or set \"s3_key\" on the index row.'
            )

    tried = ", ".join(repr(b) for b in basename_candidates)
    if dup_path:
        if truncated:
            return None, (
                f"No object for basenames [{tried}] in the first {max_scan} keys under prefix {pref!r}. "
                "Raise S3_FALLBACK_HARD_MAX_SCAN or set S3_FALLBACK_SEARCH_PREFIX / S3_MEDIA_PREFIX."
            )
        return None, (
            f"No object for basenames [{tried}] under prefix {pref!r} (full list for this cap). "
            "Tried copy-suffix stripping e.g. \"DSC05459 (3).JPG\" → \"DSC05459.JPG\", index filename, "
            "case and .jpg/.jpeg. If the bucket uses different names, set \"s3_key\" on the row."
        )
    return None, None


def s3_object_missing_hint(bucket: str, key: str) -> str:
    """User-facing explanation when no variant of the key exists in the bucket."""
    parts = [
        f"No object found for key {key!r} in bucket {bucket!r} "
        "(also tried Unicode NFC/NFD variants of the key)."
    ]
    if "__duplicates_review__" in key:
        parts.append(
            "The UI lists the bucket (up to S3_FALLBACK_HARD_MAX_SCAN keys by default) to find the "
            "same filename when the path contains __duplicates_review__/. Narrow the search with "
            "S3_MEDIA_PREFIX or S3_FALLBACK_SEARCH_PREFIX, add \"s3_key\" on the row, sync "
            "__duplicates_review__/ into S3, or set S3_PREVIEW_FALLBACK_FIND_BASENAME=0 to disable."
        )
    else:
        parts.append(
            "Confirm FAMILY_SEARCH_S3_PATH_ROOT / S3_INDEX_PATH_ROOT and S3_MEDIA_PREFIX match "
            "how objects were uploaded, or set \"s3_key\" on the item."
        )
    return " ".join(parts)


def presigned_s3_get_url(
    bucket: str,
    key: str,
    *,
    expires: int = 900,
) -> str:
    """HTTPS GET URL for a private object; use short TTL. Requires boto3 and AWS credentials."""
    try:
        import boto3
        from botocore.client import Config
    except ImportError as e:
        raise RuntimeError("Install boto3 for S3 presigned URLs") from e

    bucket = normalize_s3_bucket_name(bucket)
    if not bucket or "/" in bucket or "?" in bucket or bucket.startswith("http"):
        raise ValueError(
            "Invalid S3 bucket name. Set S3_MEDIA_BUCKET to the bucket name only "
            '(e.g. "my-bucket"), not the AWS console URL.'
        )

    region = (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-east-1"
    )
    client = boto3.client(
        "s3",
        region_name=region,
        config=Config(signature_version="s3v4"),
    )
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires,
    )


def cmd_search(args):
    """Search the index for matching items."""
    if getattr(args, "no_prefilter", False):
        pm = 0
    else:
        pm = getattr(args, "prefilter_max", None)
    matches, items, err = run_search_query(
        args.query,
        search_media=args.search_media,
        top=args.top,
        no_rerank=args.no_rerank,
        batch_model=args.batch_model,
        rerank_model=args.rerank_model,
        log_print=True,
        prog_name=Path(sys.argv[0]).name,
        prefilter_max=pm,
        include_duplicates_review=(
            True if getattr(args, "include_duplicates_review", False) else None
        ),
    )
    if err:
        print(err, file=sys.stderr)
        sys.exit(1)
    if not matches:
        print("No matching items found.")
        return

    print(f"\nFound {len(matches)} match(es):\n")
    print("=" * 70)

    for rank, match in enumerate(matches, 1):
        item_id = match.get("id")
        reason = match.get("reason", "")
        confidence = match.get("confidence", "")
        mtype = match.get("type", "")
        v = items.get(item_id, {})

        filename = v.get("filename", item_id)
        date = v.get("creation_date", "")
        duration = format_duration(v["duration"]) if v.get("duration") else ""
        path = v.get("path", "")

        type_label = f" {mtype.upper()}" if mtype else ""
        dur_label = f"  |  Duration: {duration}" if duration else ""

        print(f"#{rank}  [{confidence.upper()}]{type_label}  {filename}")
        print(f"     Date: {date}{dur_label}")
        print(f"     Why: {reason}")
        if path:
            print(f"     Path: {path}")
        print(f"     Id: {item_id}")
        if mtype == "video":
            samples = v.get("sample_times_sec") or []
            sec = snap_to_nearest_sample(match.get("start_sec"), samples)
            if samples:
                print(f"     Seek to (near indexed frame): {format_duration(sec)}")
        print()

    if matches and not args.no_open:
        print(
            "Tip: videos open in QuickTime near the moment that best matched your search "
            "(from indexed frame times). Use --no-seek to start at 0:00.\n"
            "     Or copy a Path and run: open -a 'QuickTime Player' \"/path\"\n"
            "     Enter a rank below to open a match.\n"
        )
        while True:
            choice = input(
                f"Open which match (1–{len(matches)}), or blank to skip? "
            ).strip()
            if not choice:
                break
            try:
                n = int(choice)
            except ValueError:
                print("  Enter a number, or blank to finish.")
                continue
            if not 1 <= n <= len(matches):
                print(f"  Enter a number from 1 to {len(matches)}.")
                continue
            mid = matches[n - 1].get("id")
            hit = items.get(mid, {})
            hit_path = hit.get("path")
            if not hit_path:
                print("  No file path in index for that item (e.g. iCloud placeholder).")
                continue
            if not os.path.exists(hit_path):
                print(f"  File not found: {hit_path}")
                continue
            mtype = hit.get("media_type", "")
            if mtype == "video":
                samples = hit.get("sample_times_sec") or []
                want_seek = not args.no_seek and bool(samples)
                start_sec = snap_to_nearest_sample(
                    matches[n - 1].get("start_sec"), samples
                ) if want_seek else 0.0
                if want_seek:
                    if open_quicktime_at(hit_path, start_sec):
                        print(
                            f"  Opened #{n} in QuickTime Player at ~{format_duration(start_sec)}."
                        )
                    else:
                        print(
                            f"  Opened #{n} in QuickTime Player (could not seek; "
                            f"try manually to ~{format_duration(start_sec)})."
                        )
                else:
                    subprocess.run(["open", "-a", "QuickTime Player", hit_path])
                    print(f"  Opened #{n} in QuickTime Player (from start).")
            else:
                subprocess.run(["open", "-a", "Preview", hit_path])
                print(f"  Opened #{n} in Preview.")


def cmd_show(args):
    """Open a file path or index id in Preview (images) or QuickTime (videos) on macOS."""
    if getattr(args, "item_id", None) and args.path:
        print("Use either --id or a path, not both.", file=sys.stderr)
        sys.exit(1)
    path: str | None = None
    if getattr(args, "item_id", None):
        if not INDEX_FILE.exists():
            print(f"No index found: {INDEX_FILE}", file=sys.stderr)
            sys.exit(1)
        hit = load_index().get("items", {}).get(args.item_id)
        if not hit:
            print(f"Unknown index id: {args.item_id}", file=sys.stderr)
            sys.exit(1)
        path = hit.get("path")
        if not path:
            print("That index entry has no local path.", file=sys.stderr)
            sys.exit(1)
    elif args.path:
        path = str(Path(args.path).expanduser().resolve())
    else:
        print("Provide a file path or --id ITEM_ID.", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(path):
        print(f"Not a file: {path}", file=sys.stderr)
        sys.exit(1)

    ext = Path(path).suffix.lower()
    if ext in _MEDIA_VIDEO_EXT:
        subprocess.run(["open", "-a", "QuickTime Player", path], check=False)
    else:
        subprocess.run(["open", "-a", "Preview", path], check=False)


def cmd_serve(args):
    """Serve indexed files over HTTP on 127.0.0.1 for browser inspection."""
    if not INDEX_FILE.exists():
        print(f"No index found: {INDEX_FILE}", file=sys.stderr)
        sys.exit(1)

    items = load_index().get("items", {})
    host = "127.0.0.1"
    port = args.port

    class IndexServeHandler(BaseHTTPRequestHandler):
        def log_message(self, _fmt, *_args):
            return

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                self._index_page()
            elif u.path == "/image":
                qs = parse_qs(u.query)
                raw = (qs.get("id") or [""])[0]
                self._serve_item(unquote(raw))
            else:
                self.send_error(404)

        def _send_html(self, code: int, body: str):
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _index_page(self):
            rows: list[str] = []
            for uid, v in sorted(
                items.items(),
                key=lambda kv: (kv[1].get("filename") or "").lower(),
            ):
                if v.get("status") != "indexed":
                    continue
                p = v.get("path")
                if not p or not os.path.isfile(p):
                    continue
                fn = html.escape(v.get("filename") or uid)
                uid_q = quote(uid, safe="")
                rows.append(f'<li><a href="/image?id={uid_q}">{fn}</a></li>')
            note = ""
            if len(rows) > 800:
                rows = rows[:800]
                note = (
                    "<p><em>List truncated to 800 items; use search output "
                    "and <code>show --id</code> for a specific id.</em></p>"
                )
            body = (
                "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
                "<title>family_search</title></head><body>"
                "<h1>Indexed files (local)</h1><ul>"
                + "".join(rows)
                + f"</ul>{note}"
                "<p><code>family_search.py serve</code> — localhost only.</p>"
                "</body></html>"
            )
            self._send_html(200, body)

        def _serve_item(self, item_id: str):
            v = items.get(item_id)
            if not v:
                self.send_error(404, "Unknown id")
                return
            filepath = v.get("path")
            if not filepath or not os.path.isfile(filepath):
                self.send_error(404, "Missing file")
                return
            try:
                data = Path(filepath).read_bytes()
            except OSError:
                self.send_error(500, "Read error")
                return
            mime, _ = mimetypes.guess_type(filepath)
            if not mime:
                mime = "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    httpd = HTTPServer((host, port), IndexServeHandler)
    url = f"http://{host}:{port}/"
    print(f"Serving index at {url} (Ctrl+C to stop)")
    print("  Click a name to view; or copy an Id from search and run show --id.")
    if getattr(args, "open_browser", False):
        subprocess.run(["open", url], check=False)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


# ── List ───────────────────────────────────────────────────────────────────────

def cmd_list(args):
    """List all indexed items."""
    if not INDEX_FILE.exists():
        print(f"No index found. Run 'python {sys.argv[0]} index' first.")
        sys.exit(1)

    index = load_index()
    items = index.get("items", {})

    if not items:
        print("Index is empty.")
        return

    album = index.get("album", DEFAULT_ALBUM)
    indexed_at = index.get("indexed_at", "unknown")

    n_indexed = sum(1 for v in items.values() if v.get("status") == "indexed")
    n_videos = sum(1 for v in items.values() if v.get("media_type") == "video" and v.get("status") == "indexed")
    n_images = sum(1 for v in items.values() if v.get("media_type") == "image" and v.get("status") == "indexed")
    by_status = Counter(v.get("status") or "?" for v in items.values())
    status_line = ", ".join(f"{s}={n}" for s, n in sorted(by_status.items()))

    print(f"Album: {album}  |  Last indexed: {indexed_at}")
    print(f"Index file: {len(items)} entr{'y' if len(items) == 1 else 'ies'}  |  Status: {status_line}")
    print(f"Searchable: {n_indexed} ({n_videos} videos, {n_images} images)\n")

    for uid, v in sorted(items.items(), key=lambda x: x[1].get("creation_date") or ""):
        status = v.get("status", "?")
        mtype = v.get("media_type", "?")
        icon = {
            ("indexed", "video"): "V",
            ("indexed", "image"): "I",
        }.get((status, mtype), "?" if status == "indexed" else "X" if status == "error" else "O")

        filename = v.get("filename", uid)
        date = v.get("creation_date", "")
        duration = format_duration(v["duration"]) if v.get("duration") else ""
        tags = ", ".join(v.get("tags", [])[:6])

        dur_str = f", {duration}" if duration else ""
        print(f"  [{icon}]  {filename}  ({date}{dur_str})")
        if tags:
            print(f"        Tags: {tags}")


# ── Dedup ──────────────────────────────────────────────────────────────────────

def file_sha256(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def perceptual_hash(filepath: str, size: int = 8) -> str | None:
    """
    Compute a simple perceptual hash: resize to 8x8 grayscale,
    compare each pixel to the mean, return a 64-bit hex string.
    Returns None if conversion fails.
    """
    with tempfile.NamedTemporaryFile(suffix=".gray", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-i", filepath,
                "-vf", f"scale={size}:{size}", "-pix_fmt", "gray",
                "-f", "rawvideo", "-y", tmp_path,
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            return None

        with open(tmp_path, "rb") as f:
            pixels = list(f.read())

        if len(pixels) != size * size:
            return None

        mean = sum(pixels) / len(pixels)
        bits = "".join("1" if p > mean else "0" for p in pixels)
        return hex(int(bits, 2))[2:].zfill(16)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def hamming_distance(h1: str, h2: str) -> int:
    """Hamming distance between two hex hash strings."""
    b1 = int(h1, 16)
    b2 = int(h2, 16)
    xor = b1 ^ b2
    return bin(xor).count("1")


def find_duplicate_groups(
    items_info: list[dict],
    *,
    exact_only: bool,
    threshold: int,
) -> tuple[dict[str, list[dict]], list[list[dict]]]:
    """
    items_info entries must include: id, file_hash, phash (or None), is_video implied via phash.
    Returns (exact_dupe_groups by SHA256, near_duplicate_groups).
    """
    hash_groups: dict[str, list[dict]] = defaultdict(list)
    for item in items_info:
        hash_groups[item["file_hash"]].append(item)
    exact_dupe_groups = {h: items for h, items in hash_groups.items() if len(items) > 1}

    near_dupe_groups: list[list[dict]] = []
    if exact_only:
        return exact_dupe_groups, near_dupe_groups

    phash_items = [it for it in items_info if it.get("phash") is not None]
    exact_ids = set()
    for group in exact_dupe_groups.values():
        for item in group:
            exact_ids.add(item["id"])

    matched = set()
    for i, a in enumerate(phash_items):
        if a["id"] in matched:
            continue
        group = [a]
        for b in phash_items[i + 1:]:
            if b["id"] in matched:
                continue
            dist = hamming_distance(a["phash"], b["phash"])
            if dist <= threshold:
                group.append(b)
                matched.add(b["id"])
        if len(group) > 1:
            group_ids = {it["id"] for it in group}
            if not group_ids.issubset(exact_ids):
                near_dupe_groups.append(group)
                matched.add(a["id"])

    return exact_dupe_groups, near_dupe_groups


def _unique_destination(dest_dir: Path, filename: str) -> Path:
    """Return a path under dest_dir that does not yet exist."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    base = Path(filename)
    candidate = dest_dir / base.name
    n = 0
    while candidate.exists():
        n += 1
        candidate = dest_dir / f"{base.stem}_dup{n}{base.suffix}"
    return candidate


def create_photos_album(album_name: str, photo_uuids: list[str]) -> bool:
    """
    Create or reuse a top-level album in Photos and add library items by UUID.

    Uses photoscript so IDs get the Photos 5+ suffix (e.g. .../L0/001) that
    bare-UUID AppleScript references do not use — without that, the album is
    created but stays empty.
    """
    try:
        from more_itertools import chunked
        import photoscript
    except ImportError:
        print("  Error: need photoscript (install osxphotos: pip install osxphotos).")
        return False

    try:
        library = photoscript.PhotosLibrary()
        album = library.album(album_name, top_level=True)
        if album is None:
            album = library.create_album(album_name)

        ps_photos = []
        for uid in photo_uuids:
            try:
                ps_photos.append(photoscript.Photo(uid))
            except ValueError:
                pass

        if not ps_photos:
            print(
                "  No items could be linked in Photos (UUIDs not found in this library).",
                file=sys.stderr,
            )
            return False

        for batch in chunked(ps_photos, 10):
            album.add(batch)

        n_skip = len(photo_uuids) - len(ps_photos)
        if n_skip:
            print(
                f"  Note: {n_skip} item(s) skipped (not in library or not referenceable)."
            )
        return True
    except Exception as e:
        print(f"  Photos album error: {e}")
        return False


def cmd_dedup(args):
    """Find and handle duplicate photos/videos in the album."""
    try:
        import osxphotos
    except ImportError:
        print("Error: osxphotos not installed. Run: pip install osxphotos")
        sys.exit(1)

    album_name = args.album
    threshold = args.threshold  # hamming distance threshold for near-dupes

    print(f"Opening Photos library...")
    photosdb = open_photos_db(osxphotos)

    print(f"Loading items from album: '{album_name}'")
    results = photosdb.query(osxphotos.QueryOptions(album=[album_name]))

    if not results:
        all_albums = [a.title for a in photosdb.album_info if a.title]
        matches = [a for a in all_albums if a.lower() == album_name.lower()]
        if matches:
            album_name = matches[0]
            results = photosdb.query(osxphotos.QueryOptions(album=[album_name]))

    if not results:
        print(f"No items found in album '{album_name}'.")
        sys.exit(1)

    print(f"Found {len(results)} items. Scanning for duplicates...\n")

    # Phase 1: Compute hashes
    items_info = []  # (uuid, filename, path, date, file_hash, phash, is_video)
    unavailable = 0

    for i, photo in enumerate(results, 1):
        filepath = photo.path
        if not filepath or not os.path.exists(filepath):
            filepath = photo.path_edited or photo.path_raw
        if not filepath or not os.path.exists(filepath):
            unavailable += 1
            continue

        filename = photo.original_filename or photo.uuid
        is_video = photo.ismovie
        date = str(photo.date) if photo.date else ""

        if i % 100 == 0 or i == len(results):
            print(f"  Hashing... {i}/{len(results)}", end="\r")

        fhash = file_sha256(filepath)
        phash = perceptual_hash(filepath) if not is_video else None

        items_info.append({
            "id": photo.uuid,
            "filename": filename,
            "path": filepath,
            "date": date,
            "file_hash": fhash,
            "phash": phash,
            "is_video": is_video,
            "filesize": os.path.getsize(filepath),
        })

    print(f"\n  Hashed {len(items_info)} items ({unavailable} unavailable/skipped)")

    exact_dupe_groups, near_dupe_groups = find_duplicate_groups(
        items_info, exact_only=args.exact_only, threshold=threshold
    )

    # Report results
    total_exact_dupes = sum(len(g) - 1 for g in exact_dupe_groups.values())
    total_near_dupes = sum(len(g) - 1 for g in near_dupe_groups)

    print(f"\n{'=' * 60}")
    print(f"DUPLICATE SCAN RESULTS")
    print(f"{'=' * 60}")
    print(f"  Exact duplicates:  {total_exact_dupes} items in {len(exact_dupe_groups)} group(s)")
    print(f"  Near duplicates:   {total_near_dupes} items in {len(near_dupe_groups)} group(s)")
    print(f"  Total removable:   {total_exact_dupes + total_near_dupes}")
    print()

    if total_exact_dupes == 0 and total_near_dupes == 0:
        print("No duplicates found!")
        return

    # Show details
    dupes_to_remove = []  # UUIDs of items to put in the review album

    if exact_dupe_groups:
        print("EXACT DUPLICATES (identical files):")
        print("-" * 60)
        for gnum, (fhash, group) in enumerate(exact_dupe_groups.items(), 1):
            # Keep the oldest (or largest) one, mark the rest
            group.sort(key=lambda x: (x["date"], -x["filesize"]))
            keep = group[0]
            remove = group[1:]

            print(f"  Group {gnum}: {len(group)} copies")
            print(f"    KEEP:   {keep['filename']}  ({keep['date']})")
            for r in remove:
                print(f"    REMOVE: {r['filename']}  ({r['date']})")
                dupes_to_remove.append(r["id"])
            print()

    if near_dupe_groups:
        print("NEAR DUPLICATES (visually similar):")
        print("-" * 60)
        for gnum, group in enumerate(near_dupe_groups, 1):
            group.sort(key=lambda x: (-x["filesize"], x["date"]))
            keep = group[0]  # keep the largest (likely highest quality)
            remove = group[1:]

            print(f"  Group {gnum}: {len(group)} similar items")
            print(f"    KEEP:   {keep['filename']}  ({keep['date']}, {keep['filesize']//1024}KB)")
            for r in remove:
                dist = hamming_distance(keep["phash"], r["phash"])
                print(f"    REMOVE: {r['filename']}  ({r['date']}, {r['filesize']//1024}KB, distance={dist})")
                dupes_to_remove.append(r["id"])
            print()

    if not dupes_to_remove:
        return

    # Offer to create a review album
    print(f"{'=' * 60}")
    print(f"I can create a 'Duplicates to Review' album in Photos with the")
    print(f"{len(dupes_to_remove)} duplicate(s) so you can review and delete them.")
    print(f"Nothing will be deleted automatically — you stay in control.")
    print()
    answer = input("Create the review album? [y/N] ").strip().lower()

    if answer == "y":
        print(f"\nCreating album 'Duplicates to Review' with {len(dupes_to_remove)} items...")
        success = create_photos_album("Duplicates to Review", dupes_to_remove)
        if success:
            print("Done! Open Photos and look for the 'Duplicates to Review' album.")
            print("Review the items, then select all and delete if they look right.")
        else:
            print("Could not create album via AppleScript.")
            print("You may need to grant Terminal access to Photos in:")
            print("  System Settings > Privacy & Security > Automation")

    # Save a report file
    report_path = _SCRIPT_DIR / "dedup_report.json"
    report = {
        "generated_at": datetime.now().isoformat(),
        "album": album_name,
        "total_items": len(items_info),
        "exact_duplicate_groups": [
            {
                "keep": group[0]["filename"],
                "remove": [r["filename"] for r in group[1:]],
                "uuids_to_remove": [r["id"] for r in group[1:]],
            }
            for group in (sorted(g, key=lambda x: (x["date"], -x["filesize"]))
                          for g in exact_dupe_groups.values())
        ],
        "near_duplicate_groups": [
            {
                "keep": group[0]["filename"],
                "remove": [r["filename"] for r in group[1:]],
                "uuids_to_remove": [r["id"] for r in group[1:]],
            }
            for group in (sorted(g, key=lambda x: (-x["filesize"], x["date"]))
                          for g in near_dupe_groups)
        ],
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed report saved to: {report_path}")


def cmd_dedup_folder(args):
    """Find duplicate files under a directory tree (SHA256 + optional perceptual hash)."""
    root = Path(args.path).expanduser().resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    threshold = args.threshold
    include_videos = args.type in ("all", "videos")
    include_images = args.type in ("all", "images")
    files = iter_folder_media(root, include_images, include_videos)
    if not files:
        print("No image or video files found under that path.")
        sys.exit(1)

    print(f"Scanning {root}\nFound {len(files)} file(s). Hashing...\n")

    items_info = []
    for i, (path, media_type) in enumerate(files, 1):
        path_str = str(path)
        if i % 100 == 0 or i == len(files):
            print(f"  Hashing... {i}/{len(files)}", end="\r")

        is_video = media_type == "video"
        date = file_creation_date_label(path)
        fhash = file_sha256(path_str)
        phash = perceptual_hash(path_str) if not is_video else None
        items_info.append({
            "id": path_str,
            "filename": path.name,
            "path": path_str,
            "date": date,
            "file_hash": fhash,
            "phash": phash,
            "is_video": is_video,
            "filesize": path.stat().st_size,
        })

    print(f"\n  Hashed {len(items_info)} items.")

    exact_dupe_groups, near_dupe_groups = find_duplicate_groups(
        items_info, exact_only=args.exact_only, threshold=threshold
    )

    total_exact_dupes = sum(len(g) - 1 for g in exact_dupe_groups.values())
    total_near_dupes = sum(len(g) - 1 for g in near_dupe_groups)

    print(f"\n{'=' * 60}")
    print("DUPLICATE SCAN RESULTS (folder)")
    print(f"{'=' * 60}")
    print(f"  Exact duplicates:  {total_exact_dupes} items in {len(exact_dupe_groups)} group(s)")
    print(f"  Near duplicates:   {total_near_dupes} items in {len(near_dupe_groups)} group(s)")
    print(f"  Total removable:   {total_exact_dupes + total_near_dupes}")
    print()

    if total_exact_dupes == 0 and total_near_dupes == 0:
        print("No duplicates found!")
        return

    dup_paths: list[str] = []

    if exact_dupe_groups:
        print("EXACT DUPLICATES (identical files):")
        print("-" * 60)
        for gnum, (_fhash, group) in enumerate(exact_dupe_groups.items(), 1):
            group.sort(key=lambda x: (x["date"], -x["filesize"]))
            keep = group[0]
            remove = group[1:]

            print(f"  Group {gnum}: {len(group)} copies")
            print(f"    KEEP:   {keep['filename']}  ({keep['date']})")
            print(f"            {keep['path']}")
            for r in remove:
                print(f"    REMOVE: {r['filename']}  ({r['date']})")
                print(f"            {r['path']}")
                dup_paths.append(r["path"])
            print()

    if near_dupe_groups:
        print("NEAR DUPLICATES (visually similar):")
        print("-" * 60)
        for gnum, group in enumerate(near_dupe_groups, 1):
            group.sort(key=lambda x: (-x["filesize"], x["date"]))
            keep = group[0]
            remove = group[1:]

            print(f"  Group {gnum}: {len(group)} similar items")
            print(f"    KEEP:   {keep['filename']}  ({keep['date']}, {keep['filesize'] // 1024}KB)")
            print(f"            {keep['path']}")
            for r in remove:
                dist = hamming_distance(keep["phash"], r["phash"])
                print(
                    f"    REMOVE: {r['filename']}  ({r['date']}, {r['filesize'] // 1024}KB, distance={dist})"
                )
                print(f"            {r['path']}")
                dup_paths.append(r["path"])
            print()

    dup_paths = list(dict.fromkeys(dup_paths))

    report_path = _SCRIPT_DIR / "dedup_folder_report.json"
    report = {
        "generated_at": datetime.now().isoformat(),
        "root": str(root),
        "total_items": len(items_info),
        "exact_duplicate_groups": [
            {
                "keep_path": sorted(g, key=lambda x: (x["date"], -x["filesize"]))[0]["path"],
                "keep": sorted(g, key=lambda x: (x["date"], -x["filesize"]))[0]["filename"],
                "remove": [r["filename"] for r in sorted(g, key=lambda x: (x["date"], -x["filesize"]))[1:]],
                "paths_to_remove": [r["path"] for r in sorted(g, key=lambda x: (x["date"], -x["filesize"]))[1:]],
            }
            for g in exact_dupe_groups.values()
        ],
        "near_duplicate_groups": [
            {
                "keep_path": sorted(g, key=lambda x: (-x["filesize"], x["date"]))[0]["path"],
                "keep": sorted(g, key=lambda x: (-x["filesize"], x["date"]))[0]["filename"],
                "remove": [r["filename"] for r in sorted(g, key=lambda x: (-x["filesize"], x["date"]))[1:]],
                "paths_to_remove": [r["path"] for r in sorted(g, key=lambda x: (-x["filesize"], x["date"]))[1:]],
            }
            for g in near_dupe_groups
        ],
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Detailed report saved to: {report_path}")

    if not dup_paths:
        return

    print(f"{'=' * 60}")
    print(
        f"Quarantine: move {len(dup_paths)} duplicate file(s) into:\n"
        f"  {root / '__duplicates_review__' / '<timestamp>'}\n"
        "Nothing is deleted — you can review and trash the folder later."
    )
    print()
    answer = input("Move duplicates into __duplicates_review__ now? [y/N] ").strip().lower()

    if answer != "y":
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    quarantine = root / "__duplicates_review__" / stamp
    moved = 0
    for src_str in dup_paths:
        src = Path(src_str)
        if not src.is_file():
            print(f"  Skip (missing): {src}")
            continue
        try:
            dest = _unique_destination(quarantine, src.name)
            shutil.move(str(src), str(dest))
            moved += 1
        except OSError as e:
            print(f"  Could not move {src}: {e}")

    print(f"\nMoved {moved} file(s) to:\n  {quarantine}")


def cmd_dedup_numbered(args):
    """
    Move files named like 'base (n).ext' when a matching base file exists in the same directory.
    Only .jpg, .jpeg, and .mov are considered; .jpg and .jpeg originals match each other.
    """
    root = Path(args.path).expanduser().resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    include_videos = args.type in ("all", "videos")
    include_images = args.type in ("all", "images")
    files = [
        (p, t)
        for p, t in iter_folder_media(root, include_images, include_videos)
        if not _path_under_duplicates_review(p)
    ]

    to_move: list[tuple[Path, Path]] = []
    orphan_mov: list[Path] = []
    mov_named_like_copy = 0

    for path, _media_type in files:
        if path.suffix.lower() not in _NUMBERED_DEDUP_EXTENSIONS:
            continue
        m = _NUMBERED_COPY_NAME_RE.match(path.name)
        if not m:
            continue
        if path.suffix.lower() == ".mov":
            mov_named_like_copy += 1
        original = _original_for_numbered_copy(
            path, m.group("base"), m.group("ext")
        )
        if original is not None:
            to_move.append((path, original))
        elif args.without_original and path.suffix.lower() == ".mov":
            orphan_mov.append(path)

    print(f"Scanning {root} (skipping __duplicates_review__ folders)")
    if mov_named_like_copy:
        paired_mov = sum(1 for d, _o in to_move if d.suffix.lower() == ".mov")
        loose = mov_named_like_copy - paired_mov
        print(
            f"  .mov files with (n) in the name: {mov_named_like_copy} "
            f"({paired_mov} have a matching base.mov beside them"
            + (f"; {loose} do not" if loose else "")
            + ")."
        )

    if not to_move and not orphan_mov:
        print()
        if mov_named_like_copy > 0 and not args.without_original:
            print(
                f"No paired duplicates to move. Found {mov_named_like_copy} .mov file(s) with (n) in "
                "the name but no matching base.mov in the same folder.\n"
                "  Run again with --without-original to quarantine those clips (they go under "
                "_no_original_mov for review)."
            )
        else:
            print(
                "No numbered copies to move among .jpg / .jpeg / .mov "
                "(no * (n) * matches, or nothing left after filters)."
            )
        return

    if to_move:
        print(
            f"\nFound {len(to_move)} numbered copy/copies with an original in the same folder:\n"
        )
        print("-" * 60)
        for dup, orig in sorted(to_move, key=lambda p: str(p[0]).lower()):
            print(f"  MOVE: {dup.name}")
            print(f"  KEEP: {orig.name}")
            print(f"        {dup}")
            print()

    if orphan_mov:
        print("-" * 60)
        print(
            f"{len(orphan_mov)} .mov file(s) match *(n).mov but have NO base.mov in the same folder "
            "(--without-original):"
        )
        for p in sorted(orphan_mov, key=lambda x: str(x).lower()):
            print(f"  QUARANTINE: {p.name}")
            print(f"              {p}")
        print()

    report_path = _SCRIPT_DIR / "dedup_numbered_report.json"
    report = {
        "generated_at": datetime.now().isoformat(),
        "root": str(root),
        "pairs": [
            {"duplicate": str(d), "original": str(o)} for d, o in to_move
        ],
        "orphan_mov_no_original": [str(p) for p in orphan_mov],
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to: {report_path}")

    total = len(to_move) + len(orphan_mov)
    print(f"{'=' * 60}")
    print(
        f"Quarantine: move up to {total} file(s) into:\n"
        f"  {root / '__duplicates_review__' / '<timestamp>_numbered'}\n"
        "Nothing is permanently deleted."
    )
    print()
    answer = input("Move these files now? [y/N] ").strip().lower()
    if answer != "y":
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    quarantine = root / "__duplicates_review__" / f"{stamp}_numbered"
    moved = 0
    for src, _orig in to_move:
        if not src.is_file():
            print(f"  Skip (missing): {src}")
            continue
        try:
            dest = _unique_destination(quarantine, src.name)
            shutil.move(str(src), str(dest))
            moved += 1
        except OSError as e:
            print(f"  Could not move {src}: {e}")

    orphan_sub = quarantine / "_no_original_mov"
    for src in orphan_mov:
        if not src.is_file():
            print(f"  Skip (missing): {src}")
            continue
        try:
            orphan_sub.mkdir(parents=True, exist_ok=True)
            dest = _unique_destination(orphan_sub, src.name)
            shutil.move(str(src), str(dest))
            moved += 1
        except OSError as e:
            print(f"  Could not move {src}: {e}")

    print(f"\nMoved {moved} file(s) under:\n  {quarantine}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Search your Apple Photos library with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Index items from an Apple Photos album")
    p_index.add_argument("--album", default=DEFAULT_ALBUM,
                         help=f"Album name (default: '{DEFAULT_ALBUM}')")
    p_index.add_argument("--type", choices=["all", "videos", "images"], default="all",
                         help="What to index (default: all)")
    p_index.add_argument("--reindex", action="store_true",
                         help="Re-analyze already-indexed items")
    p_index.add_argument(
        "--probe",
        action="store_true",
        help="Only report how many album items exist on disk (no AI, no index write)",
    )
    p_index.set_defaults(func=cmd_index)

    p_if = sub.add_parser(
        "index-folder",
        help="Index images/videos under a folder (e.g. exported originals on external storage)",
    )
    p_if.add_argument(
        "path",
        help="Root directory to scan recursively",
    )
    p_if.add_argument(
        "--type",
        choices=["all", "videos", "images"],
        default="all",
        help="What to index (default: all)",
    )
    p_if.add_argument(
        "--reindex",
        action="store_true",
        help="Re-analyze files already marked indexed",
    )
    p_if.set_defaults(func=cmd_index_folder)

    p_search = sub.add_parser("search", help="Search for items matching a description")
    p_search.add_argument("query", help="Natural language search query")
    p_search.add_argument(
        "--type",
        dest="search_media",
        choices=["all", "videos", "images"],
        default="all",
        help="Search only indexed videos or images (default: all)",
    )
    p_search.add_argument("--no-open", action="store_true",
                          help="Don't prompt to open the top match")
    p_search.add_argument(
        "--no-seek",
        action="store_true",
        help="When opening videos, start at the beginning instead of jumping to the matched moment",
    )
    p_search.add_argument(
        "--top",
        type=int,
        default=20,
        metavar="N",
        help="Maximum matches to return (default: 20)",
    )
    p_search.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip rerank pass; cheaper, uses batch model ordering only",
    )
    p_search.add_argument(
        "--batch-model",
        default=None,
        metavar="MODEL",
        help="Model for each index batch (default: Haiku or FAMILY_SEARCH_SEARCH_BATCH_MODEL)",
    )
    p_search.add_argument(
        "--rerank-model",
        default=None,
        metavar="MODEL",
        help="Model for final ranking (default: Opus or FAMILY_SEARCH_SEARCH_RERANK_MODEL)",
    )
    p_search.add_argument(
        "--no-prefilter",
        action="store_true",
        help="Send every item to the model (slow); default uses a keyword prefilter for speed",
    )
    p_search.add_argument(
        "--prefilter-max",
        type=int,
        default=None,
        metavar="N",
        help="After keyword prefilter, max items to search (default: env FAMILY_SEARCH_SEARCH_PREFILTER_MAX or 450)",
    )
    p_search.add_argument(
        "--include-duplicates-review",
        action="store_true",
        help="Include paths under __duplicates_review__/ in search (default: skip them; or set FAMILY_SEARCH_INCLUDE_DUPLICATES_REVIEW=1)",
    )
    p_search.set_defaults(func=cmd_search)

    p_show = sub.add_parser(
        "show",
        help="Open an image or video in Preview / QuickTime (path or index --id)",
    )
    p_show.add_argument(
        "--id",
        dest="item_id",
        metavar="ITEM_ID",
        help="Index id from search (e.g. file:...)",
    )
    p_show.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to an image or video file",
    )
    p_show.set_defaults(func=cmd_show)

    p_serve = sub.add_parser(
        "serve",
        help="Browse indexed files in a browser at http://127.0.0.1:PORT/ (local only)",
    )
    p_serve.add_argument(
        "--port",
        type=int,
        default=8765,
        help="TCP port (default: 8765)",
    )
    p_serve.add_argument(
        "--open",
        dest="open_browser",
        action="store_true",
        help="Open the index page in your default browser",
    )
    p_serve.set_defaults(func=cmd_serve)

    p_list = sub.add_parser("list", help="List all indexed items")
    p_list.set_defaults(func=cmd_list)

    p_dedup = sub.add_parser("dedup", help="Find and remove duplicate photos/videos")
    p_dedup.add_argument("--album", default=DEFAULT_ALBUM,
                         help=f"Album name (default: '{DEFAULT_ALBUM}')")
    p_dedup.add_argument("--exact-only", action="store_true",
                         help="Only find exact duplicates (skip near-duplicate detection)")
    p_dedup.add_argument("--threshold", type=int, default=5,
                         help="Hamming distance threshold for near-duplicates (default: 5, lower=stricter)")
    p_dedup.set_defaults(func=cmd_dedup)

    p_dedup_f = sub.add_parser(
        "dedup-folder",
        help="Find duplicate image/video files under a folder (e.g. exports on an external drive)",
    )
    p_dedup_f.add_argument("path", help="Root directory to scan recursively")
    p_dedup_f.add_argument(
        "--type",
        choices=["all", "videos", "images"],
        default="all",
        help="Media types to include (default: all)",
    )
    p_dedup_f.add_argument(
        "--exact-only",
        action="store_true",
        help="Only find exact duplicates (skip near-duplicate detection)",
    )
    p_dedup_f.add_argument(
        "--threshold",
        type=int,
        default=5,
        help="Hamming distance threshold for near-duplicates (default: 5)",
    )
    p_dedup_f.set_defaults(func=cmd_dedup_folder)

    p_dedup_n = sub.add_parser(
        "dedup-numbered",
        help="Move .jpg/.jpeg/.mov files like IMG_0066(1).JPG when IMG_0066.jpg exists beside them",
    )
    p_dedup_n.add_argument("path", help="Root directory to scan recursively")
    p_dedup_n.add_argument(
        "--type",
        choices=["all", "videos", "images"],
        default="all",
        help="Media types to include (default: all)",
    )
    p_dedup_n.add_argument(
        "--without-original",
        action="store_true",
        help=(
            "For .mov only: also quarantine files like clip(1).mov when clip.mov is NOT in the "
            "same folder (review in _no_original_mov subfolder)"
        ),
    )
    p_dedup_n.set_defaults(func=cmd_dedup_numbered)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
