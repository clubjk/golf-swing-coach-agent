#!/usr/bin/env python3
"""
Hugging Face Spaces: Gradio UI for natural-language search over family_search_index.json
with photos/videos served from a **private** S3 bucket via short-lived presigned URLs.

Layout for the Space repo (same folder):
  app.py                  ← this file
  family_search_index.json  ← optional if not using FAMILY_SEARCH_INDEX_S3_URI
  family_search_aws.py    ← copy from your play/ repo (search engine + S3 helpers)

Secrets (HF Space → Settings → Secrets):
  ANTHROPIC_API_KEY       — required for search
  AWS_ACCESS_KEY_ID       — IAM user/role with s3:GetObject on your media + index keys
  AWS_SECRET_ACCESS_KEY
  AWS_REGION              — e.g. us-east-1
  S3_MEDIA_BUCKET         — bucket name only (e.g. my-bucket), not the S3 console URL

  Optional:
  FAMILY_SEARCH_INCLUDE_DUPLICATES_REVIEW — set to 1/true to search paths under
                               __duplicates_review__/ (default: those rows are excluded from search).
  FAMILY_SEARCH_INDEX_S3_URI — s3://bucket/path/family_search_index.json (downloaded at boot; **recommended on Spaces**).
  FAMILY_SEARCH_INDEX_S3_KEY — if S3_URI is unset, object key inside S3_MEDIA_BUCKET (default
                               family_search_index.json at bucket root).
  FAMILY_SEARCH_S3_PATH_ROOT or S3_INDEX_PATH_ROOT — exact same string you passed to
                               index-folder (e.g. /Volumes/.../apple-photo-local-index-source).
                               Required for the UI to turn each index "path" into an S3 key
                               (copy/paste from the Mac that built the index; works on Linux/HF).
  S3_MEDIA_PREFIX         — key prefix inside the bucket (no slashes at ends), if objects
                            live under photos/...
  S3_PRESIGN_TTL          — seconds for presigned media URLs (default 900). The results list loads
                            images/videos in the **browser** from S3; set bucket CORS to allow GET
                            (and Range for video) from your Space origin (e.g. https://*.hf.space).

  For keys under __duplicates_review__/, if the object is missing in S3 the app walks ListObjects
  under the prefix until it finds a unique same-filename match (not only the first N keys). Default
  walk limit is S3_FALLBACK_HARD_MAX_SCAN (default 4M keys; one walk caches all basenames in range);
  override with S3_FALLBACK_MAX_SCAN_KEYS / S3_FALLBACK_HARD_MAX_SCAN.
  Set S3_MEDIA_PREFIX or S3_FALLBACK_SEARCH_PREFIX to limit listing to one subtree (faster).
  S3_PREVIEW_FALLBACK_FIND_BASENAME=0 disables basename lookup. S3_FALLBACK_LISTING_CACHE_SEC (300).
  Needs s3:ListBucket on the listed prefix and s3:GetObject.

S3 bucket CORS (required for result thumbnails in the page):
  - Allow GET, HEAD from your Space origin (e.g. https://*.hf.space).
  - AllowedHeaders: * (or include Range — needed for <video> byte-range requests).
  - ExposeHeaders: Content-Length, Content-Range, Accept-Ranges, ETag

  Images often work with minimal CORS; video usually fails until Range + expose headers are set.

Security notes:
  - End users never see AWS keys; only the Space backend does (HF secrets).
  - Presigned URLs are temporary bearer links; use a short TTL and a private Space if needed.
  - Scope IAM to GetObject on the prefix that holds your media (and index object if used).
"""

from __future__ import annotations

import html
import os
import re
import sys
from pathlib import Path

import gradio as gr

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _bootstrap_env_files() -> None:
    """Load .env from hf_family_search/ then play/ so local keys exist before imports."""
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


def _bare_s3_bucket_name(raw: str) -> str:
    """Match family_search_aws.normalize_s3_bucket_name enough for s3:// inference."""
    s = (raw or "").strip()
    if not s:
        return ""
    if "/buckets/" in s:
        m = re.search(r"/buckets/([^/?#]+)", s, re.I)
        if m:
            return m.group(1).strip()
    if s.lower().startswith("s3://"):
        return s[5:].split("/", 1)[0].strip()
    if s.startswith("http://") or s.startswith("https://"):
        return ""
    return s.split("?", 1)[0].strip().strip("/")


def _infer_family_search_index_s3_uri() -> None:
    """
    If FAMILY_SEARCH_INDEX_S3_URI is unset, use s3://S3_MEDIA_BUCKET/FAMILY_SEARCH_INDEX_S3_KEY
    so Spaces work when the index object lives in the same bucket as media.
    """
    if os.environ.get("FAMILY_SEARCH_INDEX_S3_URI", "").strip():
        return
    b = _bare_s3_bucket_name(os.environ.get("S3_MEDIA_BUCKET", "").strip())
    if not b or "/" in b:
        return
    key = (
        os.environ.get("FAMILY_SEARCH_INDEX_S3_KEY", "family_search_index.json")
        .strip()
        .lstrip("/")
    )
    if not key:
        return
    os.environ["FAMILY_SEARCH_INDEX_S3_URI"] = f"s3://{b}/{key}"


# Filled by _prepare_index_from_s3 so the UI can explain missing index on Spaces.
_INDEX_S3_LAST_ERROR: str | None = None


def _prepare_index_from_s3() -> None:
    global _INDEX_S3_LAST_ERROR
    _INDEX_S3_LAST_ERROR = None
    uri = os.environ.get("FAMILY_SEARCH_INDEX_S3_URI", "").strip()
    if not uri.startswith("s3://"):
        return
    try:
        import boto3
        from botocore.client import Config
    except ImportError:
        msg = "boto3 required to download FAMILY_SEARCH_INDEX_S3_URI"
        print(msg, file=sys.stderr)
        _INDEX_S3_LAST_ERROR = msg
        return
    rest = uri[5:]
    if "/" not in rest:
        msg = "Invalid FAMILY_SEARCH_INDEX_S3_URI (expected s3://bucket/key)"
        print(msg, file=sys.stderr)
        _INDEX_S3_LAST_ERROR = msg
        return
    bucket, _, key = rest.partition("/")
    dest = Path(
        os.environ.get(
            "FAMILY_SEARCH_INDEX_LOCAL",
            "/tmp/family_search_index.json",
        )
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
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.download_file(bucket, key, str(dest))
    except Exception as e:
        msg = f"Could not download index from {uri!r}: {e}"
        print(msg, file=sys.stderr)
        _INDEX_S3_LAST_ERROR = str(e)
        return
    os.environ["FAMILY_SEARCH_INDEX_FILE"] = str(dest)


_bootstrap_env_files()
_infer_family_search_index_s3_uri()
_prepare_index_from_s3()

import family_search_aws as fsa  # noqa: E402

# Secrets (e.g. Hugging Face) sometimes paste the S3 console URL as the bucket — fix once at import.
_sb = os.environ.get("S3_MEDIA_BUCKET", "").strip()
if _sb:
    os.environ["S3_MEDIA_BUCKET"] = fsa.normalize_s3_bucket_name(_sb)

def _caption_for_match(item: dict, match: dict, *, key: str) -> str:
    """Plain-text caption under each gallery tile (search reason + metadata)."""
    fn = (item.get("filename") or "").strip() or "(unnamed)"
    reason = str(match.get("reason", "") or "").strip()
    conf = str(match.get("confidence", "") or "").strip()
    mtype = (item.get("media_type") or match.get("type") or "image").lower()
    lines = [f"🎬 {fn}" if mtype == "video" else f"📷 {fn}"]
    if conf:
        lines.append(f"Match strength: {conf}")
    if reason:
        lines.append(reason)
    if mtype == "video" and key.lower().endswith(".mov"):
        lines.append("Tip: .mov plays most reliably in Safari.")
    return "\n".join(lines)


def _presigned_media_url(item: dict, match: dict) -> tuple[str | None, str, bool]:
    """
    HTTPS URL for <img>/<video src="…"> in the browser, caption text, and is_video.

    gr.Gallery was only showing one tile and broke vertical scroll on Spaces; the results strip
    uses plain HTML + presigned GET URLs so every hit can appear and the panel scrolls normally.
    """
    if not item:
        return None, "Missing index row for this hit.", False
    raw_bucket = os.environ.get("S3_MEDIA_BUCKET", "").strip()
    bucket = fsa.normalize_s3_bucket_name(raw_bucket)
    if not bucket or "?" in bucket or bucket.startswith("http"):
        cap = _caption_for_match(item, match, key="") + "\n\n⚠ Set S3_MEDIA_BUCKET (bucket name only)."
        return None, cap, False
    idx_key = fsa.s3_key_for_index_item(item)
    if not idx_key:
        root = (
            os.environ.get("FAMILY_SEARCH_S3_PATH_ROOT", "").strip()
            or os.environ.get("S3_INDEX_PATH_ROOT", "").strip()
        )
        raw_path = str(item.get("path") or "").strip()
        hint = (
            "Set FAMILY_SEARCH_S3_PATH_ROOT to the folder you passed to index-folder on the Mac."
            if not root
            else f"PATH_ROOT may not match this file. Index path starts: {raw_path[:100]}…"
        )
        cap = _caption_for_match(item, match, key="") + f"\n\n⚠ No S3 key. {hint}"
        return None, cap, False
    key = idx_key
    mtype = (item.get("media_type") or match.get("type") or "image").lower()
    is_video = mtype == "video"
    ttl = int(os.environ.get("S3_PRESIGN_TTL", "900") or "900")
    try:
        import boto3
        from botocore.client import Config

        r = (
            os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "us-east-1"
        )
        cli = boto3.client(
            "s3",
            region_name=r,
            config=Config(signature_version="s3v4"),
        )
        resolved_key, spec_err = fsa.s3_resolve_preview_key(
            cli,
            bucket,
            key,
            index_filename=(item.get("filename") or None),
        )
        if not resolved_key:
            msg = spec_err or fsa.s3_object_missing_hint(bucket, key)
            cap = _caption_for_match(item, match, key=key) + f"\n\n⚠ {msg}"
            return None, cap, is_video
        url = fsa.presigned_s3_get_url(bucket, resolved_key, expires=ttl)
        cap = _caption_for_match(item, match, key=resolved_key)
        return url, cap, is_video
    except Exception as e:
        cap = _caption_for_match(item, match, key=key) + f"\n\n⚠ Presign / S3 error: {e}"
        return None, cap, is_video


def _caption_to_html(text: str) -> str:
    return html.escape(text).replace("\n", "<br/>")


def _build_results_html(matches: list, items: dict) -> tuple[str, str]:
    """Scrollable HTML: one card per hit (image/video or text-only if URL missing)."""
    parts: list[str] = []
    media_fail = 0
    for m in matches:
        iid = m.get("id")
        item = items.get(iid) if iid else None
        url, cap, is_video = _presigned_media_url(item or {}, m)
        cap_h = _caption_to_html(cap)
        if url:
            src = html.escape(url, quote=True)
            if is_video:
                media_block = (
                    f'<video class="fs-thumb" controls playsinline preload="metadata" src="{src}">'
                    f"</video>"
                )
            else:
                media_block = (
                    f'<img class="fs-thumb" src="{src}" alt="" loading="lazy" '
                    f'decoding="async" />'
                )
            parts.append(
                f'<div class="fs-card">{media_block}<div class="fs-cap">{cap_h}</div></div>'
            )
        else:
            media_fail += 1
            parts.append(
                f'<div class="fs-card fs-card-missing">'
                f'<div class="fs-cap">{cap_h}</div></div>'
            )
    inner = "\n".join(parts)
    html_out = f'<div class="fs-scroll" role="region" aria-label="Search results">{inner}</div>'
    note = ""
    if media_fail:
        note = f"{media_fail} hit(s) have no media URL (see cards below). "
    return html_out, note


def _run_search(query: str, media: str, top: int):
    q = (query or "").strip()
    empty = "<p class='fs-placeholder'>Results will show here after you search.</p>"
    if not q:
        return "Type something to search — beach, birthday, red car…", gr.update(value=empty)
    matches, items, err = fsa.run_search_query(
        q,
        search_media=media,
        top=int(top),
        no_rerank=False,
        log_print=False,
        prog_name="family_search_aws.py",
        prefilter_max=None,
    )
    if err:
        return f"Search error: {err}", gr.update(value=f"<p>{html.escape(err)}</p>")
    if not matches:
        return "No matches — try different words.", gr.update(value="<p>No matching items.</p>")
    body, note = _build_results_html(matches, items)
    head = (
        f"**{len(matches)}** hits — scroll the **results** panel below (each hit is one card). "
    )
    if note:
        head += note
    head += (
        "_Thumbnails load from S3 in your browser; if images are blank, add CORS on the bucket "
        "for this Space’s domain (GET, and Range for video)._"
    )
    return head, gr.update(value=body)


def _index_banner_html() -> str:
    if fsa.INDEX_FILE.is_file():
        return ""
    tried = os.environ.get("FAMILY_SEARCH_INDEX_S3_URI", "").strip()
    lines = [
        "<div style='background:#fff7ed;border:1px solid #fdba74;padding:12px;margin-bottom:12px;border-radius:8px;'>",
        "<b>No index JSON found.</b> Spaces usually do not include <code>family_search_index.json</code> in the repo.",
    ]
    if tried:
        lines.append(
            "<p>Boot tried to download "
            f"<code>{html.escape(tried)}</code>. Verify the object exists and IAM allows "
            "<code>s3:GetObject</code> on it (and the correct <code>AWS_REGION</code>).</p>"
        )
    else:
        lines.append(
            "<p><strong>Hugging Face does not load your laptop&rsquo;s <code>play/.env</code>.</strong> "
            "Add <code>FAMILY_SEARCH_INDEX_S3_URI</code> (and AWS keys) under "
            "<em>Space settings &rarr; Secrets</em> (or Variables), or commit a <code>.env</code> "
            "next to <code>app.py</code> in the Space repo (avoid putting API keys there).</p>"
            "<p>Alternatively set <code>FAMILY_SEARCH_INDEX_S3_KEY</code> (default "
            "<code>family_search_index.json</code> at bucket root) so the URI is inferred from "
            "<code>S3_MEDIA_BUCKET</code>.</p>"
        )
    if _INDEX_S3_LAST_ERROR and tried:
        lines.append(
            f"<p>Last download error: <code>{html.escape(_INDEX_S3_LAST_ERROR)}</code></p>"
        )
    lines.append("</div>")
    return "".join(lines)


_FS_CSS = """
.fs-hero-wrap {
  margin-bottom: 1.25rem;
}
.fs-hero {
  text-align: center;
  padding: 1.5rem 1.25rem 1.35rem;
  border-radius: 24px;
  background: linear-gradient(125deg, #6d28d9 0%, #db2777 42%, #f97316 100%);
  box-shadow: 0 18px 50px rgba(109, 40, 217, 0.35);
  border: 1px solid rgba(255,255,255,0.2);
}
.fs-hero h1 {
  margin: 0;
  font-size: clamp(1.65rem, 4vw, 2.15rem);
  font-weight: 800;
  color: #fff;
  letter-spacing: -0.03em;
  text-shadow: 0 2px 24px rgba(0,0,0,0.2);
}
.fs-hero p {
  margin: 0.55rem 0 0;
  font-size: 1.05rem;
  color: rgba(255,255,255,0.94);
  line-height: 1.45;
}
.fs-scroll {
  max-height: min(78vh, 820px);
  overflow-y: auto;
  overflow-x: hidden;
  padding: 0.75rem 0.5rem 1.5rem;
  border-radius: 18px;
  border: 1px solid rgba(0,0,0,0.08);
  background: rgba(255,255,255,0.03);
  -webkit-overflow-scrolling: touch;
}
.fs-card {
  margin-bottom: 1.75rem;
  padding-bottom: 1.5rem;
  border-bottom: 1px solid rgba(0,0,0,0.08);
}
.fs-card:last-child {
  margin-bottom: 0;
  padding-bottom: 0;
  border-bottom: none;
}
.fs-card-missing {
  background: rgba(244, 63, 94, 0.06);
  border-radius: 14px;
  padding: 1rem 1rem 1.25rem !important;
  border-bottom: 1px solid rgba(244, 63, 94, 0.15) !important;
}
.fs-thumb {
  max-width: 100%;
  width: auto;
  height: auto;
  max-height: min(65vh, 640px);
  object-fit: contain;
  border-radius: 14px;
  display: block;
  box-shadow: 0 8px 28px rgba(0,0,0,0.12);
}
.fs-cap {
  margin-top: 0.75rem;
  font-size: 0.95rem;
  line-height: 1.5;
  text-align: left;
}
.fs-placeholder {
  opacity: 0.75;
  margin: 0.5rem 0;
}
"""

with gr.Blocks(
    title="Memory Lane — family search",
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.pink,
        secondary_hue=gr.themes.colors.amber,
        neutral_hue=gr.themes.colors.zinc,
        font=gr.themes.GoogleFont("Outfit"),
        font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
    ),
    css=_FS_CSS,
) as demo:
    gr.HTML(
        "<div class='fs-hero-wrap'><div class='fs-hero'>"
        "<h1>✨ Memory lane</h1>"
        "<p>Describe a moment — <strong>beach sunset</strong>, <strong>first bike ride</strong>, "
        "<strong>Thanksgiving 2019</strong> — and scroll through everything the search found.</p>"
        "</div></div>"
    )
    _banner = _index_banner_html()
    if _banner:
        gr.HTML(_banner)
    with gr.Row():
        q = gr.Textbox(
            label="What are you looking for?",
            placeholder="e.g. kids in costumes, golden retriever, ski trip…",
            scale=5,
            lines=1,
        )
        go = gr.Button("Search 🌈", variant="primary", scale=1, min_width=120)
    with gr.Row():
        media = gr.Radio(
            choices=["all", "images", "videos"],
            value="all",
            label="Show",
            elem_classes=["fs-filter"],
        )
        top = gr.Slider(1, 100, value=20, step=1, label="How many hits", scale=2)
    status = gr.Markdown(
        "*Each hit is a card below — **scroll inside the bordered results area**. "
        "Media loads from S3 in your browser (presigned URLs).*"
    )
    results = gr.HTML(
        value="<p class='fs-placeholder'>Run a search to see photos and descriptions here.</p>",
        label="Results",
        elem_classes=["fs-results-html"],
        container=True,
        padding=True,
    )

    _search_in = [q, media, top]
    _search_out = [status, results]
    go.click(_run_search, inputs=_search_in, outputs=_search_out)
    q.submit(_run_search, inputs=_search_in, outputs=_search_out)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=port)
