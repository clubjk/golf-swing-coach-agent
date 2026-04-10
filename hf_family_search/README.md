---
title: Family media search
emoji: 🖼
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.31.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Family media search

Natural-language search over `family_search_index.json`; previews load from a **private S3** bucket (credentials only on the Space).

## Secrets (Space → Settings → **Variables and secrets**)

| Name | Required |
|------|----------|
| `ANTHROPIC_API_KEY` | Yes |
| `AWS_ACCESS_KEY_ID` | Yes |
| `AWS_SECRET_ACCESS_KEY` | Yes |
| `AWS_REGION` | Yes (e.g. `us-east-1`) |
| `S3_MEDIA_BUCKET` | Yes (bucket name only) |
| `FAMILY_SEARCH_S3_PATH_ROOT` or `S3_INDEX_PATH_ROOT` | Yes — same path string used with `index-folder` on the Mac |
| `FAMILY_SEARCH_INDEX_S3_URI` | Recommended for large indexes (`s3://bucket/path/family_search_index.json`) |
| `S3_MEDIA_PREFIX` | If objects live under a prefix |
| `FAMILY_SEARCH_INCLUDE_DUPLICATES_REVIEW` | Set `1` only if you want `__duplicates_review__/` in search (default: excluded) |

Do not commit `.env` or keys. IAM needs `s3:GetObject` on media (and index if using `FAMILY_SEARCH_INDEX_S3_URI`), plus `s3:ListBucket` if basename fallback runs on the bucket/prefix.

## Repo layout

- `app.py` — Gradio UI  
- `family_search_aws.py` — search + S3 helpers (keep in sync with your main repo)  
- `requirements.txt` — Python deps  
- `deploy_to_hf.py` — push to the Space via API (watch mode = auto-deploy)  
- `deploy_to_hf.sh` — run deploy with `play/.venv` Python when `python` on PATH is broken  
- `family_search_index.json` — optional if using `FAMILY_SEARCH_INDEX_S3_URI`

## Auto-deploy from this machine (no GitHub)

1. Create the Space once on Hugging Face (Gradio), note the repo id `username/space-name`.
2. Create a **write** token: [HF settings → Access Tokens](https://huggingface.co/settings/tokens).
3. Locally (e.g. in `play/.env`, gitignored):

   ```bash
   HF_TOKEN=hf_...
   HF_SPACE_REPO_ID=yourname/your-space
   ```

4. Install deploy deps (once):

   ```bash
   pip install huggingface_hub watchdog
   ```

5. **Auto-deploy (default):** from repo root `play/`, leave this running while you code. It deploys once at startup, then whenever you save under `hf_family_search/` (any `.py`, `requirements.txt`, `README.md`, `.gitignore`) or when **`play/family_search_aws.py`** changes (it copies into `hf_family_search/` then deploys). Debounce ~2s.

   From `play/`: `.venv/bin/python hf_family_search/deploy_to_hf.py`  
   From `hf_family_search/`: `./deploy_to_hf.sh` (uses `../.venv`; avoids broken Homebrew `python`).

   In Cursor: **Tasks → Run Task → “HF Space: auto-deploy on save (default)”** (`play/.vscode/tasks.json`).

   One-shot push: `…/deploy_to_hf.py --once`  
   Skip the first deploy: `--no-initial`

Optional: `DEPLOY_INCLUDE_INDEX=1` to upload `family_search_index.json` (large).

HF sets `PORT` automatically; `app.py` uses it for `demo.launch`.

### Optional: git push to the Space instead

You can also `git remote add` the Space URL and `git push`; GitHub is not required.
