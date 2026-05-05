#!/usr/bin/env python3
"""
Script to upload golf swing coach agent files to Hugging Face Space
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, upload_file, upload_folder

# Configuration
SPACE_ID = "clubjk/golf-swing-coach-agent"
LOCAL_DIR = Path("/Users/john/ai/play/golf-swing-coach-agent")

# Files to upload
FILES_TO_UPLOAD = [
    "Dockerfile",
    "requirements.txt",
    "app.py",
    "crew.py",
    "README.md"
]

FOLDERS_TO_UPLOAD = [
    "tools",
    "utils"
]

def main():
    # Get token from environment
    token = os.getenv("HF_TOKEN")
    if not token:
        print("❌ Please set your HF_TOKEN environment variable:")
        print("export HF_TOKEN='your-huggingface-write-token'")
        return

    api = HfApi(token=token)

    print(f"🚀 Uploading files to {SPACE_ID}...")

    # Upload individual files
    for file_path in FILES_TO_UPLOAD:
        local_path = LOCAL_DIR / file_path
        if local_path.exists():
            print(f"📤 Uploading {file_path}...")
            upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=file_path,
                repo_id=SPACE_ID,
                repo_type="space",
                token=token
            )
        else:
            print(f"⚠️  File {file_path} not found, skipping...")

    # Upload folders
    for folder in FOLDERS_TO_UPLOAD:
        local_folder = LOCAL_DIR / folder
        if local_folder.exists():
            print(f"📤 Uploading {folder}/...")
            upload_folder(
                folder_path=str(local_folder),
                path_in_repo=folder,
                repo_id=SPACE_ID,
                repo_type="space",
                token=token
            )
        else:
            print(f"⚠️  Folder {folder}/ not found, skipping...")

    print("✅ Upload complete!")
    print(f"🌐 Your space: https://huggingface.co/spaces/{SPACE_ID}")
    print("The space will rebuild automatically with the new files.")

if __name__ == "__main__":
    main()