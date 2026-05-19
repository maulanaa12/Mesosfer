#!/usr/bin/env python3
"""
Upload checkpoint ke HuggingFace Hub (private repo).

Usage:
    python scripts/upload_checkpoint_to_hf.py --step 8000
    python scripts/upload_checkpoint_to_hf.py --step 8000 --model-only
    python scripts/upload_checkpoint_to_hf.py --step 8000 --repo Dummy9898/mesosfer-checkpoints
"""

import os
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Upload checkpoint ke HuggingFace Hub")
    parser.add_argument("--step", type=int, required=True, help="Step checkpoint yang mau diupload (contoh: 8000)")
    parser.add_argument("--depth", type=str, default="d24", help="Model depth tag (default: d24)")
    parser.add_argument("--repo", type=str, default="Dummy9898/mesosfer-checkpoints", help="HF repo ID")
    parser.add_argument("--model-only", action="store_true", help="Upload model + meta saja (skip optimizer state)")
    parser.add_argument("--base-dir", type=str, default=None, help="Override base cache dir")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub tidak terinstall. Jalankan: pip install huggingface_hub")
        return

    api = HfApi()

    # Verifikasi login
    try:
        user = api.whoami()
        print(f"Login sebagai: {user['name']}")
    except Exception as e:
        print(f"ERROR: Tidak bisa login ke HuggingFace — {e}")
        print("Jalankan: hf auth login")
        return

    # Tentukan path checkpoint
    base_dir = args.base_dir or os.path.expanduser("~/.cache/mesosfer")
    ckpt_dir = Path(base_dir) / "base_checkpoints" / args.depth
    step_str = f"{args.step:06d}"

    # File yang akan diupload
    files = [
        f"model_{step_str}.pt",
        f"meta_{step_str}.json",
    ]
    if not args.model_only:
        files.append(f"optim_{step_str}_rank0.pt")

    # Pastikan repo ada
    api.create_repo(repo_id=args.repo, repo_type="model", private=True, exist_ok=True)
    print(f"Repo: {args.repo}")
    print(f"Checkpoint dir: {ckpt_dir}")
    print()

    # Upload satu per satu
    total_uploaded = 0
    for filename in files:
        filepath = ckpt_dir / filename
        if not filepath.exists():
            print(f"SKIP: {filename} tidak ditemukan di {filepath}")
            continue
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"Uploading {filename} ({size_mb:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(filepath),
            path_in_repo=f"{args.depth}/{filename}",
            repo_id=args.repo,
            repo_type="model",
        )
        total_uploaded += 1
        print(f"  ✓ {filename} uploaded")

    print()
    print(f"Selesai! {total_uploaded}/{len(files)} file diupload ke {args.repo}/{args.depth}/")


if __name__ == "__main__":
    main()
