#!/usr/bin/env python3
"""
Upload checkpoint ke HuggingFace Hub (private repo).

Bisa upload step tertentu, atau otomatis cari checkpoint terbaik
berdasarkan val_bpb terendah dari file meta_*.json.

Usage:
    # Upload checkpoint terbaik (val_bpb terendah) secara otomatis
    python scripts/upload_checkpoint_to_hf.py --best

    # Upload step tertentu
    python scripts/upload_checkpoint_to_hf.py --step 8000

    # Upload model + meta saja (skip optimizer state, lebih cepat)
    python scripts/upload_checkpoint_to_hf.py --best --model-only

    # Custom repo
    python scripts/upload_checkpoint_to_hf.py --best --repo Dummy9898/mesosfer-checkpoints
"""

import os
import json
import argparse
from pathlib import Path


def find_best_checkpoint(ckpt_dir: Path) -> tuple[int, float]:
    """Cari checkpoint dengan val_bpb terendah dari semua meta_*.json."""
    meta_files = sorted(ckpt_dir.glob("meta_*.json"))
    if not meta_files:
        raise FileNotFoundError(f"Tidak ada file meta_*.json di {ckpt_dir}")

    best_step = None
    best_bpb = float("inf")

    for meta_path in meta_files:
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            val_bpb = meta.get("val_bpb")
            if val_bpb is None:
                continue
            step = int(meta_path.stem.split("_")[1])  # meta_008000.json -> 8000
            if val_bpb < best_bpb:
                best_bpb = val_bpb
                best_step = step
        except Exception as e:
            print(f"  WARN: Gagal baca {meta_path.name}: {e}")

    if best_step is None:
        raise ValueError("Tidak ada checkpoint dengan val_bpb yang valid")

    return best_step, best_bpb


def list_checkpoints(ckpt_dir: Path):
    """Tampilkan semua checkpoint beserta val_bpb-nya."""
    meta_files = sorted(ckpt_dir.glob("meta_*.json"))
    if not meta_files:
        print("Tidak ada checkpoint ditemukan.")
        return

    print(f"\n{'Step':<10} {'val_bpb':<12} {'Status'}")
    print("-" * 35)
    best_bpb = float("inf")
    rows = []
    for meta_path in meta_files:
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            val_bpb = meta.get("val_bpb", "N/A")
            step = int(meta_path.stem.split("_")[1])
            rows.append((step, val_bpb))
            if isinstance(val_bpb, float) and val_bpb < best_bpb:
                best_bpb = val_bpb
        except Exception:
            pass

    for step, val_bpb in rows:
        is_best = isinstance(val_bpb, float) and val_bpb == best_bpb
        status = "← BEST" if is_best else ""
        bpb_str = f"{val_bpb:.6f}" if isinstance(val_bpb, float) else str(val_bpb)
        print(f"{step:<10} {bpb_str:<12} {status}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Upload checkpoint ke HuggingFace Hub")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--step", type=int, help="Step checkpoint yang mau diupload (contoh: 8000)")
    group.add_argument("--best", action="store_true", help="Otomatis upload checkpoint dengan val_bpb terbaik")
    group.add_argument("--list", action="store_true", help="Tampilkan semua checkpoint dan val_bpb-nya")
    parser.add_argument("--depth", type=str, default="d24", help="Model depth tag (default: d24)")
    parser.add_argument("--repo", type=str, default="Dummy9898/mesosfer-checkpoints", help="HF repo ID")
    parser.add_argument("--model-only", action="store_true", help="Upload model + meta saja (skip optimizer state)")
    parser.add_argument("--base-dir", type=str, default=None, help="Override base cache dir")
    args = parser.parse_args()

    base_dir = args.base_dir or os.path.expanduser("~/.cache/mesosfer")
    ckpt_dir = Path(base_dir) / "base_checkpoints" / args.depth

    if not ckpt_dir.exists():
        print(f"ERROR: Checkpoint dir tidak ditemukan: {ckpt_dir}")
        return

    # Mode --list: tampilkan semua checkpoint
    if args.list:
        print(f"Checkpoint dir: {ckpt_dir}")
        list_checkpoints(ckpt_dir)
        return

    # Tentukan step yang akan diupload
    if args.best:
        step, best_bpb = find_best_checkpoint(ckpt_dir)
        print(f"Checkpoint terbaik: step {step} (val_bpb={best_bpb:.6f})")
    elif args.step:
        step = args.step
        # Baca val_bpb dari meta untuk info
        meta_path = ckpt_dir / f"meta_{step:06d}.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            val_bpb = meta.get("val_bpb", "N/A")
            print(f"Upload step {step} (val_bpb={val_bpb})")
        else:
            print(f"Upload step {step} (meta tidak ditemukan)")
    else:
        parser.print_help()
        return

    # Login check
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub tidak terinstall. Jalankan: pip install huggingface_hub")
        return

    api = HfApi()
    try:
        user = api.whoami()
        print(f"Login sebagai: {user['name']}")
    except Exception as e:
        print(f"ERROR: Tidak bisa login ke HuggingFace — {e}")
        print("Jalankan: hf auth login")
        return

    # File yang akan diupload
    step_str = f"{step:06d}"
    files = [f"model_{step_str}.pt", f"meta_{step_str}.json"]
    if not args.model_only:
        files.append(f"optim_{step_str}_rank0.pt")

    # Pastikan repo ada
    api.create_repo(repo_id=args.repo, repo_type="model", private=True, exist_ok=True)
    print(f"Repo: {args.repo}")
    print()

    # Upload
    total_uploaded = 0
    for filename in files:
        filepath = ckpt_dir / filename
        if not filepath.exists():
            print(f"SKIP: {filename} tidak ditemukan")
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
