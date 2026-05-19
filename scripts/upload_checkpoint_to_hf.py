#!/usr/bin/env python3
"""
Upload checkpoint ke HuggingFace Hub (private repo).

Jalankan tanpa argumen untuk mode interaktif (menu pilihan).
Atau gunakan flag langsung untuk non-interaktif / scripting.

Usage:
    # Mode interaktif (direkomendasikan)
    python scripts/upload_checkpoint_to_hf.py

    # Upload checkpoint terbaru (step tertinggi)
    python scripts/upload_checkpoint_to_hf.py --latest

    # Upload checkpoint terbaik (val_bpb terendah)
    python scripts/upload_checkpoint_to_hf.py --best

    # Upload step tertentu
    python scripts/upload_checkpoint_to_hf.py --step 8000

    # Upload model + meta saja (skip optimizer state, lebih cepat)
    python scripts/upload_checkpoint_to_hf.py --best --model-only

    # Custom repo
    python scripts/upload_checkpoint_to_hf.py --best --repo Dummy9898/mesosfer-checkpoints
"""

import os
import sys
import json
import argparse
from pathlib import Path


# ── Helpers ──────────────────────────────────────────────────────────────────

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


def find_latest_checkpoint(ckpt_dir: Path) -> tuple[int, float | None]:
    """Cari checkpoint dengan step tertinggi (paling baru)."""
    meta_files = sorted(ckpt_dir.glob("meta_*.json"))
    if not meta_files:
        raise FileNotFoundError(f"Tidak ada file meta_*.json di {ckpt_dir}")

    latest_meta = meta_files[-1]
    step = int(latest_meta.stem.split("_")[1])

    try:
        with open(latest_meta, encoding="utf-8") as f:
            meta = json.load(f)
        val_bpb = meta.get("val_bpb")
    except Exception:
        val_bpb = None

    return step, val_bpb


def load_all_checkpoints(ckpt_dir: Path) -> list[dict]:
    """
    Baca semua meta_*.json dan kembalikan list dict:
    [{"step": int, "val_bpb": float|None, "label": str}, ...]
    diurutkan dari step terkecil ke terbesar.
    """
    meta_files = sorted(ckpt_dir.glob("meta_*.json"))
    rows = []
    best_bpb = float("inf")

    for meta_path in meta_files:
        try:
            step = int(meta_path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        val_bpb = None
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            val_bpb = meta.get("val_bpb")
        except Exception:
            pass
        if isinstance(val_bpb, float) and val_bpb < best_bpb:
            best_bpb = val_bpb
        rows.append({"step": step, "val_bpb": val_bpb})

    # Tandai best
    for row in rows:
        bpb = row["val_bpb"]
        bpb_str = f"{bpb:.6f}" if isinstance(bpb, float) else "N/A    "
        is_best = isinstance(bpb, float) and bpb == best_bpb
        tag = " ← BEST" if is_best else ""
        row["label"] = f"step {row['step']:>7,}   val_bpb={bpb_str}{tag}"

    return rows


def list_checkpoints(ckpt_dir: Path):
    """Tampilkan semua checkpoint beserta val_bpb-nya."""
    rows = load_all_checkpoints(ckpt_dir)
    if not rows:
        print("Tidak ada checkpoint ditemukan.")
        return

    print(f"\n{'Step':<10} {'val_bpb':<12} {'Status'}")
    print("-" * 35)
    for row in rows:
        bpb = row["val_bpb"]
        bpb_str = f"{bpb:.6f}" if isinstance(bpb, float) else "N/A"
        tag = "← BEST" if "← BEST" in row["label"] else ""
        print(f"{row['step']:<10} {bpb_str:<12} {tag}")
    print()


# ── Checkbox multi-select (prompt_toolkit) ───────────────────────────────────

def checkbox_select(rows: list[dict]) -> list[int]:
    """
    Tampilkan daftar checkpoint sebagai checkbox interaktif.
    Navigasi: ↑/↓ atau j/k, SPACE untuk toggle, ENTER untuk konfirmasi, q/ESC untuk batal.
    Kembalikan list step yang dipilih.
    """
    try:
        from prompt_toolkit import Application
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import Layout
        from prompt_toolkit.layout.containers import Window
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.formatted_text import HTML
        from prompt_toolkit.styles import Style
    except ImportError:
        print("ERROR: prompt_toolkit tidak terinstall.")
        print("Jalankan: pip install prompt_toolkit")
        sys.exit(1)

    if not rows:
        print("Tidak ada checkpoint tersedia.")
        return []

    n = len(rows)
    cursor = [0]          # posisi kursor saat ini
    selected = set()      # index yang dipilih
    cancelled = [False]

    style = Style.from_dict({
        "cursor":   "bold ansicyan",
        "checked":  "bold ansigreen",
        "unchecked": "",
        "tag":      "ansiyellow",
        "hint":     "ansibrightblack",
    })

    def get_text():
        lines = []
        lines.append(HTML(
            "<hint>  ↑/↓ navigasi   SPACE pilih/batal   ENTER konfirmasi   q batal\n\n</hint>"
        ))
        for i, row in enumerate(rows):
            is_cursor = (i == cursor[0])
            is_sel = (i in selected)

            checkbox = "[x]" if is_sel else "[ ]"
            label = row["label"]

            if is_cursor and is_sel:
                lines.append(HTML(f"<cursor>  ▶ <checked>{checkbox} {label}</checked></cursor>\n"))
            elif is_cursor:
                lines.append(HTML(f"<cursor>  ▶ {checkbox} {label}</cursor>\n"))
            elif is_sel:
                lines.append(HTML(f"    <checked>{checkbox} {label}</checked>\n"))
            else:
                lines.append(HTML(f"    {checkbox} {label}\n"))

        sel_count = len(selected)
        lines.append(HTML(f"\n<hint>  {sel_count} checkpoint dipilih</hint>\n"))
        return lines

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("k")
    def move_up(event):
        cursor[0] = (cursor[0] - 1) % n
        event.app.invalidate()

    @kb.add("down")
    @kb.add("j")
    def move_down(event):
        cursor[0] = (cursor[0] + 1) % n
        event.app.invalidate()

    @kb.add("space")
    def toggle(event):
        idx = cursor[0]
        if idx in selected:
            selected.discard(idx)
        else:
            selected.add(idx)
        event.app.invalidate()

    @kb.add("enter")
    def confirm(event):
        event.app.exit()

    @kb.add("q")
    @kb.add("escape")
    @kb.add("c-c")
    def cancel(event):
        cancelled[0] = True
        event.app.exit()

    layout = Layout(Window(content=FormattedTextControl(get_text, focusable=True)))
    app = Application(layout=layout, key_bindings=kb, style=style, full_screen=False)
    app.run()

    if cancelled[0]:
        return []

    return sorted(rows[i]["step"] for i in selected)


# ── Menu utama ────────────────────────────────────────────────────────────────

def interactive_menu(ckpt_dir: Path) -> tuple[str, list[int]]:
    """
    Tampilkan menu interaktif.
    Kembalikan (mode, steps):
      mode: 'latest' | 'best' | 'choose' | 'list' | 'quit'
      steps: list step untuk mode 'choose', kosong untuk mode lain
    """
    print("\n" + "=" * 45)
    print("  Upload Checkpoint ke HuggingFace Hub")
    print("=" * 45)
    print(f"  Checkpoint dir: {ckpt_dir}")
    print()

    meta_files = sorted(ckpt_dir.glob("meta_*.json"))
    if meta_files:
        steps_avail = []
        for m in meta_files:
            try:
                steps_avail.append(int(m.stem.split("_")[1]))
            except Exception:
                pass
        if steps_avail:
            print(f"  Checkpoint tersedia: {len(steps_avail)} ({min(steps_avail):,} – {max(steps_avail):,})")
    print()

    print("  Pilih mode upload:")
    print("  [1] Save Latest      — upload checkpoint step terbaru")
    print("  [2] Best Checkpoint  — upload checkpoint val_bpb terbaik")
    print("  [3] Choose Checkpoints — pilih manual (multi-select)")
    print("  [4] Lihat semua checkpoint")
    print("  [q] Keluar")
    print()

    while True:
        try:
            choice = input("  Pilihan (1/2/3/4/q): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nDibatalkan.")
            return "quit", []

        if choice in ("1", "latest"):
            return "latest", []
        elif choice in ("2", "best"):
            return "best", []
        elif choice in ("3", "choose"):
            return "choose", []
        elif choice in ("4", "list"):
            return "list", []
        elif choice in ("q", "quit", "exit"):
            return "quit", []
        else:
            print("  Input tidak valid. Masukkan 1, 2, 3, 4, atau q.")


# ── Upload satu step ──────────────────────────────────────────────────────────

def upload_step(api, step: int, ckpt_dir: Path, repo: str, depth: str, model_only: bool):
    """Upload semua file untuk satu step. Return (uploaded, total)."""
    step_str = f"{step:06d}"
    files = [f"model_{step_str}.pt", f"meta_{step_str}.json"]
    if not model_only:
        files.append(f"optim_{step_str}_rank0.pt")

    uploaded = 0
    for filename in files:
        filepath = ckpt_dir / filename
        if not filepath.exists():
            print(f"  SKIP: {filename} tidak ditemukan")
            continue
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  Uploading {filename} ({size_mb:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(filepath),
            path_in_repo=f"{depth}/{filename}",
            repo_id=repo,
            repo_type="model",
        )
        uploaded += 1
        print(f"  ✓ {filename} uploaded")

    return uploaded, len(files)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Upload checkpoint ke HuggingFace Hub")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--step", type=int, help="Step checkpoint yang mau diupload (contoh: 8000)")
    group.add_argument("--best", action="store_true", help="Otomatis upload checkpoint dengan val_bpb terbaik")
    group.add_argument("--latest", action="store_true", help="Upload checkpoint step terbaru")
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

    # ── Mode interaktif jika tidak ada flag ──────────────────────────────────
    chosen_steps: list[int] = []
    no_flag_given = not (args.step or args.best or args.latest or args.list)

    if no_flag_given:
        mode, _ = interactive_menu(ckpt_dir)
        if mode == "quit":
            return
        elif mode == "list":
            print(f"\nCheckpoint dir: {ckpt_dir}")
            list_checkpoints(ckpt_dir)
            return
        elif mode == "latest":
            args.latest = True
        elif mode == "best":
            args.best = True
        elif mode == "choose":
            rows = load_all_checkpoints(ckpt_dir)
            if not rows:
                print("Tidak ada checkpoint ditemukan.")
                return
            print("\n  Gunakan SPACE untuk memilih, ENTER untuk konfirmasi:\n")
            chosen_steps = checkbox_select(rows)
            if not chosen_steps:
                print("\nTidak ada checkpoint dipilih. Dibatalkan.")
                return
            print(f"\n  Dipilih: {len(chosen_steps)} checkpoint — {chosen_steps}")

    # Mode --list
    if args.list:
        print(f"Checkpoint dir: {ckpt_dir}")
        list_checkpoints(ckpt_dir)
        return

    # Tentukan daftar step yang akan diupload
    steps_to_upload: list[int] = []

    if chosen_steps:
        # Mode choose: sudah ditentukan dari checkbox
        steps_to_upload = chosen_steps
        for s in steps_to_upload:
            meta_path = ckpt_dir / f"meta_{s:06d}.json"
            val_bpb = "N/A"
            if meta_path.exists():
                try:
                    with open(meta_path, encoding="utf-8") as f:
                        val_bpb = json.load(f).get("val_bpb", "N/A")
                except Exception:
                    pass
            bpb_str = f"{val_bpb:.6f}" if isinstance(val_bpb, float) else str(val_bpb)
            print(f"  • step {s:,}  (val_bpb={bpb_str})")
    elif args.best:
        step, best_bpb = find_best_checkpoint(ckpt_dir)
        print(f"\nCheckpoint terbaik: step {step} (val_bpb={best_bpb:.6f})")
        steps_to_upload = [step]
    elif args.latest:
        step, val_bpb = find_latest_checkpoint(ckpt_dir)
        bpb_info = f"val_bpb={val_bpb:.6f}" if isinstance(val_bpb, float) else "val_bpb=N/A"
        print(f"\nCheckpoint terbaru: step {step} ({bpb_info})")
        steps_to_upload = [step]
    elif args.step:
        step = args.step
        meta_path = ckpt_dir / f"meta_{step:06d}.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                val_bpb = json.load(f).get("val_bpb", "N/A")
            print(f"Upload step {step} (val_bpb={val_bpb})")
        else:
            print(f"Upload step {step} (meta tidak ditemukan)")
        steps_to_upload = [step]
    else:
        parser.print_help()
        return

    # ── Login HuggingFace ─────────────────────────────────────────────────────
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub tidak terinstall. Jalankan: pip install huggingface_hub")
        return

    api = HfApi()
    try:
        user = api.whoami()
        print(f"\nLogin sebagai: {user['name']}")
    except Exception as e:
        print(f"ERROR: Tidak bisa login ke HuggingFace — {e}")
        print("Jalankan: hf auth login")
        return

    # Pastikan repo ada
    api.create_repo(repo_id=args.repo, repo_type="model", private=True, exist_ok=True)
    print(f"Repo: {args.repo}\n")

    # ── Upload semua step yang dipilih ────────────────────────────────────────
    grand_uploaded = 0
    grand_total = 0

    for i, step in enumerate(steps_to_upload, 1):
        if len(steps_to_upload) > 1:
            print(f"── [{i}/{len(steps_to_upload)}] step {step:,} ──")
        uploaded, total = upload_step(api, step, ckpt_dir, args.repo, args.depth, args.model_only)
        grand_uploaded += uploaded
        grand_total += total
        if len(steps_to_upload) > 1:
            print()

    print(f"Selesai! {grand_uploaded}/{grand_total} file diupload ke {args.repo}/{args.depth}/")


if __name__ == "__main__":
    main()
