#!/usr/bin/env python3
"""
Upload checkpoint to HuggingFace Hub (private repo).

Run without arguments for interactive mode (selection menu).
Or use flags directly for non-interactive / scripting.

Usage:
    # Interactive mode (recommended)
    python scripts/upload_checkpoint_to_hf.py

    # Upload the latest base checkpoint
    python scripts/upload_checkpoint_to_hf.py --latest

    # Upload the best SFT checkpoint
    python scripts/upload_checkpoint_to_hf.py --best --source sft

    # Upload the best RL checkpoint
    python scripts/upload_checkpoint_to_hf.py --best --source rl

    # Upload the best checkpoint (lowest val_bpb)
    python scripts/upload_checkpoint_to_hf.py --best

    # Upload a specific step
    python scripts/upload_checkpoint_to_hf.py --step 8000

    # Upload model + meta only (skip optimizer state, faster)
    python scripts/upload_checkpoint_to_hf.py --best --model-only

    # Custom repo
    python scripts/upload_checkpoint_to_hf.py --best --repo Dummy9898/mesosfer-checkpoints

Checkpoint sources:
    base  → ~/.cache/mesosfer/base_checkpoints/<depth>/
    sft   → ~/.cache/mesosfer/chatsft_checkpoints/<depth>/
    rl    → ~/.cache/mesosfer/chatrl_checkpoints/<depth>/
"""

import os
import sys
import json
import argparse
from pathlib import Path


# ── Helpers ──────────────────────────────────────────────────────────────────

def find_best_checkpoint(ckpt_dir: Path) -> tuple[int, float]:
    """Search for the checkpoint with the lowest val_bpb across all meta_*.json files."""
    meta_files = sorted(ckpt_dir.glob("meta_*.json"))
    if not meta_files:
        raise FileNotFoundError(f"Could not find any meta_*.json files. {ckpt_dir}")

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
            print(f"  WARN: Failed to read {meta_path.name}: {e}")

    if best_step is None:
        raise ValueError("Could not find any checkpoints with a valid val_bpb value.")

    return best_step, best_bpb


def find_latest_checkpoint(ckpt_dir: Path) -> tuple[int, float | None]:
    """Find the checkpoint with the highest step (latest)."""
    meta_files = sorted(ckpt_dir.glob("meta_*.json"))
    if not meta_files:
        raise FileNotFoundError(f"No meta_*.json files found in the current directory. {ckpt_dir}")

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
    Read all meta_*.json files and return a list of dicts:
    [{"step": int, "val_bpb": float|None, "label": str}, ...]
    sorted from lowest to highest step.
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

    # Mark best
    for row in rows:
        bpb = row["val_bpb"]
        bpb_str = f"{bpb:.6f}" if isinstance(bpb, float) else "N/A    "
        is_best = isinstance(bpb, float) and bpb == best_bpb
        tag = " ← BEST" if is_best else ""
        row["label"] = f"step {row['step']:>7,}   val_bpb={bpb_str}{tag}"

    return rows


def list_checkpoints(ckpt_dir: Path):
    """Show all checkpoints along with their val_bpb."""
    rows = load_all_checkpoints(ckpt_dir)
    if not rows:
        print("No checkpoints found.")
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
    Display checkpoints as an interactive checkbox list.
    Navigation: ↑/↓ or j/k to move, SPACE to toggle, ENTER to confirm, q/ESC to cancel.
    Return a list of selected steps..
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
        print("ERROR: prompt_toolkit is not installed.")
        print("Run: pip install prompt_toolkit")
        sys.exit(1)

    if not rows:
        print("No checkpoints found.")
        return []

    n = len(rows)
    cursor = [0]          # current cursor position
    selected = set()      # selected indices
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
            "<hint>  ↑/↓ navigate   SPACE select/deselect   ENTER confirm   q cancel\n\n</hint>"
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
        lines.append(HTML(f"\n<hint>  {sel_count} checkpoint(s) selected</hint>\n"))
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


# ── Main menu ────────────────────────────────────────────────────────────────

def interactive_menu(ckpt_dir: Path) -> tuple[str, list[int]]:
    """
    Display interactive menu.
    Returns (mode, steps):
      mode: 'latest' | 'best' | 'choose' | 'list' | 'quit'
      steps: list of steps for 'choose' mode, empty for other modes
    """
    print("\n" + "=" * 45)
    print("  Upload Checkpoint to HuggingFace Hub")
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
            print(f"  Available checkpoints: {len(steps_avail)} ({min(steps_avail):,} – {max(steps_avail):,})")
    print()

    print("  Select upload mode:")
    print("  [1] Save Latest      — upload the latest checkpoint (highest step)")
    print("  [2] Best Checkpoint  — upload the checkpoint with the best val_bpb")
    print("  [3] Choose Checkpoints — manual selection (multi-select)")
    print("  [4] View all checkpoints")
    print("  [q] Exit")
    print()

    while True:
        try:
            choice = input("  Choice (1/2/3/4/q): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled.")
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
            print("  Invalid input. Enter 1, 2, 3, 4, or q.")


# ── Upload satu step ──────────────────────────────────────────────────────────

def upload_step(api, step: int, ckpt_dir: Path, repo: str, depth: str, source: str, model_only: bool):
    """Upload all files for one step. Returns (uploaded, total).

    Files are stored in the repo under:
      <source>/<depth>/model_XXXXXX.pt   (e.g. sft/d24/model_001000.pt)
    This keeps base, sft, and rl checkpoints cleanly separated.
    """
    step_str = f"{step:06d}"
    files = [f"model_{step_str}.pt", f"meta_{step_str}.json"]
    if not model_only:
        files.append(f"optim_{step_str}_rank0.pt")

    # Repo path prefix: e.g. "sft/d24/" or "base/d24/"
    repo_prefix = f"{source}/{depth}"

    uploaded = 0
    for filename in files:
        filepath = ckpt_dir / filename
        if not filepath.exists():
            print(f"  SKIP: {filename} not found")
            continue
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  Uploading {filename} ({size_mb:.1f} MB) → {repo_prefix}/{filename}")
        api.upload_file(
            path_or_fileobj=str(filepath),
            path_in_repo=f"{repo_prefix}/{filename}",
            repo_id=repo,
            repo_type="model",
        )
        uploaded += 1
        print(f"  ✓ {filename} uploaded")

    return uploaded, len(files)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Upload checkpoint to HuggingFace Hub")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--step", type=int, help="Checkpoint step to upload (e.g. 8000)")
    group.add_argument("--best", action="store_true", help="Automatically upload the checkpoint with the best val_bpb")
    group.add_argument("--latest", action="store_true", help="Upload the latest checkpoint (highest step)")
    group.add_argument("--list", action="store_true", help="List all checkpoints and their val_bpb")
    parser.add_argument("--depth", type=str, default="d24", help="Model depth tag (default: d24)")
    parser.add_argument("--source", type=str, default="base", choices=["base", "sft", "rl"],
                        help="Checkpoint source: base (pretraining), sft (chat SFT), rl (RL/GRPO). Default: base")
    parser.add_argument("--repo", type=str, default="Dummy9898/mesosfer-checkpoints", help="HF repo ID")
    parser.add_argument("--model-only", action="store_true", help="Upload model + meta only (skip optimizer state)")
    parser.add_argument("--base-dir", type=str, default=None, help="Override base cache dir")
    args = parser.parse_args()

    base_dir = args.base_dir or os.path.expanduser("~/.cache/mesosfer")

    # Map source to checkpoint subdirectory
    source_dir_map = {
        "base": "base_checkpoints",
        "sft":  "chatsft_checkpoints",
        "rl":   "chatrl_checkpoints",
    }
    ckpt_subdir = source_dir_map[args.source]
    ckpt_dir = Path(base_dir) / ckpt_subdir / args.depth

    if not ckpt_dir.exists():
        print(f"ERROR: Checkpoint directory not found: {ckpt_dir}")
        print(f"  source={args.source!r} maps to: {ckpt_subdir}/{args.depth}/")
        if args.source == "sft":
            print("  Make sure SFT training has completed and saved a checkpoint.")
        elif args.source == "rl":
            print("  Make sure RL training has completed and saved a checkpoint.")
        return

    # ── Interactive mode if no flags given ───────────────────────────────────
    chosen_steps: list[int] = []
    no_flag_given = not (args.step or args.best or args.latest or args.list)

    if no_flag_given:
        print(f"\n  Source: {args.source} ({ckpt_dir})")
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
                print("No checkpoints found.")
                return
            print("\n  Use SPACE to select, ENTER to confirm:\n")
            chosen_steps = checkbox_select(rows)
            if not chosen_steps:
                print("\nNo checkpoints selected. Cancelled.")
                return
            print(f"\n  Selected: {len(chosen_steps)} checkpoint(s) — {chosen_steps}")

    # Mode --list
    if args.list:
        print(f"Checkpoint dir: {ckpt_dir}")
        list_checkpoints(ckpt_dir)
        return

    # Determine the list of steps to upload
    steps_to_upload: list[int] = []

    if chosen_steps:
        # Choose mode: already determined from checkbox selection
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
        print(f"\nBest checkpoint: step {step} (val_bpb={best_bpb:.6f})")
        steps_to_upload = [step]
    elif args.latest:
        step, val_bpb = find_latest_checkpoint(ckpt_dir)
        bpb_info = f"val_bpb={val_bpb:.6f}" if isinstance(val_bpb, float) else "val_bpb=N/A"
        print(f"\nLatest checkpoint: step {step} ({bpb_info})")
        steps_to_upload = [step]
    elif args.step:
        step = args.step
        meta_path = ckpt_dir / f"meta_{step:06d}.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                val_bpb = json.load(f).get("val_bpb", "N/A")
            print(f"Uploading step {step} (val_bpb={val_bpb})")
        else:
            print(f"Uploading step {step} (meta not found)")
        steps_to_upload = [step]
    else:
        parser.print_help()
        return

    # ── Login HuggingFace ─────────────────────────────────────────────────────
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub is not installed. Run: pip install huggingface_hub")
        return

    api = HfApi()
    try:
        user = api.whoami()
        print(f"\nLogged in as: {user['name']}")
    except Exception as e:
        print(f"ERROR: Could not log in to HuggingFace — {e}")
        print("Run: hf auth login")
        return

    # Ensure repo exists
    api.create_repo(repo_id=args.repo, repo_type="model", private=True, exist_ok=True)
    print(f"Repo: {args.repo}\n")

    # ── Upload all selected steps ─────────────────────────────────────────────
    grand_uploaded = 0
    grand_total = 0

    for i, step in enumerate(steps_to_upload, 1):
        if len(steps_to_upload) > 1:
            print(f"── [{i}/{len(steps_to_upload)}] step {step:,} ──")
        uploaded, total = upload_step(api, step, ckpt_dir, args.repo, args.depth, args.source, args.model_only)
        grand_uploaded += uploaded
        grand_total += total
        if len(steps_to_upload) > 1:
            print()

    print(f"Done! {grand_uploaded}/{grand_total} file(s) uploaded to {args.repo}/{args.source}/{args.depth}/")


if __name__ == "__main__":
    main()
