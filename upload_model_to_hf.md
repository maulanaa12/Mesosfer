# Upload Checkpoint to HuggingFace Hub

Script to upload training checkpoints (model weights, optimizer state, and metadata) to HuggingFace Hub, with both interactive and CLI modes.

---

## Prerequisites

### 1. Install dependencies

```bash
pip install huggingface_hub prompt_toolkit
```

### 2. Login to HuggingFace

```bash
hf auth login
```

Generate a token at https://huggingface.co/settings/tokens with `write` permission.

---

## Files Uploaded

Each checkpoint consists of 3 files:

| File | Description |
|------|-------------|
| `model_XXXXXX.pt` | Model weights |
| `meta_XXXXXX.json` | Training metadata (step, val_bpb, loss, etc.) |
| `optim_XXXXXX_rank0.pt` | Optimizer state (required for resuming training) |

> Use `--model-only` to skip the optimizer state for faster uploads (cannot resume training without it).

---

## Usage

### Interactive Mode (recommended)

Run without arguments to launch the interactive menu:

```bash
python scripts/upload_checkpoint_to_hf.py
```

```
=============================================
  Upload Checkpoint to HuggingFace Hub
=============================================
  Checkpoint dir: ~/.cache/mesosfer/base_checkpoints/d24

  Checkpoints available: 5 (2,000 – 10,000)

  Select upload mode:
  [1] Save Latest        — upload the most recent checkpoint (highest step)
  [2] Best Checkpoint    — upload the checkpoint with the lowest val_bpb
  [3] Choose Checkpoints — manually select checkpoints (multi-select)
  [4] List all checkpoints
  [q] Quit

  Choice (1/2/3/4/q):
```

---

### Menu Options

#### [1] Save Latest
Uploads the checkpoint with the highest step number (most recent).

#### [2] Best Checkpoint
Uploads the checkpoint with the lowest `val_bpb` (best performance).

#### [3] Choose Checkpoints — Multi-Select
Displays all available checkpoints as an interactive checkbox list. Multiple checkpoints can be selected.

```
  ↑/↓ navigate   SPACE select/deselect   ENTER confirm   q cancel

    [ ] step   2,000   val_bpb=1.234567
    [x] step   4,000   val_bpb=1.198432
  ▶ [x] step   6,000   val_bpb=1.187654 ← BEST
    [ ] step   8,000   val_bpb=1.201234
    [ ] step  10,000   val_bpb=1.195678

  2 checkpoints selected
```

**Navigation controls:**

| Key | Action |
|-----|--------|
| `↑` / `k` | Move up |
| `↓` / `j` | Move down |
| `SPACE` | Toggle select / deselect |
| `ENTER` | Confirm and start upload |
| `q` / `ESC` | Cancel |

After confirming, all selected checkpoints are uploaded sequentially:

```
── [1/2] step 4,000 ──
  Uploading model_004000.pt (312.4 MB)...
  ✓ model_004000.pt uploaded
  Uploading meta_004000.json (0.0 MB)...
  ✓ meta_004000.json uploaded
  Uploading optim_004000_rank0.pt (624.8 MB)...
  ✓ optim_004000_rank0.pt uploaded

── [2/2] step 6,000 ──
  Uploading model_006000.pt (312.4 MB)...
  ...

Done! 6/6 files uploaded to Dummy9898/mesosfer-checkpoints/d24/
```

#### [4] List all checkpoints
Displays a table of all checkpoints with their `val_bpb` values — no upload performed.

---

### CLI Mode (non-interactive)

#### Upload the latest checkpoint (highest step)

```bash
python scripts/upload_checkpoint_to_hf.py --latest
```

#### Upload the best checkpoint (lowest val_bpb)

```bash
python scripts/upload_checkpoint_to_hf.py --best
```

#### Upload a specific step

```bash
python scripts/upload_checkpoint_to_hf.py --step 8000
```

#### List all available checkpoints

```bash
python scripts/upload_checkpoint_to_hf.py --list
```

Example output:
```
Step       val_bpb      Status
-----------------------------------
2000       1.234567
4000       1.198432
6000       1.187654     ← BEST
8000       1.201234
10000      1.195678
```

---

## Additional Options

| Flag | Default | Description |
|------|---------|-------------|
| `--depth` | `d24` | Model depth tag (subfolder inside the HF repo) |
| `--repo` | `Dummy9898/mesosfer-checkpoints` | HuggingFace repo ID |
| `--model-only` | `false` | Skip optimizer state, upload model + meta only |
| `--base-dir` | `~/.cache/mesosfer` | Override the checkpoint directory path |

### Examples with custom options

```bash
# Upload best checkpoint to a different repo, skip optimizer
python scripts/upload_checkpoint_to_hf.py --best \
    --repo username/my-model \
    --depth d12 \
    --model-only

# Upload from a custom directory
python scripts/upload_checkpoint_to_hf.py --latest \
    --base-dir /mnt/storage/mesosfer
```

---

## HuggingFace Repo Structure

Files are uploaded into a subfolder based on `--depth`:

```
Dummy9898/mesosfer-checkpoints/
└── d24/
    ├── model_004000.pt
    ├── meta_004000.json
    ├── optim_004000_rank0.pt
    ├── model_006000.pt
    ├── meta_006000.json
    └── optim_006000_rank0.pt
```

---

## Troubleshooting

**`ERROR: Checkpoint directory not found`**
- Make sure training has run and saved at least one checkpoint
- Check the path: `~/.cache/mesosfer/base_checkpoints/d24/`
- Use `--base-dir` if checkpoints are stored elsewhere

**`ERROR: Cannot login to HuggingFace`**
- Run `hf auth login` and enter a valid token
- Make sure the token has `write` permission

**`SKIP: optim_XXXXXX_rank0.pt not found`**
- The optimizer file is missing from the checkpoint directory
- Use `--model-only` to skip it
- Check if the optimizer filename format differs (e.g., multi-rank training)

**`ERROR: prompt_toolkit is not installed`**
- Run `pip install prompt_toolkit`
- Only required for the Choose Checkpoints mode (option [3])
