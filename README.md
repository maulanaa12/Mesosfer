# Mesosfer

A minimal full-stack cybersecurity-focused language model — from pretraining to chat.

Mesosfer is inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) and follows Andrej Karpathy's educational approach to LLM training: clean, readable code that actually works. It is specifically optimized for cybersecurity and secure coding domains.

## Features

- **Pretraining** — Train base models from scratch using scaling laws
- **Tokenizer Training** — Custom BPE tokenizer (64K vocab, cybersec-aware)
- **Supervised Fine-Tuning (SFT)** — Instruction tuning with cybersecurity datasets
- **Reinforcement Learning (RL)** — GRPO-style RL on cybersecurity tasks
- **RLHF Data Collection** — Human feedback via thumbs up/down UI, stored to `data/rlhf/`
- **Evaluation** — CORE benchmark + cybersecurity domain probes
- **Chat Interface** — CLI and WebUI for model interaction
  - Syntax-highlighted code blocks (Python, Rust, JSON, etc.)
  - Markdown rendering (headings, lists, bold/italic, inline code)
  - Welcome screen with centered input on new conversation

## Architecture Highlights

| Feature | Implementation |
|---------|----------------|
| Attention | Group-Query Attention (GQA) with Flash Attention 2/3 |
| Positional Encoding | Rotary embeddings (RoPE) |
| Activation | ReLU² |
| Normalization | RMSNorm (no learnable parameters) |
| Optimizer | MuonAdamW (Muon for matrices, AdamW for embeddings) |
| Training Precision | BF16 (all GPUs) / FP8 (Hopper/H100 only) |
| Value Embeddings | ResFormer-style alternating layers |

## Hardware Support

| Vendor | GPUs | Backend |
|--------|------|---------|
| NVIDIA | H100, H200, B200, A100, L40S, L4, RTX 4090, T4 | CUDA |
| AMD | MI355X, MI350X, MI325X, MI300X, MI250X, MI210 | ROCm 7.0+ |
| Apple | M1/M2/M3 Pro/Max | MPS |
| Intel/x86 | CPU | CPU |

> For detailed setup instructions, see:
> - [RUN_NVIDIA.md](RUN_NVIDIA.md) — NVIDIA GPU setup
> - [RUN_AMD.md](RUN_AMD.md) — AMD GPU setup (ROCm 7.0 pre-built image)

---

## Quick Start

### 1. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/your-repo/mesosfer.git
cd mesosfer

# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.12
source .venv/bin/activate

# NVIDIA CUDA
uv sync --extra gpu

# AMD ROCm 7.0 (pre-built image — PyTorch already installed)
# uv sync --extra rocm --no-build-isolation
# pip install flash-attn --no-build-isolation
```

---

### 2. Prepare Data

Data preparation must be completed **before** training. Run these steps in order:

#### Step 2a — Download ClimbMix general pretraining data (~17 GB)

```bash
# Download 170 shards of ClimbMix-400B (enough for depth 24 at ratio 10)
python -m mesosfer.data.dataset -n 170
```

#### Step 2b — Prepare cybersecurity dataset

Downloads and interleaves CVE feeds, HuggingFace cybersec datasets, and local files.
Output: `~/.cache/mesosfer/base_data_cybersecurity/`

```bash
python -m scripts.data.prepare_data

# Check progress
python -m scripts.data.prepare_data --status

# Dry-run to preview sources
python -m scripts.data.prepare_data --dry-run
```

> Steps 2a and 2b can run **in parallel** in separate terminals.

#### Step 2c — Convert raw security logs to natural language

Converts `data/log/` and `data/cloud/` files to NL narratives (prevents loss spikes).
Output: `data/log_nl/` and `data/cloud_nl/`

```bash
python -m scripts.data.convert_logs_to_nl

# Preview first 3 documents without writing
python -m scripts.data.convert_logs_to_nl --dry-run
```

> Step 2c can run in parallel with 2a and 2b. It only reads from `data/` in the repo.

---

### 3. Train Tokenizer

Train a 64K BPE tokenizer on the prepared data. Requires Step 2a to be complete.

```bash
python -m scripts.train.tok_train

# Evaluate tokenizer compression ratio vs GPT-2 and GPT-4
python -m scripts.eval.tok_eval
```

---

### 4. Pretrain Base Model

Requires Steps 2a, 2b, 2c, and 3 to be complete.

```bash
# Depth 24 — recommended config for MI300X / single GPU
python -m scripts.train.base_train \
    --depth=24 \
    --target-param-data-ratio=10 \
    --device-batch-size=32 \
    --warmup-steps=200 \
    --window-pattern=L \
    --save-every=1000 \
    --core-metric-every=5000 \
    --run=d24_run

# Or use the full pipeline script (handles setup + tokenizer + pretrain + SFT)
WANDB_RUN=my_run bash runs/speedrun.sh
```

> **AMD ROCm note:** Use `--window-pattern=L` (full attention). Sliding window (`SSL`, `SSSL`)
> is not yet supported in the ROCm FA2 Triton backward pass.

---

### 5. Evaluate Base Model

```bash
python -m scripts.eval.base_eval \
    --model-tag d24 \
    --device-batch-size 32
```

---

### 6. Supervised Fine-Tuning (SFT)

Requires Step 4 to be complete. Automatically includes cybersecurity SFT datasets
from `data/sft/`.

```bash
python -m scripts.chat.chat_sft \
    --device-batch-size=32 \
    --run=sft_run

# Disable cybersec SFT (for ablation)
# python -m scripts.chat.chat_sft --disable-cybersec-sft
```

---

### 7. Chat with Your Model

```bash
# CLI chat
python -m scripts.chat.chat_cli -p "Explain CVE-2021-44228 (Log4Shell)"

# Web UI (opens http://localhost:8000)
python -m scripts.chat.chat_web

# Web UI — multi-GPU (4 workers)
python -m scripts.chat.chat_web --num-gpus 4

# Web UI — load specific checkpoint
python -m scripts.chat.chat_web --model-tag d24 --step 14000
```

The Web UI includes:
- **Welcome screen** — centered input shown before the first message
- **Syntax-highlighted code blocks** — Python, Rust, JSON, Bash, and more, with a one-click copy button
- **Markdown rendering** — headings, lists, bold/italic, inline code
- **Thumbs up / down feedback** — per-response rating that saves to `data/rlhf/feedback.jsonl`

---

### 8. Collect RLHF Feedback

Human preference data is collected automatically while chatting via the Web UI.
Each 👍 or 👎 click on an assistant response appends a record to `data/rlhf/feedback.jsonl`.

```jsonc
{
  "timestamp": "2026-05-20T10:00:00+00:00",
  "message_index": 1,
  "rating": "negative",
  "reason": "factually_incorrect",
  "comment": "The CVE number is wrong.",
  "conversation": [
    { "role": "user",      "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```

Use this data to train a reward model or run DPO fine-tuning in a future step.

---

## Web UI API Endpoints

The Web UI server (`scripts/chat/chat_web.py`) exposes the following endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serve the chat UI (`mesosfer/ui.html`) |
| `GET` | `/logo.svg` | Serve the Mesosfer logo |
| `GET` | `/interface/*` | Serve static assets (CSS, JS) |
| `POST` | `/chat/completions` | Streaming chat completion (SSE) |
| `POST` | `/feedback` | Submit thumbs up/down feedback (saved to `data/rlhf/feedback.jsonl`) |
| `GET` | `/health` | Health check + worker pool status |
| `GET` | `/stats` | Worker pool statistics and GPU utilization |

### Feedback endpoint payload

```json
POST /feedback
{
  "message_index": 1,
  "rating": "negative",
  "reason": "factually_incorrect",
  "comment": "Optional free-text",
  "conversation": [
    { "role": "user",      "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```

Valid `reason` values: `inappropriate_response`, `continuous_repetition`, `factually_incorrect`, `too_verbose`, `formatting_issues`, `other`.

---

## Full Pipeline Summary

```
Step 2a: mesosfer.data.dataset -n 170     ─┐
Step 2b: scripts.data.prepare_data         ├─ can run in parallel
Step 2c: scripts.data.convert_logs_to_nl  ─┘
         ↓
Step 3:  scripts.train.tok_train
         scripts.eval.tok_eval
         ↓
Step 4:  scripts.train.base_train
         ↓
Step 5:  scripts.eval.base_eval
         ↓
Step 6:  scripts.chat.chat_sft
         ↓
Step 7:  scripts.chat.chat_cli / chat_web
         ↓
Step 8:  Collect RLHF feedback via Web UI  →  data/rlhf/feedback.jsonl
         (future: reward model training / DPO)
```

---

## Project Structure

```
mesosfer/
├── mesosfer/               # Library (importable)
│   ├── model/              # GPT model, attention, optimization
│   ├── data/               # Dataset download, dataloader, tokenizer
│   ├── eval/               # CORE eval, BPB, engine
│   ├── utils/              # Common utilities, checkpointing, reporting
│   ├── interface/          # Web UI static assets
│   │   ├── style.css       # All UI styles (chat, code blocks, feedback, empty state)
│   │   ├── chat.js         # Chat logic, streaming, slash commands, markdown rendering
│   │   ├── feedback.js     # Thumbs up/down, feedback modal, POST /feedback
│   │   └── markdown.js     # Markdown parser + syntax-highlighted code block renderer
│   └── ui.html             # HTML shell (loads interface/ assets)
├── scripts/
│   ├── train/              # base_train.py, tok_train.py
│   ├── chat/               # chat_sft.py, chat_rl.py, chat_cli.py, chat_web.py
│   ├── eval/               # base_eval.py, tok_eval.py
│   └── data/               # prepare_data.py, convert_logs_to_nl.py
├── tasks/                  # Eval tasks (MMLU, GSM8K, cybersec_sft, cybersec_rl)
├── data/
│   ├── log/                # Raw security logs (input to convert_logs_to_nl)
│   ├── cloud/              # Raw cloud audit logs (input to convert_logs_to_nl)
│   ├── log_nl/             # NL narratives from logs (output, used in training)
│   ├── cloud_nl/           # NL narratives from cloud logs (output, used in training)
│   ├── sft/                # Cybersecurity SFT conversations
│   ├── rlhf/               # Human feedback collected via Web UI
│   │   └── feedback.jsonl  # Appended at runtime (gitignored)
│   ├── synthetic-ir/       # Synthetic incident response data
│   ├── synthetic-soc/      # Synthetic SOC analyst data
│   └── reverse-engineering/ # RE/exploitation analysis data
├── runs/                   # Training run scripts (speedrun, miniseries, etc.)
└── tests/                  # Unit tests
```

---

## Training Scripts

| Script | Description | Hardware |
|--------|-------------|----------|
| `runs/speedrun.sh` | Full pipeline: pretrain + SFT + eval | 8x H100/A100 or 1x MI300X |
| `runs/scaling_laws.sh` | Research: optimal model configs | 8x GPU |
| `runs/miniseries.sh` | Train multiple depths (12-26) | 8x GPU |
| `runs/runcpu.sh` | Demo on CPU/MacBook | CPU/MPS |

---

## Key Training Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--depth` | 20 | Transformer depth (24 recommended) |
| `--aspect-ratio` | 64 | Model dimension = depth × 64 |
| `--head-dim` | 128 | Attention head dimension |
| `--max-seq-len` | 2048 | Maximum context length |
| `--device-batch-size` | 32 | Batch size per GPU |
| `--target-param-data-ratio` | 12 | Data-to-parameters ratio (10 recommended) |
| `--warmup-steps` | 200 | LR warmup steps (200 for depth 20+) |
| `--window-pattern` | L | Attention window: L=full, S=sliding (L required for ROCm) |
| `--save-every` | 1000 | Checkpoint every N steps |
| `--core-metric-every` | 5000 | CORE eval every N steps |

---

## Configuration

### Environment Variables

```bash
# Backend selection (auto-detected if not set)
export mesosfer_TORCH_BACKEND=cuda    # cuda, rocm, cpu

# Compute dtype override
export mesosfer_DTYPE=bfloat16        # bfloat16, float16, float32

# Cache directory (datasets, checkpoints, tokenizer)
export mesosfer_BASE_DIR="$HOME/.cache/mesosfer"

# Wandb logging
export WANDB_RUN=my_training_run
```

### Dataset Configuration

See [DATASET.md](DATASET.md) for dataset sources, sampling weights, and token budgets.
See [DATASET2.md](DATASET2.md) for advanced configuration and dynamic source definitions.

---

## Requirements

- Python 3.12+
- PyTorch 2.6.0+ (ROCm 7.0) or 2.9.1+ (CUDA 12.8)
- NVIDIA GPU (CUDA 12.8+) or AMD GPU (ROCm 7.0+)
- 16GB+ GPU VRAM (32GB+ recommended for depth 24)

---

## Development

```bash
# Install dev dependencies
uv sync --dev

# Run tests
pytest tests/

# Format code
ruff format .

# Lint
ruff check .
```

---

## Model Performance

| Model | Parameters | Tokens | Validation BPB | CORE Score |
|-------|-----------|--------|----------------|------------|
| GPT-2 (reference) | ~124M | ~5B | ~0.97 | ~25 |
| mesosfer d24 (ratio=10) | ~1.38B | ~7.3B | 0.7337 | 0.2541 |

> CORE score: average of MMLU (5-shot), GSM8K (COT), ARC-C, HumanEval (pass@1)

---

## Scaling to Larger Models

The architecture is designed to scale without fundamental changes. Model size is controlled by two arguments:

```
model_dim = depth × aspect_ratio  (rounded up to nearest multiple of head_dim)
num_heads = model_dim / head_dim
```

### Depth × Aspect Ratio → Parameter Count

| Depth | Aspect Ratio | Model Dim | ~Total Params | Dataset needed (ratio=10) | Covered by ~8.5B dataset? |
|-------|-------------|-----------|---------------|---------------------------|---------------------------|
| 16 | 64 | 1024 | ~0.9B | ~2.7B | ✅ |
| 18 | 64 | 1152 | ~1.2B | ~3.6B | ✅ |
| 20 | 64 | 1280 | ~1.5B | ~4.8B | ✅ |
| 22 | 64 | 1408 | ~1.8B | ~6.2B | ✅ |
| 24 | 64 | 1536 | ~2.2B | ~7.8B | ✅ |
| 28 | 64 | 1792 | ~3.1B | ~12.0B | ❌ need more data |
| 32 | 128 | 4096 | ~11.5B | ~67B | ❌ need more data |
| 36 | 128 | 4608 | ~15.5B | ~95B | ❌ need more data |
| 40 | 128 | 5120 | ~20.3B | ~129B | ❌ need more data |
| 44 | 128 | 5632 | ~26.0B | ~171B | ❌ need more data |
| 48 | 128 | 6144 | ~32.6B | ~222B | ❌ need more data |

> "Dataset needed" = scaling params × ratio=10. Scaling params = transformer matrices + lm_head (excludes embeddings).
> Current dataset (~8.5B tokens) fully covers depth 16–24. For depth 28+, additional data sources are required.
> For Chinchilla-optimal training (ratio=20), double the token requirements above.

To train a larger model, simply pass the desired depth and aspect ratio:

```bash
# ~3B model (depth 28, aspect-ratio 64)
python -m scripts.train.base_train \
    --depth=28 \
    --aspect-ratio=64 \
    --head-dim=128 \
    --target-param-data-ratio=10 \
    --run=d28_run

# ~7B model (depth 40, aspect-ratio 64)
python -m scripts.train.base_train \
    --depth=40 \
    --aspect-ratio=64 \
    --head-dim=128 \
    --target-param-data-ratio=10 \
    --run=d40_run
```

Features like GQA, Flash Attention 2/3, RoPE, RMSNorm, and BF16/FP8 are already implemented and designed for large-scale training. The main practical constraints for 7B+ models are **data volume** (current dataset ~8.5B tokens covers up to ~850M scaling params at ratio=10) and **hardware** (7B in BF16 requires ~14GB weights + optimizer state, recommend 2–4× A100/H100 80GB or 1× MI300X).

---

## Acknowledgments

- [Andrej Karpathy](https://karpathy.ai/) — nanoGPT inspiration
- [PyTorch](https://pytorch.org/) — Deep learning framework
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) — Efficient attention
- [Muon optimizer](https://kellerjordan.github.io/posts/muon/) — Matrix parameter optimization

## License

MIT License
