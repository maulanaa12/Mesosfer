# Ozon

A minimal full-stack ChatGPT clone for training small-to-medium language models.

Ozon is inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) and follows Andrej Karpathy's educational approach to LLM training: clean, readable code that actually works.

## Features

- **Pretraining** — Train base models from scratch using scaling laws
- **Tokenizer Training** — Custom BPE tokenizer (32K vocab)
- **Supervised Fine-Tuning (SFT)** — Instruction tuning for conversational capability
- **Evaluation** — CORE benchmark metrics (MMLU, GSM8K, ARC, HumanEval, etc.)
- **Chat Interface** — CLI and WebUI for model interaction

## Architecture Highlights

| Feature | Implementation |
|---------|----------------|
| Attention | Group-Query Attention (GQA) with Flash Attention 3/2 |
| Positional Encoding | Rotary embeddings |
| Activation | ReLU² |
| Normalization | RMSNorm (no learnable parameters) |
| Optimizer | MuonAdamW (Muon for matrices, AdamW for embeddings) |
| Training Precision | BF16 (Ampere+) / FP8 (Hopper+) |
| Sliding Window | Configurable pattern (SLSLSL...) |

## Hardware Support

| Vendor | GPUs | Backend |
|--------|------|---------|
| NVIDIA | H100, H200, B200, A100, L40S, L4, RTX 4090, T4 | CUDA |
| AMD | MI355X, MI350X, MI325X, MI300X, MI250X, MI210 | ROCm 6.4+ |
| Apple | M1/M2/M3 Pro/Max | MPS |
| Intel/x86 | CPU | CPU |

> For detailed setup instructions, see:
> - [RUN_NVIDIA.md](RUN_NVIDIA.md) — NVIDIA GPU setup
> - [RUN_AMD.md](RUN_AMD.md) — AMD GPU setup

## Quick Start

### 1. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/your-repo/ozon.git
cd ozon

# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv
source .venv/bin/activate

# Install with GPU support (NVIDIA CUDA)
uv sync --extra gpu

# Or for AMD ROCm
# uv sync --extra rocm
```

### 2. Train a Model

```bash
# GPT-2 class model (~3 hours on 8x H100)
bash runs/speedrun.sh

# With Weights & Biases logging
WANDB_RUN=my_run bash runs/speedrun.sh
```

### 3. Chat with Your Model

```bash
# CLI chat
python -m scripts.chat_cli -p "Why is the sky blue?"

# Web UI (opens http://localhost:8000)
python -m scripts.chat_web
```

## Project Structure

```
ozon/
├── ozon/
│   ├── model/          # GPT model, attention, optimization
│   ├── data/           # Dataset download and preprocessing
│   ├── eval/           # Evaluation metrics
│   └── utils/          # Common utilities
├── scripts/
│   ├── train/          # Training scripts
│   ├── chat/           # Chat interface
│   ├── eval/           # Evaluation scripts
│   └── setup/          # Installation helpers
├── runs/               # Training run scripts
└── tests/              # Unit tests
```

## Training Scripts

| Script | Description | Hardware |
|--------|-------------|----------|
| `runs/speedrun.sh` | Full pipeline: pretrain + SFT + eval | 8x H100/A100 |
| `runs/scaling_laws.sh` | Research: optimal model configs | 8x GPU |
| `runs/miniseries.sh` | Train multiple depths (12-26) | 8x GPU |
| `runs/runcpu.sh` | Demo on CPU/MacBook | CPU/MPS |

## Command Line Arguments

```bash
python -m scripts.base_train -- \
    --depth 20              # Transformer layers
    --max-seq-len 2048      # Context length
    --device-batch-size 16  # Per-GPU batch
    --fp8                   # FP8 training (H100+)
    --run my_run            # wandb run name
```

### Key Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--depth` | 20 | Transformer depth |
| `--aspect-ratio` | 64 | Model dimension = depth × 64 |
| `--head-dim` | 128 | Attention head dimension |
| `--max-seq-len` | 2048 | Maximum context length |
| `--device-batch-size` | 32 | Batch size per GPU |
| `--fp8` | false | Enable FP8 (H100+ only) |
| `--target-param-data-ratio` | 12 | Data-to-parameters ratio |

## Configuration

### Environment Variables

```bash
# Backend selection (auto-detected if not set)
export OZON_TORCH_BACKEND=cuda    # cuda, rocm, cpu

# Compute dtype override
export ozon_DTYPE=bfloat16        # bfloat16, float16, float32

# Cache directory
export ozon_BASE_DIR="$HOME/.cache/ozon"

# Wandb logging
export WANDB_RUN=my_training_run
```

### Dataset Configuration

See [DATASET.md](DATASET.md) for dataset sampling weights and sources.

## Requirements

- Python 3.12+
- PyTorch 2.9.1+
- NVIDIA GPU (CUDA 12.8+) or AMD GPU (ROCm 6.4+)
- 16GB+ GPU memory recommended

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

## Model Performance

| Model | Parameters | Tokens | Validation BPB | CORE Score |
|-------|-----------|--------|----------------|------------|
| GPT-2 (reference) | ~124M | ~5B | ~0.97 | ~25 |
| Ozon (speedrun) | ~350M | ~2.8B | ~0.95 | ~28 |

> CORE score: average of MMLU (5-shot), GSM8K (COT), ARC-C, HumanEval (pass@1)

## Acknowledgments

- [Andrej Karpathy](https://karpathy.ai/) — nanoGPT inspiration
- [PyTorch](https://pytorch.org/) — Deep learning framework
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) — Efficient attention implementation
- [Karpathy's educational resources](https://makereading.com/) — LLM training insights

## License

MIT License