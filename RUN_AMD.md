# Running Ozon on AMD GPUs

This guide covers the setup and execution of Ozon on AMD GPUs using ROCm.

## Prerequisites

### Hardware

| GPU | Architecture | Compute Unit | BF16 FLOPS | Memory | Notes |
|-----|-------------|--------------|------------|--------|-------|
| MI355X | CDNA 4 | ~304 | 2500 TFLOPS | 256GB HBM3 | Latest, compute-focused |
| MI350X | CDNA 3 | ~304 | 1.6 PetaFLOPS | 256GB HBM3 | Compute powerhouse |
| MI325X | CDNA 3 | ~304 | 1.3 PetaFLOPS | 256GB HBM3 | Good value |
| MI300X | CDNA 3 | ~228 | 1.3 PetaFLOPS | 192GB HBM3 | Popular for inference |
| MI300A | CDNA 3 | ~228 | 980 TFLOPS | 192GB HBM3 | APU variant |
| MI250X | CDNA 2 | ~208 | 383 TFLOPS | 128GB HBM2 | Older compute GPU |
| MI250 | CDNA 2 | ~208 | 362 TFLOPS | 128GB HBM2 | MI250X variant |
| MI210 | CDNA 2 | ~104 | 181 TFLOPS | 64GB HBM2 | Entry compute GPU |

> **Note:** AMD ROCm GPUs all support BF16 natively, including MI210 (older generation).

### Software

- **OS:** Linux only (Ubuntu 20.04+, RHEL/CentOS 8+, SLES 15+)
- **Python:** 3.12+
- **ROCm:** 6.4+ (required for MI355X/MI350X/MI325X, recommended for all)
- **ROCm Driver:** Included with ROCm installation

> **Important:** ROCm is Linux-only. Windows support via WSL2 is experimental.

### Check ROCm Installation

```bash
# Check ROCm version
rocm-smi --version

# List available GPUs
rocm-smi

# Check rocminfo
rocminfo | grep -A 10 "Agent"
```

---

## Installation

### 1. Install ROCm (if not already installed)

Follow the official ROCm installation guide for your distribution:

**Ubuntu:**
```bash
# Add ROCm repository
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.4/ jammy main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rocm-llvm rocm-smi rocminfo

# For MI355X/MI350X/MI325X:
sudo apt install rocm-6.4.0
```

**For detailed installation:** [ROCm Installation Guide](https://rocm.docs.amd.com/projects/install/en/latest/)

### 2. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Create Virtual Environment

```bash
cd /path/to/ozon
uv venv
source .venv/bin/activate
```

### 4. Install with ROCm Support

```bash
uv sync --extra rocm
```

This installs:
- PyTorch with ROCm 6.4 support
- `pytorch-triton-rocm==3.5.1` for Triton on AMD

### 5. Verify Installation

```bash
# Check ROCm PyTorch
python -c "import torch; print(f'ROCm available: {torch.cuda.is_available()}, Version: {torch.version.hip}')"

# Verify GPU detected
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

---

## Environment Variables

### Required

```bash
# Set backend explicitly (required for AMD)
export OZON_TORCH_BACKEND=rocm

# Optimize memory allocator
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# Limit OpenMP threads
export OMP_NUM_THREADS=1

# ROCm specific
export HIP_VISIBLE_DEVICES=0,1,2,3  # Select GPUs (0-indexed)
```

### Optional

```bash
# Override compute dtype (bfloat16 recommended for AMD)
export ozon_DTYPE=bfloat16

# Cache directory (default: ~/.cache/ozon)
export ozon_BASE_DIR="$HOME/.cache/ozon"

# Weights & Biases logging
export WANDB_RUN=my_training_run
# To enable wandb: wandb login

# ROCm tuning
export ROCM_BLIS_LC=1  # Enable ROCm BLIS optimizations
```

### Multi-GPU Setup

```bash
# For 8-GPU MI300X node:
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# For specific GPU selection:
export HIP_VISIBLE_DEVICES=0,2,4,6  # Use GPUs 0, 2, 4, 6 only
```

---

## GPU-Specific Notes

### MI355X / MI350X / MI325X (CDNA 3/4)

Latest AMD compute GPUs with highest performance:
- Full BF16 support
- Flash Attention via ROCm libraries
- ROCm 6.4+ required

```bash
export OZON_TORCH_BACKEND=rocm
export ROCM_BLIS_LC=1

# 8-GPU configuration
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 \
    --device-batch-size=16 \
    --run=$WANDB_RUN
```

### MI300X / MI300A

Popular for large-scale inference and training:
- 8 GCDs (MI300X) or 4 GCDs (MI300A)
- 192GB HBM3 per GPU
- ROCm 6.0+ recommended

```bash
# MI300X (8 GCDs)
export HIP_VISIBLE_DEVICES=0,1  # Each GCD pair is one "device"
# Note: MI300X exposes 8 GCDs as 2 "devices" in PyTorch

# MI300A (4 GCDs)
export HIP_VISIBLE_DEVICES=0
```

### MI250X / MI250

Older but capable compute GPUs:
- 2 GCDs per GPU
- 128GB HBM2 per GPU
- ROCm 5.7+ recommended

```bash
# Each GCD is a separate "device" in PyTorch
# MI250X shows as 2 devices
export HIP_VISIBLE_DEVICES=0,1,2,3  # First 2 GPUs (4 GCDs)
```

### MI210

Entry-level compute GPU:
- Single GCD
- 64GB HBM2
- Good for single-GPU training or development

```bash
# Single MI210
export HIP_VISIBLE_DEVICES=0

python -m scripts.base_train -- \
    --depth=12 \
    --device-batch-size=8 \
    --run=dummy
```

---

## Training Scripts

### Quick Start: GPT-2 Class Training

**Note:** Training times on AMD GPUs vary by GPU model. MI300X 8x similar to H100.

```bash
# Basic run (no wandb logging)
bash runs/speedrun.sh

# With wandb logging
WANDB_RUN=my_run bash runs/speedrun.sh

# In a screen session
screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh
```

> **Note:** Some scripts default to `--extra gpu` (NVIDIA). For AMD, ensure your `uv sync` used `--extra rocm`.

### Scaling Laws Research

```bash
export NPROC_PER_NODE=8
export WANDB_RUN=scaling_rocm
export ozon_BASE_DIR="$HOME/.cache/ozon"

bash runs/scaling_laws.sh
```

### Miniseries (Multiple Depth Sweep)

```bash
# Default series
bash runs/miniseries.sh

# Custom name
bash runs/miniseries.sh mi300x_run

# Skip setup (if already done)
SKIP_SETUP=1 bash runs/miniseries.sh
```

### CPU Demo

```bash
bash runs/runcpu.sh
```

---

## Command Line Arguments

Common arguments for `scripts/base_train.py`:

| Argument | Default | Description |
|----------|---------|-------------|
| `--depth` | 20 | Transformer depth |
| `--aspect-ratio` | 64 | model_dim = depth * aspect_ratio |
| `--head-dim` | 128 | Attention head dimension |
| `--max-seq-len` | 2048 | Context length |
| `--device-batch-size` | 32 | Per-GPU batch size |
| `--total-batch-size` | -1 | Auto-compute if -1 |
| `--target-param-data-ratio` | 12 | Chinchilla-optimal ~20 |
| `--target-flops` | -1 | Fixed FLOP budget |
| `--num-iterations` | -1 | Auto-compute if -1 |
| `--run` | dummy | wandb run name |

### Batch Size Guidelines by GPU

```bash
# MI355X/MI350X/MI325X (256GB memory)
--device-batch-size=16

# MI300X (192GB memory)
--device-batch-size=12

# MI250X (128GB memory)
--device-batch-size=8

# MI210 (64GB memory)
--device-batch-size=4
```

---

## Post-Training: Chat Interface

```bash
# CLI Chat
python -m scripts.chat_cli -p "Why is the sky blue?"

# Web UI
python -m scripts.chat_web
```

---

## Troubleshooting

### "ROCm not available" Error

1. **Verify ROCm installation:**
   ```bash
   rocm-smi
   ```

2. **Check PyTorch ROCm support:**
   ```bash
   python -c "import torch; print(f'ROCm: {torch.cuda.is_available()}, Version: {torch.version.hip}')"
   ```

3. **Check HIP_VISIBLE_DEVICES:**
   ```bash
   echo $HIP_VISIBLE_DEVICES
   rocm-smi
   ```

4. **Reinstall with rocm extra:**
   ```bash
   uv sync --extra rocm
   ```

### Out of Memory (OOM)

1. **Reduce batch size:**
   ```bash
   --device-batch-size=4
   ```

2. **Reduce sequence length:**
   ```bash
   --max-seq-len=1024
   ```

3. **Reduce model size:**
   ```bash
   --depth=8
   --head-dim=64
   ```

4. **Check memory:**
   ```bash
   rocm-smi --showmeminfo vram
   ```

### Flash Attention Issues

AMD uses ROCm's implementation of Flash Attention:

```bash
# Check attention backend at startup
# Should show ROCm FA or SDPA fallback

# If issues, ensure ROCm 6.4+:
sudo apt install rocm-6.4.0

# Or install triton-rocm explicitly:
pip install pytorch-triton-rocm==3.5.1
```

### Multi-GPU Training Fails

1. **Verify NCCL (ROCm variant):**
   ```bash
   python -c "import torch; print(torch.distributed.is_nccl_available())"
   ```

2. **Check all GPUs visible:**
   ```bash
   rocm-smi
   python -c "import torch; print(torch.cuda.device_count())"
   ```

3. **Test with fewer GPUs:**
   ```bash
   export HIP_VISIBLE_DEVICES=0,1  # Only 2 GPUs
   torchrun --standalone --nproc_per_node=2 -m scripts.base_train -- ...
   ```

### Performance Issues

1. **Enable BLIS optimizations:**
   ```bash
   export ROCM_BLIS_LC=1
   ```

2. **Check MFU in logs:**
   ```bash
   # Model FLOP Utilization should be 45-55% for well-tuned runs
   ```

3. **Profile with ROCm tools:**
   ```bash
   # Enable ROCM profiling
   export ROCM_PROFILER_ENABLE=1
   ```

### Common ROCm Errors

| Error | Solution |
|-------|----------|
| `HIP error: hipErrorNoBinaryForGpu` | Update ROCm, check GPU compatibility |
| `ROCm version mismatch` | Ensure PyTorch ROCm version matches installed ROCm |
| `Out of memory` | Reduce batch size, sequence length, or model depth |
| `NCCL timeout` | Increase NCCL timeout, check network connectivity |

---

## GPU Comparison Table

| GPU | CDNA | Memory | BF16 FLOPS | GCDs | Best For |
|-----|------|--------|------------|-------|----------|
| MI355X | 4 | 256GB HBM3 | 2.5 PFLOPS | 8 | Research/compute |
| MI350X | 3 | 256GB HBM3 | 1.6 PFLOPS | 8 | Research/compute |
| MI325X | 3 | 256GB HBM3 | 1.3 PFLOPS | 8 | Balanced compute |
| MI300X | 3 | 192GB HBM3 | 1.3 PFLOPS | 8 | LLM inference/training |
| MI300A | 3 | 192GB HBM3 | 980 TFLOPS | 4 | APU workloads |
| MI250X | 2 | 128GB HBM2 | 383 TFLOPS | 2 | Older compute |
| MI210 | 2 | 64GB HBM2 | 181 TFLOPS | 1 | Entry/dev workloads |

---

## Quick Verification

```bash
# 1. Verify ROCm detection
rocm-smi

# 2. Verify PyTorch ROCm
python -c "import torch; print(f'ROCm: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}, Version: {torch.version.hip}')"

# 3. Verify compute dtype (should be bf16)
python -c "import torch; print(f'Compute dtype: bfloat16')"

# 4. Quick test run
export OZON_TORCH_BACKEND=rocm
python -m scripts.base_train --depth=4 --num-iterations=10 --run=dummy --device-batch-size=1
```

---

## Additional Resources

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [ROCm Installation Guide](https://rocm.docs.amd.com/projects/install/en/latest/)
- [PyTorch ROCm Support](https://pytorch.org/)