#!/usr/bin/env python3
"""
Download and convert external HuggingFace datasets into mesosfer SFT format.

Each output file is a JSONL where every line is a JSON array of
{"role": "user"|"assistant", "content": "..."} message objects.

Usage:
    python -m scripts.data.download_sft_data
    python -m scripts.data.download_sft_data --sources primus_instruct ultrachat
    python -m scripts.data.download_sft_data --list
"""

import os
import json
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load .env file (same pattern as prepare_data.py)
def _load_env_file():
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value

_load_env_file()

try:
    from datasets import load_dataset
except ImportError:
    logger.error("Install 'datasets': pip install datasets")
    raise

_REPO_ROOT = Path(__file__).resolve().parents[2]
SFT_DIR = _REPO_ROOT / "data" / "sft"

# =============================================================================
# Source definitions
# =============================================================================

SOURCES = {
    # -------------------------------------------------------------------------
    # Primus-Instruct (gated — requires HF token + accept terms)
    # ~100K high-quality cybersecurity instruction pairs from Trend Micro
    "primus_instruct": {
        "hf_name": "trendmicro-ailab/Primus-Instruct",
        "split": "train",
        "streaming": True,
        "format": "messages",
        "output_file": "primus_instruct.jsonl",
        "max_rows": 100_000,
        "gated": True,
    },
    # -------------------------------------------------------------------------
    # Primus-Reasoning (gated — requires HF token + accept terms)
    # Reasoning distillation from o1-preview on cybersecurity tasks
    "primus_reasoning": {
        "hf_name": "trendmicro-ailab/Primus-Reasoning",
        "split": "train",
        "streaming": True,
        "format": "messages",
        "output_file": "primus_reasoning.jsonl",
        "max_rows": 50_000,
        "gated": True,
    },
    # -------------------------------------------------------------------------
    # CyberNative Code Vulnerability DPO
    # 4.6K vulnerable vs fixed code pairs — convert to instruction format
    "cybernative_vuln_dpo": {
        "hf_name": "CyberNative/Code_Vulnerability_Security_DPO",
        "split": "train",
        "streaming": False,
        "format": "dpo_to_sft",
        "output_file": "cybernative_vuln_dpo_sft.jsonl",
        "max_rows": 10_000,
        "gated": False,
    },
    # -------------------------------------------------------------------------
    # OpenHermes-2.5 (general instruction, includes security/coding)
    # Large dataset — we cap at 50K rows to avoid dominating the SFT mix
    "openhermes": {
        "hf_name": "teknium/OpenHermes-2.5",
        "split": "train",
        "streaming": True,
        "format": "messages",
        "output_file": "openhermes_sft.jsonl",
        "max_rows": 50_000,
        "gated": False,
    },
    # -------------------------------------------------------------------------
    # UltraChat 200K — high-quality multi-turn conversations
    "ultrachat": {
        "hf_name": "HuggingFaceH4/ultrachat_200k",
        "split": "train_sft",
        "streaming": True,
        "format": "messages",
        "output_file": "ultrachat_sft.jsonl",
        "max_rows": 100_000,
        "gated": False,
    },
    # -------------------------------------------------------------------------
    # Trendyol Cybersecurity Instruction Tuning — 53K rows, defensive cybersec
    "trendyol_cyber_sft": {
        "hf_name": "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset",
        "split": "train",
        "streaming": False,
        "format": "system_user_assistant",
        "output_file": "trendyol_cyber_sft.jsonl",
        "max_rows": 53_000,
        "gated": False,
    },
    # -------------------------------------------------------------------------
    # Tiamz cybersecurity instruction dataset — 12K Q&A pairs
    "tiamz_cybersec": {
        "hf_name": "Tiamz/cybersecurity-instruction-dataset",
        "split": "train",
        "streaming": False,
        "format": "instruction_answer",
        "output_file": "tiamz_cybersec_sft.jsonl",
        "max_rows": 15_000,
        "gated": False,
    },
}

# =============================================================================
# Format converters
# =============================================================================

def _extract_messages_field(row):
    """Try common field names for conversation lists."""
    for key in ("messages", "conversations", "conversation", "dialogue"):
        val = row.get(key)
        if isinstance(val, list) and len(val) >= 2:
            return val
    return None


def _normalize_role(role: str) -> str | None:
    role = str(role).lower().strip()
    if role in ("user", "human"):
        return "user"
    if role in ("assistant", "gpt", "bot", "model"):
        return "assistant"
    return None  # system, tool, etc. — skip


def _convert_messages(raw_messages) -> list | None:
    """Convert raw message list to mesosfer format. Returns None if invalid."""
    result = []
    for msg in raw_messages:
        if not isinstance(msg, dict):
            return None
        role = _normalize_role(msg.get("role") or msg.get("from") or "")
        content = msg.get("content") or msg.get("value") or msg.get("text") or ""
        if not role or not isinstance(content, str) or not content.strip():
            continue
        result.append({"role": role, "content": content.strip()})

    # Validate: must alternate user/assistant starting with user
    if len(result) < 2:
        return None
    for i, msg in enumerate(result):
        expected = "user" if i % 2 == 0 else "assistant"
        if msg["role"] != expected:
            return None
    return result


def convert_messages_format(row) -> list | None:
    """Handle datasets with a messages/conversations field."""
    raw = _extract_messages_field(row)
    if raw is None:
        return None
    return _convert_messages(raw)


def convert_dpo_to_sft(row) -> list | None:
    """Convert DPO format (prompt + chosen) to SFT conversation."""
    prompt = row.get("prompt") or row.get("instruction") or row.get("question") or ""
    chosen = row.get("chosen") or row.get("accepted") or row.get("response") or ""

    # chosen can be a list of messages in some DPO datasets
    if isinstance(chosen, list):
        return _convert_messages(chosen)

    if not prompt or not chosen:
        return None
    if not isinstance(prompt, str) or not isinstance(chosen, str):
        return None

    return [
        {"role": "user", "content": prompt.strip()},
        {"role": "assistant", "content": chosen.strip()},
    ]


CONVERTERS = {
    "messages": convert_messages_format,
    "dpo_to_sft": convert_dpo_to_sft,
}


def convert_system_user_assistant(row) -> list | None:
    """Handle datasets with separate system/user/assistant columns (e.g. Trendyol)."""
    user = row.get("user") or row.get("instruction") or row.get("question") or ""
    assistant = row.get("assistant") or row.get("response") or row.get("output") or ""
    if not user or not assistant:
        return None
    if not isinstance(user, str) or not isinstance(assistant, str):
        return None
    return [
        {"role": "user", "content": user.strip()},
        {"role": "assistant", "content": assistant.strip()},
    ]


def convert_instruction_answer(row) -> list | None:
    """Handle datasets with instruction/answer columns (e.g. Tiamz)."""
    instruction = row.get("instruction") or row.get("question") or row.get("input") or ""
    answer = row.get("answer") or row.get("output") or row.get("response") or ""
    if not instruction or not answer:
        return None
    if not isinstance(instruction, str) or not isinstance(answer, str):
        return None
    return [
        {"role": "user", "content": instruction.strip()},
        {"role": "assistant", "content": answer.strip()},
    ]


CONVERTERS = {
    "messages": convert_messages_format,
    "dpo_to_sft": convert_dpo_to_sft,
    "system_user_assistant": convert_system_user_assistant,
    "instruction_answer": convert_instruction_answer,
}

# =============================================================================
# Download logic
# =============================================================================

def download_source(name: str, config: dict, hf_token: str | None = None):
    output_path = SFT_DIR / config["output_file"]

    if output_path.exists():
        existing = sum(1 for _ in open(output_path, encoding="utf-8"))
        logger.info(f"  {name}: already exists ({existing:,} rows) — skipping. Delete to re-download.")
        return existing

    if config.get("gated") and not hf_token:
        logger.warning(f"  {name}: SKIPPED — gated dataset requires HF_TOKEN in environment")
        return 0

    hf_name = config["hf_name"]
    split = config.get("split", "train")
    streaming = config.get("streaming", True)
    fmt = config.get("format", "messages")
    max_rows = config.get("max_rows", 100_000)
    converter = CONVERTERS.get(fmt)

    if converter is None:
        logger.error(f"  {name}: unknown format '{fmt}'")
        return 0

    logger.info(f"  Downloading {hf_name} (split={split}, max={max_rows:,})...")

    load_kwargs = {"streaming": streaming, "trust_remote_code": False}
    if hf_token:
        load_kwargs["token"] = hf_token
    if config.get("hf_subset"):
        load_kwargs["name"] = config["hf_subset"]

    try:
        ds = load_dataset(hf_name, split=split, **load_kwargs)
    except Exception as e:
        logger.error(f"  {name}: failed to load dataset — {e}")
        return 0

    written = 0
    skipped = 0
    tmp_path = output_path.with_suffix(".tmp")

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            for row in ds:
                if written >= max_rows:
                    break
                try:
                    messages = converter(row)
                except Exception:
                    skipped += 1
                    continue

                if messages is None:
                    skipped += 1
                    continue

                f.write(json.dumps(messages, ensure_ascii=False) + "\n")
                written += 1

                if written % 10_000 == 0:
                    logger.info(f"    {name}: {written:,} rows written...")

        tmp_path.rename(output_path)
        logger.info(f"  ✓ {name}: {written:,} rows written to {output_path.name} (skipped {skipped:,})")
        return written

    except Exception as e:
        logger.error(f"  {name}: error during download — {e}")
        if tmp_path.exists():
            tmp_path.unlink()
        return 0


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Download external SFT datasets for mesosfer")
    parser.add_argument("--sources", nargs="*", default=None,
                        help="Specific sources to download (default: all)")
    parser.add_argument("--list", action="store_true",
                        help="List available sources and exit")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable SFT sources:")
        for name, cfg in SOURCES.items():
            gated = " [GATED - needs HF_TOKEN]" if cfg.get("gated") else ""
            print(f"  {name:<25} → {cfg['output_file']}{gated}")
        return

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token and hf_token != "your_hugging_face_token_here":
        logger.info("✓ HF token found")
    else:
        hf_token = None
        logger.warning("⚠ No HF token — gated datasets will be skipped")

    SFT_DIR.mkdir(parents=True, exist_ok=True)

    sources_to_run = args.sources if args.sources else list(SOURCES.keys())
    unknown = [s for s in sources_to_run if s not in SOURCES]
    if unknown:
        logger.error(f"Unknown sources: {unknown}. Use --list to see available sources.")
        return

    total_written = 0
    for name in sources_to_run:
        logger.info(f"\n[{name}]")
        n = download_source(name, SOURCES[name], hf_token)
        total_written += n

    logger.info(f"\n✓ Done. Total rows written: {total_written:,}")
    logger.info(f"Output directory: {SFT_DIR}")


if __name__ == "__main__":
    main()
