#!/usr/bin/env python3
"""Select the PyTorch wheel extra for the current machine.

This intentionally does not import torch. It is meant to run before installing
the project dependencies.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


VALID_BACKENDS = {"cpu", "cuda", "rocm"}


def _default_command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def _default_path_exists(path: str) -> bool:
    return Path(path).exists()


def detect_backend(
    env: dict[str, str] | None = None,
    command_exists=_default_command_exists,
    path_exists=_default_path_exists,
) -> str:
    """Return `cuda`, `rocm`, or `cpu` based on local accelerator signals."""
    env = os.environ if env is None else env
    override = env.get("mesosfer_TORCH_BACKEND", "").strip().lower()
    if override:
        if override not in VALID_BACKENDS:
            valid = ", ".join(sorted(VALID_BACKENDS))
            raise ValueError(f"mesosfer_TORCH_BACKEND must be one of: {valid}")
        return override

    if command_exists("nvidia-smi"):
        return "cuda"

    if (
        command_exists("rocminfo")
        or command_exists("rocm-smi")
        or path_exists("/opt/rocm")
    ):
        return "rocm"

    return "cpu"


def install_command(backend: str) -> str:
    if backend not in VALID_BACKENDS:
        valid = ", ".join(sorted(VALID_BACKENDS))
        raise ValueError(f"backend must be one of: {valid}")
    return f"uv sync --extra {backend}"


def main() -> None:
    backend = detect_backend()
    print(f"Detected torch backend: {backend}")
    print(install_command(backend))


if __name__ == "__main__":
    main()
