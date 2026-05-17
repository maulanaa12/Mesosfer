from scripts.setup import torch_backend


def test_backend_selector_prefers_explicit_override():
    backend = torch_backend.detect_backend(
        env={"OZON_TORCH_BACKEND": "rocm"},
        command_exists=lambda name: name == "nvidia-smi",
        path_exists=lambda path: False,
    )

    assert backend == "rocm"


def test_backend_selector_detects_nvidia_before_rocm():
    backend = torch_backend.detect_backend(
        env={},
        command_exists=lambda name: name in {"nvidia-smi", "rocminfo"},
        path_exists=lambda path: False,
    )

    assert backend == "cuda"


def test_backend_selector_detects_rocm_without_nvidia():
    backend = torch_backend.detect_backend(
        env={},
        command_exists=lambda name: name == "rocminfo",
        path_exists=lambda path: False,
    )

    assert backend == "rocm"


def test_backend_selector_falls_back_to_cpu():
    backend = torch_backend.detect_backend(
        env={},
        command_exists=lambda name: False,
        path_exists=lambda path: False,
    )

    assert backend == "cpu"


def test_install_command_uses_selected_extra():
    assert torch_backend.install_command("cuda") == "uv sync --extra cuda"
    assert torch_backend.install_command("rocm") == "uv sync --extra rocm"
    assert torch_backend.install_command("cpu") == "uv sync --extra cpu"
