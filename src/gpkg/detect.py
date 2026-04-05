"""Environment detection: platform, Python, CUDA, PyTorch."""

from __future__ import annotations

import platform as platform_mod
import re
import sys
from typing import Optional


def detect_platform() -> str:
    if sys.platform.startswith("linux"):
        return "linux_aarch64" if platform_mod.machine() == "aarch64" else "linux_x86_64"
    if sys.platform == "win32":
        return "win_amd64"
    if sys.platform == "darwin":
        return "macosx_arm64" if platform_mod.machine() == "arm64" else "macosx_x86_64"
    return "linux_x86_64"


def detect_python() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def detect_cuda() -> Optional[str]:
    """Try to detect CUDA version from nvcc or torch."""
    import shutil
    import subprocess

    # Try nvcc
    if shutil.which("nvcc"):
        try:
            out = subprocess.check_output(["nvcc", "--version"], text=True)
            m = re.search(r"release (\d+)\.(\d+)", out)
            if m:
                return f"{m.group(1)}{m.group(2)}"
        except Exception:
            pass
    # Try torch
    try:
        import torch

        if torch.version.cuda:
            parts = torch.version.cuda.split(".")
            return f"{parts[0]}{parts[1]}"
    except Exception:
        pass
    return None


def detect_torch() -> Optional[str]:
    """Try to detect PyTorch version from installed package.

    Returns full version string (e.g. '2.11.0').
    """
    try:
        import torch

        v = torch.__version__
        # Strip local version specifiers like '+cu130'
        return v.split("+")[0]
    except Exception:
        pass
    return None
