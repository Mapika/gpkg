"""Build environment: GPU detection, optimized compilation, wheel caching."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

from gpkg import console

def _normalize_pkg(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).lower()


_GPU_ARCH_MAP = {
    "B100": "10.0", "B200": "10.0", "B300": "10.0", "GB200": "10.0", "GB300": "10.0",
    "5090": "12.0", "5080": "12.0", "5070": "12.0", "5060": "12.0", "B580": "12.0",
    "H100": "9.0", "H200": "9.0", "GH200": "9.0",
    "A100": "8.0", "A800": "8.0",
    "4090": "8.9", "4080": "8.9", "4070": "8.9", "L40": "8.9",
    "3090": "8.6", "3080": "8.6", "3070": "8.6", "A6000": "8.6", "A5000": "8.6",
    "A30": "8.6", "A10": "8.6",
    "V100": "7.0",
    "T4": "7.5",
}


def detect_gpu_arch() -> Optional[str]:
    """Detect GPU compute capability via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        arch = result.stdout.strip().split("\n")[0].strip()
        if arch and arch[0].isdigit():
            return arch
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    # Fallback: map GPU name to arch
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            name = result.stdout.strip().split("\n")[0]
            for key, arch in _GPU_ARCH_MAP.items():
                if key in name:
                    return arch
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def detect_build_jobs() -> int:
    """Detect optimal parallel job count: cores capped by available RAM / 3GB."""
    cores = os.cpu_count() or 4
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    mem_kb = int(line.split()[1])
                    mem_limited = mem_kb // (3 * 1024 * 1024)  # 3 GB per job
                    if 0 < mem_limited < cores:
                        cores = mem_limited
                    break
    except (FileNotFoundError, ValueError):
        pass
    return max(1, cores)


def build_env_vars(gpu_arch: str, jobs: int) -> dict[str, str]:
    """Return environment variables for optimized CUDA compilation."""
    return {
        "TORCH_CUDA_ARCH_LIST": gpu_arch,
        "MAX_JOBS": str(jobs),
        "CMAKE_BUILD_PARALLEL_LEVEL": str(jobs),
        "MAKEFLAGS": f"-j{jobs}",
        "TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES": "1",
    }


def ensure_ninja() -> bool:
    """Ensure ninja build system is available. Install if missing."""
    import importlib.util
    if importlib.util.find_spec("ninja") is not None:
        return True
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "ninja", "--quiet"],
            capture_output=True, timeout=60,
        )
        return importlib.util.find_spec("ninja") is not None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _wheel_cache_dir() -> Path:
    """Return the path for locally-built wheel cache."""
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    d = base / "gpkg" / "wheels"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _wheel_cache_subdir(torch_ver: str, cuda_ver: str) -> Path:
    """Return a cache subdirectory scoped to torch+cuda combo."""
    d = _wheel_cache_dir() / f"torch{torch_ver}_cu{cuda_ver}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def find_cached_wheel(
    package: str, python_tag: str, target_platform: str,
    torch_ver: str = "", cuda_ver: str = "",
) -> Optional[Path]:
    """Find a cached locally-built wheel matching package, python, platform, torch, and cuda.

    Returns the path to the newest matching wheel, or None.
    """
    cache = _wheel_cache_subdir(torch_ver, cuda_ver) if torch_ver and cuda_ver else _wheel_cache_dir()
    pkg_normalized = _normalize_pkg(package)
    pytag_short = python_tag.split("-")[0] if "-" in python_tag else python_tag
    matches: list[Path] = []
    for whl in cache.glob(f"{pkg_normalized}-*.whl"):
        name = whl.name
        if f"-{pytag_short}-" not in name:
            continue
        if target_platform == "linux_x86_64":
            if "x86_64" not in name:
                continue
        elif target_platform == "linux_aarch64":
            if "aarch64" not in name:
                continue
        elif target_platform not in name:
            continue
        matches.append(whl)
    if not matches:
        return None
    matches.sort(key=lambda p: p.name, reverse=True)
    return matches[0]


def _find_pip_command() -> list[str]:
    """Find a working pip command that runs inside the current venv.

    The venv's python must be used so --no-build-isolation can find torch.
    If pip isn't in the venv, install it via uv first.
    """
    import shutil
    # Check if pip is already in the venv
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True, timeout=5,
        )
        if result.returncode == 0:
            return [sys.executable, "-m", "pip"]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # Try installing pip into the venv via uv
    uv_path = shutil.which("uv")
    if uv_path:
        try:
            subprocess.run(
                [uv_path, "pip", "install", "pip", "--quiet"],
                capture_output=True, timeout=60,
            )
            result = subprocess.run(
                [sys.executable, "-m", "pip", "--version"],
                capture_output=True, timeout=5,
            )
            if result.returncode == 0:
                return [sys.executable, "-m", "pip"]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return []


def build_wheel(
    package: str, build_env: dict[str, str], *,
    torch_ver: str = "", cuda_ver: str = "", timeout: int = 1800,
) -> Optional[Path]:
    """Build a wheel from source using pip wheel. Returns path to .whl or None."""
    cache = _wheel_cache_subdir(torch_ver, cuda_ver) if torch_ver and cuda_ver else _wheel_cache_dir()
    env = {**os.environ, **build_env}
    pip_cmd = _find_pip_command()
    if not pip_cmd:
        console.print("          [red]build failed:[/red] pip not found")
        return None
    try:
        subprocess.run(
            [*pip_cmd, "wheel", package,
             "--no-deps", "--no-binary", package,
             "--no-build-isolation",
             "--wheel-dir", str(cache)],
            env=env,
            stdout=subprocess.DEVNULL,
            check=True,
            timeout=timeout,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        if isinstance(e, subprocess.CalledProcessError):
            console.print("          [red]build failed[/red]")
        elif isinstance(e, subprocess.TimeoutExpired):
            console.print(f"          [red]build timed out[/red] after {timeout}s")
        return None

    # Find the built wheel
    pkg_normalized = _normalize_pkg(package)
    built = sorted(cache.glob(f"{pkg_normalized}-*.whl"), key=lambda p: p.name, reverse=True)
    return built[0] if built else None
