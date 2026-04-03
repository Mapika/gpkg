#!/bin/bash
# uvforge-build — compile CUDA extensions at full speed
#
# Usage:
#   source <(curl -LsSf https://raw.githubusercontent.com/you/uvforge/main/build-env.sh)
#   uv add causal-conv1d       # now compiles 5-10x faster
#
#   # or wrap a single command:
#   ./build-env.sh uv pip install flash-attn --no-binary flash-attn
#
# What it does:
#   1. Detects your GPU compute capability from nvidia-smi
#   2. Sets TORCH_CUDA_ARCH_LIST to ONLY your GPU (skips all others)
#   3. Sets MAX_JOBS to all available cores
#   4. Ensures ninja is installed (parallel build backend)
#   5. Sets CMAKE_BUILD_PARALLEL_LEVEL for CMake-based packages
#
# The difference:
#   Without:  causal-conv1d compiles in ~10 min (2 jobs, 8 GPU archs)
#   With:     causal-conv1d compiles in ~60 sec (all cores, 1 GPU arch)

set -euo pipefail

# ── Colors ───────────────────────────────────────────────────────
red()   { printf '\033[1;31m%s\033[0m\n' "$*"; }
green() { printf '\033[1;32m%s\033[0m\n' "$*"; }
dim()   { printf '\033[2m%s\033[0m\n' "$*"; }

# ── Detect GPU architecture ──────────────────────────────────────
detect_gpu_arch() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        red "nvidia-smi not found"
        return 1
    fi

    # Get compute capability from nvidia-smi
    # Format: major.minor (e.g., 9.0 for H100, 8.9 for RTX 4090)
    local arch
    arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '[:space:]')

    if [ -z "$arch" ]; then
        # Fallback: parse from GPU name
        local gpu_name
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        case "$gpu_name" in
            *B100*|*B200*|*B300*|*GB200*|*GB300*)  arch="10.0" ;;
            *5090*|*5080*|*5070*|*5060*|*B580*)     arch="12.0" ;;
            *H100*|*H200*|*GH200*)                  arch="9.0"  ;;
            *A100*|*A800*)                           arch="8.0"  ;;
            *4090*|*4080*|*4070*|*L40*)              arch="8.9"  ;;
            *3090*|*3080*|*3070*|*A6000*|*A5000*)    arch="8.6"  ;;
            *A30*|*A10*)                             arch="8.6"  ;;
            *V100*)                                  arch="7.0"  ;;
            *T4*)                                    arch="7.5"  ;;
            *)
                red "Could not detect GPU arch for: $gpu_name"
                return 1
                ;;
        esac
    fi

    echo "$arch"
}

# ── Detect number of usable cores ────────────────────────────────
detect_jobs() {
    local cores
    if [ -f /proc/cpuinfo ]; then
        cores=$(nproc 2>/dev/null || grep -c ^processor /proc/cpuinfo)
    elif command -v sysctl >/dev/null 2>&1; then
        cores=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
    else
        cores=4
    fi

    # For CUDA compilation, each job can use ~2-4 GB RAM.
    # Cap at available memory / 3GB to avoid OOM during nvcc.
    if [ -f /proc/meminfo ]; then
        local mem_gb
        mem_gb=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null || echo 999)
        local mem_limited=$((mem_gb / 3))
        if [ "$mem_limited" -lt "$cores" ] && [ "$mem_limited" -gt 0 ]; then
            cores=$mem_limited
        fi
    fi

    echo "$cores"
}

# ── Ensure ninja is installed ────────────────────────────────────
ensure_ninja() {
    if python3 -c "import ninja" 2>/dev/null; then
        return 0
    fi

    dim "Installing ninja build system..."
    if command -v uv >/dev/null 2>&1; then
        uv pip install ninja --quiet 2>/dev/null || pip install ninja --quiet
    else
        pip install ninja --quiet 2>/dev/null || pip install ninja --user --quiet
    fi
}

# ── Main ─────────────────────────────────────────────────────────

GPU_ARCH=$(detect_gpu_arch)
NUM_JOBS=$(detect_jobs)

export TORCH_CUDA_ARCH_LIST="$GPU_ARCH"
export MAX_JOBS="$NUM_JOBS"
export CMAKE_BUILD_PARALLEL_LEVEL="$NUM_JOBS"
export MAKEFLAGS="-j${NUM_JOBS}"

# Ninja env — torch cpp_extension checks this
export TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES=1   # faster incremental builds

green "uvforge build environment"
echo "  TORCH_CUDA_ARCH_LIST     = $GPU_ARCH"
echo "  MAX_JOBS                 = $NUM_JOBS"
echo "  CMAKE_BUILD_PARALLEL_LEVEL = $NUM_JOBS"

ensure_ninja && dim "  ninja: installed" || dim "  ninja: not available (will use make)"

echo ""

# If called with arguments, run them in this environment
if [ $# -gt 0 ]; then
    green "Running: $*"
    exec "$@"
fi

# If sourced, the exports persist in the caller's shell
dim "Environment set. Run your build command now."
dim "Example: uv add causal-conv1d"
