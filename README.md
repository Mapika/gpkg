# uvforge
[![CI](https://github.com/Mapika/uvforge/actions/workflows/ci.yml/badge.svg)](https://github.com/Mapika/uvforge/actions/workflows/ci.yml) [![PyPI](https://img.shields.io/pypi/v/uvforge)](https://pypi.org/project/uvforge/) [![Python](https://img.shields.io/pypi/pyversions/uvforge)](https://pypi.org/project/uvforge/) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Stop compiling CUDA extensions. Find prebuilt wheels in seconds.

```bash
curl -LsSf https://raw.githubusercontent.com/Mapika/uvforge/main/install.sh | sh
```

```bash
uvforge --torch 2.11 --cuda 130 flash-attn flash-attn-3 causal-conv1d mamba-ssm -o pyproject.toml
uv sync  # done
```

## The problem

These packages take 20-60 minutes to compile and regularly fail:

| Package | Typical compile time | Common failure |
|---|---|---|
| flash-attn | 25 min | OOM during build, CUDA mismatch |
| flash-attn-3 | 20 min | SM90+ only, rare wheel coverage |
| causal-conv1d | 10 min | Torch ABI mismatch |
| mamba-ssm | 15 min | Cascading causal-conv1d failure |
| natten | 30 min | CUTLASS dependency, arch-specific |
| sageattention | 10 min | Windows build nightmare |
| grouped-gemm | 10 min | MoE stack dependency |

Prebuilt wheels exist across dozens of GitHub repos. Finding the right wheel for your exact `python + torch + cuda + platform` combo is a scavenger hunt nobody should repeat.

## Install

```bash
# one-liner (uses uv, falls back to pipx, then pip)
curl -LsSf https://raw.githubusercontent.com/Mapika/uvforge/main/install.sh | sh

# or directly with uv
uv tool install uvforge

# or from source
git clone https://github.com/Mapika/uvforge && cd uvforge
uv tool install .
```

## Usage

### Find wheels and generate pyproject.toml

```bash
# Explicit versions
uvforge --torch 2.11 --cuda 130 flash-attn flash-attn-3 -o pyproject.toml

# Auto-detect torch and CUDA from your environment
uvforge flash-attn causal-conv1d mamba-ssm

# The full Mamba + attention stack
uvforge --torch 2.10 --cuda 130 flash-attn causal-conv1d mamba-ssm natten grouped-gemm
```

### See what's available

```bash
# What torch/cuda combos exist for a package?
uvforge --available causal-conv1d natten
```

```
causal-conv1d -- available wheels (linux_x86_64)
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ CUDA ┃ PyTorch versions        ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 12   │ 2.10, 2.9, 2.8, 2.7    │
│ 13   │ 2.10, 2.9              │
└──────┴────────────────────────┘
```

### List all tracked sources

```bash
uvforge --list
```

### Use a custom registry

```bash
# Your team's private wheels
uvforge --extra-registry ./our-wheels.toml --torch 2.11 --cuda 130 flash-attn
```

## When there's no wheel: compile fast

When a prebuilt wheel doesn't exist, compilation defaults are terrible: 2 parallel jobs, 8 GPU architectures you don't own. The included `build-env.sh` fixes this:

```bash
source build-env.sh
uv add causal-conv1d    # now 5-10x faster
```

What it does:
- Detects your GPU via `nvidia-smi`, sets `TORCH_CUDA_ARCH_LIST` to only that arch
- Sets `MAX_JOBS` to all cores (capped by available RAM to prevent OOM)
- Ensures `ninja` is installed for parallel builds
- Sets `CMAKE_BUILD_PARALLEL_LEVEL` and `MAKEFLAGS`

```
uvforge build environment
  TORCH_CUDA_ARCH_LIST      = 9.0       # only YOUR GPU, not 8 architectures
  MAX_JOBS                   = 64        # all cores, not 2
  CMAKE_BUILD_PARALLEL_LEVEL = 64
  ninja: installed
```

## How it works

1. Loads `registry.toml`, a community-maintained list of wheel sources
2. Queries GitHub releases API across all sources (cached per-repo)
3. Matches wheels against your torch + cuda + python + platform
4. Picks the best match per package (latest version, prefers non-manylinux)
5. Emits a valid `pyproject.toml` with `[tool.uv.sources]` pointing at direct URLs

The registry tries to fetch from GitHub first (for latest updates), falls back to the bundled copy if offline.

## Registry

The registry tracks **10 sources** across **7 packages**:

| Package | Sources |
|---|---|
| flash-attn | mjun0812/flash-attention-prebuild-wheels, Dao-AILab/flash-attention |
| flash-attn-3 | mjun0812/flash-attention-prebuild-wheels, windreamer/flash-attention3-wheels |
| causal-conv1d | Dao-AILab/causal-conv1d |
| mamba-ssm | state-spaces/mamba |
| natten | SHI-Labs/NATTEN |
| grouped-gemm | fanshiqing/grouped_gemm |
| sageattention | woct0rdho/SageAttention, mobcat40/sageattention-blackwell |

### Adding a source

Edit `src/uvforge/registry.toml` and open a PR:

```toml
[[sources]]
package      = "causal-conv1d"
description  = "causal-conv1d -- your torch 2.11 builds"
type         = "github"
repo         = "yourname/causal-conv1d-wheels"
wheel_name   = "causal_conv1d-{version}+cu{cuda}torch{torch}-{pytag}-{platform}.whl"
cuda_style   = "full"
scan_tags    = 5
```

Template placeholders: `{version}`, `{cuda}`, `{torch}`, `{pytag}`, `{platform}`, `{abi}`, `{hash}`.

### Packages that don't need uvforge

| Package | Just run |
|---|---|
| xformers | `pip install xformers` (stable ABI since v0.0.34) |
| bitsandbytes | `pip install bitsandbytes` (bundles all CUDA) |
| deepspeed | `pip install deepspeed` (JIT at runtime) |
| flash-linear-attention | `pip install flash-linear-attention` (pure Triton) |
| transformer-engine | `pip install transformer-engine --index-url https://developer.download.nvidia.com/...` |
| torch-scatter/sparse | `pip install torch-scatter -f https://data.pyg.org/whl/torch-{ver}+{cuda}.html` |

## Configuration

| Env var | Purpose |
|---|---|
| `GITHUB_TOKEN` | Raise API rate limit 60 to 5000 req/hr |
| `UVFORGE_REGISTRY` | Override default registry path/URL |

## Development

```bash
git clone https://github.com/Mapika/uvforge && cd uvforge
uv sync
uv run pytest -v
uv run ruff check src/
```

## License

MIT
