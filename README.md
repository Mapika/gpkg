# gpkg
[![PyPI](https://img.shields.io/pypi/v/gpkg)](https://pypi.org/project/gpkg/) [![Python](https://img.shields.io/pypi/pyversions/gpkg)](https://pypi.org/project/gpkg/) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

GPU package manager. Stop compiling CUDA extensions.

![demo](https://raw.githubusercontent.com/Mapika/gpkg/main/demo.gif)

```bash
curl -LsSf https://raw.githubusercontent.com/Mapika/gpkg/main/install.sh | sh
```

```bash
gpkg add flash-attn causal-conv1d mamba-ssm
# done. resolved, locked, installed.
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

Prebuilt wheels exist across dozens of GitHub repos and pip indexes. Finding the right wheel for your exact `python + torch + cuda + platform` combo is a scavenger hunt nobody should repeat.

## Install

```bash
pip install gpkg

# or with uv
uv tool install gpkg

# or from source
git clone https://github.com/Mapika/gpkg && cd gpkg
uv tool install .
```

## Usage

### Just add packages

```bash
# Auto-detect torch + cuda, resolve wheels, lock, install
gpkg add flash-attn causal-conv1d mamba-ssm

# If no prebuilt wheel exists, it builds from source (optimized)
gpkg add flash-attn causal-conv1d mamba-ssm  # --build-missing is automatic
```

### Resolve without installing

```bash
# Explicit versions
gpkg --torch 2.11.0 --cuda 130 flash-attn flash-attn-3 -o pyproject.toml

# Auto-detect torch and CUDA from your environment
gpkg flash-attn causal-conv1d mamba-ssm
```

### Lockfile for reproducible installs

```bash
# Write a lockfile with exact versions and URLs
gpkg --torch 2.11.0 --cuda 130 flash-attn causal-conv1d --lock

# Install from lockfile (no network needed)
gpkg install -o pyproject.toml

# Install + sync in one step
gpkg install --sync
```

### Verify your environment

```bash
gpkg test flash-attn causal-conv1d mamba-ssm
```

```
  ok  torch 2.11.0+cu128  GPU: NVIDIA GeForce RTX 5070 Ti
  ok  flash-attn 2.8.3
  ok  causal-conv1d 1.6.1
  ok  mamba-ssm 2.3.1
```

### See what's available

```bash
gpkg --available causal-conv1d natten
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

### Diagnostics

```bash
gpkg --explain --torch 2.11 --cuda 130 flash-attn   # why a wheel was/wasn't selected
gpkg --doctor --torch 2.11 --cuda 130 flash-attn     # verify URLs are accessible
```

### Cache management

```bash
gpkg --cache-info                    # show cache statistics
gpkg --cache-clean                   # clean all cached data
gpkg --cache-clean --older-than 1h   # clean entries older than 1 hour
```

## When there's no wheel: compile fast

When `gpkg add` can't find a prebuilt wheel, it automatically builds from source with optimized settings:

- Detects your GPU arch via `nvidia-smi` → builds for only that arch
- Uses all CPU cores (capped by RAM to prevent OOM)
- Enables ninja for parallel builds
- Caches the built wheel so you never compile the same package twice

You can also use `build-env.sh` manually:

```bash
source build-env.sh
uv add causal-conv1d    # 5-10x faster than default
```

## How it works

1. Checks the hosted registry at `wheels.mapika.dev` for cached wheels (fast)
2. Falls back to GitHub releases API and pip find-links indexes
3. Matches wheels against your torch + cuda + python + platform
4. Picks the best match per package (latest version, prefers non-manylinux)
5. If no wheel exists and `--build-missing` is set, compiles from source
6. Emits a valid `pyproject.toml` with `[tool.uv.sources]` pointing at direct URLs

## Registry

The registry tracks **9 sources** across **7 packages**:

| Package | Sources |
|---|---|
| flash-attn | mjun0812/flash-attention-prebuild-wheels, Dao-AILab/flash-attention |
| flash-attn-3 | mjun0812/flash-attention-prebuild-wheels |
| causal-conv1d | Dao-AILab/causal-conv1d |
| mamba-ssm | state-spaces/mamba |
| natten | SHI-Labs/NATTEN |
| grouped-gemm | fanshiqing/grouped_gemm |
| sageattention | woct0rdho/SageAttention, mobcat40/sageattention-blackwell |

### Adding a source

Edit `src/gpkg/registry.toml` and open a PR:

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

## Configuration

| Env var | Purpose |
|---|---|
| `GITHUB_TOKEN` | Raise API rate limit 60 to 5000 req/hr |
| `UVFORGE_TOKEN` | Bearer token for private registries |
| `UVFORGE_TOKEN_<HOST>` | Host-specific token (e.g. `UVFORGE_TOKEN_WHEELS_MYCO_COM`) |
| `UVFORGE_REGISTRY` | Override default registry path/URL |

Private registries also support `~/.netrc` for credential storage.

## CI Usage

```yaml
- name: Install GPU packages
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  run: |
    pip install gpkg
    gpkg add flash-attn causal-conv1d mamba-ssm
```

For machine-readable output:

```bash
gpkg --list --json                           # all registered sources
gpkg --available flash-attn --json           # cuda/torch combos
gpkg --torch 2.10 --cuda 128 flash-attn --json  # resolved wheel URLs
```

## Development

```bash
git clone https://github.com/Mapika/gpkg && cd gpkg
uv sync
uv run pytest -v
uv run ruff check src/
```

## License

MIT
