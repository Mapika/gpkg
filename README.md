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

Tab completion for commands and flags:

```bash
pip install 'gpkg[completion]'
activate-global-python-argcomplete   # once, then restart your shell
```

## Usage

### Just add packages

```bash
# Auto-detect torch + cuda, resolve wheels, lock, install
gpkg add flash-attn causal-conv1d mamba-ssm

# If no prebuilt wheel exists, it builds from source (optimized)
gpkg add flash-attn causal-conv1d mamba-ssm  # --build-missing is automatic
```

When you add multiple packages, gpkg doesn't just grab the latest wheel of each — it
checks that the chosen versions are mutually compatible (dependency metadata +
a trial resolve with uv) and picks a set that actually installs together.

If a package pins torch (vllm pins `torch==2.6.0`), gpkg detects the pin from PyPI
and adjusts the target torch version before resolving.

### Check compatibility without installing

```bash
gpkg compat flash-attn causal-conv1d mamba-ssm
```

```
  ✓ Compatible set found

  causal-conv1d        1.6.2.post1  Dao-AILab official
  flash-attn           2.8.3        gpkg hosted registry
  mamba-ssm            2.3.2.post1  state-spaces official
```

On conflict it shows what clashes, offers alternative sets, and analyzes whether
the blocking constraint is actually load-bearing.

### Find out which pins actually matter

Some packages pin dependencies far tighter than they need to. `gpkg analyze` crawls
the full transitive dependency tree, intersects every constraint, and tells you
which pins are load-bearing and which can be safely relaxed:

```bash
gpkg analyze boltz            # any PyPI package
gpkg analyze boltz==2.2.1     # or a specific version
gpkg analyze                  # your pyproject.toml dependencies
```

```
boltz 2.2.1 — constraint analysis

  ✓ boltz's numpy<2.0,>=1.26 is relaxable → safe range: numpy>=2.0.0,<2.2
    numba requires numpy<2.2,>=1.24
    aeon requires numpy<2.5.0,>=2.0.0
    scipy requires numpy<2.3,>=1.22.4
    ...

  ✓ boltz's scipy==1.13.1 is relaxable → safe range: scipy>=1.9.0,<1.18.0
    scikit_learn requires scipy>=1.6.0
    aeon requires scipy<1.18.0,>=1.9.0
```

Pins that a transitive dependency genuinely requires are marked `✗ load-bearing`.
Results are cached, so repeat runs take under a second.

### Curated stacks

Known-good torch + cuda + package combos, tested on real GPUs:

```bash
gpkg stack list             # show available stacks
gpkg stack info mamba       # details + exact versions
gpkg stack install mamba    # install the verified combo
```

### Resolve without installing

```bash
# Explicit versions
gpkg resolve --torch 2.11.0 --cuda 130 flash-attn flash-attn-3 -o pyproject.toml

# Auto-detect torch and CUDA from your environment ('resolve' is implied)
gpkg flash-attn causal-conv1d mamba-ssm
```

### Lockfile for reproducible installs

```bash
# Write a lockfile with exact versions and URLs
gpkg resolve --torch 2.11.0 --cuda 130 flash-attn causal-conv1d --lock

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
gpkg available causal-conv1d natten
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
gpkg resolve --explain --torch 2.11 --cuda 130 flash-attn   # why a wheel was/wasn't selected
gpkg doctor --torch 2.11 --cuda 130 flash-attn              # verify URLs are accessible
gpkg doctor --verify --torch 2.11 --cuda 130 flash-attn     # download + check sha256
gpkg doctor                                                 # health-check every registry source
```

### Hashes and provenance

Every resolved wheel records its source-attested sha256 (GitHub asset digest or
index hash fragment), the source it came from, and the release tag — all written
into `gpkg.lock.toml`. Locally built wheels are hashed at build time.

For hash-enforced installs (CI, air-gapped, supply-chain-sensitive):

```bash
gpkg export -o requirements.txt        # emits --hash=sha256: pins from the lockfile
pip install --no-deps --require-hashes -r requirements.txt
```

Both pip and uv reject any artifact whose hash doesn't match. `gpkg doctor
--verify` does the same check without installing. A nightly CI job re-validates
that every registry source still serves wheels matching its pattern.

### Windows

flash-attn (kingbri1 builds), sageattention, nunchaku, exllamav2, kaolin, and
llama-cpp-python all publish `win_amd64` wheels — resolve them the same way:

```bash
gpkg resolve --torch 2.9.0 --cuda 128 --platform win_amd64 flash-attn sageattention
```

### Cache management

```bash
gpkg cache info                     # show cache statistics
gpkg cache clean                    # clean all cached data
gpkg cache clean --older-than 1h    # clean entries older than 1 hour
```

Every command has its own `--help`. The pre-0.5 flag forms (`gpkg --list`,
`gpkg --available ...`, `gpkg --cache-info`) still work.

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
5. For multi-package installs, verifies the set is mutually compatible
   (dependency metadata + trial resolve with uv) before committing
6. If no wheel exists and `--build-missing` is set, compiles from source
7. Emits a valid `pyproject.toml` with `[tool.uv.sources]` pointing at direct URLs

## Registry

The registry tracks **16 packages** across GitHub releases, find-links indexes,
and the hosted registry:

| Package | Sources |
|---|---|
| flash-attn | mjun0812/flash-attention-prebuild-wheels, Dao-AILab/flash-attention, kingbri1/flash-attention (Windows) |
| flash-attn-3 | mjun0812/flash-attention-prebuild-wheels, windreamer/flash-attention3-wheels |
| causal-conv1d | Dao-AILab/causal-conv1d |
| mamba-ssm | state-spaces/mamba |
| natten | SHI-Labs/NATTEN |
| grouped-gemm | fanshiqing/grouped_gemm |
| sageattention | woct0rdho/SageAttention |
| exllamav2 | turboderp-org/exllamav2 |
| nunchaku | nunchaku-ai/nunchaku |
| kaolin | NVIDIA official (S3 index) |
| llama-cpp-python | abetlen official CUDA builds |
| torch-scatter, torch-sparse, torch-cluster, torch-spline-conv, pyg-lib | data.pyg.org (PyG official) |

Every source is health-checked nightly in CI — a source that stops serving
pattern-matching wheels fails the build.

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
| `GPKG_TOKEN` | Bearer token for private registries |
| `GPKG_TOKEN_<HOST>` | Host-specific token (e.g. `GPKG_TOKEN_WHEELS_MYCO_COM`) |
| `GPKG_REGISTRY` | Override default registry path/URL |

The legacy `UVFORGE_*` names are still accepted. Private registries also support
`~/.netrc` for credential storage.

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
gpkg list --json                                       # all registered sources
gpkg available flash-attn --json                       # cuda/torch combos
gpkg resolve --torch 2.10 --cuda 128 flash-attn --json  # resolved wheel URLs
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
