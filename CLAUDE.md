# CLAUDE.md — development guide

## Quick start

```bash
uv sync
uv run pytest -v
uv run ruff check src/
```

## Project structure

```
src/gpkg/
  cli.py           — main() + argparse
  registry.py      — Source dataclass, registry loading
  matching.py      — WheelMatch, matchers, search pipeline
  cache.py         — disk cache, find-links fetch, auth
  build.py         — GPU detection, ninja, wheel cache, build_wheel
  detect.py        — platform, python, cuda, torch detection
  doctor.py        — URL verification
  lockfile.py      — lockfile read/write/compare
  output.py        — TOML generation, formatters
  registry.toml    — community-maintained wheel sources
```

## Key concepts

- **Registry**: TOML file mapping packages to GitHub repos or find-links indexes
- **Wheel matching**: regex-based filename parsing against user's torch + cuda + python + platform
- **Source types**: "github" (release assets) or "find-links" (pip index URL)
- **cuda_style**: "full" (cu130 = CUDA 13.0) vs "short" (cu13 = CUDA 13.x major only)
- **torch_format**: "minor" (2.11), "packed" (2110), "full" (2.11.0)
- **Hosted registry**: wheels.mapika.dev — Cloudflare Worker + R2

## Testing

```bash
uv run pytest -v              # run all tests
uv run pytest -k cuda         # run only CUDA matching tests
uv run ruff check src/        # lint
```

## Adding a new wheel source

1. Add a `[[sources]]` block to `src/gpkg/registry.toml`
2. Use `{version}`, `{cuda}`, `{torch}`, `{pytag}`, `{platform}` placeholders in `wheel_name`
3. Add a test in `tests/test_matching.py` for the new filename pattern
