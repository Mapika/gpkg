# CLAUDE.md — development guide

## Quick start

```bash
uv sync
uv run pytest -v
uv run ruff check src/
```

## Project structure

- `src/uvforge/cli.py` — single-file CLI: registry loading, GitHub API, wheel matching, TOML generation
- `src/uvforge/registry.toml` — community-maintained list of CUDA wheel sources
- `tests/test_matching.py` — unit tests for matching logic
- `build-env.sh` — helper script to optimize CUDA compilation speed
- `install.sh` — one-liner installer

## Key concepts

- **Registry**: TOML file mapping packages to GitHub repos that host prebuilt wheels
- **Wheel matching**: regex-based filename parsing against user's torch + cuda + python + platform
- **Source types**: "github" (release assets) or "find-links" (pip index URL)
- **cuda_style**: "full" (cu130 = CUDA 13.0) vs "short" (cu13 = CUDA 13.x major only)
- **torch_format**: "minor" (2.11), "packed" (2110), "full" (2.11.0)

## Testing

```bash
uv run pytest -v              # run all tests
uv run pytest -k cuda         # run only CUDA matching tests
uv run ruff check src/        # lint
```

## Adding a new wheel source

1. Add a `[[sources]]` block to `src/uvforge/registry.toml`
2. Use `{version}`, `{cuda}`, `{torch}`, `{pytag}`, `{platform}` placeholders in `wheel_name`
3. Add a test in `tests/test_matching.py` for the new filename pattern
