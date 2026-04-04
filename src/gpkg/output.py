"""Output formatting: TOML generation and display helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote

import tomlkit

if TYPE_CHECKING:
    from gpkg.matching import WheelMatch


def _fmt_size(b: int) -> str:
    if b < 1024:
        return f"{b} B"
    if b < 1024 * 1024:
        return f"{b / 1024:.1f} KB"
    return f"{b / 1024 / 1024:.1f} MB"


def _fmt_age(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s ago"
    if s < 3600:
        return f"{s / 60:.0f}m ago"
    return f"{s / 3600:.1f}h ago"


def _cuda_index_tag(cuda_version: str) -> str:
    cuda_clean = cuda_version.replace(".", "")
    return f"cu{cuda_clean}" if len(cuda_clean) >= 3 else f"cu{cuda_clean}0"


def generate_toml(
    project_name: str,
    python_version: str,
    torch_version: str,
    cuda_version: str,
    wheel_matches: dict[str, WheelMatch],
) -> str:
    cuda_idx = _cuda_index_tag(cuda_version)

    deps = [f'"torch=={torch_version}"']
    for name in wheel_matches:
        deps.append(f'"{name}"')

    src = [f'torch = {{ index = "pytorch-{cuda_idx}" }}']
    for name, m in wheel_matches.items():
        src.append(f'{name} = {{ url = "{unquote(m.url)}" }}')

    return "\n".join([
        "[project]",
        f'name = "{project_name}"',
        'version = "0.1.0"',
        f'requires-python = ">={python_version}"',
        "dependencies = [",
        "    " + ",\n    ".join(deps) + ",",
        "]",
        "",
        "# -- uv configuration -------------------------------------------------",
        "",
        "[[tool.uv.index]]",
        f'name = "pytorch-{cuda_idx}"',
        f'url = "https://download.pytorch.org/whl/{cuda_idx}"',
        "explicit = true",
        "",
        "[tool.uv.sources]",
        "\n".join(src),
        "",
    ])


def merge_into_pyproject(
    path: Path,
    torch_version: str,
    cuda_version: str,
    wheel_matches: dict,
) -> str:
    """Merge resolved wheels into an existing pyproject.toml. Returns the updated TOML string.

    Adds/updates: dependencies, [[tool.uv.index]] for pytorch, [tool.uv.sources] for wheels.
    Preserves everything else (comments, formatting, other config).
    """
    if path.exists():
        doc = tomlkit.parse(path.read_text())
    else:
        doc = tomlkit.document()

    cuda_idx = _cuda_index_tag(cuda_version)

    # Ensure [project] exists
    if "project" not in doc:
        doc["project"] = tomlkit.table()

    # Ensure dependencies list exists
    project = doc["project"]
    if "dependencies" not in project:
        project["dependencies"] = tomlkit.array()

    # Add torch and packages to dependencies if not present
    deps = project["dependencies"]
    existing_deps = {d.split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip().strip('"') for d in deps}
    if "torch" not in existing_deps:
        deps.append(f"torch=={torch_version}")
    for name in wheel_matches:
        if name not in existing_deps:
            deps.append(name)

    # Ensure [tool] and [tool.uv] exist
    if "tool" not in doc:
        doc["tool"] = tomlkit.table(is_super_table=True)
    if "uv" not in doc["tool"]:
        doc["tool"]["uv"] = tomlkit.table(is_super_table=True)
    uv = doc["tool"]["uv"]

    # Ensure [[tool.uv.index]] has pytorch index
    if "index" not in uv:
        uv["index"] = tomlkit.aot()
    index_array = uv["index"]
    pytorch_idx_name = f"pytorch-{cuda_idx}"
    has_pytorch_idx = any(
        idx.get("name") == pytorch_idx_name for idx in index_array
    )
    if not has_pytorch_idx:
        idx_entry = tomlkit.table()
        idx_entry["name"] = pytorch_idx_name
        idx_entry["url"] = f"https://download.pytorch.org/whl/{cuda_idx}"
        idx_entry["explicit"] = True
        index_array.append(idx_entry)

    # Ensure [tool.uv.sources] exists and has wheel URLs
    if "sources" not in uv:
        uv["sources"] = tomlkit.table()
    sources = uv["sources"]
    sources["torch"] = tomlkit.inline_table()
    sources["torch"]["index"] = pytorch_idx_name
    for name, m in wheel_matches.items():
        url = unquote(m.url) if hasattr(m, "url") else m["url"]
        sources[name] = tomlkit.inline_table()
        sources[name]["url"] = url

    return tomlkit.dumps(doc)
