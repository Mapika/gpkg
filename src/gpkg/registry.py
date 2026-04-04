"""Registry: wheel source definitions and loading."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

HOSTED_REGISTRY = "https://wheels.mapika.dev/registry.toml"
REMOTE_REGISTRY = (
    "https://raw.githubusercontent.com/Mapika/gpkg/main/src/gpkg/registry.toml"
)
LOCAL_REGISTRY = str(Path(__file__).parent / "registry.toml")

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class Source:
    """One place that publishes wheels for a package."""

    package: str
    description: str
    source_type: str  # "github" or "find-links"
    repo: str = ""
    url_template: str = ""
    wheel_name: str = ""
    cuda_style: str = "full"  # "full" (cu130) or "short" (cu13)
    has_abi: bool = False
    torch_format: str = "minor"  # "minor" | "packed" | "full"
    torch_compat: str = ""  # e.g. ">=2.4,<2.6" — empty means any
    scan_tags: int = 8

    _regex: Optional[re.Pattern] = field(default=None, repr=False, compare=False)

    @property
    def regex(self) -> re.Pattern:
        if self._regex is None:
            self._regex = _template_to_regex(self.wheel_name)
        return self._regex


def _template_to_regex(template: str) -> re.Pattern:
    """Convert a wheel_name template to a compiled regex.

    Placeholders: {version}, {cuda}, {torch}, {pytag}, {platform}, {abi}, {hash}, {_}
    {_} is a wildcard that matches any characters (non-greedy).
    """
    parts = re.split(r"(\{[a-z_]+\})", template)
    out: list[str] = []
    for p in parts:
        if p == "{version}":
            out.append(r"(?P<version>\d+\.\d+\.\d+(?:\.(?:post|dev|rc)\d+)?)")
        elif p == "{cuda}":
            out.append(r"(?P<cuda>\d+)")
        elif p == "{torch}":
            out.append(r"(?P<torch>[\d.]+[a-z]*(?:\.\w+)*)")  # 2.11, 2110, 2.9.0andhigher.post4
        elif p == "{pytag}":
            out.append(r"(?P<pytag>cp\d+-(?:cp\d+[a-z]*|abi3))")
        elif p == "{platform}":
            out.append(r"(?P<platform>[a-zA-Z0-9_.]+(?:\.[a-zA-Z0-9_.]+)*)")
        elif p == "{abi}":
            out.append(r"(?P<abi>TRUE|FALSE)")
        elif p == "{hash}":
            out.append(r"(?:git[0-9a-f]+)?")
        elif p == "{_}":
            out.append(r".*?")
        else:
            out.append(re.escape(p))
    return re.compile("".join(out))


def load_registry(path_or_url: str, client: httpx.Client) -> list[Source]:
    if path_or_url.startswith(("http://", "https://")):
        resp = client.get(path_or_url, timeout=5)
        resp.raise_for_status()
        data = tomllib.loads(resp.text)
    else:
        with open(path_or_url, "rb") as f:
            data = tomllib.load(f)
    return [
        Source(
            package=e["package"],
            description=e.get("description", ""),
            source_type=e.get("type", "github"),
            repo=e.get("repo", ""),
            url_template=e.get("url_template", ""),
            wheel_name=e.get("wheel_name", ""),
            cuda_style=e.get("cuda_style", "full"),
            has_abi=e.get("has_abi", False),
            torch_format=e.get("torch_format", "minor"),
            torch_compat=e.get("torch_compat", ""),
            scan_tags=e.get("scan_tags", 8),
        )
        for e in data.get("sources", [])
    ]


def load_registry_with_fallback(client: httpx.Client) -> list[Source]:
    """Load registries: hosted (priority) + GitHub/bundled (complete coverage).

    The hosted registry at wheels.mapika.dev has stable, fast URLs but limited
    coverage. The GitHub/bundled registry has all known sources. Both are merged
    so hosted sources are tried first per package.
    """
    sources: list[Source] = []
    # Hosted registry — fast, stable, limited coverage
    try:
        hosted = load_registry(HOSTED_REGISTRY, client)
        sources.extend(hosted)
    except Exception:
        pass
    # GitHub registry (or bundled fallback) — complete coverage
    try:
        github = load_registry(REMOTE_REGISTRY, client)
        if github:
            sources.extend(github)
            return sources
    except Exception:
        pass
    sources.extend(load_registry(LOCAL_REGISTRY, client))
    return sources
