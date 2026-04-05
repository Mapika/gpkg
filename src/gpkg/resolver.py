"""Compatibility resolver: data classes and conflict detection."""

from __future__ import annotations

import hashlib
import itertools
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from packaging.specifiers import SpecifierSet
from packaging.version import Version

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from gpkg.matching import WheelMatch
from gpkg.registry import Source, get_requires_for_version


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PackageMetadata:
    """PyPI-side metadata for one package version."""

    package: str
    version: str
    requires_dist: list[str]  # raw PEP 508 strings from wheel metadata


@dataclass
class Conflict:
    """A detected dependency conflict across a set of packages."""

    dependency: str                  # normalized dep name, e.g. "numpy"
    specifiers: dict[str, str]       # {package_name: specifier_string}
    merged_range: str                # human-readable merged constraint or "empty"


@dataclass
class Combo:
    """A candidate set of wheels together with its conflict analysis."""

    matches: list[WheelMatch]
    conflicts: list[Conflict]
    score: float


@dataclass
class CompatSet:
    """A resolved set of compatible packages."""

    packages: list[dict]             # {package, version, url, ...}
    constraints: dict                # user-supplied constraints
    status: str                      # "ok" | "conflict" | "partial"
    resolved_at: str                 # ISO timestamp


@dataclass
class ResolveResult:
    """Top-level result returned by the resolver."""

    chosen: Combo
    alternatives: list[Combo]
    from_cache: bool
    cache_key: str


# ---------------------------------------------------------------------------
# Dependency parsing
# ---------------------------------------------------------------------------

# Strip environment markers: everything after ";"
_MARKER_RE = re.compile(r";.*$")
# Strip extras: "package[extra]" -> "package"
_EXTRAS_RE = re.compile(r"\[.*?\]")
# Split name from specifier: "numpy>=2.0,<3" -> ("numpy", ">=2.0,<3")
_DEP_RE = re.compile(
    r"^\s*([A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?)\s*(.*?)\s*$"
)


def _normalize_name(name: str) -> str:
    """PEP 503 normalization: lowercase, collapse [-_.] to '-'."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_dep(req: str) -> tuple[str, Optional[SpecifierSet]]:
    """Parse a PEP 508 requirement string into (name, SpecifierSet).

    Returns (name, None) for bare deps with no version constraint (e.g. "einops").
    Strips environment markers and extras before parsing.
    """
    # Drop markers
    req = _MARKER_RE.sub("", req).strip()
    # Drop extras
    req = _EXTRAS_RE.sub("", req).strip()

    m = _DEP_RE.match(req)
    if not m:
        return _normalize_name(req), None

    name = _normalize_name(m.group(1))
    spec_str = m.group(3).strip()

    if not spec_str:
        return name, None

    try:
        return name, SpecifierSet(spec_str, prereleases=True)
    except Exception:
        return name, None


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

# Representative versions to probe (covers the realistic numpy/scipy/etc range)
_PROBE_VERSIONS = [
    Version(f"{major}.{minor}.{patch}")
    for major in range(0, 4)
    for minor in range(0, 30)
    for patch in (0,)
]


def check_conflicts(combo_metadata: list[PackageMetadata]) -> list[Conflict]:
    """Detect dependency conflicts across a set of packages.

    For each dependency that appears in 2+ packages WITH version specifiers,
    test whether any representative version satisfies ALL specifiers
    simultaneously. If none does, emit a Conflict.

    Returns a list of Conflict objects (empty = no conflicts).
    """
    # Collect all specifiers per dependency name
    # dep_specs: {dep_name: {pkg_name: SpecifierSet}}
    dep_specs: dict[str, dict[str, SpecifierSet]] = {}

    for meta in combo_metadata:
        for raw in meta.requires_dist:
            name, spec = _parse_dep(raw)
            if spec is None:
                # Bare dep — skip
                continue
            if name not in dep_specs:
                dep_specs[name] = {}
            existing = dep_specs[name].get(meta.package)
            if existing is not None:
                # Merge multiple constraints from the same package
                dep_specs[name][meta.package] = SpecifierSet(
                    str(existing) + "," + str(spec), prereleases=True
                )
            else:
                dep_specs[name][meta.package] = spec

    conflicts: list[Conflict] = []

    for dep_name, pkg_specs in dep_specs.items():
        if len(pkg_specs) < 2:
            # Only one package constrains this dep — can't conflict
            continue

        all_specs = list(pkg_specs.values())

        # Check if any probe version satisfies ALL specifiers
        satisfiable = any(
            all(spec.contains(v) for spec in all_specs)
            for v in _PROBE_VERSIONS
        )

        if not satisfiable:
            conflicts.append(
                Conflict(
                    dependency=dep_name,
                    specifiers={pkg: str(spec) for pkg, spec in pkg_specs.items()},
                    merged_range="empty",
                )
            )

    return conflicts


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_combo(matches: list[WheelMatch], conflicts: list[Conflict]) -> float:
    """Score a combination of wheels.

    Higher is better.
    - Recency: sum of version tuples (component-wise, padded to length 3)
    - Penalty: 100_000 per conflict
    """
    recency = 0.0
    for m in matches:
        vt = m.version_tuple
        # Weight major > minor > patch
        padded = (vt + (0, 0, 0))[:3]
        recency += padded[0] * 10_000 + padded[1] * 100 + padded[2]

    penalty = len(conflicts) * 100_000

    return recency - penalty


# ---------------------------------------------------------------------------
# Compat cache
# ---------------------------------------------------------------------------


def _compat_cache_dir() -> Path:
    """Return ~/.cache/gpkg/compat/, creating it if needed."""
    d = Path.home() / ".cache" / "gpkg" / "compat"
    d.mkdir(parents=True, exist_ok=True)
    return d


def compat_cache_key(
    packages: list[str],
    torch: str,
    cuda: str,
    python: str,
    platform: str,
) -> str:
    """Return a stable 16-char hex cache key for the given packages + env.

    Package names are normalized (lowercase, _ -> -) and sorted so that
    order doesn't matter.
    """
    normalized = sorted(
        re.sub(r"[-_.]+", "-", p).lower() for p in packages
    )
    pkg_part = "+".join(normalized)
    raw = f"{pkg_part}_cu{cuda}_torch{torch}_py{python}_{platform}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def write_compat_cache(
    packages: list[str],
    env: dict,
    compat: CompatSet,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Write a CompatSet to TOML at cache_dir/<key>.toml."""
    if cache_dir is None:
        cache_dir = _compat_cache_dir()

    key = compat_cache_key(
        packages,
        env["torch"],
        env["cuda"],
        env["python"],
        env["platform"],
    )
    path = Path(cache_dir) / f"{key}.toml"

    lines: list[str] = []

    # [environment]
    lines.append("[environment]")
    lines.append(f'torch = "{env["torch"]}"')
    lines.append(f'cuda = "{env["cuda"]}"')
    lines.append(f'python = "{env["python"]}"')
    lines.append(f'platform = "{env["platform"]}"')
    lines.append("")

    # [resolution]
    lines.append("[resolution]")
    lines.append(f'status = "{compat.status}"')
    lines.append(f'resolved_at = "{compat.resolved_at}"')
    lines.append("")

    # [[packages]]
    for pkg in compat.packages:
        lines.append("[[packages]]")
        for k, v in pkg.items():
            lines.append(f'{k} = "{v}"')
        lines.append("")

    # [constraints]
    if compat.constraints:
        lines.append("[constraints]")
        for k, v in compat.constraints.items():
            lines.append(f'{k} = "{v}"')
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def read_compat_cache(
    packages: list[str],
    env: dict,
    cache_dir: Optional[Path] = None,
) -> Optional[CompatSet]:
    """Read a CompatSet from the local cache; return None on miss."""
    if cache_dir is None:
        cache_dir = _compat_cache_dir()

    key = compat_cache_key(
        packages,
        env["torch"],
        env["cuda"],
        env["python"],
        env["platform"],
    )
    path = Path(cache_dir) / f"{key}.toml"

    if not path.exists():
        return None

    data = tomllib.loads(path.read_text(encoding="utf-8"))

    resolution = data.get("resolution", {})
    return CompatSet(
        packages=data.get("packages", []),
        constraints=data.get("constraints", {}),
        status=resolution.get("status", ""),
        resolved_at=resolution.get("resolved_at", ""),
    )


def lookup_known_good(
    packages: list[str],
    env: dict,
    client=None,
) -> Optional[CompatSet]:
    """Check local cache first; if miss and client provided, try hosted registry.

    Falls back to https://wheels.mapika.dev/compat/<key>.toml on cache miss.
    Caches locally on a hosted hit.
    """
    local = read_compat_cache(packages, env)
    if local is not None:
        return local

    if client is None:
        return None

    key = compat_cache_key(
        packages,
        env["torch"],
        env["cuda"],
        env["python"],
        env["platform"],
    )
    url = f"https://wheels.mapika.dev/compat/{key}.toml"

    try:
        response = client.get(url)
        if response.status_code != 200:
            return None

        data = tomllib.loads(response.text)
        resolution = data.get("resolution", {})
        compat = CompatSet(
            packages=data.get("packages", []),
            constraints=data.get("constraints", {}),
            status=resolution.get("status", ""),
            resolved_at=resolution.get("resolved_at", ""),
        )
        write_compat_cache(packages, env, compat)
        return compat
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


def generate_candidates(
    all_versions: dict[str, list[WheelMatch]],
    max_per_package: int = 5,
) -> list[Combo]:
    """Generate candidate combos from all matching versions.

    1. Cap each package to top N versions (newest first)
    2. Cartesian product
    3. Prune combos where torch/cuda/abi don't align
    4. Sort by total recency (newest combo first)
    """
    if not all_versions:
        return []

    capped = {
        pkg: versions[:max_per_package]
        for pkg, versions in all_versions.items()
    }

    pkg_names = sorted(capped.keys())
    version_lists = [capped[pkg] for pkg in pkg_names]

    combos: list[Combo] = []
    for combo_tuple in itertools.product(*version_lists):
        matches = list(combo_tuple)

        # GPU constraint pruning
        torch_versions = {m.torch_version for m in matches if m.torch_version}
        if len(torch_versions) > 1:
            continue

        cuda_tags = {m.cuda_tag for m in matches if m.cuda_tag}
        if len(cuda_tags) > 1:
            continue

        abi_values = {m.cxx11_abi for m in matches if m.cxx11_abi is not None}
        if len(abi_values) > 1:
            continue

        combos.append(Combo(matches=matches, conflicts=[], score=0.0))

    # Sort by total recency (newest combo first)
    combos.sort(
        key=lambda c: sum(
            sum(t * (100 ** i) for i, t in enumerate(reversed(m.version_tuple)))
            for m in c.matches
        ),
        reverse=True,
    )

    return combos
