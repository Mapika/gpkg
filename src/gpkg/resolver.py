"""Compatibility resolver: data classes and conflict detection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from gpkg.matching import WheelMatch


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
