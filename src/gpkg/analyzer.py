"""Interval arithmetic engine for PEP 440 version specifiers."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from packaging.version import Version


@dataclass
class VersionInterval:
    lower: Optional[Version] = None
    upper: Optional[Version] = None
    lower_inclusive: bool = True
    upper_inclusive: bool = False


def _compatible_release_upper(version: Version) -> Version:
    """Compute upper bound for ~= (compatible release) operator.

    ~=1.4.2 -> <1.5.0  (bump second-to-last, zero last)
    ~=2.4   -> <3.0    (bump first component, zero second)
    """
    release = list(version.release)
    if len(release) < 2:
        raise ValueError(f"Compatible release requires at least 2 version components: {version}")
    # Bump second-to-last, zero the last
    release[-2] += 1
    release[-1] = 0
    return Version(".".join(str(x) for x in release))


def _merge_lower(iv: VersionInterval, ver: Version, inclusive: bool) -> VersionInterval:
    """Take the tighter (higher) lower bound."""
    if iv.lower is None:
        return VersionInterval(
            lower=ver, upper=iv.upper,
            lower_inclusive=inclusive, upper_inclusive=iv.upper_inclusive,
        )
    if ver > iv.lower:
        return VersionInterval(
            lower=ver, upper=iv.upper,
            lower_inclusive=inclusive, upper_inclusive=iv.upper_inclusive,
        )
    if ver == iv.lower and not inclusive:
        # exclusive is tighter than inclusive at same boundary
        return VersionInterval(
            lower=ver, upper=iv.upper,
            lower_inclusive=False, upper_inclusive=iv.upper_inclusive,
        )
    return iv


def _merge_upper(iv: VersionInterval, ver: Version, inclusive: bool) -> VersionInterval:
    """Take the tighter (lower) upper bound."""
    if iv.upper is None:
        return VersionInterval(
            lower=iv.lower, upper=ver,
            lower_inclusive=iv.lower_inclusive, upper_inclusive=inclusive,
        )
    if ver < iv.upper:
        return VersionInterval(
            lower=iv.lower, upper=ver,
            lower_inclusive=iv.lower_inclusive, upper_inclusive=inclusive,
        )
    if ver == iv.upper and not inclusive:
        # exclusive is tighter than inclusive at same boundary
        return VersionInterval(
            lower=iv.lower, upper=ver,
            lower_inclusive=iv.lower_inclusive, upper_inclusive=False,
        )
    return iv


_OP_RE = re.compile(r"(~=|===|==|!=|>=|<=|>|<)\s*(.+)")


def specifier_to_interval(spec_str: str) -> tuple[VersionInterval, set[Version]]:
    """Convert a PEP 440 specifier string to a VersionInterval and exclusion set.

    Handles compound specifiers like '>=1.26,<3.0'.
    Returns (interval, exclusions) where exclusions contains != versions.
    """
    iv = VersionInterval()
    exclusions: set[Version] = set()

    parts = [p.strip() for p in spec_str.split(",")]
    for part in parts:
        m = _OP_RE.match(part)
        if not m:
            raise ValueError(f"Cannot parse specifier part: {part!r}")
        op, ver_str = m.group(1), m.group(2).strip()

        if op == "!=":
            exclusions.add(Version(ver_str))
            continue

        if op == "==":
            # Handle wildcard: ==2.4.*
            if ver_str.endswith(".*"):
                base = ver_str[:-2]
                lower = Version(base)
                # upper: bump last component of base
                parts_base = base.split(".")
                parts_base[-1] = str(int(parts_base[-1]) + 1)
                upper = Version(".".join(parts_base))
                iv = _merge_lower(iv, lower, inclusive=True)
                iv = _merge_upper(iv, upper, inclusive=False)
            else:
                ver = Version(ver_str)
                iv = _merge_lower(iv, ver, inclusive=True)
                iv = _merge_upper(iv, ver, inclusive=True)

        elif op == "~=":
            ver = Version(ver_str)
            upper = _compatible_release_upper(ver)
            iv = _merge_lower(iv, ver, inclusive=True)
            iv = _merge_upper(iv, upper, inclusive=False)

        elif op == ">=":
            iv = _merge_lower(iv, Version(ver_str), inclusive=True)

        elif op == ">":
            iv = _merge_lower(iv, Version(ver_str), inclusive=False)

        elif op == "<=":
            iv = _merge_upper(iv, Version(ver_str), inclusive=True)

        elif op == "<":
            iv = _merge_upper(iv, Version(ver_str), inclusive=False)

        elif op == "===":
            # Arbitrary equality: treat as exact point
            ver = Version(ver_str)
            iv = _merge_lower(iv, ver, inclusive=True)
            iv = _merge_upper(iv, ver, inclusive=True)

    return iv, exclusions


def intersect_intervals(intervals: list[VersionInterval]) -> Optional[VersionInterval]:
    """Intersect a list of VersionIntervals.

    Returns the tightest interval satisfying all constraints, or None if empty.
    """
    result = VersionInterval()

    for iv in intervals:
        if iv.lower is not None:
            result = _merge_lower(result, iv.lower, iv.lower_inclusive)
        if iv.upper is not None:
            result = _merge_upper(result, iv.upper, iv.upper_inclusive)

    # Check for empty interval
    if result.lower is not None and result.upper is not None:
        if result.lower > result.upper:
            return None
        if result.lower == result.upper:
            if not (result.lower_inclusive and result.upper_inclusive):
                return None

    return result


def interval_to_specifier(iv: VersionInterval) -> str:
    """Convert a VersionInterval back to a PEP 440 specifier string."""
    parts = []

    if iv.lower is not None:
        op = ">=" if iv.lower_inclusive else ">"
        parts.append(f"{op}{iv.lower}")

    if iv.upper is not None:
        op = "<=" if iv.upper_inclusive else "<"
        parts.append(f"{op}{iv.upper}")

    return ",".join(parts)


def _pypi_cache_dir() -> Path:
    """Return the PyPI metadata cache directory."""
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    d = base / "gpkg" / "pypi"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _matches_environment(marker_str: str, python_version: str) -> bool:
    """Check if a PEP 508 environment marker matches current env."""
    marker_str = marker_str.strip()
    if not marker_str:
        return True
    # Check python_version markers
    m = re.search(r'python_version\s*([<>=!]+)\s*["\'](\d+\.\d+)["\']', marker_str)
    if m:
        op, ver = m.group(1), m.group(2)
        from packaging.specifiers import SpecifierSet
        try:
            spec = SpecifierSet(f"{op}{ver}")
            return python_version in spec
        except Exception:
            return True
    # extra markers → skip
    if "extra" in marker_str:
        return False
    return True


def parse_pypi_requires_dist(pypi_data: dict, python_version: str = "3.12") -> list[str]:
    """Extract requires_dist from a PyPI JSON response.
    Filters out marker-gated deps that don't match, strips markers.
    """
    raw = pypi_data.get("info", {}).get("requires_dist") or []
    result = []
    for req in raw:
        if ";" in req:
            dep_part, marker = req.split(";", 1)
            if not _matches_environment(marker, python_version):
                continue
            result.append(dep_part.strip())
        else:
            result.append(req.strip())
    return result


def fetch_pypi_metadata(package: str, version: str, client, cache_dir: Optional[Path] = None) -> Optional[dict]:
    """Fetch package metadata from PyPI JSON API with local caching.
    Cache is permanent (PyPI releases are immutable).
    """
    cache_dir = cache_dir or _pypi_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    normalized = package.lower().replace("-", "_").replace(".", "_")
    cache_file = cache_dir / f"{normalized}-{version}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text())

    if client is None:
        return None

    try:
        resp = client.get(f"https://pypi.org/pypi/{package}/{version}/json", timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            cache_file.write_text(json.dumps(data))
            return data
    except Exception:
        pass
    return None
