"""Wheel matching: dataclasses, matchers, search pipeline."""

from __future__ import annotations

import re
from functools import lru_cache
from dataclasses import dataclass
from typing import Optional

import httpx

from gpkg import console
from gpkg.registry import Source
from gpkg.cache import (
    fetch_releases,
    fetch_find_links,
    _render_find_links_url,
    _get_registry_auth,
)
import gpkg.cache as _cache_mod

# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


@dataclass
class WheelMatch:
    package: str
    filename: str
    url: str
    version: str
    torch_version: str
    cuda_tag: str
    python_tag: str
    platform_tag: str
    source_desc: str
    cxx11_abi: Optional[str] = None
    release_tag: str = ""

    @property
    def version_tuple(self) -> tuple[int, ...]:
        """Parse version for proper ordering: '2.8.3' -> (2, 8, 3)."""
        nums = re.findall(r"\d+", self.version.split("+")[0])
        return tuple(int(n) for n in nums)


@dataclass
class RejectReason:
    """Why a candidate wheel was rejected."""
    filename: str
    stage: str  # "regex" | "cuda" | "torch" | "python" | "platform" | "abi" | "torch_compat" | "fetch"
    detail: str = ""

MAX_REJECTIONS = 500


@dataclass
class ExplainReport:
    """Diagnostic report for one source's matching attempt."""
    source: Source
    regex_pattern: str
    releases_scanned: int
    assets_scanned: int
    rejected: list[RejectReason]
    matched: list[WheelMatch]


def torch_minor(version: str) -> str:
    """Extract minor version: '2.11.0' -> '2.11', '2.11' -> '2.11'."""
    parts = version.split(".")
    return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else version


def normalize_cuda(cuda_tag: str, style: str) -> str:
    if style == "full":
        if len(cuda_tag) == 3:
            return f"{cuda_tag[:2]}.{cuda_tag[2]}"
        if len(cuda_tag) == 2:
            return f"{cuda_tag[0]}.{cuda_tag[1]}"
    return cuda_tag


@lru_cache(maxsize=64)
def _parse_version(v: str) -> tuple[int, ...]:
    """Parse '2.11' or '2.4.1' into a comparable tuple."""
    return tuple(int(x) for x in re.findall(r"\d+", v))


def torch_compat_matches(torch_compat: str, target_torch: str) -> bool:
    """Check if target_torch satisfies a torch_compat specifier string.

    Examples: ">=2.4", ">=2.4,<2.6", "<3.0"
    Empty string always matches.
    """
    if not torch_compat:
        return True
    target = _parse_version(target_torch)
    for spec in torch_compat.split(","):
        spec = spec.strip()
        if spec.startswith(">="):
            if target < _parse_version(spec[2:]):
                return False
        elif spec.startswith(">"):
            if target <= _parse_version(spec[1:]):
                return False
        elif spec.startswith("<="):
            if target > _parse_version(spec[2:]):
                return False
        elif spec.startswith("<"):
            if target >= _parse_version(spec[1:]):
                return False
        elif spec.startswith("=="):
            if target != _parse_version(spec[2:]):
                return False
        elif spec.startswith("!="):
            if target == _parse_version(spec[2:]):
                return False
    return True


def cuda_matches(tag: str, style: str, target_cuda: str) -> bool:
    target = target_cuda.replace(".", "")
    if len(target) >= 3:
        major, minor = target[:2], target[2:]
    elif len(target) == 2:
        major, minor = target[0], target[1]
    else:
        major, minor = target, "0"
    if style == "full":
        return tag == f"{major}{minor}"
    if style == "short":
        return tag == major
    return False


def torch_matches(tag: str, target: str, fmt: str = "minor") -> bool:
    """Compare wheel torch tag against user target.

    target can be minor ('2.11') or full ('2.11.0').
    fmt: "minor" (2.11), "packed" (2110), "full" (2.11.0 or 2.9.0andhigher)
    """
    target_m = torch_minor(target)
    if fmt == "minor":
        return tag == target_m
    if fmt == "packed":
        d = tag
        if len(d) == 4:
            return f"{d[0]}.{d[1:3]}" == target_m
        if len(d) == 3:
            return f"{d[0]}.{d[1]}" == target_m
        return tag == target_m
    if fmt == "full":
        tag_clean = re.split(r"[^0-9.]", tag)[0]
        # If target has patch component, match exactly
        if len(target.split(".")) >= 3:
            return tag_clean == target
        # Otherwise, match on minor (any patch)
        parts = tag_clean.split(".")
        if len(parts) >= 2:
            return f"{parts[0]}.{parts[1]}" == target_m
    return tag == target_m


def python_tag_matches(pytag: str, target_python: str) -> bool:
    target_cp = "cp" + target_python.replace(".", "")
    if "abi3" in pytag:
        return int(target_cp[2:]) >= int(pytag.split("-")[0][2:])
    return pytag.startswith(target_cp)


def platform_matches(plat: str, target: str) -> bool:
    if target == "linux_x86_64":
        return "linux_x86_64" in plat or "manylinux" in plat
    if target == "win_amd64":
        return "win_amd64" in plat
    if target == "linux_aarch64":
        return "aarch64" in plat
    return target in plat


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def _check_wheel(
    fname: str,
    source: Source,
    pattern: re.Pattern,
    target_torch: str,
    target_cuda: str,
    target_python: str,
    target_platform: str,
    cxx11_abi: str,
) -> tuple[Optional[dict], Optional[tuple[str, str]]]:
    """Match a single wheel filename against all criteria.

    Returns (groupdict, None) on match, or (None, (stage, detail)) on rejection.
    """
    m = pattern.match(fname)
    if not m:
        return None, ("regex", "")
    g = m.groupdict()
    if not cuda_matches(g["cuda"], source.cuda_style, target_cuda):
        return None, ("cuda", f"wheel={g['cuda']} target={target_cuda}")
    if not torch_matches(g["torch"], target_torch, source.torch_format):
        return None, ("torch", f"wheel={g['torch']} target={target_torch}")
    if not python_tag_matches(g["pytag"], target_python):
        return None, ("python", f"wheel={g['pytag']} target={target_python}")
    if not platform_matches(g["platform"], target_platform):
        return None, ("platform", f"wheel={g['platform']} target={target_platform}")
    if source.has_abi and "abi" in g and g["abi"] != cxx11_abi:
        return None, ("abi", f"wheel={g['abi']} target={cxx11_abi}")
    return g, None


def _build_wheel_match(
    source: Source, fname: str, asset: dict, g: dict, tag: str = "",
) -> WheelMatch:
    """Construct a WheelMatch from matched regex groups and asset metadata."""
    url = asset.get("browser_download_url") or asset.get("url", "")
    return WheelMatch(
        package=source.package,
        filename=fname,
        url=url,
        version=g["version"],
        torch_version=g["torch"],
        cuda_tag=g["cuda"],
        python_tag=g["pytag"],
        platform_tag=g["platform"],
        source_desc=source.description,
        cxx11_abi=g.get("abi"),
        release_tag=tag,
    )


def search_source(
    client: httpx.Client,
    source: Source,
    target_torch: str,
    target_cuda: str,
    target_python: str,
    target_platform: str,
    cxx11_abi: str,
) -> list[WheelMatch]:
    if not source.wheel_name:
        return []
    if not torch_compat_matches(source.torch_compat, target_torch):
        return []
    pattern = source.regex
    matches: list[WheelMatch] = []

    if source.source_type == "find-links":
        url = _render_find_links_url(source.url_template, target_cuda, target_torch)
        try:
            auth_headers = _get_registry_auth(url)
            assets = fetch_find_links(client, url, use_cache=_cache_mod._use_cache, headers=auth_headers)
        except Exception as e:
            console.print(f"    [dim]warn: {source.url_template}: {e}[/dim]")
            return []
        for asset in assets:
            fname = asset["name"]
            g, _ = _check_wheel(
                fname, source, pattern,
                target_torch, target_cuda, target_python, target_platform, cxx11_abi,
            )
            if g is None:
                continue
            matches.append(_build_wheel_match(source, fname, asset, g))
        return matches

    if source.source_type != "github":
        return []
    try:
        releases = fetch_releases(client, source.repo, source.scan_tags, use_cache=_cache_mod._use_cache)
    except Exception as e:
        console.print(f"    [dim]warn: {source.repo}: {e}[/dim]")
        return []
    for release in releases:
        tag = release["tag_name"]
        for asset in release.get("assets", []):
            fname = asset["name"]
            g, _ = _check_wheel(
                fname, source, pattern,
                target_torch, target_cuda, target_python, target_platform, cxx11_abi,
            )
            if g is None:
                continue
            matches.append(_build_wheel_match(source, fname, asset, g, tag))
    return matches


def search_source_explain(
    client: httpx.Client,
    source: Source,
    target_torch: str,
    target_cuda: str,
    target_python: str,
    target_platform: str,
    cxx11_abi: str,
) -> ExplainReport:
    """Like search_source but collects detailed rejection reasons."""
    pattern = source.regex if source.wheel_name else None
    report = ExplainReport(
        source=source,
        regex_pattern=pattern.pattern if pattern else "(no wheel_name)",
        releases_scanned=0,
        assets_scanned=0,
        rejected=[],
        matched=[],
    )
    if not source.wheel_name:
        return report
    if not torch_compat_matches(source.torch_compat, target_torch):
        report.rejected.append(RejectReason("", "torch_compat", f"source requires {source.torch_compat}, target={target_torch}"))
        return report

    if source.source_type == "find-links":
        url = _render_find_links_url(source.url_template, target_cuda, target_torch)
        try:
            auth_headers = _get_registry_auth(url)
            assets = fetch_find_links(client, url, use_cache=_cache_mod._use_cache, headers=auth_headers)
        except Exception as e:
            report.rejected.append(RejectReason("", "fetch", f"HTTP error: {e}"))
            return report
        report.releases_scanned = 1
        for asset in assets:
            fname = asset["name"]
            if not fname.endswith(".whl"):
                continue
            report.assets_scanned += 1
            g, rejection = _check_wheel(
                fname, source, pattern,
                target_torch, target_cuda, target_python, target_platform, cxx11_abi,
            )
            if g is None:
                if len(report.rejected) < MAX_REJECTIONS:
                    report.rejected.append(RejectReason(fname, rejection[0], rejection[1]))
                continue
            report.matched.append(_build_wheel_match(source, fname, asset, g))
        return report

    if source.source_type != "github":
        return report
    try:
        releases = fetch_releases(client, source.repo, source.scan_tags, use_cache=_cache_mod._use_cache)
    except Exception as e:
        report.rejected.append(RejectReason("", "fetch", f"API error: {e}"))
        return report
    report.releases_scanned = len(releases)
    for release in releases:
        tag = release["tag_name"]
        for asset in release.get("assets", []):
            fname = asset["name"]
            if not fname.endswith(".whl"):
                continue
            report.assets_scanned += 1
            g, rejection = _check_wheel(
                fname, source, pattern,
                target_torch, target_cuda, target_python, target_platform, cxx11_abi,
            )
            if g is None:
                if len(report.rejected) < MAX_REJECTIONS:
                    report.rejected.append(RejectReason(fname, rejection[0], rejection[1]))
                continue
            report.matched.append(_build_wheel_match(source, fname, asset, g, tag))
    return report


_NON_NUMERIC_DOT = re.compile(r"[^0-9.]")


def _normalize_torch_display(tv_raw: str, fmt: str) -> Optional[str]:
    """Normalize a raw torch tag to 'major.minor' for display. Returns None for nightlies."""
    if fmt == "packed":
        d = tv_raw
        tv = f"{d[0]}.{d[1:3]}" if len(d) == 4 else f"{d[0]}.{d[1]}" if len(d) == 3 else d
    elif fmt == "full":
        parts = _NON_NUMERIC_DOT.split(tv_raw)[0].split(".")
        tv = f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else tv_raw
    else:
        tv = tv_raw
    try:
        major = int(tv.split(".")[0])
    except (ValueError, IndexError):
        return None
    if major >= 20:  # skip nightlies (CalVer like 25.09)
        return None
    return tv


def scan_available_combos(
    client: httpx.Client,
    sources: list[Source],
    target_platform: str,
) -> set[tuple[str, str]]:
    combos: set[tuple[str, str]] = set()
    for source in sources:
        if not source.wheel_name:
            continue

        if source.source_type == "find-links":
            # Skip sources with URL placeholders — would require O(cuda*torch) fetches
            if "{cuda}" in source.url_template or "{torch}" in source.url_template:
                continue
            try:
                auth_headers = _get_registry_auth(source.url_template)
                assets = fetch_find_links(client, source.url_template, use_cache=_cache_mod._use_cache, headers=auth_headers)
            except Exception:
                continue
            pattern = source.regex
            for asset in assets:
                m = pattern.match(asset["name"])
                if not m:
                    continue
                g = m.groupdict()
                if not platform_matches(g["platform"], target_platform):
                    continue
                tv = _normalize_torch_display(g["torch"], source.torch_format)
                if tv is None:
                    continue
                combos.add((normalize_cuda(g["cuda"], source.cuda_style), tv))
            continue

        if source.source_type != "github":
            continue
        pattern = source.regex
        try:
            releases = fetch_releases(client, source.repo, source.scan_tags, use_cache=_cache_mod._use_cache)
        except Exception:
            continue
        for release in releases:
            for asset in release.get("assets", []):
                m = pattern.match(asset["name"])
                if not m:
                    continue
                g = m.groupdict()
                if not platform_matches(g["platform"], target_platform):
                    continue
                tv = _normalize_torch_display(g["torch"], source.torch_format)
                if tv is None:
                    continue
                combos.add((normalize_cuda(g["cuda"], source.cuda_style), tv))
    return combos


def pick_best(matches: list[WheelMatch]) -> Optional[WheelMatch]:
    if not matches:
        return None

    def key(m: WheelMatch) -> tuple:
        manylinux = 1 if "manylinux" in m.platform_tag else 0
        return (m.version_tuple, -manylinux)

    return max(matches, key=key)
