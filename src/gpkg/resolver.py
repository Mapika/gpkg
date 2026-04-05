"""Compatibility resolver: data classes and conflict detection."""

from __future__ import annotations

import hashlib
import itertools
import re
import shutil
import subprocess
import tempfile
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
    cache_dir: Optional[Path] = None,
) -> Optional[CompatSet]:
    """Check local cache first; if miss and client provided, try hosted registry.

    Falls back to https://wheels.mapika.dev/compat/<key>.toml on cache miss.
    Caches locally on a hosted hit.
    """
    local = read_compat_cache(packages, env, cache_dir=cache_dir)
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
        write_compat_cache(packages, env, compat, cache_dir=cache_dir)
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

        # Normalize CUDA tags: "128" and "12" both mean CUDA 12.x
        cuda_majors = {m.cuda_tag[:2] if len(m.cuda_tag) >= 2 else m.cuda_tag
                       for m in matches if m.cuda_tag}
        if len(cuda_majors) > 1:
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


# ---------------------------------------------------------------------------
# Metadata fetching
# ---------------------------------------------------------------------------

_METADATA_TTL = 600  # 10 minutes in seconds


def parse_requires_dist(metadata_text: str) -> list[str]:
    """Extract all Requires-Dist entries from wheel METADATA content."""
    result = []
    for line in metadata_text.splitlines():
        if line.startswith("Requires-Dist:"):
            value = line[len("Requires-Dist:"):].strip()
            if value:
                result.append(value)
    return result


def _metadata_cache_dir() -> Path:
    """Return ~/.cache/gpkg/metadata/, creating it if needed."""
    d = Path.home() / ".cache" / "gpkg" / "metadata"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fetch_wheel_metadata_http(url: str, client) -> Optional[str]:
    """Fetch METADATA from a remote wheel via HTTP range requests.

    Tries to read just the ZIP end-of-central-directory to locate
    the METADATA entry, then fetches only that entry. Falls back to
    downloading the full wheel if range requests are not supported.
    Returns the METADATA text content, or None on failure.
    """
    import struct
    import zipfile
    import io

    try:
        # Step 1: fetch last 65536 bytes to find ZIP EOCD
        resp = client.get(url, headers={"Range": "bytes=-65536"})

        if resp.status_code == 206:
            tail = resp.content
            # Search for EOCD signature (PK\x05\x06)
            eocd_sig = b"PK\x05\x06"
            eocd_pos = tail.rfind(eocd_sig)

            if eocd_pos != -1 and len(tail) - eocd_pos >= 22:
                eocd = tail[eocd_pos:]
                cd_size = struct.unpack_from("<I", eocd, 12)[0]
                cd_offset = struct.unpack_from("<I", eocd, 16)[0]

                # Step 2: fetch central directory
                cd_resp = client.get(
                    url,
                    headers={"Range": f"bytes={cd_offset}-{cd_offset + cd_size - 1}"},
                )
                if cd_resp.status_code == 206:
                    cd_data = cd_resp.content
                    # Scan central directory for *.dist-info/METADATA
                    sig = b"PK\x01\x02"
                    pos = 0
                    while pos < len(cd_data):
                        entry_pos = cd_data.find(sig, pos)
                        if entry_pos == -1:
                            break
                        if entry_pos + 46 > len(cd_data):
                            break
                        fname_len = struct.unpack_from("<H", cd_data, entry_pos + 28)[0]
                        extra_len = struct.unpack_from("<H", cd_data, entry_pos + 30)[0]
                        comment_len = struct.unpack_from("<H", cd_data, entry_pos + 32)[0]
                        local_offset = struct.unpack_from("<I", cd_data, entry_pos + 42)[0]
                        fname_bytes = cd_data[entry_pos + 46: entry_pos + 46 + fname_len]
                        try:
                            fname_str = fname_bytes.decode("utf-8")
                        except UnicodeDecodeError:
                            pos = entry_pos + 46 + fname_len + extra_len + comment_len
                            continue
                        if fname_str.endswith(".dist-info/METADATA"):
                            # Step 3: fetch local file header (30 bytes) to get offsets
                            hdr_resp = client.get(
                                url,
                                headers={"Range": f"bytes={local_offset}-{local_offset + 29}"},
                            )
                            if hdr_resp.status_code == 206 and len(hdr_resp.content) >= 30:
                                local_fname_len = struct.unpack_from("<H", hdr_resp.content, 26)[0]
                                local_extra_len = struct.unpack_from("<H", hdr_resp.content, 28)[0]
                                data_start = local_offset + 30 + local_fname_len + local_extra_len
                                comp_size = struct.unpack_from("<I", cd_data, entry_pos + 20)[0]
                                data_resp = client.get(
                                    url,
                                    headers={"Range": f"bytes={data_start}-{data_start + comp_size - 1}"},
                                )
                                if data_resp.status_code == 206:
                                    compress_method = struct.unpack_from("<H", hdr_resp.content, 8)[0]
                                    if compress_method == 0:
                                        return data_resp.content.decode("utf-8", errors="replace")
                                    elif compress_method == 8:
                                        import zlib
                                        decompressed = zlib.decompress(data_resp.content, -15)
                                        return decompressed.decode("utf-8", errors="replace")
                        pos = entry_pos + 46 + fname_len + extra_len + comment_len

        # Fallback: download full wheel and extract with zipfile
        full_resp = client.get(url)
        if full_resp.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(full_resp.content)) as zf:
                for name in zf.namelist():
                    if name.endswith(".dist-info/METADATA"):
                        return zf.read(name).decode("utf-8", errors="replace")

    except Exception:
        pass

    return None


def fetch_metadata(
    match: WheelMatch,
    sources: list[Source],
    client=None,
) -> Optional[list[str]]:
    """Fetch requires_dist for a wheel match.

    Priority:
    1. Registry requires override (curated data in registry.toml)
    2. Local metadata cache (~/.cache/gpkg/metadata/{package}-{version}.txt, 10 min TTL)
    3. On-demand HTTP fetch via _fetch_wheel_metadata_http
    """
    import time

    # 1. Registry override
    for source in sources:
        if source.package == match.package:
            reqs = get_requires_for_version(source, match.version)
            if reqs is not None:
                return reqs

    # 2. Local cache
    cache_dir = _metadata_cache_dir()
    cache_file = cache_dir / f"{match.package}-{match.version}.txt"
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < _METADATA_TTL:
            content = cache_file.read_text(encoding="utf-8")
            return parse_requires_dist(content)

    # 3. HTTP fetch
    if client is None:
        return None

    metadata_text = _fetch_wheel_metadata_http(match.url, client)
    if metadata_text is None:
        return None

    # Cache the result
    cache_file.write_text(metadata_text, encoding="utf-8")
    return parse_requires_dist(metadata_text)


# ---------------------------------------------------------------------------
# Trial resolution
# ---------------------------------------------------------------------------


def _build_trial_requirements(combo: Combo) -> list[str]:
    """Build a requirements list for uv pip compile from a combo."""
    return [f"{m.package}=={m.version}" for m in combo.matches]


def trial_resolve(combo: Combo, timeout: int = 30) -> bool:
    """Test if a combo resolves using uv pip compile --dry-run.

    Returns True if uv can find a compatible set of transitive dependencies.
    Returns True if uv is not available (can't verify — assume it works).
    """
    uv_path = shutil.which("uv")
    if not uv_path:
        return True

    reqs = _build_trial_requirements(combo)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("\n".join(reqs))
        f.flush()
        req_file = f.name

    try:
        find_links: set[str] = set()
        for m in combo.matches:
            if m.url:
                base = m.url.rsplit("/", 1)[0] + "/"
                find_links.add(base)

        cmd = [
            uv_path, "pip", "compile", req_file,
            "--quiet", "--no-header",
        ]
        for fl in find_links:
            cmd.extend(["--find-links", fl])

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode == 0

    except (subprocess.TimeoutExpired, FileNotFoundError):
        return True
    finally:
        Path(req_file).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main resolver pipeline
# ---------------------------------------------------------------------------


def resolve(
    all_versions: dict[str, list[WheelMatch]],
    env: dict[str, str],
    sources: list[Source],
    client=None,
    *,
    skip_trial: bool = False,
    max_per_package: int = 5,
    max_trial: int = 3,
    cache_dir: Optional[Path] = None,
) -> Optional[ResolveResult]:
    """Main resolver pipeline.

    1. Check known-good cache
    2. Generate candidate combos (GPU-pruned cartesian product)
    3. Fetch metadata + detect conflicts for each combo
    4. Score and rank
    5. Trial-resolve top candidates with uv
    6. Cache the result
    """
    packages = sorted(all_versions.keys())

    if not packages:
        return None

    # 1. Known-good cache lookup
    cached = lookup_known_good(packages, env, client, cache_dir=cache_dir)
    if cached is not None:
        cache_matches = []
        for pkg_info in cached.packages:
            pkg_name = pkg_info["package"]
            if pkg_name in all_versions:
                for m in all_versions[pkg_name]:
                    if m.version == pkg_info["version"]:
                        cache_matches.append(m)
                        break

        if len(cache_matches) == len(packages):
            key = compat_cache_key(
                packages, env["torch"], env["cuda"], env["python"], env["platform"]
            )
            return ResolveResult(
                chosen=Combo(matches=cache_matches, conflicts=[], score=0.0),
                alternatives=[],
                from_cache=True,
                cache_key=key,
            )

    # 2. Generate candidates
    combos = generate_candidates(all_versions, max_per_package)

    if not combos:
        return None

    # 3. Fetch metadata + detect conflicts for each combo
    for combo in combos:
        metadata_list: list[PackageMetadata] = []
        for match in combo.matches:
            requires = fetch_metadata(match, sources, client)
            metadata_list.append(PackageMetadata(
                package=match.package,
                version=match.version,
                requires_dist=requires or [],
            ))

        combo.conflicts = check_conflicts(metadata_list)
        combo.score = score_combo(combo.matches, combo.conflicts)

    # 4. Sort by score (highest first)
    combos.sort(key=lambda c: c.score, reverse=True)

    # 5. Trial resolution (top N zero-conflict combos)
    if not skip_trial:
        verified: list[Combo] = []
        for combo in combos:
            if combo.conflicts:
                continue
            if trial_resolve(combo):
                verified.append(combo)
            if len(verified) >= max_trial:
                break

        if verified:
            combos = verified + [c for c in combos if c not in verified]

    # 6. Cache the winner
    winner = combos[0]
    cache_key = compat_cache_key(
        packages, env["torch"], env["cuda"], env["python"], env["platform"]
    )

    if not winner.conflicts:
        compat = CompatSet(
            packages=[
                {"package": m.package, "version": m.version, "url": m.url}
                for m in winner.matches
            ],
            constraints={},
            status="user-resolved",
            resolved_at=datetime.now(timezone.utc).isoformat(),
        )
        write_compat_cache(packages, env, compat, cache_dir=cache_dir)

    return ResolveResult(
        chosen=winner,
        alternatives=combos[1:3],
        from_cache=False,
        cache_key=cache_key,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_resolve_result(result: ResolveResult) -> str:
    """Format a ResolveResult as plain text for display."""
    lines: list[str] = []

    if result.from_cache and not result.chosen.conflicts:
        lines.append("  \u2713 Known compatible set found")
        lines.append("")
        for m in result.chosen.matches:
            source_short = m.source_desc.split(" \u2014 ")[-1] if " \u2014 " in m.source_desc else m.source_desc
            lines.append(f"  {m.package:<20s} {m.version:<10s} {source_short}")
        return "\n".join(lines)

    if not result.chosen.conflicts:
        lines.append("  \u2713 Compatible set found")
        lines.append("")
        for m in result.chosen.matches:
            source_short = m.source_desc.split(" \u2014 ")[-1] if " \u2014 " in m.source_desc else m.source_desc
            lines.append(f"  {m.package:<20s} {m.version:<10s} {source_short}")
        return "\n".join(lines)

    # Conflict path
    lines.append("  \u2717 Conflicts detected")
    lines.append("")
    for c in result.chosen.conflicts:
        lines.append(f"  Conflict: {c.dependency}")
        for pkg, spec in c.specifiers.items():
            lines.append(f"    {pkg} requires {c.dependency}{spec}")
    lines.append("")

    if result.alternatives:
        lines.append(f"  Found {len(result.alternatives)} alternative set(s):")
        for i, alt in enumerate(result.alternatives, 1):
            lines.append("")
            conflict_note = f" ({len(alt.conflicts)} conflict(s))" if alt.conflicts else ""
            lines.append(f"  [{i}]{conflict_note}")
            for m in alt.matches:
                lines.append(f"      {m.package} {m.version}")
    else:
        lines.append("  Suggestions:")
        lines.append("    \u2022 Remove one of the conflicting packages")
        lines.append("    \u2022 Pin a specific version: gpkg add pkg==VERSION")
        lines.append("    \u2022 Use --build-missing to build from source")

    return "\n".join(lines)


def present_options(result: ResolveResult) -> Optional[Combo]:
    """Present options to the user and get their choice.

    Returns the chosen Combo, or None if the user aborts.
    """
    from gpkg import console

    if not result.alternatives:
        return result.chosen

    console.print(format_resolve_result(result))
    console.print("")

    all_options = [result.chosen] + result.alternatives
    clean_options = [c for c in all_options if not c.conflicts]

    if not clean_options:
        console.print("  [red]No conflict-free sets available.[/red]")
        return None

    if len(clean_options) == 1:
        return clean_options[0]

    console.print(f"  Found {len(clean_options)} compatible sets:\n")
    for i, combo in enumerate(clean_options, 1):
        label = " [dim](recommended)[/dim]" if i == 1 else ""
        console.print(f"  [{i}]{label}")
        for m in combo.matches:
            console.print(f"      {m.package} {m.version}")
        console.print("")

    try:
        choice = input(f"  Choose [1-{len(clean_options)}] or 'q' to abort: ").strip()
        if choice.lower() == "q":
            return None
        idx = int(choice) - 1
        if 0 <= idx < len(clean_options):
            return clean_options[idx]
    except (ValueError, EOFError, KeyboardInterrupt):
        return None

    return clean_options[0]


# ---------------------------------------------------------------------------
# Build reports
# ---------------------------------------------------------------------------


def format_build_report(
    package: str,
    version: str,
    torch: str,
    cuda: str,
    python: str,
    platform: str,
    gpu_arch: str,
    build_time: int,
    gpkg_version: str,
) -> dict:
    """Format a build report for submission to the hosted registry."""
    return {
        "package": package,
        "version": version,
        "torch": torch,
        "cuda": cuda,
        "python": python,
        "platform": platform,
        "gpu_arch": gpu_arch,
        "build_time_seconds": build_time,
        "success": True,
        "gpkg_version": gpkg_version,
    }


def send_build_report(report: dict, client) -> bool:
    """Send a build report to the hosted registry.

    Returns True if the report was accepted.
    """
    try:
        resp = client.post(
            "https://wheels.mapika.dev/api/report",
            json=report,
            timeout=10,
        )
        return resp.status_code in (200, 201, 202)
    except Exception:
        return False
