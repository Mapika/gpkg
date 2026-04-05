"""Lockfile: write, read, compare resolved wheels."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal, Optional
from urllib.parse import unquote

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

if TYPE_CHECKING:
    from gpkg.matching import WheelMatch

LOCKFILE_NAME = "gpkg.lock.toml"


def write_lockfile(
    path: str,
    torch_ver: str,
    cuda_ver: str,
    python_ver: str,
    platform_tag: str,
    cxx11_abi: str,
    wheel_matches: dict[str, WheelMatch],
) -> None:
    """Write resolved wheels to a TOML lockfile for reproducible installs."""
    lines = [
        "# gpkg lockfile -- do not edit manually",
        f'# generated = "{datetime.now(tz=timezone.utc).isoformat()}"',
        "",
        "[environment]",
        f'torch = "{torch_ver}"',
        f'cuda = "{cuda_ver}"',
        f'python = "{python_ver}"',
        f'platform = "{platform_tag}"',
        f'cxx11_abi = "{cxx11_abi}"',
        "",
    ]
    for name, m in wheel_matches.items():
        lines.extend([
            "[[wheels]]",
            f'package = "{name}"',
            f'version = "{m.version}"',
            f'filename = "{m.filename}"',
            f'url = "{unquote(m.url)}"',
            f'source = "{m.source_desc}"',
            f'release_tag = "{m.release_tag}"',
            "",
        ])
    with open(path, "w") as f:
        f.write("\n".join(lines))


def read_lockfile(path: str) -> Optional[dict]:
    """Read and parse a gpkg lockfile. Returns None if not found."""
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return None


@dataclass
class LockChange:
    """A single difference between old lockfile and new resolution."""
    kind: Literal["added", "updated", "removed", "url_changed"]
    package: str
    old_version: Optional[str] = None
    new_version: Optional[str] = None


def compare_lock(
    old_lock: dict,
    new_results: dict[str, WheelMatch],
) -> list[LockChange]:
    """Compare old lockfile with new results, return list of changes."""
    changes: list[LockChange] = []
    old_wheels = {w["package"]: w for w in old_lock.get("wheels", [])}

    for name, m in new_results.items():
        old = old_wheels.get(name)
        if old is None:
            changes.append(LockChange("added", name, new_version=m.version))
        elif old["version"] != m.version:
            changes.append(LockChange("updated", name, old["version"], m.version))
        elif old["url"] != unquote(m.url):
            changes.append(LockChange("url_changed", name, new_version=m.version))

    for name in old_wheels:
        if name not in new_results:
            changes.append(LockChange("removed", name, old_version=old_wheels[name]["version"]))

    return changes


def lockfile_to_wheel_matches(lock_data: dict) -> tuple[dict[str, str], dict]:
    """Convert lockfile data to environment dict + wheel match dicts.

    Returns (env, wheels) where env has keys: torch, cuda, python, platform, cxx11_abi
    and wheels is {package: {url, filename, version, source_desc, release_tag}}.
    """
    env = dict(lock_data.get("environment", {}))
    wheels = {}
    for w in lock_data.get("wheels", []):
        wheels[w["package"]] = {
            "url": w["url"],
            "filename": w["filename"],
            "version": w["version"],
            "source_desc": w.get("source", ""),
            "release_tag": w.get("release_tag", ""),
        }
    return env, wheels


def format_lock_change(change: LockChange) -> str:
    """Format a LockChange for Rich console output."""
    if change.kind == "added":
        return f"  [green]+[/green] {change.package} {change.new_version} (new)"
    if change.kind == "updated":
        return f"  [yellow]~[/yellow] {change.package} {change.old_version} → {change.new_version}"
    if change.kind == "url_changed":
        return f"  [yellow]~[/yellow] {change.package} {change.new_version} (URL changed)"
    # kind == "removed"
    return f"  [red]-[/red] {change.package} {change.old_version} (removed)"
