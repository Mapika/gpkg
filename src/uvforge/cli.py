"""uvforge CLI -- find prebuilt CUDA wheels, generate uv pyproject.toml."""

from __future__ import annotations

import argparse
import json
import os
import platform as platform_mod
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

import httpx
from rich.console import Console
from rich.table import Table

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

console = Console(highlight=False)

REMOTE_REGISTRY = (
    "https://raw.githubusercontent.com/Mapika/uvforge/main/src/uvforge/registry.toml"
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
    scan_tags: int = 8

    _regex: Optional[re.Pattern] = field(default=None, repr=False, compare=False)

    @property
    def regex(self) -> re.Pattern:
        if self._regex is None:
            self._regex = _template_to_regex(self.wheel_name)
        return self._regex


def _template_to_regex(template: str) -> re.Pattern:
    """Convert a wheel_name template to a compiled regex.

    Placeholders: {version}, {cuda}, {torch}, {pytag}, {platform}, {abi}, {hash}
    """
    parts = re.split(r"(\{[a-z]+\})", template)
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
            scan_tags=e.get("scan_tags", 8),
        )
        for e in data.get("sources", [])
    ]


def load_registry_with_fallback(client: httpx.Client) -> list[Source]:
    """Try remote registry first, fall back to bundled."""
    try:
        sources = load_registry(REMOTE_REGISTRY, client)
        if sources:
            return sources
    except Exception:
        pass
    return load_registry(LOCAL_REGISTRY, client)


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


def normalize_cuda(cuda_tag: str, style: str) -> str:
    if style == "full":
        if len(cuda_tag) == 3:
            return f"{cuda_tag[:2]}.{cuda_tag[2]}"
        if len(cuda_tag) == 2:
            return f"{cuda_tag[0]}.{cuda_tag[1]}"
    return cuda_tag


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
    """Compare wheel torch tag against user target (e.g. '2.11').

    fmt: "minor" (2.11), "packed" (2110), "full" (2.11.0 or 2.9.0andhigher)
    """
    if fmt == "minor":
        return tag == target
    if fmt == "packed":
        # "2110" -> "2.11", "280" -> "2.8", "2100" -> "2.10"
        d = tag
        if len(d) == 4:
            return f"{d[0]}.{d[1:3]}" == target
        if len(d) == 3:
            return f"{d[0]}.{d[1]}" == target
        return tag == target
    if fmt == "full":
        # "2.9.0" -> "2.9", "2.9.0andhigher" -> "2.9"
        parts = re.split(r"[^0-9.]", tag)[0].split(".")
        if len(parts) >= 2:
            return f"{parts[0]}.{parts[1]}" == target
    return tag == target


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
# GitHub API (cached)
# ---------------------------------------------------------------------------

_release_cache: dict[str, list[dict]] = {}


def get_headers() -> dict:
    h = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def fetch_releases(client: httpx.Client, repo: str, count: int) -> list[dict]:
    key = f"{repo}:{count}"
    if key in _release_cache:
        return _release_cache[key]
    resp = client.get(
        f"https://api.github.com/repos/{repo}/releases",
        params={"per_page": count},
    )
    if resp.status_code == 403:
        console.print("[red]GitHub API rate limit hit.[/red] Set GITHUB_TOKEN env var.")
        sys.exit(1)
    resp.raise_for_status()
    data = resp.json()
    _release_cache[key] = data
    return data


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def search_source(
    client: httpx.Client,
    source: Source,
    target_torch: str,
    target_cuda: str,
    target_python: str,
    target_platform: str,
    cxx11_abi: str,
) -> list[WheelMatch]:
    if source.source_type != "github" or not source.wheel_name:
        return []
    pattern = source.regex
    matches: list[WheelMatch] = []
    try:
        releases = fetch_releases(client, source.repo, source.scan_tags)
    except Exception as e:
        console.print(f"    [dim]warn: {source.repo}: {e}[/dim]")
        return []
    for release in releases:
        tag = release["tag_name"]
        for asset in release.get("assets", []):
            fname = asset["name"]
            m = pattern.match(fname)
            if not m:
                continue
            g = m.groupdict()
            if not cuda_matches(g["cuda"], source.cuda_style, target_cuda):
                continue
            if not torch_matches(g["torch"], target_torch, source.torch_format):
                continue
            if not python_tag_matches(g["pytag"], target_python):
                continue
            if not platform_matches(g["platform"], target_platform):
                continue
            if source.has_abi and "abi" in g and g["abi"] != cxx11_abi:
                continue
            matches.append(
                WheelMatch(
                    package=source.package,
                    filename=fname,
                    url=asset["browser_download_url"],
                    version=g["version"],
                    torch_version=g["torch"],
                    cuda_tag=g["cuda"],
                    python_tag=g["pytag"],
                    platform_tag=g["platform"],
                    source_desc=source.description,
                    cxx11_abi=g.get("abi"),
                    release_tag=tag,
                )
            )
    return matches


def scan_available_combos(
    client: httpx.Client,
    sources: list[Source],
    target_platform: str,
) -> set[tuple[str, str]]:
    combos: set[tuple[str, str]] = set()
    for source in sources:
        if source.source_type != "github" or not source.wheel_name:
            continue
        pattern = source.regex
        try:
            releases = fetch_releases(client, source.repo, source.scan_tags)
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
                tv_raw = g["torch"]
                # Normalize torch for display
                if source.torch_format == "packed":
                    d = tv_raw
                    tv = f"{d[0]}.{d[1:3]}" if len(d) == 4 else f"{d[0]}.{d[1]}" if len(d) == 3 else d
                elif source.torch_format == "full":
                    parts = re.split(r"[^0-9.]", tv_raw)[0].split(".")
                    tv = f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else tv_raw
                else:
                    tv = tv_raw
                if len(tv) > 5:  # skip nightlies like 25.09
                    continue
                combos.add((normalize_cuda(g["cuda"], source.cuda_style), tv))
    return combos


def pick_best(matches: list[WheelMatch]) -> Optional[WheelMatch]:
    if not matches:
        return None

    def key(m: WheelMatch) -> tuple:
        manylinux = 1 if "manylinux" in m.platform_tag else 0
        return (m.version_tuple, -manylinux)

    matches.sort(key=key, reverse=True)
    return matches[0]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect_platform() -> str:
    if sys.platform.startswith("linux"):
        return "linux_aarch64" if platform_mod.machine() == "aarch64" else "linux_x86_64"
    if sys.platform == "win32":
        return "win_amd64"
    if sys.platform == "darwin":
        return "macosx_arm64" if platform_mod.machine() == "arm64" else "macosx_x86_64"
    return "linux_x86_64"


def detect_python() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def detect_cuda() -> Optional[str]:
    """Try to detect CUDA version from nvcc or torch."""
    import shutil
    import subprocess

    # Try nvcc
    if shutil.which("nvcc"):
        try:
            out = subprocess.check_output(["nvcc", "--version"], text=True)
            m = re.search(r"release (\d+)\.(\d+)", out)
            if m:
                return f"{m.group(1)}{m.group(2)}"
        except Exception:
            pass
    # Try torch
    try:
        import torch

        if torch.version.cuda:
            parts = torch.version.cuda.split(".")
            return f"{parts[0]}{parts[1]}"
    except Exception:
        pass
    return None


def detect_torch() -> Optional[str]:
    """Try to detect PyTorch version from installed package."""
    try:
        import torch

        parts = torch.__version__.split(".")
        return f"{parts[0]}.{parts[1]}"
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# TOML generation
# ---------------------------------------------------------------------------


def generate_toml(
    project_name: str,
    python_version: str,
    torch_version: str,
    cuda_version: str,
    wheel_matches: dict[str, WheelMatch],
) -> str:
    cuda_clean = cuda_version.replace(".", "")
    cuda_idx = f"cu{cuda_clean}" if len(cuda_clean) >= 3 else f"cu{cuda_clean}0"

    deps = [f'"torch>={torch_version}"']
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        prog="uvforge",
        description="Find prebuilt CUDA wheels across community sources, generate a uv pyproject.toml.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              uvforge --torch 2.11 --cuda 130 flash-attn flash-attn-3
              uvforge --torch 2.10 --cuda 130 causal-conv1d mamba-ssm natten
              uvforge flash-attn causal-conv1d          # auto-detect torch + cuda
              uvforge --available causal-conv1d natten
              uvforge --list
        """),
    )
    p.add_argument("packages", nargs="*", help="packages to resolve")
    p.add_argument("--torch", default=None, help="PyTorch version (e.g. 2.11; default: auto-detect)")
    p.add_argument("--cuda", default=None, help="CUDA version (e.g. 130; default: auto-detect)")
    p.add_argument("--python", default=None, help="Python version (default: detect)")
    p.add_argument("--platform", default=None, help="Platform (default: detect)")
    p.add_argument("--cxx11-abi", default="TRUE", choices=["TRUE", "FALSE"])
    p.add_argument("--project", default="my-project", help="name in pyproject.toml")
    p.add_argument("-o", "--output", default=None, help="write pyproject.toml to file")
    p.add_argument("--registry", default=None, help="path or URL to registry.toml")
    p.add_argument("--extra-registry", action="append", default=[], help="merge additional registries")
    p.add_argument("--list", action="store_true", help="list all sources")
    p.add_argument("--available", action="store_true", help="show cuda/torch combos")
    p.add_argument("--all", action="store_true", help="show all matching wheels")
    p.add_argument("--json", action="store_true", dest="as_json")
    p.add_argument("-V", "--version", action="version", version=f"%(prog)s {__import__('uvforge').__version__}")

    # Tab completion (optional: pip install argcomplete)
    try:
        import argcomplete
        argcomplete.autocomplete(p)
    except ImportError:
        pass

    args = p.parse_args()
    client = httpx.Client(headers=get_headers(), timeout=30, follow_redirects=True)

    # -- Load registries ---------------------------------------------------
    all_sources: list[Source] = []
    if args.registry:
        all_sources.extend(load_registry(args.registry, client))
    else:
        all_sources.extend(load_registry_with_fallback(client))
    for extra in args.extra_registry:
        try:
            all_sources.extend(load_registry(extra, client))
        except Exception as e:
            console.print(f"[red]Error loading {extra}:[/red] {e}")

    pkg_sources: dict[str, list[Source]] = {}
    for s in all_sources:
        pkg_sources.setdefault(s.package, []).append(s)

    # -- --list ------------------------------------------------------------
    if args.list:
        t = Table(title="Registered wheel sources")
        t.add_column("Package", style="bold cyan")
        t.add_column("Description")
        t.add_column("Repo / URL", style="dim")
        t.add_column("CUDA")
        t.add_column("ABI")
        for s in all_sources:
            loc = s.repo if s.source_type == "github" else (s.url_template or "-")
            t.add_row(s.package, s.description, loc, s.cuda_style, "yes" if s.has_abi else "-")
        console.print(t)
        client.close()
        return

    if not args.packages:
        p.error("specify packages, or use --list")

    plat = args.platform or detect_platform()

    # -- --available -------------------------------------------------------
    if args.available:
        for pkg in args.packages:
            srcs = pkg_sources.get(pkg, [])
            if not srcs:
                console.print(f"[yellow]{pkg}:[/yellow] not in registry")
                continue
            combos = scan_available_combos(client, srcs, plat)
            if not combos:
                console.print(f"[yellow]{pkg}:[/yellow] nothing found for {plat}")
                continue
            t = Table(title=f"{pkg} -- available wheels ({plat})")
            t.add_column("CUDA", style="cyan")
            t.add_column("PyTorch versions", style="green")
            by_cuda: dict[str, list[str]] = {}
            for c, tv in sorted(combos):
                by_cuda.setdefault(c, []).append(tv)
            for c in sorted(by_cuda):
                t.add_row(c, ", ".join(sorted(by_cuda[c], reverse=True)))
            console.print(t)
            console.print()
        client.close()
        return

    # -- Auto-detect torch/cuda if not provided ----------------------------
    torch_ver = args.torch or detect_torch()
    cuda_ver = args.cuda or detect_cuda()

    if not torch_ver:
        p.error("--torch required (could not auto-detect; is PyTorch installed?)")
    if not cuda_ver:
        p.error("--cuda required (could not auto-detect; is nvcc or PyTorch+CUDA available?)")

    py_ver = args.python or detect_python()

    # -- Main search -------------------------------------------------------
    console.print("\n[bold]uvforge[/bold] -- searching prebuilt wheels")
    console.print(
        f"  torch={torch_ver}  cuda={cuda_ver}  python={py_ver}  "
        f"platform={plat}  cxx11abi={args.cxx11_abi}\n"
    )

    results: dict[str, WheelMatch] = {}
    all_matches: dict[str, list[WheelMatch]] = {}

    for pkg in args.packages:
        srcs = pkg_sources.get(pkg, [])
        if not srcs:
            console.print(f"  [yellow]![/yellow]  {pkg}: not in registry")
            continue

        n = len(srcs)
        console.print(f"  [dim]{n} source{'s' if n > 1 else ''}[/dim] {pkg}", end="  ")
        matches: list[WheelMatch] = []
        for src in srcs:
            matches.extend(
                search_source(client, src, torch_ver, cuda_ver, py_ver, plat, args.cxx11_abi)
            )
        all_matches[pkg] = matches

        if not matches:
            console.print("[red]no match[/red]")
            combos = scan_available_combos(client, srcs, plat)
            nearby = [
                (c, tv) for c, tv in combos
                if c.startswith(cuda_ver.replace(".", "")[:2]) and tv.startswith("2.")
            ]
            if nearby:
                tvs = sorted({tv for _, tv in nearby}, reverse=True)
                console.print(f"          [dim]nearest: torch {', '.join(tvs[:6])}[/dim]")
            continue

        best = pick_best(matches)
        results[pkg] = best
        console.print(f"[green]ok[/green] {best.filename}")
        console.print(f"          [dim]via {best.source_desc} ({best.release_tag})[/dim]")

    # -- Show all ----------------------------------------------------------
    if args.all and any(all_matches.values()):
        console.print()
        t = Table(title="All matching wheels")
        t.add_column("Package")
        t.add_column("Ver")
        t.add_column("Source")
        t.add_column("Tag")
        t.add_column("Filename")
        for pkg, ms in all_matches.items():
            for m in ms:
                t.add_row(pkg, m.version, m.source_desc[:35], m.release_tag, m.filename)
        console.print(t)

    if not results:
        console.print("\n[red]No wheels resolved.[/red]")
        console.print("  [dim]•[/dim] Try [bold]--available[/bold] to see what torch/cuda combos exist")
        console.print("  [dim]•[/dim] Need to compile? [bold]source build-env.sh[/bold] for 5-10x faster builds")
        console.print("  [dim]•[/dim] Set [bold]GITHUB_TOKEN[/bold] to avoid API rate limits\n")
        client.close()
        sys.exit(1)

    # -- JSON output -------------------------------------------------------
    if args.as_json:
        out = {
            n: {
                "url": unquote(m.url),
                "filename": m.filename,
                "version": m.version,
                "source": m.source_desc,
            }
            for n, m in results.items()
        }
        print(json.dumps(out, indent=2))
        client.close()
        return

    # -- TOML output -------------------------------------------------------
    toml = generate_toml(args.project, py_ver, torch_ver, cuda_ver, results)

    console.print("\n[bold green]pyproject.toml[/bold green]\n")
    if args.output:
        with open(args.output, "w") as f:
            f.write(toml)
        console.print(f"  Written to [bold]{args.output}[/bold]\n")
    else:
        print(toml)
    console.print("[dim]Next: uv sync[/dim]")
    client.close()


if __name__ == "__main__":
    main()
