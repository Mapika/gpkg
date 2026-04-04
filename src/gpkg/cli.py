"""gpkg CLI — GPU package manager. Find prebuilt CUDA wheels, build missing ones."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from urllib.parse import unquote

import httpx
from rich.table import Table

from gpkg import console
from gpkg.detect import detect_platform, detect_python, detect_cuda, detect_torch
from gpkg.output import generate_toml, merge_into_pyproject, _fmt_size, _fmt_age
from gpkg.lockfile import (
    LOCKFILE_NAME, write_lockfile, read_lockfile, compare_lock,
    format_lock_change, lockfile_to_wheel_matches,
)
from gpkg.registry import Source, load_registry, load_registry_with_fallback
import gpkg.cache as _cache_mod
from gpkg.cache import parse_duration, cache_info, cache_clean, get_headers
from gpkg.matching import (
    WheelMatch, torch_minor,
    search_source, search_source_explain, scan_available_combos, pick_best,
)
from gpkg.doctor import doctor_check
from gpkg.build import (
    detect_gpu_arch, detect_build_jobs, build_env_vars, ensure_ninja,
    find_cached_wheel, build_wheel,
)

# Package name -> Python import name
IMPORT_MAP = {
    "flash-attn": "flash_attn",
    "flash-attn-3": "flash_attn_3",
    "causal-conv1d": "causal_conv1d",
    "mamba-ssm": "mamba_ssm",
    "grouped-gemm": "grouped_gemm",
    "sageattention": "sageattention",
    "natten": "natten",
}


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_test(args: argparse.Namespace) -> None:
    """Verify installed packages and GPU."""
    packages = args.packages or list(IMPORT_MAP.keys())
    all_ok = True

    # Test torch + CUDA
    result = subprocess.run(
        [sys.executable, "-c",
         "import torch; print(f'{torch.__version__},{torch.version.cuda or \"none\"},{torch.cuda.is_available()},{torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode == 0:
        parts = result.stdout.strip().split(",")
        ver, cuda, gpu_ok, gpu_name = parts[0], parts[1], parts[2] == "True", parts[3]
        if gpu_ok:
            console.print(f"  [green]ok[/green]  torch {ver}+cu{cuda}  GPU: {gpu_name}")
        else:
            console.print(f"  [yellow]!![/yellow]  torch {ver}+cu{cuda}  [yellow]no GPU detected[/yellow]")
            all_ok = False
    else:
        console.print("  [red]FAIL[/red]  torch  not installed")
        all_ok = False

    # Test each package
    for pkg in packages:
        mod_name = IMPORT_MAP.get(pkg, pkg.replace("-", "_"))
        result = subprocess.run(
            [sys.executable, "-c",
             f"import {mod_name}; print(getattr({mod_name}, '__version__', 'ok'))"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            console.print(f"  [green]ok[/green]  {pkg} {result.stdout.strip()}")
        else:
            err = result.stderr.strip().split("\n")[-1] if result.stderr else "import failed"
            console.print(f"  [red]FAIL[/red]  {pkg}  ({err})")
            all_ok = False

    sys.exit(0 if all_ok else 1)


def cmd_install(args: argparse.Namespace) -> None:
    """Install from lockfile — generate pyproject.toml without network."""
    lock_data = read_lockfile(args.lockfile)
    if lock_data is None:
        console.print(f"[red]No lockfile found at {args.lockfile}.[/red] Run gpkg --lock first.")
        sys.exit(1)
    env, wheels = lockfile_to_wheel_matches(lock_data)
    if not wheels:
        console.print("[yellow]Lockfile has no wheels.[/yellow]")
        return

    class _Wheel:
        def __init__(self, url: str):
            self.url = url

    locked = {name: _Wheel(w["url"]) for name, w in wheels.items()}
    toml = generate_toml(
        args.project,
        env.get("python", detect_python()),
        env.get("torch", ""),
        env.get("cuda", ""),
        locked,
    )

    if args.output:
        with open(args.output, "w") as f:
            f.write(toml)
        console.print(f"[bold green]Written to {args.output}[/bold green] (from {args.lockfile})")
    else:
        print(toml)

    if args.sync:
        console.print("\n[bold]Running uv sync...[/bold]")
        sys.exit(subprocess.run(["uv", "sync"], timeout=600).returncode)


def cmd_cache_info() -> None:
    """Show cache statistics."""
    info = cache_info()
    console.print(f"Cache: {info['path']}")
    console.print(f"Files: {info['files']} ({_fmt_size(info['bytes'])})")
    if info["files"]:
        console.print(f"Oldest: {_fmt_age(info['oldest_age_s'])}")
        console.print(f"Newest: {_fmt_age(info['newest_age_s'])}")


def cmd_cache_clean(args: argparse.Namespace) -> None:
    """Clean cached data."""
    max_age = parse_duration(args.older_than) if args.older_than else None
    count, freed = cache_clean(max_age)
    console.print(f"Cleaned {count} files ({_fmt_size(freed)} freed)")


def _version_sort_key(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))


def cmd_list(args: argparse.Namespace, sources: list[Source], client: httpx.Client) -> None:
    """List all registered sources."""
    if args.as_json:
        out = [
            {
                "package": s.package,
                "description": s.description,
                "type": s.source_type,
                "repo": s.repo if s.source_type == "github" else s.url_template,
            }
            for s in sources
        ]
        print(json.dumps(out, indent=2))
        return
    t = Table(title="Registered wheel sources")
    t.add_column("Package", style="bold cyan")
    t.add_column("Description")
    t.add_column("Repo / URL", style="dim")
    t.add_column("CUDA")
    t.add_column("ABI")
    for s in sources:
        loc = s.repo if s.source_type == "github" else (s.url_template or "-")
        t.add_row(s.package, s.description, loc, s.cuda_style, "yes" if s.has_abi else "-")
    console.print(t)


def cmd_available(args: argparse.Namespace, pkg_sources: dict, plat: str, client: httpx.Client) -> None:
    """Show available torch/cuda combos."""
    if args.as_json:
        out = {}
        for pkg in args.packages:
            srcs = pkg_sources.get(pkg, [])
            if not srcs:
                continue
            combos = scan_available_combos(client, srcs, plat)
            out[pkg] = {plat: [{"cuda": c, "torch": t} for c, t in sorted(combos)]}
        print(json.dumps(out, indent=2))
        return
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
            t.add_row(c, ", ".join(sorted(by_cuda[c], key=_version_sort_key, reverse=True)))
        console.print(t)
        console.print()


def _resolve_packages(
    args: argparse.Namespace,
    pkg_sources: dict[str, list[Source]],
    client: httpx.Client,
    torch_ver: str,
    cuda_ver: str,
    py_ver: str,
    plat: str,
) -> tuple[dict[str, WheelMatch], dict[str, list[WheelMatch]]]:
    """Run the main search loop. Returns (results, all_matches)."""
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

        if args.explain:
            console.print()
            for src in srcs:
                report = search_source_explain(
                    client, src, torch_ver, cuda_ver, py_ver, plat, args.cxx11_abi
                )
                matches.extend(report.matched)
                _print_explain_report(src, report)
        else:
            for src in srcs:
                matches.extend(
                    search_source(client, src, torch_ver, cuda_ver, py_ver, plat, args.cxx11_abi)
                )

        all_matches[pkg] = matches

        if not matches:
            if args.build_missing:
                built_match = _try_build(pkg, py_ver, plat, torch_ver, cuda_ver, args.explain)
                if built_match:
                    results[pkg] = built_match
                continue
            if not args.explain:
                console.print("[red]no match[/red]")
            else:
                console.print("\n    [red]no match across all sources[/red]")
            _print_nearest(srcs, cuda_ver, plat, client)
            continue

        best = pick_best(matches)
        results[pkg] = best
        if args.explain:
            console.print(f"\n    [bold green]Best pick:[/bold green] {best.filename}")
            console.print(f"    [dim]version={best.version}  source={best.source_desc}  tag={best.release_tag}[/dim]")
        else:
            console.print(f"[green]ok[/green] {best.filename}")
            console.print(f"          [dim]via {best.source_desc} ({best.release_tag})[/dim]")

    return results, all_matches


def _print_explain_report(src: Source, report) -> None:
    """Print detailed matching diagnostics for one source."""
    console.print(f"\n    [bold cyan]Source:[/bold cyan] {src.description}")
    console.print(f"    [dim]repo:[/dim]        {src.repo}")
    console.print(f"    [dim]wheel_name:[/dim]  {src.wheel_name}")
    console.print(f"    [dim]regex:[/dim]       {report.regex_pattern}")
    compat_str = f"  torch_compat: {src.torch_compat}" if src.torch_compat else ""
    console.print(f"    [dim]cuda_style:[/dim]  {src.cuda_style}  torch_format: {src.torch_format}  has_abi: {src.has_abi}{compat_str}")
    console.print(f"    [dim]releases:[/dim]    {report.releases_scanned}  assets (wheels): {report.assets_scanned}")
    console.print(f"    [dim]matched:[/dim]     {len(report.matched)}  rejected: {len(report.rejected)}")
    if report.rejected:
        by_stage: dict[str, int] = {}
        for r in report.rejected:
            by_stage[r.stage] = by_stage.get(r.stage, 0) + 1
        summary = "  ".join(f"{s}={c}" for s, c in sorted(by_stage.items()))
        console.print(f"    [dim]rejected by:[/dim] {summary}")
        shown = 0
        for r in report.rejected:
            if r.stage != "regex" and r.detail and shown < 5:
                console.print(f"      [dim]✗ {r.stage}:[/dim] {r.filename}  ({r.detail})")
                shown += 1
    if report.matched:
        for m in report.matched:
            console.print(f"      [green]✓[/green] {m.filename}  (v{m.version}, {m.release_tag})")


def _local_wheel_match(pkg: str, whl: Path, py_ver: str, plat: str, torch_ver: str, cuda_ver: str) -> WheelMatch:
    """Create a WheelMatch from a locally-built or cached wheel."""
    pytag = f"cp{py_ver.replace('.', '')}"
    return WheelMatch(
        package=pkg, filename=whl.name, url=whl.as_uri(),
        version=whl.name.split("-")[1], torch_version=torch_minor(torch_ver),
        cuda_tag=cuda_ver, python_tag=pytag, platform_tag=plat,
        source_desc="local-build",
    )


def _try_build(
    pkg: str, py_ver: str, plat: str, torch_ver: str, cuda_ver: str, explain: bool,
) -> WheelMatch | None:
    """Try cached wheel or build from source. Returns WheelMatch or None."""
    pytag = f"cp{py_ver.replace('.', '')}"
    cached = find_cached_wheel(pkg, pytag, plat, torch_ver, cuda_ver)
    if cached:
        if not explain:
            console.print(f"[cyan]cached[/cyan] {cached.name}")
        return _local_wheel_match(pkg, cached, py_ver, plat, torch_ver, cuda_ver)

    gpu_arch = detect_gpu_arch()
    if gpu_arch is None:
        if not explain:
            console.print("[red]no match[/red] (--build-missing requires nvidia-smi)")
        return None

    jobs = detect_build_jobs()
    has_ninja = ensure_ninja()
    env = build_env_vars(gpu_arch, jobs)
    console.print("[yellow]building[/yellow]")
    console.print(f"          [dim]from source (arch={gpu_arch}, jobs={jobs}, ninja={'yes' if has_ninja else 'no'})...[/dim]")

    start = time.time()
    built = build_wheel(pkg, env, torch_ver=torch_ver, cuda_ver=cuda_ver)
    if built:
        console.print(f"          [green]built[/green] {built.name} ({time.time() - start:.0f}s)")
        return _local_wheel_match(pkg, built, py_ver, plat, torch_ver, cuda_ver)

    console.print("          [red]build failed[/red]")
    return None


def _print_nearest(srcs: list[Source], cuda_ver: str, plat: str, client: httpx.Client) -> None:
    """Print nearest available torch versions as a hint."""
    combos = scan_available_combos(client, srcs, plat)
    nearby = [
        (c, tv) for c, tv in combos
        if c.startswith(cuda_ver.replace(".", "")[:2]) and tv.startswith("2.")
    ]
    if nearby:
        tvs = sorted({tv for _, tv in nearby}, key=_version_sort_key, reverse=True)
        console.print(f"          [dim]nearest: torch {', '.join(tvs[:6])}[/dim]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gpkg",
        description="GPU package manager — find prebuilt CUDA wheels, build missing ones.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            commands:
              gpkg add flash-attn causal-conv1d    resolve + lock + install
              gpkg install                         install from lockfile
              gpkg test flash-attn                 verify packages work

            examples:
              gpkg --torch 2.11 --cuda 130 flash-attn flash-attn-3
              gpkg flash-attn causal-conv1d        # auto-detect torch + cuda
              gpkg --available causal-conv1d natten
              gpkg --list
        """),
    )
    p.add_argument("command_or_packages", nargs="*", metavar="packages",
                   help="packages to resolve, or a command (add, install, test)")
    p.add_argument("--torch", default=None, help="PyTorch version (e.g. 2.11)")
    p.add_argument("--cuda", default=None, help="CUDA version (e.g. 130)")
    p.add_argument("--python", default=None, help="Python version")
    p.add_argument("--platform", default=None, help="platform")
    p.add_argument("--cxx11-abi", default="TRUE", choices=["TRUE", "FALSE"])
    p.add_argument("--project", default="my-project", help="name in pyproject.toml")
    p.add_argument("-o", "--output", default=None, help="write pyproject.toml to file")
    p.add_argument("--registry", default=None, help="path or URL to registry.toml")
    p.add_argument("--extra-registry", action="append", default=[], help="merge additional registries")
    p.add_argument("--list", action="store_true", help="list all sources")
    p.add_argument("--available", action="store_true", help="show cuda/torch combos")
    p.add_argument("--all", action="store_true", help="show all matching wheels")
    p.add_argument("--explain", action="store_true", help="detailed matching diagnostics")
    p.add_argument("--doctor", action="store_true", help="verify resolved wheel URLs")
    p.add_argument("--lock", action="store_true", help="write gpkg.lock.toml")
    p.add_argument("--update", action="store_true", help="show changes vs lockfile")
    p.add_argument("--lockfile", default=LOCKFILE_NAME, help=f"lockfile path (default: {LOCKFILE_NAME})")
    p.add_argument("--json", action="store_true", dest="as_json")
    p.add_argument("--no-cache", action="store_true", help="bypass disk cache")
    p.add_argument("--build-missing", action="store_true", help="compile from source when no wheel exists")
    p.add_argument("--sync", action="store_true", help="run uv sync after install")
    p.add_argument("--cache-info", action="store_true", help="show cache statistics")
    p.add_argument("--cache-clean", action="store_true", help="delete cached data")
    p.add_argument("--older-than", default=None, help="with --cache-clean: duration (5m, 1h, 2d)")
    p.add_argument("-V", "--version", action="version", version=f"%(prog)s {__import__('gpkg').__version__}")

    try:
        import argcomplete
        argcomplete.autocomplete(p)
    except ImportError:
        pass

    return p


def main() -> None:
    p = _build_parser()
    args = p.parse_args()
    _cache_mod._use_cache = not args.no_cache

    # Detect subcommand from first positional arg
    command = None
    positionals = args.command_or_packages or []
    if positionals and positionals[0] in ("add", "install", "test"):
        command = positionals[0]
        args.packages = positionals[1:]
    else:
        args.packages = positionals

    if command == "add":
        if not args.packages:
            p.error("gpkg add requires package names (e.g. gpkg add flash-attn)")
        args.build_missing = True

    # -- Commands that don't need network ----------------------------------

    if command == "test":
        cmd_test(args)
        return

    if args.cache_info:
        cmd_cache_info()
        return

    if args.cache_clean:
        cmd_cache_clean(args)
        return

    if args.older_than:
        p.error("--older-than requires --cache-clean")

    if command == "install":
        cmd_install(args)
        return

    # -- Commands that need network ----------------------------------------

    client = httpx.Client(headers=get_headers(), timeout=30, follow_redirects=True)

    all_sources: list[Source] = []
    registry = args.registry or os.environ.get("UVFORGE_REGISTRY")
    if registry:
        all_sources.extend(load_registry(registry, client))
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

    if args.list:
        cmd_list(args, all_sources, client)
        client.close()
        return

    if not args.packages:
        p.error("specify packages, or use --list")

    plat = args.platform or detect_platform()

    if args.available:
        cmd_available(args, pkg_sources, plat, client)
        client.close()
        return

    # -- Resolve packages --------------------------------------------------

    torch_ver = args.torch or detect_torch()
    cuda_ver = args.cuda or detect_cuda()
    if not torch_ver:
        p.error("--torch required (could not auto-detect; is PyTorch installed?)")
    if not cuda_ver:
        p.error("--cuda required (could not auto-detect; is nvcc or PyTorch+CUDA available?)")
    py_ver = args.python or detect_python()

    console.print("\n[bold]gpkg[/bold] -- searching prebuilt wheels")
    console.print(f"  torch={torch_ver}  cuda={cuda_ver}  python={py_ver}  platform={plat}  cxx11abi={args.cxx11_abi}\n")

    results, all_matches = _resolve_packages(args, pkg_sources, client, torch_ver, cuda_ver, py_ver, plat)

    # -- Show all matches --------------------------------------------------
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
        console.print("  [dim]•[/dim] Try [bold]--build-missing[/bold] to compile from source")
        console.print("  [dim]•[/dim] Set [bold]GITHUB_TOKEN[/bold] to avoid API rate limits\n")
        client.close()
        sys.exit(1)

    # -- Doctor ------------------------------------------------------------
    if args.doctor:
        console.print("\n[bold]Doctor:[/bold] verifying wheel URLs...")
        checks = doctor_check(results)
        for c in checks:
            if c.url_ok:
                size = f"  ({c.content_length / 1024 / 1024:.1f} MB)" if c.content_length else ""
                console.print(f"  [green]✓[/green] {c.package}: {c.filename}{size}")
            else:
                console.print(f"  [red]✗[/red] {c.package}: {c.filename}  ({c.error or f'HTTP {c.status_code}'})")

    # -- Lock / Update -----------------------------------------------------
    if args.update:
        old_lock = read_lockfile(args.lockfile)
        if old_lock is None:
            console.print(f"\n[yellow]No lockfile found at {args.lockfile}.[/yellow] Use --lock to create one.")
        else:
            changes = compare_lock(old_lock, results)
            if changes:
                console.print(f"\n[bold]Changes vs {args.lockfile}:[/bold]")
                for c in changes:
                    console.print(format_lock_change(c))
            else:
                console.print(f"\n[green]No changes vs {args.lockfile}.[/green]")

    if args.lock:
        write_lockfile(args.lockfile, torch_ver, cuda_ver, py_ver, plat, args.cxx11_abi, results)
        console.print(f"\n[bold green]Lockfile written:[/bold green] {args.lockfile}")

    # -- JSON output -------------------------------------------------------
    if args.as_json:
        print(json.dumps({
            n: {"url": unquote(m.url), "filename": m.filename, "version": m.version, "source": m.source_desc}
            for n, m in results.items()
        }, indent=2))
        client.close()
        return

    # -- TOML output -------------------------------------------------------
    if command == "add":
        pyproject_path = Path(args.output or "pyproject.toml")
        toml = merge_into_pyproject(pyproject_path, torch_ver, cuda_ver, results)
        pyproject_path.write_text(toml)
        console.print(f"\n[bold green]Updated {pyproject_path}[/bold green]")
        write_lockfile(args.lockfile, torch_ver, cuda_ver, py_ver, plat, args.cxx11_abi, results)
        console.print(f"[bold green]Lockfile written:[/bold green] {args.lockfile}")
        console.print("\n[bold]Running uv sync...[/bold]\n")
        sync_result = subprocess.run(["uv", "sync"], timeout=600)
        client.close()
        sys.exit(sync_result.returncode)

    toml = generate_toml(args.project, py_ver, torch_ver, cuda_ver, results)
    console.print("\n[bold green]pyproject.toml[/bold green]\n")
    if args.output:
        with open(args.output, "w") as f:
            f.write(toml)
        console.print(f"  Written to [bold]{args.output}[/bold]\n")
    else:
        print(toml)
    client.close()


if __name__ == "__main__":
    main()
