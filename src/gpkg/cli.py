"""gpkg CLI — GPU package manager. Find prebuilt CUDA wheels, build missing ones."""

from __future__ import annotations

import argparse
import difflib
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

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from gpkg import console
from gpkg.detect import detect_platform, detect_python, detect_cuda, detect_torch
from gpkg.output import generate_toml, merge_into_pyproject, _fmt_size, _fmt_age
from gpkg.lockfile import (
    LOCKFILE_NAME, write_lockfile, read_lockfile, compare_lock,
    format_lock_change, lockfile_to_wheel_matches,
)
from gpkg.registry import Source, load_registry, load_registry_with_fallback
import gpkg.cache as _cache_mod
from gpkg.cache import parse_duration, cache_info, cache_clean
from gpkg.matching import (
    WheelMatch, torch_minor,
    search_source, search_source_explain, scan_available_combos, pick_best,
)
from gpkg.doctor import doctor_check, check_source_health
from gpkg.build import (
    detect_gpu_arch, detect_build_jobs, build_env_vars, ensure_ninja,
    find_cached_wheel, build_wheel,
)

STACKS_URL = "https://wheels.mapika.dev/stacks.toml"

# Package name -> Python import name
IMPORT_MAP = {
    "flash-attn": "flash_attn",
    "flash-attn-3": "flash_attn_3",
    "causal-conv1d": "causal_conv1d",
    "mamba-ssm": "mamba_ssm",
    "grouped-gemm": "grouped_gemm",
    "sageattention": "sageattention",
    "natten": "natten",
    "torch-scatter": "torch_scatter",
    "torch-sparse": "torch_sparse",
    "torch-cluster": "torch_cluster",
    "torch-spline-conv": "torch_spline_conv",
    "pyg-lib": "pyg_lib",
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


def _load_stacks() -> list[dict]:
    """Fetch curated stacks from the registry."""
    try:
        resp = httpx.get(STACKS_URL, timeout=5)
        resp.raise_for_status()
        return tomllib.loads(resp.text).get("stacks", [])
    except Exception:
        console.print("[dim]Could not fetch stacks from registry[/dim]")
        return []


def _find_matching_stack(stacks: list[dict], name: str, torch_ver: str = "", cuda_ver: str = "",
                         py_ver: str = "", plat: str = "") -> dict | None:
    """Find the best matching stack entry for the user's environment."""
    candidates = [s for s in stacks if s["name"] == name]
    if not candidates:
        return None
    # Prefer verified over untested
    verified = [s for s in candidates if s.get("status") == "verified"]
    pool = verified or candidates
    # Try exact match on torch+cuda+python+platform
    for s in pool:
        if torch_ver and s.get("torch") != torch_ver:
            continue
        if cuda_ver and s.get("cuda") != cuda_ver:
            continue
        if py_ver and s.get("python") and s["python"] != py_ver:
            continue
        if plat and s.get("platform") and s["platform"] != plat:
            continue
        return s
    # Fall back to first verified, or first candidate
    return pool[0]


def cmd_stack(args: argparse.Namespace) -> None:
    """List or install curated, tested package stacks."""
    stacks = _load_stacks()
    sub = args.packages[0] if args.packages else "list"
    remaining = args.packages[1:] if len(args.packages) > 1 else []

    if sub == "list":
        # Group by name, show unique stacks
        seen = {}
        for s in stacks:
            name = s["name"]
            if name not in seen:
                seen[name] = s
            elif s.get("status") == "verified" and seen[name].get("status") != "verified":
                seen[name] = s

        t = Table(title="Curated GPU stacks")
        t.add_column("Stack", style="bold cyan")
        t.add_column("Packages")
        t.add_column("Torch")
        t.add_column("Status")
        t.add_column("Tested on", style="dim")
        for s in seen.values():
            pkgs = ", ".join(s.get("packages", {}).keys()) if isinstance(s.get("packages"), dict) else str(s.get("packages", ""))
            status = "[green]verified[/green]" if s.get("status") == "verified" else "[dim]untested[/dim]"
            tested = ", ".join(s.get("tested_on", []))
            t.add_row(s["name"], pkgs, s.get("torch", ""), status, tested)
        console.print(t)
        console.print("\n[dim]gpkg stack info <name>     details + versions[/dim]")
        console.print("[dim]gpkg stack install <name>  install verified combo[/dim]")
        return

    if sub == "info":
        name = remaining[0] if remaining else None
        if not name:
            console.print("[red]Usage: gpkg stack info <name>[/red]")
            sys.exit(1)
        entries = [s for s in stacks if s["name"] == name]
        if not entries:
            console.print(f"[red]Unknown stack: {name}[/red]")
            sys.exit(1)
        for s in entries:
            status = "[green]verified[/green]" if s.get("status") == "verified" else "[dim]untested[/dim]"
            console.print(f"\n[bold]{s['name']}[/bold] — {s['description']}  {status}")
            console.print(f"  [dim]Use case:[/dim] {s.get('use_case', '')}")
            console.print(f"  [dim]Env:[/dim] torch={s.get('torch', '?')} cuda={s.get('cuda', '?')} python={s.get('python', '?')} {s.get('platform', '')}")
            if s.get("tested_on"):
                console.print(f"  [dim]Tested on:[/dim] {', '.join(s['tested_on'])} ({s.get('tested_date', '?')})")
            pkgs = s.get("packages", {})
            if isinstance(pkgs, dict):
                for pkg, info in pkgs.items():
                    ver = info.get("version", "?") if isinstance(info, dict) else info
                    src = ""
                    if isinstance(info, dict) and info.get("url"):
                        src = " [dim](prebuilt)[/dim]"
                    elif isinstance(info, dict) and info.get("source") == "local-build":
                        src = " [dim](build from source)[/dim]"
                    console.print(f"    {pkg}=={ver}{src}")
        console.print(f"\n[dim]Install: gpkg stack install {name}[/dim]")
        return

    if sub == "install":
        name = remaining[0] if remaining else None
        if not name:
            console.print("[red]Usage: gpkg stack install <name>[/red]")
            console.print("[dim]Run gpkg stack list to see available stacks[/dim]")
            sys.exit(1)

        # Detect environment for matching
        torch_ver = args.torch or detect_torch() or ""
        cuda_ver = args.cuda or detect_cuda() or ""
        py_ver = args.python or detect_python()
        plat = args.platform or detect_platform()

        stack = _find_matching_stack(stacks, name, torch_ver, cuda_ver, py_ver, plat)
        if not stack:
            console.print(f"[red]Unknown stack: {name}[/red]")
            available = sorted(set(s["name"] for s in stacks))
            console.print(f"[dim]Available: {', '.join(available)}[/dim]")
            sys.exit(1)

        status = "[green]verified[/green]" if stack.get("status") == "verified" else "[yellow]untested[/yellow]"
        console.print(f"\n[bold]Stack:[/bold] {stack['name']}  {status}")
        console.print(f"[dim]{stack['description']}[/dim]")
        if stack.get("tested_on"):
            console.print(f"[dim]Tested on: {', '.join(stack['tested_on'])}[/dim]")
        console.print(f"[dim]Env: torch={stack.get('torch', '?')} cuda={stack.get('cuda', '?')} python={stack.get('python', '?')}[/dim]")

        pkgs = stack.get("packages", {})
        if not isinstance(pkgs, dict):
            console.print("[red]Stack has no pinned packages[/red]")
            sys.exit(1)

        # Install CUDA runtime if specified
        runtime_key = stack.get("runtime", "")
        # Find runtime in the raw TOML data
        try:
            resp = httpx.get(STACKS_URL, timeout=5)
            full_data = tomllib.loads(resp.text)
            cuda_runtimes = full_data.get("cuda_runtimes", {})
        except Exception:
            cuda_runtimes = {}

        if runtime_key and runtime_key in cuda_runtimes:
            rt = cuda_runtimes[runtime_key]
            console.print(f"\n[bold]CUDA runtime:[/bold] {runtime_key}")
            rt_pkgs = []
            for pkg, val in rt.items():
                if pkg == "torch":
                    torch_info = val if isinstance(val, dict) else {"version": val}
                    torch_ver = torch_info.get("version", "")
                    torch_idx = torch_info.get("index", "https://download.pytorch.org/whl/cu128")
                    console.print(f"  torch=={torch_ver}")
                    rt_pkgs.extend(["torch==" + torch_ver, "--index-url", torch_idx])
                else:
                    ver = val if isinstance(val, str) else val.get("version", "")
                    console.print(f"  [dim]{pkg}=={ver}[/dim]")
                    rt_pkgs.append(f"{pkg}=={ver}")
            console.print(f"\n[dim]Installing runtime ({len(rt_pkgs)} packages)...[/dim]")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", *rt_pkgs],
                timeout=600,
            )

        console.print("\n[bold]Packages:[/bold]")
        urls = []
        build_pkgs = []
        for pkg, info in pkgs.items():
            ver = info.get("version", "") if isinstance(info, dict) else info
            url = info.get("url") if isinstance(info, dict) else None
            source = info.get("source", "") if isinstance(info, dict) else ""
            if url:
                console.print(f"  {pkg}=={ver}  [green]prebuilt[/green]")
                urls.append(url)
            elif source == "local-build":
                console.print(f"  {pkg}=={ver}  [yellow]build from source[/yellow]")
                build_pkgs.append(f"{pkg}=={ver}")
            else:
                console.print(f"  {pkg}=={ver}  [cyan]resolve[/cyan]")
                build_pkgs.append(f"{pkg}=={ver}")

        # Install prebuilt wheels
        if urls:
            console.print(f"\n[dim]Installing {len(urls)} prebuilt wheels...[/dim]")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-deps", *urls],
                timeout=300,
            )

        # Resolve remaining via gpkg
        if build_pkgs:
            pkg_names = [p.split("==")[0] for p in build_pkgs]
            console.print(f"\n[dim]Resolving {len(build_pkgs)} packages via gpkg...[/dim]")
            args.packages = pkg_names
            args.build_missing = True
            args.torch = stack.get("torch") or args.torch
            args.cuda = stack.get("cuda") or args.cuda
            return  # fall through to resolve

        console.print("\n[bold green]Stack installed.[/bold green]")
        sys.exit(0)

    console.print(f"[red]Unknown stack command: {sub}[/red]")
    console.print("[dim]Usage: gpkg stack list | install <name> | info <name>[/dim]")
    sys.exit(1)


def cmd_export(args: argparse.Namespace) -> None:
    """Write a hash-pinned requirements.txt from the lockfile."""
    lock_data = read_lockfile(args.lockfile)
    if lock_data is None:
        console.print(f"[red]No lockfile found at {args.lockfile}.[/red] Run gpkg add or --lock first.")
        sys.exit(1)
    env, wheels = lockfile_to_wheel_matches(lock_data)
    if not wheels:
        console.print("[yellow]Lockfile has no wheels.[/yellow]")
        sys.exit(1)

    lines = [
        f"# generated by gpkg from {args.lockfile}",
        f"# env: torch={env.get('torch', '?')} cuda={env.get('cuda', '?')} "
        f"python={env.get('python', '?')} {env.get('platform', '')}",
        "# install with: pip install --no-deps --require-hashes -r requirements.txt",
        "# (torch itself comes from the pytorch index; install it first)",
        "",
    ]
    missing = []
    for name, w in wheels.items():
        if w.get("sha256"):
            lines.append(f"{name} @ {w['url']} \\")
            lines.append(f"    --hash=sha256:{w['sha256']}")
        else:
            lines.append(f"{name} @ {w['url']}")
            missing.append(name)
    text = "\n".join(lines) + "\n"

    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
        console.print(f"[bold green]Written to {args.output}[/bold green]")
    else:
        print(text)
    if missing:
        console.print(f"[yellow]No recorded hash for: {', '.join(missing)}[/yellow] "
                      "— --require-hashes will reject these lines")


def cmd_doctor_health(args: argparse.Namespace, sources: list[Source], client: httpx.Client) -> None:
    """Registry-wide source health check: every source must still serve wheels."""
    console.print("\n[bold]Doctor:[/bold] checking all registry sources...\n")
    checks = check_source_health(client, sources)
    broken = [c for c in checks if not c.healthy]
    for c in checks:
        if c.healthy:
            console.print(f"  [green]✓[/green] {c.package:<20s} {c.wheel_count:>4d} wheels  [dim]{c.location}[/dim]")
        else:
            console.print(f"  [red]✗[/red] {c.package:<20s}    0 wheels  [dim]{c.location}[/dim]")
            console.print(f"      [red]{c.detail}[/red]")
    console.print(f"\n  {len(checks) - len(broken)}/{len(checks)} sources healthy")
    sys.exit(1 if broken else 0)


def cmd_analyze(args: argparse.Namespace, client: httpx.Client) -> None:
    """Analyze which constraint pins are load-bearing vs relaxable."""
    from gpkg.analyzer import analyze_package, format_analysis

    py_ver = args.python or detect_python()
    exit_code = 0
    for pkg in args.packages:
        try:
            with console.status(f"Crawling dependency tree of {pkg} (cached after first run)..."):
                name, version, analyses = analyze_package(pkg, client, python_version=py_ver)
        except LookupError as e:
            console.print(f"  [red]✗[/red] {e}")
            exit_code = 1
            continue
        console.print(f"\n[bold]{name} {version}[/bold] — constraint analysis\n")
        if not analyses:
            console.print("  No restrictive constraints (pins or upper bounds) found.")
            continue
        for analysis in analyses:
            console.print(format_analysis(analysis))
            console.print()
    sys.exit(exit_code)


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
            close = difflib.get_close_matches(pkg, pkg_sources, n=1, cutoff=0.75)
            hint = f" (did you mean [bold]{close[0]}[/bold]?)" if close else ""
            console.print(f"  [yellow]![/yellow]  {pkg}: not in registry{hint}")
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
            if args.build_missing and not args.defer_builds:
                built_match = _try_build(pkg, py_ver, plat, torch_ver, cuda_ver, args.explain)
                if built_match:
                    results[pkg] = built_match
                continue
            if args.build_missing and args.defer_builds:
                console.print("[yellow]deferred[/yellow] (will build after uv sync)")
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


def _file_sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _local_wheel_match(pkg: str, whl: Path, py_ver: str, plat: str, torch_ver: str, cuda_ver: str) -> WheelMatch:
    """Create a WheelMatch from a locally-built or cached wheel."""
    pytag = f"cp{py_ver.replace('.', '')}"
    return WheelMatch(
        package=pkg, filename=whl.name, url=whl.as_uri(),
        version=whl.name.split("-")[1], torch_version=torch_minor(torch_ver),
        cuda_tag=cuda_ver, python_tag=pytag, platform_tag=plat,
        source_desc="local-build",
        sha256=_file_sha256(whl),
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


COMMANDS = (
    "add", "resolve", "install", "test", "compat", "analyze",
    "stack", "list", "available", "doctor", "cache", "export",
)

# Mode flags from the pre-0.5 flat CLI, mapped to their subcommand
_LEGACY_MODE_FLAGS = {
    "--list": ["list"],
    "--available": ["available"],
    "--cache-info": ["cache", "info"],
    "--cache-clean": ["cache", "clean"],
}

# Options that consume the next token, so the argv scanner can find the
# first true positional
_VALUE_FLAGS = {
    "--torch", "--cuda", "--python", "--platform", "--cxx11-abi",
    "--project", "-o", "--output", "--registry", "--extra-registry",
    "--lockfile", "--older-than",
}


def _rewrite_legacy_argv(argv: list[str]) -> list[str]:
    """Map pre-0.5 invocations onto the subcommand grammar.

    gpkg flash-attn          -> gpkg resolve flash-attn
    gpkg --torch 2.11 pkg    -> gpkg resolve --torch 2.11 pkg
    gpkg --list              -> gpkg list
    gpkg --available pkg     -> gpkg available pkg
    gpkg --cache-clean ...   -> gpkg cache clean ...
    """
    if not argv or all(a in ("-h", "--help", "-V", "--version") for a in argv):
        return argv

    for flag, cmd in _LEGACY_MODE_FLAGS.items():
        if flag in argv:
            return cmd + [a for a in argv if a != flag]

    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in _VALUE_FLAGS:
            i += 2
            continue
        if tok.startswith("-"):
            i += 1
            continue
        # First true positional
        if tok in COMMANDS:
            # Move the command to the front so flags typed before it still parse
            return argv if i == 0 else [tok] + argv[:i] + argv[i + 1:]
        close = difflib.get_close_matches(tok, COMMANDS, n=1, cutoff=0.75)
        if close:
            console.print(f"[dim]Hint: did you mean [bold]gpkg {close[0]}[/bold]? "
                          f"Treating '{tok}' as a package name.[/dim]")
        return ["resolve"] + argv

    # Flags but no positional: assume resolve so the error mentions packages
    return ["resolve"] + argv


def _add_env_opts(sp: argparse.ArgumentParser) -> None:
    g = sp.add_argument_group("environment (auto-detected if omitted)")
    g.add_argument("--torch", default=None, help="PyTorch version (e.g. 2.11)")
    g.add_argument("--cuda", default=None, help="CUDA version (e.g. 130)")
    g.add_argument("--python", default=None, help="Python version (e.g. 3.12)")
    g.add_argument("--platform", default=None, help="platform tag (e.g. linux_x86_64)")
    g.add_argument("--cxx11-abi", default="TRUE", choices=["TRUE", "FALSE"],
                   help="filter wheels by C++11 ABI tag")


def _add_registry_opts(sp: argparse.ArgumentParser) -> None:
    g = sp.add_argument_group("registry")
    g.add_argument("--registry", default=None, help="path or URL to registry.toml")
    g.add_argument("--extra-registry", action="append", default=[],
                   help="merge additional registries")
    g.add_argument("--no-cache", action="store_true", help="bypass disk cache")


def _make_resolve_like(sub, name: str, help_text: str, epilog: str,
                       packages_required: bool = True) -> argparse.ArgumentParser:
    sp = sub.add_parser(name, help=help_text, description=help_text,
                        formatter_class=argparse.RawDescriptionHelpFormatter,
                        epilog=epilog)
    sp.add_argument("packages", nargs="+" if packages_required else "*",
                    help="packages to resolve")
    _add_env_opts(sp)
    _add_registry_opts(sp)
    sp.add_argument("-o", "--output", default=None, help="write pyproject.toml to file")
    sp.add_argument("--project", default="my-project", help="name in generated pyproject.toml")
    sp.add_argument("--json", action="store_true", dest="as_json",
                    help="machine-readable JSON output")
    sp.add_argument("--lock", action="store_true", help="write gpkg.lock.toml")
    sp.add_argument("--update", action="store_true", help="show changes vs lockfile")
    sp.add_argument("--lockfile", default=LOCKFILE_NAME,
                    help=f"lockfile path (default: {LOCKFILE_NAME})")
    sp.add_argument("--explain", action="store_true", help="detailed matching diagnostics")
    sp.add_argument("--all", action="store_true", help="show all matching wheels")
    sp.add_argument("--build-missing", action="store_true",
                    help="compile from source when no wheel exists")
    sp.add_argument("--no-resolve", action="store_true",
                    help="skip compatibility resolution, pick latest per package")
    return sp


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gpkg",
        description="GPU package manager — find prebuilt CUDA wheels, build missing ones.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              gpkg add flash-attn causal-conv1d mamba-ssm   resolve + lock + install
              gpkg flash-attn causal-conv1d                 resolve only (same as gpkg resolve)
              gpkg analyze boltz                            which pins are relaxable
              gpkg stack install mamba                      install a tested combo

            run 'gpkg <command> --help' for command-specific options.
        """),
    )
    p.add_argument("-V", "--version", action="version",
                   version=f"%(prog)s {__import__('gpkg').__version__}")

    sub = p.add_subparsers(dest="command", metavar="<command>")

    sp = sub.add_parser("add", help="resolve wheels, lock, and install into the current project",
                        description="Resolve wheels, write pyproject.toml + lockfile, run uv sync. "
                                    "Builds from source when no wheel exists.",
                        formatter_class=argparse.RawDescriptionHelpFormatter,
                        epilog="example:\n  gpkg add flash-attn causal-conv1d mamba-ssm")
    sp.add_argument("packages", nargs="+", help="packages to add")
    _add_env_opts(sp)
    _add_registry_opts(sp)
    sp.add_argument("-o", "--output", default=None,
                    help="pyproject.toml path (default: ./pyproject.toml)")
    sp.add_argument("--lockfile", default=LOCKFILE_NAME,
                    help=f"lockfile path (default: {LOCKFILE_NAME})")
    sp.add_argument("--explain", action="store_true", help="detailed matching diagnostics")
    sp.add_argument("--all", action="store_true", help="show all matching wheels")
    sp.add_argument("--no-resolve", action="store_true",
                    help="skip compatibility resolution, pick latest per package")
    sp.set_defaults(build_missing=True, defer_builds=True)

    _make_resolve_like(
        sub, "resolve", "find wheels and emit pyproject.toml (the default command)",
        "examples:\n"
        "  gpkg resolve flash-attn causal-conv1d      # auto-detect torch + cuda\n"
        "  gpkg resolve --torch 2.11 --cuda 130 flash-attn -o pyproject.toml\n"
        "  gpkg flash-attn                            # 'resolve' is implied",
    ).add_argument("--doctor", action="store_true", help="verify resolved wheel URLs")

    sp = _make_resolve_like(
        sub, "doctor", "verify wheel URLs — or, with no packages, check every registry source",
        "examples:\n"
        "  gpkg doctor                                  # health-check all registry sources\n"
        "  gpkg doctor --torch 2.11 --cuda 130 flash-attn\n"
        "  gpkg doctor --verify --torch 2.11 --cuda 130 flash-attn   # download + check sha256",
        packages_required=False,
    )
    sp.add_argument("--verify", action="store_true",
                    help="download wheels and verify sha256 against source-attested hashes")
    sp.set_defaults(doctor=True)

    sp = sub.add_parser("install", help="install from lockfile (no network)",
                        description="Generate pyproject.toml from gpkg.lock.toml without network.",
                        formatter_class=argparse.RawDescriptionHelpFormatter,
                        epilog="example:\n  gpkg install --sync")
    sp.add_argument("--lockfile", default=LOCKFILE_NAME,
                    help=f"lockfile path (default: {LOCKFILE_NAME})")
    sp.add_argument("-o", "--output", default=None, help="write pyproject.toml to file")
    sp.add_argument("--project", default="my-project", help="name in generated pyproject.toml")
    sp.add_argument("--sync", action="store_true", help="run uv sync after generating")

    sp = sub.add_parser("export", help="export lockfile as hash-pinned requirements.txt",
                        description="Write a requirements.txt with --hash=sha256: lines from "
                                    "gpkg.lock.toml, for pip/uv --require-hashes installs.",
                        formatter_class=argparse.RawDescriptionHelpFormatter,
                        epilog="example:\n  gpkg export -o requirements.txt")
    sp.add_argument("--lockfile", default=LOCKFILE_NAME,
                    help=f"lockfile path (default: {LOCKFILE_NAME})")
    sp.add_argument("-o", "--output", default=None, help="write to file (default: stdout)")

    sp = sub.add_parser("test", help="verify installed packages import and see the GPU",
                        formatter_class=argparse.RawDescriptionHelpFormatter,
                        epilog="example:\n  gpkg test flash-attn causal-conv1d")
    sp.add_argument("packages", nargs="*", help="packages to test (default: all known)")

    sp = sub.add_parser("compat", help="dry-run: check a package set is mutually compatible",
                        description="Find a mutually compatible version set without installing.",
                        formatter_class=argparse.RawDescriptionHelpFormatter,
                        epilog="example:\n  gpkg compat flash-attn causal-conv1d mamba-ssm")
    sp.add_argument("packages", nargs="+", help="packages to check")
    _add_env_opts(sp)
    _add_registry_opts(sp)
    sp.add_argument("--explain", action="store_true", help="detailed matching diagnostics")

    sp = sub.add_parser("analyze", help="show which constraint pins are load-bearing vs relaxable",
                        description="Crawl a package's dependency tree and report which of its "
                                    "version pins are load-bearing and which can be safely relaxed.",
                        formatter_class=argparse.RawDescriptionHelpFormatter,
                        epilog="examples:\n"
                               "  gpkg analyze boltz\n"
                               "  gpkg analyze boltz==2.2.1\n"
                               "  gpkg analyze          # dependencies from ./pyproject.toml")
    sp.add_argument("packages", nargs="*",
                    help="PyPI packages (default: dependencies from ./pyproject.toml)")
    _add_env_opts(sp)
    _add_registry_opts(sp)

    sp = sub.add_parser("stack", help="curated, tested torch+cuda+package combos",
                        formatter_class=argparse.RawDescriptionHelpFormatter,
                        epilog="examples:\n"
                               "  gpkg stack list\n"
                               "  gpkg stack info mamba\n"
                               "  gpkg stack install mamba")
    sp.add_argument("packages", nargs="*", metavar="action",
                    help="list | info <name> | install <name>")
    _add_env_opts(sp)
    _add_registry_opts(sp)

    sp = sub.add_parser("list", help="list registered wheel sources")
    _add_registry_opts(sp)
    sp.add_argument("--json", action="store_true", dest="as_json",
                    help="machine-readable JSON output")

    sp = sub.add_parser("available", help="show published cuda/torch combos per package",
                        formatter_class=argparse.RawDescriptionHelpFormatter,
                        epilog="example:\n  gpkg available causal-conv1d natten")
    sp.add_argument("packages", nargs="+", help="packages to look up")
    _add_env_opts(sp)
    _add_registry_opts(sp)
    sp.add_argument("--json", action="store_true", dest="as_json",
                    help="machine-readable JSON output")

    sp = sub.add_parser("cache", help="show or clean the disk cache",
                        formatter_class=argparse.RawDescriptionHelpFormatter,
                        epilog="examples:\n"
                               "  gpkg cache info\n"
                               "  gpkg cache clean --older-than 1h")
    sp.add_argument("cache_action", choices=["info", "clean"], metavar="action",
                    help="info | clean")
    sp.add_argument("--older-than", default=None,
                    help="with clean: only entries older than this (5m, 1h, 2d)")

    try:
        import argcomplete
        argcomplete.autocomplete(p)
    except ImportError:
        pass

    return p


# Attributes the shared code paths read; subparsers that don't define them
# get these defaults after parsing.
_ARG_DEFAULTS: dict = {
    "packages": [], "torch": None, "cuda": None, "python": None, "platform": None,
    "cxx11_abi": "TRUE", "project": "my-project", "output": None,
    "registry": None, "extra_registry": [], "as_json": False, "no_cache": False,
    "build_missing": False, "defer_builds": False, "sync": False,
    "no_resolve": False, "explain": False, "all": False, "doctor": False,
    "verify": False,
    "lock": False, "update": False, "lockfile": LOCKFILE_NAME, "older_than": None,
}


def _ensure_defaults(args: argparse.Namespace) -> None:
    for k, v in _ARG_DEFAULTS.items():
        if not hasattr(args, k):
            setattr(args, k, list(v) if isinstance(v, list) else v)


def main() -> None:
    p = _build_parser()
    args = p.parse_args(_rewrite_legacy_argv(sys.argv[1:]))
    _ensure_defaults(args)
    _cache_mod._use_cache = not args.no_cache
    command = args.command

    if command is None:
        p.print_help()
        return

    if command == "analyze" and not args.packages:
        pyproject_path = Path("pyproject.toml")
        if pyproject_path.exists():
            with open(pyproject_path, "rb") as f:
                pyproject = tomllib.load(f)
            deps = pyproject.get("project", {}).get("dependencies", [])
            if deps:
                import re as _re
                args.packages = []
                for dep in deps:
                    name = _re.split(r"[>=<!\s\[]", dep)[0].strip()
                    if name:
                        args.packages.append(name)
        if not args.packages:
            p.error("gpkg analyze requires package names or a pyproject.toml with dependencies")

    # -- Commands that don't need network ----------------------------------

    if command == "test":
        cmd_test(args)
        return

    if command == "stack":
        old_packages = list(args.packages)
        cmd_stack(args)
        if args.packages == old_packages or not args.packages:
            # stack list/info — done, don't resolve
            return
        # stack install sets args.packages and falls through to resolve
        command = "add"

    if command == "cache":
        if args.cache_action == "info":
            cmd_cache_info()
        else:
            cmd_cache_clean(args)
        return

    if command == "install":
        cmd_install(args)
        return

    if command == "export":
        cmd_export(args)
        return

    # -- Commands that need network ----------------------------------------

    # No client-wide auth: GitHub tokens are scoped to API requests inside
    # fetch_releases — S3-style hosts reject foreign Authorization headers.
    client = httpx.Client(timeout=30, follow_redirects=True)

    if command == "analyze":
        cmd_analyze(args, client)
        return

    all_sources: list[Source] = []
    registry = args.registry or os.environ.get("GPKG_REGISTRY") or os.environ.get("UVFORGE_REGISTRY")
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

    if command == "list":
        cmd_list(args, all_sources, client)
        client.close()
        return

    if command == "doctor" and not args.packages:
        cmd_doctor_health(args, all_sources, client)
        return

    plat = args.platform or detect_platform()

    if command == "available":
        cmd_available(args, pkg_sources, plat, client)
        client.close()
        return

    # -- Resolve packages --------------------------------------------------

    torch_ver = args.torch or detect_torch()
    cuda_ver = args.cuda or detect_cuda()
    py_ver = args.python or detect_python()

    # -- Detect torch pins from packages being added ----------------------
    # Must run BEFORE the torch_ver check so we can set it from the pin
    torch_adjustments: list[tuple[str, str]] = []
    if command in ("add", "compat") and not args.torch:
        try:
            from gpkg.analyzer import parse_pypi_requires_dist
            import re as _re
            for pkg in args.packages:
                if pkg in pkg_sources:
                    continue  # GPU package in registry, skip
                # Check PyPI for torch pins
                data = None
                try:
                    resp = client.get(f"https://pypi.org/pypi/{pkg}/json", timeout=10)
                    if resp.status_code == 200:
                        data = resp.json()
                except Exception:
                    pass
                if data is None:
                    continue
                requires = parse_pypi_requires_dist(data, py_ver or "3.12")
                for req in requires:
                    m = _re.match(r"torch\s*==\s*([0-9.]+)", req.strip())
                    if m:
                        pinned = m.group(1)
                        cur = torch_ver or ""
                        if pinned != cur and pinned != cur.split("+")[0]:
                            torch_adjustments.append((pkg, pinned))
                        break
        except Exception:
            pass

    if torch_adjustments:
        pkg_name, pinned_torch = torch_adjustments[0]
        if torch_ver:
            console.print(f"\n[bold yellow]⚠[/bold yellow]  {pkg_name} requires [bold]torch=={pinned_torch}[/bold] (you have {torch_ver})")
            console.print(f"  Adjusting torch version to {pinned_torch} for compatibility\n")
        else:
            console.print(f"\n[bold]ℹ[/bold]  {pkg_name} requires [bold]torch=={pinned_torch}[/bold] — using that version\n")
        torch_ver = pinned_torch

    if not torch_ver:
        p.error("--torch required (could not auto-detect; is PyTorch installed?)")
    if not cuda_ver:
        p.error("--cuda required (could not auto-detect; is nvcc or PyTorch+CUDA available?)")

    console.print("\n[bold]gpkg[/bold] -- searching prebuilt wheels")
    console.print(f"  torch={torch_ver}  cuda={cuda_ver}  python={py_ver}  platform={plat}  cxx11abi={args.cxx11_abi}\n")

    results, all_matches = _resolve_packages(args, pkg_sources, client, torch_ver, cuda_ver, py_ver, plat)

    # -- Compatibility resolution ------------------------------------------
    if command == "compat" or (command == "add" and len(args.packages) > 1 and not args.no_resolve):
        from gpkg.resolver import resolve, format_resolve_result, present_options
        from gpkg.matching import search_all_versions

        versioned = search_all_versions(all_matches)
        env = {"torch": torch_ver, "cuda": cuda_ver, "python": py_ver, "platform": plat}

        resolve_result = resolve(
            all_versions=versioned,
            env=env,
            sources=all_sources,
            client=client,
        )

        if resolve_result is None and command == "compat":
            missing = [pkg for pkg in args.packages if not all_matches.get(pkg)]
            if missing:
                console.print(f"\n  [red]✗[/red] No compatible set: no wheels found for "
                              f"[bold]{', '.join(missing)}[/bold] "
                              f"(torch {torch_ver}, cuda {cuda_ver})")
                console.print("  [dim]Try --available to see supported torch/cuda combos[/dim]")
            else:
                console.print("\n  [red]✗[/red] No compatible set found")
            client.close()
            sys.exit(1)

        if resolve_result is not None:
            if hasattr(resolve_result, 'analyses') and resolve_result.analyses:
                from gpkg.analyzer import format_analysis
                console.print()
                for analysis in resolve_result.analyses:
                    console.print(format_analysis(analysis))
                console.print()

            if command == "compat":
                # Dry-run: just show results
                console.print(format_resolve_result(resolve_result))
                client.close()
                return

            if resolve_result.chosen.conflicts and resolve_result.alternatives:
                chosen = present_options(resolve_result)
                if chosen is None:
                    console.print("\n[yellow]Aborted.[/yellow]")
                    client.close()
                    return
                # Update results with chosen versions
                results = {m.package: m for m in chosen.matches}
            elif not resolve_result.chosen.conflicts:
                if not resolve_result.from_cache:
                    console.print(format_resolve_result(resolve_result))
                else:
                    console.print("\n  [green]✓[/green] Known compatible set")
                results = {m.package: m for m in resolve_result.chosen.matches}

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
        verb = "downloading + verifying hashes" if args.verify else "verifying wheel URLs"
        console.print(f"\n[bold]Doctor:[/bold] {verb}...")
        checks = doctor_check(results, verify=args.verify)
        for c in checks:
            if c.url_ok:
                size = f"  ({c.content_length / 1024 / 1024:.1f} MB)" if c.content_length else ""
                if c.sha256_ok is True:
                    hash_note = "  [green]sha256 verified[/green]"
                elif c.sha256_ok is False:
                    hash_note = f"  [red]{c.error}[/red]"
                elif c.sha256_expected:
                    hash_note = "  [dim]sha256 recorded[/dim]"
                else:
                    hash_note = "  [dim]no source hash[/dim]" if args.verify else ""
                console.print(f"  [green]✓[/green] {c.package}: {c.filename}{size}{hash_note}")
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
        toml = merge_into_pyproject(pyproject_path, torch_ver, cuda_ver, results, py_ver)
        pyproject_path.write_text(toml)
        console.print(f"\n[bold green]Updated {pyproject_path}[/bold green]")
        write_lockfile(args.lockfile, torch_ver, cuda_ver, py_ver, plat, args.cxx11_abi, results)
        console.print(f"[bold green]Lockfile written:[/bold green] {args.lockfile}")

        # Find packages that need building (requested but not resolved)
        needs_build = [pkg for pkg in args.packages if pkg not in results and pkg in pkg_sources]

        # First sync: install torch + prebuilt wheels
        console.print("\n[bold]Running uv sync...[/bold]\n")
        sync_result = subprocess.run(["uv", "sync"], timeout=600)
        if sync_result.returncode != 0:
            client.close()
            sys.exit(sync_result.returncode)

        # Deferred builds: now that torch is installed, build from source
        if needs_build:
            console.print(f"\n[bold]Building {len(needs_build)} package(s) from source...[/bold]\n")
            for pkg in needs_build:
                console.print(f"  {pkg}", end="  ")
                built = _try_build(pkg, py_ver, plat, torch_ver, cuda_ver, args.explain)
                if built:
                    results[pkg] = built

            if any(pkg in results for pkg in needs_build):
                # Update pyproject + lockfile with newly built wheels
                toml = merge_into_pyproject(pyproject_path, torch_ver, cuda_ver, results, py_ver)
                pyproject_path.write_text(toml)
                write_lockfile(args.lockfile, torch_ver, cuda_ver, py_ver, plat, args.cxx11_abi, results)
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
