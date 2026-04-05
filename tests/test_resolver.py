"""Tests for the compatibility resolver."""

from gpkg.resolver import (
    Conflict,
    Combo,
    CompatSet,
    ResolveResult,
    PackageMetadata,
    check_conflicts,
    score_combo,
)


def test_no_conflict_disjoint_deps():
    """Packages with no shared deps produce no conflicts."""
    meta = [
        PackageMetadata("flash-attn", "2.8.3", ["torch>=2.0", "einops"]),
        PackageMetadata("mamba-ssm", "2.2.4", ["torch>=2.0", "causal-conv1d"]),
    ]
    assert check_conflicts(meta) == []


def test_no_conflict_compatible_ranges():
    """Shared dep with overlapping ranges is fine."""
    meta = [
        PackageMetadata("pkg-a", "1.0", ["numpy>=1.26,<3.0"]),
        PackageMetadata("pkg-b", "2.0", ["numpy>=2.0"]),
    ]
    assert check_conflicts(meta) == []


def test_conflict_incompatible_numpy():
    """Classic numpy<2 vs numpy>=2 conflict."""
    meta = [
        PackageMetadata("flash-attn", "2.8.3", ["numpy>=2.0"]),
        PackageMetadata("natten", "0.17.4", ["numpy<2.0"]),
    ]
    conflicts = check_conflicts(meta)
    assert len(conflicts) == 1
    assert conflicts[0].dependency == "numpy"
    assert "flash-attn" in conflicts[0].specifiers
    assert "natten" in conflicts[0].specifiers


def test_conflict_multiple_deps():
    """Multiple shared deps, one conflicts."""
    meta = [
        PackageMetadata("pkg-a", "1.0", ["numpy>=2.0", "scipy>=1.13"]),
        PackageMetadata("pkg-b", "2.0", ["numpy<1.26", "scipy>=1.10"]),
    ]
    conflicts = check_conflicts(meta)
    assert len(conflicts) == 1
    assert conflicts[0].dependency == "numpy"


def test_conflict_three_packages():
    """Three packages, conflict only visible when all three combined."""
    meta = [
        PackageMetadata("a", "1.0", ["numpy>=1.20,<2.0"]),
        PackageMetadata("b", "1.0", ["numpy>=1.26"]),
        PackageMetadata("c", "1.0", ["numpy>=2.0"]),
    ]
    conflicts = check_conflicts(meta)
    assert len(conflicts) == 1
    assert conflicts[0].dependency == "numpy"


def test_deps_without_version_spec_ignored():
    """Deps like 'einops' (no version) don't cause conflicts."""
    meta = [
        PackageMetadata("a", "1.0", ["einops", "torch>=2.0"]),
        PackageMetadata("b", "1.0", ["einops", "torch>=2.4"]),
    ]
    assert check_conflicts(meta) == []


def test_score_no_conflicts_higher():
    """Zero-conflict combo scores higher than conflicted."""
    from gpkg.matching import WheelMatch

    m1 = WheelMatch(
        package="a", filename="a.whl", url="", version="2.0.0",
        torch_version="2.11", cuda_tag="128", python_tag="cp312-cp312",
        platform_tag="linux_x86_64", source_desc="test", cxx11_abi=None,
        release_tag="",
    )
    m2 = WheelMatch(
        package="b", filename="b.whl", url="", version="1.0.0",
        torch_version="2.11", cuda_tag="128", python_tag="cp312-cp312",
        platform_tag="linux_x86_64", source_desc="test", cxx11_abi=None,
        release_tag="",
    )

    conflict = Conflict(
        dependency="numpy",
        specifiers={"a": ">=2.0", "b": "<2.0"},
        merged_range="empty",
    )

    clean_score = score_combo([m1, m2], [])
    dirty_score = score_combo([m1, m2], [conflict])
    assert clean_score > dirty_score


def test_score_newer_versions_higher():
    """Among conflict-free combos, newer versions score higher."""
    from gpkg.matching import WheelMatch

    def make(pkg, ver):
        return WheelMatch(
            package=pkg, filename=f"{pkg}.whl", url="", version=ver,
            torch_version="2.11", cuda_tag="128", python_tag="cp312-cp312",
            platform_tag="linux_x86_64", source_desc="test", cxx11_abi=None,
            release_tag="",
        )

    newer = score_combo([make("a", "3.0.0"), make("b", "2.0.0")], [])
    older = score_combo([make("a", "1.0.0"), make("b", "1.0.0")], [])
    assert newer > older


def test_registry_requires_field():
    """Source dataclass accepts requires blocks."""
    from gpkg.registry import Source, RequiresBlock

    rb = RequiresBlock(
        versions=["2.8.3", "2.8.2"],
        requires_dist=["torch>=2.0", "einops", "packaging", "ninja"],
    )
    s = Source(
        package="flash-attn",
        description="test",
        source_type="github",
        requires=[rb],
    )
    assert len(s.requires) == 1
    assert "2.8.3" in s.requires[0].versions
    assert "torch>=2.0" in s.requires[0].requires_dist


def test_registry_requires_lookup():
    """Can look up requires_dist for a specific version."""
    from gpkg.registry import Source, RequiresBlock, get_requires_for_version

    rb = RequiresBlock(
        versions=["2.8.3", "2.8.2"],
        requires_dist=["torch>=2.0", "einops"],
    )
    s = Source(package="flash-attn", description="t", source_type="github", requires=[rb])
    assert get_requires_for_version(s, "2.8.3") == ["torch>=2.0", "einops"]
    assert get_requires_for_version(s, "2.8.3.post1") == ["torch>=2.0", "einops"]
    assert get_requires_for_version(s, "2.7.0") is None


def test_search_all_versions_returns_multiple():
    """search_all_versions returns all matches, not just best."""
    from gpkg.matching import WheelMatch, search_all_versions

    matches_by_pkg = search_all_versions(
        all_matches={
            "flash-attn": [
                WheelMatch("flash-attn", "a.whl", "", "2.8.3", "2.11", "128",
                           "cp312-cp312", "linux_x86_64", "test", None, ""),
                WheelMatch("flash-attn", "b.whl", "", "2.8.2", "2.11", "128",
                           "cp312-cp312", "linux_x86_64", "test", None, ""),
                WheelMatch("flash-attn", "c.whl", "", "2.7.0", "2.11", "128",
                           "cp312-cp312", "linux_x86_64", "test", None, ""),
            ],
        },
        max_per_package=2,
    )
    assert len(matches_by_pkg["flash-attn"]) == 2
    assert matches_by_pkg["flash-attn"][0].version == "2.8.3"
    assert matches_by_pkg["flash-attn"][1].version == "2.8.2"


def test_compat_cache_key_deterministic():
    """Same packages + env always produce the same cache key."""
    from gpkg.resolver import compat_cache_key

    key1 = compat_cache_key(
        ["flash-attn", "mamba-ssm"], "2.11", "128", "3.12", "linux_x86_64"
    )
    key2 = compat_cache_key(
        ["mamba-ssm", "flash-attn"], "2.11", "128", "3.12", "linux_x86_64"
    )
    assert key1 == key2  # order-independent


def test_compat_cache_key_differs_by_env():
    """Different environments produce different keys."""
    from gpkg.resolver import compat_cache_key

    key1 = compat_cache_key(["flash-attn"], "2.11", "128", "3.12", "linux_x86_64")
    key2 = compat_cache_key(["flash-attn"], "2.10", "128", "3.12", "linux_x86_64")
    assert key1 != key2


def test_compat_cache_roundtrip(tmp_path):
    """Write and read back a compat cache entry."""
    from gpkg.resolver import write_compat_cache, read_compat_cache, CompatSet

    compat = CompatSet(
        packages=[
            {"package": "flash-attn", "version": "2.8.3", "url": "https://example.com/a.whl"},
        ],
        constraints={"numpy": ">=2.0,<3.0"},
        status="user-resolved",
        resolved_at="2026-04-04T15:30:00Z",
    )
    env = {"torch": "2.11", "cuda": "128", "python": "3.12", "platform": "linux_x86_64"}

    write_compat_cache(["flash-attn"], env, compat, cache_dir=tmp_path)
    loaded = read_compat_cache(["flash-attn"], env, cache_dir=tmp_path)

    assert loaded is not None
    assert loaded.status == "user-resolved"
    assert loaded.packages[0]["package"] == "flash-attn"
    assert loaded.constraints["numpy"] == ">=2.0,<3.0"


def test_compat_cache_miss(tmp_path):
    """Cache miss returns None."""
    from gpkg.resolver import read_compat_cache

    env = {"torch": "2.11", "cuda": "128", "python": "3.12", "platform": "linux_x86_64"}
    assert read_compat_cache(["nonexistent"], env, cache_dir=tmp_path) is None


def test_fetch_metadata_registry_override():
    """Registry requires block is used when available."""
    from gpkg.resolver import fetch_metadata
    from gpkg.registry import Source, RequiresBlock
    from gpkg.matching import WheelMatch

    source = Source(
        package="flash-attn", description="test", source_type="github",
        requires=[RequiresBlock(
            versions=["2.8.3"],
            requires_dist=["torch>=2.0", "einops"],
        )],
    )
    match = WheelMatch(
        "flash-attn", "a.whl", "https://example.com/a.whl", "2.8.3",
        "2.11", "128", "cp312-cp312", "linux_x86_64", "test", None, "",
    )

    result = fetch_metadata(match, sources=[source], client=None)
    assert result == ["torch>=2.0", "einops"]


def test_fetch_metadata_no_registry_returns_none_without_client():
    """Without registry data and no client, returns None."""
    from gpkg.resolver import fetch_metadata
    from gpkg.registry import Source
    from gpkg.matching import WheelMatch

    source = Source(package="flash-attn", description="test", source_type="github")
    match = WheelMatch(
        "flash-attn", "a.whl", "https://example.com/a.whl", "2.8.3",
        "2.11", "128", "cp312-cp312", "linux_x86_64", "test", None, "",
    )

    result = fetch_metadata(match, sources=[source], client=None)
    assert result is None


def test_parse_wheel_metadata():
    """Parse Requires-Dist from METADATA content."""
    from gpkg.resolver import parse_requires_dist

    metadata = """Metadata-Version: 2.1
Name: flash-attn
Version: 2.8.3
Requires-Dist: torch>=2.0
Requires-Dist: einops
Requires-Dist: packaging
Requires-Dist: ninja ; extra == "build"
"""
    result = parse_requires_dist(metadata)
    assert "torch>=2.0" in result
    assert "einops" in result
    assert "packaging" in result


def test_generate_candidates_basic():
    """Generate combos from 2 packages x 2 versions each."""
    from gpkg.resolver import generate_candidates, Combo
    from gpkg.matching import WheelMatch

    def make(pkg, ver):
        return WheelMatch(pkg, f"{pkg}.whl", "", ver, "2.11", "128",
                          "cp312-cp312", "linux_x86_64", "test", None, "")

    all_versions = {
        "a": [make("a", "2.0.0"), make("a", "1.0.0")],
        "b": [make("b", "3.0.0"), make("b", "2.0.0")],
    }
    combos = generate_candidates(all_versions)
    assert len(combos) == 4  # 2x2
    # First combo should be newest of each (highest score)
    versions = {m.package: m.version for m in combos[0].matches}
    assert versions["a"] == "2.0.0"
    assert versions["b"] == "3.0.0"


def test_generate_candidates_single_package():
    """Single package produces one combo per version."""
    from gpkg.resolver import generate_candidates
    from gpkg.matching import WheelMatch

    def make(ver):
        return WheelMatch("a", "a.whl", "", ver, "2.11", "128",
                          "cp312-cp312", "linux_x86_64", "test", None, "")

    combos = generate_candidates({"a": [make("2.0.0"), make("1.0.0")]})
    assert len(combos) == 2


def test_generate_candidates_gpu_pruning():
    """Combos with mismatched torch versions are pruned."""
    from gpkg.resolver import generate_candidates
    from gpkg.matching import WheelMatch

    all_versions = {
        "a": [WheelMatch("a", "a.whl", "", "2.0.0", "2.11", "128",
                          "cp312-cp312", "linux_x86_64", "test", None, "")],
        "b": [WheelMatch("b", "b.whl", "", "1.0.0", "2.10", "128",
                          "cp312-cp312", "linux_x86_64", "test", None, "")],
    }
    combos = generate_candidates(all_versions)
    assert len(combos) == 0  # torch 2.11 vs 2.10 mismatch


def test_trial_resolve_builds_requirements():
    """trial_resolve builds correct requirements list for uv."""
    from gpkg.resolver import _build_trial_requirements, Combo
    from gpkg.matching import WheelMatch

    combo = Combo(
        matches=[
            WheelMatch("flash-attn", "a.whl", "https://example.com/a.whl",
                       "2.8.3", "2.11", "128", "cp312-cp312",
                       "linux_x86_64", "test", None, ""),
            WheelMatch("mamba-ssm", "b.whl", "https://example.com/b.whl",
                       "2.2.4", "2.11", "128", "cp312-cp312",
                       "linux_x86_64", "test", None, ""),
        ],
        conflicts=[],
        score=0.0,
    )
    reqs = _build_trial_requirements(combo)
    assert "flash-attn==2.8.3" in reqs
    assert "mamba-ssm==2.2.4" in reqs


def test_resolve_single_package_no_conflicts(tmp_path):
    """Single package resolves trivially."""
    from gpkg.resolver import resolve, ResolveResult
    from gpkg.matching import WheelMatch

    match = WheelMatch(
        "flash-attn", "a.whl", "https://example.com/a.whl", "2.8.3",
        "2.11", "128", "cp312-cp312", "linux_x86_64", "test", None, "",
    )
    all_versions = {"flash-attn": [match]}
    env = {"torch": "2.11", "cuda": "128", "python": "3.12", "platform": "linux_x86_64"}

    result = resolve(
        all_versions=all_versions,
        env=env,
        sources=[],
        client=None,
        skip_trial=True,
        cache_dir=tmp_path,
    )

    assert result is not None
    assert len(result.chosen.matches) == 1
    assert result.chosen.matches[0].package == "flash-attn"
    assert result.chosen.conflicts == []


def test_resolve_detects_conflict(tmp_path):
    """Resolver detects numpy conflict between packages."""
    from gpkg.resolver import resolve
    from gpkg.matching import WheelMatch
    from gpkg.registry import Source, RequiresBlock

    sources = [
        Source(
            package="pkg-a", description="t", source_type="github",
            requires=[RequiresBlock(["1.0"], ["numpy>=2.0"])],
        ),
        Source(
            package="pkg-b", description="t", source_type="github",
            requires=[RequiresBlock(["1.0"], ["numpy<2.0"])],
        ),
    ]

    all_versions = {
        "pkg-a": [WheelMatch("pkg-a", "a.whl", "", "1.0", "2.11", "128",
                              "cp312-cp312", "linux_x86_64", "t", None, "")],
        "pkg-b": [WheelMatch("pkg-b", "b.whl", "", "1.0", "2.11", "128",
                              "cp312-cp312", "linux_x86_64", "t", None, "")],
    }
    env = {"torch": "2.11", "cuda": "128", "python": "3.12", "platform": "linux_x86_64"}

    result = resolve(
        all_versions=all_versions,
        env=env,
        sources=sources,
        client=None,
        skip_trial=True,
        cache_dir=tmp_path,
    )

    assert result is not None
    assert len(result.chosen.conflicts) > 0
    assert result.chosen.conflicts[0].dependency == "numpy"


def test_format_resolve_result_happy_path():
    """Happy path produces clean output."""
    from gpkg.resolver import format_resolve_result, ResolveResult, Combo
    from gpkg.matching import WheelMatch

    match = WheelMatch(
        "flash-attn", "flash_attn-2.8.3.whl",
        "https://wheels.mapika.dev/flash-attn/flash_attn-2.8.3.whl",
        "2.8.3", "2.11", "128", "cp312-cp312", "linux_x86_64",
        "flash-attn \u2014 gpkg hosted registry", None, "",
    )
    result = ResolveResult(
        chosen=Combo(matches=[match], conflicts=[], score=100.0),
        alternatives=[],
        from_cache=True,
        cache_key="abc123",
    )

    output = format_resolve_result(result)
    assert "flash-attn" in output
    assert "2.8.3" in output


def test_format_resolve_result_conflict():
    """Conflict path shows diagnosis."""
    from gpkg.resolver import format_resolve_result, ResolveResult, Combo, Conflict
    from gpkg.matching import WheelMatch

    def make(pkg, ver):
        return WheelMatch(pkg, f"{pkg}.whl", "", ver, "2.11", "128",
                          "cp312-cp312", "linux_x86_64", "test", None, "")

    conflict = Conflict("numpy", {"pkg-a": ">=2.0", "pkg-b": "<2.0"}, "empty")
    result = ResolveResult(
        chosen=Combo(matches=[make("pkg-a", "1.0"), make("pkg-b", "1.0")],
                     conflicts=[conflict], score=-1000.0),
        alternatives=[],
        from_cache=False,
        cache_key="abc123",
    )

    output = format_resolve_result(result)
    assert "numpy" in output
    assert "conflict" in output.lower() or "Conflict" in output


def test_build_report_format():
    """Build report contains required fields."""
    from gpkg.resolver import format_build_report

    report = format_build_report(
        package="mamba-ssm",
        version="2.2.4",
        torch="2.11.0",
        cuda="128",
        python="3.12",
        platform="linux_x86_64",
        gpu_arch="9.0",
        build_time=847,
        gpkg_version="0.5.0",
    )
    assert report["package"] == "mamba-ssm"
    assert report["version"] == "2.2.4"
    assert report["torch"] == "2.11.0"
    assert report["success"] is True
    assert report["build_time_seconds"] == 847
    assert report["gpkg_version"] == "0.5.0"


def test_full_pipeline_mamba_stack(tmp_path):
    """End-to-end: resolve the mamba stack (flash-attn + causal-conv1d + mamba-ssm)."""
    from gpkg.resolver import resolve, ResolveResult, read_compat_cache
    from gpkg.matching import WheelMatch
    from gpkg.registry import Source, RequiresBlock

    sources = [
        Source(
            package="flash-attn", description="t", source_type="github",
            requires=[RequiresBlock(["2.8.3"], ["torch>=2.0", "einops", "packaging"])],
        ),
        Source(
            package="causal-conv1d", description="t", source_type="github",
            requires=[RequiresBlock(["1.5.0"], ["torch>=2.0", "packaging"])],
        ),
        Source(
            package="mamba-ssm", description="t", source_type="github",
            requires=[RequiresBlock(["2.2.4"], ["torch>=2.0", "causal-conv1d>=1.4.0", "packaging"])],
        ),
    ]

    def make(pkg, ver):
        return WheelMatch(
            pkg, f"{pkg}-{ver}.whl", f"https://wheels.mapika.dev/{pkg}/{pkg}-{ver}.whl",
            ver, "2.11", "128", "cp312-cp312", "linux_x86_64",
            f"{pkg} — gpkg hosted registry", "TRUE", "",
        )

    all_versions = {
        "flash-attn": [make("flash-attn", "2.8.3"), make("flash-attn", "2.8.2")],
        "causal-conv1d": [make("causal-conv1d", "1.5.0"), make("causal-conv1d", "1.4.0")],
        "mamba-ssm": [make("mamba-ssm", "2.2.4"), make("mamba-ssm", "2.2.3")],
    }
    env = {"torch": "2.11", "cuda": "128", "python": "3.12", "platform": "linux_x86_64"}

    result = resolve(
        all_versions=all_versions,
        env=env,
        sources=sources,
        client=None,
        skip_trial=True,
        cache_dir=tmp_path,
    )

    assert result is not None
    assert not result.from_cache
    assert len(result.chosen.matches) == 3
    assert result.chosen.conflicts == []

    # Should pick newest versions (no conflicts in this stack)
    versions = {m.package: m.version for m in result.chosen.matches}
    assert versions["flash-attn"] == "2.8.3"
    assert versions["causal-conv1d"] == "1.5.0"
    assert versions["mamba-ssm"] == "2.2.4"

    # Should have cached the result
    cached = read_compat_cache(["flash-attn", "causal-conv1d", "mamba-ssm"], env, cache_dir=tmp_path)
    assert cached is not None
    assert cached.status == "user-resolved"

    # Re-resolve should hit cache
    result2 = resolve(
        all_versions=all_versions,
        env=env,
        sources=sources,
        client=None,
        skip_trial=True,
        cache_dir=tmp_path,
    )
    assert result2 is not None
    assert result2.from_cache
