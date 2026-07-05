"""Tests for the constraint analyzer."""

from gpkg.analyzer import (
    VersionInterval,
    specifier_to_interval,
    intersect_intervals,
    interval_to_specifier,
)
from packaging.version import Version


def test_gte_to_interval():
    iv, excl = specifier_to_interval(">=2.0")
    assert iv.lower == Version("2.0")
    assert iv.lower_inclusive is True
    assert iv.upper is None
    assert excl == set()


def test_lt_to_interval():
    iv, excl = specifier_to_interval("<2.0")
    assert iv.upper == Version("2.0")
    assert iv.upper_inclusive is False
    assert iv.lower is None


def test_eq_to_interval():
    iv, excl = specifier_to_interval("==2.0")
    assert iv.lower == Version("2.0")
    assert iv.upper == Version("2.0")
    assert iv.lower_inclusive is True
    assert iv.upper_inclusive is True


def test_compat_release_to_interval():
    """~=2.4 means >=2.4, <3.0"""
    iv, excl = specifier_to_interval("~=2.4")
    assert iv.lower == Version("2.4")
    assert iv.lower_inclusive is True
    assert iv.upper == Version("3.0")
    assert iv.upper_inclusive is False


def test_compat_release_minor_to_interval():
    """~=1.4.2 means >=1.4.2, <1.5.0"""
    iv, excl = specifier_to_interval("~=1.4.2")
    assert iv.lower == Version("1.4.2")
    assert iv.upper == Version("1.5.0")
    assert iv.upper_inclusive is False


def test_neq_to_exclusion():
    iv, excl = specifier_to_interval("!=1.5.0")
    assert iv.lower is None
    assert iv.upper is None
    assert Version("1.5.0") in excl


def test_compound_specifier():
    """>=1.26,<3.0 is two constraints intersected."""
    iv, excl = specifier_to_interval(">=1.26,<3.0")
    assert iv.lower == Version("1.26")
    assert iv.upper == Version("3.0")
    assert iv.lower_inclusive is True
    assert iv.upper_inclusive is False


def test_intersect_overlapping():
    a, _ = specifier_to_interval(">=1.0,<3.0")
    b, _ = specifier_to_interval(">=2.0,<4.0")
    result = intersect_intervals([a, b])
    assert result is not None
    assert result.lower == Version("2.0")
    assert result.upper == Version("3.0")


def test_intersect_disjoint():
    a, _ = specifier_to_interval(">=1.0,<2.0")
    b, _ = specifier_to_interval(">=3.0,<4.0")
    result = intersect_intervals([a, b])
    assert result is None


def test_intersect_single():
    a, _ = specifier_to_interval(">=2.0")
    result = intersect_intervals([a])
    assert result is not None
    assert result.lower == Version("2.0")
    assert result.upper is None


def test_intersect_tight():
    """Numpy conflict scenario: >=2.0 vs <2.0 -> empty."""
    a, _ = specifier_to_interval(">=2.0")
    b, _ = specifier_to_interval("<2.0")
    result = intersect_intervals([a, b])
    assert result is None


def test_intersect_touching_inclusive():
    """>=2.0 and <=2.0 -> [2.0, 2.0]"""
    a, _ = specifier_to_interval(">=2.0")
    b, _ = specifier_to_interval("<=2.0")
    result = intersect_intervals([a, b])
    assert result is not None
    assert result.lower == Version("2.0")
    assert result.upper == Version("2.0")


def test_interval_to_specifier_range():
    iv, _ = specifier_to_interval(">=2.0,<3.0")
    s = interval_to_specifier(iv)
    assert ">=" in s
    assert "2.0" in s
    assert "<" in s
    assert "3.0" in s


def test_interval_to_specifier_lower_only():
    iv, _ = specifier_to_interval(">=2.0")
    s = interval_to_specifier(iv)
    assert s == ">=2.0"


def test_interval_to_specifier_upper_only():
    iv, _ = specifier_to_interval("<3.0")
    s = interval_to_specifier(iv)
    assert s == "<3.0"


def test_parse_pypi_requires_dist():
    """Parse requires_dist from PyPI JSON response."""
    from gpkg.analyzer import parse_pypi_requires_dist

    pypi_data = {
        "info": {
            "requires_dist": [
                "numpy (>=1.24,<2.2)",
                "llvmlite (<0.44,>=0.43.0dev0)",
                "importlib-metadata ; python_version < \"3.9\"",
            ]
        }
    }
    result = parse_pypi_requires_dist(pypi_data, python_version="3.12")
    assert any("numpy" in r for r in result)
    assert any("llvmlite" in r for r in result)
    assert len(result) == 2


def test_parse_pypi_requires_dist_no_deps():
    """Package with no dependencies."""
    from gpkg.analyzer import parse_pypi_requires_dist

    pypi_data = {"info": {"requires_dist": None}}
    result = parse_pypi_requires_dist(pypi_data, python_version="3.12")
    assert result == []


def test_crawl_dep_tree_simple(tmp_path):
    """Crawl a simple dep tree with mocked PyPI data."""
    from gpkg.analyzer import crawl_dep_tree
    import json

    cache = tmp_path / "pypi"
    cache.mkdir()

    (cache / "boltz-2.2.1.json").write_text(json.dumps({
        "info": {
            "requires_dist": [
                "numpy (>=1.26,<2.0)",
                "numba (==0.61.0)",
                "scipy (==1.13.1)",
            ]
        }
    }))
    (cache / "numba-0.61.0.json").write_text(json.dumps({
        "info": {
            "requires_dist": [
                "numpy (>=1.24,<2.2)",
                "llvmlite (>=0.43)",
            ]
        }
    }))
    (cache / "scipy-1.13.1.json").write_text(json.dumps({
        "info": {
            "requires_dist": [
                "numpy (>=1.22,<2.3)",
            ]
        }
    }))
    (cache / "llvmlite-0.43.0.json").write_text(json.dumps({
        "info": {"requires_dist": []}
    }))

    tree = crawl_dep_tree("boltz", "2.2.1", client=None, cache_dir=cache)

    assert "numpy" in tree
    assert "boltz" in tree["numpy"]
    assert "numba" in tree["numpy"]
    assert "scipy" in tree["numpy"]
    assert tree["numpy"]["boltz"] == ">=1.26,<2.0"


def test_crawl_dep_tree_cycle(tmp_path):
    """Crawler handles cycles without infinite loop."""
    from gpkg.analyzer import crawl_dep_tree
    import json

    cache = tmp_path / "pypi"
    cache.mkdir()

    (cache / "a-1.0.json").write_text(json.dumps({
        "info": {"requires_dist": ["b (>=1.0)"]}
    }))
    (cache / "b-1.0.json").write_text(json.dumps({
        "info": {"requires_dist": ["a (>=1.0)"]}
    }))

    tree = crawl_dep_tree("a", "1.0", client=None, cache_dir=cache)
    assert isinstance(tree, dict)


def test_crawl_dep_tree_missing_package(tmp_path):
    """Crawler handles missing packages gracefully."""
    from gpkg.analyzer import crawl_dep_tree
    import json

    cache = tmp_path / "pypi"
    cache.mkdir()

    (cache / "a-1.0.json").write_text(json.dumps({
        "info": {"requires_dist": ["nonexistent-pkg (>=1.0)"]}
    }))

    tree = crawl_dep_tree("a", "1.0", client=None, cache_dir=cache)
    assert isinstance(tree, dict)


def test_analyze_relaxable_pin(tmp_path):
    """Boltz's numpy<2.0 is relaxable when no transitive dep needs it."""
    from gpkg.analyzer import analyze_constraint, ConstraintAnalysis
    import json

    cache = tmp_path / "pypi"
    cache.mkdir()

    (cache / "boltz-2.2.1.json").write_text(json.dumps({
        "info": {
            "requires_dist": [
                "numpy (>=1.26,<2.0)",
                "numba (==0.61.0)",
                "scipy (==1.13.1)",
            ]
        }
    }))
    (cache / "numba-0.61.0.json").write_text(json.dumps({
        "info": {"requires_dist": ["numpy (>=1.24,<2.2)"]}
    }))
    (cache / "scipy-1.13.1.json").write_text(json.dumps({
        "info": {"requires_dist": ["numpy (>=1.22,<2.3)"]}
    }))

    result = analyze_constraint(
        blocker_pkg="boltz",
        blocker_version="2.2.1",
        dep_name="numpy",
        stated_spec=">=1.26,<2.0",
        required_spec=">=2.0",
        client=None,
        cache_dir=cache,
    )

    assert result.relaxable is True
    assert result.safe_range is not None
    assert "2.0" in result.safe_range
    assert "2.2" in result.safe_range
    assert len(result.evidence) >= 2


def test_analyze_load_bearing_pin(tmp_path):
    """Pin is load-bearing when a transitive dep also requires it."""
    from gpkg.analyzer import analyze_constraint
    import json

    cache = tmp_path / "pypi"
    cache.mkdir()

    (cache / "pkg_a-1.0.json").write_text(json.dumps({
        "info": {
            "requires_dist": [
                "numpy (>=1.26,<2.0)",
                "old-lib (==1.0)",
            ]
        }
    }))
    (cache / "old_lib-1.0.json").write_text(json.dumps({
        "info": {"requires_dist": ["numpy (>=1.20,<2.0)"]}
    }))

    result = analyze_constraint(
        blocker_pkg="pkg-a",
        blocker_version="1.0",
        dep_name="numpy",
        stated_spec=">=1.26,<2.0",
        required_spec=">=2.0",
        client=None,
        cache_dir=cache,
    )

    assert result.relaxable is False
    assert result.safe_range is None


def test_analyze_no_transitive_constraints(tmp_path):
    """When no transitive dep constrains the dep, pin is relaxable."""
    from gpkg.analyzer import analyze_constraint
    import json

    cache = tmp_path / "pypi"
    cache.mkdir()

    (cache / "pkg_a-1.0.json").write_text(json.dumps({
        "info": {
            "requires_dist": [
                "numpy (>=1.26,<2.0)",
                "requests (>=2.0)",
            ]
        }
    }))
    (cache / "requests-2.32.3.json").write_text(json.dumps({
        "info": {"requires_dist": ["urllib3 (>=1.21)"]}
    }))
    (cache / "urllib3-2.0.0.json").write_text(json.dumps({
        "info": {"requires_dist": []}
    }))

    result = analyze_constraint(
        blocker_pkg="pkg-a",
        blocker_version="1.0",
        dep_name="numpy",
        stated_spec=">=1.26,<2.0",
        required_spec=">=2.0",
        client=None,
        cache_dir=cache,
    )

    assert result.relaxable is True


def test_format_analysis_relaxable():
    """Relaxable analysis shows evidence and safe range."""
    from gpkg.analyzer import format_analysis, ConstraintAnalysis

    analysis = ConstraintAnalysis(
        blocker="boltz",
        dependency="numpy",
        stated_range=">=1.26,<2.0",
        real_range=">=1.24,<2.2",
        safe_range=">=2.0,<2.2",
        relaxable=True,
        evidence=[
            "numba requires numpy>=1.24,<2.2",
            "scipy requires numpy>=1.22,<2.3",
        ],
    )
    output = format_analysis(analysis)
    assert "relaxable" in output.lower() or "Relaxable" in output
    assert "numpy" in output
    assert ">=2.0,<2.2" in output
    assert "numba" in output


def test_format_analysis_load_bearing():
    """Load-bearing analysis shows which dep needs the pin."""
    from gpkg.analyzer import format_analysis, ConstraintAnalysis

    analysis = ConstraintAnalysis(
        blocker="pkg-a",
        dependency="numpy",
        stated_range=">=1.26,<2.0",
        real_range=">=1.20,<2.0",
        safe_range=None,
        relaxable=False,
        evidence=["old-lib requires numpy>=1.20,<2.0"],
    )
    output = format_analysis(analysis)
    assert "load-bearing" in output.lower() or "Load-bearing" in output
    assert "old-lib" in output


def test_resolver_calls_analyzer_on_conflict(tmp_path):
    """Resolver runs analyzer when it detects a conflict."""
    from gpkg.resolver import resolve
    from gpkg.matching import WheelMatch
    from gpkg.registry import Source, RequiresBlock
    import json

    pypi_cache = tmp_path / "pypi"
    pypi_cache.mkdir()
    (pypi_cache / "boltz-2.2.1.json").write_text(json.dumps({
        "info": {
            "requires_dist": [
                "numpy (>=1.26,<2.0)",
                "numba (==0.61.0)",
            ]
        }
    }))
    (pypi_cache / "numba-0.61.0.json").write_text(json.dumps({
        "info": {"requires_dist": ["numpy (>=1.24,<2.2)"]}
    }))

    sources = [
        Source(package="flash-attn", description="t", source_type="github",
               requires=[RequiresBlock(["3.0.0"], ["numpy>=2.0"])]),
        Source(package="boltz", description="t", source_type="github",
               requires=[RequiresBlock(["2.2.1"], ["numpy>=1.26,<2.0"])]),
    ]

    def make(pkg, ver):
        return WheelMatch(pkg, f"{pkg}-{ver}.whl", "", ver, "2.10", "128",
                          "cp312-cp312", "linux_x86_64", "t", None, "")

    all_versions = {
        "flash-attn": [make("flash-attn", "3.0.0")],
        "boltz": [make("boltz", "2.2.1")],
    }
    env = {"torch": "2.10", "cuda": "128", "python": "3.12", "platform": "linux_x86_64"}

    result = resolve(
        all_versions=all_versions,
        env=env,
        sources=sources,
        client=None,
        skip_trial=True,
        cache_dir=tmp_path,
        pypi_cache_dir=pypi_cache,
    )

    assert result is not None
    assert len(result.analyses) > 0
    assert result.analyses[0].relaxable is True
    assert "2.0" in result.analyses[0].safe_range


def test_full_boltz_scenario(tmp_path):
    """End-to-end: boltz pins numpy<2.0, flash-attn needs numpy>=2.0.

    Analyzer should find that boltz's pin is relaxable because
    none of its transitive deps actually need numpy<2.0.
    """
    from gpkg.analyzer import analyze_constraint, format_analysis
    import json

    cache = tmp_path / "pypi"
    cache.mkdir()

    (cache / "boltz-2.2.1.json").write_text(json.dumps({
        "info": {
            "requires_dist": [
                "numpy (>=1.26,<2.0)",
                "numba (==0.61.0)",
                "scipy (==1.13.1)",
                "torch (>=2.2)",
                "einops (==0.8.0)",
            ]
        }
    }))
    (cache / "numba-0.61.0.json").write_text(json.dumps({
        "info": {
            "requires_dist": [
                "numpy (>=1.24,<2.2)",
                "llvmlite (==0.43.0)",
            ]
        }
    }))
    (cache / "scipy-1.13.1.json").write_text(json.dumps({
        "info": {
            "requires_dist": [
                "numpy (>=1.22.4,<2.3)",
            ]
        }
    }))
    (cache / "llvmlite-0.43.0.json").write_text(json.dumps({
        "info": {"requires_dist": []}
    }))
    (cache / "einops-0.8.0.json").write_text(json.dumps({
        "info": {"requires_dist": ["numpy"]}
    }))

    result = analyze_constraint(
        blocker_pkg="boltz",
        blocker_version="2.2.1",
        dep_name="numpy",
        stated_spec=">=1.26,<2.0",
        required_spec=">=2.0",
        client=None,
        cache_dir=cache,
    )

    # Should be relaxable
    assert result.relaxable is True

    # Safe range should be >=2.0,<2.2 (bounded by numba's upper)
    assert result.safe_range is not None
    assert "2.0" in result.safe_range
    assert "2.2" in result.safe_range

    # Evidence should mention numba and scipy
    evidence_text = " ".join(result.evidence)
    assert "numba" in evidence_text
    assert "scipy" in evidence_text

    # Format should be readable
    output = format_analysis(result)
    assert "relaxable" in output.lower() or "Relaxable" in output
    assert "boltz" in output

    # Now test the opposite: if llvmlite also pins numpy<2.0, NOT relaxable
    (cache / "llvmlite-0.43.0.json").write_text(json.dumps({
        "info": {"requires_dist": ["numpy (>=1.20,<2.0)"]}
    }))

    result2 = analyze_constraint(
        blocker_pkg="boltz",
        blocker_version="2.2.1",
        dep_name="numpy",
        stated_spec=">=1.26,<2.0",
        required_spec=">=2.0",
        client=None,
        cache_dir=cache,
    )

    assert result2.relaxable is False
    assert result2.safe_range is None


def test_analyze_package_end_to_end(tmp_path):
    """The gpkg analyze entry point: pick out restrictive constraints only."""
    from gpkg.analyzer import analyze_package
    import json

    cache = tmp_path / "pypi"
    cache.mkdir()

    (cache / "boltz-2.2.1.json").write_text(json.dumps({
        "info": {
            "requires_dist": [
                "numpy (>=1.26,<2.0)",
                "torch (>=2.2)",
                "requests",
            ]
        }
    }))

    name, version, analyses = analyze_package("boltz==2.2.1", client=None, cache_dir=cache)

    assert name == "boltz"
    assert version == "2.2.1"
    # numpy has an upper bound; torch (lower bound only) and requests (bare) are skipped
    assert [a.dependency for a in analyses] == ["numpy"]
    assert analyses[0].relaxable is True


def test_analyze_package_unknown(tmp_path):
    """Unknown package (no client, no cache) raises LookupError."""
    from gpkg.analyzer import analyze_package
    import pytest

    with pytest.raises(LookupError):
        analyze_package("no-such-package==1.0", client=None, cache_dir=tmp_path)
