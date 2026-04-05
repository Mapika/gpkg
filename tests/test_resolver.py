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
