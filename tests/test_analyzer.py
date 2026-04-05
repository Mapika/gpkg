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
