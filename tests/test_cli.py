"""Tests for the CLI: legacy argv rewriting and the subcommand parser."""

import pytest

from gpkg.cli import _build_parser, _ensure_defaults, _rewrite_legacy_argv


def _parse(argv):
    args = _build_parser().parse_args(_rewrite_legacy_argv(argv))
    _ensure_defaults(args)
    return args


def test_bare_packages_default_to_resolve():
    args = _parse(["flash-attn", "causal-conv1d"])
    assert args.command == "resolve"
    assert args.packages == ["flash-attn", "causal-conv1d"]


def test_flags_before_packages():
    args = _parse(["--torch", "2.11", "--cuda", "130", "flash-attn"])
    assert args.command == "resolve"
    assert args.torch == "2.11"
    assert args.cuda == "130"
    assert args.packages == ["flash-attn"]


def test_flags_before_command_are_relocated():
    args = _parse(["--torch", "2.9", "compat", "flash-attn"])
    assert args.command == "compat"
    assert args.torch == "2.9"
    assert args.packages == ["flash-attn"]


def test_legacy_list_flag():
    assert _rewrite_legacy_argv(["--list"]) == ["list"]
    assert _rewrite_legacy_argv(["--list", "--json"]) == ["list", "--json"]


def test_legacy_available_flag():
    assert _rewrite_legacy_argv(["--available", "natten"]) == ["available", "natten"]
    args = _parse(["--available", "causal-conv1d", "natten"])
    assert args.command == "available"
    assert args.packages == ["causal-conv1d", "natten"]


def test_legacy_cache_flags():
    assert _rewrite_legacy_argv(["--cache-info"]) == ["cache", "info"]
    assert _rewrite_legacy_argv(["--cache-clean", "--older-than", "1h"]) == \
        ["cache", "clean", "--older-than", "1h"]


def test_help_and_version_untouched():
    assert _rewrite_legacy_argv([]) == []
    assert _rewrite_legacy_argv(["--help"]) == ["--help"]
    assert _rewrite_legacy_argv(["-V"]) == ["-V"]


def test_add_sets_build_missing_and_defer():
    args = _parse(["add", "flash-attn"])
    assert args.command == "add"
    assert args.build_missing is True
    assert args.defer_builds is True


def test_add_requires_packages():
    with pytest.raises(SystemExit):
        _parse(["add"])


def test_doctor_command_sets_doctor_flag():
    args = _parse(["doctor", "flash-attn"])
    assert args.command == "doctor"
    assert args.doctor is True


def test_legacy_doctor_flag_on_resolve():
    args = _parse(["--doctor", "--torch", "2.11", "flash-attn"])
    assert args.command == "resolve"
    assert args.doctor is True


def test_cache_parser():
    args = _parse(["cache", "clean", "--older-than", "2d"])
    assert args.command == "cache"
    assert args.cache_action == "clean"
    assert args.older_than == "2d"


def test_defaults_filled_for_sparse_commands():
    args = _parse(["test"])
    assert args.command == "test"
    assert args.packages == []
    assert args.as_json is False
    assert args.no_cache is False
    assert args.build_missing is False


def test_analyze_packages_optional():
    args = _parse(["analyze"])
    assert args.command == "analyze"
    assert args.packages == []


def test_lock_flag_on_resolve():
    args = _parse(["--torch", "2.11", "--cuda", "130", "flash-attn", "--lock"])
    assert args.command == "resolve"
    assert args.lock is True
