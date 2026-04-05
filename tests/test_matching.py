"""Tests for gpkg matching logic."""

import subprocess
from unittest.mock import patch

import pytest

from gpkg.cache import (
    _find_links_cache_key,
    _get_registry_auth,
    cache_clean,
    cache_info,
    fetch_find_links,
    parse_duration,
    parse_find_links_html,
)
from gpkg.build import (
    _wheel_cache_dir,
    build_env_vars,
    build_wheel,
    detect_build_jobs,
    detect_gpu_arch,
    ensure_ninja,
    find_cached_wheel,
)
from gpkg.lockfile import (
    compare_lock,
    read_lockfile,
    write_lockfile,
)
from gpkg.matching import (
    cuda_matches,
    ExplainReport,
    normalize_cuda,
    pick_best,
    platform_matches,
    python_tag_matches,
    RejectReason,
    scan_available_combos,
    search_source,
    search_source_explain,
    torch_compat_matches,
    torch_matches,
    torch_minor,
    WheelMatch,
)
from gpkg.registry import Source, _template_to_regex


# -- CUDA matching ---------------------------------------------------------


class TestCudaMatches:
    def test_full_style_130(self):
        assert cuda_matches("130", "full", "130")
        assert cuda_matches("130", "full", "13.0")
        assert not cuda_matches("126", "full", "130")

    def test_full_style_126(self):
        assert cuda_matches("126", "full", "126")
        assert cuda_matches("126", "full", "12.6")

    def test_short_style(self):
        assert cuda_matches("13", "short", "130")
        assert cuda_matches("13", "short", "13.0")
        assert cuda_matches("12", "short", "126")
        assert cuda_matches("12", "short", "12.6")
        assert not cuda_matches("11", "short", "130")

    def test_normalize(self):
        assert normalize_cuda("130", "full") == "13.0"
        assert normalize_cuda("126", "full") == "12.6"
        assert normalize_cuda("13", "short") == "13"


# -- Torch matching --------------------------------------------------------


class TestTorchMatches:
    def test_minor(self):
        assert torch_matches("2.11", "2.11", "minor")
        assert torch_matches("2.7", "2.7", "minor")
        assert not torch_matches("2.10", "2.11", "minor")

    def test_packed(self):
        assert torch_matches("2110", "2.11", "packed")
        assert torch_matches("2100", "2.10", "packed")
        assert torch_matches("280", "2.8", "packed")
        assert not torch_matches("2100", "2.11", "packed")

    def test_full(self):
        assert torch_matches("2.9.0", "2.9", "full")
        assert torch_matches("2.11.0", "2.11", "full")
        assert torch_matches("2.9.0andhigher", "2.9", "full")
        assert not torch_matches("2.8.0", "2.9", "full")


# -- Python tag matching ---------------------------------------------------


class TestPythonTagMatches:
    def test_exact(self):
        assert python_tag_matches("cp312-cp312", "3.12")
        assert not python_tag_matches("cp311-cp311", "3.12")

    def test_abi3(self):
        assert python_tag_matches("cp39-abi3", "3.12")
        assert python_tag_matches("cp39-abi3", "3.9")
        assert not python_tag_matches("cp312-abi3", "3.11")


# -- Platform matching -----------------------------------------------------


class TestPlatformMatches:
    def test_linux(self):
        assert platform_matches("linux_x86_64", "linux_x86_64")
        assert platform_matches("manylinux_2_24_x86_64.manylinux_2_28_x86_64", "linux_x86_64")
        assert not platform_matches("win_amd64", "linux_x86_64")

    def test_windows(self):
        assert platform_matches("win_amd64", "win_amd64")
        assert not platform_matches("linux_x86_64", "win_amd64")

    def test_aarch64(self):
        assert platform_matches("linux_aarch64", "linux_aarch64")


# -- Template to regex -----------------------------------------------------


class TestTemplateRegex:
    def test_mjun0812_fa2(self):
        tpl = "flash_attn-{version}+cu{cuda}torch{torch}-{pytag}-{platform}.whl"
        rx = _template_to_regex(tpl)
        m = rx.match("flash_attn-2.8.3+cu130torch2.11-cp312-cp312-linux_x86_64.whl")
        assert m is not None
        assert m.group("version") == "2.8.3"
        assert m.group("cuda") == "130"
        assert m.group("torch") == "2.11"
        assert m.group("pytag") == "cp312-cp312"

    def test_daoai_conv1d(self):
        tpl = "causal_conv1d-{version}+cu{cuda}torch{torch}cxx11abi{abi}-{pytag}-{platform}.whl"
        rx = _template_to_regex(tpl)
        m = rx.match("causal_conv1d-1.6.1+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl")
        assert m is not None
        assert m.group("version") == "1.6.1"
        assert m.group("cuda") == "13"
        assert m.group("torch") == "2.10"
        assert m.group("abi") == "TRUE"

    def test_fa3_with_hash(self):
        tpl = "flash_attn_3-{version}+cu{cuda}torch{torch}{hash}-{pytag}-{platform}.whl"
        rx = _template_to_regex(tpl)
        m = rx.match("flash_attn_3-3.0.0+cu130torch2.11gite2743ab-cp39-abi3-linux_x86_64.whl")
        assert m is not None
        assert m.group("version") == "3.0.0"
        assert m.group("pytag") == "cp39-abi3"

    def test_natten_packed(self):
        tpl = "natten-{version}+torch{torch}cu{cuda}-{pytag}-{platform}.whl"
        rx = _template_to_regex(tpl)
        m = rx.match("natten-0.21.5+torch2100cu130-cp312-cp312-linux_x86_64.whl")
        assert m is not None
        assert m.group("version") == "0.21.5"
        assert m.group("torch") == "2100"
        assert m.group("cuda") == "130"

    def test_post_version(self):
        tpl = "causal_conv1d-{version}+cu{cuda}torch{torch}cxx11abi{abi}-{pytag}-{platform}.whl"
        rx = _template_to_regex(tpl)
        # .post4 is not a separate release, it's part of the version in the tag
        # The regex handles it via the version group
        # Actually the filename from Dao-AILab is just "1.6.1" even from the v1.6.1.post4 release
        # So this specific test checks our regex handles the .post variant if it appears
        assert rx.match("causal_conv1d-1.6.1.post4+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl") is not None

    def test_windreamer_wildcard(self):
        tpl = "flash_attn_3-{version}+{_}.cu{cuda}torch{torch}cxx11abi{_}-{pytag}-{platform}.whl"
        rx = _template_to_regex(tpl)
        m = rx.match("flash_attn_3-3.0.0+20260330.cu130torch2110cxx11abitrue.98024f-cp39-abi3-linux_x86_64.whl")
        assert m is not None
        assert m.group("version") == "3.0.0"
        assert m.group("cuda") == "130"
        assert m.group("torch") == "2110"
        assert m.group("pytag") == "cp39-abi3"
        assert m.group("platform") == "linux_x86_64"

    def test_sageattention_full_torch(self):
        tpl = "sageattention-{version}+cu{cuda}torch{torch}-{pytag}-{platform}.whl"
        rx = _template_to_regex(tpl)
        m = rx.match("sageattention-2.2.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl")
        assert m is not None
        # Regex captures full suffix; torch_matches("full") normalizes it
        assert m.group("torch") == "2.9.0andhigher.post4"
        assert torch_matches(m.group("torch"), "2.9", "full")


# -- pick_best (version ordering) -----------------------------------------


class TestPickBest:
    def _make(self, version: str, platform: str = "linux_x86_64") -> WheelMatch:
        return WheelMatch(
            package="test",
            filename=f"test-{version}.whl",
            url="https://example.com",
            version=version,
            torch_version="2.11",
            cuda_tag="130",
            python_tag="cp312-cp312",
            platform_tag=platform,
            source_desc="test",
        )

    def test_higher_version_wins(self):
        best = pick_best([self._make("2.8.3"), self._make("2.7.4"), self._make("2.10.0")])
        assert best.version == "2.10.0"

    def test_plain_preferred_over_manylinux(self):
        best = pick_best([
            self._make("2.8.3", "manylinux_2_24_x86_64.manylinux_2_28_x86_64"),
            self._make("2.8.3", "linux_x86_64"),
        ])
        assert best.platform_tag == "linux_x86_64"

    def test_empty(self):
        assert pick_best([]) is None


# -- torch_compat matching -------------------------------------------------


class TestTorchCompat:
    def test_empty_always_matches(self):
        assert torch_compat_matches("", "2.11")
        assert torch_compat_matches("", "1.0")

    def test_gte(self):
        assert torch_compat_matches(">=2.4", "2.4")
        assert torch_compat_matches(">=2.4", "2.11")
        assert not torch_compat_matches(">=2.4", "2.3")

    def test_lt(self):
        assert torch_compat_matches("<2.6", "2.5")
        assert torch_compat_matches("<2.6", "2.4")
        assert not torch_compat_matches("<2.6", "2.6")
        assert not torch_compat_matches("<2.6", "2.11")

    def test_range(self):
        assert torch_compat_matches(">=2.4,<2.6", "2.4")
        assert torch_compat_matches(">=2.4,<2.6", "2.5")
        assert not torch_compat_matches(">=2.4,<2.6", "2.3")
        assert not torch_compat_matches(">=2.4,<2.6", "2.6")

    def test_eq_neq(self):
        assert torch_compat_matches("==2.11", "2.11")
        assert not torch_compat_matches("==2.11", "2.10")
        assert torch_compat_matches("!=2.10", "2.11")
        assert not torch_compat_matches("!=2.10", "2.10")

    def test_gt_lte(self):
        assert torch_compat_matches(">2.4", "2.5")
        assert not torch_compat_matches(">2.4", "2.4")
        assert torch_compat_matches("<=2.6", "2.6")
        assert not torch_compat_matches("<=2.6", "2.7")


# -- Lockfile round-trip ---------------------------------------------------


class TestLockfile:
    def _make_match(self, pkg: str, version: str) -> WheelMatch:
        return WheelMatch(
            package=pkg,
            filename=f"{pkg}-{version}.whl",
            url=f"https://example.com/{pkg}-{version}.whl",
            version=version,
            torch_version="2.11",
            cuda_tag="130",
            python_tag="cp312-cp312",
            platform_tag="linux_x86_64",
            source_desc="test source",
            release_tag="v1.0",
        )

    def test_write_and_read(self, tmp_path):
        lockfile = str(tmp_path / "gpkg.lock.toml")
        matches = {"flash-attn": self._make_match("flash-attn", "2.8.3")}
        write_lockfile(lockfile, "2.11", "130", "3.12", "linux_x86_64", "TRUE", matches)
        data = read_lockfile(lockfile)
        assert data is not None
        assert data["environment"]["torch"] == "2.11"
        assert data["environment"]["cuda"] == "130"
        assert len(data["wheels"]) == 1
        assert data["wheels"][0]["package"] == "flash-attn"
        assert data["wheels"][0]["version"] == "2.8.3"

    def test_read_missing(self):
        assert read_lockfile("/nonexistent/path.toml") is None

    def test_compare_no_changes(self, tmp_path):
        matches = {"flash-attn": self._make_match("flash-attn", "2.8.3")}
        lockfile = str(tmp_path / "gpkg.lock.toml")
        write_lockfile(lockfile, "2.11", "130", "3.12", "linux_x86_64", "TRUE", matches)
        old = read_lockfile(lockfile)
        changes = compare_lock(old, matches)
        assert changes == []

    def test_compare_version_bump(self, tmp_path):
        old_matches = {"flash-attn": self._make_match("flash-attn", "2.8.3")}
        lockfile = str(tmp_path / "gpkg.lock.toml")
        write_lockfile(lockfile, "2.11", "130", "3.12", "linux_x86_64", "TRUE", old_matches)
        old = read_lockfile(lockfile)

        new_matches = {"flash-attn": self._make_match("flash-attn", "2.9.0")}
        changes = compare_lock(old, new_matches)
        assert len(changes) == 1
        assert changes[0].kind == "updated"
        assert changes[0].old_version == "2.8.3"
        assert changes[0].new_version == "2.9.0"

    def test_compare_added_removed(self, tmp_path):
        old_matches = {"flash-attn": self._make_match("flash-attn", "2.8.3")}
        lockfile = str(tmp_path / "gpkg.lock.toml")
        write_lockfile(lockfile, "2.11", "130", "3.12", "linux_x86_64", "TRUE", old_matches)
        old = read_lockfile(lockfile)

        new_matches = {"mamba-ssm": self._make_match("mamba-ssm", "1.0.0")}
        changes = compare_lock(old, new_matches)
        assert len(changes) == 2  # one added, one removed
        kinds = {c.kind for c in changes}
        assert "added" in kinds
        assert "removed" in kinds
        packages = {c.package for c in changes}
        assert "mamba-ssm" in packages
        assert "flash-attn" in packages


# -- Explain report --------------------------------------------------------


class TestExplainReport:
    def test_reject_reason_fields(self):
        r = RejectReason("test.whl", "cuda", "wheel=126 target=130")
        assert r.filename == "test.whl"
        assert r.stage == "cuda"
        assert r.detail == "wheel=126 target=130"

    def test_explain_report_fields(self):
        src = Source(
            package="test",
            description="test source",
            source_type="github",
            repo="test/test",
            wheel_name="test-{version}+cu{cuda}torch{torch}-{pytag}-{platform}.whl",
        )
        report = ExplainReport(
            source=src,
            regex_pattern=".*",
            releases_scanned=5,
            assets_scanned=20,
            rejected=[RejectReason("a.whl", "cuda")],
            matched=[],
        )
        assert report.releases_scanned == 5
        assert report.assets_scanned == 20
        assert len(report.rejected) == 1
        assert len(report.matched) == 0


# -- GitHub token auto-detection -------------------------------------------


class TestGetHeaders:
    def test_github_token_env_var(self):
        with patch.dict("os.environ", {"GITHUB_TOKEN": "tok_123"}):
            from gpkg.cache import get_headers
            h = get_headers()
            assert h["Authorization"] == "Bearer tok_123"

    def test_gh_cli_fallback(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("gpkg.cache.subprocess.run") as mock_run:
                mock_run.return_value = subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="ghp_faketoken\n"
                )
                import gpkg.cache as cache_mod
                cache_mod._gh_token_cache = None
                h = cache_mod.get_headers()
                assert h["Authorization"] == "Bearer ghp_faketoken"

    def test_no_token_available(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("gpkg.cache.subprocess.run") as mock_run:
                mock_run.side_effect = FileNotFoundError
                import gpkg.cache as cache_mod
                cache_mod._gh_token_cache = None
                h = cache_mod.get_headers()
                assert "Authorization" not in h


# -- Disk cache ------------------------------------------------------------


class TestDiskCache:
    def test_cache_write_and_read(self, tmp_path):
        from gpkg.cache import _write_disk_cache, _read_disk_cache
        data = [{"tag_name": "v1.0", "assets": []}]
        _write_disk_cache(tmp_path, "owner/repo", 5, data)
        result = _read_disk_cache(tmp_path, "owner/repo", 5, ttl=600)
        assert result == data

    def test_cache_expired(self, tmp_path):
        from gpkg.cache import _write_disk_cache, _read_disk_cache
        data = [{"tag_name": "v1.0", "assets": []}]
        _write_disk_cache(tmp_path, "owner/repo", 5, data)
        result = _read_disk_cache(tmp_path, "owner/repo", 5, ttl=0)
        assert result is None

    def test_cache_missing(self, tmp_path):
        from gpkg.cache import _read_disk_cache
        result = _read_disk_cache(tmp_path, "owner/repo", 5, ttl=600)
        assert result is None

    def test_cache_key_format(self, tmp_path):
        from gpkg.cache import _write_disk_cache
        _write_disk_cache(tmp_path, "Dao-AILab/flash-attention", 8, [])
        expected = tmp_path / "Dao-AILab_flash-attention_8.json"
        assert expected.exists()


# -- JSON output -----------------------------------------------------------


class TestJsonOutput:
    def test_list_json_structure(self):
        """--list --json should produce a list of source dicts."""
        sources = [
            Source(
                package="flash-attn",
                description="Flash Attention 2 — mjun0812",
                source_type="github",
                repo="mjun0812/flash-attention-prebuild-wheels",
            ),
        ]
        result = [
            {
                "package": s.package,
                "description": s.description,
                "type": s.source_type,
                "repo": s.repo if s.source_type == "github" else s.url_template,
            }
            for s in sources
        ]
        assert result[0]["package"] == "flash-attn"
        assert result[0]["type"] == "github"
        assert result[0]["repo"] == "mjun0812/flash-attention-prebuild-wheels"

    def test_available_json_structure(self):
        """--available --json should produce dict of pkg -> platform -> combos."""
        combos = {("12.8", "2.10"), ("12.8", "2.9")}
        plat = "linux_x86_64"
        result = {
            plat: [{"cuda": c, "torch": t} for c, t in sorted(combos)]
        }
        assert len(result[plat]) == 2
        assert result[plat][0]["cuda"] == "12.8"


class TestFindLinks:
    def test_parse_simple_html(self):
        html = """
        <html><body>
        <a href="flash_attn_3-3.0.0+cu130torch2110-cp39-abi3-linux_x86_64.whl">flash_attn_3-3.0.0+cu130torch2110-cp39-abi3-linux_x86_64.whl</a>
        <a href="flash_attn_3-3.0.0+cu130torch2110-cp39-abi3-win_amd64.whl">flash_attn_3-3.0.0+cu130torch2110-cp39-abi3-win_amd64.whl</a>
        <a href="README.md">README.md</a>
        </body></html>
        """
        results = parse_find_links_html(html, "https://example.com/wheels/")
        assert len(results) == 2
        assert results[0]["name"] == "flash_attn_3-3.0.0+cu130torch2110-cp39-abi3-linux_x86_64.whl"
        assert results[0]["url"].startswith("https://example.com/wheels/")

    def test_parse_absolute_urls(self):
        html = '<a href="https://github.com/releases/download/v1.0/pkg-1.0.0-cp312-cp312-linux_x86_64.whl">pkg</a>'
        results = parse_find_links_html(html, "https://example.com/simple/pkg/")
        assert len(results) == 1
        assert results[0]["url"] == "https://github.com/releases/download/v1.0/pkg-1.0.0-cp312-cp312-linux_x86_64.whl"

    def test_parse_url_encoded(self):
        html = '<a href="torch-2.5.0%2Bcu124/pyg_lib-0.4.0%2Bpt25cu124-cp310-cp310-linux_x86_64.whl">pyg_lib</a>'
        results = parse_find_links_html(html, "https://data.pyg.org/whl/")
        assert len(results) == 1
        assert "+" in results[0]["name"]  # URL-decoded
        assert "%2B" not in results[0]["name"]

    def test_parse_empty_page(self):
        results = parse_find_links_html("<html></html>", "https://example.com/")
        assert results == []


class TestFetchFindLinks:
    def test_fetch_and_parse(self):
        from unittest.mock import MagicMock
        html = '<a href="pkg-1.0.0+cu130-cp312-cp312-linux_x86_64.whl">pkg</a>'
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = html
        mock_resp.headers = {}
        mock_client.get.return_value = mock_resp
        results = fetch_find_links(mock_client, "https://example.com/wheels/", use_cache=False)
        assert len(results) == 1
        assert results[0]["name"] == "pkg-1.0.0+cu130-cp312-cp312-linux_x86_64.whl"
        mock_client.get.assert_called_once()

    def test_fetch_uses_disk_cache(self, tmp_path):
        import json
        import time
        from unittest.mock import MagicMock, patch
        # Pre-populate cache
        cache_file = tmp_path / "findlinks_aHR0cHM6Ly9leGFtcGxlLmNvbS93aGVlbHMv.json"
        payload = {
            "_timestamp": time.time(),
            "data": [{"name": "cached.whl", "url": "https://example.com/cached.whl"}],
        }
        cache_file.write_text(json.dumps(payload))
        mock_client = MagicMock()
        with patch("gpkg.cache._cache_dir", return_value=tmp_path), \
             patch("gpkg.cache._find_links_cache", {}):
            results = fetch_find_links(mock_client, "https://example.com/wheels/", use_cache=True)
        assert len(results) == 1
        assert results[0]["name"] == "cached.whl"
        mock_client.get.assert_not_called()


class TestFindLinksExplainAvailable:
    def test_explain_find_links(self):
        from unittest.mock import MagicMock, patch

        source = Source(
            package="flash-attn-3",
            description="test find-links",
            source_type="find-links",
            url_template="https://example.com/wheels/cu{cuda}_torch{torch}",
            wheel_name="flash_attn_3-{version}+cu{cuda}torch{torch}-{pytag}-{platform}.whl",
            cuda_style="full",
        )

        html = (
            '<a href="flash_attn_3-3.0.0+cu130torch2.11-cp312-cp312-linux_x86_64.whl">f</a>'
            '<a href="flash_attn_3-3.0.0+cu126torch2.11-cp312-cp312-linux_x86_64.whl">f</a>'
        )
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = html
        mock_resp.headers = {}
        mock_client.get.return_value = mock_resp

        with patch("gpkg.cache._use_cache", False):
            report = search_source_explain(
                mock_client, source,
                target_torch="2.11", target_cuda="130",
                target_python="3.12", target_platform="linux_x86_64",
                cxx11_abi="TRUE",
            )

        assert len(report.matched) == 1
        assert len(report.rejected) == 1  # cu126 rejected
        assert report.assets_scanned == 2

    def test_available_find_links_no_placeholders(self):
        """find-links with no URL placeholders can scan available combos."""
        from unittest.mock import MagicMock, patch

        source = Source(
            package="pkg",
            description="test",
            source_type="find-links",
            url_template="https://example.com/simple/pkg/",
            wheel_name="pkg-{version}+cu{cuda}torch{torch}-{pytag}-{platform}.whl",
            cuda_style="full",
        )

        html = (
            '<a href="pkg-1.0.0+cu130torch2.11-cp312-cp312-linux_x86_64.whl">p</a>'
            '<a href="pkg-1.0.0+cu126torch2.10-cp312-cp312-linux_x86_64.whl">p</a>'
        )
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = html
        mock_resp.headers = {}
        mock_client.get.return_value = mock_resp

        with patch("gpkg.cache._use_cache", False):
            combos = scan_available_combos(mock_client, [source], "linux_x86_64")

        assert ("13.0", "2.11") in combos
        assert ("12.6", "2.10") in combos

    def test_available_find_links_with_placeholders_skipped(self):
        """find-links with URL placeholders should be skipped in --available."""
        from unittest.mock import MagicMock, patch

        source = Source(
            package="pkg",
            description="test",
            source_type="find-links",
            url_template="https://example.com/wheels/cu{cuda}_torch{torch}",
            wheel_name="pkg-{version}+cu{cuda}torch{torch}-{pytag}-{platform}.whl",
            cuda_style="full",
        )

        mock_client = MagicMock()
        with patch("gpkg.cache._use_cache", False):
            combos = scan_available_combos(mock_client, [source], "linux_x86_64")

        assert combos == set()
        mock_client.get.assert_not_called()


class TestSearchFindLinks:
    def test_search_source_find_links(self):
        from unittest.mock import MagicMock, patch

        source = Source(
            package="flash-attn-3",
            description="test find-links",
            source_type="find-links",
            url_template="https://example.com/wheels/cu{cuda}_torch{torch}",
            wheel_name="flash_attn_3-{version}+cu{cuda}torch{torch}-{pytag}-{platform}.whl",
            cuda_style="full",
            torch_format="minor",
        )

        html = (
            '<a href="flash_attn_3-3.0.0+cu130torch2.11-cp312-cp312-linux_x86_64.whl">'
            'flash_attn_3-3.0.0+cu130torch2.11-cp312-cp312-linux_x86_64.whl</a>'
            '<a href="flash_attn_3-3.0.0+cu130torch2.11-cp311-cp311-linux_x86_64.whl">'
            'flash_attn_3-3.0.0+cu130torch2.11-cp311-cp311-linux_x86_64.whl</a>'
        )
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = html
        mock_resp.headers = {}
        mock_client.get.return_value = mock_resp

        with patch("gpkg.cache._use_cache", False):
            matches = search_source(
                mock_client, source,
                target_torch="2.11", target_cuda="130",
                target_python="3.12", target_platform="linux_x86_64",
                cxx11_abi="TRUE",
            )

        assert len(matches) == 1  # only cp312 matches
        assert matches[0].filename == "flash_attn_3-3.0.0+cu130torch2.11-cp312-cp312-linux_x86_64.whl"
        assert "example.com" in matches[0].url

    def test_search_source_find_links_no_wheel_name(self):
        """find-links source without wheel_name should return empty."""
        from unittest.mock import MagicMock, patch

        source = Source(
            package="flash-attn-3",
            description="test find-links no pattern",
            source_type="find-links",
            url_template="https://example.com/wheels/cu{cuda}_torch{torch}",
            wheel_name="",
            cuda_style="full",
            torch_format="minor",
        )

        mock_client = MagicMock()
        with patch("gpkg.cache._use_cache", False):
            matches = search_source(
                mock_client, source,
                target_torch="2.11", target_cuda="130",
                target_python="3.12", target_platform="linux_x86_64",
                cxx11_abi="TRUE",
            )

        assert matches == []


class TestRegistryAuth:
    def test_host_specific_env_var(self):
        with patch.dict("os.environ", {"UVFORGE_TOKEN_WHEELS_MYCOMPANY_COM": "tok_host"}):
            auth = _get_registry_auth("https://wheels.mycompany.com/simple/pkg/")
            assert auth == {"Authorization": "Bearer tok_host"}

    def test_generic_env_var(self):
        env = {"UVFORGE_TOKEN": "tok_generic"}
        with patch.dict("os.environ", env, clear=True):
            auth = _get_registry_auth("https://example.com/wheels/")
            assert auth == {"Authorization": "Bearer tok_generic"}

    def test_no_auth_returns_none(self):
        with patch.dict("os.environ", {}, clear=True):
            auth = _get_registry_auth("https://example.com/wheels/")
            assert auth is None

    def test_host_specific_overrides_generic(self):
        env = {
            "UVFORGE_TOKEN": "tok_generic",
            "UVFORGE_TOKEN_EXAMPLE_COM": "tok_specific",
        }
        with patch.dict("os.environ", env, clear=True):
            auth = _get_registry_auth("https://example.com/wheels/")
            assert auth == {"Authorization": "Bearer tok_specific"}

    def test_host_with_hyphens(self):
        env = {"UVFORGE_TOKEN_MY_COOL_REGISTRY_IO": "tok_hyphens"}
        with patch.dict("os.environ", env, clear=True):
            auth = _get_registry_auth("https://my-cool-registry.io/pkg/")
            assert auth == {"Authorization": "Bearer tok_hyphens"}


# -- Exact torch pinning ---------------------------------------------------


class TestExactTorchPinning:
    def test_detect_torch_returns_full_version(self):
        """detect_torch should return full version like '2.11.0'."""
        from unittest.mock import MagicMock
        import sys
        mock_torch = MagicMock()
        mock_torch.__version__ = "2.11.0"
        with patch.dict(sys.modules, {"torch": mock_torch}):
            from gpkg.detect import detect_torch
            result = detect_torch()
            assert result == "2.11.0"

    def test_torch_minor_helper(self):
        assert torch_minor("2.11.0") == "2.11"
        assert torch_minor("2.11") == "2.11"
        assert torch_minor("2.9.1") == "2.9"

    def test_torch_matches_minor_with_full_target(self):
        """minor format: '2.11' should match target '2.11.0' (truncate target)."""
        assert torch_matches("2.11", "2.11.0", "minor")
        assert torch_matches("2.11", "2.11", "minor")
        assert not torch_matches("2.10", "2.11.0", "minor")

    def test_torch_matches_packed_with_full_target(self):
        """packed format: '2110' should match target '2.11.0' (truncate target)."""
        assert torch_matches("2110", "2.11.0", "packed")
        assert not torch_matches("2100", "2.11.0", "packed")

    def test_torch_matches_full_exact(self):
        """full format: '2.11.0' should match '2.11.0' exactly."""
        assert torch_matches("2.11.0", "2.11.0", "full")
        assert not torch_matches("2.11.1", "2.11.0", "full")

    def test_torch_matches_full_with_minor_target(self):
        """full format: '2.11.0' should match target '2.11' (any patch)."""
        assert torch_matches("2.11.0", "2.11", "full")
        assert torch_matches("2.11.1", "2.11", "full")
        assert not torch_matches("2.10.0", "2.11", "full")

    def test_lockfile_records_full_version(self, tmp_path):
        lockfile = str(tmp_path / "gpkg.lock.toml")
        match = WheelMatch(
            package="flash-attn", filename="f.whl", url="https://x.com/f.whl",
            version="2.8.3", torch_version="2.11", cuda_tag="130",
            python_tag="cp312-cp312", platform_tag="linux_x86_64",
            source_desc="test", release_tag="v1.0",
        )
        write_lockfile(lockfile, "2.11.0", "130", "3.12", "linux_x86_64", "TRUE", {"flash-attn": match})
        data = read_lockfile(lockfile)
        assert data["environment"]["torch"] == "2.11.0"


class TestCacheManagement:
    def test_parse_duration(self):
        assert parse_duration("5m") == 300
        assert parse_duration("1h") == 3600
        assert parse_duration("2d") == 172800
        assert parse_duration("30m") == 1800

    def test_parse_duration_invalid(self):
        with pytest.raises(ValueError):
            parse_duration("abc")
        with pytest.raises(ValueError):
            parse_duration("5x")

    def test_cache_info(self, tmp_path):
        import json
        import time
        from unittest.mock import patch
        for i in range(2):
            f = tmp_path / f"test_{i}.json"
            f.write_text(json.dumps({"_timestamp": time.time() - (i * 60), "data": []}))
        with patch("gpkg.cache._cache_dir", return_value=tmp_path):
            info = cache_info()
        assert info["files"] == 2
        assert info["bytes"] > 0
        assert info["path"] == str(tmp_path)

    def test_cache_clean_all(self, tmp_path):
        import json
        import time
        from unittest.mock import patch
        for i in range(3):
            f = tmp_path / f"test_{i}.json"
            f.write_text(json.dumps({"_timestamp": time.time(), "data": []}))
        with patch("gpkg.cache._cache_dir", return_value=tmp_path):
            count, freed = cache_clean()
        assert count == 3
        assert freed > 0
        assert list(tmp_path.glob("*.json")) == []

    def test_cache_clean_older_than(self, tmp_path):
        import json
        import os
        import time
        from unittest.mock import patch
        fresh = tmp_path / "fresh.json"
        fresh.write_text(json.dumps({"_timestamp": time.time(), "data": []}))
        old = tmp_path / "old.json"
        old.write_text(json.dumps({"_timestamp": time.time() - 7200, "data": []}))
        # Set old file's mtime to 2 hours ago so cache_clean detects it
        old_mtime = time.time() - 7200
        os.utime(old, (old_mtime, old_mtime))
        with patch("gpkg.cache._cache_dir", return_value=tmp_path):
            count, freed = cache_clean(max_age=3600)
        assert count == 1
        assert fresh.exists()
        assert not old.exists()


class TestETagCaching:
    def test_etag_stored_in_cache(self, tmp_path):
        import json
        from unittest.mock import MagicMock, patch

        html = '<a href="pkg-1.0.0-cp312-cp312-linux_x86_64.whl">pkg</a>'
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = html
        mock_resp.headers = {"etag": '"abc123"', "last-modified": "Thu, 03 Apr 2026 15:00:00 GMT"}
        mock_client.get.return_value = mock_resp

        with patch("gpkg.cache._cache_dir", return_value=tmp_path), \
             patch("gpkg.cache._find_links_cache", {}):
            fetch_find_links(mock_client, "https://example.com/wheels/", use_cache=True)

        cache_files = list(tmp_path.glob("findlinks_*.json"))
        assert len(cache_files) == 1
        payload = json.loads(cache_files[0].read_text())
        assert payload["_etag"] == '"abc123"'
        assert payload["_last_modified"] == "Thu, 03 Apr 2026 15:00:00 GMT"

    def test_304_returns_cached_data(self, tmp_path):
        import json
        import time
        from unittest.mock import MagicMock, patch

        url = "https://example.com/wheels/"
        cache_file = tmp_path / _find_links_cache_key(url)
        cached_data = [{"name": "cached.whl", "url": "https://example.com/cached.whl"}]
        payload = {
            "_timestamp": time.time() - 9999,  # expired TTL
            "_etag": '"abc123"',
            "_last_modified": "Thu, 03 Apr 2026 15:00:00 GMT",
            "data": cached_data,
        }
        cache_file.write_text(json.dumps(payload))

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 304
        mock_client.get.return_value = mock_resp

        with patch("gpkg.cache._cache_dir", return_value=tmp_path), \
             patch("gpkg.cache._find_links_cache", {}):
            results = fetch_find_links(mock_client, url, use_cache=True)

        assert results == cached_data
        # Verify conditional headers were sent
        sent_headers = mock_client.get.call_args[1].get("headers", {})
        assert sent_headers.get("If-None-Match") == '"abc123"'


class TestBuildEnv:
    def test_detect_gpu_arch_from_compute_cap(self):
        from unittest.mock import patch
        with patch("gpkg.build.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="12.0\n"
            )
            result = detect_gpu_arch()
            assert result == "12.0"

    def test_detect_gpu_arch_from_name_fallback(self):
        from unittest.mock import patch
        def side_effect(*args, **kwargs):
            cmd = args[0]
            if "compute_cap" in str(cmd):
                return subprocess.CompletedProcess(args=[], returncode=0, stdout="\n")
            if "name" in str(cmd):
                return subprocess.CompletedProcess(args=[], returncode=0, stdout="NVIDIA GeForce RTX 5070 Ti\n")
            return subprocess.CompletedProcess(args=[], returncode=1, stdout="")
        with patch("gpkg.build.subprocess.run", side_effect=side_effect):
            result = detect_gpu_arch()
            assert result == "12.0"

    def test_detect_gpu_arch_no_nvidia_smi(self):
        from unittest.mock import patch
        with patch("gpkg.build.subprocess.run", side_effect=FileNotFoundError):
            result = detect_gpu_arch()
            assert result is None

    def test_detect_build_jobs(self):
        from unittest.mock import patch
        with patch("os.cpu_count", return_value=64):
            jobs = detect_build_jobs()
            assert jobs >= 1
            assert jobs <= 64

    def test_build_env_vars(self):
        env = build_env_vars("12.0", 21)
        assert env["TORCH_CUDA_ARCH_LIST"] == "12.0"
        assert env["MAX_JOBS"] == "21"
        assert env["CMAKE_BUILD_PARALLEL_LEVEL"] == "21"
        assert env["MAKEFLAGS"] == "-j21"
        assert env["TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES"] == "1"

    def test_ensure_ninja_already_installed(self):
        from unittest.mock import patch, MagicMock
        with patch("importlib.util.find_spec", return_value=MagicMock()):
            assert ensure_ninja() is True

    def test_ensure_ninja_not_installed(self):
        from unittest.mock import patch
        with patch("importlib.util.find_spec", return_value=None):
            with patch("gpkg.build.subprocess.run") as mock_run:
                mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
                ensure_ninja()
                mock_run.assert_called_once()


class TestWheelCache:
    def test_wheel_cache_dir(self):
        d = _wheel_cache_dir()
        assert "gpkg" in str(d)
        assert "wheels" in str(d)

    def test_find_cached_wheel_hit(self, tmp_path):
        from unittest.mock import patch
        whl = tmp_path / "causal_conv1d-1.6.1-cp312-cp312-linux_x86_64.whl"
        whl.write_text("fake wheel")
        with patch("gpkg.build._wheel_cache_dir", return_value=tmp_path):
            result = find_cached_wheel("causal-conv1d", "cp312", "linux_x86_64")
            assert result is not None
            assert result.name == "causal_conv1d-1.6.1-cp312-cp312-linux_x86_64.whl"

    def test_find_cached_wheel_miss(self, tmp_path):
        from unittest.mock import patch
        with patch("gpkg.build._wheel_cache_dir", return_value=tmp_path):
            result = find_cached_wheel("causal-conv1d", "cp312", "linux_x86_64")
            assert result is None

    def test_find_cached_wheel_picks_latest(self, tmp_path):
        from unittest.mock import patch
        (tmp_path / "causal_conv1d-1.5.0-cp312-cp312-linux_x86_64.whl").write_text("old")
        (tmp_path / "causal_conv1d-1.6.1-cp312-cp312-linux_x86_64.whl").write_text("new")
        with patch("gpkg.build._wheel_cache_dir", return_value=tmp_path):
            result = find_cached_wheel("causal-conv1d", "cp312", "linux_x86_64")
            assert "1.6.1" in result.name

    def test_find_cached_wheel_manylinux(self, tmp_path):
        from unittest.mock import patch
        whl = tmp_path / "causal_conv1d-1.6.1-cp312-cp312-manylinux_2_28_x86_64.whl"
        whl.write_text("fake")
        with patch("gpkg.build._wheel_cache_dir", return_value=tmp_path):
            result = find_cached_wheel("causal-conv1d", "cp312", "linux_x86_64")
            assert result is not None

    def test_build_wheel_success(self, tmp_path):
        from unittest.mock import patch
        def fake_run(*args, **kwargs):
            whl = tmp_path / "causal_conv1d-1.6.1-cp312-cp312-linux_x86_64.whl"
            whl.write_text("built wheel")
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with patch("gpkg.build.subprocess.run", side_effect=fake_run), \
             patch("gpkg.build._wheel_cache_dir", return_value=tmp_path):
            env = {"TORCH_CUDA_ARCH_LIST": "12.0", "MAX_JOBS": "8"}
            result = build_wheel("causal-conv1d", env, timeout=60)
            assert result is not None
            assert result.exists()
            assert result.name.endswith(".whl")

    def test_build_wheel_failure(self, tmp_path):
        from unittest.mock import patch
        call_count = 0
        def fake_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # _find_pip_command version check
                return subprocess.CompletedProcess(args=[], returncode=0)
            raise subprocess.CalledProcessError(1, "pip")  # actual build fails
        with patch("gpkg.build.subprocess.run", side_effect=fake_run), \
             patch("gpkg.build._wheel_cache_dir", return_value=tmp_path):
            env = {"TORCH_CUDA_ARCH_LIST": "12.0"}
            result = build_wheel("nonexistent-pkg", env, timeout=60)
            assert result is None
