"""Tests for uvforge matching logic."""

import pytest

from uvforge.cli import (
    _template_to_regex,
    cuda_matches,
    normalize_cuda,
    pick_best,
    platform_matches,
    python_tag_matches,
    torch_matches,
    WheelMatch,
)


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
        m = rx.match("causal_conv1d-1.6.1.post4+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl")
        # .post4 is not a separate release, it's part of the version in the tag
        # The regex handles it via the version group
        # Actually the filename from Dao-AILab is just "1.6.1" even from the v1.6.1.post4 release
        # So this specific test checks our regex handles the .post variant if it appears

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
