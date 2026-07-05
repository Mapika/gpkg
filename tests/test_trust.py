"""Tests for the trust layer: hash capture, lockfile provenance, export."""

from gpkg.cache import parse_find_links_html
from gpkg.lockfile import write_lockfile, read_lockfile, lockfile_to_wheel_matches
from gpkg.matching import WheelMatch, _build_wheel_match
from gpkg.registry import Source


def _wheel(name="flash-attn", sha256=""):
    return WheelMatch(
        package=name,
        filename="flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_x86_64.whl",
        url="https://example.com/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_x86_64.whl",
        version="2.8.3", torch_version="2.9", cuda_tag="128",
        python_tag="cp312-cp312", platform_tag="linux_x86_64",
        source_desc="test source", release_tag="v2.8.3", sha256=sha256,
    )


# -- find-links hash fragments ------------------------------------------------


def test_parse_find_links_sha256_fragment():
    """PyPI-style indexes carry the hash as a #sha256= URL fragment."""
    html = (
        '<a href="/whl/cu128/xformers-0.0.30-cp310-cp310-manylinux_2_28_x86_64.whl'
        '#sha256=fc3cc23baf901e2ecb33f525bca12321bd275d2ad69360beb6edaf6dda3ab064">'
        "xformers-0.0.30-cp310-cp310-manylinux_2_28_x86_64.whl</a>"
    )
    results = parse_find_links_html(html, "https://download.pytorch.org/whl/cu128/xformers/")
    assert len(results) == 1
    assert results[0]["name"] == "xformers-0.0.30-cp310-cp310-manylinux_2_28_x86_64.whl"
    assert results[0]["url"] == (
        "https://download.pytorch.org/whl/cu128/xformers-0.0.30-cp310-cp310-manylinux_2_28_x86_64.whl"
    )
    assert results[0]["sha256"] == "fc3cc23baf901e2ecb33f525bca12321bd275d2ad69360beb6edaf6dda3ab064"


def test_parse_find_links_no_fragment():
    """Indexes without fragments still parse; sha256 is empty."""
    html = '<a href="torch-2.7.0%2Bcu128/pyg_lib-0.4.0%2Bpt27cu128-cp310-cp310-linux_x86_64.whl">x</a>'
    results = parse_find_links_html(html, "https://data.pyg.org/whl/torch-2.7.0+cu128.html")
    assert len(results) == 1
    assert results[0]["name"] == "pyg_lib-0.4.0+pt27cu128-cp310-cp310-linux_x86_64.whl"
    assert results[0]["sha256"] == ""


# -- GitHub digest capture ----------------------------------------------------


def test_build_wheel_match_github_digest():
    """GitHub asset 'digest' (sha256:<hex>) lands in WheelMatch.sha256."""
    src = Source(
        package="causal-conv1d", description="test", source_type="github",
        repo="Dao-AILab/causal-conv1d",
        wheel_name="causal_conv1d-{version}+cu{cuda}torch{torch}cxx11abi{abi}-{pytag}-{platform}.whl",
        cuda_style="short", has_abi=True,
    )
    fname = "causal_conv1d-1.6.2.post1+cu11torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    asset = {
        "name": fname,
        "browser_download_url": f"https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/{fname}",
        "digest": "sha256:6751d5b818e989cbd6fa5074a1b503a05a0751bc01f61e8afe713139b688ce91",
    }
    g = src.regex.match(fname).groupdict()
    m = _build_wheel_match(src, fname, asset, g, "v1.6.2.post1")
    assert m.sha256 == "6751d5b818e989cbd6fa5074a1b503a05a0751bc01f61e8afe713139b688ce91"


# -- lockfile roundtrip ---------------------------------------------------------


def test_lockfile_sha256_roundtrip(tmp_path):
    path = str(tmp_path / "gpkg.lock.toml")
    sha = "6751d5b818e989cbd6fa5074a1b503a05a0751bc01f61e8afe713139b688ce91"
    write_lockfile(path, "2.9.0", "128", "3.12", "linux_x86_64", "TRUE",
                   {"flash-attn": _wheel(sha256=sha)})
    lock = read_lockfile(path)
    env, wheels = lockfile_to_wheel_matches(lock)
    assert wheels["flash-attn"]["sha256"] == sha
    assert wheels["flash-attn"]["release_tag"] == "v2.8.3"
    assert wheels["flash-attn"]["source_desc"] == "test source"


def test_lockfile_without_sha256_omits_key(tmp_path):
    path = str(tmp_path / "gpkg.lock.toml")
    write_lockfile(path, "2.9.0", "128", "3.12", "linux_x86_64", "TRUE",
                   {"flash-attn": _wheel(sha256="")})
    lock = read_lockfile(path)
    assert "sha256" not in lock["wheels"][0]
    _, wheels = lockfile_to_wheel_matches(lock)
    assert wheels["flash-attn"]["sha256"] == ""


# -- local wheel hashing --------------------------------------------------------


def test_local_wheel_match_hashes_file(tmp_path):
    from gpkg.cli import _local_wheel_match, _file_sha256
    import hashlib
    whl = tmp_path / "mypkg-1.0.0-cp312-cp312-linux_x86_64.whl"
    whl.write_bytes(b"fake wheel bytes")
    expected = hashlib.sha256(b"fake wheel bytes").hexdigest()
    assert _file_sha256(whl) == expected
    m = _local_wheel_match("mypkg", whl, "3.12", "linux_x86_64", "2.9.0", "128")
    assert m.sha256 == expected
    assert m.source_desc == "local-build"
