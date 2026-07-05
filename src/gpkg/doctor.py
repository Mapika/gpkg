"""Doctor: verify resolved wheel URLs, hashes, and registry source health."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import httpx

import gpkg.cache as _cache_mod
from gpkg.cache import _get_registry_auth, fetch_find_links, fetch_releases, _render_find_links_url
from gpkg.matching import WheelMatch
from gpkg.registry import Source


@dataclass
class DoctorResult:
    """Result of verifying a resolved wheel."""
    package: str
    filename: str
    url: str
    url_ok: bool
    status_code: int
    content_length: Optional[int]
    error: str = ""
    sha256_expected: str = ""
    sha256_ok: Optional[bool] = None  # None = not verified


def _hash_url(client: httpx.Client, url: str, headers: dict) -> str:
    """Stream-download a wheel and return its sha256 hex digest."""
    h = hashlib.sha256()
    with client.stream("GET", url, headers=headers, follow_redirects=True, timeout=300) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_bytes(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _doctor_check_one(
    client: httpx.Client, name: str, m: WheelMatch, verify: bool = False,
) -> DoctorResult:
    """Verify a single wheel URL via HEAD request (full download if verify=True)."""
    expected = getattr(m, "sha256", "")
    # Skip local file:// URLs — can't HEAD them over HTTP
    if m.url.startswith("file://"):
        from pathlib import Path
        from urllib.parse import unquote
        local_path = Path(unquote(m.url.removeprefix("file://")))
        return DoctorResult(
            package=name, filename=m.filename, url=m.url,
            url_ok=local_path.exists(),
            status_code=200 if local_path.exists() else 0,
            content_length=local_path.stat().st_size if local_path.exists() else None,
            sha256_expected=expected,
        )
    try:
        headers = _get_registry_auth(m.url) or {}
        resp = client.head(m.url, headers=headers, follow_redirects=True, timeout=15)
        length = resp.headers.get("content-length")
        result = DoctorResult(
            package=name,
            filename=m.filename,
            url=m.url,
            url_ok=resp.status_code == 200,
            status_code=resp.status_code,
            content_length=int(length) if length else None,
            sha256_expected=expected,
        )
        if verify and result.url_ok and expected:
            actual = _hash_url(client, m.url, headers)
            result.sha256_ok = actual == expected
            if not result.sha256_ok:
                result.error = f"hash mismatch: expected {expected[:16]}…, got {actual[:16]}…"
        return result
    except Exception as e:
        return DoctorResult(
            package=name,
            filename=m.filename,
            url=m.url,
            url_ok=False,
            status_code=0,
            content_length=None,
            error=str(e),
            sha256_expected=expected,
        )


def doctor_check(
    wheel_matches: dict[str, WheelMatch], verify: bool = False,
) -> list[DoctorResult]:
    """Verify resolved wheel URLs are accessible via concurrent HEAD requests.

    With verify=True, also downloads each wheel and checks its sha256 against
    the source-attested hash recorded at resolution time.
    """
    if not wheel_matches:
        return []
    client = httpx.Client(follow_redirects=True)
    try:
        with ThreadPoolExecutor(max_workers=min(4, len(wheel_matches))) as pool:
            futures = {
                pool.submit(_doctor_check_one, client, name, m, verify): name
                for name, m in wheel_matches.items()
            }
            results = [future.result() for future in as_completed(futures)]
    finally:
        client.close()
    order = list(wheel_matches.keys())
    results.sort(key=lambda r: order.index(r.package))
    return results


# ---------------------------------------------------------------------------
# Registry source health
# ---------------------------------------------------------------------------


@dataclass
class SourceHealth:
    """Health of one registry source: does it still serve matching wheels?"""
    package: str
    description: str
    location: str
    wheel_count: int
    detail: str = ""

    @property
    def healthy(self) -> bool:
        return self.wheel_count > 0


# Representative (cuda, torch) combos used to render templated find-links URLs.
# Ordered newest-first; the first URL that returns matching wheels wins.
_PROBE_COMBOS = [
    ("130", "2.9.0"), ("128", "2.9.0"), ("128", "2.8.0"), ("128", "2.7.0"),
    ("126", "2.6.0"), ("124", "2.6.0"), ("124", "2.4.0"), ("121", "2.4.0"),
]


def _count_regex_matches(assets: list[dict], source: Source) -> int:
    pattern = source.regex
    return sum(1 for a in assets if pattern.match(a["name"]))


def check_source_health(client: httpx.Client, sources: list[Source]) -> list[SourceHealth]:
    """Check every registry source still serves wheels matching its pattern.

    No environment filtering — this asks "is the source alive and does its
    wheel_name pattern still match reality", which catches renamed repos,
    restructured indexes, and empty-200 responses (e.g. data.dgl.ai).
    """
    results: list[SourceHealth] = []
    for source in sources:
        if not source.wheel_name:
            continue

        if source.source_type == "github":
            try:
                releases = fetch_releases(
                    client, source.repo, source.scan_tags, use_cache=_cache_mod._use_cache)
            except Exception as e:
                results.append(SourceHealth(
                    source.package, source.description, source.repo, 0, f"fetch failed: {e}"))
                continue
            assets = [a for r in releases for a in r.get("assets", [])]
            count = _count_regex_matches(assets, source)
            detail = f"{len(releases)} releases, {len(assets)} assets"
            results.append(SourceHealth(
                source.package, source.description, source.repo, count, detail))
            continue

        if source.source_type != "find-links":
            continue

        templated = "{cuda}" in source.url_template or "{torch}" in source.url_template
        probe_urls = (
            [_render_find_links_url(source.url_template, c, t, source.torch_format)
             for c, t in _PROBE_COMBOS]
            if templated else [source.url_template]
        )
        count, detail = 0, "no probe URL returned matching wheels"
        for url in probe_urls:
            try:
                auth = _get_registry_auth(url)
                assets = fetch_find_links(
                    client, url, use_cache=_cache_mod._use_cache, headers=auth)
            except Exception:
                continue
            n = _count_regex_matches(assets, source)
            if n > 0:
                count, detail = n, url
                break
        results.append(SourceHealth(
            source.package, source.description, source.url_template, count, detail))
    return results
