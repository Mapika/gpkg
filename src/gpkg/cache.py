"""Caching, fetching, and authentication for wheel sources."""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urljoin, urlparse

import httpx

from gpkg import console

# ---------------------------------------------------------------------------
# GitHub API (cached)
# ---------------------------------------------------------------------------

_ANCHOR_RE = re.compile(r'<a\s[^>]*href="([^"]+)"')
_release_cache: dict[str, list[dict]] = {}
_use_cache: bool = True
_CACHE_TTL = 600  # 10 minutes


def _cache_dir() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    d = base / "gpkg" / "releases"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(repo: str, count: int) -> str:
    return f"{repo.replace('/', '_')}_{count}.json"


def _write_disk_cache(cache_path: Path, repo: str, count: int, data: list[dict]) -> None:
    payload = {"_timestamp": time.time(), "data": data}
    filepath = cache_path / _cache_key(repo, count)
    filepath.write_text(json.dumps(payload))


def _read_disk_cache(cache_path: Path, repo: str, count: int, ttl: int) -> Optional[list[dict]]:
    filepath = cache_path / _cache_key(repo, count)
    if not filepath.exists():
        return None
    try:
        payload = json.loads(filepath.read_text())
        if time.time() - payload["_timestamp"] > ttl:
            return None
        return payload["data"]
    except (json.JSONDecodeError, KeyError):
        return None


def parse_duration(s: str) -> int:
    """Parse a duration string like '5m', '1h', '2d' to seconds."""
    m = re.fullmatch(r"(\d+)([mhd])", s)
    if not m:
        raise ValueError(f"Invalid duration: {s!r} (use e.g. 5m, 1h, 2d)")
    val, unit = int(m.group(1)), m.group(2)
    return val * {"m": 60, "h": 3600, "d": 86400}[unit]


def cache_info() -> dict:
    """Return cache statistics: path, file count, total bytes, oldest/newest age."""
    path = _cache_dir()
    files = list(path.glob("*.json"))
    now = time.time()
    total_bytes = 0
    ages: list[float] = []
    for f in files:
        st = f.stat()
        total_bytes += st.st_size
        ages.append(now - st.st_mtime)
    return {
        "path": str(path),
        "files": len(files),
        "bytes": total_bytes,
        "oldest_age_s": max(ages) if ages else 0,
        "newest_age_s": min(ages) if ages else 0,
    }


def cache_clean(max_age: Optional[int] = None) -> tuple[int, int]:
    """Delete cache files. Returns (files_deleted, bytes_freed).

    If max_age is set (seconds), only delete files older than that.
    """
    path = _cache_dir()
    now = time.time()
    deleted = 0
    freed = 0
    for f in path.glob("*.json"):
        st = f.stat()
        if max_age is not None:
            if now - st.st_mtime < max_age:
                continue
        f.unlink()
        deleted += 1
        freed += st.st_size
    return deleted, freed


_gh_token_cache: Optional[str] = None


def _get_github_token() -> Optional[str]:
    global _gh_token_cache
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    if _gh_token_cache is not None:
        return _gh_token_cache or None
    try:
        result = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            _gh_token_cache = result.stdout.strip()
            return _gh_token_cache
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    _gh_token_cache = ""
    return None


def get_headers() -> dict:
    h = {"Accept": "application/vnd.github+json"}
    token = _get_github_token()
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _get_registry_auth(url: str) -> Optional[dict[str, str]]:
    """Get auth headers for a non-GitHub registry URL.

    Resolution chain:
    1. UVFORGE_TOKEN_<HOST> (host-specific, dots/hyphens -> underscores, uppercased)
    2. UVFORGE_TOKEN (generic fallback)
    3. None (let httpx use ~/.netrc)
    """
    host = urlparse(url).hostname or ""
    host_key = host.upper().replace(".", "_").replace("-", "_")
    token = os.environ.get(f"UVFORGE_TOKEN_{host_key}")
    if token:
        return {"Authorization": f"Bearer {token}"}
    token = os.environ.get("UVFORGE_TOKEN")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return None


def fetch_releases(
    client: httpx.Client, repo: str, count: int, *, use_cache: bool = True
) -> list[dict]:
    key = f"{repo}:{count}"
    if key in _release_cache:
        return _release_cache[key]
    cache_path = _cache_dir()
    if use_cache:
        cached = _read_disk_cache(cache_path, repo, count, _CACHE_TTL)
        if cached is not None:
            _release_cache[key] = cached
            return cached
    resp = client.get(
        f"https://api.github.com/repos/{repo}/releases",
        params={"per_page": count},
    )
    if resp.status_code == 403:
        console.print("[red]GitHub API rate limit hit.[/red] Set GITHUB_TOKEN env var.")
        sys.exit(1)
    resp.raise_for_status()
    data = resp.json()
    _release_cache[key] = data
    if use_cache:
        _write_disk_cache(cache_path, repo, count, data)
    return data


def _render_find_links_url(url_template: str, cuda: str, torch_ver: str) -> str:
    """Substitute {cuda} and {torch} placeholders in a find-links URL template."""
    return url_template.replace("{cuda}", cuda).replace("{torch}", torch_ver)


def parse_find_links_html(html: str, base_url: str) -> list[dict]:
    """Parse an HTML page for wheel links (PEP 503 simple index style).

    Returns list of {"name": decoded_filename, "url": resolved_url}.
    """
    results: list[dict] = []
    if not base_url.endswith("/"):
        base_url += "/"
    for m in _ANCHOR_RE.finditer(html):
        href = m.group(1)
        url = urljoin(base_url, href)
        decoded_url = unquote(url)
        fname = decoded_url.rsplit("/", 1)[-1]
        if not fname.endswith(".whl"):
            continue
        results.append({"name": fname, "url": decoded_url})
    return results


_find_links_cache: dict[str, list[dict]] = {}


def _find_links_cache_key(url: str) -> str:
    """Generate a cache filename for a find-links URL."""
    encoded = base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")
    return f"findlinks_{encoded}.json"


def fetch_find_links(
    client: httpx.Client, url: str, *, use_cache: bool = True,
    headers: Optional[dict] = None,
) -> list[dict]:
    """Fetch a find-links HTML page and return wheel entries.

    Returns list of {"name": filename, "url": download_url}.
    Supports ETag/Last-Modified conditional caching.
    """
    if url in _find_links_cache:
        return _find_links_cache[url]

    cache_path = _cache_dir()
    cache_file = cache_path / _find_links_cache_key(url)
    cached_payload: Optional[dict] = None

    if use_cache and cache_file.exists():
        try:
            cached_payload = json.loads(cache_file.read_text())
            if time.time() - cached_payload["_timestamp"] <= _CACHE_TTL:
                _find_links_cache[url] = cached_payload["data"]
                return cached_payload["data"]
        except (json.JSONDecodeError, KeyError):
            cached_payload = None

    # Build request headers
    req_headers = dict(headers or {})
    if cached_payload:
        etag = cached_payload.get("_etag")
        if etag:
            req_headers["If-None-Match"] = etag
        last_mod = cached_payload.get("_last_modified")
        if last_mod:
            req_headers["If-Modified-Since"] = last_mod

    resp = client.get(url, headers=req_headers, timeout=15)

    # 304 Not Modified — use cached data
    if resp.status_code == 304 and cached_payload:
        cached_payload["_timestamp"] = time.time()
        cache_file.write_text(json.dumps(cached_payload))
        _find_links_cache[url] = cached_payload["data"]
        return cached_payload["data"]

    resp.raise_for_status()
    results = parse_find_links_html(resp.text, url)

    _find_links_cache[url] = results
    if use_cache:
        payload: dict = {"_timestamp": time.time(), "data": results}
        etag = resp.headers.get("etag")
        if etag:
            payload["_etag"] = etag
        last_mod = resp.headers.get("last-modified")
        if last_mod:
            payload["_last_modified"] = last_mod
        cache_file.write_text(json.dumps(payload))

    return results
