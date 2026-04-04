"""Doctor: verify resolved wheel URLs are accessible."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import httpx

from gpkg.cache import _get_registry_auth
from gpkg.matching import WheelMatch


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


def _doctor_check_one(
    client: httpx.Client, name: str, m: WheelMatch,
) -> DoctorResult:
    """Verify a single wheel URL via HEAD request."""
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
        )
    try:
        headers = _get_registry_auth(m.url) or {}
        resp = client.head(m.url, headers=headers, follow_redirects=True, timeout=15)
        length = resp.headers.get("content-length")
        return DoctorResult(
            package=name,
            filename=m.filename,
            url=m.url,
            url_ok=resp.status_code == 200,
            status_code=resp.status_code,
            content_length=int(length) if length else None,
        )
    except Exception as e:
        return DoctorResult(
            package=name,
            filename=m.filename,
            url=m.url,
            url_ok=False,
            status_code=0,
            content_length=None,
            error=str(e),
        )


def doctor_check(
    wheel_matches: dict[str, WheelMatch],
) -> list[DoctorResult]:
    """Verify resolved wheel URLs are accessible via concurrent HEAD requests."""
    if not wheel_matches:
        return []
    client = httpx.Client(follow_redirects=True)
    try:
        with ThreadPoolExecutor(max_workers=min(4, len(wheel_matches))) as pool:
            futures = {
                pool.submit(_doctor_check_one, client, name, m): name
                for name, m in wheel_matches.items()
            }
            results = [future.result() for future in as_completed(futures)]
    finally:
        client.close()
    order = list(wheel_matches.keys())
    results.sort(key=lambda r: order.index(r.package))
    return results
