from __future__ import annotations

from hashlib import sha256
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit


_TRACKING_PREFIXES = ("utm_",)
_TRACKING_KEYS = {"spm", "from", "source", "share", "share_token", "fbclid", "gclid"}


def canonicalize_url(url: str, base_url: str | None = None) -> str:
    absolute = urljoin(base_url or "", str(url).strip())
    parts = urlsplit(absolute)
    scheme = (parts.scheme or "https").lower()
    netloc = parts.netloc.lower()
    path = parts.path or "/"
    query_items = []
    for key, value in parse_qsl(parts.query, keep_blank_values=False):
        lower_key = key.lower()
        if lower_key in _TRACKING_KEYS or any(lower_key.startswith(prefix) for prefix in _TRACKING_PREFIXES):
            continue
        query_items.append((key, value))
    query = urlencode(sorted(query_items), doseq=True)
    return urlunsplit((scheme, netloc, path, query, ""))


def url_host(url: str) -> str:
    return urlsplit(url).netloc.lower()


def stable_content_hash(*parts: str) -> str:
    payload = "\n".join(str(part or "") for part in parts)
    return sha256(payload.encode("utf-8")).hexdigest()
