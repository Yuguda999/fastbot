from urllib.parse import urljoin, urlparse, urlunparse

def is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

def is_internal_url(url: str, base_url: str) -> bool:
    base_domain = urlparse(base_url).netloc.lower()
    return urlparse(url).netloc.lower() == base_domain

def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip('/')
    return urlunparse((scheme, netloc, path, '', '', ''))
