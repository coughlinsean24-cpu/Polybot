"""
News provider interface.

A provider turns one upstream feed into ``RawNewsItem`` records.  It does not
deduplicate, classify, or judge credibility beyond declaring its own
``SourceProfile`` — that is the normaliser's job — so adding or swapping a
provider never touches the engine.
"""

from __future__ import annotations

import html
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from src.news_trader.http import HttpClient
from src.news_trader.models import RawNewsItem

# Latency classes, from best to worst.  None of the free sources here are
# push streams; "poll_fast" means an official feed published at release time
# and polled every <=60s, so detection latency ~= poll interval + CDN cache.
LATENCY_CLASSES = ("stream", "poll_fast", "poll", "delayed")


@dataclass(frozen=True)
class SourceProfile:
    key: str
    name: str
    url: str
    default_category: str
    credibility: float          # 0..1 prior for items from this source
    primary_official: bool      # the issuer of the information itself
    latency_class: str          # one of LATENCY_CLASSES
    expected_latency_seconds: float
    poll_seconds: float
    official_domains: tuple[str, ...] = ()
    notes: str = ""

    @property
    def is_realtime(self) -> bool:
        return self.latency_class in ("stream", "poll_fast")


@dataclass(frozen=True)
class SourceAssessment:
    """Per-item source judgement made by the provider that fetched it."""
    # Organisation identity used for corroboration: the same outlet reached
    # through two providers (e.g. Reuters on Bluesky and via GDELT) is ONE source.
    independence_key: str
    credibility: float
    primary_official: bool
    default_category: str


# Credibility priors for publishers reached through aggregators (GDELT) or
# that appear as link domains.  Unknown domains get UNKNOWN_DOMAIN_CREDIBILITY.
DOMAIN_CREDIBILITY: dict[str, float] = {
    "reuters.com": 0.92, "apnews.com": 0.92, "bloomberg.com": 0.88,
    "wsj.com": 0.86, "ft.com": 0.86, "nytimes.com": 0.85, "washingtonpost.com": 0.82,
    "cnbc.com": 0.80, "marketwatch.com": 0.78, "bbc.co.uk": 0.85, "bbc.com": 0.85,
    "npr.org": 0.80, "axios.com": 0.78, "politico.com": 0.80, "theguardian.com": 0.78,
    "cnn.com": 0.75, "abcnews.go.com": 0.75, "cbsnews.com": 0.75, "nbcnews.com": 0.75,
    "aljazeera.com": 0.72, "barrons.com": 0.80, "economist.com": 0.82,
    "espn.com": 0.80, "coindesk.com": 0.70, "theblock.co": 0.68,
    "federalreserve.gov": 1.0, "sec.gov": 0.97, "bls.gov": 1.0, "bea.gov": 1.0,
    "whitehouse.gov": 0.92, "treasury.gov": 0.97,
}
UNKNOWN_DOMAIN_CREDIBILITY = 0.40
OFFICIAL_DOMAINS = frozenset({
    "federalreserve.gov", "sec.gov", "bls.gov", "bea.gov", "whitehouse.gov", "treasury.gov",
})


def registrable_domain(host: str) -> str:
    """Best-effort eTLD+1 (handles the common two-part suffixes we see)."""
    host = (host or "").lower().split(":")[0].strip(".")
    if host.startswith("www."):
        host = host[4:]
    parts = host.split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in ("co.uk", "com.au", "go.com"):
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


class NewsProvider(ABC):
    def __init__(self, profile: SourceProfile):
        self.profile = profile

    @property
    def key(self) -> str:
        return self.profile.key

    @abstractmethod
    async def fetch(self, http: HttpClient, now: float) -> list[RawNewsItem]:
        """Return the provider's current items.  Raise on transport/format errors."""

    def assess(self, item: RawNewsItem) -> SourceAssessment:
        p = self.profile
        key = p.official_domains[0] if p.official_domains else p.key
        return SourceAssessment(key, p.credibility, p.primary_official, p.default_category)


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def strip_html(text: str | None, limit: int = 2000) -> str:
    if not text:
        return ""
    text = _TAG_RE.sub(" ", html.unescape(text))
    text = html.unescape(text)
    return _WS_RE.sub(" ", text).strip()[:limit]


def parse_timestamp(value: str | None) -> float | None:
    """Parse RFC-822, ISO-8601 and GDELT compact timestamps to epoch seconds.

    Returns None (never raises) for missing or unparseable values; a missing
    publication time is handled downstream by falling back to first-seen.
    """
    if not value:
        return None
    value = value.strip()
    try:
        dt = parsedate_to_datetime(value)
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
    except (TypeError, ValueError, IndexError):
        pass
    try:
        if re.fullmatch(r"\d{8}T\d{6}Z", value):  # GDELT: 20261022T153000Z
            dt = datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            return dt.timestamp()
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None
