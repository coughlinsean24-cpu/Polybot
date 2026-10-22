"""
GDELT 2.0 DOC API provider (free, global, ~15-minute update cadence).

Documented limit: one request every 5 seconds per client; the HTTP client is
configured with host spacing for api.gdeltproject.org.  Response shape
(verified live): {"articles": [{url, title, seendate "YYYYMMDDTHHMMSSZ",
domain, language, sourcecountry, ...}]}.  GDELT is an aggregator: the
credibility of each item comes from its publishing domain, not from GDELT.
"""

from __future__ import annotations

from urllib.parse import urlparse

from src.news_trader.http import HttpClient, SchemaError
from src.news_trader.models import RawNewsItem
from src.news_trader.sources.base import (
    DOMAIN_CREDIBILITY,
    OFFICIAL_DOMAINS,
    UNKNOWN_DOMAIN_CREDIBILITY,
    NewsProvider,
    SourceAssessment,
    SourceProfile,
    parse_timestamp,
    registrable_domain,
    strip_html,
)

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


class GdeltProvider(NewsProvider):
    def __init__(self, profile: SourceProfile, query: str, max_records: int = 75,
                 timespan: str = "30min"):
        super().__init__(profile)
        self.query = query
        self.max_records = max_records
        self.timespan = timespan

    async def fetch(self, http: HttpClient, now: float) -> list[RawNewsItem]:
        data = await http.get_json(GDELT_DOC_URL, params={
            "query": self.query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": str(self.max_records),
            "sort": "datedesc",
            "timespan": self.timespan,
        })
        return self.parse(data, now)

    def assess(self, item: RawNewsItem) -> SourceAssessment:
        domain = registrable_domain(urlparse(item.url).netloc)
        cred = DOMAIN_CREDIBILITY.get(domain, UNKNOWN_DOMAIN_CREDIBILITY)
        # A GDELT sighting of an official page is still second-hand and
        # ~15 minutes late, so it never counts as the primary release itself.
        return SourceAssessment(domain, min(cred, 0.9), False,
                                "general" if domain not in OFFICIAL_DOMAINS else self.profile.default_category)

    def parse(self, data, now: float) -> list[RawNewsItem]:
        if data == {} or data is None:
            return []  # GDELT returns {} when nothing matches
        if not isinstance(data, dict) or not isinstance(data.get("articles", []), list):
            raise SchemaError(f"{self.key}: unexpected GDELT payload type")
        out: list[RawNewsItem] = []
        for art in data.get("articles", []):
            if not isinstance(art, dict):
                continue
            url = art.get("url")
            title = strip_html(art.get("title"), 500)
            if not url or not title:
                continue
            if art.get("language") and art["language"] != "English":
                continue
            out.append(RawNewsItem(
                provider=self.key,
                source_name=str(art.get("domain") or self.profile.name),
                url=url,
                guid=url,
                title=title,
                summary="",
                published_at=parse_timestamp(art.get("seendate")),
                fetched_at=now,
            ))
        return out
