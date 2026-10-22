"""
Default free news sources (each URL and response format verified live on
2026-10-22) and the factory that builds providers from settings.

Latency honesty: everything here is polled.  Official feeds publish at
release time, so polling every 30-60s gives ~30-90s detection latency.
GDELT is an aggregator with a ~15 minute cadence and is classed "delayed" —
useful for corroboration and coverage, too slow to win a race on its own.
"""

from __future__ import annotations

from src.news_trader.config import NewsTraderSettings
from src.news_trader.sources.base import NewsProvider, SourceProfile
from src.news_trader.sources.bluesky import BlueskyAccount, BlueskyProvider
from src.news_trader.sources.gdelt import GdeltProvider
from src.news_trader.sources.rss import RssAtomProvider

RSS_PROFILES: dict[str, SourceProfile] = {p.key: p for p in [
    SourceProfile(
        "fed_press", "Federal Reserve (all press releases)",
        "https://www.federalreserve.gov/feeds/press_all.xml",
        "monetary_policy", 1.0, True, "poll_fast", 60, 45, ("federalreserve.gov",),
    ),
    SourceProfile(
        "fed_monetary", "Federal Reserve (monetary policy)",
        "https://www.federalreserve.gov/feeds/press_monetary.xml",
        "monetary_policy", 1.0, True, "poll_fast", 45, 30, ("federalreserve.gov",),
    ),
    SourceProfile(
        "sec_8k", "SEC EDGAR current 8-K filings",
        "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K"
        "&company=&dateb=&owner=include&start=0&count=40&output=atom",
        "corporate", 0.97, True, "poll_fast", 90, 60, ("sec.gov",),
        notes="SEC fair-access policy: declare a contact User-Agent; <=10 req/s.",
    ),
    SourceProfile(
        "bls_latest", "BLS major economic indicators",
        "https://www.bls.gov/feed/bls_latest.rss",
        "economics", 1.0, True, "poll_fast", 60, 45, ("bls.gov",),
    ),
    SourceProfile(
        "bea_releases", "BEA news releases",
        "https://apps.bea.gov/rss/rss.xml",
        "economics", 1.0, True, "poll_fast", 90, 60, ("bea.gov",),
    ),
    SourceProfile(
        "whitehouse", "White House news",
        "https://www.whitehouse.gov/news/feed/",
        "politics", 0.92, True, "poll", 180, 90, ("whitehouse.gov",),
    ),
    SourceProfile(
        "cnbc_top", "CNBC top news", "https://search.cnbc.com/rs/search/combinedcms/view.xml"
        "?partnerId=wrss01&id=100003114",
        "general", 0.80, False, "poll", 180, 60, ("cnbc.com",),
    ),
    SourceProfile(
        "marketwatch_top", "MarketWatch top stories",
        "https://feeds.content.dowjones.io/public/rss/mw_topstories",
        "general", 0.78, False, "poll", 180, 60, ("marketwatch.com",),
    ),
    SourceProfile(
        "bbc_world", "BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml",
        "general", 0.85, False, "poll", 300, 90, ("bbc.co.uk",),
    ),
    SourceProfile(
        "npr_news", "NPR News", "https://feeds.npr.org/1001/rss.xml",
        "general", 0.80, False, "poll", 300, 90, ("npr.org",),
    ),
]}

GDELT_PROFILE = SourceProfile(
    "gdelt_markets", "GDELT DOC 2.0 (aggregated)", "https://api.gdeltproject.org/api/v2/doc/doc",
    "general", 0.6, False, "delayed", 900, 300,
    notes="Aggregator; per-item credibility comes from the publishing domain. "
          "Rate limit: one request per 5 seconds.",
)
GDELT_QUERY = (
    '(acquisition OR resigns OR ceasefire OR indicted OR tariffs OR bankruptcy '
    'OR "rate cut" OR elected OR approves) sourcelang:english'
)

BLUESKY_PROFILE = SourceProfile(
    "bluesky_official", "Bluesky verified official/wire accounts",
    "https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed",
    "general", 0.85, False, "poll_fast", 60, 45,
    notes="Only verified accounts listed in BLUESKY_ACCOUNTS; reposts ignored.",
)
# Handles confirmed to exist with verifiedStatus == "valid" on 2026-10-22.
BLUESKY_ACCOUNTS = [
    BlueskyAccount("federalreserve.gov", "Federal Reserve (Bluesky)", 1.0, True, "monetary_policy"),
    BlueskyAccount("reuters.com", "Reuters (Bluesky)", 0.92),
    BlueskyAccount("apnews.com", "Associated Press (Bluesky)", 0.92),
    BlueskyAccount("bloomberg.com", "Bloomberg (Bluesky)", 0.88),
    BlueskyAccount("cnbc.com", "CNBC (Bluesky)", 0.80),
    BlueskyAccount("nytimes.com", "New York Times (Bluesky)", 0.85),
]

ALL_PROFILES: dict[str, SourceProfile] = {
    **RSS_PROFILES, GDELT_PROFILE.key: GDELT_PROFILE, BLUESKY_PROFILE.key: BLUESKY_PROFILE,
}

# Hosts that need request spacing (seconds) on the shared HTTP client.
HOST_MIN_INTERVAL = {
    "api.gdeltproject.org": 5.5,
    "www.sec.gov": 0.2,
}


def build_providers(settings: NewsTraderSettings) -> list[NewsProvider]:
    providers: list[NewsProvider] = []
    for key in settings.sources.enabled:
        if key in RSS_PROFILES:
            providers.append(RssAtomProvider(RSS_PROFILES[key]))
        elif key == GDELT_PROFILE.key:
            providers.append(GdeltProvider(GDELT_PROFILE, GDELT_QUERY))
        elif key == BLUESKY_PROFILE.key:
            providers.append(BlueskyProvider(BLUESKY_PROFILE, BLUESKY_ACCOUNTS))
        else:
            raise ValueError(f"unknown news source '{key}'")
    return providers
