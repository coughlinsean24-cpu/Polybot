"""News providers, ingestion resilience, normalisation and verification."""

import asyncio
import unittest

from src.news_trader.clock import ManualClock
from src.news_trader.http import HttpError, NetworkError, RateLimited, SchemaError
from src.news_trader.ingestion import IngestionService
from src.news_trader.models import RawNewsItem, Verification
from src.news_trader.normalize import Normalizer, canonical_url, classify_category
from src.news_trader.sources.base import SourceAssessment, parse_timestamp
from src.news_trader.sources.bluesky import BlueskyProvider
from src.news_trader.sources.gdelt import GdeltProvider
from src.news_trader.sources.registry import (
    BLUESKY_ACCOUNTS, BLUESKY_PROFILE, GDELT_PROFILE, GDELT_QUERY, RSS_PROFILES,
)
from src.news_trader.sources.rss import RssAtomProvider
from tests.news_fakes import T0, FakeHttp

RSS = """<?xml version="1.0"?><rss version="2.0"><channel><title>Fed</title>
<item><title>Federal Reserve issues FOMC statement</title>
<link>https://www.federalreserve.gov/newsevents/pressreleases/monetary20261028a.htm</link>
<guid>https://www.federalreserve.gov/newsevents/pressreleases/monetary20261028a.htm</guid>
<description>Federal Reserve issues FOMC statement</description><category>Monetary Policy</category>
<pubDate>Wed, 28 Oct 2026 18:00:00 GMT</pubDate></item>
<item><title></title><link></link></item>
</channel></rss>"""

SEC_ATOM = """<?xml version="1.0" encoding="ISO-8859-1" ?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>Latest Filings</title>
<entry><title>8-K - ACME CORP (0000123456) (Filer)</title>
<link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/123456/x-index.htm"/>
<summary type="html"> &lt;b&gt;Filed:&lt;/b&gt; 2026-10-22 &lt;b&gt;AccNo:&lt;/b&gt; 0000123456-26-000001 &lt;br&gt;Item 1.01: Entry into a Material Definitive Agreement&lt;br&gt;Item 9.01: Financial Statements and Exhibits</summary>
<updated>2026-10-22T12:07:27-04:00</updated><category scheme="https://www.sec.gov/" label="form type" term="8-K"/>
<id>urn:tag:sec.gov,2008:accession-number=0000123456-26-000001</id></entry></feed>"""

BSKY = {"feed": [
    {"post": {"uri": "at://did:plc:x/app.bsky.feed.post/abc", "author": {
        "handle": "reuters.com", "verification": {"verifiedStatus": "valid"}},
        "record": {"text": "Acme to buy Widget for $4 bln\nmore", "createdAt": "2026-10-22T15:00:00Z",
                   "embed": {"external": {"uri": "https://reut.rs/abc"}}},
        "indexedAt": "2026-10-22T15:00:05Z"}},
    {"post": {"uri": "at://did:plc:y/app.bsky.feed.post/def", "author": {"handle": "someone.else"},
              "record": {"text": "reposted"}, "indexedAt": "2026-10-22T15:00:05Z"},
     "reason": {"$type": "app.bsky.feed.defs#reasonRepost"}},
]}


def raw(title, provider="cnbc_top", guid=None, url=None, published=T0, fetched=T0, summary=""):
    return RawNewsItem(provider, provider, url or f"https://example.com/{abs(hash(title))}", guid or title,
                       title, summary, published, fetched)


def assess(key="cnbc.com", cred=0.8, primary=False, cat="general"):
    return SourceAssessment(key, cred, primary, cat)


class TestParsers(unittest.TestCase):
    def test_rss_parses_and_skips_empty_items(self):
        p = RssAtomProvider(RSS_PROFILES["fed_monetary"])
        items = p.parse(RSS, T0)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].published_at, parse_timestamp("Wed, 28 Oct 2026 18:00:00 GMT"))
        self.assertTrue(p.assess(items[0]).primary_official)

    def test_sec_atom_headline_includes_items(self):
        items = RssAtomProvider(RSS_PROFILES["sec_8k"]).parse(SEC_ATOM, T0)
        self.assertEqual(len(items), 1)
        self.assertIn("Acme Corp files 8-K", items[0].title)
        self.assertIn("Material Definitive Agreement", items[0].title)

    def test_malformed_xml_raises_schema_error(self):
        with self.assertRaises(SchemaError):
            RssAtomProvider(RSS_PROFILES["cnbc_top"]).parse("<rss><channel><item>", T0)
        with self.assertRaises(SchemaError):
            RssAtomProvider(RSS_PROFILES["cnbc_top"]).parse("<html><body>blocked</body></html>", T0)

    def test_gdelt_schema_change_and_empty(self):
        g = GdeltProvider(GDELT_PROFILE, GDELT_QUERY)
        self.assertEqual(g.parse({}, T0), [])
        with self.assertRaises(SchemaError):
            g.parse({"articles": "not-a-list"}, T0)
        items = g.parse({"articles": [{"url": "https://www.reuters.com/a", "title": "T", "seendate": "20261022T153000Z",
                                       "domain": "reuters.com", "language": "English"},
                                      {"url": "https://x.example/b", "title": "Foreign", "language": "French"}]}, T0)
        self.assertEqual(len(items), 1)
        self.assertAlmostEqual(g.assess(items[0]).credibility, 0.9)  # capped: aggregator, not primary
        self.assertFalse(g.assess(items[0]).primary_official)

    def test_bluesky_skips_reposts_and_requires_verification(self):
        b = BlueskyProvider(BLUESKY_PROFILE, BLUESKY_ACCOUNTS)
        items = b.parse("reuters.com", BSKY, T0)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].url, "https://reut.rs/abc")
        self.assertEqual(items[0].title, "Acme to buy Widget for $4 bln")
        self.assertGreater(b.assess(items[0]).credibility, 0.9)
        items[0].author_verified = False  # impersonation / lost verification
        self.assertLessEqual(b.assess(items[0]).credibility, 0.2)
        with self.assertRaises(SchemaError):
            b.parse("reuters.com", {"unexpected": True}, T0)

    def test_timestamp_formats(self):
        self.assertIsNotNone(parse_timestamp("Tue, 20 Oct 2026 10:02:35 -0400"))
        self.assertIsNotNone(parse_timestamp("2026-10-22T12:07:27-04:00"))
        self.assertIsNotNone(parse_timestamp("20261022T153000Z"))
        self.assertIsNone(parse_timestamp("not a date"))
        self.assertIsNone(parse_timestamp(None))


class TestIngestionResilience(unittest.IsolatedAsyncioTestCase):
    def make(self, routes):
        clock = ManualClock(T0)
        providers = [RssAtomProvider(RSS_PROFILES["fed_monetary"]), RssAtomProvider(RSS_PROFILES["cnbc_top"])]
        q = asyncio.Queue()
        return IngestionService(providers, FakeHttp(routes), Normalizer(), clock, q), clock, q

    async def test_one_provider_failure_does_not_affect_others(self):
        ing, clock, q = self.make({"federalreserve.gov": RSS, "cnbc.com": NetworkError("connection reset")})
        for p in ing.providers.values():
            await ing.poll_once(p)
        self.assertEqual(ing.health["fed_monetary"].state, "ok")
        self.assertEqual(ing.health["cnbc_top"].state, "degraded")
        self.assertEqual(q.qsize(), 1)
        self.assertGreater(ing.health["cnbc_top"].next_poll_at, clock.now() + 100)  # backed off

    async def test_rate_limit_respects_retry_after(self):
        ing, clock, _ = self.make({"federalreserve.gov": RateLimited("429", retry_after=900), "cnbc.com": RSS})
        await ing.poll_once(ing.providers["fed_monetary"])
        h = ing.health["fed_monetary"]
        self.assertEqual(h.state, "rate_limited")
        self.assertGreaterEqual(h.next_poll_at - clock.now(), 900)

    async def test_repeated_failures_mark_down_and_recover(self):
        state = {"fail": True}

        def route(url, params):
            return HttpError("503", 503) if state["fail"] else RSS
        ing, clock, q = self.make({"federalreserve.gov": route, "cnbc.com": RSS})
        p = ing.providers["fed_monetary"]
        for _ in range(5):
            await ing.poll_once(p)
        self.assertEqual(ing.health["fed_monetary"].state, "down")
        state["fail"] = False
        await ing.poll_once(p)
        self.assertEqual(ing.health["fed_monetary"].state, "ok")
        self.assertEqual(ing.health["fed_monetary"].consecutive_failures, 0)

    async def test_parser_crash_is_contained(self):
        ing, _, _ = self.make({"federalreserve.gov": "<rss><channel><item><title>x</title><link>y</link>"
                                                     "<pubDate>bogus</pubDate></item></channel></rss>",
                               "cnbc.com": RSS})

        class Boom(RssAtomProvider):
            def parse(self, body, now):
                raise RuntimeError("bug")
        ing.providers["cnbc_top"] = Boom(RSS_PROFILES["cnbc_top"])
        await ing.poll_once(ing.providers["cnbc_top"])
        self.assertIn("unexpected RuntimeError", ing.health["cnbc_top"].last_error)

    async def test_repeat_polls_do_not_reemit(self):
        ing, _, q = self.make({"federalreserve.gov": RSS, "cnbc.com": RSS})
        p = ing.providers["fed_monetary"]
        await ing.poll_once(p)
        await ing.poll_once(p)
        self.assertEqual(q.qsize(), 1)


class TestNormalizer(unittest.TestCase):
    def test_url_canonicalisation(self):
        self.assertEqual(canonical_url("https://www.BBC.co.uk/news/x/?at_medium=RSS&at_campaign=rss#0"),
                         canonical_url("https://bbc.co.uk/news/x"))

    def test_near_duplicates_cluster_and_corroborate(self):
        n = Normalizer()
        a = n.process(raw("Acme agrees to acquire Widget for $4 billion", url="https://cnbc.com/a"),
                      assess("cnbc.com", 0.8), T0, True)
        self.assertEqual(a.event.verification, Verification.SINGLE_SOURCE)
        b = n.process(raw("Acme agrees to acquire Widget in $4 billion deal", provider="bluesky_official",
                          url="https://reut.rs/z"), assess("reuters.com", 0.92), T0 + 60, True)
        self.assertTrue(b.event.is_duplicate)
        self.assertEqual(b.event.cluster_id, a.event.cluster_id)
        self.assertEqual(b.event.first_seen_at, T0)  # age measured from first sighting
        self.assertEqual(b.event.verification, Verification.CORROBORATED)
        self.assertIsNotNone(b.upgraded_leader)  # leader re-emitted for re-evaluation
        self.assertEqual(b.upgraded_leader.event_id, a.event.event_id)

    def test_same_outlet_twice_is_not_corroboration(self):
        n = Normalizer()
        n.process(raw("Acme agrees to acquire Widget", url="https://cnbc.com/a"), assess("cnbc.com"), T0, True)
        r = n.process(raw("Acme agrees to acquire Widget", provider="gdelt_markets", url="https://www.cnbc.com/a/"),
                      assess("cnbc.com"), T0 + 5, False)
        self.assertTrue(r.event.is_duplicate)
        self.assertEqual(r.event.verification, Verification.SINGLE_SOURCE)
        self.assertIsNone(r.upgraded_leader)

    def test_denial_creates_conflict(self):
        n = Normalizer()
        n.process(raw("Acme to acquire Widget, sources say"), assess("cnbc.com"), T0, True)
        r = n.process(raw("Acme denies report it will acquire Widget", provider="bbc_world"),
                      assess("bbc.co.uk", 0.85), T0 + 30, True)
        self.assertEqual(r.event.verification, Verification.CONFLICTING)
        self.assertIn("conflict", r.event.flags)

    def test_unverified_low_credibility_source(self):
        n = Normalizer(min_credibility=0.6)
        r = n.process(raw("Shocking: Widget CEO arrested"), assess("randomblog.net", 0.4), T0, False)
        self.assertEqual(r.event.verification, Verification.UNVERIFIED)

    def test_primary_source_and_flags(self):
        n = Normalizer()
        r = n.process(raw("Acme reportedly in talks to buy Widget", published=None),
                      assess("sec.gov", 0.97, primary=True, cat="corporate"), T0, True)
        self.assertEqual(r.event.verification, Verification.PRIMARY_SOURCE)
        self.assertIn("speculative", r.event.flags)
        self.assertIn("missing_timestamp", r.event.flags)
        future = n.process(raw("Different story entirely about tariffs", published=T0 + 3600),
                           assess(), T0, True)
        self.assertIn("future_timestamp", future.event.flags)
        self.assertEqual(future.event.reference_time, T0)  # never in the future

    def test_categories(self):
        self.assertEqual(classify_category("Fed cuts interest rates by 25 basis points", "general"), "monetary_policy")
        self.assertEqual(classify_category("Israel and Hamas agree to ceasefire", "general"), "geopolitics")
        self.assertEqual(classify_category("A pleasant story about waterfalls", "general"), "general")


if __name__ == "__main__":
    unittest.main()
