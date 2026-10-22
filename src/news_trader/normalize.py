"""
Event normalisation, deduplication, corroboration and verification.

Each provider poll returns overlapping items, and the same story arrives from
several outlets.  The normaliser:

  * drops exact repeats (same provider + guid) silently;
  * clusters near-duplicates (same canonical URL, or similar headline within a
    time window) so one story produces one tradeable event;
  * counts *independent* organisations per cluster for corroboration;
  * flags conflicts (a denial/retraction alongside affirming reports);
  * assigns a category and a verification status.

When a later item upgrades a cluster's verification (e.g. a second wire
service confirms), the cluster leader is re-emitted so the engine can
re-evaluate it — with its age still measured from the cluster's first sighting.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from src.news_trader import text as T
from src.news_trader.models import NewsEvent, RawNewsItem, Verification
from src.news_trader.sources.base import SourceAssessment, registrable_domain

# Category keyword sets operate on alias-normalised tokens from text.tokens().
_RAW_CATEGORY_KEYWORDS: dict[str, set[str]] = {
    "monetary_policy": {"fed", "rates", "bps", "ecb", "boj",
                        "central", "monetary", "discount", "yield", "treasury"},
    "economics": {"cpi", "payrolls", "unemployment", "gdp", "retail", "pce", "ppi",
                  "housing", "deficit", "jobless", "indicator", "economy", "economic",
                  "wage", "consumer", "manufacturing"},
    "corporate": {"acquisition", "ipo", "bankruptcy", "ceo", "earnings", "revenue",
                  "shares", "stock", "8-k", "quarterly", "dividend", "layoff", "company",
                  "inc", "corp", "deal", "files"},
    "politics": {"election", "senate", "congress", "president", "trump", "governor",
                 "vote", "poll", "impeach", "nominee", "primary", "campaign", "scotus",
                 "bill", "legislation", "shutdown", "democrat", "republican", "whitehouse",
                 "cabinet", "resign", "executive", "order", "midterm", "speaker"},
    "geopolitics": {"war", "ceasefire", "invasion", "missile", "strike", "sanction",
                    "troop", "nato", "israel", "iran", "ukraine", "russia", "china",
                    "taiwan", "gaza", "hamas", "military", "attack", "nuclear", "summit"},
    "crypto": {"btc", "eth", "crypto", "solana", "stablecoin", "coinbase", "binance",
               "blockchain", "token", "defi"},
    "sports": {"nfl", "nba", "mlb", "nhl", "fifa", "cup", "league", "playoff", "match",
               "tournament", "championship", "coach", "season", "game", "esport"},
}
# Keywords pass through the same stemmer as event text so they compare equal.
CATEGORY_KEYWORDS: dict[str, set[str]] = {
    cat: {T.stem(k) for k in kws} for cat, kws in _RAW_CATEGORY_KEYWORDS.items()
}

_TRACKING_PARAMS = re.compile(r"^(utm_|at_|mod$|ref$|cmpid$|src$|smid$|guccounter)")


def canonical_url(url: str) -> str:
    try:
        p = urlparse(url.strip())
    except ValueError:
        return url.strip()
    query = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=False)
             if not _TRACKING_PARAMS.match(k.lower())]
    host = p.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = p.path.rstrip("/") or "/"
    return urlunparse((p.scheme.lower() or "https", host, path, "", urlencode(query), ""))


def classify_category(text: str, default: str) -> str:
    toks = set(T.tokens(text))
    best, best_hits = default, 0
    for cat, kws in CATEGORY_KEYWORDS.items():
        hits = len(toks & kws)
        if hits > best_hits or (hits == best_hits and hits > 0 and cat == default):
            best, best_hits = cat, hits
    return best


def make_event_id(provider: str, guid: str) -> str:
    return hashlib.sha1(f"{provider}|{guid}".encode()).hexdigest()[:16]


@dataclass
class _Cluster:
    cluster_id: str
    first_seen: float
    leader: NewsEvent
    shingles: set[str]
    tokens: set[str]
    sources: dict[str, float] = field(default_factory=dict)  # independence key -> best credibility
    primary: bool = False
    denial_seen: bool = False
    affirm_seen: bool = False
    verification: Verification = Verification.UNVERIFIED


@dataclass
class NormalizeResult:
    event: NewsEvent
    # Set when this item upgraded the cluster's verification; the engine
    # should re-evaluate the (updated) leader event.
    upgraded_leader: NewsEvent | None = None


class Normalizer:
    def __init__(self, min_credibility: float = 0.6, window_seconds: float = 6 * 3600,
                 shingle_threshold: float = 0.5, token_threshold: float = 0.6,
                 denial_token_threshold: float = 0.35):
        self.min_credibility = min_credibility
        self.window_seconds = window_seconds
        self.shingle_threshold = shingle_threshold
        self.token_threshold = token_threshold
        self.denial_token_threshold = denial_token_threshold
        self._seen_ids: dict[str, float] = {}
        self._url_to_cluster: dict[str, str] = {}
        self._clusters: dict[str, _Cluster] = {}

    def process(self, raw: RawNewsItem, assessment: SourceAssessment, now: float,
                is_realtime_source: bool) -> NormalizeResult | None:
        """Normalise one item.  Returns None for an exact repeat."""
        self._evict(now)
        event_id = make_event_id(raw.provider, raw.guid)
        if event_id in self._seen_ids:
            return None
        self._seen_ids[event_id] = now

        url = canonical_url(raw.url)
        text = f"{raw.title}. {raw.summary}"
        denial = T.is_denial(raw.title)
        flags: list[str] = []
        if raw.published_at is None:
            flags.append("missing_timestamp")
        elif raw.published_at > raw.fetched_at + 120:
            flags.append("future_timestamp")
        if T.is_speculative(raw.title):
            flags.append("speculative")
        if denial:
            flags.append("denial")

        event = NewsEvent(
            event_id=event_id,
            cluster_id=event_id,
            provider=raw.provider,
            source_name=raw.source_name,
            source_domain=registrable_domain(urlparse(url).netloc),
            url=raw.url,
            title=raw.title,
            summary=raw.summary,
            category=classify_category(text, assessment.default_category),
            credibility=assessment.credibility,
            published_at=raw.published_at,
            first_seen_at=raw.fetched_at,
            ingested_at=now,
            is_realtime_source=is_realtime_source,
            is_primary_source=assessment.primary_official,
            flags=flags,
        )

        cluster = self._find_cluster(url, raw.title, denial)
        if cluster is None:
            cluster = _Cluster(
                cluster_id=event_id, first_seen=raw.fetched_at, leader=event,
                shingles=T.shingles(raw.title), tokens=T.token_set(raw.title),
            )
            self._clusters[cluster.cluster_id] = cluster
            prev_verification = None
        else:
            prev_verification = cluster.verification
            event.is_duplicate = True
            event.duplicate_of = cluster.leader.event_id
            event.cluster_id = cluster.cluster_id
            event.first_seen_at = cluster.first_seen
            event.category = cluster.leader.category
        self._url_to_cluster[url] = cluster.cluster_id

        if assessment.credibility >= cluster.sources.get(assessment.independence_key, 0.0):
            cluster.sources[assessment.independence_key] = assessment.credibility
        cluster.primary = cluster.primary or (assessment.primary_official and assessment.credibility >= 0.9)
        cluster.denial_seen = cluster.denial_seen or denial
        cluster.affirm_seen = cluster.affirm_seen or not denial
        cluster.verification = self._verify(cluster)

        for ev in (event, cluster.leader):
            ev.verification = cluster.verification
            ev.corroborating_sources = sorted(cluster.sources)
            if cluster.verification == Verification.CONFLICTING and "conflict" not in ev.flags:
                ev.flags.append("conflict")

        upgraded = None
        if prev_verification is not None and cluster.verification != prev_verification \
                and cluster.verification in (Verification.CORROBORATED, Verification.PRIMARY_SOURCE,
                                             Verification.CONFLICTING):
            upgraded = cluster.leader
        return NormalizeResult(event, upgraded)

    def _verify(self, c: _Cluster) -> Verification:
        if c.denial_seen and c.affirm_seen:
            return Verification.CONFLICTING
        if c.primary:
            return Verification.PRIMARY_SOURCE
        credible = [k for k, cred in c.sources.items() if cred >= self.min_credibility]
        if len(credible) >= 2:
            return Verification.CORROBORATED
        if credible:
            return Verification.SINGLE_SOURCE
        return Verification.UNVERIFIED

    def _find_cluster(self, url: str, title: str, denial: bool) -> _Cluster | None:
        cid = self._url_to_cluster.get(url)
        if cid and cid in self._clusters:
            return self._clusters[cid]
        sh = T.shingles(title)
        tk = T.token_set(title)
        best, best_score = None, 0.0
        for c in self._clusters.values():
            s_sh = T.jaccard(sh, c.shingles)
            s_tk = T.jaccard(tk, c.tokens)
            hit = s_sh >= self.shingle_threshold or s_tk >= self.token_threshold
            # Denials rarely share phrasing with the claim they deny.
            if not hit and denial != (c.denial_seen and not c.affirm_seen):
                hit = s_tk >= self.denial_token_threshold
            score = max(s_sh, s_tk)
            if hit and score > best_score:
                best, best_score = c, score
        return best

    def cluster_first_seen(self, cluster_id: str) -> float | None:
        c = self._clusters.get(cluster_id)
        return c.first_seen if c else None

    def _evict(self, now: float) -> None:
        cutoff = now - self.window_seconds
        stale = [cid for cid, c in self._clusters.items() if c.first_seen < cutoff]
        for cid in stale:
            del self._clusters[cid]
        if stale:
            self._url_to_cluster = {u: c for u, c in self._url_to_cluster.items() if c in self._clusters}
        # Keep exact-id memory longer than clusters: feeds re-serve old items for days.
        id_cutoff = now - 7 * 24 * 3600
        if len(self._seen_ids) > 50_000:
            self._seen_ids = {k: t for k, t in self._seen_ids.items() if t >= id_cutoff}
