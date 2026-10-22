"""
Core typed records shared across the news-trading pipeline.

Plain dataclasses are used for internal records (matching the rest of the
repo); boundary data from external APIs and the LLM is validated with
pydantic in the module that parses it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Verification(str, Enum):
    PRIMARY_SOURCE = "PRIMARY_SOURCE"  # official issuer of the information
    CORROBORATED = "CORROBORATED"      # >= 2 independent credible sources
    SINGLE_SOURCE = "SINGLE_SOURCE"    # one credible secondary source
    UNVERIFIED = "UNVERIFIED"          # low-credibility or unknown source
    CONFLICTING = "CONFLICTING"        # credible sources disagree / denial seen


class Action(str, Enum):
    BUY_YES = "BUY_YES"
    BUY_NO = "BUY_NO"
    NO_TRADE = "NO_TRADE"


@dataclass
class RawNewsItem:
    """One item exactly as a provider returned it (after format parsing)."""
    provider: str
    source_name: str
    url: str
    guid: str
    title: str
    summary: str
    published_at: float | None  # epoch seconds from the source, if given
    fetched_at: float
    raw_category: str = ""
    author_handle: str = ""
    author_verified: bool = False


@dataclass
class NewsEvent:
    event_id: str
    cluster_id: str
    provider: str
    source_name: str
    source_domain: str
    url: str
    title: str
    summary: str
    category: str
    credibility: float
    published_at: float | None
    first_seen_at: float      # first time ANY source in this cluster was seen
    ingested_at: float        # when this specific item was normalised
    is_realtime_source: bool
    is_primary_source: bool
    is_duplicate: bool = False
    duplicate_of: str | None = None
    verification: Verification = Verification.UNVERIFIED
    corroborating_sources: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def reference_time(self) -> float:
        """Earliest time the information is known to have been public."""
        if self.published_at is not None:
            return min(self.published_at, self.first_seen_at)
        return self.first_seen_at

    def age(self, now: float) -> float:
        return max(0.0, now - self.reference_time)

    @property
    def text(self) -> str:
        return f"{self.title}. {self.summary}".strip()


@dataclass
class MarketRecord:
    """An active, binary Polymarket market from the Gamma catalog."""
    market_id: str
    condition_id: str
    question: str
    description: str
    slug: str
    event_id: str
    event_title: str
    event_slug: str
    tags: list[str]
    category: str
    outcomes: list[str]          # [yes_label, no_label]
    token_ids: list[str]         # [yes_token, no_token]
    end_ts: float | None
    active: bool
    closed: bool
    archived: bool
    accepting_orders: bool
    enable_order_book: bool
    neg_risk: bool
    liquidity: float
    volume_24h: float
    best_bid: float | None
    best_ask: float | None
    last_trade_price: float | None
    tick_size: float
    min_order_size: float
    fees_enabled: bool
    fee_rate: float
    resolution_source: str
    uma_resolution_status: str
    group_item_title: str = ""
    price_change_1h: float | None = None
    price_change_24h: float | None = None

    @property
    def yes_token(self) -> str:
        return self.token_ids[0]

    @property
    def no_token(self) -> str:
        return self.token_ids[1]

    @property
    def search_text(self) -> str:
        parts = [self.question, self.event_title, self.group_item_title]
        return " ".join(p for p in parts if p)


@dataclass(frozen=True)
class BookLevel:
    price: float
    size: float  # shares


@dataclass
class OrderBook:
    token_id: str
    bids: list[BookLevel]   # best (highest) first
    asks: list[BookLevel]   # best (lowest) first
    exchange_ts: float | None  # exchange timestamp, epoch seconds
    fetched_at: float
    tick_size: float = 0.01
    min_order_size: float = 5.0
    neg_risk: bool = False

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    def age(self, now: float) -> float:
        """Quote age, using the older of exchange and fetch timestamps."""
        ref = self.fetched_at
        if self.exchange_ts is not None:
            ref = min(ref, self.exchange_ts)
        return max(0.0, now - ref)


@dataclass
class ResolutionCheck:
    ok: bool
    ambiguity_score: float
    resolution_source: str
    has_explicit_rule: bool
    deadline_text: str
    issues: list[str] = field(default_factory=list)


@dataclass
class MarketMatch:
    market: MarketRecord
    score: float
    matched_terms: list[str]
    reasons: list[str]
    resolution: ResolutionCheck | None = None


@dataclass
class Rejection:
    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"
