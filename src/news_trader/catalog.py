"""
Continuously refreshed catalog of active, eligible Polymarket markets.

Source: Gamma ``GET /markets`` (closed=false, include_tag=true, ordered by
24h volume, server-side liquidity floor), paginated 100 at a time.  The
response shape was verified live: list fields such as ``outcomes``,
``outcomePrices`` and ``clobTokenIds`` arrive as JSON-encoded strings.

Records are validated with pydantic.  A market that fails validation is
skipped and counted; if most of a page fails, the refresh aborts and the
previous catalog stays in place (an API schema change must not silently
empty or corrupt the catalog).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src.news_trader import text as T
from src.news_trader.clock import Clock
from src.news_trader.config import CatalogSettings
from src.news_trader.http import HttpClient, SchemaError
from src.news_trader.models import MarketRecord
from src.news_trader.sources.base import parse_timestamp

logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"

# Gamma tag slug -> engine category.  First match in tag order wins.
TAG_CATEGORY: dict[str, str] = {
    "fed": "monetary_policy", "fed-rates": "monetary_policy", "fomc": "monetary_policy",
    "interest-rates": "monetary_policy", "jerome-powell": "monetary_policy",
    "economy": "economics", "inflation": "economics", "cpi": "economics", "jobs": "economics",
    "gdp": "economics", "recession": "economics", "economic-policy": "economics",
    "business": "corporate", "stocks": "corporate", "earnings": "corporate", "tech": "corporate",
    "big-tech": "corporate", "ipos": "corporate", "ipo": "corporate", "companies": "corporate",
    "mergers": "corporate", "ai": "corporate",
    "crypto": "crypto", "bitcoin": "crypto", "ethereum": "crypto", "crypto-prices": "crypto",
    "solana": "crypto",
    "sports": "sports", "soccer": "sports", "nba": "sports", "nfl": "sports", "mlb": "sports",
    "nhl": "sports", "tennis": "sports", "esports": "sports", "games": "sports", "ufc": "sports",
    "geopolitics": "geopolitics", "world": "geopolitics", "middle-east": "geopolitics",
    "ukraine": "geopolitics", "russia": "geopolitics", "israel": "geopolitics", "iran": "geopolitics",
    "china": "geopolitics", "gaza": "geopolitics",
    "politics": "politics", "elections": "politics", "us-presidential-election": "politics",
    "midterms": "politics", "trump": "politics", "congress": "politics",
    "global-elections": "politics", "world-elections": "politics", "us-politics": "politics",
}
_CATEGORY_PRIORITY = ("monetary_policy", "economics", "corporate", "crypto", "sports",
                      "geopolitics", "politics")


def _json_list(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError as e:
            raise ValueError(f"not a JSON list: {v[:60]}") from e
    return v


def _float_or_none(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


class GammaTag(BaseModel):
    model_config = ConfigDict(extra="ignore")
    slug: str = ""
    label: str = ""


class GammaEventStub(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = ""
    title: str = ""
    slug: str = ""


class GammaFeeSchedule(BaseModel):
    model_config = ConfigDict(extra="ignore")
    rate: float = 0.0
    exponent: float = 1.0
    takerOnly: bool = True


class GammaMarket(BaseModel):
    """The subset of the Gamma market schema the engine relies on."""
    model_config = ConfigDict(extra="ignore")

    id: str
    question: str
    conditionId: str
    slug: str = ""
    description: str = ""
    resolutionSource: str = ""
    endDate: str | None = None
    outcomes: list[str]
    clobTokenIds: list[str]
    active: bool = False
    closed: bool = True
    archived: bool = False
    acceptingOrders: bool = False
    enableOrderBook: bool = False
    negRisk: bool = False
    liquidityNum: float | None = None
    volume24hr: float | None = None
    bestBid: float | None = None
    bestAsk: float | None = None
    lastTradePrice: float | None = None
    oneHourPriceChange: float | None = None
    oneDayPriceChange: float | None = None
    orderPriceMinTickSize: float | None = None
    orderMinSize: float | None = None
    feesEnabled: bool = False
    feeSchedule: GammaFeeSchedule | None = None
    umaResolutionStatus: str = ""
    groupItemTitle: str = ""
    tags: list[GammaTag] = Field(default_factory=list)
    events: list[GammaEventStub] = Field(default_factory=list)

    @field_validator("outcomes", "clobTokenIds", mode="before")
    @classmethod
    def _decode_list(cls, v: Any) -> Any:
        return _json_list(v)

    @field_validator("liquidityNum", "volume24hr", "bestBid", "bestAsk", "lastTradePrice",
                     "oneHourPriceChange", "oneDayPriceChange", "orderPriceMinTickSize", "orderMinSize", mode="before")
    @classmethod
    def _number(cls, v: Any) -> float | None:
        return _float_or_none(v)

    @field_validator("id", "conditionId", mode="before")
    @classmethod
    def _str_id(cls, v: Any) -> str:
        if v is None or v == "":
            raise ValueError("missing id")
        return str(v)


def categorize_tags(tag_slugs: list[str]) -> str:
    cats = {TAG_CATEGORY[s] for s in tag_slugs if s in TAG_CATEGORY}
    for c in _CATEGORY_PRIORITY:
        if c in cats:
            return c
    return "general"


def to_record(m: GammaMarket) -> MarketRecord | None:
    """Convert a validated Gamma market; None if it is not a binary CLOB market."""
    if len(m.outcomes) != 2 or len(m.clobTokenIds) != 2 or not all(m.clobTokenIds):
        return None
    tag_slugs = [t.slug for t in m.tags if t.slug]
    ev = m.events[0] if m.events else GammaEventStub()
    fee_rate = m.feeSchedule.rate if (m.feesEnabled and m.feeSchedule) else 0.0
    return MarketRecord(
        market_id=m.id,
        condition_id=m.conditionId,
        question=m.question.strip(),
        description=m.description or "",
        slug=m.slug,
        event_id=ev.id,
        event_title=ev.title,
        event_slug=ev.slug,
        tags=tag_slugs,
        category=categorize_tags(tag_slugs),
        outcomes=list(m.outcomes),
        token_ids=list(m.clobTokenIds),
        end_ts=parse_timestamp(m.endDate),
        active=m.active,
        closed=m.closed,
        archived=m.archived,
        accepting_orders=m.acceptingOrders,
        enable_order_book=m.enableOrderBook,
        neg_risk=m.negRisk,
        liquidity=m.liquidityNum or 0.0,
        volume_24h=m.volume24hr or 0.0,
        best_bid=m.bestBid,
        best_ask=m.bestAsk,
        last_trade_price=m.lastTradePrice,
        tick_size=m.orderPriceMinTickSize or 0.01,
        min_order_size=m.orderMinSize or 5.0,
        fees_enabled=m.feesEnabled,
        fee_rate=fee_rate,
        resolution_source=m.resolutionSource or "",
        uma_resolution_status=m.umaResolutionStatus or "",
        group_item_title=m.groupItemTitle or "",
        price_change_1h=m.oneHourPriceChange,
        price_change_24h=m.oneDayPriceChange,
    )


def parse_market_page(payload: Any) -> tuple[list[MarketRecord], int]:
    """Parse one Gamma /markets page. Returns (records, invalid_count)."""
    if not isinstance(payload, list):
        raise SchemaError(f"Gamma /markets returned {type(payload).__name__}, expected list")
    records, invalid = [], 0
    for raw in payload:
        try:
            rec = to_record(GammaMarket.model_validate(raw))
        except ValidationError:
            invalid += 1
            continue
        if rec is not None:
            records.append(rec)
    return records, invalid


def market_tradeable(m: MarketRecord, now: float, min_time_to_close_seconds: float) -> tuple[bool, str]:
    """Deterministic status gate.  Expired or suspended markets never trade."""
    if m.closed or m.archived:
        return False, "market closed/archived"
    if not m.active:
        return False, "market inactive"
    if not m.accepting_orders or not m.enable_order_book:
        return False, "market not accepting orders (suspended or paused)"
    if m.end_ts is None:
        return False, "market has no end date"
    if m.end_ts - now < min_time_to_close_seconds:
        return False, f"market closes in {(m.end_ts - now) / 3600:.1f}h"
    if m.uma_resolution_status.lower() in ("proposed", "disputed", "resolved", "challenged"):
        return False, f"market already in resolution ({m.uma_resolution_status})"
    return True, "ok"


@dataclass
class CatalogStats:
    markets: int
    invalid: int
    pages: int
    refreshed_at: float
    duration_seconds: float


class MarketIndex:
    """Inverted token index with IDF weights over market search text."""

    def __init__(self, markets: list[MarketRecord]):
        self.postings: dict[str, set[str]] = {}
        self.doc_tokens: dict[str, set[str]] = {}
        self.doc_entities: dict[str, set[str]] = {}
        for m in markets:
            toks = T.token_set(m.search_text)
            self.doc_tokens[m.market_id] = toks
            self.doc_entities[m.market_id] = T.entities(m.search_text)
            for t in toks:
                self.postings.setdefault(t, set()).add(m.market_id)
        n = max(1, len(markets))
        self.idf = {t: math.log(1 + n / len(ids)) for t, ids in self.postings.items()}
        self.max_idf = math.log(1 + n)

    def weight(self, token: str) -> float:
        return self.idf.get(token, self.max_idf)


class MarketCatalog:
    def __init__(self, http: HttpClient, clock: Clock, settings: CatalogSettings):
        self.http = http
        self.clock = clock
        self.settings = settings
        self.markets: dict[str, MarketRecord] = {}
        self.by_token: dict[str, MarketRecord] = {}
        self.index = MarketIndex([])
        self.last_stats: CatalogStats | None = None
        self.last_error: str = ""

    @property
    def ready(self) -> bool:
        return bool(self.markets)

    def age(self) -> float | None:
        return None if self.last_stats is None else self.clock.now() - self.last_stats.refreshed_at

    async def refresh(self) -> CatalogStats:
        started = self.clock.now()
        end_min = datetime.fromtimestamp(started, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        records: dict[str, MarketRecord] = {}
        invalid_total, pages = 0, 0
        offset = 0
        while offset < self.settings.max_markets:
            payload = await self.http.get_json(f"{GAMMA_API}/markets", params={
                "closed": "false",
                "limit": str(self.settings.page_size),
                "offset": str(offset),
                "order": "volume24hr",
                "ascending": "false",
                "include_tag": "true",
                "liquidity_num_min": str(self.settings.min_liquidity),
                "end_date_min": end_min,
            })
            page, invalid = parse_market_page(payload)
            pages += 1
            if payload and invalid > 0.5 * len(payload) and len(payload) >= 20:
                self.last_error = f"schema check failed: {invalid}/{len(payload)} markets invalid"
                raise SchemaError(f"Gamma /markets {self.last_error}; keeping previous catalog")
            invalid_total += invalid
            for rec in page:
                if rec.volume_24h >= self.settings.min_volume_24h:
                    records[rec.market_id] = rec
            if len(payload) < self.settings.page_size:
                break
            offset += self.settings.page_size

        self.markets = records
        self.by_token = {t: m for m in records.values() for t in m.token_ids}
        self.index = MarketIndex(list(records.values()))
        self.last_error = ""
        self.last_stats = CatalogStats(len(records), invalid_total, pages,
                                       self.clock.now(), self.clock.now() - started)
        logger.info("market catalog refreshed: %d markets (%d invalid) in %.1fs",
                    len(records), invalid_total, self.last_stats.duration_seconds)
        return self.last_stats

    async def fetch_markets_status(self, condition_ids: list[str]) -> dict[str, dict]:
        """Raw Gamma records (open, closed or resolved) keyed by condition id.

        Gamma accepts up to 100 repeated ``condition_ids`` per request.
        """
        out: dict[str, dict] = {}
        for i in range(0, len(condition_ids), 100):
            chunk = condition_ids[i:i + 100]
            params = [("condition_ids", c) for c in chunk] + [("closed", "any")]
            payload = await self.http.get_json(f"{GAMMA_API}/markets", params=params)
            if not isinstance(payload, list):
                raise SchemaError("Gamma /markets by condition_ids did not return a list")
            for raw in payload:
                if isinstance(raw, dict) and raw.get("conditionId") in chunk:
                    out[raw["conditionId"]] = raw
        return out
