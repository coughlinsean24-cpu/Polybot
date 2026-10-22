"""
CLOB order books, depth walking and fee maths.

``GET https://clob.polymarket.com/book?token_id=...`` (verified live) returns
bids sorted ascending and asks sorted descending — the best price is the LAST
element of each list — with string prices/sizes, a millisecond ``timestamp``,
``tick_size``, ``min_order_size`` and ``neg_risk``.  We normalise to
best-first lists and never trust list position.

Fees (docs.polymarket.com/trading/fees): takers pay
``fee = shares * feeRate * p * (1 - p)``, rounded to 5 decimals; makers pay
nothing.  We model every entry as a taker fill (we cross the spread to act on
news), which is the conservative assumption.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from src.news_trader.clock import Clock
from src.news_trader.http import HttpClient, SchemaError
from src.news_trader.models import BookLevel, OrderBook

CLOB_API = "https://clob.polymarket.com"


class _RawLevel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    price: float
    size: float


class _RawBook(BaseModel):
    model_config = ConfigDict(extra="ignore")
    asset_id: str
    bids: list[_RawLevel] = []
    asks: list[_RawLevel] = []
    timestamp: float | None = None
    tick_size: float | None = None
    min_order_size: float | None = None
    neg_risk: bool = False


def parse_book(payload: Any, fetched_at: float) -> OrderBook:
    try:
        raw = _RawBook.model_validate(payload)
    except ValidationError as e:
        raise SchemaError(f"CLOB /book schema mismatch: {e.errors()[:2]}") from e

    def clean(levels: list[_RawLevel]) -> list[BookLevel]:
        return [BookLevel(lv.price, lv.size) for lv in levels
                if 0.0 < lv.price < 1.0 and lv.size > 0 and math.isfinite(lv.size)]

    bids = sorted(clean(raw.bids), key=lambda lv: -lv.price)
    asks = sorted(clean(raw.asks), key=lambda lv: lv.price)
    exchange_ts = raw.timestamp / 1000.0 if raw.timestamp and raw.timestamp > 1e11 else raw.timestamp
    return OrderBook(
        token_id=raw.asset_id,
        bids=bids,
        asks=asks,
        exchange_ts=exchange_ts,
        fetched_at=fetched_at,
        tick_size=raw.tick_size or 0.01,
        min_order_size=raw.min_order_size or 5.0,
        neg_risk=raw.neg_risk,
    )


class OrderBookClient:
    def __init__(self, http: HttpClient, clock: Clock):
        self.http = http
        self.clock = clock

    async def get_book(self, token_id: str) -> OrderBook:
        payload = await self.http.get_json(f"{CLOB_API}/book", params={"token_id": token_id})
        if isinstance(payload, dict) and payload.get("error"):
            raise SchemaError(f"CLOB /book error for {token_id[:12]}…: {payload['error']}")
        return parse_book(payload, self.clock.now())


def taker_fee(shares: float, price: float, fee_rate: float) -> float:
    if fee_rate <= 0 or shares <= 0:
        return 0.0
    return round(shares * fee_rate * price * (1.0 - price), 5)


def round_to_tick(price: float, tick: float, direction: str = "down") -> float:
    """Round a limit price onto the tick grid (down for buys = never overpay)."""
    steps = price / tick
    steps = math.floor(steps + 1e-9) if direction == "down" else math.ceil(steps - 1e-9)
    decimals = max(0, -int(math.floor(math.log10(tick)))) if tick < 1 else 0
    return round(steps * tick, decimals)


@dataclass
class FillEstimate:
    shares: float = 0.0
    notional: float = 0.0       # sum(price * shares), excluding fees
    fees: float = 0.0
    levels: list[tuple[float, float]] = field(default_factory=list)  # (price, shares)
    depth_exhausted: bool = False

    @property
    def cost(self) -> float:
        return self.notional + self.fees

    @property
    def avg_price(self) -> float | None:
        return self.notional / self.shares if self.shares > 0 else None

    @property
    def effective_price(self) -> float | None:
        """Average all-in cost per share, fees included."""
        return self.cost / self.shares if self.shares > 0 else None

    @property
    def worst_price(self) -> float | None:
        return self.levels[-1][0] if self.levels else None


def walk_asks(asks: list[BookLevel], budget_usd: float, limit_price: float, fee_rate: float,
              depth_fraction: float = 1.0, share_step: float = 0.01) -> FillEstimate:
    """Simulate a marketable limit BUY against best-first asks.

    Consumes at most ``depth_fraction`` of each displayed level, never pays
    above ``limit_price``, and keeps total cost (notional + taker fees) within
    ``budget_usd``.  Shares are floored to ``share_step``.
    """
    est = FillEstimate()
    remaining = budget_usd
    for lv in asks:
        if lv.price > limit_price + 1e-12:
            break
        per_share = lv.price + fee_rate * lv.price * (1.0 - lv.price)
        available = lv.size * depth_fraction
        take = min(available, remaining / per_share if per_share > 0 else 0.0)
        take = math.floor(take / share_step + 1e-9) * share_step
        if take <= 0:
            break
        fee = taker_fee(take, lv.price, fee_rate)
        est.shares += take
        est.notional += take * lv.price
        est.fees += fee
        est.levels.append((lv.price, take))
        remaining -= take * lv.price + fee
        if remaining <= per_share * share_step:
            break
    # Usable depth (within the limit) ran out before the budget did.
    est.depth_exhausted = remaining > 0.01 + (limit_price * share_step)
    est.shares = round(est.shares, 6)
    return est


def depth_usd_within(asks: list[BookLevel], limit_price: float) -> float:
    return sum(lv.price * lv.size for lv in asks if lv.price <= limit_price + 1e-12)
