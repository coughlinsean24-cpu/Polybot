"""
Execution adapters: paper (default) and live (gated, disabled).

Both take limit orders only.  There are no market orders anywhere.

PaperExecutor is deliberately pessimistic:
  * it waits ``paper_latency_ms`` and then re-fetches the book — the fill is
    computed against the book that exists when our order would arrive, not
    the one we decided on (faster bots may already have taken the edge);
  * it only consumes ``paper_depth_haircut`` of each displayed level;
  * while "resting" it fills only against NEW asks that appear strictly
    below our limit (evidence sellers actually came to us), never by
    re-counting liquidity it already saw;
  * unfilled remainder is cancelled at the timeout (partial fills recorded).

LiveExecutor wraps the existing ``PolymarketClient`` (same signing/auth as the
BTC engine).  It never assumes a fill: fills come only from exchange order
status, and the remainder is cancelled and re-checked at timeout.  It can only
be constructed when ``LiveAuthorization`` grants it, which requires (among
other things) changing ``LIVE_TRADING_RELEASED`` in a reviewed commit.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Protocol

from src.news_trader.clock import Clock
from src.news_trader.config import ExecutionSettings, NewsTraderSettings
from src.news_trader.http import HttpClient
from src.news_trader.models import MarketRecord
from src.news_trader.orderbook import OrderBookClient, round_to_tick, taker_fee, walk_asks

logger = logging.getLogger(__name__)

# Real-money execution for this engine is released only by changing this
# constant in a reviewed commit, after the paper-trading evidence described in
# docs/NEWS_TRADER.md exists.  No config value, env var or dashboard toggle
# can override it.
LIVE_TRADING_RELEASED = False

GEOBLOCK_URL = "https://polymarket.com/api/geoblock"


@dataclass
class OrderRequest:
    client_order_id: str
    market: MarketRecord
    outcome: str            # "YES" | "NO"
    token_id: str
    limit_price: float
    max_cost_usd: float
    fee_rate: float
    created_at: float


@dataclass
class OrderResult:
    client_order_id: str
    status: str             # FILLED | PARTIAL | UNFILLED | REJECTED
    filled_shares: float = 0.0
    notional: float = 0.0
    fees: float = 0.0
    submitted_at: float = 0.0
    completed_at: float = 0.0
    exchange_order_id: str | None = None
    error: str = ""
    side_ask_at_submit: float | None = None
    fills: list[tuple[float, float, float]] = field(default_factory=list)  # (ts, price, shares)

    @property
    def avg_price(self) -> float | None:
        return self.notional / self.filled_shares if self.filled_shares > 0 else None


@dataclass
class ExitResult:
    sold_shares: float
    proceeds: float
    fees: float
    avg_price: float | None


class ExecutionAdapter(Protocol):
    mode: str

    async def execute(self, req: OrderRequest) -> OrderResult: ...

    async def sell(self, token_id: str, shares: float, limit_price: float, fee_rate: float) -> ExitResult: ...


class PaperExecutor:
    mode = "paper"

    def __init__(self, books: OrderBookClient, clock: Clock, settings: ExecutionSettings,
                 max_quote_age_seconds: float = 10.0):
        self.books = books
        self.clock = clock
        self.settings = settings
        self.max_quote_age = max_quote_age_seconds
        self._submitted: set[str] = set()

    async def execute(self, req: OrderRequest) -> OrderResult:
        if req.client_order_id in self._submitted:
            return OrderResult(req.client_order_id, "REJECTED", error="duplicate client order id")
        self._submitted.add(req.client_order_id)

        await self.clock.sleep(self.settings.paper_latency_ms / 1000.0)
        submitted_at = self.clock.now()
        res = OrderResult(req.client_order_id, "UNFILLED", submitted_at=submitted_at)
        try:
            book = await self.books.get_book(req.token_id)
        except Exception as e:
            res.status, res.error, res.completed_at = "REJECTED", f"book unavailable at submit: {e}", self.clock.now()
            return res
        res.side_ask_at_submit = book.best_ask
        if book.age(self.clock.now()) > self.max_quote_age:
            res.status, res.error, res.completed_at = "REJECTED", "stale book at submit", self.clock.now()
            return res

        budget = req.max_cost_usd
        est = walk_asks(book.asks, budget, req.limit_price, req.fee_rate, self.settings.paper_depth_haircut)
        if est.shares > 0:
            self._apply(res, est.levels, req.fee_rate, submitted_at)
            budget -= est.cost

        deadline = submitted_at + self.settings.order_timeout_seconds
        seen = {(lv.price, lv.size) for lv in book.asks}
        tick = book.tick_size
        while budget > req.limit_price * book.min_order_size and self.clock.now() < deadline:
            await self.clock.sleep(self.settings.poll_interval_seconds)
            try:
                book = await self.books.get_book(req.token_id)
            except Exception as e:
                logger.debug("paper resting poll failed: %s", e)
                continue
            new = [lv for lv in book.asks
                   if (lv.price, lv.size) not in seen and lv.price <= req.limit_price - tick / 2]
            seen = {(lv.price, lv.size) for lv in book.asks}
            if not new:
                continue
            est = walk_asks(new, budget, req.limit_price, req.fee_rate, self.settings.paper_depth_haircut)
            if est.shares > 0:
                self._apply(res, est.levels, req.fee_rate, self.clock.now())
                budget -= est.cost

        res.completed_at = self.clock.now()
        if res.filled_shares <= 0:
            res.status = "UNFILLED"
        elif budget > req.limit_price * book.min_order_size:
            res.status = "PARTIAL"   # remainder cancelled at timeout
        else:
            res.status = "FILLED"
        return res

    @staticmethod
    def _apply(res: OrderResult, levels: list[tuple[float, float]], fee_rate: float, ts: float) -> None:
        for price, shares in levels:
            res.filled_shares += shares
            res.notional += price * shares
            res.fees += taker_fee(shares, price, fee_rate)
            res.fills.append((ts, price, shares))
        res.filled_shares = round(res.filled_shares, 6)

    async def sell(self, token_id: str, shares: float, limit_price: float, fee_rate: float) -> ExitResult:
        """Paper exit: marketable limit SELL against the current bids (taker)."""
        book = await self.books.get_book(token_id)
        remaining, proceeds, fees, sold = shares, 0.0, 0.0, 0.0
        for lv in book.bids:
            if lv.price < limit_price - 1e-12 or remaining <= 0:
                break
            take = math.floor(min(remaining, lv.size * self.settings.paper_depth_haircut) * 100) / 100
            if take <= 0:
                continue
            sold += take
            proceeds += take * lv.price
            fees += taker_fee(take, lv.price, fee_rate)
            remaining -= take
        return ExitResult(sold, proceeds, fees, proceeds / sold if sold else None)


# ── live trading (disabled) ─────────────────────────────────────────────────


@dataclass
class AuthorizationStatus:
    granted: bool
    reasons: list[str]
    geoblock: dict | None = None


class LiveAuthorization:
    """Every gate must pass; any failure keeps the engine in paper mode."""

    ACK_ENV = "NEWS_TRADER_LIVE_ACK"
    ACK_VALUE = "I-ACCEPT-REAL-MONEY-RISK"

    def __init__(self, settings: NewsTraderSettings, cli_flag: bool, http: HttpClient | None,
                 env: dict | None = None):
        self.settings = settings
        self.cli_flag = cli_flag
        self.http = http
        self.env = os.environ if env is None else env

    async def evaluate(self) -> AuthorizationStatus:
        reasons: list[str] = []
        if not LIVE_TRADING_RELEASED:
            reasons.append("LIVE_TRADING_RELEASED is False in src/news_trader/execution.py "
                           "(requires a reviewed code change after paper validation)")
        if self.settings.mode != "live":
            reasons.append("news_trader.mode is not 'live'")
        if not self.cli_flag:
            reasons.append("process not started with --enable-live-news-trading")
        if self.env.get(self.ACK_ENV) != self.ACK_VALUE:
            reasons.append(f"{self.ACK_ENV} acknowledgement not set")
        for var in ("POLYMARKET_PRIVATE_KEY",):
            if not self.env.get(var):
                reasons.append(f"{var} not set")
        geo = None
        if self.http is None:
            reasons.append("geoblock status unknown (no HTTP client)")
        else:
            try:
                geo = await self.http.get_json(GEOBLOCK_URL)
                if not isinstance(geo, dict) or geo.get("blocked") is not False:
                    reasons.append(f"Polymarket geoblock reports trading not permitted from here: "
                                   f"{geo if isinstance(geo, dict) else 'unparseable response'}")
            except Exception as e:
                reasons.append(f"geoblock check failed ({type(e).__name__}); refusing live mode")
        return AuthorizationStatus(granted=not reasons, reasons=reasons, geoblock=geo)


class LiveExecutor:
    """Wraps src.polymarket_client.PolymarketClient (sync) for limit orders."""

    mode = "live"

    def __init__(self, poly_client, auth: AuthorizationStatus, books: OrderBookClient, clock: Clock,
                 settings: ExecutionSettings):
        if not auth.granted or not LIVE_TRADING_RELEASED:
            raise PermissionError("live execution not authorised: " + "; ".join(auth.reasons))
        self.client = poly_client
        self.books = books
        self.clock = clock
        self.settings = settings
        self._submitted: set[str] = set()

    @staticmethod
    def _market_info(m: MarketRecord):
        from src.polymarket_client import MarketInfo
        return MarketInfo(
            condition_id=m.condition_id, question=m.question,
            up_token_id=m.yes_token, down_token_id=m.no_token,
            up_price=m.best_ask or 0.0, down_price=0.0, end_date="", volume=m.volume_24h,
            event_start_time="", slug=m.slug,
        )

    async def _run(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(None, fn, *args)

    async def execute(self, req: OrderRequest) -> OrderResult:
        if req.client_order_id in self._submitted:
            return OrderResult(req.client_order_id, "REJECTED", error="duplicate client order id")
        self._submitted.add(req.client_order_id)
        res = OrderResult(req.client_order_id, "UNFILLED", submitted_at=self.clock.now())

        balance = await self._run(self.client.get_live_balance)
        if balance is None or balance < req.max_cost_usd:
            res.status, res.error = "REJECTED", f"insufficient balance ({balance}) for ${req.max_cost_usd:.2f}"
            return res
        # The shared client rounds prices to 0.01 to nearest; pre-round DOWN so
        # it can never round above our limit.
        limit = round_to_tick(req.limit_price, max(0.01, req.market.tick_size), "down")
        book = await self.books.get_book(req.token_id)
        res.side_ask_at_submit = book.best_ask
        if book.best_ask is None or book.best_ask > limit:
            res.status, res.error = "REJECTED", f"pre-trade recheck: ask {book.best_ask} > limit {limit}"
            return res
        shares = math.floor(req.max_cost_usd / (limit * (1 + req.fee_rate)) * 100) / 100
        placed = await self._run(self.client.place_limit_order, self._market_info(req.market),
                                 req.outcome, shares, limit)
        if not placed.success or not placed.order_id:
            res.status, res.error = "REJECTED", f"exchange rejected order: {placed.error}"
            return res
        res.exchange_order_id = placed.order_id

        matched = 0.0
        deadline = res.submitted_at + self.settings.order_timeout_seconds
        while self.clock.now() < deadline:
            await self.clock.sleep(self.settings.poll_interval_seconds)
            st = await self._run(self.client.check_position_status, placed.order_id)
            if st.get("status") == "error":
                continue
            matched = float(st.get("filled", 0.0) or 0.0)
            if matched >= shares - 1e-9 or str(st.get("status", "")).upper().startswith("CANCEL"):
                break
        if matched < shares - 1e-9:
            await self._run(self.client.cancel_order, placed.order_id)
            st = await self._run(self.client.check_position_status, placed.order_id)
            if st.get("status") != "error":
                matched = max(matched, float(st.get("filled", 0.0) or 0.0))

        res.completed_at = self.clock.now()
        if matched > 0:
            # Conservative: assume fills at our limit (the exchange may do better).
            res.filled_shares, res.notional = matched, matched * limit
            res.fees = taker_fee(matched, limit, req.fee_rate)
            res.fills.append((res.completed_at, limit, matched))
        res.status = "FILLED" if matched >= shares - 1e-9 else ("PARTIAL" if matched > 0 else "UNFILLED")
        return res

    async def sell(self, token_id: str, shares: float, limit_price: float, fee_rate: float) -> ExitResult:
        raise NotImplementedError("live exits are placed by an operator until live mode is validated")
