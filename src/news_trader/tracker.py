"""
Post-decision tracking: price marks, position marking/exits, resolutions.

* Price marks are scheduled for EVERY AI-assessed signal (traded or not) at
  fixed horizons, so we can measure what the market did after the news —
  the raw material for missed-opportunity and "already priced in" analysis.
* Open positions are marked at the best BID (what we could actually sell
  for), net of the taker fee an exit would pay.
* Resolution is read from Gamma (closed market + outcomePrices); a position
  pays out shares * final price of its outcome (1, 0, or 0.5 for 50-50).
"""

from __future__ import annotations

import json
import logging

from src.news_trader.catalog import MarketCatalog
from src.news_trader.clock import Clock
from src.news_trader.config import NewsTraderSettings
from src.news_trader.execution import ExecutionAdapter
from src.news_trader.journal import Journal
from src.news_trader.orderbook import OrderBookClient, taker_fee

logger = logging.getLogger(__name__)


def parse_resolution(raw: dict) -> tuple[bool, list[float]] | None:
    """Return (resolved, [yes_price, no_price]) if the market has settled."""
    if not isinstance(raw, dict) or not raw.get("closed"):
        return None
    prices = raw.get("outcomePrices")
    if isinstance(prices, str):
        try:
            prices = json.loads(prices)
        except json.JSONDecodeError:
            return None
    try:
        p = [float(x) for x in prices][:2]
    except (TypeError, ValueError):
        return None
    if len(p) != 2:
        return None
    decisive = max(p) >= 0.99
    fifty = abs(p[0] - 0.5) < 0.01 and abs(p[1] - 0.5) < 0.01 and \
        str(raw.get("umaResolutionStatus", "")).lower() == "resolved"
    if not (decisive or fifty):
        return None
    if decisive:
        p = [1.0, 0.0] if p[0] >= 0.99 else [0.0, 1.0]
    return True, p


class OutcomeTracker:
    def __init__(self, settings: NewsTraderSettings, journal: Journal, books: OrderBookClient,
                 catalog: MarketCatalog, executor: ExecutionAdapter, clock: Clock):
        self.settings = settings
        self.journal = journal
        self.books = books
        self.catalog = catalog
        self.executor = executor
        self.clock = clock
        self._last_resolution_check = 0.0

    async def run_once(self) -> None:
        await self.fill_due_marks()
        await self.mark_positions()
        now = self.clock.now()
        if now - self._last_resolution_check >= self.settings.tracking.resolution_check_seconds:
            self._last_resolution_check = now
            await self.check_resolutions()

    async def fill_due_marks(self) -> int:
        n = 0
        for mark in self.journal.due_marks(self.clock.now()):
            try:
                book = await self.books.get_book(mark.yes_token)
            except Exception as e:
                logger.debug("mark fetch failed for %s: %s", mark.market_id, e)
                # Leave unobserved; a market that has closed will keep failing,
                # so give up after a day past due.
                if self.clock.now() - mark.due_at > 86400:
                    self.journal.fill_mark(mark.id, self.clock.now(), None, None)
                continue
            self.journal.fill_mark(mark.id, self.clock.now(), book.best_bid, book.best_ask)
            n += 1
        return n

    async def mark_positions(self) -> None:
        ex = self.settings.execution
        for pos in self.journal.open_positions(self.executor.mode):
            try:
                book = await self.books.get_book(pos.token_id)
            except Exception as e:
                logger.debug("position mark failed %s: %s", pos.position_id, e)
                continue
            bid = book.best_bid
            fee_rate = (self.catalog.markets.get(pos.market_id).fee_rate
                        if pos.market_id in self.catalog.markets else 0.0)
            now = self.clock.now()
            if bid is None:
                self.journal.update_position(pos.position_id, last_mark_at=now)
                continue
            value = pos.shares * bid - taker_fee(pos.shares, bid, fee_rate)
            self.journal.update_position(pos.position_id, last_mark_bid=bid, last_mark_at=now,
                                         unrealized_pnl=value - pos.entry_cost)

            reason = None
            if ex.exit_policy == "take_profit" and pos.p_used is not None and bid >= pos.p_used:
                reason = "take_profit_at_fair_value"
            elif ex.max_hold_hours > 0 and now - pos.opened_at >= ex.max_hold_hours * 3600:
                reason = "max_hold_time"
            if reason:
                await self._exit(pos, bid, fee_rate, reason)

    async def _exit(self, pos, limit: float, fee_rate: float, reason: str) -> None:
        try:
            res = await self.executor.sell(pos.token_id, pos.shares, limit, fee_rate)
        except NotImplementedError:
            return
        except Exception as e:
            logger.warning("exit failed for %s: %s", pos.position_id, e)
            return
        if res.sold_shares < pos.shares - 0.01:
            logger.info("exit for %s not fully fillable at %.3f (%.2f/%.2f); holding",
                        pos.position_id, limit, res.sold_shares, pos.shares)
            return
        realized = res.proceeds - res.fees - pos.entry_cost
        self.journal.update_position(
            pos.position_id, status="CLOSED", exit_price=res.avg_price, exit_proceeds=res.proceeds,
            exit_fees=res.fees, closed_at=self.clock.now(), close_reason=reason,
            realized_pnl=realized, unrealized_pnl=0.0,
        )
        logger.info("closed %s (%s): realized %+.2f", pos.position_id, reason, realized)

    async def check_resolutions(self) -> None:
        pending = dict(self.journal.conditions_needing_resolution())
        if not pending:
            return
        try:
            raws = await self.catalog.fetch_markets_status(sorted(pending))
        except Exception as e:
            logger.warning("resolution check failed: %s", e)
            return
        for condition_id, market_id in pending.items():
            raw = raws.get(condition_id)
            parsed = parse_resolution(raw) if raw else None
            now = self.clock.now()
            if parsed is None:
                self.journal.record_resolution(condition_id, market_id, None, [], now)
                continue
            _, prices = parsed
            resolved_yes = True if prices[0] == 1.0 else (False if prices[1] == 1.0 else None)
            self.journal.record_resolution(condition_id, market_id, resolved_yes, prices, now)
            self.settle(condition_id, prices, resolved_yes)

    def settle(self, condition_id: str, prices: list[float], resolved_yes: bool | None) -> None:
        for pos in self.journal.open_positions():
            if pos.condition_id != condition_id:
                continue
            payout_price = prices[0] if pos.outcome == "YES" else prices[1]
            payout = pos.shares * payout_price
            self.journal.update_position(
                pos.position_id, status="RESOLVED", exit_price=payout_price, exit_proceeds=payout,
                exit_fees=0.0, closed_at=self.clock.now(), close_reason="market_resolved",
                realized_pnl=payout - pos.entry_cost, unrealized_pnl=0.0, resolved_yes=resolved_yes,
            )
            logger.info("position %s resolved: payout %.2f, P&L %+.2f",
                        pos.position_id, payout, payout - pos.entry_cost)
