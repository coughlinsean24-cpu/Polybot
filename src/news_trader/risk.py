"""
Deterministic risk gate.  Only this module can approve an order.

Checks are grouped by pipeline stage so that a cheap rejection happens
before any paid AI call:

  pre_event   halts, category, source quality, verification, freshness
  pre_market  market status, rules, quote freshness, spread, depth, price range
  post_ai     model confidence/uncertainty, corroboration, decision latency
  sizing      cash, per-trade, per-market, daily, correlated exposure, loss limit

Every rejection carries a stable code (for analytics) and a human detail.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.news_trader.ai import CandidateContext, MarketAssessment
from src.news_trader.catalog import market_tradeable
from src.news_trader.clock import Clock
from src.news_trader.config import NewsTraderSettings
from src.news_trader.edge import SizingLimits
from src.news_trader.journal import Journal, day_start
from src.news_trader.models import NewsEvent, OrderBook, Rejection, Verification
from src.news_trader.orderbook import depth_usd_within

logger = logging.getLogger(__name__)


@dataclass
class Exposure:
    open_positions: int
    open_cost: float
    market_cost: dict[str, float]
    group_cost: dict[str, float]      # Gamma event id (correlated outcomes)
    cluster_cost: dict[str, float]    # same news story
    entry_cost_today: float
    realized_today: float
    realized_total: float
    unrealized: float


class RiskManager:
    def __init__(self, settings: NewsTraderSettings, journal: Journal, clock: Clock,
                 halt_files: list[Path], mode: str = "paper"):
        self.settings = settings
        self.journal = journal
        self.clock = clock
        self.halt_files = halt_files
        self.mode = mode
        self._latched_halt: str | None = None

    # ── global halts ────────────────────────────────────────────────────

    def halted(self) -> str | None:
        if self._latched_halt:
            return self._latched_halt
        if self.settings.emergency_stop:
            return "news_trader.emergency_stop is set"
        for f in self.halt_files:
            if f.exists():
                return f"kill switch file present: {f.name}"
        if not self.settings.enabled:
            return "news engine disabled in config"
        return None

    def latch_halt(self, reason: str) -> None:
        """Circuit breaker: stays halted until the process restarts."""
        if not self._latched_halt:
            logger.error("NEWS ENGINE HALTED: %s", reason)
            self._latched_halt = reason

    # ── exposure ────────────────────────────────────────────────────────

    def exposure(self) -> Exposure:
        now = self.clock.now()
        open_pos = self.journal.open_positions(self.mode)
        market_cost: dict[str, float] = {}
        group_cost: dict[str, float] = {}
        cluster_cost: dict[str, float] = {}
        unrealized = 0.0
        for p in open_pos:
            market_cost[p.market_id] = market_cost.get(p.market_id, 0.0) + p.entry_cost
            if p.event_group:
                group_cost[p.event_group] = group_cost.get(p.event_group, 0.0) + p.entry_cost
            if p.cluster_id:
                cluster_cost[p.cluster_id] = cluster_cost.get(p.cluster_id, 0.0) + p.entry_cost
            unrealized += p.unrealized_pnl or 0.0
        start = day_start(now)
        return Exposure(
            open_positions=len(open_pos),
            open_cost=sum(market_cost.values()),
            market_cost=market_cost, group_cost=group_cost, cluster_cost=cluster_cost,
            entry_cost_today=self.journal.entry_cost_since(start, self.mode),
            realized_today=self.journal.realized_pnl_since(start, self.mode),
            realized_total=self.journal.realized_pnl_total(self.mode),
            unrealized=unrealized,
        )

    def bankroll(self, exp: Exposure) -> float:
        """Paper equity: starting bankroll + realized P&L (marks excluded)."""
        return self.settings.risk.paper_bankroll_usd + exp.realized_total

    def check_loss_limit(self, exp: Exposure) -> Rejection | None:
        loss = -(exp.realized_today + exp.unrealized)
        if loss >= self.settings.risk.daily_loss_limit_usd:
            reason = (f"daily loss ${loss:.2f} (realized {exp.realized_today:+.2f}, "
                      f"unrealized {exp.unrealized:+.2f}) >= limit ${self.settings.risk.daily_loss_limit_usd:.2f}")
            return Rejection("daily_loss_limit", reason)
        return None

    def sizing_limits(self, exp: Exposure, market_id: str, event_group: str, cluster_id: str) -> SizingLimits:
        r = self.settings.risk
        return SizingLimits(
            cash_available=self.bankroll(exp) - exp.open_cost,
            trade_cap=r.max_trade_usd,
            market_cap=r.max_market_exposure_usd - exp.market_cost.get(market_id, 0.0),
            daily_cap=r.max_daily_exposure_usd - exp.entry_cost_today,
            correlated_cap=r.max_correlated_exposure_usd - max(
                exp.group_cost.get(event_group, 0.0) if event_group else 0.0,
                exp.cluster_cost.get(cluster_id, 0.0),
            ),
        )

    # ── stage checks ────────────────────────────────────────────────────

    def pre_event(self, ev: NewsEvent) -> list[Rejection]:
        r = self.settings.risk
        now = self.clock.now()
        out: list[Rejection] = []
        halt = self.halted()
        if halt:
            out.append(Rejection("halted", halt))
        if ev.category not in self.settings.categories:
            out.append(Rejection("category_disabled", f"category '{ev.category}' not enabled"))
        if ev.provider not in self.settings.sources.enabled:
            out.append(Rejection("source_disabled", f"source '{ev.provider}' filtered out"))
        if ev.credibility < self.settings.sources.min_credibility:
            out.append(Rejection("low_credibility", f"credibility {ev.credibility:.2f} < "
                                                    f"{self.settings.sources.min_credibility:.2f}"))
        if ev.verification == Verification.CONFLICTING:
            out.append(Rejection("conflicting_reports", "a denial/retraction conflicts with this story"))
        elif ev.verification == Verification.UNVERIFIED:
            out.append(Rejection("unverified", "no credible source for this story"))
        elif self.needs_corroboration(ev) and ev.verification == Verification.SINGLE_SOURCE:
            out.append(Rejection("awaiting_corroboration",
                                 f"category '{ev.category}' requires >= 2 independent sources or a primary source"))
        if "speculative" in ev.flags:
            out.append(Rejection("speculative", "headline is speculative (could/may/reportedly/in talks)"))
        if "denial" in ev.flags and ev.verification != Verification.PRIMARY_SOURCE:
            out.append(Rejection("denial", "denial/retraction story; not acted on without a primary source"))
        age = ev.age(now)
        if age > r.max_news_age_seconds:
            out.append(Rejection("stale_news", f"news age {age:.0f}s > {r.max_news_age_seconds:.0f}s"))
        return out

    def needs_corroboration(self, ev: NewsEvent) -> bool:
        return (not ev.is_primary_source
                and ev.category in self.settings.risk.require_corroboration_categories)

    def pre_market(self, ctx: CandidateContext) -> list[Rejection]:
        r = self.settings.risk
        now = self.clock.now()
        out: list[Rejection] = []
        ok, why = market_tradeable(ctx.market, now, r.min_time_to_close_hours * 3600)
        if not ok:
            out.append(Rejection("market_status", why))
        if not ctx.resolution.ok:
            out.append(Rejection("ambiguous_rules",
                                 f"ambiguity {ctx.resolution.ambiguity_score:.2f}: {'; '.join(ctx.resolution.issues)}"))
        out.extend(self.book_checks(ctx.yes_book, ctx.no_book))
        return out

    def book_checks(self, yes: OrderBook, no: OrderBook) -> list[Rejection]:
        r = self.settings.risk
        now = self.clock.now()
        out: list[Rejection] = []
        for label, book in (("YES", yes), ("NO", no)):
            if book.age(now) > r.max_quote_age_seconds:
                out.append(Rejection("stale_quote", f"{label} book {book.age(now):.1f}s old"))
        viable = []
        for label, book in (("YES", yes), ("NO", no)):
            spread = book.spread
            if spread is None:
                continue
            if spread > r.max_spread:
                continue
            if book.best_ask is None or not (r.min_entry_price <= book.best_ask <= r.max_entry_price):
                continue
            if depth_usd_within(book.asks, book.best_ask + 0.02) < r.min_depth_usd:
                continue
            viable.append(label)
        if not viable:
            spreads = f"YES spread {yes.spread}, NO spread {no.spread}"
            out.append(Rejection("no_viable_side",
                                 f"neither side has spread <= {r.max_spread}, ask in "
                                 f"[{r.min_entry_price}, {r.max_entry_price}] and depth >= ${r.min_depth_usd:.0f} ({spreads})"))
        return out

    def post_ai(self, a: MarketAssessment, ev: NewsEvent) -> list[Rejection]:
        r = self.settings.risk
        now = self.clock.now()
        out: list[Rejection] = []
        if a.confidence < r.min_confidence:
            out.append(Rejection("low_confidence", f"confidence {a.confidence:.2f} < {r.min_confidence:.2f}"))
        width = a.probability_high - a.probability_low
        if width > r.max_uncertainty:
            out.append(Rejection("high_uncertainty", f"probability interval width {width:.2f} > {r.max_uncertainty:.2f}"))
        if a.requires_corroboration and ev.verification not in (Verification.CORROBORATED,
                                                                 Verification.PRIMARY_SOURCE):
            out.append(Rejection("awaiting_corroboration", "model requires corroboration; story is single-source"))
        age = ev.age(now)
        if age > r.max_news_age_seconds:
            out.append(Rejection("stale_news", f"news age at decision {age:.0f}s > {r.max_news_age_seconds:.0f}s"))
        latency = now - ev.first_seen_at
        if latency > r.max_decision_latency_seconds:
            out.append(Rejection("slow_decision", f"detection-to-decision {latency:.0f}s > "
                                                  f"{r.max_decision_latency_seconds:.0f}s"))
        return out

    def duplicate(self, cluster_id: str, market_id: str) -> Rejection | None:
        now = self.clock.now()
        if (cluster_id, market_id) in self.journal.traded_pairs(now - 7 * 86400):
            return Rejection("duplicate_signal", "this story already traded this market")
        last = self.journal.last_trade_at(market_id)
        cooldown = self.settings.risk.signal_cooldown_seconds
        if last is not None and now - last < cooldown:
            return Rejection("market_cooldown", f"traded this market {now - last:.0f}s ago (cooldown {cooldown:.0f}s)")
        return None

    def pre_execution(self, exp: Exposure) -> list[Rejection]:
        out: list[Rejection] = []
        halt = self.halted()
        if halt:
            out.append(Rejection("halted", halt))
        if exp.open_positions >= self.settings.risk.max_open_positions:
            out.append(Rejection("max_positions", f"{exp.open_positions} open positions"))
        loss = self.check_loss_limit(exp)
        if loss:
            self.latch_halt(str(loss))
            out.append(loss)
        return out
