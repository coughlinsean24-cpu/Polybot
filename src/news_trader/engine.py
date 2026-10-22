"""
News-trading engine: wires ingestion -> matching -> AI -> edge -> risk ->
execution -> journal, plus background catalog refresh, outcome tracking and
a status file for the dashboard.

``process_event`` is the whole decision pipeline for one event and is fully
testable with injected fakes (clock, HTTP, analyzer, executor, journal).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from src.news_trader.ai import AnalysisResult, CandidateContext, SignalAnalyzer, validate_assessment
from src.news_trader.catalog import MarketCatalog, market_tradeable
from src.news_trader.clock import Clock
from src.news_trader.config import NewsTraderSettings
from src.news_trader.edge import plan_trade
from src.news_trader.execution import AuthorizationStatus, ExecutionAdapter, OrderRequest
from src.news_trader.ingestion import IngestedEvent, IngestionService
from src.news_trader.journal import Journal, utc_day
from src.news_trader.matching import EventMarketMatcher
from src.news_trader.models import Action, MarketMatch, NewsEvent, Rejection
from src.news_trader.orderbook import OrderBookClient
from src.news_trader.resolution import check_resolution
from src.news_trader.risk import RiskManager
from src.news_trader.tracker import OutcomeTracker

logger = logging.getLogger(__name__)


@dataclass
class EventOutcome:
    event_id: str
    status: str
    detail: str = ""
    rejections: list[Rejection] = field(default_factory=list)
    signals: int = 0
    trades: int = 0


def _codes(rs: list[Rejection]) -> list[str]:
    return sorted({r.code for r in rs})


def _detail(rs: list[Rejection]) -> str:
    return " | ".join(str(r) for r in rs)


class NewsTradingEngine:
    def __init__(self, settings: NewsTraderSettings, *, catalog: MarketCatalog, books: OrderBookClient,
                 analyzer: SignalAnalyzer, executor: ExecutionAdapter, journal: Journal,
                 risk: RiskManager, clock: Clock, ingestion: IngestionService | None = None,
                 tracker: OutcomeTracker | None = None, live_auth: AuthorizationStatus | None = None,
                 config_path: str | None = None):
        self.settings = settings
        self.catalog = catalog
        self.books = books
        self.analyzer = analyzer
        self.executor = executor
        self.journal = journal
        self.risk = risk
        self.clock = clock
        self.ingestion = ingestion
        self.tracker = tracker
        self.live_auth = live_auth
        self.config_path = config_path
        self.matcher = EventMarketMatcher(catalog, max_candidates=settings.matching.max_candidates,
                                          min_score=settings.matching.min_score)
        self.recent_headlines: deque[tuple[float, str, set[str]]] = deque(maxlen=500)
        self.counters: dict[str, int] = {}
        self.latencies: deque[float] = deque(maxlen=200)
        self.started_at = clock.now()
        self._config_mtime: float | None = None
        # Exposure is read, sized against, and committed (order + position)
        # under one lock so concurrent events cannot jointly exceed a limit.
        self._exec_lock = asyncio.Lock()

    # ── pipeline ─────────────────────────────────────────────────────────

    def _count(self, key: str) -> None:
        self.counters[key] = self.counters.get(key, 0) + 1

    def _finish(self, ev: NewsEvent, status: str, rs: list[Rejection] | None = None,
                detail: str = "", **kw) -> EventOutcome:
        rs = rs or []
        text = detail or _detail(rs)
        self.journal.set_event_status(ev.event_id, status, text, self.clock.now())
        self._count(f"event_{status}")
        return EventOutcome(ev.event_id, status, text, rs, **kw)

    async def process_event(self, item: IngestedEvent) -> EventOutcome:
        ev = item.event
        self.journal.record_event(ev, item.kind, received_at=self.clock.now())
        self._count("events_total")
        if item.kind == "duplicate":
            return self._finish(ev, "duplicate", detail=f"duplicate of {ev.duplicate_of}")

        rs = self.risk.pre_event(ev)
        if rs:
            return self._finish(ev, "prefilter_rejected", rs)
        if not self.catalog.ready:
            return self._finish(ev, "catalog_unavailable", [Rejection("catalog_unavailable", "no market catalog")])

        matches = self.matcher.match(ev)
        matched_at = self.clock.now()
        if not matches:
            return self._finish(ev, "no_match", detail="no candidate market above match threshold")

        candidates, rejected = await self._screen_candidates(ev, matches, matched_at)
        if not candidates:
            return self._finish(ev, "markets_rejected", rejected)
        self._count("events_qualifying")

        related = self._related_headlines(ev)
        ai_started = self.clock.now()
        result = await self.analyzer.analyze(ev, [c for c, _ in candidates], related)
        ai_done = self.clock.now()
        self.journal.record_analysis(ev, [c.market.market_id for c, _ in candidates], result, ai_done)
        self._count("ai_calls" if not result.cached else "ai_cache_hits")
        if not result.ok:
            # Fail safe: no analysis means NO TRADE for every candidate.
            for ctx, snap in candidates:
                self._record_signal(ev, ctx, snap, None, result, matched_at, ai_started, ai_done,
                                    [Rejection("ai_failed", result.error)], stage="ai")
            return self._finish(ev, "ai_failed", [Rejection("ai_failed", result.error)])

        by_id = {a.market_id: a for a in result.analysis.assessments}
        signals = trades = 0
        all_rs: list[Rejection] = []
        for ctx, snap in candidates:
            a = by_id.get(ctx.market.market_id)
            if a is None:
                r = [Rejection("ai_missing", "model returned no assessment for this market")]
                self._record_signal(ev, ctx, snap, None, result, matched_at, ai_started, ai_done, r, "ai")
                all_rs += r
                continue
            signals += 1
            traded, r = await self._decide(ev, ctx, snap, a, result, matched_at, ai_started, ai_done)
            trades += int(traded)
            all_rs += r
        status = "traded" if trades else "signals_rejected"
        return self._finish(ev, status, all_rs, signals=signals, trades=trades)

    async def _screen_candidates(self, ev: NewsEvent, matches: list[MarketMatch], at: float):
        min_close = self.settings.risk.min_time_to_close_hours * 3600
        out: list[tuple[CandidateContext, dict]] = []
        rejected: list[Rejection] = []
        eligible: list[MarketMatch] = []
        for mm in matches:
            m = mm.market
            ok, why = market_tradeable(m, self.clock.now(), min_close)
            res = check_resolution(m, self.settings.matching.max_ambiguity)
            mm.resolution = res
            self.journal.record_match(ev.event_id, m, mm.score, mm.matched_terms, mm.reasons, ok, why, res, at)
            if not ok:
                rejected.append(Rejection("market_status", f"{m.question[:60]}: {why}"))
            elif not res.ok:
                rejected.append(Rejection("ambiguous_rules", f"{m.question[:60]}: {'; '.join(res.issues)}"))
            elif dup := self.risk.duplicate(ev.cluster_id, m.market_id):
                rejected.append(dup)
            else:
                eligible.append(mm)

        # Fetch every candidate's YES and NO books concurrently (latency matters).
        fetched = await asyncio.gather(*(
            asyncio.gather(self.books.get_book(mm.market.yes_token), self.books.get_book(mm.market.no_token))
            for mm in eligible), return_exceptions=True)
        for mm, books in zip(eligible, fetched):
            m, res = mm.market, mm.resolution
            if isinstance(books, BaseException):
                rejected.append(Rejection("book_unavailable", f"{m.question[:60]}: {type(books).__name__}: {books}"))
                continue
            yes, no = books
            self.journal.record_book(yes, m.market_id)
            self.journal.record_book(no, m.market_id)
            ctx = CandidateContext(m, yes, no, res, mm.score)
            book_rs = self.risk.pre_market(ctx)
            if book_rs:
                rejected.extend(Rejection(r.code, f"{m.question[:60]}: {r.detail}") for r in book_rs)
                continue
            snap = {"yes_bid": yes.best_bid, "yes_ask": yes.best_ask, "no_ask": no.best_ask}
            out.append((ctx, snap))
        return out, rejected

    async def _decide(self, ev, ctx: CandidateContext, snap: dict, a, result: AnalysisResult,
                      matched_at, ai_started, ai_done) -> tuple[bool, list[Rejection]]:
        rs = validate_assessment(a, ctx, ev, result.analysis)
        if a.action != Action.NO_TRADE:
            rs += self.risk.post_ai(a, ev)
        if rs:
            self._record_signal(ev, ctx, snap, a, result, matched_at, ai_started, ai_done, rs, "ai")
            return False, rs

        # Pre-trade recheck: the books we analysed are seconds old and faster
        # bots may have moved them.  Re-fetch and re-run book checks.
        try:
            yes, no = await asyncio.gather(self.books.get_book(ctx.market.yes_token),
                                           self.books.get_book(ctx.market.no_token))
        except Exception as e:
            r = [Rejection("book_unavailable", f"pre-trade recheck failed: {e}")]
            self._record_signal(ev, ctx, snap, a, result, matched_at, ai_started, ai_done, r, "risk")
            return False, r
        self.journal.record_book(yes, ctx.market.market_id)
        self.journal.record_book(no, ctx.market.market_id)
        ctx = CandidateContext(ctx.market, yes, no, ctx.resolution, ctx.match_score)
        async with self._exec_lock:
            return await self._size_and_execute(ev, ctx, snap, a, result, matched_at, ai_started, ai_done)

    async def _size_and_execute(self, ev, ctx: CandidateContext, snap: dict, a, result: AnalysisResult,
                                matched_at, ai_started, ai_done) -> tuple[bool, list[Rejection]]:
        rs = self.risk.book_checks(ctx.yes_book, ctx.no_book)
        # Re-check under the lock: another item from the same story may have
        # traded this market while we were waiting for the AI or the lock.
        dup = self.risk.duplicate(ev.cluster_id, ctx.market.market_id)
        if dup:
            rs.append(dup)
        exp = self.risk.exposure()
        rs += self.risk.pre_execution(exp)
        plan = None
        if not rs:
            limits = self.risk.sizing_limits(exp, ctx.market.market_id, ctx.market.event_id, ev.cluster_id)
            plan, plan_rs = plan_trade(a, ctx, self.settings.risk, self.settings.ai.probability_weight,
                                       self.risk.bankroll(exp), limits)
            rs += plan_rs
        rs += self.risk.post_ai(a, ev)  # latency/age again, after the recheck
        if rs or plan is None:
            self._record_signal(ev, ctx, snap, a, result, matched_at, ai_started, ai_done, rs, "risk", plan)
            return False, rs

        signal_id = self._record_signal(ev, ctx, snap, a, result, matched_at, ai_started, ai_done, [],
                                        "execution", plan)
        req = OrderRequest(signal_id, ctx.market, plan.outcome, plan.token_id, plan.limit_price,
                           plan.budget_usd, ctx.market.fee_rate, self.clock.now())
        order = await self.executor.execute(req)
        self.journal.record_order(
            order_id=signal_id, signal_id=signal_id, mode=self.executor.mode, market_id=ctx.market.market_id,
            token_id=plan.token_id, outcome=plan.outcome, limit_price=plan.limit_price,
            requested_cost=plan.budget_usd, status=order.status, filled_shares=order.filled_shares,
            avg_price=order.avg_price, fees=order.fees, notional=order.notional,
            exchange_order_id=order.exchange_order_id, error=order.error, submitted_at=order.submitted_at,
            completed_at=order.completed_at, side_ask_at_submit=order.side_ask_at_submit, fills=order.fills,
        )
        self._count(f"orders_{order.status.lower()}")
        if order.filled_shares <= 0:
            r = [Rejection("not_filled", f"order {order.status}: {order.error or 'no liquidity at limit'}")]
            return False, r
        baseline = snap["yes_ask"] if plan.outcome == "YES" else snap["no_ask"]
        self.journal.open_position(
            position_id=signal_id, signal_id=signal_id, mode=self.executor.mode,
            market_id=ctx.market.market_id, condition_id=ctx.market.condition_id,
            question=ctx.market.question, event_group=ctx.market.event_id, cluster_id=ev.cluster_id,
            token_id=plan.token_id, outcome=plan.outcome, outcome_label=plan.outcome_label,
            market_category=ctx.market.category, source_name=ev.source_name, provider=ev.provider,
            shares=order.filled_shares, entry_price=order.avg_price, entry_cost=order.notional + order.fees,
            entry_fees=order.fees, p_used=plan.p_used_side, baseline_price=baseline,
            news_age_at_decision=ev.age(ai_done), detection_to_decision=ai_done - ev.first_seen_at,
            opened_at=order.completed_at, unrealized_pnl=0.0,
        )
        self._count("positions_opened")
        logger.info("%s %s %.2f sh %s @ %.3f on '%s' (edge %.3f, news: %s)", self.executor.mode.upper(), plan.outcome,
                    order.filled_shares, plan.outcome_label, order.avg_price, ctx.market.question[:60],
                    plan.edge_after_costs, ev.title[:60])
        return True, []

    def _record_signal(self, ev: NewsEvent, ctx: CandidateContext, snap: dict, a, result: AnalysisResult,
                       matched_at, ai_started, ai_done, rs: list[Rejection], stage: str, plan=None) -> str:
        now = self.clock.now()
        signal_id = uuid.uuid4().hex[:16]
        action = a.action.value if a is not None else Action.NO_TRADE.value
        side_ask_match = None
        side_ask_now = None
        if a is not None and a.action == Action.BUY_NO:
            side_ask_match, side_ask_now = snap.get("no_ask"), ctx.no_book.best_ask
        elif a is not None:
            side_ask_match, side_ask_now = snap.get("yes_ask"), ctx.yes_book.best_ask
        self.journal.record_signal(
            signal_id=signal_id, event_id=ev.event_id, cluster_id=ev.cluster_id,
            market_id=ctx.market.market_id, condition_id=ctx.market.condition_id,
            question=ctx.market.question, market_category=ctx.market.category,
            event_category=ev.category, event_group=ctx.market.event_id, source_name=ev.source_name,
            provider=ev.provider, verification=ev.verification.value, stage=stage, action=action,
            decision="REJECT" if rs else "TRADE", rejection_codes=_codes(rs), rejection_detail=_detail(rs)[:2000],
            ai_model=result.model, ai_ok=result.ok, ai_error=result.error[:500] if result.error else None,
            relevance=a.relevance.value if a else None,
            rule_compatibility=a.rule_compatibility.value if a else None,
            rule_analysis=a.rule_analysis[:2000] if a else None,
            priced_in=a.priced_in_assessment[:1000] if a else None, evidence=a.evidence if a else [],
            p_ai_yes=a.fair_probability_yes if a else None, p_low=a.probability_low if a else None,
            p_high=a.probability_high if a else None, confidence=a.confidence if a else None,
            p_used_side=plan.p_used_side if plan else None, limit_price=plan.limit_price if plan else None,
            planned_shares=plan.fill.shares if plan else None, planned_cost=plan.budget_usd if plan else None,
            edge_after_costs=plan.edge_after_costs if plan else None,
            expected_profit=plan.expected_profit_usd if plan else None,
            binding_limit=plan.binding_limit if plan else None,
            published_at=ev.published_at, first_seen_at=ev.first_seen_at, matched_at=matched_at,
            ai_started_at=ai_started, ai_completed_at=ai_done, decision_at=now,
            yes_bid_at_match=snap.get("yes_bid"), yes_ask_at_match=snap.get("yes_ask"),
            yes_mid_at_decision=ctx.yes_book.mid, side_ask_at_match=side_ask_match,
            side_ask_at_decision=side_ask_now, spread_at_decision=ctx.yes_book.spread,
            news_age_at_decision=ev.age(now), detection_to_decision=now - ev.first_seen_at,
        )
        if a is not None:
            self.journal.schedule_marks(signal_id, ctx.market.market_id, ctx.market.yes_token,
                                        self.settings.tracking.mark_horizons_seconds, now)
        self.latencies.append(now - ev.first_seen_at)
        self._count("signals_trade" if not rs else "signals_rejected")
        return signal_id

    def _related_headlines(self, ev: NewsEvent) -> list[str]:
        from src.news_trader import text as T
        toks = T.token_set(ev.title)
        now = self.clock.now()
        out = [h for ts, h, ht in reversed(self.recent_headlines)
               if now - ts < 86400 and h != ev.title and T.jaccard(toks, ht) >= 0.25]
        self.recent_headlines.append((now, ev.title, toks))
        return out[:6]

    # ── runtime ──────────────────────────────────────────────────────────

    def maybe_reload_settings(self) -> None:
        """Hot-reload risk/AI/matching settings edited from the dashboard."""
        if not self.config_path:
            return
        try:
            mtime = os.path.getmtime(self.config_path)
        except OSError:
            return
        if self._config_mtime is None:
            self._config_mtime = mtime
            return
        if mtime == self._config_mtime:
            return
        self._config_mtime = mtime
        from src.news_trader.config import load_settings_file
        try:
            new = load_settings_file(self.config_path)
        except Exception as e:
            logger.error("config reload rejected (keeping previous settings): %s", e)
            return
        # Mode and source list need a restart; everything else applies live.
        new.mode = self.settings.mode
        new.sources.enabled = [s for s in new.sources.enabled if s in self.settings.sources.enabled] \
            or self.settings.sources.enabled
        self.settings = new
        self.risk.settings = new
        self.matcher.max_candidates = new.matching.max_candidates
        self.matcher.min_score = new.matching.min_score
        if hasattr(self.analyzer, "settings"):
            self.analyzer.settings = new.ai
            if hasattr(self.analyzer, "budget"):
                self.analyzer.budget.max_calls = new.ai.max_calls_per_day
                self.analyzer.budget.max_usd = new.ai.daily_budget_usd
        if self.tracker:
            self.tracker.settings = new
        logger.info("news_trader settings reloaded from %s", self.config_path)

    def status(self) -> dict:
        now = self.clock.now()
        budget = getattr(self.analyzer, "budget", None)
        exp = self.risk.exposure()
        lat = sorted(self.latencies)
        return {
            "heartbeat": now,
            "started_at": self.started_at,
            "mode": self.executor.mode,
            "requested_mode": self.settings.mode,
            "live_authorization": None if self.live_auth is None else {
                "granted": self.live_auth.granted, "reasons": self.live_auth.reasons,
                "geoblock": self.live_auth.geoblock},
            "halted": self.risk.halted(),
            "enabled": self.settings.enabled,
            "providers": [h.to_dict() for h in self.ingestion.health.values()] if self.ingestion else [],
            "queue_size": self.ingestion.queue.qsize() if self.ingestion else 0,
            "catalog": None if self.catalog.last_stats is None else {
                **self.catalog.last_stats.__dict__, "age_seconds": self.catalog.age(),
                "error": self.catalog.last_error},
            "ai": {
                "model": self.settings.ai.model, "provider": self.settings.ai.provider,
                "unavailable_reason": getattr(self.analyzer, "reason", None),
                "calls_today": getattr(budget, "calls", None), "spent_today_usd": getattr(budget, "spent_usd", None),
                "max_calls": self.settings.ai.max_calls_per_day, "budget_usd": self.settings.ai.daily_budget_usd,
            },
            "exposure": {
                "open_positions": exp.open_positions, "open_cost": round(exp.open_cost, 2),
                "entry_cost_today": round(exp.entry_cost_today, 2),
                "realized_today": round(exp.realized_today, 2), "unrealized": round(exp.unrealized, 2),
                "bankroll": round(self.risk.bankroll(exp), 2),
            },
            "latency": {
                "samples": len(lat),
                "median_detection_to_decision": lat[len(lat) // 2] if lat else None,
                "p90_detection_to_decision": lat[int(len(lat) * 0.9)] if lat else None,
            },
            "counters": dict(self.counters),
        }

    def write_status(self) -> None:
        path = Path(self.settings.status_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.status(), default=str, indent=1))
        tmp.replace(path)

    async def _catalog_loop(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await self.catalog.refresh()
            except Exception as e:
                self.catalog.last_error = f"{type(e).__name__}: {e}"
                logger.warning("catalog refresh failed (keeping previous): %s", e)
            await self._sleep_or_stop(stop, self.settings.catalog.refresh_seconds if self.catalog.ready else 30)

    async def _consumer(self, stop: asyncio.Event, sem: asyncio.Semaphore) -> None:
        assert self.ingestion is not None
        tasks: set[asyncio.Task] = set()
        while not stop.is_set():
            if not self.catalog.ready:
                # Hold events (they keep their first-seen time) until markets
                # are loaded, rather than rejecting them as unmatched.
                await self._sleep_or_stop(stop, 0.5)
                continue
            try:
                item = await asyncio.wait_for(self.ingestion.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            await sem.acquire()

            async def run(it=item):
                try:
                    await self.process_event(it)
                except Exception:
                    logger.exception("event pipeline crashed for %s", it.event.event_id)
                    self.journal.set_event_status(it.event.event_id, "error", "pipeline exception",
                                                  self.clock.now())
                finally:
                    sem.release()
            t = asyncio.create_task(run())
            tasks.add(t)
            t.add_done_callback(tasks.discard)
        for t in tasks:
            t.cancel()

    async def _housekeeping(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            self.maybe_reload_settings()
            if self.tracker:
                try:
                    await self.tracker.run_once()
                except Exception:
                    logger.exception("outcome tracker failed")
            try:
                self.write_status()
            except Exception:
                logger.exception("status write failed")
            await self._sleep_or_stop(stop, min(5.0, self.settings.tracking.loop_seconds))

    @staticmethod
    async def _sleep_or_stop(stop: asyncio.Event, seconds: float) -> None:
        try:
            await asyncio.wait_for(stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    async def run(self, stop: asyncio.Event) -> None:
        budget = getattr(self.analyzer, "budget", None)
        if budget is not None:
            budget.seed(*self.journal.ai_usage_on(utc_day(self.clock.now())))
        sem = asyncio.Semaphore(self.settings.max_concurrent_events)
        tasks = [
            asyncio.create_task(self._catalog_loop(stop)),
            asyncio.create_task(self._consumer(stop, sem)),
            asyncio.create_task(self._housekeeping(stop)),
        ]
        if self.ingestion:
            tasks.append(asyncio.create_task(self.ingestion.run(stop)))
        logger.info("news engine running in %s mode", self.executor.mode.upper())
        try:
            await stop.wait()
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.write_status()
