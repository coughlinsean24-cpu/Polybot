"""End-to-end pipeline, tracking/settlement, analytics and look-ahead-free replay."""

import json
import tempfile
import unittest
from pathlib import Path

from sqlalchemy import select

from src.news_trader.analytics import max_drawdown, performance_report
from src.news_trader.backtest import LookAheadError, PointInTimeBookStore, replay
from src.news_trader.catalog import MarketCatalog, MarketIndex
from src.news_trader.clock import ManualClock
from src.news_trader.config import CatalogSettings
from src.news_trader.engine import NewsTradingEngine
from src.news_trader.execution import PaperExecutor
from src.news_trader.ingestion import IngestedEvent
from src.news_trader.journal import BookSnapshotRow, Journal, MarkRow, OrderRow, PositionRow, SignalRow
from src.news_trader.risk import RiskManager
from src.news_trader.tracker import OutcomeTracker, parse_resolution
from tests.news_fakes import (
    T0, FakeBooks, FakeHttp, StubAnalyzer, analysis, assessment, make_book, make_event, make_market, settings,
)

RESOLVED_YES = [{"conditionId": "0xcond-m1", "closed": True, "outcomePrices": '["1", "0"]', "umaResolutionStatus": "resolved"}]


def books_ok():
    return {"ty1": make_book("ty1", bids=((0.30, 500),), asks=((0.31, 500), (0.33, 500))),
            "tn1": make_book("tn1", bids=((0.68, 500),), asks=((0.70, 500),))}


class Harness:
    def __init__(self, analyzer=None, books=None, db=":memory:", http_routes=None, **risk_kw):
        self.clock = ManualClock(T0)
        self.settings = settings(**risk_kw)
        self.journal = Journal(db)
        self.markets = [make_market("m1"), make_market("m2", "Will the Acme Corp CEO resign in 2026?")]
        self.catalog = MarketCatalog(FakeHttp(http_routes or {"/markets": RESOLVED_YES}), self.clock,
                                     CatalogSettings())
        self.catalog.markets = {m.market_id: m for m in self.markets}
        self.catalog.index = MarketIndex(self.markets)
        self.books = FakeBooks(self.clock, books or books_ok())
        self.analyzer = analyzer or StubAnalyzer(analysis(assessment("m1", p=0.8, low=0.7, high=0.9,
                                                                     max_entry=0.6)))
        self.executor = PaperExecutor(self.books, self.clock, self.settings.execution)
        self.risk = RiskManager(self.settings, self.journal, self.clock, [])
        self.tracker = OutcomeTracker(self.settings, self.journal, self.books, self.catalog, self.executor,
                                      self.clock)
        self.engine = NewsTradingEngine(self.settings, catalog=self.catalog, books=self.books,
                                        analyzer=self.analyzer, executor=self.executor, journal=self.journal,
                                        risk=self.risk, clock=self.clock, tracker=self.tracker)

    def rows(self, model):
        with self.journal.Session() as s:
            return list(s.scalars(select(model)))


class TestPipeline(unittest.IsolatedAsyncioTestCase):
    async def test_news_to_paper_trade_end_to_end(self):
        h = Harness()
        out = await h.engine.process_event(IngestedEvent(make_event(), "new"))
        self.assertEqual(out.status, "traded", out.detail)
        self.assertEqual(h.analyzer.calls, 1)
        [pos] = h.rows(PositionRow)
        self.assertEqual(pos.outcome, "YES")
        self.assertLessEqual(pos.entry_cost, h.settings.risk.max_trade_usd + 1e-9)
        self.assertAlmostEqual(pos.baseline_price, 0.31)
        [order] = h.rows(OrderRow)
        self.assertEqual(order.status, "FILLED")
        self.assertEqual(order.mode, "paper")
        sig = [s for s in h.rows(SignalRow) if s.decision == "TRADE"][0]
        for ts in (sig.first_seen_at, sig.matched_at, sig.ai_started_at, sig.ai_completed_at, sig.decision_at):
            self.assertIsNotNone(ts)
        self.assertGreater(sig.edge_after_costs, h.settings.risk.min_edge)
        self.assertEqual(len([m for m in h.rows(MarkRow) if m.signal_id == sig.signal_id]),
                         len(h.settings.tracking.mark_horizons_seconds))

    async def test_ai_failure_means_no_trade(self):
        h = Harness(analyzer=StubAnalyzer(error="ai_timeout"))
        out = await h.engine.process_event(IngestedEvent(make_event(), "new"))
        self.assertEqual(out.status, "ai_failed")
        self.assertEqual(h.rows(OrderRow), [])
        sigs = h.rows(SignalRow)
        self.assertTrue(sigs and all(s.decision == "REJECT" and s.action == "NO_TRADE" for s in sigs))
        self.assertIn("ai_failed", sigs[0].rejection_codes)

    async def test_halt_blocks_before_paid_ai(self):
        h = Harness()
        h.settings.emergency_stop = True
        out = await h.engine.process_event(IngestedEvent(make_event(), "new"))
        self.assertEqual(out.status, "prefilter_rejected")
        self.assertEqual(h.analyzer.calls, 0)
        self.assertEqual(h.rows(OrderRow), [])

    async def test_duplicate_story_does_not_trade_twice(self):
        h = Harness()
        await h.engine.process_event(IngestedEvent(make_event(), "new"))
        again = make_event(event_id="e2", cluster_id="e1")
        out = await h.engine.process_event(IngestedEvent(again, "upgrade"))
        self.assertNotEqual(out.status, "traded")
        self.assertIn("duplicate_signal", out.detail)
        self.assertEqual(len(h.rows(OrderRow)), 1)
        dup = await h.engine.process_event(IngestedEvent(make_event(event_id="e3", cluster_id="e1"), "duplicate"))
        self.assertEqual(dup.status, "duplicate")

    async def test_pretrade_recheck_catches_faster_bot(self):
        b = books_ok()
        moved = make_book("ty1", bids=((0.60, 500),), asks=((0.62, 500),))
        b["ty1"] = [b["ty1"], moved]  # analysed at 0.31, re-fetched at 0.62
        h = Harness(books=b)
        out = await h.engine.process_event(IngestedEvent(make_event(), "new"))
        self.assertEqual(out.status, "signals_rejected")
        self.assertIn("edge_insufficient", out.detail)
        self.assertEqual(h.rows(OrderRow), [])

    async def test_model_no_trade_and_irrelevant_market(self):
        h = Harness(analyzer=StubAnalyzer(analysis(assessment("m1", action="NO_TRADE"))))
        out = await h.engine.process_event(IngestedEvent(make_event(), "new"))
        self.assertEqual(out.status, "signals_rejected")
        self.assertEqual(h.rows(OrderRow), [])

    async def test_book_outage_rejects_market(self):
        from src.news_trader.http import NetworkError
        b = books_ok()
        b["ty1"] = NetworkError("connection reset")
        h = Harness(books=b)
        out = await h.engine.process_event(IngestedEvent(make_event(), "new"))
        self.assertEqual(out.status, "markets_rejected")
        self.assertIn("book_unavailable", out.detail)
        self.assertEqual(h.analyzer.calls, 0)

    async def test_concurrent_events_cannot_jointly_breach_correlated_cap(self):
        import asyncio
        b = books_ok()
        b["ty3"] = make_book("ty3", bids=((0.30, 500),), asks=((0.31, 500),))
        b["tn3"] = make_book("tn3", bids=((0.68, 500),), asks=((0.70, 500),))
        stub = StubAnalyzer(analysis(
            assessment("m1", p=0.8, low=0.7, high=0.9, max_entry=0.6),
            assessment("m3", p=0.8, low=0.7, high=0.9, max_entry=0.6, evidence=["agrees to acquire Initech"])))
        h = Harness(analyzer=stub, books=b, max_correlated_exposure_usd=12.0, max_trade_usd=10.0)
        m3 = make_market("m3", "Will Globex acquire Initech by December 31, 2026?", yes="ty3", no="tn3",
                         event_id="ev1")  # same Gamma event group as m1 => correlated
        h.catalog.markets["m3"] = m3
        h.catalog.index = MarketIndex(list(h.catalog.markets.values()))
        e2 = make_event("Globex agrees to acquire Initech for $2 billion", event_id="e2")
        await asyncio.gather(h.engine.process_event(IngestedEvent(make_event(), "new")),
                             h.engine.process_event(IngestedEvent(e2, "new")))
        total = sum(p.entry_cost for p in h.rows(PositionRow))
        self.assertLessEqual(total, 12.0 + 1e-9)  # without the exec lock both size to $10 (~$20)
        bound = sorted(s.binding_limit for s in h.rows(SignalRow) if s.decision == "TRADE")
        self.assertEqual(bound, ["correlated_exposure", "max_trade"])

    async def test_concurrent_items_from_same_story_trade_once(self):
        import asyncio
        h = Harness()
        await asyncio.gather(h.engine.process_event(IngestedEvent(make_event(), "new")),
                             h.engine.process_event(IngestedEvent(make_event(event_id="e2", cluster_id="e1"),
                                                                  "upgrade")))
        self.assertEqual(len(h.rows(OrderRow)), 1)
        self.assertIn("duplicate_signal", [c for s in h.rows(SignalRow) for c in json.loads(s.rejection_codes)])

    async def test_events_wait_for_catalog_instead_of_being_dropped(self):
        import asyncio
        from types import SimpleNamespace
        h = Harness()
        markets = h.catalog.markets
        h.catalog.markets = {}  # catalog still loading at startup
        q = asyncio.Queue()
        q.put_nowait(IngestedEvent(make_event(), "new"))
        h.engine.ingestion = SimpleNamespace(queue=q)
        stop = asyncio.Event()
        task = asyncio.create_task(h.engine._consumer(stop, asyncio.Semaphore(2)))
        await asyncio.sleep(0.2)
        self.assertEqual(q.qsize(), 1)  # held, not rejected
        h.catalog.markets = markets
        for _ in range(40):
            await asyncio.sleep(0.05)
            if h.rows(OrderRow):
                break
        stop.set()
        await task
        self.assertEqual(len(h.rows(OrderRow)), 1)

    async def test_stale_news_never_reaches_ai(self):
        h = Harness()
        out = await h.engine.process_event(IngestedEvent(make_event(published=T0 - 7200, first_seen=T0 - 7000), "new"))
        self.assertEqual(out.status, "prefilter_rejected")
        self.assertEqual(h.analyzer.calls, 0)


class TestTrackingAndAnalytics(unittest.IsolatedAsyncioTestCase):
    async def test_resolution_settles_position_and_report(self):
        h = Harness()
        await h.engine.process_event(IngestedEvent(make_event(), "new"))
        h.clock.advance(4000)
        await h.tracker.run_once()
        [pos] = h.rows(PositionRow)
        self.assertEqual(pos.status, "RESOLVED")
        self.assertAlmostEqual(pos.realized_pnl, pos.shares * 1.0 - pos.entry_cost)
        self.assertTrue(pos.resolved_yes)
        self.assertTrue(all(m.observed_at is not None for m in h.rows(MarkRow) if m.due_at <= h.clock.now()))
        rep = performance_report(h.journal)
        self.assertEqual(rep["performance"]["trades"], 1)
        self.assertEqual(rep["performance"]["win_rate"], 1.0)
        self.assertEqual(rep["fills"]["filled"], 1)
        self.assertEqual(rep["calibration"]["resolved_signals"], 1)
        self.assertIsNotNone(rep["calibration"]["brier_model"])
        self.assertIn("immediate_entry_pnl", rep["vs_immediate_entry"])

    async def test_mark_to_market_feeds_loss_limit(self):
        b = books_ok()
        h = Harness(books=b, daily_loss_limit_usd=2.0)
        await h.engine.process_event(IngestedEvent(make_event(), "new"))
        b["ty1"] = make_book("ty1", bids=((0.02, 500),), asks=((0.03, 500),))  # collapse
        h.catalog.http.routes = {"/markets": [{"closed": False}]}
        await h.tracker.mark_positions()
        [pos] = h.rows(PositionRow)
        self.assertLess(pos.unrealized_pnl, -2.0)
        self.assertIn("daily_loss_limit", [r.code for r in h.risk.pre_execution(h.risk.exposure())])

    def test_parse_resolution_variants(self):
        self.assertEqual(parse_resolution(RESOLVED_YES[0]), (True, [1.0, 0.0]))
        self.assertEqual(parse_resolution({"closed": True, "outcomePrices": ["0.0005", "0.9995"]}), (True, [0.0, 1.0]))
        self.assertEqual(parse_resolution({"closed": True, "outcomePrices": '["0.5","0.5"]',
                                           "umaResolutionStatus": "resolved"}), (True, [0.5, 0.5]))
        self.assertIsNone(parse_resolution({"closed": False, "outcomePrices": '["1","0"]'}))
        self.assertIsNone(parse_resolution({"closed": True, "outcomePrices": '["0.6","0.4"]'}))
        self.assertIsNone(parse_resolution({"closed": True, "outcomePrices": "garbage"}))

    def test_max_drawdown(self):
        self.assertEqual(max_drawdown([5, -3, -4, 10, -2]), 7)
        self.assertEqual(max_drawdown([]), 0)


class TestNoLookAhead(unittest.IsolatedAsyncioTestCase):
    def snap(self, t, ask):
        return BookSnapshotRow(token_id="ty1", market_id="m1", observed_at=t, exchange_ts=t, tick_size=0.01,
                               min_order_size=5, bids=json.dumps([[ask - 0.01, 100]]),
                               asks=json.dumps([[ask, 100]]))

    def test_store_never_returns_future_books(self):
        store = PointInTimeBookStore([self.snap(T0 + 10, 0.40), self.snap(T0 + 20, 0.60)])
        with self.assertRaises(LookAheadError):
            store.get("ty1", T0 + 9.999)
        self.assertEqual(store.get("ty1", T0 + 15).best_ask, 0.40)
        self.assertEqual(store.get("ty1", T0 + 20).best_ask, 0.60)
        self.assertEqual(store.get("ty1", T0 + 15).fetched_at, T0 + 10)  # true age preserved

    async def test_replay_reproduces_decision_without_ai_calls(self):
        with tempfile.TemporaryDirectory() as d:
            h = Harness(db=str(Path(d) / "src.db"))
            await h.engine.process_event(IngestedEvent(make_event(), "new"))
            target = Journal(str(Path(d) / "replay.db"))
            report = await replay(h.journal, target, h.settings)
            self.assertEqual(report["events"]["by_status"].get("traded"), 1)
            self.assertEqual(report["ai"]["cost_usd"], 0.0)
            # Tighter thresholds on the same history: no trade, still no AI spend.
            strict = settings(min_edge=0.5)
            report2 = await replay(h.journal, Journal(str(Path(d) / "replay2.db")), strict)
            self.assertNotIn("traded", report2["events"]["by_status"])


if __name__ == "__main__":
    unittest.main()
