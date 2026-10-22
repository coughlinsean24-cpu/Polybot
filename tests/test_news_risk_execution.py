"""Risk gate, kill switches, exposure limits, paper/live execution and authorisation."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.news_trader import execution as X
from src.news_trader.ai import AIAnalysis, CandidateContext
from src.news_trader.clock import ManualClock
from src.news_trader.config import ExecutionSettings
from src.news_trader.execution import (
    AuthorizationStatus, LiveAuthorization, LiveExecutor, OrderRequest, PaperExecutor,
)
from src.news_trader.journal import Journal
from src.news_trader.models import Verification
from src.news_trader.resolution import check_resolution
from src.news_trader.risk import RiskManager
from tests.news_fakes import T0, FakeBooks, FakeHttp, analysis, assessment, make_book, make_event, make_market, settings


def risk_mgr(clock=None, halt_files=(), **risk_kw):
    clock = clock or ManualClock(T0)
    s = settings(**risk_kw)
    j = Journal(":memory:")
    return RiskManager(s, j, clock, list(halt_files)), j, clock, s


def a(**kw):
    return AIAnalysis.model_validate(analysis(assessment(**kw))).assessments[0]


def open_pos(j, pid, market="m1", group="ev1", cluster="c1", cost=10.0, opened=T0, unreal=0.0):
    j.open_position(position_id=pid, signal_id=pid, mode="paper", market_id=market, condition_id=f"0x{market}",
                    question="q", event_group=group, cluster_id=cluster, token_id="ty1", outcome="YES",
                    outcome_label="Yes", market_category="corporate", source_name="s", provider="p", shares=20,
                    entry_price=0.5, entry_cost=cost, entry_fees=0.0, p_used=0.6, baseline_price=0.5,
                    news_age_at_decision=10, detection_to_decision=5, opened_at=opened, unrealized_pnl=unreal)


class TestHalts(unittest.TestCase):
    def test_global_emergency_stop_file_halts(self):
        with tempfile.TemporaryDirectory() as d:
            stop = Path(d) / "EMERGENCY_STOP"
            rm, *_ = risk_mgr(halt_files=[stop])
            self.assertIsNone(rm.halted())
            stop.write_text("x")
            self.assertIn("EMERGENCY_STOP", rm.halted())
            self.assertIn("halted", [r.code for r in rm.pre_event(make_event())])

    def test_config_emergency_stop_and_disable(self):
        rm, _, _, s = risk_mgr()
        s.emergency_stop = True
        self.assertIsNotNone(rm.halted())
        s.emergency_stop, s.enabled = False, False
        self.assertIsNotNone(rm.halted())

    def test_daily_loss_limit_latches_halt(self):
        rm, j, clock, _ = risk_mgr(daily_loss_limit_usd=5)
        open_pos(j, "p1", unreal=-6.0)
        rs = rm.pre_execution(rm.exposure())
        self.assertIn("daily_loss_limit", [r.code for r in rs])
        j.update_position("p1", unrealized_pnl=0.0)
        self.assertIsNotNone(rm.halted())  # stays halted until restart


class TestEventChecks(unittest.TestCase):
    def codes(self, ev, **kw):
        rm, *_ = risk_mgr(**kw)
        return [r.code for r in rm.pre_event(ev)]

    def test_fresh_primary_event_passes(self):
        self.assertEqual(self.codes(make_event()), [])

    def test_stale_news(self):
        self.assertIn("stale_news", self.codes(make_event(published=T0 - 5000, first_seen=T0 - 4000)))

    def test_fake_or_unverified_story(self):
        ev = make_event(verification=Verification.UNVERIFIED, credibility=0.3, primary=False)
        c = self.codes(ev)
        self.assertIn("unverified", c)
        self.assertIn("low_credibility", c)

    def test_conflicting_reports(self):
        self.assertIn("conflicting_reports", self.codes(make_event(verification=Verification.CONFLICTING)))

    def test_single_source_needs_corroboration_in_politics(self):
        ev = make_event(category="politics", verification=Verification.SINGLE_SOURCE, primary=False,
                        provider="cnbc_top", credibility=0.8)
        self.assertIn("awaiting_corroboration", self.codes(ev))
        ev.verification = Verification.CORROBORATED
        self.assertNotIn("awaiting_corroboration", self.codes(ev))

    def test_speculative_and_category_filter(self):
        self.assertIn("speculative", self.codes(make_event(flags=["speculative"])))
        rm, _, _, s = risk_mgr()
        s.categories = ["sports"]
        self.assertIn("category_disabled", [r.code for r in rm.pre_event(make_event())])


class TestMarketAndSignalChecks(unittest.TestCase):
    def ctx(self, yes=None, no=None, market=None):
        m = market or make_market()
        return CandidateContext(m, yes or make_book("ty1"), no or make_book("tn1", bids=((0.66, 500),),
                                                                                 asks=((0.68, 500),)),
                                check_resolution(m, 0.5), 0.9)

    def test_stale_quote(self):
        rm, _, clock, _ = risk_mgr(max_quote_age_seconds=5)
        c = self.ctx(yes=make_book("ty1", at=T0 - 30))
        self.assertIn("stale_quote", [r.code for r in rm.pre_market(c)])

    def test_wide_spread_and_extreme_prices(self):
        rm, *_ = risk_mgr(max_spread=0.05)
        wide = self.ctx(yes=make_book("ty1", bids=((0.20, 500),), asks=((0.40, 500),)),
                        no=make_book("tn1", bids=((0.58, 500),), asks=((0.80, 500),)))
        self.assertIn("no_viable_side", [r.code for r in rm.pre_market(wide)])
        extreme = self.ctx(yes=make_book("ty1", bids=((0.97, 500),), asks=((0.98, 500),)),
                           no=make_book("tn1", bids=((0.01, 500),), asks=((0.02, 500),)))
        self.assertIn("no_viable_side", [r.code for r in rm.pre_market(extreme)])

    def test_suspended_market_and_ambiguous_rules(self):
        rm, *_ = risk_mgr()
        codes = [r.code for r in rm.pre_market(self.ctx(market=make_market(accepting_orders=False,
                                                                             description="TBD")))]
        self.assertIn("market_status", codes)
        self.assertIn("ambiguous_rules", codes)

    def test_post_ai_confidence_uncertainty_latency(self):
        rm, _, clock, _ = risk_mgr(min_confidence=0.7, max_uncertainty=0.2, max_decision_latency_seconds=60)
        ev = make_event()
        self.assertEqual(rm.post_ai(a(), ev), [])
        codes = [r.code for r in rm.post_ai(a(conf=0.5, low=0.2, high=0.9), ev)]
        self.assertIn("low_confidence", codes)
        self.assertIn("high_uncertainty", codes)
        clock.advance(120)
        self.assertIn("slow_decision", [r.code for r in rm.post_ai(a(), ev)])

    def test_model_requested_corroboration(self):
        rm, *_ = risk_mgr()
        ev = make_event(verification=Verification.SINGLE_SOURCE, primary=False)
        self.assertIn("awaiting_corroboration", [r.code for r in rm.post_ai(a(corroborate=True), ev)])


class TestExposure(unittest.TestCase):
    def test_limits_reflect_open_positions(self):
        rm, j, _, _ = risk_mgr(max_market_exposure_usd=20, max_correlated_exposure_usd=30,
                               max_daily_exposure_usd=100, paper_bankroll_usd=500)
        open_pos(j, "p1", market="m1", group="fed", cost=15)
        open_pos(j, "p2", market="m2", group="fed", cost=10)
        lim = rm.sizing_limits(rm.exposure(), "m1", "fed", "new-cluster")
        self.assertAlmostEqual(lim.market_cap, 5)
        self.assertAlmostEqual(lim.correlated_cap, 5)  # same Gamma event group
        self.assertAlmostEqual(lim.daily_cap, 75)
        self.assertAlmostEqual(lim.cash_available, 475)

    def test_insufficient_paper_balance(self):
        rm, j, _, _ = risk_mgr(paper_bankroll_usd=20, max_market_exposure_usd=1000)
        open_pos(j, "p1", market="m9", group="g9", cost=19)
        cap, name = rm.sizing_limits(rm.exposure(), "m1", "g1", "c").binding()
        self.assertEqual(name, "cash")
        self.assertAlmostEqual(cap, 1)

    def test_max_open_positions(self):
        rm, j, _, _ = risk_mgr(max_open_positions=1)
        open_pos(j, "p1")
        self.assertIn("max_positions", [r.code for r in rm.pre_execution(rm.exposure())])


class TestPaperExecutor(unittest.IsolatedAsyncioTestCase):
    def req(self, cost=10.0, limit=0.42, oid="o1"):
        return OrderRequest(oid, make_market(), "YES", "ty1", limit, cost, 0.0, T0)

    def ex(self, books, clock, **kw):
        es = ExecutionSettings(paper_latency_ms=500, order_timeout_seconds=6, poll_interval_seconds=2, **kw)
        return PaperExecutor(FakeBooks(clock, books), clock, es)

    async def test_fill_uses_book_after_latency_with_haircut(self):
        clock = ManualClock(T0)
        # Decision saw a deep 0.40 ask; by arrival a faster bot took it.
        after = make_book("ty1", asks=((0.41, 10), (0.50, 1000)))
        res = await self.ex({"ty1": [after]}, clock, paper_depth_haircut=0.5).execute(self.req(cost=100))
        self.assertEqual(res.status, "PARTIAL")
        self.assertAlmostEqual(res.filled_shares, 5.0)  # half of 10 displayed at 0.41
        self.assertAlmostEqual(res.avg_price, 0.41)
        self.assertGreaterEqual(res.submitted_at, T0 + 0.5)

    async def test_unfilled_when_market_moved_away(self):
        clock = ManualClock(T0)
        res = await self.ex({"ty1": [make_book("ty1", asks=((0.60, 1000),))]}, clock).execute(self.req())
        self.assertEqual(res.status, "UNFILLED")
        self.assertEqual(res.filled_shares, 0)

    async def test_resting_order_fills_only_on_new_crossing_liquidity(self):
        clock = ManualClock(T0)
        seq = [make_book("ty1", asks=((0.45, 100),)),             # nothing at limit 0.42
               make_book("ty1", asks=((0.45, 100),)),             # unchanged: no fill
               make_book("ty1", asks=((0.40, 30), (0.45, 100)))]  # new seller below limit
        res = await self.ex({"ty1": seq}, clock, paper_depth_haircut=1.0).execute(self.req(cost=100))
        self.assertAlmostEqual(res.filled_shares, 30)
        self.assertEqual(res.status, "PARTIAL")

    async def test_full_fill(self):
        clock = ManualClock(T0)
        res = await self.ex({"ty1": [make_book("ty1", asks=((0.40, 1000),))]}, clock).execute(self.req(cost=10))
        self.assertEqual(res.status, "FILLED")
        self.assertLessEqual(res.notional + res.fees, 10 + 1e-9)

    async def test_duplicate_order_rejected(self):
        clock = ManualClock(T0)
        ex = self.ex({"ty1": [make_book("ty1", asks=((0.40, 1000),))]}, clock)
        await ex.execute(self.req())
        self.assertEqual((await ex.execute(self.req())).status, "REJECTED")

    async def test_network_failure_and_stale_book_at_submit(self):
        clock = ManualClock(T0)
        from src.news_trader.http import NetworkError
        res = await self.ex({"ty1": [NetworkError("reset")]}, clock).execute(self.req())
        self.assertEqual(res.status, "REJECTED")


class TestLiveGates(unittest.IsolatedAsyncioTestCase):
    async def test_denied_by_default_with_every_reason(self):
        s = settings()
        auth = await LiveAuthorization(s, cli_flag=False, http=None, env={}).evaluate()
        self.assertFalse(auth.granted)
        text = " ".join(auth.reasons)
        for needle in ("LIVE_TRADING_RELEASED", "mode is not 'live'", "--enable-live-news-trading",
                       "NEWS_TRADER_LIVE_ACK", "POLYMARKET_PRIVATE_KEY", "geoblock"):
            self.assertIn(needle, text)

    async def test_dashboard_mode_toggle_alone_is_not_enough(self):
        s = settings()
        s.mode = "live"
        env = {LiveAuthorization.ACK_ENV: LiveAuthorization.ACK_VALUE, "POLYMARKET_PRIVATE_KEY": "k"}
        http = FakeHttp({"geoblock": {"blocked": False, "country": "XX"}})
        auth = await LiveAuthorization(s, cli_flag=True, http=http, env=env).evaluate()
        self.assertFalse(auth.granted)  # code release flag still False
        self.assertEqual(len(auth.reasons), 1)

    async def test_geoblocked_region_never_authorised(self):
        s = settings()
        s.mode = "live"
        env = {LiveAuthorization.ACK_ENV: LiveAuthorization.ACK_VALUE, "POLYMARKET_PRIVATE_KEY": "k"}
        with mock.patch.object(X, "LIVE_TRADING_RELEASED", True):
            blocked = FakeHttp({"geoblock": {"blocked": True, "country": "US", "region": "NY"}})
            auth = await LiveAuthorization(s, True, blocked, env).evaluate()
            self.assertFalse(auth.granted)
            failing = FakeHttp({"geoblock": RuntimeError("dns")})
            self.assertFalse((await LiveAuthorization(s, True, failing, env).evaluate()).granted)
            ok = FakeHttp({"geoblock": {"blocked": False}})
            self.assertTrue((await LiveAuthorization(s, True, ok, env).evaluate()).granted)

    def test_live_executor_refuses_without_authorisation(self):
        with self.assertRaises(PermissionError):
            LiveExecutor(object(), AuthorizationStatus(False, ["no"]), None, ManualClock(T0), ExecutionSettings())
        with self.assertRaises(PermissionError):  # even a granted status needs the code flag
            LiveExecutor(object(), AuthorizationStatus(True, []), None, ManualClock(T0), ExecutionSettings())


class FakePoly:
    def __init__(self, balance=100.0, statuses=None, success=True):
        self.balance = balance
        self.statuses = list(statuses or [])
        self.success = success
        self.placed, self.cancelled = [], []

    def get_live_balance(self):
        return self.balance

    def place_limit_order(self, market, side, shares, price):
        self.placed.append((side, shares, price))
        return type("R", (), {"success": self.success, "order_id": "ex1" if self.success else None,
                              "error": None if self.success else "rejected"})()

    def check_position_status(self, oid):
        return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]

    def cancel_order(self, oid):
        self.cancelled.append(oid)
        return True


class TestLiveExecutorLogic(unittest.IsolatedAsyncioTestCase):
    def make(self, poly, clock):
        books = FakeBooks(clock, {"ty1": [make_book("ty1", asks=((0.40, 1000),))]})
        es = ExecutionSettings(order_timeout_seconds=4, poll_interval_seconds=1)
        with mock.patch.object(X, "LIVE_TRADING_RELEASED", True):
            return LiveExecutor(poly, AuthorizationStatus(True, []), books, clock, es)

    def req(self, limit=0.4459):
        return OrderRequest("c1", make_market(tick_size=0.001), "YES", "ty1", limit, 10.0, 0.0, T0)

    async def test_partial_fill_cancels_remainder_and_never_assumes_fill(self):
        clock = ManualClock(T0)
        poly = FakePoly(statuses=[{"status": "LIVE", "filled": 0}, {"status": "LIVE", "filled": 5.0}])
        res = await self.make(poly, clock).execute(self.req())
        self.assertEqual(res.status, "PARTIAL")
        self.assertEqual(res.filled_shares, 5.0)
        self.assertEqual(poly.cancelled, ["ex1"])
        self.assertEqual(poly.placed[0][2], 0.44)  # rounded DOWN to the 0.01 grid, never above limit

    async def test_no_confirmation_means_no_fill(self):
        clock = ManualClock(T0)
        poly = FakePoly(statuses=[{"status": "error", "error": "timeout"}])
        res = await self.make(poly, clock).execute(self.req())
        self.assertEqual(res.status, "UNFILLED")
        self.assertEqual(res.filled_shares, 0)

    async def test_insufficient_balance_and_exchange_rejection(self):
        clock = ManualClock(T0)
        res = await self.make(FakePoly(balance=3.0, statuses=[{}]), clock).execute(self.req())
        self.assertEqual(res.status, "REJECTED")
        self.assertIn("insufficient balance", res.error)
        res = await self.make(FakePoly(success=False, statuses=[{}]), ManualClock(T0)).execute(self.req())
        self.assertEqual(res.status, "REJECTED")

    async def test_duplicate_client_order_id(self):
        clock = ManualClock(T0)
        poly = FakePoly(statuses=[{"status": "MATCHED", "filled": 22.42}])
        ex = self.make(poly, clock)
        await ex.execute(self.req())
        self.assertEqual((await ex.execute(self.req())).status, "REJECTED")
        self.assertEqual(len(poly.placed), 1)


if __name__ == "__main__":
    unittest.main()
