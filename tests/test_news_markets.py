"""Market catalog, resolution screening, matching, order books, fees and edge."""

import unittest

from src.news_trader.ai import AIAnalysis, CandidateContext
from src.news_trader.catalog import MarketCatalog, MarketIndex, market_tradeable, parse_market_page
from src.news_trader.clock import ManualClock
from src.news_trader.config import CatalogSettings, RiskSettings
from src.news_trader.edge import SizingLimits, blend_probability, kelly_fraction, plan_trade
from src.news_trader.http import SchemaError
from src.news_trader.matching import EventMarketMatcher
from src.news_trader.models import BookLevel
from src.news_trader.orderbook import parse_book, round_to_tick, taker_fee, walk_asks
from src.news_trader.resolution import check_resolution
from tests.news_fakes import T0, FakeHttp, analysis, assessment, make_book, make_event, make_market


def gamma_market(i, **kw):
    m = {
        "id": str(i), "question": f"Question {i}?", "conditionId": f"0x{i}", "slug": f"q-{i}",
        "description": 'This market will resolve to "Yes" if X happens by December 31, 2026. '
                      'Otherwise it will resolve to "No". The resolution source will be official statements.',
        "endDate": "2026-12-31T23:59:00Z", "outcomes": '["Yes", "No"]',
        "clobTokenIds": f'["{i}1", "{i}2"]', "active": True, "closed": False, "acceptingOrders": True,
        "enableOrderBook": True, "liquidityNum": 5000, "volume24hr": "1234.5", "bestBid": 0.4,
        "bestAsk": 0.42, "orderPriceMinTickSize": 0.01, "orderMinSize": 5, "feesEnabled": True,
        "feeSchedule": {"exponent": 1, "rate": 0.04, "takerOnly": True, "rebateRate": 0.25},
        "tags": [{"slug": "politics", "label": "Politics"}], "events": [{"id": "77", "title": "E"}],
    }
    m.update(kw)
    return m


class TestCatalog(unittest.TestCase):
    def test_parses_json_string_lists_and_fees(self):
        recs, bad = parse_market_page([gamma_market(1)])
        self.assertEqual(bad, 0)
        r = recs[0]
        self.assertEqual(r.token_ids, ["11", "12"])
        self.assertEqual(r.fee_rate, 0.04)
        self.assertEqual(r.category, "politics")
        self.assertEqual(r.event_id, "77")

    def test_invalid_and_non_binary_markets_skipped(self):
        page = [gamma_market(1), gamma_market(2, clobTokenIds="not json"), {"question": "no ids"},
                gamma_market(3, outcomes='["A","B","C"]', clobTokenIds='["a","b","c"]')]
        recs, bad = parse_market_page(page)
        self.assertEqual([r.market_id for r in recs], ["1"])
        self.assertEqual(bad, 2)

    def test_schema_change_keeps_previous_catalog(self):
        import asyncio
        clock = ManualClock(T0)
        broken = [{"id": str(i), "question": "q"} for i in range(30)]  # required fields gone
        cat = MarketCatalog(FakeHttp({"/markets": broken}), clock, CatalogSettings())
        cat.markets = {"keep": make_market("keep")}
        with self.assertRaises(SchemaError):
            asyncio.run(cat.refresh())
        self.assertIn("keep", cat.markets)
        with self.assertRaises(SchemaError):
            parse_market_page({"error": "unexpected"})

    def test_expired_suspended_and_resolving_markets_not_tradeable(self):
        ok, _ = market_tradeable(make_market(), T0, 3600)
        self.assertTrue(ok)
        cases = [
            make_market(closed=True), make_market(accepting_orders=False), make_market(active=False),
            make_market(end_ts=T0 - 1), make_market(end_ts=T0 + 600), make_market(end_ts=None),
            make_market(uma_resolution_status="proposed"), make_market(uma_resolution_status="disputed"),
        ]
        for m in cases:
            self.assertFalse(market_tradeable(m, T0, 3600)[0], m)


class TestResolution(unittest.TestCase):
    def test_explicit_rule_passes(self):
        self.assertTrue(check_resolution(make_market(), 0.5).ok)

    def test_reversed_if_clause_recognised(self):
        m = make_market(description='In the upcoming game, if the Cavaliers win, the market will resolve to '
                                    '"Cavaliers". If the 76ers win, the market will resolve to "76ers".')
        self.assertTrue(check_resolution(m, 0.5).has_explicit_rule)

    def test_ambiguous_rules_rejected(self):
        m = make_market(description="This market will resolve at the sole discretion of the market creator, "
                                    "in the opinion of whom the outcome may be resolved as appropriate. "
                                    "Ambiguous cases are subjective.")
        c = check_resolution(m, 0.5)
        self.assertFalse(c.ok)
        self.assertIn("discretionary resolution", c.issues)

    def test_missing_rules_rejected(self):
        self.assertFalse(check_resolution(make_market(description="TBD"), 0.5).ok)


class TestMatching(unittest.TestCase):
    def setUp(self):
        self.markets = [
            make_market("acq", "Will Acme Corp acquire Widget Inc by December 31, 2026?"),
            make_market("cap", "Will Acme Corp hit a $1T market cap in 2026?"),
            make_market("ceo", "Will the Acme Corp CEO resign in 2026?"),
            make_market("other", "Will Globex acquire Initech by December 31?"),
            make_market("sport", "Will Acme FC win the league?", category="sports"),
        ]
        cat = MarketCatalog(None, ManualClock(T0), CatalogSettings())
        cat.markets = {m.market_id: m for m in self.markets}
        cat.index = MarketIndex(self.markets)
        self.matcher = EventMarketMatcher(cat, max_candidates=4, min_score=0.3)

    def test_acquisition_matches_acquisition_market_not_every_company_market(self):
        ids = [m.market.market_id for m in self.matcher.match(make_event())]
        self.assertEqual(ids[0], "acq")
        self.assertNotIn("ceo", ids)       # action mismatch (acquisition vs resign)
        self.assertNotIn("sport", ids)     # category incompatible
        self.assertNotIn("other", ids)     # different companies

    def test_unrelated_headline_matches_nothing(self):
        self.assertEqual(self.matcher.match(make_event("Heavy rain expected across the region")), [])


class TestOrderBookAndFees(unittest.TestCase):
    def test_parse_book_normalises_reverse_sorted_lists(self):
        payload = {"asset_id": "t", "timestamp": "1792685431122", "tick_size": "0.01", "min_order_size": "5",
                   "neg_risk": False,
                   "bids": [{"price": "0.01", "size": "100"}, {"price": "0.45", "size": "10"}],
                   "asks": [{"price": "0.99", "size": "100"}, {"price": "0.47", "size": "20"}]}
        b = parse_book(payload, T0)
        self.assertEqual(b.best_bid, 0.45)
        self.assertEqual(b.best_ask, 0.47)
        self.assertAlmostEqual(b.exchange_ts, 1792685431.122)
        self.assertAlmostEqual(b.spread, 0.02)

    def test_parse_book_schema_change(self):
        with self.assertRaises(SchemaError):
            parse_book({"bids": "nope"}, T0)

    def test_fee_formula_matches_documented_examples(self):
        self.assertAlmostEqual(taker_fee(100, 0.50, 0.07), 1.75)
        self.assertAlmostEqual(taker_fee(100, 0.30, 0.07), 1.47)
        self.assertAlmostEqual(taker_fee(100, 0.70, 0.07), 1.47)
        self.assertAlmostEqual(taker_fee(100, 0.50, 0.04), 1.00)
        self.assertEqual(taker_fee(100, 0.50, 0.0), 0.0)

    def test_walk_asks_respects_limit_budget_and_haircut(self):
        asks = [BookLevel(0.40, 10), BookLevel(0.42, 100), BookLevel(0.50, 1000)]
        est = walk_asks(asks, budget_usd=1000, limit_price=0.45, fee_rate=0.0)
        self.assertAlmostEqual(est.shares, 110)
        self.assertTrue(est.depth_exhausted)
        self.assertEqual(est.worst_price, 0.42)
        half = walk_asks(asks, budget_usd=1000, limit_price=0.45, fee_rate=0.0, depth_fraction=0.5)
        self.assertAlmostEqual(half.shares, 55)
        small = walk_asks(asks, budget_usd=10, limit_price=0.45, fee_rate=0.04)
        self.assertLessEqual(small.cost, 10 + 1e-9)
        self.assertGreater(small.fees, 0)

    def test_round_to_tick_never_overpays(self):
        self.assertEqual(round_to_tick(0.6449, 0.01), 0.64)
        self.assertEqual(round_to_tick(0.646, 0.01), 0.64)
        self.assertEqual(round_to_tick(0.6459, 0.001), 0.645)
        self.assertEqual(round_to_tick(0.64, 0.01), 0.64)


class TestEdgeAndSizing(unittest.TestCase):
    def ctx(self, yes_asks=((0.40, 200), (0.41, 200)), no_asks=((0.61, 200),)):
        m = make_market()
        yes = make_book("ty1", bids=((0.39, 200),), asks=yes_asks)
        no = make_book("tn1", bids=((0.59, 200),), asks=no_asks)
        return CandidateContext(m, yes, no, check_resolution(m, 0.5), 0.9)

    def limits(self, cap=1000.0):
        return SizingLimits(cap, cap, cap, cap, cap)

    def a(self, **kw):
        return AIAnalysis.model_validate(analysis(assessment(**kw))).assessments[0]

    def test_helpers(self):
        self.assertAlmostEqual(blend_probability(0.8, 0.4, 0.5), 0.6)
        self.assertAlmostEqual(kelly_fraction(0.6, 0.4), 1 / 3)
        self.assertEqual(kelly_fraction(0.3, 0.4), 0.0)

    def test_buy_yes_plan_uses_shrunk_probability_and_executable_prices(self):
        risk = RiskSettings(min_edge=0.04, safety_margin=0.01, kelly_fraction=0.25, max_trade_usd=10)
        limits = SizingLimits(cash_available=500, trade_cap=10, market_cap=20, daily_cap=100, correlated_cap=30)
        plan, rs = plan_trade(self.a(p=0.80, low=0.7, high=0.9, max_entry=0.75), self.ctx(), risk, 0.5, 500,
                              limits)
        self.assertEqual(rs, [])
        self.assertAlmostEqual(plan.p_used_side, 0.395 + 0.5 * (0.80 - 0.395))
        self.assertLessEqual(plan.limit_price, plan.p_used_side - 0.05 + 1e-9)
        self.assertLessEqual(plan.fill.cost, 10 + 1e-9)
        self.assertEqual(plan.binding_limit, "max_trade")
        self.assertGreaterEqual(plan.edge_after_costs, 0.04)

    def test_no_edge_when_ask_above_justified_price(self):
        risk = RiskSettings(min_edge=0.04)
        plan, rs = plan_trade(self.a(p=0.45, low=0.4, high=0.5), self.ctx(), risk, 0.5, 500, self.limits())
        self.assertIsNone(plan)
        self.assertEqual(rs[0].code, "edge_insufficient")

    def test_buy_no_uses_no_book(self):
        risk = RiskSettings(min_edge=0.02, max_trade_usd=50)
        plan, rs = plan_trade(self.a(action="BUY_NO", p=0.10, low=0.05, high=0.2, max_entry=0.9),
                              self.ctx(no_asks=((0.50, 300),)), risk, 0.5, 500, self.limits())
        self.assertEqual(rs, [])
        self.assertEqual(plan.outcome, "NO")
        self.assertEqual(plan.token_id, "tn1")

    def test_size_below_exchange_minimum(self):
        plan, rs = plan_trade(self.a(p=0.8, low=0.7, high=0.9), self.ctx(), RiskSettings(), 0.5, 500,
                              self.limits(cap=1.0))
        self.assertIsNone(plan)
        self.assertEqual(rs[0].code, "size_below_minimum")

    def test_thin_depth_rejected(self):
        plan, rs = plan_trade(self.a(p=0.8, low=0.7, high=0.9), self.ctx(yes_asks=((0.40, 2),)), RiskSettings(),
                              0.5, 500, self.limits())
        self.assertIsNone(plan)
        self.assertEqual(rs[0].code, "depth_insufficient")

    def test_kelly_bound_when_smaller_than_caps(self):
        risk = RiskSettings(kelly_fraction=0.1, min_edge=0.01)
        plan, rs = plan_trade(self.a(p=0.8, low=0.7, high=0.9), self.ctx(), risk, 0.5, 500, self.limits(1000))
        self.assertEqual(rs, [])
        self.assertEqual(plan.binding_limit, "kelly")


if __name__ == "__main__":
    unittest.main()
