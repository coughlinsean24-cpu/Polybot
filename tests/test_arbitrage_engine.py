"""Tests for ArbitrageEngine - BTC vs target price trade decisions."""

import unittest

from src.arbitrage_engine import ArbitrageEngine, TradeDecision
from src.polymarket_client import MarketInfo


def make_config():
    return {
        "strategy": {
            "initial_bet": 5,
            "max_bet": 160,
            "reset_profit_target": 200,
            "min_bankroll": 50,
            "min_edge": 0.05,
            "max_slippage": 0.02,
        },
    }


def make_market(up_price=0.50, down_price=0.50):
    return MarketInfo(
        condition_id="test",
        question="BTC Up or Down 5m",
        up_token_id="up_tok",
        down_token_id="down_tok",
        up_price=up_price,
        down_price=down_price,
        end_date="",
        volume=1000,
        event_start_time="",
        slug="btc-updown-5m-0",
    )


class TestArbitrageEngine(unittest.TestCase):
    def setUp(self):
        self.engine = ArbitrageEngine(make_config())

    def test_btc_above_target_bets_up(self):
        """BTC above target → should bet UP."""
        market = make_market(up_price=0.50, down_price=0.50)
        decision = self.engine.analyze_opportunity(
            btc_price=68600, target_price=68500, market=market, seconds_left=200
        )
        self.assertEqual(decision.direction, "UP")

    def test_btc_below_target_bets_down(self):
        """BTC below target → should bet DOWN."""
        market = make_market(up_price=0.50, down_price=0.50)
        decision = self.engine.analyze_opportunity(
            btc_price=68400, target_price=68500, market=market, seconds_left=200
        )
        self.assertEqual(decision.direction, "DOWN")

    def test_rejects_wide_spread(self):
        """Should reject when spread is too wide."""
        market = make_market(up_price=0.55, down_price=0.50)  # spread = 0.05
        decision = self.engine.analyze_opportunity(
            btc_price=68700, target_price=68500, market=market, seconds_left=200
        )
        self.assertFalse(decision.should_trade)
        self.assertIn("Spread", decision.reason)

    def test_rejects_small_edge(self):
        """Should reject when edge is below minimum."""
        # BTC barely above target, market already prices Up high
        market = make_market(up_price=0.70, down_price=0.30)
        decision = self.engine.analyze_opportunity(
            btc_price=68510, target_price=68500, market=market, seconds_left=200
        )
        self.assertFalse(decision.should_trade)
        self.assertIn("Edge too small", decision.reason)

    def test_accepts_good_opportunity(self):
        """Should accept when BTC strongly above target and odds are cheap."""
        market = make_market(up_price=0.50, down_price=0.50)
        # BTC is 0.15% above target → significant edge
        decision = self.engine.analyze_opportunity(
            btc_price=68600, target_price=68500, market=market, seconds_left=150
        )
        self.assertTrue(decision.should_trade)
        self.assertEqual(decision.direction, "UP")
        self.assertGreater(decision.edge, 0.05)

    def test_rejects_too_close_to_expiry(self):
        """Should not trade in the last 30 seconds."""
        market = make_market(up_price=0.50, down_price=0.50)
        decision = self.engine.analyze_opportunity(
            btc_price=68700, target_price=68500, market=market, seconds_left=20
        )
        self.assertFalse(decision.should_trade)
        self.assertIn("expiry", decision.reason)

    def test_probability_estimate_bounded(self):
        """Estimated probability should be between 0.45 and 0.95."""
        # Large gap
        prob = self.engine._estimate_probability(diff_pct=0.5, seconds_left=60)
        self.assertLessEqual(prob, 0.95)
        self.assertGreaterEqual(prob, 0.45)

        # Tiny gap
        prob = self.engine._estimate_probability(diff_pct=0.001, seconds_left=290)
        self.assertLessEqual(prob, 0.95)
        self.assertGreaterEqual(prob, 0.45)

    def test_rejects_tiny_btc_diff(self):
        """Should reject when BTC diff is too small (noise).
        Even with a loose min_edge, the 0.005% minimum diff filter kicks in.
        """
        engine = ArbitrageEngine({
            "strategy": {"min_edge": 0.001, "max_slippage": 0.10},
        })
        market = make_market(up_price=0.50, down_price=0.50)
        decision = engine.analyze_opportunity(
            btc_price=68501.0, target_price=68500, market=market, seconds_left=200
        )
        self.assertFalse(decision.should_trade)
        self.assertIn("noise", decision.reason)

    def test_less_time_increases_probability(self):
        """With same price gap, less time remaining → higher probability."""
        prob_early = self.engine._estimate_probability(diff_pct=0.05, seconds_left=250)
        prob_late = self.engine._estimate_probability(diff_pct=0.05, seconds_left=30)
        self.assertGreater(prob_late, prob_early)

    def test_bigger_gap_increases_probability(self):
        """With same time left, bigger gap → higher probability."""
        prob_small = self.engine._estimate_probability(diff_pct=0.01, seconds_left=150)
        prob_big = self.engine._estimate_probability(diff_pct=0.10, seconds_left=150)
        self.assertGreater(prob_big, prob_small)

    def test_ev_calculation(self):
        decision = TradeDecision(
            should_trade=True,
            direction="UP",
            edge=0.10,
            confidence=0.7,
            price=0.50,
            reason="test",
        )
        ev = self.engine.calculate_expected_value(decision, bet_size=10)
        # win_prob = 0.60, loss_prob = 0.40
        # payout_ratio = (1/0.5 - 1) = 1.0
        # ev = 0.60 * 1.0 * 10 - 0.40 * 10 = 6 - 4 = 2
        self.assertAlmostEqual(ev, 2.0)


if __name__ == "__main__":
    unittest.main()
