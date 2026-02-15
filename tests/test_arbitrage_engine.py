"""Tests for ArbitrageEngine - edge detection and trade decisions."""

import unittest

from src.arbitrage_engine import ArbitrageEngine, TradeDecision
from src.price_monitor import ArbitrageSignal
from src.polymarket_client import MarketInfo


def make_config():
    return {
        "strategy": {
            "initial_bet": 5,
            "max_bet": 160,
            "reset_profit_target": 200,
            "min_bankroll": 50,
            "min_price_delta": 0.15,
            "min_edge": 0.05,
            "max_slippage": 0.02,
        },
    }


def make_signal(direction="UP", delta=0.3, confidence=0.6):
    return ArbitrageSignal(
        direction=direction,
        delta_pct=delta,
        confidence=confidence,
        btc_price_start=100000,
        btc_price_end=100300 if direction == "UP" else 99700,
        timestamp=1000.0,
    )


def make_market(yes_price=0.55, no_price=0.45):
    return MarketInfo(
        condition_id="test",
        question="Will BTC go up?",
        yes_token_id="yes_tok",
        no_token_id="no_tok",
        yes_price=yes_price,
        no_price=no_price,
        end_date="",
        volume=1000,
    )


class TestArbitrageEngine(unittest.TestCase):
    def setUp(self):
        self.engine = ArbitrageEngine(make_config())

    def test_rejects_small_edge(self):
        """Should reject when edge is below minimum."""
        signal = make_signal(delta=0.15, confidence=0.1)
        # Price is already high enough that our estimate doesn't beat it
        market = make_market(yes_price=0.70, no_price=0.30)
        decision = self.engine.analyze_opportunity(signal, market, 0.70, 0.30)
        self.assertFalse(decision.should_trade)

    def test_rejects_wide_spread(self):
        """Should reject when spread is too wide."""
        signal = make_signal(delta=0.5, confidence=0.8)
        # Spread = |0.55 + 0.50 - 1.0| = 0.05, which > max_slippage of 0.02
        decision = self.engine.analyze_opportunity(signal, make_market(), 0.55, 0.50)
        self.assertFalse(decision.should_trade)
        self.assertIn("Spread", decision.reason)

    def test_accepts_good_opportunity(self):
        """Should accept when edge and spread are favorable."""
        signal = make_signal(delta=0.5, confidence=0.8)
        # Tight spread: 0.50 + 0.50 = 1.0
        decision = self.engine.analyze_opportunity(signal, make_market(), 0.50, 0.50)
        self.assertTrue(decision.should_trade)
        self.assertGreater(decision.edge, 0.05)

    def test_up_signal_bets_yes(self):
        signal = make_signal(direction="UP")
        decision = self.engine.analyze_opportunity(signal, make_market(), 0.50, 0.50)
        self.assertEqual(decision.direction, "YES")

    def test_down_signal_bets_no(self):
        signal = make_signal(direction="DOWN")
        decision = self.engine.analyze_opportunity(signal, make_market(), 0.50, 0.50)
        self.assertEqual(decision.direction, "NO")

    def test_probability_estimate_bounded(self):
        """Estimated probability should be between 0.40 and 0.85."""
        # Very large delta
        signal = make_signal(delta=5.0, confidence=1.0)
        prob = self.engine._estimate_probability(signal)
        self.assertLessEqual(prob, 0.85)
        self.assertGreaterEqual(prob, 0.40)

        # Very small delta
        signal = make_signal(delta=0.01, confidence=0.0)
        prob = self.engine._estimate_probability(signal)
        self.assertLessEqual(prob, 0.85)
        self.assertGreaterEqual(prob, 0.40)

    def test_ev_calculation(self):
        decision = TradeDecision(
            should_trade=True,
            direction="YES",
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
