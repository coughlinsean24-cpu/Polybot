"""Tests for PositionManager - bankroll tracking and progression logic."""

import time
import unittest

from src.position_manager import PositionManager


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
        "risk": {
            "max_concurrent_positions": 1,
            "cooldown_after_loss": 60,
            "max_consecutive_losses": 3,
        },
    }


class TestPositionManager(unittest.TestCase):
    def setUp(self):
        self.pm = PositionManager(make_config())

    def test_initial_state(self):
        self.assertEqual(self.pm.bankroll, 200.0)
        self.assertEqual(self.pm.current_bet_size, 5)
        self.assertEqual(self.pm.consecutive_wins, 0)
        self.assertEqual(self.pm.total_trades, 0)

    def test_doubling_on_wins(self):
        """Bet should double after each win."""
        self.pm.update_after_win(5)
        self.assertEqual(self.pm.current_bet_size, 10)

        self.pm.update_after_win(10)
        self.assertEqual(self.pm.current_bet_size, 20)

        self.pm.update_after_win(20)
        self.assertEqual(self.pm.current_bet_size, 40)

        self.pm.update_after_win(40)
        self.assertEqual(self.pm.current_bet_size, 80)

        self.pm.update_after_win(80)
        self.assertEqual(self.pm.current_bet_size, 160)

    def test_max_bet_cap(self):
        """Bet should not exceed max_bet."""
        self.pm.current_bet_size = 160
        self.pm.update_after_win(160)
        self.assertEqual(self.pm.current_bet_size, 160)

    def test_reset_on_loss(self):
        """Bet should reset to initial after a loss."""
        self.pm.update_after_win(5)
        self.pm.update_after_win(10)
        self.assertEqual(self.pm.current_bet_size, 20)

        self.pm.update_after_loss(20)
        self.assertEqual(self.pm.current_bet_size, 5)

    def test_bankroll_tracking(self):
        """Bankroll should update correctly on wins and losses."""
        self.pm.update_after_win(10)
        self.assertEqual(self.pm.bankroll, 210.0)

        self.pm.update_after_loss(5)
        self.assertEqual(self.pm.bankroll, 205.0)

    def test_win_rate(self):
        self.pm.update_after_win(5)
        self.pm.update_after_win(5)
        self.pm.update_after_loss(5)
        self.assertAlmostEqual(self.pm.win_rate(), 2 / 3)

    def test_circuit_breaker_bankroll(self):
        """Should block trading when bankroll is too low."""
        self.pm.bankroll = 40
        can, reason = self.pm.can_trade()
        self.assertFalse(can)
        self.assertIn("below minimum", reason)

    def test_circuit_breaker_consecutive_losses(self):
        """Should block trading after too many consecutive losses."""
        self.pm.consecutive_losses = 3
        can, reason = self.pm.can_trade()
        self.assertFalse(can)
        self.assertIn("consecutive losses", reason)

    def test_cooldown_after_loss(self):
        """Should enforce cooldown period after a loss."""
        self.pm.update_after_loss(5)
        can, reason = self.pm.can_trade()
        self.assertFalse(can)
        self.assertIn("cooldown", reason)

    def test_session_reset_on_profit_target(self):
        """Session should reset when profit target is hit."""
        # Simulate hitting $200 profit
        self.pm.update_after_win(200)
        # After reset: bet goes back to initial, session_profit resets
        self.assertEqual(self.pm.current_bet_size, 5)
        self.assertEqual(self.pm.session_profit, 0.0)
        self.assertEqual(self.pm.consecutive_wins, 0)

    def test_full_winning_streak(self):
        """Simulate the proposed $5->$10->$20->$40->$80->$160 streak.
        After win 6, session profit = $315 which exceeds reset target ($200),
        so bet resets to initial $5 and session_profit resets to 0.
        """
        profits = [5, 10, 20, 40, 80, 160]
        # After 6th win, session reset triggers: bet goes back to $5
        expected_bets = [10, 20, 40, 80, 160, 5]

        for i, profit in enumerate(profits):
            self.pm.update_after_win(profit)
            self.assertEqual(
                self.pm.current_bet_size, expected_bets[i],
                f"After win {i+1}, bet should be {expected_bets[i]}"
            )

        total = sum(profits)
        self.assertEqual(self.pm.bankroll, 200 + total)
        # Session profit should have been reset
        self.assertEqual(self.pm.session_profit, 0.0)


class TestPositionManagerState(unittest.TestCase):
    def test_get_state(self):
        pm = PositionManager(make_config())
        pm.update_after_win(5)
        state = pm.get_state()
        self.assertEqual(state.bankroll, 205.0)
        self.assertEqual(state.consecutive_wins, 1)
        self.assertEqual(state.total_trades, 1)


if __name__ == "__main__":
    unittest.main()
