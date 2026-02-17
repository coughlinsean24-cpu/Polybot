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
            "bet_mode": "doubling",  # tests are written for doubling mode
            "kelly_fraction": 0.25,
        },
        "risk": {
            "max_concurrent_positions": 1,
            "cooldown_after_loss": 60,
            "max_consecutive_losses": 3,
        },
    }


def make_kelly_config():
    return {
        "strategy": {
            "initial_bet": 5,
            "max_bet": 50,
            "reset_profit_target": 200,
            "min_bankroll": 50,
            "min_price_delta": 0.15,
            "min_edge": 0.05,
            "max_slippage": 0.02,
            "bet_mode": "kelly",
            "kelly_fraction": 0.25,
        },
        "risk": {
            "max_concurrent_positions": 1,
            "cooldown_after_loss": 30,
            "max_consecutive_losses": 5,
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


class TestKellySizing(unittest.TestCase):
    """Tests for the Kelly criterion bet sizing mode."""

    def setUp(self):
        self.pm = PositionManager(make_kelly_config())

    def test_kelly_mode_set(self):
        self.assertEqual(self.pm.bet_mode, "kelly")
        self.assertEqual(self.pm.kelly_fraction, 0.25)

    def test_kelly_with_edge(self):
        """Kelly should bet more when edge is larger."""
        small_edge_bet = self.pm.calculate_bet_size(edge=0.05, probability=0.55)
        large_edge_bet = self.pm.calculate_bet_size(edge=0.20, probability=0.70)
        self.assertGreater(large_edge_bet, small_edge_bet)

    def test_kelly_respects_max_bet(self):
        """Kelly should not exceed max_bet."""
        self.pm.bankroll = 10000.0
        bet = self.pm.calculate_bet_size(edge=0.30, probability=0.80)
        self.assertLessEqual(bet, self.pm.max_bet)

    def test_kelly_respects_bankroll(self):
        """Kelly should not exceed bankroll (but never below initial_bet).

        The bankroll is a virtual Kelly tracker. When bankroll drops below
        initial_bet (e.g. after a loss), we still bet initial_bet because
        real solvency is checked by _check_balance_before_trade.
        """
        # When bankroll > initial_bet, Kelly is capped at bankroll
        self.pm.bankroll = 8.0
        bet = self.pm.calculate_bet_size(edge=0.20, probability=0.70)
        self.assertLessEqual(bet, 8.0)

        # When bankroll < initial_bet, bet is floored at initial_bet
        self.pm.bankroll = 3.0
        bet = self.pm.calculate_bet_size(edge=0.20, probability=0.70)
        self.assertEqual(bet, self.pm.initial_bet)

    def test_kelly_minimum_is_initial_bet(self):
        """Kelly should bet at least initial_bet."""
        bet = self.pm.calculate_bet_size(edge=0.001, probability=0.501)
        self.assertGreaterEqual(bet, self.pm.initial_bet)

    def test_kelly_no_edge_returns_initial(self):
        """With zero edge, Kelly falls back to initial_bet."""
        bet = self.pm.calculate_bet_size(edge=0.0, probability=0.50)
        self.assertEqual(bet, self.pm.initial_bet)

    def test_flat_mode(self):
        """Flat mode should always return initial_bet."""
        pm = PositionManager(make_config())
        pm.bet_mode = "flat"
        pm.bankroll = 10000.0
        bet = pm.calculate_bet_size(edge=0.30, probability=0.80)
        self.assertEqual(bet, pm.initial_bet)

    def test_kelly_scales_with_bankroll(self):
        """Larger bankroll should produce larger Kelly bets (up to max_bet)."""
        self.pm.bankroll = 200.0
        bet_small = self.pm.calculate_bet_size(edge=0.10, probability=0.60)

        self.pm.bankroll = 2000.0
        bet_large = self.pm.calculate_bet_size(edge=0.10, probability=0.60)

        self.assertGreater(bet_large, bet_small)


def make_compound_config():
    """Config with smart compounding fully enabled."""
    return {
        "strategy": {
            "initial_bet": 5,
            "max_bet": 50,
            "reset_profit_target": 200,
            "min_bankroll": 5,
            "min_price_delta": 0.15,
            "min_edge": 0.05,
            "max_slippage": 0.02,
            "bet_mode": "kelly",
            "kelly_fraction": 0.25,
            "compound": {
                "enabled": True,
                "profit_reinvest_pct": 0.50,
                "max_bet_pct": 0.10,
                "floor": 10.0,
                "ratchet_after": 5.0,
                "gears": {
                    "enabled": True,
                    "cold_win_rate": 0.45,
                    "hot_win_rate": 0.60,
                    "cold_reinvest_pct": 0.25,
                    "hot_reinvest_pct": 0.70,
                },
                "drawdown": {
                    "enabled": True,
                    "threshold": 0.15,
                    "multiplier": 0.50,
                    "severe_threshold": 0.30,
                    "severe_multiplier": 0.25,
                },
                "streak": {
                    "enabled": True,
                    "threshold": 3,
                    "boost_pct": 0.25,
                    "max_boost": 2.0,
                },
                "lock_in": {
                    "enabled": True,
                    "pct": 0.10,
                },
            },
        },
        "risk": {
            "max_concurrent_positions": 1,
            "cooldown_after_loss": 30,
            "max_consecutive_losses": 5,
        },
    }


class TestSmartCompounding(unittest.TestCase):
    """Tests for the gear-shifting compound system."""

    def setUp(self):
        self.pm = PositionManager(make_compound_config())
        self.pm.bankroll = 200.0
        self.pm.high_water_mark = 200.0

    def test_compound_enabled(self):
        self.assertTrue(self.pm.compound_enabled)
        self.assertTrue(self.pm.gear_enabled)
        self.assertTrue(self.pm.drawdown_enabled)
        self.assertTrue(self.pm.streak_enabled)
        self.assertTrue(self.pm.lockin_enabled)

    def test_gear_starts_normal(self):
        self.assertEqual(self.pm._current_gear, "normal")

    def test_gear_shifts_hot_on_high_win_rate(self):
        """After enough wins, gear should shift to 'hot'."""
        # 4 wins, 1 loss => 80% WR (above hot threshold of 60%)
        for _ in range(4):
            self.pm.update_after_win(5)
        self.pm._last_loss_time = 0  # clear cooldown
        self.pm.update_after_loss(5)
        # 5 trades, 80% WR
        self.assertEqual(self.pm._current_gear, "hot")

    def test_gear_shifts_cold_on_low_win_rate(self):
        """After many losses, gear should shift to 'cold'."""
        # 1 win, 4 losses => 20% WR
        self.pm.update_after_win(5)
        for _ in range(4):
            self.pm._last_loss_time = 0
            self.pm.update_after_loss(5)
        self.assertEqual(self.pm._current_gear, "cold")

    def test_hot_gear_reinvests_more(self):
        """Hot gear should use 70% reinvest vs normal 50%."""
        self.pm._current_gear = "hot"
        self.assertEqual(self.pm._get_gear_reinvest_pct(), 0.70)

    def test_cold_gear_reinvests_less(self):
        """Cold gear should use 25% reinvest."""
        self.pm._current_gear = "cold"
        self.assertEqual(self.pm._get_gear_reinvest_pct(), 0.25)

    def test_high_water_mark_updates_on_win(self):
        """HWM should go up when bankroll hits new peak."""
        initial_hwm = self.pm.high_water_mark
        self.pm.update_after_win(10)
        self.assertGreater(self.pm.high_water_mark, initial_hwm)
        self.assertEqual(self.pm.high_water_mark, self.pm.bankroll)

    def test_high_water_mark_stays_on_loss(self):
        """HWM should NOT go down on a loss."""
        self.pm.update_after_win(10)
        hwm_after_win = self.pm.high_water_mark
        self.pm._last_loss_time = 0
        self.pm.update_after_loss(5)
        self.assertEqual(self.pm.high_water_mark, hwm_after_win)

    def test_lock_in_raises_floor_on_win(self):
        """10% of each win should be locked into the floor."""
        old_floor = self.pm.compound_floor
        self.pm.update_after_win(10.0)
        # 10% of $10 = $1 locked
        expected_floor = old_floor + 1.0
        # Ratchet may also fire, so floor should be >= expected
        self.assertGreaterEqual(self.pm.compound_floor, expected_floor)

    def test_ratchet_raises_floor_on_headroom(self):
        """Floor should ratchet up when bankroll is $5+ above floor."""
        self.pm.compound_floor = 10.0
        self.pm.bankroll = 20.0  # $10 above floor, > ratchet_after=$5
        self.pm.update_after_win(5.0)
        # lock_in adds $0.50, then ratchet fires because headroom >= $5
        self.assertGreater(self.pm.compound_floor, 10.0)

    def test_drawdown_throttle_moderate(self):
        """Bet should be throttled at 15%+ drawdown from peak."""
        self.pm.bankroll = 200.0
        self.pm.high_water_mark = 240.0  # 16.7% drawdown
        normal_bet = self.pm.calculate_bet_size(edge=0.10, probability=0.60)
        # The bet should be cut by drawdown_multiplier (0.50)
        # Since it's a moderate drawdown, the bet should be < what it would be
        # without drawdown protection
        self.pm.drawdown_enabled = False
        unthrottled_bet = self.pm.calculate_bet_size(edge=0.10, probability=0.60)
        self.pm.drawdown_enabled = True
        self.assertLess(normal_bet, unthrottled_bet)

    def test_drawdown_throttle_severe(self):
        """Bet should be heavily throttled at 30%+ drawdown."""
        self.pm.bankroll = 140.0
        self.pm.high_water_mark = 200.0  # 30% drawdown
        severe_bet = self.pm.calculate_bet_size(edge=0.10, probability=0.60)
        # Should be cut to 25% of base
        self.pm.drawdown_enabled = False
        unthrottled_bet = self.pm.calculate_bet_size(edge=0.10, probability=0.60)
        self.pm.drawdown_enabled = True
        self.assertLess(severe_bet, unthrottled_bet)

    def test_streak_boost_activates(self):
        """After 3 consecutive wins, bet should get streak bonus."""
        self.pm.bankroll = 200.0
        self.pm.high_water_mark = 200.0
        self.pm.consecutive_wins = 2
        base_bet = self.pm.calculate_bet_size(edge=0.10, probability=0.60)

        self.pm.consecutive_wins = 4  # 2 levels above threshold of 3
        streak_bet = self.pm.calculate_bet_size(edge=0.10, probability=0.60)

        self.assertGreater(streak_bet, base_bet)

    def test_streak_boost_capped(self):
        """Streak boost should never exceed max_boost (2x)."""
        self.pm.bankroll = 500.0
        self.pm.high_water_mark = 500.0
        self.pm.consecutive_wins = 100  # huge streak
        bet = self.pm.calculate_bet_size(edge=0.10, probability=0.60)
        # Even with 100 wins, boost is capped at 2x, and overall bet capped at max_bet
        self.assertLessEqual(bet, self.pm.max_bet)

    def test_floor_never_decreases(self):
        """The compound floor should only go up, never down."""
        self.pm.update_after_win(10)
        floor_after_win = self.pm.compound_floor
        self.pm._last_loss_time = 0
        self.pm.update_after_loss(10)
        self.assertEqual(self.pm.compound_floor, floor_after_win)

    def test_compounding_summary(self):
        """get_compounding_summary should return all expected keys."""
        summary = self.pm.get_compounding_summary()
        expected_keys = {
            "gear", "win_rate", "high_water_mark", "compound_floor",
            "drawdown_pct", "reinvest_pct", "streak_bonus_active",
            "protected_profit",
        }
        self.assertEqual(set(summary.keys()), expected_keys)

    def test_compound_bet_never_below_initial(self):
        """With compounding enabled, bet should still be >= initial_bet."""
        self.pm.bankroll = 11.0  # just above floor
        self.pm.high_water_mark = 200.0  # big drawdown
        bet = self.pm.calculate_bet_size(edge=0.05, probability=0.55)
        self.assertGreaterEqual(bet, self.pm.initial_bet)


if __name__ == "__main__":
    unittest.main()
