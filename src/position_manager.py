"""
Bankroll and position sizing management.
Implements Kelly-style doubling progression with circuit breakers.
"""

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PositionState:
    bankroll: float
    bet_size: float
    consecutive_wins: int
    consecutive_losses: int
    total_profit: float
    total_trades: int
    total_wins: int
    session_profit: float  # profit since last reset


class PositionManager:
    def __init__(self, config: dict):
        strategy = config["strategy"]
        risk = config["risk"]

        self.initial_bet = strategy["initial_bet"]
        self.max_bet = strategy["max_bet"]
        self.reset_profit_target = strategy["reset_profit_target"]
        self.min_bankroll = strategy["min_bankroll"]
        self.max_consecutive_losses = risk["max_consecutive_losses"]
        self.cooldown_after_loss = risk["cooldown_after_loss"]

        # State
        self.bankroll: float = 200.0
        self.current_bet_size: float = self.initial_bet
        self.consecutive_wins: int = 0
        self.consecutive_losses: int = 0
        self.total_profit: float = 0.0
        self.total_trades: int = 0
        self.total_wins: int = 0
        self.session_profit: float = 0.0
        self._last_loss_time: float = 0.0
        self._paused: bool = False

    def get_state(self) -> PositionState:
        return PositionState(
            bankroll=self.bankroll,
            bet_size=self.current_bet_size,
            consecutive_wins=self.consecutive_wins,
            consecutive_losses=self.consecutive_losses,
            total_profit=self.total_profit,
            total_trades=self.total_trades,
            total_wins=self.total_wins,
            session_profit=self.session_profit,
        )

    def calculate_bet_size(self) -> float:
        """Calculate next bet size based on progression strategy."""
        # Don't bet more than we have
        bet = min(self.current_bet_size, self.bankroll)
        # Don't exceed max bet
        bet = min(bet, self.max_bet)
        return bet

    def can_trade(self) -> tuple[bool, str]:
        """Check if we're allowed to trade right now."""
        # Circuit breaker: minimum bankroll
        if self.bankroll < self.min_bankroll:
            return False, f"Bankroll ${self.bankroll:.2f} below minimum ${self.min_bankroll}"

        # Circuit breaker: consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False, f"Hit {self.consecutive_losses} consecutive losses, pausing"

        # Cooldown after loss
        if self._last_loss_time > 0:
            elapsed = time.time() - self._last_loss_time
            if elapsed < self.cooldown_after_loss:
                remaining = self.cooldown_after_loss - elapsed
                return False, f"Loss cooldown: {remaining:.0f}s remaining"

        # Can't bet more than bankroll
        if self.current_bet_size > self.bankroll:
            return False, f"Bet size ${self.current_bet_size} exceeds bankroll ${self.bankroll:.2f}"

        return True, "OK"

    def update_after_win(self, profit: float):
        """Update state after a winning trade."""
        self.bankroll += profit
        self.total_profit += profit
        self.session_profit += profit
        self.total_trades += 1
        self.total_wins += 1
        self.consecutive_wins += 1
        self.consecutive_losses = 0

        # Doubling progression: double bet on win
        self.current_bet_size = min(self.current_bet_size * 2, self.max_bet)

        logger.info(
            f"WIN +${profit:.2f} | Bankroll: ${self.bankroll:.2f} | "
            f"Next bet: ${self.current_bet_size:.2f} | Streak: {self.consecutive_wins}W"
        )

        # Check if we hit the reset target
        self._check_reset_condition()

    def update_after_loss(self, loss: float):
        """Update state after a losing trade."""
        self.bankroll -= loss
        self.total_profit -= loss
        self.session_profit -= loss
        self.total_trades += 1
        self.consecutive_losses += 1
        self.consecutive_wins = 0
        self._last_loss_time = time.time()

        # Reset bet size to initial on loss
        self.current_bet_size = self.initial_bet

        logger.info(
            f"LOSS -${loss:.2f} | Bankroll: ${self.bankroll:.2f} | "
            f"Reset bet to: ${self.current_bet_size:.2f} | Streak: {self.consecutive_losses}L"
        )

    def _check_reset_condition(self):
        """Reset session if profit target reached."""
        if self.session_profit >= self.reset_profit_target:
            logger.info(
                f"SESSION RESET: Profit target ${self.reset_profit_target} reached! "
                f"Session profit: ${self.session_profit:.2f}"
            )
            self.session_profit = 0.0
            self.current_bet_size = self.initial_bet
            self.consecutive_wins = 0

    def reset_consecutive_loss_counter(self):
        """Manually reset the consecutive loss counter (after pause/review)."""
        self.consecutive_losses = 0
        self._paused = False
        logger.info("Consecutive loss counter reset, trading resumed")

    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.total_wins / self.total_trades
