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
        self.max_loss = strategy.get("max_loss", 50.0)
        self.conviction_max_bet = strategy.get("conviction_max_bet", self.max_bet)
        self.conviction_edge = strategy.get("conviction_edge", 0.30)
        self.max_consecutive_losses = risk["max_consecutive_losses"]
        self.cooldown_after_loss = risk["cooldown_after_loss"]
        self.kelly_fraction = strategy.get("kelly_fraction", 0.25)
        self.bet_mode = strategy.get("bet_mode", "kelly")  # "kelly", "flat", "doubling"

        # Compounding config
        compound_cfg = strategy.get("compound", {})
        self.compound_enabled = compound_cfg.get("enabled", False)
        self.compound_reinvest_pct = compound_cfg.get("profit_reinvest_pct", 0.50)
        self.compound_max_bet_pct = compound_cfg.get("max_bet_pct", 0.10)
        self.compound_floor = compound_cfg.get("floor", self.initial_bet)
        self.compound_ratchet_step = compound_cfg.get("ratchet_after", 5.0)

        # State
        self.bankroll: float = 200.0
        self.starting_bankroll: float = 200.0  # set on startup from exchange balance
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

    def calculate_bet_size(self, edge: float = 0.0, probability: float = 0.0) -> float:
        """Calculate next bet size based on the configured strategy.

        Modes:
          - "kelly": fractional Kelly criterion — bet proportional to edge
            and bankroll.  Safer than full Kelly, survives variance.
          - "flat": always bet initial_bet.  Simplest.
          - "doubling": Martingale-style doubling on wins (legacy).

        The Kelly formula for a binary market paying 1:b odds is:
            f* = (bp - q) / b
        where p = win probability, q = 1 - p, b = net payout ratio.
        We bet kelly_fraction * f* * bankroll (quarter-Kelly by default).
        """
        if self.bet_mode == "flat":
            bet = self.initial_bet

        elif self.bet_mode == "kelly" and edge > 0 and probability > 0:
            # For Polymarket: buying at price p, paying out $1 on win.
            # Net odds b = (1/p) - 1.  Win prob = probability.
            price = probability - edge  # market price (what we pay)
            if price <= 0 or price >= 1:
                bet = self.initial_bet
            else:
                b = (1.0 / price) - 1.0  # net payout ratio
                p = probability
                q = 1.0 - p
                kelly_f = (b * p - q) / b
                kelly_f = max(0.0, kelly_f)
                bet = self.kelly_fraction * kelly_f * self.bankroll
                bet = max(self.initial_bet, bet)  # at least initial_bet

        elif self.bet_mode == "doubling":
            bet = self.current_bet_size

        else:
            bet = self.initial_bet

        # ── Compounding: scale bet with profits while protecting base ──
        if self.compound_enabled:
            # "Investable" bankroll = only the portion above the floor
            # that we're willing to reinvest.  The floor is sacred.
            investable = max(0.0, self.bankroll - self.compound_floor)
            reinvest_pool = investable * self.compound_reinvest_pct
            compound_bankroll = self.compound_floor + reinvest_pool

            # Cap bet at a percentage of the compound bankroll
            compound_max = compound_bankroll * self.compound_max_bet_pct
            # The compound max should be at least initial_bet
            compound_max = max(compound_max, self.initial_bet)
            bet = min(bet, compound_max)

            logger.debug(
                f"Compound sizing: bankroll=${self.bankroll:.2f} "
                f"floor=${self.compound_floor:.2f} "
                f"investable=${investable:.2f} → bet=${bet:.2f}"
            )

        # Don't bet more than we have — but never go below initial_bet.
        # The bankroll is a *virtual* Kelly tracker; real solvency is
        # checked by _check_balance_before_trade against exchange balance.
        # Without this floor, a single early loss (bankroll $10 → $5)
        # permanently halves all subsequent bets.
        bet = min(bet, max(self.bankroll, self.initial_bet))
        # Don't exceed max bet (or conviction_max_bet for high-edge trades)
        if edge >= self.conviction_edge and self.conviction_max_bet > self.max_bet:
            bet = min(bet, self.conviction_max_bet)
        else:
            bet = min(bet, self.max_bet)
        # Round to cents
        bet = round(bet, 2)
        return bet

    def can_trade(self, bot_pnl: float = 0.0) -> tuple[bool, str]:
        """Check if we're allowed to trade right now.

        Args:
            bot_pnl: The bot's own net P&L from its trade records.
                This is the sum of profit_loss on all resolved live
                trades — completely independent of the account balance.
                A negative number means the bot has lost money.
        """
        # Circuit breaker: bot's own realized losses exceed max_loss
        if bot_pnl < -self.max_loss:
            return False, (f"Max loss reached: bot P&L ${bot_pnl:.2f} "
                           f"exceeds -${self.max_loss:.2f} limit")

        # Circuit breaker: consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False, f"Hit {self.consecutive_losses} consecutive losses, pausing"

        # Cooldown after loss
        if self._last_loss_time > 0:
            elapsed = time.time() - self._last_loss_time
            if elapsed < self.cooldown_after_loss:
                remaining = self.cooldown_after_loss - elapsed
                return False, f"Loss cooldown: {remaining:.0f}s remaining"

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

        # Only double in "doubling" mode (legacy).
        # For kelly/flat, the bet size is computed fresh each trade.
        if self.bet_mode == "doubling":
            self.current_bet_size = min(self.current_bet_size * 2, self.max_bet)

        logger.info(
            f"WIN +${profit:.2f} | Bankroll: ${self.bankroll:.2f} | "
            f"Streak: {self.consecutive_wins}W | Mode: {self.bet_mode}"
        )

        # ── Compound: ratchet up the floor as profits grow ──
        if self.compound_enabled and self.compound_ratchet_step > 0:
            headroom = self.bankroll - self.compound_floor
            if headroom >= self.compound_ratchet_step:
                old_floor = self.compound_floor
                # Move floor up by one step (bank those profits permanently)
                self.compound_floor += self.compound_ratchet_step
                logger.info(
                    f"🔒 Compound floor raised: ${old_floor:.2f} → "
                    f"${self.compound_floor:.2f} (protected profits)"
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

    def load_state_from_db(self, trade_logger) -> bool:
        """Restore bankroll and cumulative stats from the trade database.

        Called on startup so restarts don't lose progress.
        The bankroll is computed as initial_bet (the bot's starting
        capital concept) + the bot's total realised P&L.  We do NOT
        read bankroll_after from the DB because that field was
        historically set to the exchange balance which includes
        the user's manual trades.

        Returns True if state was loaded, False if starting fresh.
        """
        cumulative = trade_logger.get_cumulative_stats()
        if cumulative["total_trades"] == 0:
            logger.info("No prior trades in DB — starting fresh")
            return False

        bot_pnl = trade_logger.get_bot_live_pnl()
        # Bot bankroll = max_bet as starting capital + cumulative P&L
        # This is a virtual bankroll used only for Kelly sizing.
        self.bankroll = max(self.max_bet, self.max_bet + bot_pnl)
        self.total_trades = cumulative.get("total_trades", 0)
        self.total_wins = cumulative.get("total_wins", 0)
        self.total_profit = cumulative.get("total_pnl", 0.0)

        # Reset per-session counters (streaks don't carry across restarts)
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.session_profit = 0.0
        self.current_bet_size = self.initial_bet
        self._last_loss_time = 0.0

        logger.info(
            f"State restored from DB: bankroll=${self.bankroll:.2f}, "
            f"bot P&L=${bot_pnl:+.2f}, "
            f"{self.total_trades} lifetime trades"
        )
        return True

    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.total_wins / self.total_trades
