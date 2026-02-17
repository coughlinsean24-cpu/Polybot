"""
Bankroll and position sizing management.
Implements Kelly-style sizing with smart compounding and circuit breakers.

Smart Compounding Strategy ("Gear Shifting"):
  - Tiered reinvestment: reinvest more aggressively when win rate is strong
  - Drawdown throttle: auto-shrink bets when bankroll drops from peak
  - Win-streak gearing: compound faster during hot streaks (hard ceiling)
  - Profit lock-in: permanently lock away profits at regular intervals
  - High-water mark: the protected floor only goes up, never down
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
        self.min_bankroll = strategy.get("min_bankroll", 50.0)
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

        # Smart compounding: tiered gears based on performance
        gears_cfg = compound_cfg.get("gears", {})
        self.gear_enabled = gears_cfg.get("enabled", True)
        # Gear thresholds: win-rate -> reinvest multiplier
        # Below "cold" win rate = defensive (reinvest less)
        # Above "hot" win rate  = aggressive (reinvest more)
        self.gear_cold_wr = gears_cfg.get("cold_win_rate", 0.45)
        self.gear_hot_wr = gears_cfg.get("hot_win_rate", 0.60)
        self.gear_cold_reinvest = gears_cfg.get("cold_reinvest_pct", 0.25)  # defensive
        self.gear_normal_reinvest = compound_cfg.get("profit_reinvest_pct", 0.50)
        self.gear_hot_reinvest = gears_cfg.get("hot_reinvest_pct", 0.70)  # aggressive

        # Drawdown throttle: reduce bets when bankroll drops from peak
        drawdown_cfg = compound_cfg.get("drawdown", {})
        self.drawdown_enabled = drawdown_cfg.get("enabled", True)
        self.drawdown_threshold = drawdown_cfg.get("threshold", 0.15)  # 15% drop triggers throttle
        self.drawdown_multiplier = drawdown_cfg.get("multiplier", 0.50)  # cut bets in half
        self.drawdown_severe_threshold = drawdown_cfg.get("severe_threshold", 0.30)  # 30% = severe
        self.drawdown_severe_multiplier = drawdown_cfg.get("severe_multiplier", 0.25)  # quarter bets

        # Win-streak gearing: boost bet slightly during hot streaks
        streak_cfg = compound_cfg.get("streak", {})
        self.streak_enabled = streak_cfg.get("enabled", True)
        self.streak_threshold = streak_cfg.get("threshold", 3)  # consecutive wins to trigger
        self.streak_boost_pct = streak_cfg.get("boost_pct", 0.25)  # +25% per streak level
        self.streak_max_boost = streak_cfg.get("max_boost", 2.0)  # never more than 2x base bet

        # Lock-in: lock a % of every win into the floor immediately
        lockin_cfg = compound_cfg.get("lock_in", {})
        self.lockin_enabled = lockin_cfg.get("enabled", True)
        self.lockin_pct = lockin_cfg.get("pct", 0.10)  # lock 10% of each win into floor

        # State -- bankroll starts from the configured initial_bankroll.
        # It grows/shrinks purely from trade P&L (Kelly compound on trades).
        self._initial_bankroll = strategy.get("initial_bankroll", 200.0)
        self.bankroll: float = self._initial_bankroll
        self.starting_bankroll: float = self._initial_bankroll
        self.high_water_mark: float = self._initial_bankroll
        self.current_bet_size: float = self.initial_bet
        self.consecutive_wins: int = 0
        self.consecutive_losses: int = 0
        self.total_profit: float = 0.0
        self.total_trades: int = 0
        self.total_wins: int = 0
        self.session_profit: float = 0.0
        self._last_loss_time: float = 0.0
        self._current_gear: str = "normal"  # "cold", "normal", "hot"

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
          - "kelly": fractional Kelly criterion -- bet proportional to edge
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

        # -- Smart Compounding: scale bet with profits while protecting base --
        if self.compound_enabled:
            # 1. Pick the reinvest % based on current performance gear
            reinvest_pct = self._get_gear_reinvest_pct()

            # 2. "Investable" bankroll = only the portion above the floor
            investable = max(0.0, self.bankroll - self.compound_floor)
            reinvest_pool = investable * reinvest_pct
            compound_bankroll = self.compound_floor + reinvest_pool

            # 3. Cap bet at a percentage of the compound bankroll
            compound_max = compound_bankroll * self.compound_max_bet_pct
            compound_max = max(compound_max, self.initial_bet)
            bet = min(bet, compound_max)

            # 4. Win-streak gearing: boost bet during hot streaks
            if self.streak_enabled and self.consecutive_wins >= self.streak_threshold:
                streak_levels = self.consecutive_wins - self.streak_threshold + 1
                streak_mult = 1.0 + (self.streak_boost_pct * streak_levels)
                streak_mult = min(streak_mult, self.streak_max_boost)
                bet_before_streak = bet
                bet = bet * streak_mult
                logger.debug(
                    f"Streak boost: {self.consecutive_wins}W -> "
                    f"{streak_mult:.2f}x (${bet_before_streak:.2f} -> ${bet:.2f})"
                )

            # 5. Drawdown throttle: shrink bets when below high-water mark
            if self.drawdown_enabled and self.high_water_mark > 0:
                drawdown_pct = 1.0 - (self.bankroll / self.high_water_mark)
                if drawdown_pct >= self.drawdown_severe_threshold:
                    bet *= self.drawdown_severe_multiplier
                    logger.info(
                        f"[DRAWDOWN] Severe ({drawdown_pct:.0%} from peak) "
                        f"-> bet cut to ${bet:.2f}"
                    )
                elif drawdown_pct >= self.drawdown_threshold:
                    bet *= self.drawdown_multiplier
                    logger.info(
                        f"[DRAWDOWN] Moderate ({drawdown_pct:.0%} from peak) "
                        f"-> bet cut to ${bet:.2f}"
                    )

            logger.debug(
                f"Smart compound: bankroll=${self.bankroll:.2f} "
                f"floor=${self.compound_floor:.2f} gear={self._current_gear} "
                f"investable=${investable:.2f} reinvest={reinvest_pct:.0%} "
                f"hwm=${self.high_water_mark:.2f} -> bet=${bet:.2f}"
            )

            # 6. Never let compounding push below initial_bet
            bet = max(bet, self.initial_bet)

        # Don't bet more than we have -- but never go below initial_bet.
        # The bankroll is a *virtual* Kelly tracker; real solvency is
        # checked by _check_balance_before_trade against exchange balance.
        # Without this floor, a single early loss (bankroll $10 -> $5)
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
                trades -- completely independent of the account balance.
                A negative number means the bot has lost money.
        """
        # Circuit breaker: bot's own realized losses exceed max_loss
        if bot_pnl < -self.max_loss:
            return False, (f"Max loss reached: bot P&L ${bot_pnl:.2f} "
                           f"exceeds -${self.max_loss:.2f} limit")

        # Circuit breaker: bankroll below minimum
        if self.bankroll < self.min_bankroll:
            return False, (f"Bankroll ${self.bankroll:.2f} below minimum "
                           f"${self.min_bankroll:.2f}")

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

        # Update high-water mark (peak bankroll)
        if self.bankroll > self.high_water_mark:
            self.high_water_mark = self.bankroll

        # Update performance gear
        self._update_gear()

        logger.info(
            f"WIN +${profit:.2f} | Bankroll: ${self.bankroll:.2f} | "
            f"Streak: {self.consecutive_wins}W | Gear: {self._current_gear} | "
            f"HWM: ${self.high_water_mark:.2f} | Floor: ${self.compound_floor:.2f}"
        )

        if self.compound_enabled:
            # -- Lock-in: immediately lock a % of each win into the floor --
            if self.lockin_enabled and profit > 0:
                lock_amount = profit * self.lockin_pct
                old_floor = self.compound_floor
                self.compound_floor += lock_amount
                logger.info(
                    f"[LOCK] Win lock-in: ${lock_amount:.2f} of ${profit:.2f} win "
                    f"added to floor (${old_floor:.2f} -> ${self.compound_floor:.2f})"
                )

            # -- Ratchet: move floor up in bigger steps when bankroll grows --
            if self.compound_ratchet_step > 0:
                headroom = self.bankroll - self.compound_floor
                if headroom >= self.compound_ratchet_step:
                    old_floor = self.compound_floor
                    # Move floor up by one step (bank those profits permanently)
                    self.compound_floor += self.compound_ratchet_step
                    logger.info(
                        f"[LOCK] Ratchet raised floor: ${old_floor:.2f} -> "
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

        # Update performance gear (may downshift)
        self._update_gear()

        # Calculate drawdown from peak for logging
        drawdown_pct = 0.0
        if self.high_water_mark > 0:
            drawdown_pct = 1.0 - (self.bankroll / self.high_water_mark)

        logger.info(
            f"LOSS -${loss:.2f} | Bankroll: ${self.bankroll:.2f} | "
            f"Gear: {self._current_gear} | Drawdown: {drawdown_pct:.1%} | "
            f"Floor: ${self.compound_floor:.2f} | Streak: {self.consecutive_losses}L"
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
        cumulative = trade_logger.get_session_stats(all_sessions=True)
        if cumulative["total_trades"] == 0:
            logger.info("No prior trades in DB -- starting fresh")
            return False

        bot_pnl = trade_logger.get_bot_live_pnl()
        # Bot bankroll = configured initial_bankroll + cumulative P&L.
        # This is a self-contained pool that grows/shrinks from trades only.
        initial = self._initial_bankroll
        self.bankroll = max(initial, initial + bot_pnl)
        self.total_trades = cumulative.get("total_trades", 0)
        self.total_wins = cumulative.get("wins", 0)
        self.total_profit = cumulative.get("total_pnl", 0.0)

        # Set high-water mark to current bankroll (best we know on restart)
        self.high_water_mark = self.bankroll

        # Initialize the performance gear based on restored win rate
        self._update_gear()

        # Reset per-session counters (streaks don't carry across restarts)
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.session_profit = 0.0
        self.current_bet_size = self.initial_bet
        self._last_loss_time = 0.0

        logger.info(
            f"State restored from DB: bankroll=${self.bankroll:.2f}, "
            f"bot P&L=${bot_pnl:+.2f}, "
            f"gear={self._current_gear}, hwm=${self.high_water_mark:.2f}, "
            f"{self.total_trades} lifetime trades"
        )
        return True

    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.total_wins / self.total_trades

    # -- Smart Compounding helpers ---------------------------------

    def _get_gear_reinvest_pct(self) -> float:
        """Return the reinvestment percentage based on current gear.

        Gears shift automatically based on rolling win rate:
          - cold (<45% WR): defensive, reinvest only 25% of profits
          - normal (45-60% WR): standard 50% reinvestment
          - hot (>60% WR): aggressive 70% reinvestment

        This means when you're winning, profits compound faster.
        When you're losing, the bot automatically gets more protective.
        """
        if not self.gear_enabled:
            return self.compound_reinvest_pct

        if self._current_gear == "hot":
            return self.gear_hot_reinvest
        elif self._current_gear == "cold":
            return self.gear_cold_reinvest
        return self.gear_normal_reinvest

    def _update_gear(self):
        """Update the current performance gear based on win rate.

        Uses a minimum sample of 5 trades before shifting gears
        to avoid overreacting to early variance.
        """
        if not self.gear_enabled or self.total_trades < 5:
            self._current_gear = "normal"
            return

        wr = self.win_rate()
        old_gear = self._current_gear

        if wr >= self.gear_hot_wr:
            self._current_gear = "hot"
        elif wr <= self.gear_cold_wr:
            self._current_gear = "cold"
        else:
            self._current_gear = "normal"

        if self._current_gear != old_gear:
            logger.info(
                f"[GEAR] Shifted {old_gear} -> {self._current_gear} "
                f"(WR: {wr:.1%} over {self.total_trades} trades)"
            )

    def get_compounding_summary(self) -> dict:
        """Return a summary of the smart compounding state for dashboards."""
        drawdown_pct = 0.0
        if self.high_water_mark > 0:
            drawdown_pct = 1.0 - (self.bankroll / self.high_water_mark)

        return {
            "gear": self._current_gear,
            "win_rate": self.win_rate(),
            "high_water_mark": round(self.high_water_mark, 2),
            "compound_floor": round(self.compound_floor, 2),
            "drawdown_pct": round(drawdown_pct, 4),
            "reinvest_pct": self._get_gear_reinvest_pct(),
            "streak_bonus_active": (
                self.streak_enabled
                and self.consecutive_wins >= self.streak_threshold
            ),
            "protected_profit": round(
                self.compound_floor - self.initial_bet, 2
            ),
        }
