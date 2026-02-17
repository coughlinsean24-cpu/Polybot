"""
Core trading decision engine for BTC Up/Down 5-minute markets.

Strategy:
  1. Get the "price to beat" (BTC price at the start of the 5-min window)
  2. Compare current real-time BTC price against that target
  3. If BTC is ABOVE the target -> bet UP, if BELOW -> bet DOWN
  4. Only trade if the Polymarket odds offer value vs. our confidence

Enhanced edge model (v2):
  - Uses ACTUAL measured BTC volatility instead of a static 0.10% guess
  - Factors in price velocity (momentum) and acceleration
  - Multi-feed consensus: when all exchanges agree, confidence increases
  - Big-move detection: large, accelerating moves get an edge boost
"""

import logging
import math
from dataclasses import dataclass

from src.polymarket_client import MarketInfo

logger = logging.getLogger(__name__)


@dataclass
class TradeDecision:
    should_trade: bool
    direction: str  # "UP" or "DOWN"
    edge: float  # our estimated prob minus market price
    confidence: float  # how far BTC is from the target (0-1)
    price: float  # price we'd pay for the position (market odds)
    reason: str


class ArbitrageEngine:
    def __init__(self, config: dict):
        self.max_slippage = config["strategy"]["max_slippage"]
        self.min_edge = config["strategy"].get("min_edge", 0.01)

    def analyze_opportunity(
        self,
        btc_price: float,
        target_price: float,
        market: MarketInfo,
        seconds_left: float,
        measured_vol: float = 0.10,
        velocity: float = 0.0,
        acceleration: float = 0.0,
        feed_agreement: float = 0.5,
        active_feeds: int = 1,
    ) -> TradeDecision:
        """
        Decide whether to trade based on where BTC is relative to the
        market's "price to beat".

        Args:
            btc_price:      Current real-time BTC price
            target_price:   BTC price at start of the 5-min window
            market:         Polymarket MarketInfo with Up/Down odds
            seconds_left:   Seconds remaining in this 5-min window
            measured_vol:   Actual measured 5-min volatility (%)
            velocity:       Price velocity (%/sec, positive = rising)
            acceleration:   Is the move speeding up? (positive = accelerating)
            feed_agreement: 0-1, fraction of feeds agreeing on direction
            active_feeds:   Number of active exchange feeds
        """
        diff = btc_price - target_price
        diff_pct = (diff / target_price) * 100  # e.g. +0.05%

        # Which side of the target are we on?
        if diff >= 0:
            direction = "UP"
            market_price = market.up_price
        else:
            direction = "DOWN"
            market_price = market.down_price

        # Estimate our probability using the enhanced model
        estimated_prob = self._estimate_probability(
            diff_pct, seconds_left, measured_vol, velocity, acceleration,
            feed_agreement, active_feeds,
        )

        # Edge = our estimate - what the market charges
        edge = estimated_prob - market_price

        # Confidence = how strongly the price signal is
        confidence = min(abs(diff_pct) / 0.10, 1.0)

        # Check spread
        spread = abs(market.up_price + market.down_price - 1.0)
        if spread > self.max_slippage:
            return TradeDecision(
                should_trade=False,
                direction=direction,
                edge=edge,
                confidence=confidence,
                price=market_price,
                reason=f"Spread too wide: {spread:.4f} > {self.max_slippage}",
            )

        # Check minimum edge
        if edge < self.min_edge:
            return TradeDecision(
                should_trade=False,
                direction=direction,
                edge=edge,
                confidence=confidence,
                price=market_price,
                reason=(
                    f"Edge too small: {edge:.4f} < {self.min_edge} | "
                    f"BTC diff={diff_pct:+.4f}% | "
                    f"Our prob={estimated_prob:.3f} vs market={market_price:.3f}"
                ),
            )

        # Don't trade in the last 30 seconds
        if seconds_left < 30:
            return TradeDecision(
                should_trade=False,
                direction=direction,
                edge=edge,
                confidence=confidence,
                price=market_price,
                reason=f"Too close to expiry: {seconds_left:.0f}s left",
            )

        # Don't trade if BTC hasn't moved enough from target.
        # PURE LATENCY ARB: lower threshold to catch more delay-based moves.
        min_diff_pct = 0.05
        if abs(diff_pct) < min_diff_pct:
            return TradeDecision(
                should_trade=False,
                direction=direction,
                edge=edge,
                confidence=confidence,
                price=market_price,
                reason=f"BTC diff too small ({diff_pct:+.4f}% < {min_diff_pct}%), waiting for bigger move",
            )

        return TradeDecision(
            should_trade=True,
            direction=direction,
            edge=edge,
            confidence=confidence,
            price=market_price,
            reason=(
                f"BTC {'above' if direction == 'UP' else 'below'} target by "
                f"{abs(diff_pct):.4f}% | Edge={edge:.4f} | "
                f"Our prob={estimated_prob:.3f} vs market={market_price:.3f} | "
                f"{seconds_left:.0f}s left | "
                f"vol={measured_vol:.3f}% vel={velocity:+.5f}%/s "
                f"feeds={active_feeds} agree={feed_agreement:.0%}"
            ),
        )

    def _estimate_probability(
        self,
        diff_pct: float,
        seconds_left: float,
        measured_vol: float = 0.10,
        velocity: float = 0.0,
        acceleration: float = 0.0,
        feed_agreement: float = 0.5,
        active_feeds: int = 1,
    ) -> float:
        """
        Enhanced probability model (v2).

        Base model:
          Same normal-CDF z-score approach, but now using ACTUAL measured
          volatility instead of a static 0.10% guess.

        Momentum adjustment:
          If BTC is moving WITH our direction (velocity confirms diff),
          the probability of reversal is lower. This is a small additive
          boost to estimated_prob.

        Multi-feed consensus:
          When 3+ feeds all agree on the same direction, the price signal
          is more trustworthy (not a glitch on one exchange).

        Big-move acceleration:
          If the move is accelerating (getting bigger, not mean-reverting),
          it's more likely to hold through expiry.
        """
        abs_diff = abs(diff_pct)

        # Use actual measured vol instead of hardcoded 0.10%
        vol_5min = max(measured_vol, 0.03)

        # Scale remaining vol by sqrt(time_left / 300)
        time_frac_remaining = max(seconds_left, 1.0) / 300.0
        remaining_vol = vol_5min * math.sqrt(time_frac_remaining)

        # z-score
        if remaining_vol > 0:
            z = abs_diff / remaining_vol
        else:
            z = 10.0

        # Base probability via normal CDF
        base_prob = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

        # ── Momentum adjustment ──────────────────────────────────
        # velocity is %/sec, positive = price rising
        # If direction matches velocity, add a boost
        momentum_boost = 0.0
        if diff_pct != 0 and velocity != 0:
            # Check if velocity is in the same direction as our bet
            velocity_confirms = (diff_pct > 0 and velocity > 0) or \
                                (diff_pct < 0 and velocity < 0)
            if velocity_confirms:
                # Scale boost by velocity magnitude (typical: 0.001-0.01 %/sec)
                vel_magnitude = abs(velocity)
                # Cap at +5% probability boost for very strong momentum
                momentum_boost = min(vel_magnitude * 500, 0.05)

                # Extra boost if accelerating
                if acceleration > 0:
                    momentum_boost *= 1.5  # up to +7.5%
            else:
                # Velocity is AGAINST our direction — slight penalty
                vel_magnitude = abs(velocity)
                momentum_boost = -min(vel_magnitude * 300, 0.03)

        # ── Multi-feed consensus adjustment ──────────────────────
        # When multiple independent exchanges all agree, it's more reliable
        consensus_boost = 0.0
        if active_feeds >= 3 and feed_agreement >= 0.9:
            # 3+ feeds, 90%+ agreement = strong consensus
            consensus_boost = 0.02
        elif active_feeds >= 2 and feed_agreement >= 1.0:
            # 2 feeds, perfect agreement
            consensus_boost = 0.01

        # ── Big-move bonus ───────────────────────────────────────
        # Large diff + time remaining = this is a real move, not noise
        big_move_boost = 0.0
        if abs_diff > 0.15 and seconds_left > 60:
            # BTC moved >0.15% with >1 min left — unlikely to fully reverse
            big_move_boost = 0.02
        if abs_diff > 0.25 and seconds_left > 60:
            big_move_boost = 0.04  # very large move

        # ── Combine ──────────────────────────────────────────────
        estimated = base_prob + momentum_boost + consensus_boost + big_move_boost

        # Clamp to [0.45, 0.96]
        estimated = max(0.45, min(0.96, estimated))

        logger.debug(
            f"Prob estimate v2: z={z:.2f} base={base_prob:.3f} "
            f"mom={momentum_boost:+.3f} cons={consensus_boost:+.3f} "
            f"big={big_move_boost:+.3f} -> {estimated:.3f} "
            f"(diff={diff_pct:+.4f}%, vol={vol_5min:.4f}%, "
            f"vel={velocity:+.6f}%/s, {seconds_left:.0f}s left, "
            f"feeds={active_feeds} agree={feed_agreement:.0%})"
        )
        return estimated

    def calculate_expected_value(self, decision: TradeDecision, bet_size: float) -> float:
        """Calculate expected value of a trade."""
        if not decision.should_trade:
            return 0.0

        win_prob = decision.price + decision.edge
        loss_prob = 1 - win_prob

        payout_ratio = (1.0 / decision.price) - 1
        ev = (win_prob * payout_ratio * bet_size) - (loss_prob * bet_size)
        return ev
