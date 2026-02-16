"""
Core trading decision engine for BTC Up/Down 5-minute markets.

Simple strategy:
  1. Get the "price to beat" (BTC price at the start of the 5-min window)
  2. Compare current real-time BTC price against that target
  3. If BTC is ABOVE the target → bet UP, if BELOW → bet DOWN
  4. Only trade if the Polymarket odds offer value vs. our confidence
"""

import logging
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
    ) -> TradeDecision:
        """
        Decide whether to trade based on where BTC is relative to the
        market's "price to beat".

        Args:
            btc_price:    Current real-time BTC price (Binance)
            target_price: BTC price at start of the 5-min window
            market:       Polymarket MarketInfo with Up/Down odds
            seconds_left: Seconds remaining in this 5-min window
        """
        diff = btc_price - target_price
        diff_pct = (diff / target_price) * 100  # e.g. +0.05%

        # Which side of the target are we on?
        if diff >= 0:
            direction = "UP"
            market_price = market.up_price  # cost to buy "Up"
        else:
            direction = "DOWN"
            market_price = market.down_price  # cost to buy "Down"

        # Estimate our probability that this side wins
        estimated_prob = self._estimate_probability(diff_pct, seconds_left)

        # Edge = our estimate - what the market charges
        edge = estimated_prob - market_price

        # Confidence = how strongly the price signal is
        confidence = min(abs(diff_pct) / 0.10, 1.0)  # 0.10% diff = max confidence

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

        # Don't trade in the last 30 seconds — price can flip and
        # our order may not fill in time on the live CLOB
        if seconds_left < 30:
            return TradeDecision(
                should_trade=False,
                direction=direction,
                edge=edge,
                confidence=confidence,
                price=market_price,
                reason=f"Too close to expiry: {seconds_left:.0f}s left",
            )

        # Don't trade if BTC is barely above/below target (noise)
        # Data shows "tight" conditions (<0.02% diff) lose money —
        # the signal is too weak and gets washed by noise/fees.
        if abs(diff_pct) < 0.01:
            return TradeDecision(
                should_trade=False,
                direction=direction,
                edge=edge,
                confidence=confidence,
                price=market_price,
                reason=f"BTC diff too small ({diff_pct:+.4f}%), likely noise",
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
                f"{seconds_left:.0f}s left"
            ),
        )

    def _estimate_probability(self, diff_pct: float, seconds_left: float) -> float:
        """
        Estimate the true probability that the current side wins.

        Model (calibrated to BTC 5-min vol ≈ 0.05-0.15%):
          - Base rate is 50% (coin flip if exactly at target).
          - Distance factor: how far BTC is from the target, scaled by
            typical 5-min BTC volatility (~0.10%).  A move of 0.10% is
            roughly 1-sigma, giving ~68% probability.
          - Time factor: less time remaining means less room for reversal.
            This multiplies the distance factor — a 0.05% lead at 30s left
            is much stronger than at 280s left.
          - The two factors combine multiplicatively (time amplifies distance).
        """
        import math

        abs_diff = abs(diff_pct)

        # Typical 5-min BTC volatility in percent.  This is the key
        # calibration knob — if BTC can move ±0.10% in 5 min on average,
        # then a 0.10% lead ≈ 1σ ≈ 68%.
        vol_5min = 0.10  # percent

        # Scale remaining vol by sqrt(time_left / 300)
        # At 300s left, full vol remains.  At 30s left, only ~32% of vol.
        time_frac_remaining = max(seconds_left, 1.0) / 300.0
        remaining_vol = vol_5min * math.sqrt(time_frac_remaining)

        # z-score: how many remaining-vols the current lead represents
        if remaining_vol > 0:
            z = abs_diff / remaining_vol
        else:
            z = 10.0  # effectively certain

        # Convert z-score to probability using the normal CDF approximation.
        # P(staying ahead) ≈ Φ(z).  Use a fast rational approximation.
        # For z < 0 this shouldn't happen (abs_diff >= 0).
        estimated = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

        # Clamp to [0.45, 0.95]
        estimated = max(0.45, min(0.95, estimated))

        logger.debug(
            f"Prob estimate: z={z:.2f} → {estimated:.3f} "
            f"(diff={diff_pct:+.4f}%, vol_rem={remaining_vol:.4f}%, "
            f"{seconds_left:.0f}s left)"
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
