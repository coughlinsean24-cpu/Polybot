"""
Core arbitrage decision engine.
Compares BTC price delta against Polymarket odds to identify +EV opportunities.
"""

import logging
from dataclasses import dataclass

from src.price_monitor import ArbitrageSignal
from src.polymarket_client import MarketInfo

logger = logging.getLogger(__name__)


@dataclass
class TradeDecision:
    should_trade: bool
    direction: str  # "YES" or "NO"
    edge: float  # expected value edge
    confidence: float
    price: float  # price to pay for the position
    reason: str


class ArbitrageEngine:
    def __init__(self, config: dict):
        self.min_edge = config["strategy"]["min_edge"]
        self.max_slippage = config["strategy"]["max_slippage"]
        self.min_price_delta = config["strategy"]["min_price_delta"]

    def analyze_opportunity(
        self,
        signal: ArbitrageSignal,
        market: MarketInfo,
        yes_price: float,
        no_price: float,
    ) -> TradeDecision:
        """
        Determine if an arbitrage opportunity exists.

        Logic:
        - BTC is trending UP strongly -> market should resolve YES for "BTC up" markets
        - If YES price is below our estimated probability, there's an edge
        - Same logic inverted for DOWN signals on "BTC down" markets
        """
        # Estimate the true probability based on BTC price momentum
        estimated_prob = self._estimate_probability(signal)

        # Determine which side to bet
        if signal.direction == "UP":
            # We think YES is more likely
            bet_side = "YES"
            market_price = yes_price
            our_estimate = estimated_prob
        else:
            # We think NO is more likely (BTC going down)
            bet_side = "NO"
            market_price = no_price
            our_estimate = estimated_prob

        # Calculate edge: our estimate minus what the market is charging
        edge = our_estimate - market_price

        # Check spread / slippage
        spread = abs(yes_price + no_price - 1.0)
        if spread > self.max_slippage:
            return TradeDecision(
                should_trade=False,
                direction=bet_side,
                edge=edge,
                confidence=signal.confidence,
                price=market_price,
                reason=f"Spread too wide: {spread:.4f} > {self.max_slippage}",
            )

        # Check minimum edge
        if edge < self.min_edge:
            return TradeDecision(
                should_trade=False,
                direction=bet_side,
                edge=edge,
                confidence=signal.confidence,
                price=market_price,
                reason=f"Edge too small: {edge:.4f} < {self.min_edge}",
            )

        return TradeDecision(
            should_trade=True,
            direction=bet_side,
            edge=edge,
            confidence=signal.confidence,
            price=market_price,
            reason=f"Edge={edge:.4f}, Delta={signal.delta_pct:.3f}%, Confidence={signal.confidence:.2f}",
        )

    def _estimate_probability(self, signal: ArbitrageSignal) -> float:
        """
        Estimate the true probability of the market outcome based on BTC price action.

        Uses a simple model:
        - Base rate: 50% (random walk)
        - Adjust up based on momentum magnitude
        - Adjust up based on confidence score
        - Cap at 85% (nothing is certain in 5-min markets)
        """
        base = 0.50
        abs_delta = abs(signal.delta_pct)

        # Momentum factor: bigger moves = higher probability of continuation
        # 0.15% move -> small boost, 0.5%+ -> large boost
        momentum_boost = min(abs_delta / 1.0, 0.25)  # max 25% boost from momentum

        # Confidence factor
        confidence_boost = signal.confidence * 0.10  # max 10% boost from confidence

        estimated = base + momentum_boost + confidence_boost

        # Clamp between 0.40 and 0.85
        estimated = max(0.40, min(0.85, estimated))

        logger.debug(
            f"Probability estimate: base={base} + momentum={momentum_boost:.3f} "
            f"+ confidence={confidence_boost:.3f} = {estimated:.3f}"
        )
        return estimated

    def calculate_expected_value(self, decision: TradeDecision, bet_size: float) -> float:
        """Calculate expected value of a trade."""
        if not decision.should_trade:
            return 0.0

        win_prob = decision.price + decision.edge
        loss_prob = 1 - win_prob

        # If you win, you get (1/price - 1) * bet_size profit
        # If you lose, you lose bet_size
        payout_ratio = (1.0 / decision.price) - 1
        ev = (win_prob * payout_ratio * bet_size) - (loss_prob * bet_size)
        return ev
