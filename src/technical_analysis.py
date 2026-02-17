"""
Technical analysis module for Polybot.

Computes real-time technical indicators from the BTC price stream:
  - RSI (Relative Strength Index) -- momentum oscillator
  - Bollinger Band position -- volatility + mean reversion
  - Rate of Change (ROC) -- short-term velocity
  - VWAP-like moving average deviation
  - Trend strength (slope of price over rolling window)

All indicators are designed for 5-minute market windows:
  - Computed from the rolling ~20 min of 1-second tick data
  - No external libraries needed -- pure-python from price_history deque

The ArbitrageEngine queries these for additional signal before trading.
"""

import logging
import math
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────
RSI_PERIOD = 14         # 14 ticks (seconds at 1 sample/sec)
BB_PERIOD = 20          # Bollinger Band lookback
BB_STD_MULT = 2.0       # standard deviations for bands
ROC_PERIOD = 30         # rate of change lookback (30s)
TREND_PERIOD = 60       # trend slope lookback (60s)
EMA_FAST = 10           # fast EMA (seconds)
EMA_SLOW = 30           # slow EMA (seconds)


@dataclass
class TASignal:
    """Complete technical signal snapshot."""
    rsi: float              # 0-100; <30 oversold, >70 overbought
    bb_position: float      # -1 to +1; -1 = at lower band, +1 = at upper band
    bb_width: float         # current band width (volatility proxy)
    roc: float              # % change over ROC_PERIOD seconds
    trend_slope: float      # linear regression slope ($/sec) over TREND_PERIOD
    trend_r2: float         # R² of the trend line (0-1; >0.7 = strong trend)
    ema_cross: float        # fast EMA - slow EMA (positive = bullish)
    momentum_score: float   # composite -1 to +1 overall momentum reading
    volatility_regime: str  # "low", "normal", "high"

    @property
    def is_overbought(self) -> bool:
        return self.rsi > 70

    @property
    def is_oversold(self) -> bool:
        return self.rsi < 30

    @property
    def is_strong_trend(self) -> bool:
        return self.trend_r2 > 0.65

    @property
    def trend_direction(self) -> str:
        """UP, DOWN, or FLAT based on trend slope."""
        if self.trend_slope > 0.05:
            return "UP"
        elif self.trend_slope < -0.05:
            return "DOWN"
        return "FLAT"


class TechnicalAnalyzer:
    """Compute technical indicators from a price history deque."""

    def __init__(self, config: dict = None):
        ta_cfg = (config or {}).get("technical_analysis", {})
        self.rsi_period = ta_cfg.get("rsi_period", RSI_PERIOD)
        self.bb_period = ta_cfg.get("bb_period", BB_PERIOD)
        self.bb_std_mult = ta_cfg.get("bb_std_mult", BB_STD_MULT)
        self.roc_period = ta_cfg.get("roc_period", ROC_PERIOD)
        self.trend_period = ta_cfg.get("trend_period", TREND_PERIOD)
        self.ema_fast = ta_cfg.get("ema_fast", EMA_FAST)
        self.ema_slow = ta_cfg.get("ema_slow", EMA_SLOW)

    def compute(self, price_history: deque) -> TASignal | None:
        """Compute all technical indicators from the price history.

        Returns None if insufficient data (need at least 60 data points).
        """
        if len(price_history) < max(self.trend_period, self.bb_period, self.rsi_period + 1):
            return None

        prices = [pt.price for pt in price_history]

        rsi = self._compute_rsi(prices)
        bb_pos, bb_width = self._compute_bollinger(prices)
        roc = self._compute_roc(prices)
        slope, r2 = self._compute_trend(prices)
        ema_cross = self._compute_ema_cross(prices)
        momentum = self._compute_momentum_score(rsi, bb_pos, roc, slope, r2, ema_cross)
        vol_regime = self._classify_volatility(bb_width, prices)

        return TASignal(
            rsi=rsi,
            bb_position=bb_pos,
            bb_width=bb_width,
            roc=roc,
            trend_slope=slope,
            trend_r2=r2,
            ema_cross=ema_cross,
            momentum_score=momentum,
            volatility_regime=vol_regime,
        )

    # ── RSI ─────────────────────────────────────────────────────

    def _compute_rsi(self, prices: list[float]) -> float:
        """Relative Strength Index using Wilder's smoothed RS."""
        period = self.rsi_period
        if len(prices) < period + 1:
            return 50.0  # neutral default

        # Calculate price changes
        changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

        # Use the last `period * 3` changes for a more stable reading
        changes = changes[-(period * 3):]

        # Seed with simple average of first `period` changes
        gains = [max(c, 0) for c in changes[:period]]
        losses = [abs(min(c, 0)) for c in changes[:period]]
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        # Wilder smoothed average for remaining
        for c in changes[period:]:
            gain = max(c, 0)
            loss = abs(min(c, 0))
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    # ── Bollinger Bands ─────────────────────────────────────────

    def _compute_bollinger(self, prices: list[float]) -> tuple[float, float]:
        """Return (position within bands [-1,+1], band width)."""
        period = self.bb_period
        if len(prices) < period:
            return 0.0, 0.0

        window = prices[-period:]
        sma = sum(window) / period
        variance = sum((p - sma) ** 2 for p in window) / period
        std = math.sqrt(variance) if variance > 0 else 0.001

        upper = sma + self.bb_std_mult * std
        lower = sma - self.bb_std_mult * std
        band_width = upper - lower

        current = prices[-1]
        if band_width > 0:
            position = 2.0 * (current - lower) / band_width - 1.0
            position = max(-1.0, min(1.0, position))
        else:
            position = 0.0

        # Normalize width as percentage of price
        width_pct = (band_width / sma * 100) if sma > 0 else 0.0

        return position, width_pct

    # ── Rate of Change ──────────────────────────────────────────

    def _compute_roc(self, prices: list[float]) -> float:
        """Percentage price change over the ROC period."""
        period = min(self.roc_period, len(prices) - 1)
        if period < 1:
            return 0.0
        old_price = prices[-(period + 1)]
        if old_price == 0:
            return 0.0
        return ((prices[-1] - old_price) / old_price) * 100

    # ── Trend (linear regression) ───────────────────────────────

    def _compute_trend(self, prices: list[float]) -> tuple[float, float]:
        """Linear regression slope and R² over the trend period.

        Returns (slope in $/second, R²).
        """
        period = min(self.trend_period, len(prices))
        if period < 5:
            return 0.0, 0.0

        y = prices[-period:]
        n = len(y)
        x = list(range(n))

        x_mean = (n - 1) / 2.0
        y_mean = sum(y) / n

        ss_xy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        ss_xx = sum((xi - x_mean) ** 2 for xi in x)
        ss_yy = sum((yi - y_mean) ** 2 for yi in y)

        if ss_xx == 0:
            return 0.0, 0.0

        slope = ss_xy / ss_xx

        if ss_yy == 0:
            r2 = 1.0  # no variance = perfect fit
        else:
            r2 = (ss_xy ** 2) / (ss_xx * ss_yy)
            r2 = max(0.0, min(1.0, r2))

        return slope, r2

    # ── EMA Cross ───────────────────────────────────────────────

    def _compute_ema_cross(self, prices: list[float]) -> float:
        """Fast EMA minus slow EMA. Positive = bullish crossover."""
        fast_ema = self._ema(prices, self.ema_fast)
        slow_ema = self._ema(prices, self.ema_slow)
        if fast_ema is None or slow_ema is None:
            return 0.0
        return fast_ema - slow_ema

    @staticmethod
    def _ema(prices: list[float], period: int) -> float | None:
        """Exponential Moving Average."""
        if len(prices) < period:
            return None
        multiplier = 2.0 / (period + 1)
        ema = sum(prices[:period]) / period
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        return ema

    # ── Composite Momentum Score ────────────────────────────────

    def _compute_momentum_score(
        self,
        rsi: float,
        bb_pos: float,
        roc: float,
        slope: float,
        r2: float,
        ema_cross: float,
    ) -> float:
        """Combine all indicators into a single -1 to +1 score.

        Positive = bullish momentum, negative = bearish momentum.
        Magnitude indicates strength.
        """
        # Normalize each component to roughly -1..+1
        # RSI: 50 = neutral, scale deviations
        rsi_signal = (rsi - 50.0) / 50.0  # -1 to +1

        # BB position: already -1 to +1
        bb_signal = bb_pos

        # ROC: scale by typical 5-min move (~0.1%)
        roc_signal = max(-1.0, min(1.0, roc / 0.10))

        # Trend: slope weighted by R² (only count confident trends)
        # Normalize slope by typical BTC movement (~$5/sec = strong)
        trend_signal = max(-1.0, min(1.0, slope / 3.0)) * r2

        # EMA cross: normalize by typical cross magnitude (~$10)
        ema_signal = max(-1.0, min(1.0, ema_cross / 10.0))

        # Weighted combination
        score = (
            0.25 * rsi_signal
            + 0.15 * bb_signal
            + 0.25 * roc_signal
            + 0.20 * trend_signal
            + 0.15 * ema_signal
        )

        return max(-1.0, min(1.0, score))

    # ── Volatility Classification ───────────────────────────────

    def _classify_volatility(self, bb_width: float, prices: list[float]) -> str:
        """Classify current volatility regime from Bollinger Band width."""
        # BB width is in % of price
        if bb_width < 0.03:
            return "low"
        elif bb_width < 0.08:
            return "normal"
        return "high"

    # ── Trade quality assessment ────────────────────────────────

    def assess_trade_quality(
        self,
        signal: TASignal,
        direction: str,
    ) -> tuple[float, str]:
        """Score a proposed trade 0.0 to 1.0 based on TA alignment.

        Returns (quality_score, reason_string).
        A score below 0.3 means TA actively disagrees with the trade.
        A score above 0.7 means strong TA confirmation.
        """
        reasons = []
        score = 0.5  # neutral starting point

        # 1. Momentum alignment: does overall momentum agree with direction?
        if direction == "UP":
            if signal.momentum_score > 0.2:
                score += 0.15
                reasons.append(f"momentum bullish ({signal.momentum_score:+.2f})")
            elif signal.momentum_score < -0.2:
                score -= 0.15
                reasons.append(f"momentum bearish ({signal.momentum_score:+.2f})")
        else:  # DOWN
            if signal.momentum_score < -0.2:
                score += 0.15
                reasons.append(f"momentum bearish ({signal.momentum_score:+.2f})")
            elif signal.momentum_score > 0.2:
                score -= 0.15
                reasons.append(f"momentum bullish ({signal.momentum_score:+.2f})")

        # 2. RSI alignment: don't buy UP when overbought, don't buy DOWN when oversold
        if direction == "UP" and signal.is_overbought:
            score -= 0.10
            reasons.append(f"RSI overbought ({signal.rsi:.0f})")
        elif direction == "DOWN" and signal.is_oversold:
            score -= 0.10
            reasons.append(f"RSI oversold ({signal.rsi:.0f})")
        elif direction == "UP" and signal.is_oversold:
            score += 0.10
            reasons.append(f"RSI oversold reversal ({signal.rsi:.0f})")
        elif direction == "DOWN" and signal.is_overbought:
            score += 0.10
            reasons.append(f"RSI overbought reversal ({signal.rsi:.0f})")

        # 3. Trend confirmation: strong trend in our direction = good
        if signal.is_strong_trend:
            if signal.trend_direction == direction:
                score += 0.15
                reasons.append(f"strong {direction} trend (R²={signal.trend_r2:.2f})")
            elif signal.trend_direction != "FLAT" and signal.trend_direction != direction:
                score -= 0.15
                reasons.append(f"strong counter-trend {signal.trend_direction} (R²={signal.trend_r2:.2f})")

        # 4. Bollinger Band position: extreme = mean reversion risk
        if direction == "UP" and signal.bb_position > 0.8:
            score -= 0.05
            reasons.append("near upper BB (reversion risk)")
        elif direction == "DOWN" and signal.bb_position < -0.8:
            score -= 0.05
            reasons.append("near lower BB (reversion risk)")

        # 5. Volatility regime: high vol = more uncertainty
        if signal.volatility_regime == "high":
            score -= 0.05
            reasons.append("high volatility")
        elif signal.volatility_regime == "low":
            score += 0.05
            reasons.append("low volatility (stable)")

        score = max(0.0, min(1.0, score))
        reason_str = " | ".join(reasons) if reasons else "neutral TA"

        return score, reason_str
