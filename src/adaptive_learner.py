"""
Adaptive learning module for Polybot.

Tracks win/loss outcomes bucketed by observable conditions at trade time:
  - Volatility regime  (tight / moderate / strong)
  - Momentum direction (fresh / trending / reverting)
  - Edge strength      (weak / decent / big)

3 × 3 × 3 = 27 buckets — fills ~3× faster than the old 81-bucket
design while capturing the extra signal of *how much* edge we had.

After enough samples accumulate, the learner adjusts:
  1. Minimum edge threshold — raises it in conditions that lose often,
     lowers it (down to a floor) in conditions that win often.
  2. Provides a calibrated win-rate estimate per bucket that the
     arbitrage engine can use instead of the naive normal-CDF model.

State persists to a JSON file so learning survives restarts.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

STATE_FILE = "data/adaptive_state.json"
MIN_SAMPLES = 2          # need at least N trades in a bucket before adjusting
BASE_MIN_EDGE = 0.01     # never go below this
MAX_MIN_EDGE = 0.15      # never require more than this
LEARNING_RATE = 0.25     # how fast adjustments move toward observed rates


@dataclass
class BucketStats:
    """Stats for one condition bucket."""
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    avg_edge: float = 0.0
    adjusted_min_edge: float = BASE_MIN_EDGE

    @property
    def total(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        return self.wins / self.total if self.total > 0 else 0.5

    def to_dict(self) -> dict:
        return {
            "wins": self.wins,
            "losses": self.losses,
            "total_pnl": round(self.total_pnl, 4),
            "avg_edge": round(self.avg_edge, 6),
            "adjusted_min_edge": round(self.adjusted_min_edge, 6),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BucketStats":
        return cls(
            wins=d.get("wins", 0),
            losses=d.get("losses", 0),
            total_pnl=d.get("total_pnl", 0.0),
            avg_edge=d.get("avg_edge", 0.0),
            adjusted_min_edge=d.get("adjusted_min_edge", BASE_MIN_EDGE),
        )


class AdaptiveLearner:
    """Learns from past trades and adjusts strategy parameters."""

    def __init__(self, config: dict):
        self.base_min_edge = config["strategy"].get("min_edge", BASE_MIN_EDGE)
        self.buckets: dict[str, BucketStats] = {}
        self._load_state()

    # ── Bucketing logic ─────────────────────────────────────────────
    #
    # 3 dimensions: volatility × momentum × edge_strength = 27 buckets.
    # Fills fast (~54 trades to see every bucket once) while capturing
    # whether big-edge trades win at different rates than small-edge ones.

    @staticmethod
    def _volatility_bucket(diff_pct: float) -> str:
        """Classify BTC distance from target into volatility regime."""
        abs_d = abs(diff_pct)
        if abs_d < 0.02:
            return "tight"       # BTC very close to target — coin-flip
        elif abs_d < 0.06:
            return "moderate"    # mild move — most common
        else:
            return "strong"      # big move — should be high-conviction

    @staticmethod
    def _momentum_bucket(diff_pct: float, prev_diff_pct: float | None) -> str:
        """Is price moving further from target (trending) or back (reverting)?"""
        if prev_diff_pct is None:
            return "fresh"       # first scan of window — no prior data
        if abs(diff_pct) > abs(prev_diff_pct) * 1.05:
            return "trending"    # price accelerating away from target
        elif abs(diff_pct) < abs(prev_diff_pct) * 0.95:
            return "reverting"   # price coming back toward target
        else:
            return "fresh"       # barely changed — treat as fresh

    @staticmethod
    def _edge_bucket(edge: float) -> str:
        """Classify the edge at entry time."""
        if edge < 0.08:
            return "weak"        # thin edge — more likely noise
        elif edge < 0.20:
            return "decent"      # solid edge — bread & butter
        else:
            return "big"         # fat edge — high conviction

    def _make_key(
        self,
        diff_pct: float,
        edge: float,
        seconds_left: float,
        prev_diff_pct: float | None = None,
    ) -> str:
        """Build a composite bucket key from trade conditions."""
        vol = self._volatility_bucket(diff_pct)
        mom = self._momentum_bucket(diff_pct, prev_diff_pct)
        edg = self._edge_bucket(edge)
        return f"{vol}|{mom}|{edg}"

    # ── Recording outcomes ──────────────────────────────────────────

    def record_outcome(
        self,
        won: bool,
        pnl: float,
        diff_pct: float,
        edge: float,
        seconds_left: float,
        prev_diff_pct: float | None = None,
    ):
        """Record a trade outcome and update bucket stats."""
        key = self._make_key(diff_pct, edge, seconds_left, prev_diff_pct)
        bucket = self.buckets.setdefault(key, BucketStats())

        if won:
            bucket.wins += 1
        else:
            bucket.losses += 1
        bucket.total_pnl += pnl

        # Running average of edge at entry
        n = bucket.total
        bucket.avg_edge = bucket.avg_edge + (edge - bucket.avg_edge) / n

        # Adjust min_edge for this bucket after enough samples
        if bucket.total >= MIN_SAMPLES:
            self._adjust_edge(bucket)

        self._save_state()

        logger.info(
            f"📚 Learner: [{key}] → {'WIN' if won else 'LOSS'} | "
            f"Bucket: {bucket.wins}W/{bucket.losses}L ({bucket.win_rate:.0%}) | "
            f"Adj edge: {bucket.adjusted_min_edge:.4f}"
        )

    def _adjust_edge(self, bucket: BucketStats):
        """Adjust the minimum edge for a bucket based on observed win rate.

        If the bucket wins a lot (>60%), we can afford to trade with a
        smaller edge.  If it loses often (<45%), we raise the bar.
        """
        wr = bucket.win_rate

        if wr >= 0.60:
            # Good bucket — lower the bar (but not below floor)
            target = self.base_min_edge * 0.75
        elif wr >= 0.50:
            # Marginal — keep near default
            target = self.base_min_edge
        elif wr >= 0.40:
            # Losing slightly — raise the bar
            target = self.base_min_edge * 1.5
        else:
            # Bad bucket — require much more edge
            target = self.base_min_edge * 2.5

        # Smooth toward target
        bucket.adjusted_min_edge += LEARNING_RATE * (target - bucket.adjusted_min_edge)
        bucket.adjusted_min_edge = max(BASE_MIN_EDGE * 0.5, min(MAX_MIN_EDGE, bucket.adjusted_min_edge))

    # ── Query methods (used by arbitrage engine) ────────────────────

    def get_adjusted_min_edge(
        self,
        diff_pct: float,
        edge: float,
        seconds_left: float,
        prev_diff_pct: float | None = None,
    ) -> float:
        """Get the learned minimum edge for these conditions.

        Returns the default min_edge if we haven't seen enough data yet.
        """
        key = self._make_key(diff_pct, edge, seconds_left, prev_diff_pct)
        bucket = self.buckets.get(key)
        if bucket and bucket.total >= MIN_SAMPLES:
            return bucket.adjusted_min_edge
        return self.base_min_edge

    def get_adjusted_probability(
        self,
        naive_prob: float,
        diff_pct: float,
        edge: float,
        seconds_left: float,
        prev_diff_pct: float | None = None,
    ) -> float:
        """Blend the naive model probability with observed win rate.

        If we have enough data for this bucket, we nudge the probability
        toward the empirical win rate.  This corrects for model mis-
        calibration in specific regimes.
        """
        key = self._make_key(diff_pct, edge, seconds_left, prev_diff_pct)
        bucket = self.buckets.get(key)
        if bucket and bucket.total >= MIN_SAMPLES:
            empirical = bucket.win_rate
            # Blend: weight empirical more as samples grow
            weight = min(bucket.total / 50.0, 0.5)  # max 50% weight to empirical
            blended = (1 - weight) * naive_prob + weight * empirical
            logger.debug(
                f"Prob adjustment [{key}]: naive={naive_prob:.3f} "
                f"empirical={empirical:.3f} → blended={blended:.3f} "
                f"(weight={weight:.2f}, n={bucket.total})"
            )
            return blended
        return naive_prob

    def should_skip_conditions(
        self,
        diff_pct: float,
        edge: float,
        seconds_left: float,
        prev_diff_pct: float | None = None,
    ) -> tuple[bool, str]:
        """Check if we should skip trading in these conditions entirely.

        Skips if we have evidence (≥6 trades) of <35% win rate.
        Also checks the 2-dim parent bucket (vol|mom) as a fallback
        so we don't need full 3-dim data to block bad conditions.
        """
        key = self._make_key(diff_pct, edge, seconds_left, prev_diff_pct)
        bucket = self.buckets.get(key)
        if bucket and bucket.total >= 6 and bucket.win_rate < 0.35:
            return True, (
                f"Learner skip [{key}]: {bucket.win_rate:.0%} win rate "
                f"over {bucket.total} trades"
            )

        # Fallback: check 2-dim parent (vol|mom) by aggregating all edge tiers
        vol = self._volatility_bucket(diff_pct)
        mom = self._momentum_bucket(diff_pct, prev_diff_pct)
        parent_prefix = f"{vol}|{mom}|"
        parent_wins = sum(b.wins for k, b in self.buckets.items() if k.startswith(parent_prefix))
        parent_losses = sum(b.losses for k, b in self.buckets.items() if k.startswith(parent_prefix))
        parent_total = parent_wins + parent_losses
        if parent_total >= 8:
            parent_wr = parent_wins / parent_total
            if parent_wr < 0.35:
                return True, (
                    f"Learner skip [{vol}|{mom}|*]: {parent_wr:.0%} win rate "
                    f"over {parent_total} trades (aggregated)"
                )
        return False, ""

    def get_summary(self) -> dict:
        """Get a summary of all buckets for the dashboard."""
        summary = {}
        for key, bucket in sorted(self.buckets.items()):
            if bucket.total > 0:
                summary[key] = {
                    "wins": bucket.wins,
                    "losses": bucket.losses,
                    "win_rate": round(bucket.win_rate, 3),
                    "total_pnl": round(bucket.total_pnl, 2),
                    "adjusted_min_edge": round(bucket.adjusted_min_edge, 4),
                    "samples": bucket.total,
                }
        return summary

    # ── Persistence ─────────────────────────────────────────────────

    def _save_state(self):
        """Save bucket stats to disk."""
        try:
            Path(STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
            data = {key: bucket.to_dict() for key, bucket in self.buckets.items()}
            with open(STATE_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save adaptive state: {e}")

    def _load_state(self):
        """Load bucket stats from disk."""
        try:
            if Path(STATE_FILE).exists():
                with open(STATE_FILE) as f:
                    data = json.load(f)
                self.buckets = {
                    key: BucketStats.from_dict(val) for key, val in data.items()
                }
                total_trades = sum(b.total for b in self.buckets.values())
                logger.info(
                    f"Adaptive learner loaded: {len(self.buckets)} buckets, "
                    f"{total_trades} total observations"
                )
            else:
                logger.info("Adaptive learner starting fresh (no prior state)")
        except Exception as e:
            logger.warning(f"Failed to load adaptive state: {e}")
            self.buckets = {}
