"""
Probability, expected value and sizing at *executable* prices.

Nothing here trusts the model's probability or the displayed mid:

  * the model's probability is shrunk toward the market mid
    (``p = mid + w * (p_ai - mid)``) until resolved outcomes show it is
    calibrated — AI confidence is not a calibrated probability;
  * the limit price is derived from that shrunk probability minus the
    minimum edge and a safety margin, then rounded DOWN onto the tick grid;
  * size is fractional Kelly at the limit price, capped by every exposure
    limit, then the ask side of the book is walked to get the real all-in
    cost (VWAP + taker fees) — edge is recomputed from that, not the mid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.news_trader.ai import CandidateContext, MarketAssessment
from src.news_trader.config import RiskSettings
from src.news_trader.models import Action, OrderBook, Rejection
from src.news_trader.orderbook import FillEstimate, round_to_tick, walk_asks


def blend_probability(p_ai: float, market_mid: float, weight: float) -> float:
    weight = min(1.0, max(0.0, weight))
    return min(0.999, max(0.001, market_mid + weight * (p_ai - market_mid)))


def kelly_fraction(p: float, price: float) -> float:
    """Full-Kelly bankroll fraction for a $1-payout contract bought at ``price``."""
    if price <= 0 or price >= 1:
        return 0.0
    return max(0.0, (p - price) / (1.0 - price))


@dataclass
class SizingLimits:
    """Remaining room under each limit, in USD.  The binding one wins."""
    cash_available: float
    trade_cap: float
    market_cap: float
    daily_cap: float
    correlated_cap: float

    def binding(self) -> tuple[float, str]:
        caps = {
            "cash": self.cash_available, "max_trade": self.trade_cap,
            "market_exposure": self.market_cap, "daily_exposure": self.daily_cap,
            "correlated_exposure": self.correlated_cap,
        }
        name = min(caps, key=caps.get)
        return max(0.0, caps[name]), name


@dataclass
class TradePlan:
    action: Action
    outcome: str            # "YES" or "NO"
    outcome_label: str
    token_id: str
    book: OrderBook
    p_ai_side: float
    p_used_side: float
    market_mid_side: float
    limit_price: float
    budget_usd: float
    fill: FillEstimate
    edge_after_costs: float  # per share, after fees, slippage and safety margin
    expected_profit_usd: float
    kelly_raw: float
    binding_limit: str
    notes: list[str] = field(default_factory=list)


def plan_trade(a: MarketAssessment, ctx: CandidateContext, risk: RiskSettings,
               ai_weight: float, bankroll: float, limits: SizingLimits) -> tuple[TradePlan | None, list[Rejection]]:
    if a.action == Action.BUY_YES:
        outcome, idx, book, p_ai = "YES", 0, ctx.yes_book, a.fair_probability_yes
    elif a.action == Action.BUY_NO:
        outcome, idx, book, p_ai = "NO", 1, ctx.no_book, 1.0 - a.fair_probability_yes
    else:
        return None, [Rejection("no_action", "assessment has no trade direction")]

    mid = book.mid
    if mid is None or book.best_ask is None:
        return None, [Rejection("no_book", f"{outcome} book is not two-sided")]
    p_used = blend_probability(p_ai, mid, ai_weight)

    tick = max(book.tick_size, ctx.market.tick_size)
    raw_limit = min(a.max_entry_price, p_used - risk.min_edge - risk.safety_margin, risk.max_entry_price)
    limit = round_to_tick(raw_limit, tick, "down")
    if limit < risk.min_entry_price:
        return None, [Rejection("price_bounds", f"limit {limit:.3f} below min entry price {risk.min_entry_price}")]
    if book.best_ask > limit + 1e-12:
        return None, [Rejection(
            "edge_insufficient",
            f"best {outcome} ask {book.best_ask:.3f} above max justified price {limit:.3f} "
            f"(p_ai={p_ai:.3f}, p_used={p_used:.3f}, mid={mid:.3f})",
        )]

    k_raw = kelly_fraction(p_used, limit)
    kelly_usd = risk.kelly_fraction * k_raw * bankroll
    cap, binding = limits.binding()
    budget = min(kelly_usd, cap)
    if kelly_usd < cap:
        binding = "kelly"
    min_notional = max(book.min_order_size, ctx.market.min_order_size) * book.best_ask
    if budget < min_notional:
        return None, [Rejection("size_below_minimum",
                                f"budget ${budget:.2f} (bound by {binding}) < exchange minimum ${min_notional:.2f}")]

    fill = walk_asks(book.asks, budget, limit, ctx.market.fee_rate)
    min_shares = max(book.min_order_size, ctx.market.min_order_size)
    if fill.shares < min_shares:
        return None, [Rejection("depth_insufficient",
                                f"only {fill.shares:.2f} shares available at <= {limit:.3f} (min {min_shares})")]

    edge = p_used - (fill.effective_price or 1.0) - risk.safety_margin
    if edge < risk.min_edge:
        return None, [Rejection("edge_insufficient",
                                f"edge after fees/slippage {edge:.4f} < min {risk.min_edge} "
                                f"(VWAP {fill.avg_price:.4f}, fees ${fill.fees:.4f})")]
    plan = TradePlan(
        action=a.action, outcome=outcome, outcome_label=ctx.market.outcomes[idx],
        token_id=ctx.market.token_ids[idx], book=book, p_ai_side=p_ai, p_used_side=p_used,
        market_mid_side=mid, limit_price=limit, budget_usd=round(budget, 2), fill=fill,
        edge_after_costs=edge, expected_profit_usd=fill.shares * edge, kelly_raw=k_raw,
        binding_limit=binding,
    )
    if fill.depth_exhausted:
        plan.notes.append("budget exceeds depth at limit; expect partial fill")
    return plan, []
