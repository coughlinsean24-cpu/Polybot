"""
Performance analytics for the news engine.

The questions this answers are the ones that decide whether an edge exists:

  * Did trades make money after fees and realistic fills?  (P&L, win rate,
    drawdown, fill rate — split by source, category and latency.)
  * Did waiting for analysis cost us?  (Compare our entry against buying
    immediately at the price observed when the story was first matched.)
  * Does the model add information beyond the market price?  (Brier score of
    the model's probability vs the market mid at decision time, on resolved
    markets.  If the model is not better than the mid, there is no edge.)
  * What did we miss?  (Rejected signals whose market later moved >= 5c in
    the model's predicted direction.)
"""

from __future__ import annotations

import json
from collections import defaultdict

from sqlalchemy import Integer, cast, func, select

from src.news_trader.journal import (
    AnalysisRow, EventRow, Journal, MarkRow, OrderRow, PositionRow, ResolutionRow, SignalRow,
)

LATENCY_BUCKETS = [(0, 30, "<30s"), (30, 120, "30s-2m"), (120, 600, "2m-10m"), (600, 1e12, ">10m")]
MISSED_MOVE = 0.05


def _bucket(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    for lo, hi, name in LATENCY_BUCKETS:
        if lo <= seconds < hi:
            return name
    return "unknown"


def _stats(pnls: list[float]) -> dict:
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    return {
        "trades": len(pnls),
        "net_pnl": round(sum(pnls), 4),
        "win_rate": round(len(wins) / len(pnls), 4) if pnls else None,
        "avg_win": round(sum(wins) / len(wins), 4) if wins else None,
        "avg_loss": round(sum(losses) / len(losses), 4) if losses else None,
        "avg_pnl": round(sum(pnls) / len(pnls), 4) if pnls else None,
    }


def max_drawdown(pnls_in_time_order: list[float]) -> float:
    peak = cum = dd = 0.0
    for p in pnls_in_time_order:
        cum += p
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    return round(dd, 4)


def brier(pairs: list[tuple[float, float]]) -> float | None:
    return round(sum((p - y) ** 2 for p, y in pairs) / len(pairs), 5) if pairs else None


def performance_report(journal: Journal, mode: str = "paper") -> dict:
    with journal.Session() as s:
        events_total = s.scalar(select(func.count(EventRow.event_id))) or 0
        events_unique = s.scalar(select(func.count(EventRow.event_id)).where(EventRow.is_duplicate.is_(False))) or 0
        status_counts = dict(s.execute(select(EventRow.status, func.count()).group_by(EventRow.status)).all())
        signals = list(s.scalars(select(SignalRow)))
        orders = list(s.scalars(select(OrderRow).where(OrderRow.mode == mode)))
        positions = list(s.scalars(select(PositionRow).where(PositionRow.mode == mode)))
        resolutions = {r.condition_id: r for r in s.scalars(select(ResolutionRow))}
        marks = list(s.scalars(select(MarkRow).where(MarkRow.observed_at.is_not(None))))
        ai_rows = s.execute(select(
            func.coalesce(func.sum(cast(AnalysisRow.cached.is_(False), Integer)), 0),
            func.coalesce(func.sum(AnalysisRow.cost_usd), 0.0),
            func.coalesce(func.sum(cast(AnalysisRow.cached, Integer)), 0),
            func.coalesce(func.sum(cast(AnalysisRow.ok.is_(False), Integer)), 0),
        )).one()

    qualifying = sum(v for k, v in status_counts.items() if k in ("traded", "signals_rejected", "ai_failed"))
    trade_signals = [x for x in signals if x.decision == "TRADE"]
    rejection_counts: dict[str, int] = defaultdict(int)
    for x in signals:
        if x.decision == "REJECT":
            for c in json.loads(x.rejection_codes or "[]"):
                rejection_counts[c] += 1

    requested = sum((o.requested_cost or 0) / max(o.limit_price or 1, 1e-9) for o in orders)
    filled_shares = sum(o.filled_shares or 0 for o in orders)
    fills = {
        "orders": len(orders),
        "filled": sum(o.status == "FILLED" for o in orders),
        "partial": sum(o.status == "PARTIAL" for o in orders),
        "unfilled": sum(o.status == "UNFILLED" for o in orders),
        "rejected": sum(o.status == "REJECTED" for o in orders),
        "fill_rate_by_count": round(sum((o.filled_shares or 0) > 0 for o in orders) / len(orders), 4) if orders else None,
        "fill_rate_by_shares": round(filled_shares / requested, 4) if requested else None,
    }

    closed = sorted([p for p in positions if p.status != "OPEN" and p.realized_pnl is not None],
                    key=lambda p: p.closed_at or 0)
    open_pos = [p for p in positions if p.status == "OPEN"]
    pnls = [p.realized_pnl for p in closed]
    overall = _stats(pnls)
    overall["max_drawdown"] = max_drawdown(pnls)
    overall["unrealized_pnl"] = round(sum(p.unrealized_pnl or 0 for p in open_pos), 4)
    overall["open_positions"] = len(open_pos)

    def split(key) -> dict:
        groups: dict[str, list[float]] = defaultdict(list)
        for p in closed:
            groups[key(p) or "unknown"].append(p.realized_pnl)
        return {k: _stats(v) for k, v in sorted(groups.items())}

    # Baseline: same shares bought at the side ask observed when the story was
    # first matched, held the same way.  Positive "delay_cost" = waiting hurt.
    baseline_pnl = actual_pnl = 0.0
    n_base = 0
    for p in closed:
        if p.baseline_price is None or p.exit_proceeds is None:
            continue
        base_cost = p.shares * p.baseline_price + (p.entry_fees or 0)
        baseline_pnl += (p.exit_proceeds - (p.exit_fees or 0)) - base_cost
        actual_pnl += p.realized_pnl
        n_base += 1

    # Calibration on resolved markets: model vs market mid at decision.
    model_pairs, market_pairs = [], []
    for x in signals:
        if x.p_ai_yes is None or x.yes_mid_at_decision is None:
            continue
        r = resolutions.get(x.condition_id)
        if r is None or r.resolved_yes is None:
            continue
        y = 1.0 if r.resolved_yes else 0.0
        model_pairs.append((x.p_ai_yes, y))
        market_pairs.append((x.yes_mid_at_decision, y))

    # Missed opportunities: rejected, directional signals whose YES mid moved
    # >= MISSED_MOVE in the predicted direction by the 1h mark.
    sig_by_id = {x.signal_id: x for x in signals}
    missed: list[dict] = []
    moves_by_horizon: dict[int, list[float]] = defaultdict(list)
    for m in marks:
        x = sig_by_id.get(m.signal_id)
        if x is None or m.yes_mid is None or x.yes_mid_at_decision is None or x.p_ai_yes is None:
            continue
        direction = 1.0 if x.p_ai_yes > x.yes_mid_at_decision else -1.0
        move = (m.yes_mid - x.yes_mid_at_decision) * direction
        moves_by_horizon[m.horizon_seconds].append(move)
        if x.decision == "REJECT" and m.horizon_seconds == 3600 and move >= MISSED_MOVE:
            missed.append({"signal_id": x.signal_id, "question": x.question, "move": round(move, 4),
                           "rejection_codes": json.loads(x.rejection_codes or "[]")})

    return {
        "mode": mode,
        "events": {"total": events_total, "unique": events_unique, "qualifying": qualifying,
                   "by_status": status_counts},
        "signals": {"assessed": len(signals), "trade_signals": len(trade_signals),
                    "rejection_counts": dict(sorted(rejection_counts.items(), key=lambda kv: -kv[1]))},
        "ai": {"calls": int(ai_rows[0]), "cost_usd": round(float(ai_rows[1]), 4),
               "cache_hits": int(ai_rows[2]), "failures": int(ai_rows[3])},
        "fills": fills,
        "performance": overall,
        "by_source": split(lambda p: p.source_name),
        "by_category": split(lambda p: p.market_category),
        "by_latency": split(lambda p: _bucket(p.detection_to_decision)),
        "vs_immediate_entry": {
            "positions_compared": n_base,
            "actual_pnl": round(actual_pnl, 4),
            "immediate_entry_pnl": round(baseline_pnl, 4),
            "delay_cost": round(baseline_pnl - actual_pnl, 4),
        },
        "calibration": {
            "resolved_signals": len(model_pairs),
            "brier_model": brier(model_pairs),
            "brier_market_mid": brier(market_pairs),
            "model_beats_market": (brier(model_pairs) < brier(market_pairs)) if model_pairs else None,
        },
        "post_signal_moves": {str(h): {"n": len(v), "avg_move_in_predicted_direction": round(sum(v) / len(v), 4)}
                              for h, v in sorted(moves_by_horizon.items()) if v},
        "missed_opportunities": {"count": len(missed), "examples": missed[:20]},
    }
