"""
Trade journal for the news engine (separate SQLite DB, WAL mode).

Records the complete lifecycle — every event, match, AI analysis, signal
(including rejected ones and why), order, position, price mark and market
resolution — with timestamps at each stage, so performance can be analysed
and decisions replayed without look-ahead.

Kept in its own database file so the BTC engine's trades.db is untouched.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Boolean, Column, Float, Integer, String, Text, UniqueConstraint, create_engine, event, func,
    inspect, select, text,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from src.news_trader.models import MarketRecord, NewsEvent, OrderBook

logger = logging.getLogger(__name__)

Base = declarative_base()


def _j(v: Any) -> str:
    return json.dumps(v, default=str)


class EventRow(Base):
    __tablename__ = "news_events"
    event_id = Column(String, primary_key=True)
    cluster_id = Column(String, index=True)
    kind = Column(String)                 # new | duplicate | upgrade
    provider = Column(String, index=True)
    source_name = Column(String)
    source_domain = Column(String)
    url = Column(Text)
    title = Column(Text)
    summary = Column(Text)
    category = Column(String, index=True)
    credibility = Column(Float)
    verification = Column(String)
    corroborating = Column(Text)
    flags = Column(Text)
    is_duplicate = Column(Boolean)
    duplicate_of = Column(String)
    realtime_source = Column(Boolean)
    primary_source = Column(Boolean)
    published_at = Column(Float)
    first_seen_at = Column(Float, index=True)
    ingested_at = Column(Float)
    received_at = Column(Float)           # when the decision pipeline picked it up (replay clock)
    status = Column(String, index=True)   # pending|duplicate|prefilter_rejected|no_match|...|traded
    status_detail = Column(Text)
    processed_at = Column(Float)


class MatchRow(Base):
    __tablename__ = "news_matches"
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String, index=True)
    market_id = Column(String, index=True)
    condition_id = Column(String)
    question = Column(Text)
    score = Column(Float)
    matched_terms = Column(Text)
    reasons = Column(Text)
    tradeable = Column(Boolean)
    tradeable_detail = Column(Text)
    resolution_ok = Column(Boolean)
    ambiguity = Column(Float)
    resolution_issues = Column(Text)
    market_json = Column(Text)            # point-in-time market metadata for replay
    created_at = Column(Float)


class AnalysisRow(Base):
    __tablename__ = "news_ai_analyses"
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String, index=True)
    cluster_id = Column(String)
    created_at = Column(Float, index=True)
    day = Column(String, index=True)
    model = Column(String)
    served_by = Column(String)
    ok = Column(Boolean)
    error = Column(Text)
    cached = Column(Boolean)
    latency_ms = Column(Float)
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    cache_read_tokens = Column(Integer)
    cache_write_tokens = Column(Integer)
    cost_usd = Column(Float)
    request_id = Column(String)
    market_ids = Column(Text)
    analysis_json = Column(Text)


class SignalRow(Base):
    __tablename__ = "news_signals"
    signal_id = Column(String, primary_key=True)
    event_id = Column(String, index=True)
    cluster_id = Column(String, index=True)
    market_id = Column(String, index=True)
    condition_id = Column(String)
    question = Column(Text)
    market_category = Column(String)
    event_category = Column(String)
    event_group = Column(String)
    source_name = Column(String)
    provider = Column(String)
    verification = Column(String)
    stage = Column(String)                # pre_ai | ai | risk | execution
    action = Column(String)               # BUY_YES | BUY_NO | NO_TRADE
    decision = Column(String, index=True)  # TRADE | REJECT
    rejection_codes = Column(Text)
    rejection_detail = Column(Text)
    ai_model = Column(String)
    ai_ok = Column(Boolean)
    ai_error = Column(Text)
    relevance = Column(String)
    rule_compatibility = Column(String)
    rule_analysis = Column(Text)
    priced_in = Column(Text)
    evidence = Column(Text)
    p_ai_yes = Column(Float)
    p_low = Column(Float)
    p_high = Column(Float)
    confidence = Column(Float)
    p_used_side = Column(Float)
    limit_price = Column(Float)
    planned_shares = Column(Float)
    planned_cost = Column(Float)
    edge_after_costs = Column(Float)
    expected_profit = Column(Float)
    binding_limit = Column(String)
    # stage timestamps (epoch seconds)
    published_at = Column(Float)
    first_seen_at = Column(Float)
    matched_at = Column(Float)
    ai_started_at = Column(Float)
    ai_completed_at = Column(Float)
    decision_at = Column(Float, index=True)
    # prices at stages (YES token unless noted)
    yes_bid_at_match = Column(Float)
    yes_ask_at_match = Column(Float)
    yes_mid_at_decision = Column(Float)
    side_ask_at_match = Column(Float)     # ask of the side the signal points to
    side_ask_at_decision = Column(Float)
    spread_at_decision = Column(Float)
    news_age_at_decision = Column(Float)
    detection_to_decision = Column(Float)


class OrderRow(Base):
    __tablename__ = "news_orders"
    order_id = Column(String, primary_key=True)  # client order id (idempotency key)
    signal_id = Column(String, unique=True)
    mode = Column(String)
    market_id = Column(String, index=True)
    token_id = Column(String)
    outcome = Column(String)
    limit_price = Column(Float)
    requested_cost = Column(Float)
    status = Column(String)
    filled_shares = Column(Float)
    avg_price = Column(Float)
    fees = Column(Float)
    notional = Column(Float)
    exchange_order_id = Column(String)
    error = Column(Text)
    submitted_at = Column(Float)
    completed_at = Column(Float)
    side_ask_at_submit = Column(Float)
    fills = Column(Text)


class PositionRow(Base):
    __tablename__ = "news_positions"
    position_id = Column(String, primary_key=True)
    signal_id = Column(String, index=True)
    mode = Column(String)
    market_id = Column(String, index=True)
    condition_id = Column(String)
    question = Column(Text)
    event_group = Column(String, index=True)
    cluster_id = Column(String, index=True)
    token_id = Column(String)
    outcome = Column(String)
    outcome_label = Column(String)
    market_category = Column(String)
    source_name = Column(String)
    provider = Column(String)
    shares = Column(Float)
    entry_price = Column(Float)
    entry_cost = Column(Float)            # notional + fees
    entry_fees = Column(Float)
    p_used = Column(Float)
    baseline_price = Column(Float)        # side ask when the news was first matched
    news_age_at_decision = Column(Float)
    detection_to_decision = Column(Float)
    opened_at = Column(Float, index=True)
    status = Column(String, index=True)   # OPEN | CLOSED | RESOLVED
    last_mark_bid = Column(Float)
    last_mark_at = Column(Float)
    unrealized_pnl = Column(Float)
    exit_price = Column(Float)
    exit_proceeds = Column(Float)
    exit_fees = Column(Float)
    closed_at = Column(Float, index=True)
    close_reason = Column(String)
    realized_pnl = Column(Float)
    resolved_yes = Column(Boolean)


class MarkRow(Base):
    __tablename__ = "news_price_marks"
    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(String, index=True)
    market_id = Column(String)
    yes_token = Column(String)
    horizon_seconds = Column(Integer)
    due_at = Column(Float, index=True)
    observed_at = Column(Float)
    yes_bid = Column(Float)
    yes_ask = Column(Float)
    yes_mid = Column(Float)
    __table_args__ = (UniqueConstraint("signal_id", "horizon_seconds"),)


class ResolutionRow(Base):
    __tablename__ = "news_resolutions"
    condition_id = Column(String, primary_key=True)
    market_id = Column(String)
    resolved_yes = Column(Boolean)
    outcome_prices = Column(Text)
    resolved_at = Column(Float)
    last_checked_at = Column(Float)


class BookSnapshotRow(Base):
    __tablename__ = "news_book_snapshots"
    id = Column(Integer, primary_key=True, autoincrement=True)
    token_id = Column(String, index=True)
    market_id = Column(String)
    observed_at = Column(Float, index=True)
    exchange_ts = Column(Float)
    tick_size = Column(Float)
    min_order_size = Column(Float)
    bids = Column(Text)
    asks = Column(Text)


def utc_day(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def day_start(ts: float) -> float:
    d = datetime.fromtimestamp(ts, tz=timezone.utc)
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp()


def market_to_json(m: MarketRecord) -> str:
    return _j(m.__dict__)


def market_from_json(s: str) -> MarketRecord:
    return MarketRecord(**json.loads(s))


class Journal:
    def __init__(self, db_path: str):
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)

        @event.listens_for(self.engine, "connect")
        def _pragma(dbapi_conn, _):
            cur = dbapi_conn.cursor()
            if db_path != ":memory:":
                cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.close()

        Base.metadata.create_all(self.engine)
        self._add_missing_columns()
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)

    def _add_missing_columns(self) -> None:
        """Forward-only migration: add columns introduced after a DB was created."""
        insp = inspect(self.engine)
        with self.engine.begin() as conn:
            for table in Base.metadata.sorted_tables:
                existing = {c["name"] for c in insp.get_columns(table.name)}
                for col in table.columns:
                    if col.name not in existing:
                        ddl = col.type.compile(dialect=self.engine.dialect)
                        conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN "{col.name}" {ddl}'))
                        logger.info("journal migration: added %s.%s", table.name, col.name)

    # ── writes ───────────────────────────────────────────────────────────

    def record_event(self, ev: NewsEvent, kind: str, status: str = "pending", detail: str = "",
                     received_at: float | None = None) -> None:
        with self.Session() as s:
            row = s.get(EventRow, ev.event_id) or EventRow(event_id=ev.event_id)
            row.cluster_id, row.kind, row.provider = ev.cluster_id, kind, ev.provider
            row.source_name, row.source_domain, row.url = ev.source_name, ev.source_domain, ev.url
            row.title, row.summary, row.category = ev.title, ev.summary, ev.category
            row.credibility, row.verification = ev.credibility, ev.verification.value
            row.corroborating, row.flags = _j(ev.corroborating_sources), _j(ev.flags)
            row.is_duplicate, row.duplicate_of = ev.is_duplicate, ev.duplicate_of
            row.realtime_source, row.primary_source = ev.is_realtime_source, ev.is_primary_source
            row.published_at, row.first_seen_at, row.ingested_at = ev.published_at, ev.first_seen_at, ev.ingested_at
            row.status, row.status_detail = status, detail[:2000]
            row.received_at = received_at if received_at is not None else row.received_at
            s.merge(row)
            s.commit()

    def set_event_status(self, event_id: str, status: str, detail: str, at: float) -> None:
        with self.Session() as s:
            row = s.get(EventRow, event_id)
            if row:
                row.status, row.status_detail, row.processed_at = status, detail[:2000], at
                s.commit()

    def record_match(self, event_id: str, m: MarketRecord, score: float, terms: list[str], reasons: list[str],
                     tradeable: bool, tradeable_detail: str, resolution, at: float) -> None:
        with self.Session() as s:
            s.add(MatchRow(
                event_id=event_id, market_id=m.market_id, condition_id=m.condition_id,
                question=m.question, score=score, matched_terms=_j(terms), reasons=_j(reasons),
                tradeable=tradeable, tradeable_detail=tradeable_detail,
                resolution_ok=None if resolution is None else resolution.ok,
                ambiguity=None if resolution is None else resolution.ambiguity_score,
                resolution_issues=_j([] if resolution is None else resolution.issues),
                market_json=market_to_json(m), created_at=at,
            ))
            s.commit()

    def record_analysis(self, event: NewsEvent, market_ids: list[str], result, at: float) -> None:
        with self.Session() as s:
            s.add(AnalysisRow(
                event_id=event.event_id, cluster_id=event.cluster_id, created_at=at, day=utc_day(at),
                model=result.model, served_by=result.served_by, ok=result.ok, error=result.error[:2000],
                cached=result.cached, latency_ms=result.latency_ms, input_tokens=result.input_tokens,
                output_tokens=result.output_tokens, cache_read_tokens=result.cache_read_tokens,
                cache_write_tokens=result.cache_write_tokens, cost_usd=result.cost_usd,
                request_id=result.request_id, market_ids=_j(market_ids),
                analysis_json=result.analysis.model_dump_json() if result.analysis else None,
            ))
            s.commit()

    def record_signal(self, **fields) -> None:
        for k in ("rejection_codes", "evidence"):
            if k in fields and not isinstance(fields[k], str):
                fields[k] = _j(fields[k])
        with self.Session() as s:
            s.merge(SignalRow(**fields))
            s.commit()

    def record_order(self, **fields) -> None:
        if "fills" in fields and not isinstance(fields["fills"], str):
            fields["fills"] = _j(fields["fills"])
        with self.Session() as s:
            s.merge(OrderRow(**fields))
            s.commit()

    def order_exists(self, order_id: str) -> bool:
        with self.Session() as s:
            return s.get(OrderRow, order_id) is not None

    def open_position(self, **fields) -> None:
        with self.Session() as s:
            s.merge(PositionRow(status="OPEN", **fields))
            s.commit()

    def update_position(self, position_id: str, **fields) -> None:
        with self.Session() as s:
            row = s.get(PositionRow, position_id)
            if row:
                for k, v in fields.items():
                    setattr(row, k, v)
                s.commit()

    def schedule_marks(self, signal_id: str, market_id: str, yes_token: str, horizons: list[int], at: float) -> None:
        with self.Session() as s:
            for h in horizons:
                s.merge(MarkRow(signal_id=signal_id, market_id=market_id, yes_token=yes_token,
                                horizon_seconds=h, due_at=at + h))
            s.commit()

    def due_marks(self, now: float, limit: int = 50) -> list[MarkRow]:
        with self.Session() as s:
            return list(s.scalars(
                select(MarkRow).where(MarkRow.observed_at.is_(None), MarkRow.due_at <= now)
                .order_by(MarkRow.due_at).limit(limit)
            ))

    def fill_mark(self, mark_id: int, at: float, bid: float | None, ask: float | None) -> None:
        with self.Session() as s:
            row = s.get(MarkRow, mark_id)
            if row:
                row.observed_at, row.yes_bid, row.yes_ask = at, bid, ask
                row.yes_mid = (bid + ask) / 2 if bid is not None and ask is not None else None
                s.commit()

    def record_resolution(self, condition_id: str, market_id: str, resolved_yes: bool | None,
                          outcome_prices: list, at: float) -> None:
        with self.Session() as s:
            row = s.get(ResolutionRow, condition_id) or ResolutionRow(condition_id=condition_id)
            row.market_id, row.last_checked_at = market_id, at
            if resolved_yes is not None and row.resolved_yes is None:
                row.resolved_yes, row.resolved_at, row.outcome_prices = resolved_yes, at, _j(outcome_prices)
            s.merge(row)
            s.commit()

    def record_book(self, book: OrderBook, market_id: str, depth: int = 10) -> None:
        with self.Session() as s:
            s.add(BookSnapshotRow(
                token_id=book.token_id, market_id=market_id, observed_at=book.fetched_at,
                exchange_ts=book.exchange_ts, tick_size=book.tick_size, min_order_size=book.min_order_size,
                bids=_j([(lv.price, lv.size) for lv in book.bids[:depth]]),
                asks=_j([(lv.price, lv.size) for lv in book.asks[:depth]]),
            ))
            s.commit()

    # ── reads ────────────────────────────────────────────────────────────

    def ai_usage_on(self, day: str) -> tuple[int, float]:
        with self.Session() as s:
            calls, cost = s.execute(
                select(func.count(AnalysisRow.id), func.coalesce(func.sum(AnalysisRow.cost_usd), 0.0))
                .where(AnalysisRow.day == day, AnalysisRow.cached.is_(False), AnalysisRow.cost_usd > 0)
            ).one()
            return int(calls), float(cost)

    def traded_pairs(self, since: float) -> set[tuple[str, str]]:
        with self.Session() as s:
            rows = s.execute(select(SignalRow.cluster_id, SignalRow.market_id)
                             .where(SignalRow.decision == "TRADE", SignalRow.decision_at >= since)).all()
            return {(r[0], r[1]) for r in rows}

    def last_trade_at(self, market_id: str) -> float | None:
        with self.Session() as s:
            return s.scalar(select(func.max(SignalRow.decision_at))
                            .where(SignalRow.market_id == market_id, SignalRow.decision == "TRADE"))

    def open_positions(self, mode: str | None = None) -> list[PositionRow]:
        with self.Session() as s:
            q = select(PositionRow).where(PositionRow.status == "OPEN")
            if mode:
                q = q.where(PositionRow.mode == mode)
            return list(s.scalars(q))

    def positions(self, limit: int = 200) -> list[PositionRow]:
        with self.Session() as s:
            return list(s.scalars(select(PositionRow).order_by(PositionRow.opened_at.desc()).limit(limit)))

    def realized_pnl_since(self, since: float, mode: str) -> float:
        with self.Session() as s:
            return float(s.scalar(select(func.coalesce(func.sum(PositionRow.realized_pnl), 0.0))
                                  .where(PositionRow.closed_at >= since, PositionRow.mode == mode)) or 0.0)

    def realized_pnl_total(self, mode: str) -> float:
        with self.Session() as s:
            return float(s.scalar(select(func.coalesce(func.sum(PositionRow.realized_pnl), 0.0))
                                  .where(PositionRow.mode == mode, PositionRow.status != "OPEN")) or 0.0)

    def entry_cost_since(self, since: float, mode: str) -> float:
        with self.Session() as s:
            return float(s.scalar(select(func.coalesce(func.sum(PositionRow.entry_cost), 0.0))
                                  .where(PositionRow.opened_at >= since, PositionRow.mode == mode)) or 0.0)

    def resolution_for(self, condition_id: str) -> ResolutionRow | None:
        with self.Session() as s:
            return s.get(ResolutionRow, condition_id)

    def conditions_needing_resolution(self) -> list[tuple[str, str]]:
        """(condition_id, market_id) for open positions and AI-assessed signals not yet resolved."""
        with self.Session() as s:
            resolved = select(ResolutionRow.condition_id).where(ResolutionRow.resolved_yes.is_not(None))
            pos = s.execute(select(PositionRow.condition_id, PositionRow.market_id)
                            .where(PositionRow.status == "OPEN")).all()
            sig = s.execute(select(SignalRow.condition_id, SignalRow.market_id)
                            .where(SignalRow.p_ai_yes.is_not(None), SignalRow.condition_id.not_in(resolved))
                            .distinct()).all()
            return list({(c, m) for c, m in [*pos, *sig] if c})

    def book_snapshots(self, token_id: str) -> list[BookSnapshotRow]:
        with self.Session() as s:
            return list(s.scalars(select(BookSnapshotRow).where(BookSnapshotRow.token_id == token_id)
                                  .order_by(BookSnapshotRow.observed_at)))

    def rows(self, model, order_col, limit: int = 100, where=None) -> list:
        with self.Session() as s:
            q = select(model)
            if where is not None:
                q = q.where(where)
            return list(s.scalars(q.order_by(order_col.desc()).limit(limit)))
