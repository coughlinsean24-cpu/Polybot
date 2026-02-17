"""
Trade logging and analytics using SQLite via SQLAlchemy.
Records every signal, decision, and outcome for post-analysis.
"""

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    text,
    Boolean,
    create_engine,
    desc,
)
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()


class TradeRecord(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    session_id = Column(String, nullable=True)

    # BTC price data
    btc_price_start = Column(Float)
    btc_price_end = Column(Float)
    delta_pct = Column(Float)

    # Polymarket data
    market_id = Column(String)
    market_question = Column(String)
    polymarket_odds_yes = Column(Float)
    polymarket_odds_no = Column(Float)

    # Trade details
    direction = Column(String)  # YES or NO
    bet_size = Column(Float)
    fill_price = Column(Float)
    order_id = Column(String)

    # Outcome
    outcome = Column(String)  # WIN, LOSS, PENDING
    profit_loss = Column(Float)

    # State at time of trade
    bankroll_before = Column(Float)
    bankroll_after = Column(Float)
    consecutive_wins = Column(Integer)
    edge_estimate = Column(Float)
    confidence = Column(Float)


class SignalRecord(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    session_id = Column(String, nullable=True)
    direction = Column(String)
    delta_pct = Column(Float)
    confidence = Column(Float)
    btc_price_start = Column(Float)
    btc_price_end = Column(Float)
    traded = Column(Boolean, default=False)
    reason = Column(String)  # why we did or didn't trade


class BotSession(Base):
    __tablename__ = "bot_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, unique=True, nullable=False)
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    ended_at = Column(DateTime, nullable=True)
    mode = Column(String)  # always "live"
    starting_bankroll = Column(Float)
    ending_bankroll = Column(Float, nullable=True)
    config_snapshot = Column(Text, nullable=True)


class TradeLogger:
    def __init__(self, config: dict):
        db_path = config["logging"]["db_path"]
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self._migrate_add_columns()
        self.Session = sessionmaker(bind=self.engine)

        # Cached P&L -- invalidated when a trade outcome is updated
        self._cached_pnl: float | None = None

        # Create a new bot session record
        self.session_id = str(uuid.uuid4())[:8]
        session = self.Session()
        try:
            bot_session = BotSession(
                session_id=self.session_id,
                mode="live",
                starting_bankroll=config["strategy"].get("initial_bankroll", 200.0),
                config_snapshot=json.dumps(config, indent=2),
            )
            session.add(bot_session)
            session.commit()
        finally:
            session.close()

        logger.info(f"Trade logger initialized with DB: {db_path} (session {self.session_id})")

    def _migrate_add_columns(self):
        """Add new columns to existing DBs without losing data."""
        with self.engine.connect() as conn:
            # Add session_id to trades if missing
            try:
                conn.execute(text("SELECT session_id FROM trades LIMIT 1"))
            except Exception:
                conn.execute(text("ALTER TABLE trades ADD COLUMN session_id TEXT"))
                conn.commit()
            # Add session_id to signals if missing
            try:
                conn.execute(text("SELECT session_id FROM signals LIMIT 1"))
            except Exception:
                conn.execute(text("ALTER TABLE signals ADD COLUMN session_id TEXT"))
                conn.commit()

    def end_session(self, ending_bankroll: float):
        """Mark the current bot session as ended."""
        session = self.Session()
        try:
            rec = session.query(BotSession).filter_by(session_id=self.session_id).first()
            if rec:
                rec.ended_at = datetime.now(timezone.utc)
                rec.ending_bankroll = ending_bankroll
                session.commit()
        finally:
            session.close()

    def log_trade(
        self,
        btc_price_start: float,
        btc_price_end: float,
        delta_pct: float,
        market_id: str,
        market_question: str,
        odds_yes: float,
        odds_no: float,
        direction: str,
        bet_size: float,
        fill_price: float,
        order_id: str,
        bankroll_before: float,
        edge: float,
        confidence: float,
    ) -> int:
        """Log a new trade (outcome pending)."""
        session = self.Session()
        try:
            record = TradeRecord(
                session_id=self.session_id,
                btc_price_start=btc_price_start,
                btc_price_end=btc_price_end,
                delta_pct=delta_pct,
                market_id=market_id,
                market_question=market_question,
                polymarket_odds_yes=odds_yes,
                polymarket_odds_no=odds_no,
                direction=direction,
                bet_size=bet_size,
                fill_price=fill_price,
                order_id=order_id,
                outcome="PENDING",
                profit_loss=0.0,
                bankroll_before=bankroll_before,
                bankroll_after=bankroll_before,
                consecutive_wins=0,
                edge_estimate=edge,
                confidence=confidence,
            )
            session.add(record)
            session.commit()
            trade_id = record.id
            logger.info(f"Trade logged: id={trade_id} {direction} ${bet_size} on {market_id}")
            return trade_id
        finally:
            session.close()

    def update_trade_outcome(
        self,
        trade_id: int,
        outcome: str,
        profit_loss: float,
        bankroll_after: float,
        consecutive_wins: int,
        btc_price_close: float | None = None,
    ):
        """Update a trade with its final outcome and close price."""
        session = self.Session()
        try:
            record = session.query(TradeRecord).filter_by(id=trade_id).first()
            if record:
                record.outcome = outcome
                record.profit_loss = profit_loss
                record.bankroll_after = bankroll_after
                record.consecutive_wins = consecutive_wins
                if btc_price_close is not None:
                    record.btc_price_end = btc_price_close
                session.commit()
                # Invalidate P&L cache -- next call will re-query
                self._cached_pnl = None
                logger.info(f"Trade {trade_id} updated: {outcome} P&L=${profit_loss:.2f}")
        finally:
            session.close()

    def log_signal(
        self,
        direction: str,
        delta_pct: float,
        confidence: float,
        btc_start: float,
        btc_end: float,
        traded: bool,
        reason: str,
    ):
        """Log an arbitrage signal (whether traded or not).

        Throttled: non-traded signals are only logged every 30s to reduce
        DB writes.  Traded signals are always logged immediately.
        """
        now = time.time()
        if not traded:
            last = getattr(self, '_last_signal_time', 0.0)
            if now - last < 30.0:
                return  # skip -- too soon since last non-trade signal
            self._last_signal_time = now

        session = self.Session()
        try:
            record = SignalRecord(
                session_id=self.session_id,
                direction=direction,
                delta_pct=delta_pct,
                confidence=confidence,
                btc_price_start=btc_start,
                btc_price_end=btc_end,
                traded=traded,
                reason=reason,
            )
            session.add(record)
            session.commit()
        finally:
            session.close()

    def get_orphaned_trades(self) -> list[dict]:
        """Find PENDING trades from windows that have already ended.

        These are trades whose 5-min window has passed but were never
        resolved -- typically because the bot crashed mid-session.
        Returns a list of dicts with trade details needed for resolution.
        """
        import time as _time
        session = self.Session()
        try:
            pending = (
                session.query(TradeRecord)
                .filter(TradeRecord.outcome == "PENDING")
                .all()
            )
            orphans = []
            now = _time.time()
            for t in pending:
                # Extract window timestamp from market_question or market_id
                # The slug is stored indirectly -- we can reconstruct from the
                # timestamp.  A trade is orphaned if it's > 10 min old.
                if t.timestamp is None:
                    continue
                # SQLite datetimes are naive -- make them UTC-aware for comparison
                ts = t.timestamp.replace(tzinfo=timezone.utc) if t.timestamp.tzinfo is None else t.timestamp
                trade_age = (datetime.now(timezone.utc) - ts).total_seconds()
                if trade_age > 600:  # > 10 minutes old = definitely orphaned
                    orphans.append({
                        "trade_id": t.id,
                        "direction": t.direction,
                        "bet_size": t.bet_size,
                        "fill_price": t.fill_price,
                        "btc_price_start": t.btc_price_start,
                        "btc_price_end": t.btc_price_end,
                        "market_question": t.market_question,
                        "bankroll_before": t.bankroll_before,
                        "timestamp": t.timestamp,
                        "order_id": t.order_id,
                    })
            return orphans
        finally:
            session.close()

    def get_bot_live_pnl(self) -> float:
        """Get the bot's net P&L from all resolved trades.

        This sums profit_loss for all resolved trades.  It does NOT
        depend on the exchange balance -- so manual trading by the user
        from the same account is completely ignored.

        Result is cached and invalidated when update_trade_outcome() is called.
        """
        if self._cached_pnl is not None:
            return self._cached_pnl

        session = self.Session()
        try:
            trades = (
                session.query(TradeRecord)
                .filter(
                    TradeRecord.outcome != "PENDING",
                )
                .all()
            )
            pnl = sum(t.profit_loss or 0.0 for t in trades)
            self._cached_pnl = pnl
            return pnl
        finally:
            session.close()

    def get_session_stats(self, session_id: str | None = None,
                          all_sessions: bool = False) -> dict:
        """Get summary statistics.

        Args:
            session_id: Specific session to query (defaults to current).
            all_sessions: If True, query across ALL sessions (lifetime).
        """
        session = self.Session()
        try:
            query = session.query(TradeRecord).filter(TradeRecord.outcome != "PENDING")
            if not all_sessions:
                sid = session_id or self.session_id
                if sid:
                    query = query.filter(TradeRecord.session_id == sid)
            trades = query.all()
            if not trades:
                return {"total_trades": 0}

            wins = [t for t in trades if t.outcome == "WIN"]
            losses = [t for t in trades if t.outcome == "LOSS"]
            total_pnl = sum(t.profit_loss for t in trades)

            return {
                "total_trades": len(trades),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": len(wins) / len(trades) if trades else 0,
                "total_pnl": total_pnl,
                "avg_win": sum(t.profit_loss for t in wins) / len(wins) if wins else 0,
                "avg_loss": sum(t.profit_loss for t in losses) / len(losses) if losses else 0,
                "largest_win": max((t.profit_loss for t in wins), default=0),
                "largest_loss": min((t.profit_loss for t in losses), default=0),
            }
        finally:
            session.close()

