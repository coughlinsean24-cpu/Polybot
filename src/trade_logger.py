"""
Trade logging and analytics using SQLite via SQLAlchemy.
Records every signal, decision, and outcome for post-analysis.
"""

import logging
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Boolean,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()


class TradeRecord(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

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

    # Meta
    paper_trade = Column(Boolean, default=False)


class SignalRecord(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    direction = Column(String)
    delta_pct = Column(Float)
    confidence = Column(Float)
    btc_price_start = Column(Float)
    btc_price_end = Column(Float)
    traded = Column(Boolean, default=False)
    reason = Column(String)  # why we did or didn't trade


class TradeLogger:
    def __init__(self, config: dict):
        db_path = config["logging"]["db_path"]
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        logger.info(f"Trade logger initialized with DB: {db_path}")

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
        paper_trade: bool = False,
    ) -> int:
        """Log a new trade (outcome pending)."""
        session = self.Session()
        try:
            record = TradeRecord(
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
                paper_trade=paper_trade,
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
    ):
        """Update a trade with its final outcome."""
        session = self.Session()
        try:
            record = session.query(TradeRecord).filter_by(id=trade_id).first()
            if record:
                record.outcome = outcome
                record.profit_loss = profit_loss
                record.bankroll_after = bankroll_after
                record.consecutive_wins = consecutive_wins
                session.commit()
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
        """Log an arbitrage signal (whether traded or not)."""
        session = self.Session()
        try:
            record = SignalRecord(
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

    def get_session_stats(self) -> dict:
        """Get summary statistics for the current session."""
        session = self.Session()
        try:
            trades = session.query(TradeRecord).filter(TradeRecord.outcome != "PENDING").all()
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
