"""
Main orchestrator for the Polymarket 5-min BTC arbitrage bot.
Coordinates price monitoring, market scanning, trade execution, and position management.
"""

import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.price_monitor import PriceMonitor
from src.polymarket_client import PolymarketClient
from src.arbitrage_engine import ArbitrageEngine
from src.position_manager import PositionManager
from src.trade_logger import TradeLogger

logger = logging.getLogger("polybot")

# Globals for signal handling
_shutdown = False


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def check_emergency_stop() -> bool:
    """Check for emergency stop file."""
    return Path("EMERGENCY_STOP").exists()


def check_trading_enabled() -> bool:
    return os.getenv("TRADING_ENABLED", "true").lower() == "true"


class Orchestrator:
    def __init__(self, config: dict, paper_trade: bool = True):
        self.config = config
        self.paper_trade = paper_trade

        # Initialize components
        self.price_monitor = PriceMonitor(config)
        self.arbitrage_engine = ArbitrageEngine(config)
        self.position_manager = PositionManager(config)
        self.trade_logger = TradeLogger(config)

        # Polymarket client (only needed for live trading)
        self.poly_client: PolymarketClient | None = None
        if not paper_trade:
            api_key = os.getenv("POLYMARKET_API_KEY", "")
            private_key = os.getenv("POLYMARKET_PRIVATE_KEY", "")
            if not api_key or not private_key:
                raise ValueError("POLYMARKET_API_KEY and POLYMARKET_PRIVATE_KEY required for live trading")
            self.poly_client = PolymarketClient(config, api_key, private_key)

        self._active_trade_id: int | None = None
        self._scan_interval = 5  # seconds between scans

    async def run(self):
        """Main event loop."""
        global _shutdown

        logger.info(f"Starting Polybot ({'PAPER' if self.paper_trade else 'LIVE'} mode)")
        logger.info(f"Bankroll: ${self.position_manager.bankroll:.2f}")
        logger.info(f"Initial bet: ${self.position_manager.initial_bet}")

        # Authenticate if live trading
        if self.poly_client:
            if not self.poly_client.authenticate():
                logger.error("Failed to authenticate with Polymarket, exiting")
                return

        # Start price monitor in background
        price_task = asyncio.create_task(self.price_monitor.start())

        # Wait for initial price data
        logger.info("Waiting for price data...")
        for _ in range(30):
            if self.price_monitor.current_price is not None:
                break
            await asyncio.sleep(1)

        if self.price_monitor.current_price is None:
            logger.error("Failed to get initial price data after 30s, exiting")
            await self.price_monitor.stop()
            return

        logger.info(f"Initial BTC price: ${self.price_monitor.current_price:,.2f}")
        logger.info("Waiting 5 minutes for price history to build...")

        # Wait for enough price history (5 minutes)
        await asyncio.sleep(300)

        logger.info("Price history ready, starting scan loop")

        # Main scan loop
        try:
            while not _shutdown:
                if check_emergency_stop():
                    logger.warning("EMERGENCY STOP detected, shutting down")
                    break

                if not check_trading_enabled():
                    logger.info("Trading disabled, waiting...")
                    await asyncio.sleep(10)
                    continue

                await self._scan_and_trade()
                await asyncio.sleep(self._scan_interval)

        except asyncio.CancelledError:
            logger.info("Orchestrator cancelled")
        finally:
            await self.price_monitor.stop()
            price_task.cancel()
            self._print_session_summary()

    async def _scan_and_trade(self):
        """Single scan cycle: check for signals and execute if warranted."""
        # Check if we can trade
        can_trade, reason = self.position_manager.can_trade()
        if not can_trade:
            logger.debug(f"Cannot trade: {reason}")
            return

        # Check for arbitrage signal from price movement
        signal = self.price_monitor.detect_arbitrage_signal()
        if signal is None:
            return

        logger.info(
            f"Signal detected: {signal.direction} delta={signal.delta_pct:.3f}% "
            f"confidence={signal.confidence:.2f}"
        )

        if self.paper_trade:
            await self._paper_trade(signal)
        else:
            await self._live_trade(signal)

    async def _paper_trade(self, signal):
        """Simulate a trade without executing on Polymarket."""
        # Simulate market odds (in paper mode, estimate from price movement)
        # In reality, you'd check actual Polymarket odds
        simulated_yes_price = 0.50 + (signal.delta_pct / 100) * 2
        simulated_yes_price = max(0.10, min(0.90, simulated_yes_price))
        simulated_no_price = 1.0 - simulated_yes_price

        from src.polymarket_client import MarketInfo

        fake_market = MarketInfo(
            condition_id="paper_trade",
            question=f"Will BTC go {signal.direction} in 5 min?",
            yes_token_id="",
            no_token_id="",
            yes_price=simulated_yes_price,
            no_price=simulated_no_price,
            end_date="",
            volume=0,
        )

        decision = self.arbitrage_engine.analyze_opportunity(
            signal, fake_market, simulated_yes_price, simulated_no_price
        )

        # Log the signal
        self.trade_logger.log_signal(
            direction=signal.direction,
            delta_pct=signal.delta_pct,
            confidence=signal.confidence,
            btc_start=signal.btc_price_start,
            btc_end=signal.btc_price_end,
            traded=decision.should_trade,
            reason=decision.reason,
        )

        if not decision.should_trade:
            logger.info(f"Signal rejected: {decision.reason}")
            return

        bet_size = self.position_manager.calculate_bet_size()
        bankroll_before = self.position_manager.bankroll

        trade_id = self.trade_logger.log_trade(
            btc_price_start=signal.btc_price_start,
            btc_price_end=signal.btc_price_end,
            delta_pct=signal.delta_pct,
            market_id="paper_trade",
            market_question=fake_market.question,
            odds_yes=simulated_yes_price,
            odds_no=simulated_no_price,
            direction=decision.direction,
            bet_size=bet_size,
            fill_price=decision.price,
            order_id="paper",
            bankroll_before=bankroll_before,
            edge=decision.edge,
            confidence=decision.confidence,
            paper_trade=True,
        )

        # Wait for market resolution (5 minutes in real life, simulated here)
        logger.info(f"Paper trade placed: {decision.direction} ${bet_size:.2f} | Waiting for resolution...")
        await asyncio.sleep(300)  # Wait 5 minutes for actual outcome

        # Check actual BTC price movement for paper trade outcome
        new_signal = self.price_monitor.detect_arbitrage_signal()
        if new_signal and new_signal.direction == signal.direction:
            # Price continued in our direction = win
            profit = bet_size * ((1.0 / decision.price) - 1)
            self.position_manager.update_after_win(profit)
            self.trade_logger.update_trade_outcome(
                trade_id, "WIN", profit, self.position_manager.bankroll,
                self.position_manager.consecutive_wins,
            )
        else:
            # Price reversed = loss
            self.position_manager.update_after_loss(bet_size)
            self.trade_logger.update_trade_outcome(
                trade_id, "LOSS", -bet_size, self.position_manager.bankroll,
                self.position_manager.consecutive_wins,
            )

    async def _live_trade(self, signal):
        """Execute a real trade on Polymarket."""
        if not self.poly_client:
            return

        # Find active 5-min markets
        markets = self.poly_client.get_active_5min_markets()
        if not markets:
            logger.info("No active 5-min BTC markets found")
            return

        # Pick the most relevant market
        market = markets[0]  # TODO: better market selection logic

        # Get live odds
        yes_price, no_price = self.poly_client.get_current_odds(market)

        # Analyze opportunity
        decision = self.arbitrage_engine.analyze_opportunity(
            signal, market, yes_price, no_price
        )

        # Log signal
        self.trade_logger.log_signal(
            direction=signal.direction,
            delta_pct=signal.delta_pct,
            confidence=signal.confidence,
            btc_start=signal.btc_price_start,
            btc_end=signal.btc_price_end,
            traded=decision.should_trade,
            reason=decision.reason,
        )

        if not decision.should_trade:
            logger.info(f"Signal rejected: {decision.reason}")
            return

        bet_size = self.position_manager.calculate_bet_size()
        bankroll_before = self.position_manager.bankroll

        # Place order
        result = self.poly_client.place_order(
            market, decision.direction, bet_size, decision.price
        )

        if not result.success:
            logger.error(f"Order failed: {result.error}")
            return

        trade_id = self.trade_logger.log_trade(
            btc_price_start=signal.btc_price_start,
            btc_price_end=signal.btc_price_end,
            delta_pct=signal.delta_pct,
            market_id=market.condition_id,
            market_question=market.question,
            odds_yes=yes_price,
            odds_no=no_price,
            direction=decision.direction,
            bet_size=bet_size,
            fill_price=result.fill_price or decision.price,
            order_id=result.order_id or "",
            bankroll_before=bankroll_before,
            edge=decision.edge,
            confidence=decision.confidence,
        )

        logger.info(f"Order filled: {decision.direction} ${bet_size:.2f} @ {decision.price}")

        # Wait for market resolution
        logger.info("Waiting for market resolution...")
        await asyncio.sleep(330)  # 5 min + 30s buffer

        # Check outcome (simplified - in production, poll market status)
        if result.order_id:
            status = self.poly_client.check_position_status(result.order_id)
            # Determine win/loss based on position value
            # This is simplified - real implementation would check market resolution
            if status.get("status") == "resolved":
                profit = bet_size * ((1.0 / decision.price) - 1)
                self.position_manager.update_after_win(profit)
                self.trade_logger.update_trade_outcome(
                    trade_id, "WIN", profit, self.position_manager.bankroll,
                    self.position_manager.consecutive_wins,
                )
            else:
                self.position_manager.update_after_loss(bet_size)
                self.trade_logger.update_trade_outcome(
                    trade_id, "LOSS", -bet_size, self.position_manager.bankroll,
                    self.position_manager.consecutive_wins,
                )

    def _print_session_summary(self):
        """Print end-of-session summary."""
        stats = self.trade_logger.get_session_stats()
        state = self.position_manager.get_state()

        logger.info("=" * 50)
        logger.info("SESSION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total trades: {stats.get('total_trades', 0)}")
        logger.info(f"Wins: {stats.get('wins', 0)} | Losses: {stats.get('losses', 0)}")
        logger.info(f"Win rate: {stats.get('win_rate', 0):.1%}")
        logger.info(f"Total P&L: ${stats.get('total_pnl', 0):.2f}")
        logger.info(f"Final bankroll: ${state.bankroll:.2f}")
        logger.info(f"Avg win: ${stats.get('avg_win', 0):.2f} | Avg loss: ${stats.get('avg_loss', 0):.2f}")
        logger.info("=" * 50)


def handle_shutdown(signum, frame):
    global _shutdown
    logger.info("Shutdown signal received")
    _shutdown = True


def main():
    load_dotenv()

    # Parse CLI args
    paper_mode = "--live" not in sys.argv
    config_path = "config.yaml"

    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]

    config = load_config(config_path)
    setup_logging(config["logging"]["log_level"])

    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    orchestrator = Orchestrator(config, paper_trade=paper_mode)

    logger.info("=" * 50)
    logger.info("POLYBOT - BTC 5-Min Arbitrage Bot")
    logger.info(f"Mode: {'PAPER TRADE' if paper_mode else 'LIVE TRADING'}")
    logger.info(f"Config: {config_path}")
    logger.info("=" * 50)
    logger.info("Use --live flag to enable live trading")
    logger.info("Create EMERGENCY_STOP file to halt")
    logger.info("Set TRADING_ENABLED=false in env to pause")
    logger.info("=" * 50)

    asyncio.run(orchestrator.run())


if __name__ == "__main__":
    main()
