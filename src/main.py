"""
Main orchestrator for the Polymarket 5-min BTC Up/Down bot.

Strategy:
  1. Stream real-time BTC price from Binance.
  2. Every scan, fetch the current 5-min market from Polymarket (gamma-api).
  3. Compare live BTC price against the market's "price to beat"
     (BTC price at the start of the 5-min window).
  4. If BTC is above the target -> bet UP, below -> bet DOWN.
  5. Only bet if the Polymarket odds offer value vs. our confidence.
  6. Wait for resolution, record result.
"""

import asyncio
import json
import logging
import math
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

import yaml
from dotenv import load_dotenv

from src.price_monitor import PriceMonitor
from src.polymarket_client import PolymarketClient, MarketInfo, set_ws_streams
from src.polymarket_ws import PolymarketMarketStream, PolymarketUserStream
from src.arbitrage_engine import ArbitrageEngine
from src.position_manager import PositionManager
from src.trade_logger import TradeLogger
from src.adaptive_learner import AdaptiveLearner
from src.technical_analysis import TechnicalAnalyzer

logger = logging.getLogger("polybot")

# Globals for signal handling
_shutdown = False


def setup_logging(level: str = "INFO"):
    # Log to both console and a rotating file
    root_logger = logging.getLogger()
    root_logger.handlers.clear()          # Remove any default / stale handlers
    root_logger.setLevel(getattr(logging, level))

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    # Rotating file handler -- keeps history across restarts
    from logging.handlers import RotatingFileHandler
    Path("data").mkdir(exist_ok=True)
    file_handler = RotatingFileHandler(
        "data/polybot.log",
        maxBytes=5 * 1024 * 1024,  # 5 MB per file
        backupCount=10,            # keep 10 old files (50 MB total)
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def check_emergency_stop() -> bool:
    """Check for emergency stop file."""
    return Path("EMERGENCY_STOP").exists()


def check_trading_enabled() -> bool:
    return os.getenv("TRADING_ENABLED", "true").lower() == "true"


# ── Single-instance guard (PID file) ──────────────────────────────
_PID_FILE = Path("data/polybot.pid")


def _acquire_pid_lock() -> bool:
    """Ensure only one Polybot process runs at a time.

    Writes our PID to data/polybot.pid.  If such a file already exists
    and the PID inside is still alive, refuse to start.
    Returns True if lock acquired, False if another instance is running.
    """
    Path("data").mkdir(exist_ok=True)
    if _PID_FILE.exists():
        try:
            old_pid = int(_PID_FILE.read_text().strip())
            if psutil.pid_exists(old_pid):
                try:
                    proc = psutil.Process(old_pid)
                    # Only block if the process is actually python running polybot
                    cmdline = " ".join(proc.cmdline()).lower()
                    if "polybot" in cmdline or "src.main" in cmdline:
                        return False  # another instance is alive
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass  # stale PID or can't inspect -- OK to proceed
        except (ValueError, OSError):
            pass  # corrupt PID file -- overwrite it
    # Write our PID
    _PID_FILE.write_text(str(os.getpid()))
    return True


def _release_pid_lock():
    """Remove PID file on clean shutdown."""
    try:
        if _PID_FILE.exists():
            stored = int(_PID_FILE.read_text().strip())
            if stored == os.getpid():
                _PID_FILE.unlink()
    except (ValueError, OSError):
        pass


class Orchestrator:
    def __init__(self, config: dict):
        self.config = config

        # Initialize components
        self.price_monitor = PriceMonitor(config)
        self.arbitrage_engine = ArbitrageEngine(config)
        self.position_manager = PositionManager(config)
        self.trade_logger = TradeLogger(config)
        self.learner = AdaptiveLearner(config)

        # Restore bankroll and stats from DB (survives restarts)
        self.position_manager.load_state_from_db(self.trade_logger)

        # -- Strategy config --
        strategy_cfg = config.get("strategy", {})
        self._min_fill_price = strategy_cfg.get("min_fill_price", 0.30)

        # -- Straddle strategy state --
        straddle_cfg = config.get("straddle", {})
        self._straddle_enabled = straddle_cfg.get("enabled", False)
        self._straddle_trigger_secs = straddle_cfg.get("trigger_seconds", 120)
        self._straddle_max_diff = straddle_cfg.get("max_diff_pct", 0.02)
        self._straddle_price = straddle_cfg.get("limit_price", 0.05)
        self._straddle_shares = straddle_cfg.get("shares", 500)
        self._straddle_max_cost = straddle_cfg.get("max_cost", 50.0)
        self._straddle_active: dict | None = None  # tracks active straddle
        self._straddle_window_slugs: set[str] = set()  # windows we already straddled
        self._prev_diff_pct: float | None = None  # for momentum tracking

        # Polymarket client
        api_key = os.getenv("POLYMARKET_API_KEY", "")
        api_secret = os.getenv("POLYMARKET_API_SECRET", "")
        passphrase = os.getenv("POLYMARKET_PASSPHRASE", "")
        private_key = os.getenv("POLYMARKET_PRIVATE_KEY", "")
        if not api_key or not api_secret or not passphrase:
            raise ValueError(
                "POLYMARKET_API_KEY, POLYMARKET_API_SECRET, and "
                "POLYMARKET_PASSPHRASE are required for trading"
            )
        if not private_key:
            logger.warning(
                "POLYMARKET_PRIVATE_KEY not set -- order signing will fail. "
                "Add your Polygon wallet private key to .env for live trades."
            )
        self.poly_client = PolymarketClient(
            config, api_key, api_secret, passphrase, private_key
        )

        self._scan_interval = 1  # seconds between scans (1s for minimum latency)

        # -- Scalp re-entry config --
        scalp_cfg = config.get("scalp", {})
        self._scalp_enabled = scalp_cfg.get("enabled", False)
        self._scalp_max_entries = scalp_cfg.get("max_entries_per_window", 3)
        self._scalp_cooldown_tp = scalp_cfg.get("cooldown_after_take_profit", 15)
        self._scalp_cooldown_sl = scalp_cfg.get("cooldown_after_stop_loss", 30)

        # -- Hedge lock-in config --
        hedge_cfg = config.get("hedge", {})
        self._hedge_enabled = hedge_cfg.get("enabled", False)
        self._hedge_min_prob = hedge_cfg.get("min_prob", 0.80)
        self._hedge_min_profit_pct = hedge_cfg.get("min_profit_pct", 0.10)
        self._hedge_max_cost = hedge_cfg.get("max_hedge_cost", 5.0)
        self._hedge_min_secs = hedge_cfg.get("min_seconds_left", 30)

        # -- Momentum pre-position config --
        prepos_cfg = config.get("preposition", {})
        self._prepos_enabled = prepos_cfg.get("enabled", False)
        self._prepos_trigger_secs = prepos_cfg.get("trigger_seconds", 30)
        self._prepos_min_momentum = prepos_cfg.get("min_momentum_pct", 0.04)
        self._prepos_min_accel = prepos_cfg.get("min_accel_pct", 0.01)
        self._prepos_max_price = prepos_cfg.get("max_price", 0.55)
        self._prepos_bet_fraction = prepos_cfg.get("bet_fraction", 0.5)
        self._prepos_pending: dict | None = None  # tracks pre-positioned order
        self._prepos_window_slugs: set[str] = set()  # next-window slugs we already pre-positioned

        # -- Position tracking for continuous trading --
        self._open_position: dict | None = None  # currently held position
        self._background_tasks: list[asyncio.Task] = []  # resolution tasks
        self._traded_window_slugs: set[str] = set()  # windows we already traded
        self._window_entry_counts: dict[str, int] = {}  # slug -> entries this window
        self._reentry_cooldown_until: float = 0.0  # timestamp when cooldown expires
        self._window_trade_directions: dict[str, list[str]] = {}  # slug -> ["UP","DOWN"] traded
        self._last_no_target_slug: str = ""  # suppress repeated "no openPrice" logs

        # -- WebSocket streams for real-time market data --
        self._market_stream = PolymarketMarketStream()
        self._user_stream: PolymarketUserStream | None = None
        self._ws_tasks: list[asyncio.Task] = []  # background WS tasks
        self._registered_ws_slugs: set[str] = set()  # slugs we already registered

        # -- Double It & Pass (parlay) mode --
        parlay_cfg = config.get("parlay", {})
        self._parlay_enabled = parlay_cfg.get("enabled", False)
        self._parlay_initial_stake = parlay_cfg.get("initial_stake", 1.0)
        self._parlay_max_rounds = parlay_cfg.get("max_rounds", 8)
        self._parlay_take_profit = parlay_cfg.get("take_profit", 100.0)
        self._parlay_active = False     # is a parlay streak in progress?
        self._parlay_stake = 0.0        # current bet for this round
        self._parlay_round = 0          # rounds completed in current streak
        self._parlay_total_won = 0.0    # cumulative profit in current streak
        self._parlay_history: list[dict] = []  # completed streaks
        self._parlay_lifetime_profit = 0.0     # total profit across all streaks
        self._parlay_lifetime_lost = 0.0       # total lost across all streaks
        self._parlay_state_file = Path("data/parlay_state.json")
        self._parlay_control_file = Path("data/parlay_control.json")
        self._load_parlay_state()  # restore from disk on restart

        # -- Dual-side limit-order config --
        dual_cfg = config.get("dual_side", {})
        self._dual_enabled = dual_cfg.get("enabled", False)
        self._dual_budget = dual_cfg.get("budget", 40.0)
        self._dual_max_price = dual_cfg.get("max_price", 0.47)
        self._dual_min_secs_to_place = dual_cfg.get("min_seconds_to_place", 240)
        self._dual_cancel_at = dual_cfg.get("cancel_unfilled_at", 10)
        self._dual_active: dict | None = None   # tracks active dual-side orders
        self._dual_window_slugs: set[str] = set()  # windows we already placed dual orders

        # -- Technical analysis --
        self.ta = TechnicalAnalyzer(config)

        # -- Hard safety guardrails (anti-rogue) --
        safety_cfg = config.get("safety", {})
        # Absolute maximum any single trade can cost (overrides Kelly, conviction, everything)
        self._hard_max_per_trade = safety_cfg.get("hard_max_per_trade", 20.0)
        # Maximum total spent in a single 5-min window (across all entries)
        self._max_per_window = safety_cfg.get("max_per_window", 30.0)
        # Rolling session loss limit: if we lose this much in one session, halt
        self._session_loss_limit = safety_cfg.get("session_loss_limit", 50.0)
        # Minimum TA quality score to allow a trade (0.0-1.0)
        self._min_ta_quality = safety_cfg.get("min_ta_quality", 0.30)
        # Track per-window spending
        self._window_spend: dict[str, float] = {}  # slug -> total $ spent
        # Track session P&L (resets on restart)
        self._session_trades_pnl: float = 0.0
        self._session_halted: bool = False

    async def run(self):
        """Main event loop -- fully non-blocking continuous trading."""
        global _shutdown

        logger.info("Starting Polybot (LIVE mode)")
        logger.info(f"Bankroll: ${self.position_manager.bankroll:.2f}")
        logger.info(f"Initial bet: ${self.position_manager.initial_bet}")

        # Authenticate if live trading
        if self.poly_client:
            self._last_balance_sync = 0.0  # track when we last synced
            if not self.poly_client.authenticate():
                logger.error("Failed to authenticate with Polymarket, exiting")
                return

        # Sync bankroll with real exchange balance (sets starting_bankroll + floor)
        if self.poly_client:
            self._sync_bankroll_with_exchange(is_startup=True)

        # Start WebSocket streams for real-time Polymarket data
        await self._start_ws_streams()

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
        logger.info("Waiting for price history to accumulate (30s)...")

        # Wait for enough history to know the window start price
        while not self.price_monitor.has_enough_history() and not _shutdown:
            await asyncio.sleep(1)

        # Resolve any orphaned PENDING trades from crashed sessions
        await self._resolve_orphaned_trades()

        logger.info("Price history ready -- starting continuous scan loop")

        # Main scan loop -- every cycle: monitor position -> look for new entry
        _consecutive_errors = 0
        try:
            while not _shutdown:
                if check_emergency_stop():
                    logger.warning("EMERGENCY STOP detected, shutting down")
                    break

                if not check_trading_enabled():
                    logger.info("Trading disabled, waiting...")
                    await asyncio.sleep(10)
                    continue

                # Session loss circuit breaker
                if self._session_halted:
                    logger.info(
                        f"[HALT] Session halted (lost ${abs(self._session_trades_pnl):.2f} "
                        f"this session, limit ${self._session_loss_limit:.2f}). "
                        f"Monitoring positions only."
                    )
                    # Still monitor existing positions, but don't enter new ones
                    await self._monitor_open_position()
                    await asyncio.sleep(30)
                    continue

                try:
                    # 1) Monitor open position (take-profit / window-end)
                    await self._monitor_open_position()

                    # 1b) Reversal exit: DISABLED
                    # Data shows reversal-exit causes massive losses by
                    # dumping positions on small BTC wobbles and booking
                    # full -$35 losses that often would have recovered.
                    # if self._open_position is not None:
                    #     await self._check_reversal_exit()

                    # 2) Monitor active straddle orders
                    if self._straddle_active:
                        await self._monitor_straddle()

                    # 2b) Monitor active dual-side orders
                    if self._dual_active:
                        await self._monitor_dual_side()

                    # 3) Monitor / promote pre-positioned orders
                    if self._prepos_pending:
                        await self._monitor_preposition()

                    # 4) Look for new entry if we have no open position
                    if self._open_position is None:
                        # Try dual-side entry first (if enabled)
                        if self._dual_enabled and self._dual_active is None:
                            await self._check_dual_side_entry()
                        await self._scan_for_entry()

                    # 5) Check for momentum pre-position on next window
                    if (
                        self._prepos_enabled
                        and self.poly_client
                        and self._prepos_pending is None
                    ):
                        await self._check_preposition()

                    # 6) Check for parlay control signals from dashboard
                    self._check_parlay_control()

                    # 7) Reap finished background tasks
                    self._background_tasks = [
                        t for t in self._background_tasks if not t.done()
                    ]

                    # 8) Periodic bankroll sync (every 60s) so bankroll
                    #    always matches real Polymarket cash balance
                    if time.time() - getattr(self, '_last_balance_sync', 0) > 60:
                        self._sync_bankroll_with_exchange()

                    _consecutive_errors = 0  # reset on success

                except Exception as e:
                    _consecutive_errors += 1
                    logger.error(
                        f"Scan cycle error ({_consecutive_errors} consecutive): {e}",
                        exc_info=(_consecutive_errors <= 3),
                    )
                    if _consecutive_errors >= 30:
                        logger.critical(
                            "30 consecutive scan errors -- shutting down"
                        )
                        break
                    # Back off proportionally to streak length
                    await asyncio.sleep(min(_consecutive_errors * 2, 30))

                await asyncio.sleep(self._scan_interval)

        except asyncio.CancelledError:
            logger.info("Orchestrator cancelled")
        finally:
            # Cancel any pending resolution tasks
            for task in self._background_tasks:
                task.cancel()
            # Stop WebSocket streams
            await self._stop_ws_streams()
            await self.price_monitor.stop()
            price_task.cancel()
            self.trade_logger.end_session(self.position_manager.bankroll)
            self._print_session_summary()

    # -- Orphaned trade resolution (runs once on startup) ----------

    async def _resolve_orphaned_trades(self):
        """Find and resolve PENDING trades from crashed sessions.

        On startup, any trade older than 10 min that is still PENDING
        had its resolution task die with the old bot process.  We look
        up the Chainlink close price for each orphan's window and
        resolve it now.
        """
        orphans = self.trade_logger.get_orphaned_trades()
        if not orphans:
            logger.info("No orphaned trades to resolve")
            return

        logger.info(f"Found {len(orphans)} orphaned PENDING trade(s) -- resolving...")

        for orph in orphans:
            tid = orph["trade_id"]
            direction = orph["direction"]
            bet_size = orph["bet_size"]
            fill_price = orph["fill_price"]
            target = orph["btc_price_start"]  # the Chainlink open price
            orph_order_id = orph.get("order_id", "")

            # First, check if the order actually filled on the exchange.
            # If size_matched == 0 AND fill_price is near-zero, the order
            # never filled -- mark CANCELLED.  But if fill_price is > 0.10
            # we already KNOW it filled (the bot recorded a real fill), so
            # skip the cancellation path -- the CLOB API may no longer
            # return old order data.
            if orph_order_id and self.poly_client and fill_price <= 0.10:
                try:
                    status = self.poly_client.check_position_status(orph_order_id)
                    matched = status.get("filled", 0)
                    order_status = status.get("status", "")
                    if matched == 0:
                        logger.info(
                            f"  Orphan #{tid} order never filled "
                            f"(status={order_status}) -- marking CANCELLED"
                        )
                        self.trade_logger.update_trade_outcome(
                            tid, "CANCELLED", 0.0,
                            self.position_manager.bankroll,
                            self.position_manager.consecutive_wins,
                        )
                        continue
                except Exception as e:
                    logger.warning(f"  Could not check order status for #{tid}: {e}")

            # Extract the window timestamp from the market_question
            # e.g. "Bitcoin Up or Down - February 15, 4:10PM-4:15PM ET"
            # We need the slug to fetch close price.  Reconstruct from
            # the trade's timestamp -- round down to the 5-min boundary.
            trade_ts = orph["timestamp"]
            if trade_ts:
                # SQLite datetimes are naive -- ensure UTC-aware
                if trade_ts.tzinfo is None:
                    trade_ts = trade_ts.replace(tzinfo=timezone.utc)
                epoch = int(trade_ts.timestamp())
                window_ts = (epoch // 300) * 300
                slug = f"btc-updown-5m-{window_ts}"
            else:
                logger.warning(f"Orphan trade #{tid} has no timestamp, marking LOSS")
                self.trade_logger.update_trade_outcome(
                    tid, "LOSS", -bet_size,
                    self.position_manager.bankroll - bet_size, 0,
                )
                self.position_manager.update_after_loss(bet_size)
                continue

            logger.info(f"Resolving orphan trade #{tid} ({direction} ${bet_size:.2f}) from {slug}")

            # Fetch the Chainlink close price
            close_price = PolymarketClient.fetch_close_price(slug)
            open_price = target or PolymarketClient.fetch_price_to_beat(slug)

            if close_price is not None and open_price is not None:
                win = (close_price > open_price) if direction == "UP" else (close_price < open_price)
                result_str = "UP wins" if close_price > open_price else "DOWN wins"
                logger.info(
                    f"  Chainlink: close=${close_price:,.2f} vs open=${open_price:,.2f} -> {result_str}"
                )
            else:
                # Last resort: use the Binance price that was captured at entry
                btc_end = orph.get("btc_price_end")
                if btc_end and open_price:
                    win = (btc_end > open_price) if direction == "UP" else (btc_end < open_price)
                    logger.warning(
                        f"  No Chainlink close for {slug}, using entry Binance ${btc_end:,.2f} -- unreliable"
                    )
                else:
                    win = False
                    logger.warning(f"  Cannot resolve orphan #{tid} -- marking as LOSS")

            if win:
                profit = bet_size * ((1.0 / fill_price) - 1)
                self.position_manager.update_after_win(profit)
                self.trade_logger.update_trade_outcome(
                    tid, "WIN", profit,
                    self.position_manager.bankroll,
                    self.position_manager.consecutive_wins,
                    btc_price_close=close_price,
                )
                logger.info(f"  [OK] Orphan #{tid} resolved: WIN +${profit:.2f}")
            else:
                self.position_manager.update_after_loss(bet_size)
                self.trade_logger.update_trade_outcome(
                    tid, "LOSS", -bet_size,
                    self.position_manager.bankroll,
                    self.position_manager.consecutive_wins,
                    btc_price_close=close_price,
                )
                logger.info(f"  [X] Orphan #{tid} resolved: LOSS -${bet_size:.2f}")

        logger.info(f"Orphan resolution complete. Bankroll: ${self.position_manager.bankroll:.2f}")

    # -- Shared trading helpers ---------------------------------------

    @staticmethod
    def _get_sell_price_for_direction(live: dict, direction: str) -> float:
        """Get the best sell price for our shares given direction.

        For UP shares: sell at best_bid.
        For DOWN shares: sell at 1 - best_ask (complement).
        """
        if direction == "UP":
            return live.get("best_bid", 0)
        return round(1.0 - live.get("best_ask", 1.0), 2) if live.get("best_ask") else 0

    async def _sell_and_poll(
        self, market, direction: str, shares: float, price: float,
        order_type: str = "GTC", max_attempts: int = 4,
        poll_delay: float = 1.0, label: str = "Sell",
    ) -> tuple[bool, str | None]:
        """Place a sell order and poll for fill, cancelling if unfilled.

        Returns (filled, order_id).
        """
        result = self.poly_client.sell_position(
            market, direction, shares, price, order_type=order_type,
        )
        if not result.success:
            logger.warning(f"{label} order failed: {result.error}")
            return False, None

        order_id = result.order_id or ""
        if not order_id:
            return False, None

        for attempt in range(max_attempts):
            await asyncio.sleep(poll_delay)
            try:
                status = self.poly_client.check_position_status(order_id)
                if status.get("filled", 0) > 0:
                    return True, order_id
                if status.get("status", "").upper() in (
                    "CANCELED", "CANCELLED", "CANCELED_MARKET_RESOLVED"
                ):
                    return False, order_id
            except Exception:
                pass

        # Not filled -- cancel to prevent stranded shares
        try:
            self.poly_client.cancel_order(order_id)
            logger.warning(f"{label} not filled after {max_attempts} polls -- cancelled")
        except Exception:
            pass
        return False, order_id

    async def _place_limit_sell_bg_task(
        self, market, direction: str, shares: float, sell_target: float,
        max_attempts: int = 15, initial_delay: float = 0.0,
        label: str = "LIMIT SELL",
    ):
        """Background task: place limit sell IMMEDIATELY, retry if not settled.

        Pure latency arb: we need the sell on the book ASAP so it fills
        the moment Polymarket catches up to the BTC move.
        """
        if initial_delay > 0:
            await asyncio.sleep(initial_delay)
        if self._open_position is None:
            return

        for attempt in range(max_attempts):
            if attempt > 0:
                wait = 2.0 + attempt * 1.0   # fast retries: 2s, 3s, 4s...
                await asyncio.sleep(wait)
            if self._open_position is None:
                return

            sell_result = self.poly_client.sell_position(
                market, direction, shares, sell_target,
            )
            if sell_result.success:
                if self._open_position is not None:
                    self._open_position["limit_sell_order_id"] = sell_result.order_id
                    self._open_position.pop("_limit_sell_pending", None)
                logger.info(
                    f"? {label}: {shares:.1f} shares @ {sell_target:.3f} "
                    f"-> {sell_result.order_id} (attempt {attempt+1})"
                )
                return

            err = sell_result.error or ""
            if "not enough balance" in err.lower():
                logger.info(
                    f"{label} attempt {attempt+1}/{max_attempts}: "
                    f"shares not settled yet -- retrying..."
                )
            else:
                logger.warning(
                    f"{label} attempt {attempt+1}/{max_attempts}: {err}"
                )

        logger.warning(
            f"{label} failed after {max_attempts} attempts -- "
            f"will use polling take-profit"
        )
        if self._open_position is not None:
            self._open_position.pop("_limit_sell_pending", None)

    def _record_trade_outcome(self, pos: dict, won: bool, pnl: float,
                               btc_now: float):
        """Record a trade outcome in position manager, trade logger, and learner."""
        if won:
            self.position_manager.update_after_win(pnl)
        else:
            self.position_manager.update_after_loss(abs(pnl))

        # Immediately sync bankroll with real Polymarket cash balance
        self._sync_bankroll_with_exchange()

        # Track session P&L for circuit breaker
        self._session_trades_pnl += pnl
        if self._session_trades_pnl < -self._session_loss_limit and not self._session_halted:
            self._session_halted = True
            logger.critical(
                f"[HALT] SESSION LOSS LIMIT reached: ${self._session_trades_pnl:.2f} "
                f"(limit -${self._session_loss_limit:.2f}). "
                f"No new trades until restart."
            )

        self.trade_logger.update_trade_outcome(
            pos["trade_id"],
            "WIN" if won else "LOSS",
            pnl,
            self.position_manager.bankroll,
            self.position_manager.consecutive_wins,
            btc_price_close=btc_now,
        )

        self.learner.record_outcome(
            won=won, pnl=pnl,
            diff_pct=pos.get("diff_pct", 0),
            edge=pos.get("edge", 0),
            seconds_left=pos.get("seconds_left", 0),
            prev_diff_pct=pos.get("prev_diff_pct"),
            ta_quality=pos.get("ta_quality"),
            mispricing_tier=pos.get("mispricing_tier"),
            velocity=pos.get("velocity"),
        )

        # -- Double It & Pass: update parlay state --
        if self._parlay_active:
            self._update_parlay(won, pnl)

    # ── Double It & Pass (parlay) helpers ──────────────────────────

    def _start_parlay(self):
        """Begin a new parlay streak from the initial stake."""
        self._parlay_active = True
        self._parlay_stake = self._parlay_initial_stake
        self._parlay_round = 0
        self._parlay_total_won = 0.0
        logger.info(
            f"[PARLAY] Started -- initial stake ${self._parlay_initial_stake:.2f}, "
            f"max {self._parlay_max_rounds} rounds, "
            f"auto-pass at ${self._parlay_take_profit:.2f}"
        )
        self._save_parlay_state()

    def _update_parlay(self, won: bool, pnl: float):
        """Called after each trade outcome while parlay is active."""
        if won:
            self._parlay_round += 1
            profit = abs(pnl)
            self._parlay_total_won += profit
            # Next stake = double of current
            self._parlay_stake = self._parlay_stake * 2
            logger.info(
                f"[PARLAY] WIN round {self._parlay_round}! "
                f"Profit this streak: ${self._parlay_total_won:.2f} -- "
                f"next bet: ${self._parlay_stake:.2f}"
            )
            # Auto-pass conditions
            if self._parlay_round >= self._parlay_max_rounds:
                logger.info(
                    f"[PARLAY] Max rounds ({self._parlay_max_rounds}) reached "
                    f"-- auto-passing with ${self._parlay_total_won:.2f} profit!"
                )
                self._pass_parlay("max_rounds")
            elif self._parlay_total_won >= self._parlay_take_profit:
                logger.info(
                    f"[PARLAY] Take-profit ${self._parlay_take_profit:.2f} hit "
                    f"-- auto-passing with ${self._parlay_total_won:.2f} profit!"
                )
                self._pass_parlay("take_profit")
        else:
            lost = self._parlay_initial_stake
            logger.info(
                f"[PARLAY] LOSS on round {self._parlay_round + 1} -- "
                f"streak ended, lost ${lost:.2f}"
            )
            self._parlay_history.append({
                "rounds": self._parlay_round,
                "result": "LOSS",
                "profit": -lost,
                "ended_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            self._parlay_lifetime_lost += lost
            # Auto-restart the parlay
            self._parlay_stake = self._parlay_initial_stake
            self._parlay_round = 0
            self._parlay_total_won = 0.0
            logger.info("[PARLAY] Restarting streak from $%.2f", self._parlay_initial_stake)
        self._save_parlay_state()

    def _pass_parlay(self, reason: str = "manual"):
        """Cash out the current parlay streak."""
        profit = self._parlay_total_won
        self._parlay_history.append({
            "rounds": self._parlay_round,
            "result": "PASS",
            "profit": profit,
            "reason": reason,
            "ended_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        self._parlay_lifetime_profit += profit
        logger.info(
            f"[PARLAY] PASSED! Locked ${profit:.2f} profit after "
            f"{self._parlay_round} round(s). "
            f"Lifetime: +${self._parlay_lifetime_profit:.2f} / "
            f"-${self._parlay_lifetime_lost:.2f}"
        )
        # Reset for next streak
        self._parlay_stake = self._parlay_initial_stake
        self._parlay_round = 0
        self._parlay_total_won = 0.0
        self._save_parlay_state()

    def _stop_parlay(self):
        """Deactivate parlay mode entirely."""
        if self._parlay_active and self._parlay_round > 0:
            self._pass_parlay("stopped")
        self._parlay_active = False
        logger.info("[PARLAY] Mode deactivated")
        self._save_parlay_state()

    def get_parlay_state(self) -> dict:
        """Return current parlay state for the dashboard."""
        return {
            "enabled": self._parlay_enabled,
            "active": self._parlay_active,
            "current_stake": self._parlay_stake,
            "round": self._parlay_round,
            "streak_profit": self._parlay_total_won,
            "initial_stake": self._parlay_initial_stake,
            "max_rounds": self._parlay_max_rounds,
            "take_profit": self._parlay_take_profit,
            "potential_next_win": self._parlay_stake,  # what's at risk
            "lifetime_profit": self._parlay_lifetime_profit,
            "lifetime_lost": self._parlay_lifetime_lost,
            "lifetime_net": self._parlay_lifetime_profit - self._parlay_lifetime_lost,
            "history": self._parlay_history[-10:],  # last 10 streaks
        }

    def _save_parlay_state(self):
        """Persist parlay state to JSON file for the dashboard to read."""
        try:
            self._parlay_state_file.parent.mkdir(parents=True, exist_ok=True)
            self._parlay_state_file.write_text(
                json.dumps(self.get_parlay_state(), indent=2)
            )
        except Exception as e:
            logger.warning(f"[PARLAY] Failed to save state: {e}")

    def _load_parlay_state(self):
        """Restore parlay state from JSON file (survives restarts)."""
        if not self._parlay_state_file.exists():
            return
        try:
            state = json.loads(self._parlay_state_file.read_text())
            self._parlay_active = state.get("active", False)
            self._parlay_stake = state.get("current_stake", self._parlay_initial_stake)
            self._parlay_round = state.get("round", 0)
            self._parlay_total_won = state.get("streak_profit", 0.0)
            self._parlay_lifetime_profit = state.get("lifetime_profit", 0.0)
            self._parlay_lifetime_lost = state.get("lifetime_lost", 0.0)
            self._parlay_history = state.get("history", [])
            if self._parlay_active:
                logger.info(
                    f"[PARLAY] Restored active streak: round {self._parlay_round}, "
                    f"stake ${self._parlay_stake:.2f}, "
                    f"profit ${self._parlay_total_won:.2f}"
                )
        except Exception as e:
            logger.warning(f"[PARLAY] Failed to load state: {e}")

    def _check_parlay_control(self):
        """Check for control signals from the dashboard (start/pass/stop)."""
        if not self._parlay_control_file.exists():
            return
        try:
            ctrl = json.loads(self._parlay_control_file.read_text())
            self._parlay_control_file.unlink()  # consume the signal
            action = ctrl.get("action", "")
            if action == "start":
                if not self._parlay_active:
                    self._start_parlay()
                    self._save_parlay_state()
                else:
                    logger.info("[PARLAY] Already active, ignoring start signal")
            elif action == "pass":
                if self._parlay_active and self._parlay_round > 0:
                    self._pass_parlay("manual")
                    self._save_parlay_state()
                else:
                    logger.info("[PARLAY] Nothing to pass")
            elif action == "stop":
                self._stop_parlay()
                self._save_parlay_state()
        except Exception as e:
            logger.warning(f"[PARLAY] Control file error: {e}")

    def _check_reentry(self, pos: dict, secs_left: float,
                        min_secs: int = 30,
                        cooldown_key: str = "_scalp_cooldown_tp") -> bool:
        """Check if scalp re-entry is possible and set cooldown if so."""
        entries = self._window_entry_counts.get(pos['slug'], 1)
        can_reenter = (
            self._scalp_enabled
            and entries < self._scalp_max_entries
            and secs_left > min_secs
        )
        if can_reenter:
            cooldown = getattr(self, cooldown_key, 10)
            self._reentry_cooldown_until = time.time() + cooldown
        return can_reenter

    # -- Position monitoring (called every scan cycle) --------------

    async def _monitor_open_position(self):
        """Check the open position for take-profit, stop-loss, or window expiry.

        Non-blocking dispatcher: runs once per scan, delegates to focused
        sub-methods for each concern.
        """
        pos = self._open_position
        if pos is None:
            return

        btc_now = self.price_monitor.current_price
        if btc_now is None:
            return

        secs_left = self.price_monitor.seconds_left_in_window()
        target_price = pos["target_price"]
        if not target_price or target_price == 0:
            # Pre-positioned trade filled before target was available -- try to fetch it now
            slug = pos.get("slug", "")
            if slug:
                fetched = PolymarketClient.fetch_price_to_beat(slug)
                if fetched and fetched > 0:
                    pos["target_price"] = fetched
                    target_price = fetched
                    logger.info(f"Recovered target price for {slug}: ${target_price:,.2f}")
                else:
                    # Use current BTC as fallback target if we've been stuck > 30s
                    if pos.get("_target_retry_count", 0) > 10:
                        pos["target_price"] = btc_now
                        target_price = btc_now
                        logger.warning(
                            f"Using current BTC ${btc_now:,.2f} as fallback target for {slug}"
                        )
                    else:
                        pos["_target_retry_count"] = pos.get("_target_retry_count", 0) + 1
                        logger.warning(
                            f"Target price still unavailable for {slug} "
                            f"(retry {pos['_target_retry_count']}/10)"
                        )
                        return
            else:
                logger.warning("Target price is zero/missing and no slug -- skipping monitor cycle")
                return
        diff_now = ((btc_now - target_price) / target_price) * 100
        direction = pos["direction"]

        # diff_favoring_us: positive = BTC moved in our direction, negative = against us
        diff_favoring_us = diff_now if direction == "UP" else -diff_now
        if diff_favoring_us >= 0:
            # BTC is moving in our direction -- probability model works normally
            our_prob = self.arbitrage_engine._estimate_probability(
                diff_favoring_us, secs_left,
            )
        else:
            # BTC is moving AGAINST our direction -- mirror the probability
            # e.g. if moving against us by 0.5%, prob of OTHER side = 0.96,
            # so OUR prob = 1 - 0.96 = 0.04  (should trigger stop-loss)
            our_prob = 1.0 - self.arbitrage_engine._estimate_probability(
                abs(diff_favoring_us), secs_left,
            )

        logger.info(
            f"? Position: {direction} | "
            f"BTC=${btc_now:,.2f} vs target=${target_price:,.2f} "
            f"({diff_now:+.3f}%) | prob={our_prob:.3f} | {secs_left:.0f}s left"
        )

        limit_sell_id = pos.get("limit_sell_order_id")

        # 1. Check if limit sell filled
        if await self._check_limit_sell_fill(pos, btc_now, secs_left, limit_sell_id):
            return

        # 2. Hedge lock-in check
        if (
            self._hedge_enabled
            and not pos.get("hedge_placed")
            and self.poly_client
            and our_prob >= self._hedge_min_prob
            and secs_left >= self._hedge_min_secs
        ):
            await self._execute_hedge_lockin(pos, our_prob, secs_left)

        stop_loss_threshold = self.config["strategy"].get("stop_loss", 0.30)
        take_profit_threshold = self.config["strategy"].get("take_profit", 0.85)

        # Confident scalp: lower the take-profit bar for faster exit
        if pos.get("confident_scalp"):
            take_profit_threshold = min(take_profit_threshold, 0.65)

        # Mispricing trades: even more aggressive take-profit trigger
        # We want to lock in profit FAST since Poly will catch up soon.
        misp_tier = pos.get("mispricing_tier", "NONE")
        if misp_tier == "EXTREME":
            take_profit_threshold = min(take_profit_threshold, 0.52)
        elif misp_tier == "HUGE":
            take_profit_threshold = min(take_profit_threshold, 0.55)
        elif misp_tier == "BIG":
            take_profit_threshold = min(take_profit_threshold, 0.60)
        elif misp_tier == "GOOD":
            take_profit_threshold = min(take_profit_threshold, 0.65)

        # -- BTC-diff safety stop: if BTC moved > 0.05% AGAINST us, exit fast --
        # With 2% TP, we can't afford to hold a losing position.
        # diff_favoring_us < 0 means BTC moved the wrong way.
        diff_sl_pct = 0.05  # cut if BTC has moved 0.05% against us
        if diff_favoring_us < -diff_sl_pct and secs_left > 5:
            logger.info(
                f"[STOP] BTC-DIFF STOP: BTC moved {abs(diff_favoring_us):.3f}% "
                f"against {direction} (limit {diff_sl_pct}%) -- forcing stop-loss"
            )
            await self._execute_stop_loss(
                pos, our_prob, secs_left, btc_now, diff_now,
                stop_loss_threshold, direction, limit_sell_id,
            )
            return

        # 3. Stop-loss (probability-based)
        if our_prob <= stop_loss_threshold:
            await self._execute_stop_loss(
                pos, our_prob, secs_left, btc_now, diff_now,
                stop_loss_threshold, direction, limit_sell_id,
            )
            return

        # 4. Take-profit status logging & trigger
        if pos.get("hedge_placed"):
            self._log_hedge_status(pos, secs_left, limit_sell_id)
        elif limit_sell_id:
            self._log_limit_sell_status(pos, our_prob, secs_left)
        elif pos.get("_limit_sell_pending"):
            if secs_left > 10:
                logger.info(
                    f"? Settlement pending -- waiting for limit sell placement | "
                    f"prob={our_prob:.3f} | {secs_left:.0f}s left"
                )
        elif our_prob >= take_profit_threshold and secs_left > 10:
            await self._execute_take_profit(pos, our_prob, secs_left, btc_now)
            return

        # 5. Smart pre-expiry exit
        if secs_left <= 30 and secs_left > 3 and self.poly_client:
            if await self._attempt_smart_exit(
                pos, btc_now, secs_left, direction, limit_sell_id,
            ):
                return

        # 6. Window end -- background resolution
        if secs_left <= 3:
            self._handle_window_end(pos, btc_now, limit_sell_id)

    # -- Reversal exit: close losing position when BTC flips direction ------

    async def _check_reversal_exit(self):
        """If BTC reversed significantly against our open position, exit early.

        This frees `_open_position` so `_scan_for_entry()` can fire on the
        *same* loop cycle and trade the opposite direction.  Without this,
        the bot sits in a losing position and never sees the reversal edge.

        Criteria (all must be true):
          - We hold a position
          - BTC moved >= 0.08 % in the OPPOSITE direction from our trade
          - Position is underwater (current sell price < entry price)
          - At least 45 seconds remain (enough time to trade the new direction)
        """
        pos = self._open_position
        if pos is None:
            return

        btc_now = self.price_monitor.current_price
        if btc_now is None:
            return

        target_price = pos.get("target_price")
        if not target_price or target_price <= 0:
            return

        secs_left = self.price_monitor.seconds_left_in_window()
        if secs_left < 45:
            return  # not enough time to enter + profit on opposite side

        direction = pos["direction"]
        diff_now = ((btc_now - target_price) / target_price) * 100

        # Is BTC going the WRONG way for our position?
        if direction == "DOWN" and diff_now < 0.12:
            return  # BTC is not above target enough to justify flipping
        if direction == "UP" and diff_now > -0.12:
            return  # BTC is not below target enough to justify flipping

        # Check that the position is actually underwater
        slug = pos.get("slug", "")
        entry_price = pos["entry_price"]
        bet_size = pos["bet_size"]
        shares = bet_size / entry_price
        limit_sell_id = pos.get("limit_sell_order_id")

        # Fetch live sell price
        try:
            live = PolymarketClient.get_live_market_price(slug)
            if not live:
                return
            sell_price = self._get_sell_price_for_direction(live, direction)
        except Exception:
            return

        if sell_price and sell_price >= entry_price:
            return  # position is in profit -- let normal take-profit handle it

        # --- All criteria met: execute reversal exit ---
        opp_dir = "UP" if direction == "DOWN" else "DOWN"
        logger.info(
            f"[REVERSAL-EXIT] BTC {diff_now:+.3f}% vs target -- flipping! "
            f"Closing {direction} position (sell@{sell_price:.3f} < entry@{entry_price:.3f}) "
            f"to free capital for {opp_dir} trade | {secs_left:.0f}s left"
        )

        # Cancel any pending limit sell
        if limit_sell_id and self.poly_client:
            try:
                self.poly_client.cancel_order(limit_sell_id)
                logger.info(f"Cancelled limit sell {limit_sell_id} (reversal exit)")
            except Exception:
                pass

        # Try to sell at market to recover partial value
        recovered = 0.0
        if self.poly_client and sell_price and sell_price > 0.01:
            try:
                market = PolymarketClient._fetch_market_by_slug(slug)
                if market:
                    aggressive_price = round(max(sell_price - 0.02, 0.01), 2)
                    logger.info(
                        f"[REVERSAL-EXIT] Selling {shares:.1f} shares @ "
                        f"{aggressive_price:.3f} (bid={sell_price:.3f})"
                    )
                    filled, _ = await self._sell_and_poll(
                        market, direction, shares, aggressive_price,
                        max_attempts=3, label="Reversal exit sell",
                    )
                    if filled:
                        recovered = shares * aggressive_price
                        logger.info(
                            f"[OK] Reversal exit filled -- recovered "
                            f"${recovered:.2f} of ${bet_size:.2f}"
                        )
            except Exception as e:
                logger.warning(f"Reversal exit sell error: {e}")

        loss = bet_size - recovered
        logger.info(
            f"[REVERSAL-EXIT] Closed {direction} position | "
            f"Loss -${loss:.2f}"
            + (f" (recovered ${recovered:.2f})" if recovered > 0 else " (hold to resolution)")
            + f" | Freeing scanner for {opp_dir}"
        )

        self._record_trade_outcome(pos, won=False, pnl=-loss, btc_now=btc_now)
        self._open_position = None
        # Do NOT set cooldown -- we WANT the scanner to fire immediately
        # for the opposite direction (reversal gate in _scan_for_entry
        # will handle the re-entry logic).

    # -- Monitor sub-methods (extracted from _monitor_open_position) --

    async def _check_limit_sell_fill(self, pos: dict, btc_now: float,
                                      secs_left: float,
                                      limit_sell_id: str | None) -> bool:
        """Check if a resting limit sell order has filled.

        Returns True if the position was closed (caller should return).
        """
        if not limit_sell_id or not self.poly_client:
            return False

        try:
            sell_status = self.poly_client.check_position_status(limit_sell_id)
            sell_filled = sell_status.get("filled", 0)
            sell_order_status = sell_status.get("status", "")

            if sell_filled > 0:
                entry_price = pos["entry_price"]
                bet_size = pos["bet_size"]
                shares = bet_size / entry_price
                limit_sell_pct = self.config["strategy"].get("limit_sell_profit", 0.10)
                sell_price = round(entry_price * (1.0 + limit_sell_pct), 2)
                profit = (shares * sell_price) - bet_size

                self._record_trade_outcome(pos, won=True, pnl=profit, btc_now=btc_now)

                # -- Lag burst: if the sell filled quickly and the market is
                # still at 50/50 (stale), skip cooldown for instant re-entry.
                # This lets us trade repeatedly throughout the entire 30s lag.
                lag_burst = False
                entry_ts = pos.get("entry_ts", 0)
                fill_time = time.time() - entry_ts if entry_ts else 999
                if fill_time < 20 and secs_left > 30:
                    try:
                        slug = pos.get("slug", "")
                        live = PolymarketClient.get_live_market_price(slug)
                        if live:
                            up_price = live.get("best_ask", 0.5)
                            # Market is "stale" if UP price is still near 50/50
                            if 0.48 <= up_price <= 0.57:
                                lag_burst = True
                                self._reentry_cooldown_until = 0  # no cooldown
                                logger.info(
                                    f"[LAG-BURST] Limit sell filled in {fill_time:.0f}s "
                                    f"and market still stale (Up={up_price:.3f}) -- "
                                    f"instant re-entry enabled! {secs_left:.0f}s left"
                                )
                    except Exception:
                        pass

                can_reenter = self._check_reentry(pos, secs_left) if not lag_burst else True
                cooldown_tag = "0s lag-burst" if lag_burst else f"{self._scalp_cooldown_tp}s cooldown"
                logger.info(
                    f"[OK] LIMIT SELL FILLED +${profit:.2f} [{pos['slug']}] | "
                    f"Sold {shares:.1f} shares @ {sell_price:.3f} "
                    f"(entry {entry_price:.3f}) | "
                    f"Bankroll: ${self.position_manager.bankroll:.2f}"
                    + (f" | ? Re-entry eligible ({cooldown_tag})" if can_reenter or lag_burst else "")
                )

                self._open_position = None
                return True

            if sell_order_status.upper() in ("CANCELED", "CANCELLED",
                                              "CANCELED_MARKET_RESOLVED"):
                pos["limit_sell_order_id"] = None
        except Exception as e:
            logger.warning(f"Limit sell check error: {e}")

        return False

    async def _execute_stop_loss(self, pos: dict, our_prob: float,
                                  secs_left: float, btc_now: float,
                                  diff_now: float, threshold: float,
                                  direction: str, limit_sell_id: str | None):
        """Execute stop-loss: cancel limit sell, try emergency sell, record loss."""
        # Cancel any pending limit sell
        if limit_sell_id and self.poly_client:
            try:
                self.poly_client.cancel_order(limit_sell_id)
                logger.info(f"Cancelled limit sell {limit_sell_id} (stop-loss)")
            except Exception:
                pass

        entry_price = pos["entry_price"]
        bet_size = pos["bet_size"]
        shares = bet_size / entry_price

        # Try to sell shares on CLOB to recover partial value
        recovered = 0.0
        if self.poly_client:
            slug = pos["slug"]
            try:
                live = PolymarketClient.get_live_market_price(slug)
                if live:
                    sell_price = self._get_sell_price_for_direction(live, direction)
                    if sell_price and sell_price > 0.01:
                        market = PolymarketClient._fetch_market_by_slug(slug)
                        if market:
                            aggressive_price = round(max(sell_price - 0.02, 0.01), 2)
                            logger.info(
                                f"[STOP] STOP-LOSS SELL: selling {shares:.1f} shares "
                                f"@ {aggressive_price:.3f} (bid={sell_price:.3f}, "
                                f"entry {entry_price:.3f})"
                            )
                            filled, _ = await self._sell_and_poll(
                                market, direction, shares, aggressive_price,
                                max_attempts=4, label="Stop-loss sell",
                            )
                            if filled:
                                recovered = shares * aggressive_price
                                logger.info(
                                    f"[OK] Stop-loss sell filled -- recovered "
                                    f"${recovered:.2f} of ${bet_size:.2f}"
                                )
            except Exception as e:
                logger.warning(f"Stop-loss sell error: {e}")

        loss = bet_size - recovered
        loss_pct = (loss / bet_size) * 100

        logger.info(
            f"[STOP] STOP-LOSS: prob={our_prob:.3f} "
            f"(<={threshold:.0%}) | "
            f"BTC ${btc_now:,.2f} ({diff_now:+.3f}%) | "
            f"Loss -${loss:.2f} ({loss_pct:.0f}% of bet)"
            + (f" | Recovered ${recovered:.2f} from sell" if recovered > 0 else " | No recovery (hold to resolution)")
            + f" | {secs_left:.0f}s left"
        )

        self._record_trade_outcome(pos, won=False, pnl=-loss, btc_now=btc_now)

        can_reenter = self._check_reentry(
            pos, secs_left, min_secs=45,
            cooldown_key="_scalp_cooldown_sl",
        )
        logger.info(
            f"[X] LOSS (stop-loss) "
            f"-${loss:.2f} | Bankroll: ${self.position_manager.bankroll:.2f}"
            + (f" | ? Re-entry eligible ({self._scalp_cooldown_sl}s cooldown)" if can_reenter else "")
        )

        self._open_position = None

    async def _attempt_smart_exit(self, pos: dict, btc_now: float,
                                    secs_left: float, direction: str,
                                    limit_sell_id: str | None) -> bool:
        """Smart pre-expiry exit: trust CLOB price over our model.

        If the market bid is below entry or a resting limit sell hasn't
        filled close to expiry, sell aggressively to recover value.
        Returns True if position was closed.
        """
        if pos.get("_expiry_sell_attempted"):
            return False

        slug = pos["slug"]
        entry_price = pos["entry_price"]
        bet_size = pos["bet_size"]
        shares = bet_size / entry_price

        try:
            live = PolymarketClient.get_live_market_price(slug)
            if not live:
                return False

            market_bid = self._get_sell_price_for_direction(live, direction)

            market_losing = market_bid and market_bid < entry_price
            near_expiry_unsold = limit_sell_id and secs_left <= 20

            if not ((market_losing or near_expiry_unsold) and market_bid and market_bid > 0.01):
                return False

            pos["_expiry_sell_attempted"] = True

            # Cancel limit sell first
            if limit_sell_id:
                try:
                    self.poly_client.cancel_order(limit_sell_id)
                    pos["limit_sell_order_id"] = None
                except Exception:
                    pass

            market = PolymarketClient._fetch_market_by_slug(slug)
            if not market:
                return False

            reason = "market losing" if market_losing else "unsold near expiry"
            aggressive_price = round(max(market_bid - 0.02, 0.01), 2)
            logger.info(
                f"[TIME][STOP] SMART EXIT ({reason}): "
                f"bid={market_bid:.3f} vs entry={entry_price:.3f} | "
                f"selling {shares:.1f} shares @ {aggressive_price:.3f} "
                f"({secs_left:.0f}s left)"
            )

            filled = False
            sell_order_id = None
            # Retry the sell up to 3 times with settlement waits
            for sell_attempt in range(3):
                filled, sell_order_id = await self._sell_and_poll(
                    market, direction, shares, aggressive_price,
                    max_attempts=4, label=f"Smart exit (attempt {sell_attempt+1})",
                )
                if filled:
                    break
                # Wait for settlement before retrying
                if secs_left > 8:
                    await asyncio.sleep(5)
                    secs_left -= 5
                else:
                    break

            if not filled:
                return False  # fall through to window-end resolution

            recovered = shares * aggressive_price
            if recovered >= bet_size:
                profit = recovered - bet_size
                self._record_trade_outcome(pos, won=True, pnl=profit, btc_now=btc_now)
                logger.info(
                    f"[OK] SMART EXIT WIN +${profit:.2f} | "
                    f"Sold @ {aggressive_price:.3f} (entry {entry_price:.3f}) | "
                    f"Bankroll: ${self.position_manager.bankroll:.2f}"
                )
            else:
                loss = bet_size - recovered
                self._record_trade_outcome(pos, won=False, pnl=-loss, btc_now=btc_now)
                logger.info(
                    f"[X] SMART EXIT LOSS -${loss:.2f} "
                    f"(recovered ${recovered:.2f} of ${bet_size:.2f}) | "
                    f"Bankroll: ${self.position_manager.bankroll:.2f}"
                )

            self._open_position = None
            return True

        except Exception as e:
            logger.warning(f"Smart exit error: {e}")
            return False

    def _log_hedge_status(self, pos: dict, secs_left: float,
                           limit_sell_id: str | None):
        """Log hedge status and cancel limit sell if hedged."""
        hedge_profit = pos.get("hedge_shares", 0) * (
            1.0 - pos["entry_price"] - pos.get("hedge_price", 0)
        )
        if secs_left > 10 and int(secs_left) % 15 < 4:
            logger.info(
                f"[LOCK] HEDGED: {pos['direction']}@{pos['entry_price']:.3f} + "
                f"{pos.get('hedge_side','')}@{pos.get('hedge_price',0):.3f} | "
                f"Locked profit ${hedge_profit:.2f} | {secs_left:.0f}s left"
            )
        if limit_sell_id and self.poly_client:
            try:
                self.poly_client.cancel_order(limit_sell_id)
                pos["limit_sell_order_id"] = None
                logger.info("Cancelled limit sell (hedge active, no longer needed)")
            except Exception:
                pass

    def _log_limit_sell_status(self, pos: dict, our_prob: float,
                                secs_left: float):
        """Log that a limit sell order is resting on the book."""
        limit_sell_pct = self.config["strategy"].get("limit_sell_profit", 0.10)
        sell_target = round(pos["entry_price"] * (1.0 + limit_sell_pct), 2)
        if secs_left > 10:
            logger.info(
                f"? Limit sell resting @ {sell_target:.3f} "
                f"(entry {pos['entry_price']:.3f} + {limit_sell_pct:.0%}) | "
                f"prob={our_prob:.3f} | {secs_left:.0f}s left"
            )

    def _handle_window_end(self, pos: dict, btc_now: float,
                            limit_sell_id: str | None):
        """Handle window expiry: cancel limit sell, send to background resolution."""
        if limit_sell_id and self.poly_client:
            try:
                self.poly_client.cancel_order(limit_sell_id)
                logger.info(f"Cancelled limit sell {limit_sell_id} (window end)")
            except Exception:
                pass

        pos["btc_at_close"] = btc_now
        logger.info(
            f"[TIME] Window ended for {pos['slug']} -- sending to background resolution "
            f"(BTC at close: ${btc_now:,.2f})"
        )
        self._open_position = None
        task = asyncio.create_task(self._resolve_trade_background(pos))
        self._background_tasks.append(task)

    # -- Scan for new entry (called every scan cycle) -------------

    async def _scan_for_entry(self):
        """Look for a new trade opportunity. Non-blocking."""
        # Skip if there's a pre-positioned order pending for this window
        # (it will be promoted by _monitor_preposition when filled)
        if self._prepos_pending:
            window_start_ts = (int(time.time()) // 300) * 300
            window_slug = f"btc-updown-5m-{window_start_ts}"
            if self._prepos_pending.get("slug") == window_slug:
                return

        # Check if position manager allows trading.
        # Pass the bot's own realized P&L from its trade records --
        # this is independent of the account balance, so the user's
        # manual trades don't affect the max-loss check.
        bot_pnl = self.trade_logger.get_bot_live_pnl()
        can_trade, reason = self.position_manager.can_trade(bot_pnl=bot_pnl)
        if not can_trade:
            logger.info(f"Cannot trade: {reason}")
            return

        # Get the current BTC price
        btc_price = self.price_monitor.current_price
        if btc_price is None:
            return

        # How much time left in this window?
        seconds_left = self.price_monitor.seconds_left_in_window()

        # Build the slug for the current window
        window_start_ts = (int(time.time()) // 300) * 300
        window_slug = f"btc-updown-5m-{window_start_ts}"

        # -- Scalp re-entry gate --
        entries_this_window = self._window_entry_counts.get(window_slug, 0)
        if entries_this_window > 0:
            # Reversal override: if BTC moved big in the OPPOSITE direction
            # from our previous trade, allow re-entry even without scalp enabled.
            # This catches the "V-shape" pattern where BTC spikes UP then DOWN.
            prev_dirs = self._window_trade_directions.get(window_slug, [])
            is_reversal = False
            if prev_dirs and btc_price:
                # Use the price_monitor's window start price for diff calc
                ws_price, _ = self.price_monitor.get_window_start_price()
                if ws_price and ws_price > 0:
                    diff_now = ((btc_price - ws_price) / ws_price) * 100
                    if abs(diff_now) >= 0.08:
                        current_dir = "UP" if diff_now > 0 else "DOWN"
                        if current_dir not in prev_dirs:
                            is_reversal = True
                            logger.info(
                                f"[REVERSAL] BTC flipped! Previously traded {prev_dirs} "
                                f"but now {diff_now:+.3f}% -> {current_dir} | "
                                f"Allowing opposite-direction trade in same window"
                            )

            if not is_reversal:
                if not self._scalp_enabled:
                    return  # scalp disabled -- strict one-trade-per-window
                if entries_this_window >= self._scalp_max_entries:
                    return  # hit cap for this window
                if time.time() < self._reentry_cooldown_until:
                    return  # still in cooldown after last exit
                logger.info(
                    f"? Scalp re-entry scan #{entries_this_window + 1}/{self._scalp_max_entries} "
                    f"for {window_slug}"
                )

        # Evict stale slugs and counts (older than 10 min)
        cutoff = window_start_ts - 600
        self._traded_window_slugs = {
            s for s in self._traded_window_slugs
            if int(s.rsplit("-", 1)[-1]) > cutoff
        }
        self._window_entry_counts = {
            s: c for s, c in self._window_entry_counts.items()
            if int(s.rsplit("-", 1)[-1]) > cutoff
        }
        self._prepos_window_slugs = {
            s for s in self._prepos_window_slugs
            if int(s.rsplit("-", 1)[-1]) > cutoff
        }
        self._window_trade_directions = {
            s: d for s, d in self._window_trade_directions.items()
            if int(s.rsplit("-", 1)[-1]) > cutoff
        }

        # Fetch market info and price-to-beat concurrently.
        # These are independent HTTP calls -- run them in parallel to
        # save ~200-500ms per scan cycle.
        loop = asyncio.get_event_loop()
        market_future = loop.run_in_executor(
            None, PolymarketClient._fetch_market_by_slug, window_slug
        )
        price_future = loop.run_in_executor(
            None, PolymarketClient.fetch_price_to_beat, window_slug
        )

        market, target_price = await asyncio.gather(market_future, price_future)

        if not market:
            logger.info(f"Market {window_slug} not found on Polymarket")
            return

        # Register market for WebSocket streaming (if not already)
        self._register_market_for_ws(window_slug, market)

        if target_price is None:
            # Fallback: use our own price feed's snapshot at window start
            ws_price = self.price_monitor.get_price_at(float(window_start_ts))
            if ws_price is not None:
                target_price = ws_price
                logger.info(
                    f"Using WebSocket price as target for {window_slug}: "
                    f"${target_price:,.2f} (Chainlink unavailable)"
                )
            else:
                # Only log once per slug to avoid spamming every 3 seconds
                if window_slug != self._last_no_target_slug:
                    logger.info(f"Waiting for price-to-beat for {window_slug}...")
                    self._last_no_target_slug = window_slug
                return

        # Clear the "waiting" flag now that we have a target price
        if self._last_no_target_slug == window_slug:
            logger.info(f"Price-to-beat found for {window_slug}: ${target_price:,.2f}")
            self._last_no_target_slug = ""

        # Analyze: BTC above or below target? Are odds favorable?
        # Feed in live analytics for the enhanced edge model (v2)
        measured_vol = self.price_monitor.measured_volatility()
        velocity = self.price_monitor.price_velocity(30)
        acceleration = self.price_monitor.price_acceleration()
        feed_agree = self.price_monitor.feed_agreement(target_price)
        n_feeds = self.price_monitor.active_feed_count()

        decision = self.arbitrage_engine.analyze_opportunity(
            btc_price, target_price, market, seconds_left,
            measured_vol=measured_vol,
            velocity=velocity,
            acceleration=acceleration,
            feed_agreement=feed_agree,
            active_feeds=n_feeds,
        )

        diff_pct = ((btc_price - target_price) / target_price) * 100

        # PURE LATENCY ARB: no learner adjustment.
        # Our edge is the 30s delay -- if BTC moved, buy now.

        # Log the signal regardless of whether we trade
        self.trade_logger.log_signal(
            direction=decision.direction,
            delta_pct=diff_pct,
            confidence=decision.confidence,
            btc_start=target_price,
            btc_end=btc_price,
            traded=decision.should_trade,
            reason=decision.reason,
        )

        if not decision.should_trade:
            logger.info(
                f"No trade: {decision.reason} | BTC=${btc_price:,.2f} "
                f"target=${target_price:,.2f} ({diff_pct:+.4f}%) {seconds_left:.0f}s left"
            )
            # -- Straddle check: even if directional trade is rejected,
            #    check if conditions are right for a straddle --
            if (
                self._straddle_enabled
                and self.poly_client
                and seconds_left <= self._straddle_trigger_secs
                and abs(diff_pct) <= self._straddle_max_diff
                and window_slug not in self._straddle_window_slugs
                and self._straddle_active is None
            ):
                await self._place_straddle(market, window_slug, target_price, diff_pct, seconds_left)
            self._prev_diff_pct = diff_pct
            return

        # PURE LATENCY ARB: no learner gates -- trade when BTC has moved.

        logger.info(
            f"[CHART] BTC ${btc_price:,.2f} vs target ${target_price:,.2f} "
            f"({diff_pct:+.4f}%) -> {decision.direction} | "
            f"Market: Up={market.up_price:.3f} Down={market.down_price:.3f} | "
            f"Edge={decision.edge:.4f} | {seconds_left:.0f}s left | "
            f"vol={measured_vol:.3f}% vel={velocity:+.5f} feeds={n_feeds}"
        )

        await self._live_trade(decision, market, btc_price, target_price, diff_pct, seconds_left)

        self._prev_diff_pct = diff_pct

    # -- WebSocket stream management ---------------------------------

    async def _start_ws_streams(self):
        """Start Polymarket WebSocket streams for real-time data.

        - Market stream: live price/book updates (replaces HTTP polling)
        - User stream: order lifecycle events (fills, cancels)

        Both streams auto-reconnect on failure. The existing HTTP methods
        remain as automatic fallbacks when WS data is unavailable.
        """
        # Always start the market price stream
        logger.info("[WS] Starting Polymarket market price stream...")
        market_task = asyncio.create_task(self._market_stream.start())
        self._ws_tasks.append(market_task)

        # Start user stream (needs API credentials)
        if self.poly_client:
            api_key = os.getenv("POLYMARKET_API_KEY", "")
            api_secret = os.getenv("POLYMARKET_API_SECRET", "")
            passphrase = os.getenv("POLYMARKET_PASSPHRASE", "")
            if api_key:
                self._user_stream = PolymarketUserStream(
                    api_key=api_key,
                    api_secret=api_secret,
                    passphrase=passphrase,
                )
                logger.info("[WS] Starting Polymarket user order stream...")
                user_task = asyncio.create_task(self._user_stream.start())
                self._ws_tasks.append(user_task)

        # Inject WS streams into the client module for fallback logic
        set_ws_streams(
            market_stream=self._market_stream,
            user_stream=self._user_stream,
        )

        # Give streams a moment to connect
        await asyncio.sleep(2)

        ws_status = "connected" if self._market_stream.connected else "connecting"
        logger.info(f"[WS] Market stream: {ws_status}")
        if self._user_stream:
            usr_status = "connected" if self._user_stream.connected else "connecting"
            logger.info(f"[WS] User stream: {usr_status}")

    async def _stop_ws_streams(self):
        """Stop all WebSocket streams gracefully."""
        logger.info("[WS] Stopping WebSocket streams...")
        await self._market_stream.stop()
        if self._user_stream:
            await self._user_stream.stop()
        for task in self._ws_tasks:
            task.cancel()
        self._ws_tasks.clear()

    def _register_market_for_ws(self, slug: str, market: MarketInfo):
        """Register a market's tokens with the WS stream for live prices.

        Called whenever we discover a new market (current or next window).
        Idempotent -- safe to call repeatedly for the same slug.
        """
        if slug in self._registered_ws_slugs:
            return
        self._registered_ws_slugs.add(slug)
        self._market_stream.register_market(
            slug, market.up_token_id, market.down_token_id
        )
        # Clean up old slugs (keep last 5)
        if len(self._registered_ws_slugs) > 5:
            oldest = sorted(self._registered_ws_slugs)
            for old_slug in oldest[:-5]:
                self._market_stream.unregister_market(old_slug)
                self._registered_ws_slugs.discard(old_slug)

    # -- Balance safety --------------------------------------------

    def _sync_bankroll_with_exchange(self, is_startup: bool = False):
        """Sync the bot's bankroll with the real Polymarket USDC balance.

        The bankroll IS the cash on Polymarket.  Every sync overwrites
        the virtual tracker so bet sizing always reflects reality.
        """
        if not self.poly_client:
            return
        live = self.poly_client.get_live_balance()
        if live is None:
            logger.warning("[!]  Could not fetch live balance -- keeping internal bankroll")
            return
        self._cached_real_balance = live
        self._last_balance_sync = time.time()

        old_bankroll = self.position_manager.bankroll
        self.position_manager.bankroll = live

        # Keep high-water mark in sync
        if live > self.position_manager.high_water_mark:
            self.position_manager.high_water_mark = live

        if is_startup:
            self.position_manager.starting_bankroll = live
            logger.info(
                f"[$] Bankroll synced to Polymarket cash: ${live:.2f}"
            )
        elif abs(live - old_bankroll) > 0.01:
            logger.info(
                f"[$] Bankroll synced: ${old_bankroll:.2f} -> ${live:.2f} "
                f"(Polymarket cash)"
            )

    def _check_balance_before_trade(self, cost: float) -> tuple[bool, str]:
        """Pre-trade solvency check: does the account have enough USDC?

        This only checks whether the exchange balance can cover the
        trade.  Max-loss is handled separately by can_trade() using
        the bot's own P&L from its trade records.
        """
        real_balance = self._get_real_balance()
        if real_balance < cost:
            return False, (
                f"Insufficient balance: need ${cost:.2f} but "
                f"account only has ${real_balance:.2f}"
            )
        return True, "OK"

    def _get_real_balance(self) -> float:
        """Get the real exchange balance, with caching (refresh every 30s).

        Also syncs bankroll so it always matches Polymarket cash.
        """
        now = time.time()
        if (
            self.poly_client
            and now - getattr(self, '_last_balance_sync', 0) > 30
        ):
            self._sync_bankroll_with_exchange()
        return getattr(self, '_cached_real_balance', self.position_manager.bankroll)

    # -- Trade execution (instant -- no blocking) ------------------

    async def _live_trade(
        self,
        decision,
        market: MarketInfo,
        btc_price: float,
        target_price: float,
        diff_pct: float,
        seconds_left: float = 0,
    ):
        """Execute a real trade on Polymarket and register for monitoring."""
        if not self.poly_client:
            return

        # ── Duplicate-position guard ──────────────────────────────
        # If we already hold shares on this market (e.g. from a stale
        # process, manual trade, or race condition), refuse to pile on.
        buy_token = (
            market.up_token_id if decision.direction == "UP"
            else market.down_token_id
        )
        try:
            existing = self.poly_client.get_position_balance(buy_token)
            # Ignore dust (< 0.5 shares) — leftover micro-units after
            # limit sells shouldn't block re-entry.
            if existing and float(existing) >= 0.5:
                logger.warning(
                    f"[DEDUP] Already hold {existing:.2f} shares of "
                    f"{decision.direction} on {market.slug} -- skipping entry"
                )
                return
            elif existing and float(existing) > 0:
                logger.debug(
                    f"[DEDUP] Ignoring dust: {existing:.4f} shares of "
                    f"{decision.direction} on {market.slug}"
                )
        except Exception as e:
            logger.debug(f"Position check failed (proceeding): {e}")

        secs_left = seconds_left or self.price_monitor.seconds_left_in_window()

        # ── Step 1: Check REAL CLOB order book FIRST ──────────────
        # The gamma API often returns stale prices (e.g. best_ask=0.10
        # when the real book has asks at 0.95+).  Hit the CLOB directly
        # to get the actual tradeable price before doing any analysis.
        buy_token_id = market.up_token_id if decision.direction == "UP" else market.down_token_id
        comp_token_id = market.down_token_id if decision.direction == "UP" else market.up_token_id

        # Query BOTH books in parallel (saves ~0.5s vs sequential)
        loop = asyncio.get_event_loop()
        book_future = loop.run_in_executor(
            None, PolymarketClient.check_clob_book, buy_token_id, 0.95
        )
        comp_future = loop.run_in_executor(
            None, PolymarketClient.check_clob_book, comp_token_id, 0.99
        )
        book, comp_book = await asyncio.gather(book_future, comp_future)

        # Determine the real cheapest ask on the native side
        real_best_ask = book["best_ask"] if book else None

        # Check complement: if someone bids on the opposite side at >= (1-X),
        # we can effectively buy our side at X via complement matching
        comp_bid = comp_book["best_bid"] if (comp_book and comp_book.get("best_bid")) else None
        effective_comp_ask = round(1.0 - comp_bid, 2) if comp_bid else None

        # The real fill price is the cheaper of native ask vs complement
        if real_best_ask and effective_comp_ask:
            real_fill = min(real_best_ask, effective_comp_ask)
        elif real_best_ask:
            real_fill = real_best_ask
        elif effective_comp_ask:
            real_fill = effective_comp_ask
        else:
            real_fill = None

        # ── Early exit: no book at all ────────────────────────────
        if real_fill is None:
            cache_key = f"nobook_{market.slug}_{decision.direction}"
            last_log = getattr(self, '_last_priced_in_log', {})
            now = time.time()
            if now - last_log.get(cache_key, 0) > 30:
                logger.info(
                    f">>  No asks on book for {decision.direction} -- skipping"
                )
                if not hasattr(self, '_last_priced_in_log'):
                    self._last_priced_in_log = {}
                self._last_priced_in_log[cache_key] = now
            return

        # Floor: fill prices below min_fill_price are junk bets -- historically 0% win rate.
        # Block them entirely.
        if real_fill < self._min_fill_price:
            cache_key = f"floor_{market.slug}_{decision.direction}"
            last_log = getattr(self, '_last_priced_in_log', {})
            now = time.time()
            if now - last_log.get(cache_key, 0) > 30:
                logger.info(
                    f">>  Fill price too low for {decision.direction}: "
                    f"${real_fill:.2f} (<${self._min_fill_price:.2f} floor) -- skipping (0% WR bucket)"
                )
                if not hasattr(self, '_last_priced_in_log'):
                    self._last_priced_in_log = {}
                self._last_priced_in_log[cache_key] = now
            return

        # PURE LATENCY ARB: no 0.65 price ceiling.
        # We buy at whatever price the stale book has. The delay IS the edge.
        # The 0.85 hard cap below still prevents buying near-certain outcomes.

        # Hard cap: if cheapest share is > 0.85, outcome is near-certain
        # and there's no value buying.  Let the edge check handle 0.50-0.85.
        if real_fill > 0.85:
            cache_key = f"priced_in_{market.slug}_{decision.direction}"
            last_log = getattr(self, '_last_priced_in_log', {})
            now = time.time()
            if now - last_log.get(cache_key, 0) > 30:
                logger.info(
                    f">>  Market fully priced in for {decision.direction}: "
                    f"cheapest=${real_fill:.2f} (>$0.85) -- skipping"
                )
                if not hasattr(self, '_last_priced_in_log'):
                    self._last_priced_in_log = {}
                self._last_priced_in_log[cache_key] = now
            return

        # Check minimum size: need at least 5 shares available
        native_size = book["ask_size"] if book else 0
        comp_size = comp_book["bid_size"] if comp_book else 0
        if native_size < 5 and comp_size < 5:
            logger.info(
                f">>  Book too thin for {decision.direction}: "
                f"native_size={native_size:.0f}, comp_size={comp_size:.0f} -- skipping"
            )
            return

        # ── Step 2: Derive prices from CLOB (no gamma API call) ──
        # The gamma API added ~500ms latency for data we can compute
        # from the CLOB book we already have.
        up_price = market.up_price   # from gamma fetch in _scan_for_entry
        down_price = market.down_price

        # ── Step 3: Use the REAL CLOB price as fill price ─────────
        # Start from the real best ask, then add a small proportional bump
        # for priority.  Old flat bump ($0.05-$0.10) was fine at $0.05 fills
        # but killed edge at $0.50+ CLOB prices.  Use 2% of ask price.
        fill_price = real_fill
        bump_pct = 0.01  # 1% bump -- just enough for priority, saves on cost
        fill_bump = round(real_fill * bump_pct, 2)
        fill_bump = max(fill_bump, 0.01)  # at least 1 cent
        original_ask = fill_price
        fill_price = round(min(fill_price + fill_bump, 0.95), 2)
        if fill_price != original_ask:
            logger.info(
                f"[$] Price bump: {original_ask:.3f} -> {fill_price:.3f} "
                f"(+{fill_bump:.3f}, 1% of ask)"
            )

        if fill_price <= 0 or fill_price >= 1:
            logger.warning(
                f"Invalid fill price {fill_price} for {decision.direction} -- skipping"
            )
            return

        # ── Step 4: Edge check against REAL price ─────────────────
        estimated_prob = self.arbitrage_engine._estimate_probability(
            diff_pct if decision.direction == "UP" else -diff_pct,
            secs_left,
        )

        real_edge = estimated_prob - fill_price
        min_edge = self.config["strategy"].get("min_edge", 0.01)

        if real_edge < min_edge:
            logger.info(
                f"Edge vs CLOB ask too small: {real_edge:.4f} < {min_edge} "
                f"(prob={estimated_prob:.3f}, real_ask={fill_price:.3f}) -- skipping"
            )
            return

        logger.info(
            f"Execution: prob={estimated_prob:.3f} fill@={fill_price:.3f} "
            f"edge={real_edge:.4f} (CLOB verified)"
        )

        # PURE LATENCY ARB: TA gate disabled.
        # The delay is the edge, not technical indicators.
        ta_quality = 0.5

        prob = estimated_prob
        bet_size = self.position_manager.calculate_bet_size(
            edge=real_edge, probability=prob
        )

        # ── Hard safety caps (anti-rogue) ─────────────────────────
        # 1. Absolute per-trade cap -- overrides everything
        if bet_size > self._hard_max_per_trade:
            logger.info(
                f"[SAFETY] Hard cap: ${bet_size:.2f} -> ${self._hard_max_per_trade:.2f}"
            )
            bet_size = self._hard_max_per_trade

        # 2. Per-window spend cap
        window_slug = market.slug
        spent_this_window = self._window_spend.get(window_slug, 0.0)
        remaining_budget = self._max_per_window - spent_this_window
        if remaining_budget <= 0:
            logger.info(
                f"[SAFETY] Window budget exhausted: already spent "
                f"${spent_this_window:.2f} on {window_slug} (cap ${self._max_per_window:.2f})"
            )
            return
        if bet_size > remaining_budget:
            logger.info(
                f"[SAFETY] Window cap: ${bet_size:.2f} -> ${remaining_budget:.2f} "
                f"(${spent_this_window:.2f} already spent this window)"
            )
            bet_size = round(remaining_budget, 2)

        # PURE LATENCY ARB: no TA scaling, no early-confidence boost,
        # no mispricing tiers.  Flat bet size from Kelly, trade every signal.
        mispricing_tier = "NONE"

        # Grab velocity for learner tracking
        entry_velocity = self.price_monitor.price_velocity()

        # -- Double It & Pass: override bet size when parlay is active --
        if self._parlay_active:
            bet_size = self._parlay_stake
            logger.info(
                f"[PARLAY] Round {self._parlay_round + 1} "
                f"-- betting ${bet_size:.2f} (streak profit: ${self._parlay_total_won:.2f})"
            )

        # Hard safety check: real balance vs floor
        ok, reason = self._check_balance_before_trade(bet_size)
        if not ok:
            logger.warning(f"[STOP] {reason}")
            return

        bankroll_before = self.position_manager.bankroll

        # -- Place order: GTC directly --
        # FOK almost never fills on thin 5-min books and wastes 1-2s per
        # rejected round-trip.  Go straight to GTC -- the fill polling
        # below will verify it matched or cancel after 5s.
        result = self.poly_client.place_order(
            market, decision.direction, bet_size, fill_price,
            order_type="GTC",
        )

        filled = False
        if not result.success:
            logger.error(f"GTC order failed: {result.error}")
            return

        # Check if the order was immediately matched (from API response status)
        if result.matched:
            filled = True
            logger.info(f"* GTC matched instantly @ {result.fill_price:.3f}")

        order_id = result.order_id or ""

        # Use the actual cost from the exchange (accounts for share rounding)
        actual_cost = result.actual_cost or bet_size
        if actual_cost != bet_size:
            logger.info(
                f"Actual cost ${actual_cost:.2f} differs from intended "
                f"${bet_size:.2f} -- using actual"
            )
            bet_size = actual_cost

        # -- Verify the order actually filled (poll up to 5s) --
        max_polls = 5
        if not filled and order_id and self.poly_client:
            for attempt in range(max_polls):
                await asyncio.sleep(1.0)
                try:
                    status = self.poly_client.check_position_status(order_id)
                    matched = status.get("filled", 0)
                    order_status = status.get("status", "")
                    logger.info(
                        f"Fill check [{attempt+1}/{max_polls}]: status={order_status} "
                        f"filled={matched}"
                    )
                    if matched > 0:
                        filled = True
                        break
                    # If exchange already cancelled it (e.g., market resolved)
                    if order_status.upper() in ("CANCELED", "CANCELLED",
                                                 "CANCELED_MARKET_RESOLVED"):
                        break
                except Exception as e:
                    logger.warning(f"Fill check error: {e}")

            if not filled:
                logger.warning(
                    f"[!] Order {order_id} NOT filled after {max_polls}s -- cancelling"
                )
                try:
                    self.poly_client.cancel_order(order_id)
                except Exception:
                    pass  # may already be cancelled by exchange
                # Count this as an attempt so we don't spam orders endlessly
                self._window_entry_counts[market.slug] = (
                    self._window_entry_counts.get(market.slug, 0) + 1
                )
                # Cooldown before next attempt in this window
                self._reentry_cooldown_until = time.time() + 15.0
                return

        trade_id = self.trade_logger.log_trade(
            btc_price_start=target_price,
            btc_price_end=btc_price,
            delta_pct=diff_pct,
            market_id=market.condition_id,
            market_question=market.question,
            odds_yes=up_price,
            odds_no=down_price,
            direction=decision.direction,
            bet_size=bet_size,
            fill_price=result.fill_price or fill_price,
            order_id=order_id,
            bankroll_before=bankroll_before,
            edge=real_edge,
            confidence=decision.confidence,
        )

        logger.info(
            f"? LIVE {decision.direction} ${bet_size:.2f} @ "
            f"{fill_price:.3f} on {market.slug} (FILLED +)"
        )

        # -- Determine if this is a high-confidence scalp trade --
        # Data: 0.10%+ moves have 87.5% WR.  Mark these for tighter
        # take-profit so we get in and get out quickly.
        is_confident_scalp = abs(diff_pct) >= 0.10

        # Register position for monitoring -- no blocking
        actual_entry = result.fill_price or fill_price
        self._open_position = {
            "trade_id": trade_id,
            "slug": market.slug,
            "direction": decision.direction,
            "entry_price": actual_entry,
            "bet_size": bet_size,
            "target_price": target_price,
            "diff_pct": diff_pct,
            "edge": real_edge,
            "seconds_left": secs_left,
            "prev_diff_pct": self._prev_diff_pct,
            "order_id": order_id,
            "limit_sell_order_id": None,  # filled in below
            "confident_scalp": is_confident_scalp,
            "ta_quality": ta_quality,
            "mispricing_tier": mispricing_tier,
            "velocity": entry_velocity,
            "entry_ts": time.time(),
        }

        if is_confident_scalp:
            logger.info(
                f"[SCALP] HIGH-CONFIDENCE entry: BTC delta {diff_pct:+.4f}% "
                f"-> tight take-profit for quick exit"
            )
        self._traded_window_slugs.add(market.slug)
        self._window_entry_counts[market.slug] = self._window_entry_counts.get(market.slug, 0) + 1
        # Track which directions we've traded this window (for reversal detection)
        dirs = self._window_trade_directions.setdefault(market.slug, [])
        if decision.direction not in dirs:
            dirs.append(decision.direction)

        # Track per-window spending for safety cap
        self._window_spend[market.slug] = self._window_spend.get(market.slug, 0.0) + bet_size
        # Evict old window spend entries (>10 min old)
        cutoff_ts = ((int(time.time()) // 300) * 300) - 600
        self._window_spend = {
            s: v for s, v in self._window_spend.items()
            if int(s.rsplit("-", 1)[-1]) > cutoff_ts
        }

        # -- Place a GTC limit sell for take-profit (with settlement delay) --
        # PURE LATENCY ARB: flat tight TP from config.  Buy cheap stale
        # price, sell when Polymarket catches up.  Small guaranteed win.
        limit_sell_pct = self.config["strategy"].get("limit_sell_profit", 0.04)
        if limit_sell_pct and limit_sell_pct > 0:
            sell_target = round(actual_entry * (1.0 + limit_sell_pct), 2)
            sell_target = min(sell_target, 0.99)  # can't exceed 0.99
            shares = bet_size / actual_entry

            # Flag so monitor loop knows a limit sell is being set up
            if self._open_position is not None:
                self._open_position["_limit_sell_pending"] = True

            task = asyncio.create_task(
                self._place_limit_sell_bg_task(
                    market, decision.direction, shares, sell_target,
                    max_attempts=15, initial_delay=0.0, label="LIMIT SELL",
                )
            )
            self._background_tasks.append(task)

    # -- Take-profit execution -------------------------------------

    async def _execute_take_profit(self, pos: dict, our_prob: float,
                                    secs_left: float, btc_now: float):
        """Sell shares at best_bid to lock in profit.

        Uses gamma-api best_bid for instant fill.  If the sell fails
        or shares are below minimum, fall through to hold-for-resolution.
        """
        slug = pos["slug"]
        direction = pos["direction"]
        entry_price = pos["entry_price"]
        bet_size = pos["bet_size"]
        trade_id = pos["trade_id"]

        # Fetch current market prices
        live = PolymarketClient.get_live_market_price(slug)
        if not live:
            logger.warning("Cannot fetch live prices for take-profit -- holding")
            return

        # The best_bid is the best price someone will pay for UP shares.
        # For DOWN shares, the sell price = 1 - UP best_ask.
        if direction == "UP":
            sell_price = live["best_bid"]
        else:
            sell_price = round(1.0 - live["best_ask"], 2) if live["best_ask"] else None

        if not sell_price or sell_price <= 0:
            logger.warning("No valid sell price for take-profit -- holding")
            return

        # Only take profit if we'd actually make money
        if sell_price <= entry_price:
            logger.info(
                f"[GEM] HOLD: sell@{sell_price:.3f} <= entry@{entry_price:.3f} -- "
                f"no profit to take, holding ({secs_left:.0f}s left)"
            )
            return

        # Calculate shares we hold
        shares = bet_size / entry_price

        # Fetch the market object for sell_position
        market = PolymarketClient._fetch_market_by_slug(slug)
        if not market:
            logger.warning("Cannot fetch market for sell -- holding")
            return

        logger.info(
            f"[$] TAKE-PROFIT: prob={our_prob:.3f} sell@{sell_price:.3f} "
            f"(entry@{entry_price:.3f}) {shares:.1f} shares | "
            f"{secs_left:.0f}s left"
        )

        filled, order_id = await self._sell_and_poll(
            market, direction, shares, sell_price,
            max_attempts=3, poll_delay=0.8, label="Take-profit",
        )

        if not filled:
            return  # hold to resolution

        # Calculate actual profit: proceeds - cost
        profit = (shares * sell_price) - bet_size

        self._record_trade_outcome(pos, won=True, pnl=profit, btc_now=btc_now)

        can_reenter = self._check_reentry(pos, secs_left)

        logger.info(
            f"[OK] LIVE TAKE-PROFIT +${profit:.2f} [{slug}] | "
            f"Sold {shares:.1f} shares @ {sell_price:.3f} "
            f"(entry {entry_price:.3f}) | "
            f"Bankroll: ${self.position_manager.bankroll:.2f}"
            + (f" | ? Re-entry eligible ({self._scalp_cooldown_tp}s cooldown)" if can_reenter else "")
        )

        self._open_position = None

    # -- Hedge lock-in (insurance strategy) --------------------------

    async def _execute_hedge_lockin(self, pos: dict, our_prob: float, secs_left: float):
        """Buy the opposite side to lock in guaranteed profit.

        The math:
          We hold N shares of side A at entry_price.
          Cost so far = N * entry_price.
          The opposite side B is cheap (~= 1 - our_prob).
          If we buy N shares of side B at hedge_price:
            Total invested = N * (entry_price + hedge_price)
            One side ALWAYS pays $1/share -> payout = N * $1.00
            Guaranteed profit = N * (1.0 - entry_price - hedge_price)

          This only works when entry_price + hedge_price < 1.0,
          which happens when we bought our side cheap and the opposite
          side is now very cheap because we're winning.
        """
        direction = pos["direction"]
        entry_price = pos["entry_price"]
        bet_size = pos["bet_size"]
        slug = pos["slug"]
        shares = bet_size / entry_price

        opposite = "DOWN" if direction == "UP" else "UP"

        # Get live price for the opposite side
        live = PolymarketClient.get_live_market_price(slug)
        if not live:
            logger.debug("Hedge: cannot fetch live prices -- skipping")
            return

        # What does it cost to buy the opposite side right now?
        if opposite == "DOWN":
            # DOWN best ask = 1 - UP best bid
            hedge_price = round(1.0 - live["best_bid"], 2) if live["best_bid"] else live["down_price"]
        else:
            # UP best ask
            hedge_price = live["best_ask"] if live["best_ask"] else live["up_price"]

        if hedge_price <= 0 or hedge_price >= 1:
            logger.debug(f"Hedge: invalid opposite price {hedge_price} -- skipping")
            return

        # The key formula: can we lock in a profit?
        combined = entry_price + hedge_price
        if combined >= 1.0:
            # No guaranteed profit possible -- the spread eats it
            logger.debug(
                f"Hedge: entry({entry_price:.3f}) + opposite({hedge_price:.3f}) "
                f"= {combined:.3f} >= 1.00 -- no profit to lock in"
            )
            return

        # How much profit per share?
        profit_per_share = 1.0 - combined
        guaranteed_profit = shares * profit_per_share
        hedge_cost = shares * hedge_price
        total_invested = bet_size + hedge_cost
        profit_pct = guaranteed_profit / total_invested

        # Must meet minimum profit threshold
        if profit_pct < self._hedge_min_profit_pct:
            logger.debug(
                f"Hedge: profit {profit_pct:.1%} < min {self._hedge_min_profit_pct:.0%} "
                f"-- not enough upside"
            )
            return

        # Cap hedge cost
        if hedge_cost > self._hedge_max_cost:
            # Scale down: buy fewer insurance shares
            affordable_shares = self._hedge_max_cost / hedge_price
            hedge_cost = affordable_shares * hedge_price
            # We'll have a partial hedge -- only affordable_shares are insured.
            # The rest ride on the original directional bet.
            partial = True
            logger.info(
                f"? HEDGE: capping insurance to {affordable_shares:.1f} of "
                f"{shares:.1f} shares (max cost ${self._hedge_max_cost:.2f})"
            )
        else:
            affordable_shares = shares
            partial = False

        # Balance check
        ok, reason = self._check_balance_before_trade(hedge_cost)
        if not ok:
            logger.info(f"? HEDGE blocked: {reason}")
            return

        # Fetch market for order placement
        market = PolymarketClient._fetch_market_by_slug(slug)
        if not market:
            logger.warning("Hedge: cannot fetch market object -- skipping")
            return

        # Apply fill bump for better fill rate
        fill_bump = self.config["strategy"].get("fill_bump", 0.0)
        order_price = round(min(hedge_price + fill_bump, 0.95), 2)
        actual_hedge_cost = round(affordable_shares * order_price, 2)

        logger.info(
            f"? HEDGE LOCK-IN: Buying {affordable_shares:.1f} {opposite} shares "
            f"@ {order_price:.3f} (cost ${actual_hedge_cost:.2f}) | "
            f"Entry {direction}@{entry_price:.3f} + {opposite}@{order_price:.3f} "
            f"= {entry_price + order_price:.3f} | "
            f"[$] Guaranteed profit: ${shares * (1.0 - entry_price - order_price):.2f} "
            f"({(1.0 - entry_price - order_price) / (entry_price + order_price) * 100:.1f}%)"
        )

        # Place FOK order for instant fill
        result = self.poly_client.place_order(
            market, opposite, actual_hedge_cost, order_price,
            order_type="FOK",
        )

        if result.success:
            pos["hedge_placed"] = True
            pos["hedge_side"] = opposite
            pos["hedge_price"] = order_price
            pos["hedge_cost"] = actual_hedge_cost
            pos["hedge_shares"] = affordable_shares
            pos["hedge_order_id"] = result.order_id

            locked_profit = affordable_shares * (1.0 - entry_price - order_price)
            logger.info(
                f"[OK] HEDGE FILLED! {affordable_shares:.1f} {opposite} @ {order_price:.3f} | "
                f"[LOCK] Locked in ${locked_profit:.2f} profit regardless of outcome"
                + (" (partial hedge)" if partial else " (FULL hedge)")
            )
        else:
            # FOK failed -- try GTC with quick poll
            logger.info(
                f"Hedge FOK rejected ({result.error}) -- trying GTC"
            )
            result = self.poly_client.place_order(
                market, opposite, actual_hedge_cost, order_price,
                order_type="GTC",
            )
            if result.success:
                # Quick fill check (2 attempts)
                filled = False
                for attempt in range(2):
                    await asyncio.sleep(1.0)
                    try:
                        status = self.poly_client.check_position_status(result.order_id)
                        if status.get("filled", 0) > 0:
                            filled = True
                            break
                    except Exception:
                        pass

                if filled:
                    pos["hedge_placed"] = True
                    pos["hedge_side"] = opposite
                    pos["hedge_price"] = order_price
                    pos["hedge_cost"] = actual_hedge_cost
                    pos["hedge_shares"] = affordable_shares
                    pos["hedge_order_id"] = result.order_id
                    locked_profit = affordable_shares * (1.0 - entry_price - order_price)
                    logger.info(
                        f"[OK] HEDGE FILLED (GTC)! [LOCK] Locked ${locked_profit:.2f}"
                    )
                else:
                    try:
                        self.poly_client.cancel_order(result.order_id)
                    except Exception:
                        pass
                    logger.info("Hedge GTC not filled -- cancelled, will retry next scan")
            else:
                logger.warning(f"Hedge order failed: {result.error}")

    # -- Momentum pre-position (next-window orders) -----------------

    async def _check_preposition(self):
        """In the last 30s of the current window, if BTC is surging/crashing,
        place a GTC limit order on the NEXT window's market in the direction
        of the momentum.

        Rationale:
          - The next window's market is already listed on gamma-api, priced
            near 50/50 because the "price to beat" hasn't been set yet.
          - If BTC is in a strong directional move (e.g. +0.05% and
            accelerating), there's a good chance the move continues into
            the next window -- BTC will open ABOVE the new target.
          - By placing a limit order at ~$0.50 before the window starts,
            we get positioned cheaply before the market adjusts.
          - If the move reverses, the order may not fill (thin book at 50/50)
            or we cancel early in the new window when momentum fades.

        This is directional (unlike straddle) -- we only bet the momentum side.
        """
        secs_left = self.price_monitor.seconds_left_in_window()
        if secs_left > self._prepos_trigger_secs or secs_left <= 3:
            return  # not in the trigger zone

        btc_price = self.price_monitor.current_price
        if btc_price is None:
            return

        # Get the current window's target to measure momentum
        window_start_ts = (int(time.time()) // 300) * 300
        current_slug = f"btc-updown-5m-{window_start_ts}"

        # Only pre-position once per next window
        next_ts = window_start_ts + 300
        next_slug = f"btc-updown-5m-{next_ts}"
        if next_slug in self._prepos_window_slugs:
            return

        # Pre-positioning is independent of the current window's position.
        # We're placing a resting GTC order on the NEXT window's market,
        # so having a position in the current window is fine.

        # Solvency check only (skip can_trade cooldown -- prepositions are
        # for the next window, loss cooldown shouldn't block them)
        bot_pnl = self.trade_logger.get_bot_live_pnl()
        # Still respect max-loss and consecutive-loss circuit breakers
        if bot_pnl < -self.position_manager.max_loss:
            return
        if self.position_manager.consecutive_losses >= self.position_manager.max_consecutive_losses:
            return

        # Measure momentum: how far has BTC moved from the current window's open?
        target_price = PolymarketClient.fetch_price_to_beat(current_slug)
        if target_price is None:
            return

        diff_pct = ((btc_price - target_price) / target_price) * 100

        # Check momentum strength
        if abs(diff_pct) < self._prepos_min_momentum:
            self._prev_diff_pct = diff_pct  # keep fresh for acceleration check
            return  # move isn't strong enough

        # Check acceleration (is it still moving in the same direction?)
        if self._prev_diff_pct is not None:
            accel = abs(diff_pct) - abs(self._prev_diff_pct)
            # Positive accel means the move is getting bigger
            if accel < self._prepos_min_accel:
                self._prev_diff_pct = diff_pct
                return  # momentum is fading, not accelerating
            # Also check direction hasn't flipped
            if self._prev_diff_pct != 0 and (
                (diff_pct > 0) != (self._prev_diff_pct > 0)
            ):
                self._prev_diff_pct = diff_pct
                return  # direction reversed
        else:
            self._prev_diff_pct = diff_pct
            return  # need at least one prior reading for acceleration

        direction = "UP" if diff_pct > 0 else "DOWN"

        # Fetch the NEXT window's market from gamma-api
        loop = asyncio.get_event_loop()
        next_market = await loop.run_in_executor(
            None, PolymarketClient._fetch_market_by_slug, next_slug
        )
        if not next_market:
            logger.debug(f"Pre-position: next market {next_slug} not yet available")
            return

        # Register next-window market for WebSocket streaming
        self._register_market_for_ws(next_slug, next_market)

        # Use a limit price near 50/50 -- the market shouldn't have moved yet
        if direction == "UP":
            limit_price = min(next_market.up_price, self._prepos_max_price)
        else:
            limit_price = min(next_market.down_price, self._prepos_max_price)

        if limit_price <= 0 or limit_price >= 0.60:
            logger.debug(
                f"Pre-position: price {limit_price:.3f} too high "
                f"(next window already priced in?) -- skipping"
            )
            return

        # Size: fraction of normal bet
        prob = 0.55  # conservative estimate for continuation
        edge = prob - limit_price
        if edge <= 0:
            return
        bet_size = self.position_manager.calculate_bet_size(
            edge=edge, probability=prob
        )
        bet_size = round(bet_size * self._prepos_bet_fraction, 2)

        # Balance check
        ok, reason = self._check_balance_before_trade(bet_size)
        if not ok:
            logger.info(f"[GO] Pre-position blocked: {reason}")
            return

        shares = math.floor((bet_size / limit_price) * 10000) / 10000

        logger.info(
            f"[GO] PRE-POSITION: BTC momentum {diff_pct:+.4f}% (accel "
            f"+{abs(diff_pct) - abs(self._prev_diff_pct):.4f}%) -> {direction} | "
            f"Placing {shares:.1f} shares @ {limit_price:.3f} on NEXT window "
            f"{next_slug} ({secs_left:.0f}s before it starts)"
        )

        result = self.poly_client.place_limit_order(
            next_market, direction, shares, limit_price
        )

        if result.success:
            self._prepos_pending = {
                "slug": next_slug,
                "direction": direction,
                "limit_price": limit_price,
                "shares": shares,
                "bet_size": bet_size,
                "order_id": result.order_id,
                "placed_at": time.time(),
                "momentum_diff_pct": diff_pct,
                "window_start_ts": next_ts,
            }
            self._prepos_window_slugs.add(next_slug)

            logger.info(
                f"  [OK] Pre-position order placed: {direction} {shares:.1f} shares "
                f"@ {limit_price:.3f} on {next_slug} -> {result.order_id}"
            )

            # Log the trade
            bankroll = self.position_manager.bankroll
            trade_id = self.trade_logger.log_trade(
                btc_price_start=0.0,  # target not yet known
                btc_price_end=btc_price,
                delta_pct=diff_pct,
                market_id=next_market.condition_id,
                market_question=f"PREPOS {direction}: {next_market.question}",
                odds_yes=next_market.up_price,
                odds_no=next_market.down_price,
                direction=direction,
                bet_size=bet_size,
                fill_price=limit_price,
                order_id=result.order_id or "",
                bankroll_before=bankroll,
                edge=edge,
                confidence=abs(diff_pct) / 0.10,
            )
            # Store trade_id so _monitor_preposition can reuse it
            self._prepos_pending["trade_id"] = trade_id
        else:
            logger.warning(f"  [X] Pre-position failed: {result.error}")

    async def _monitor_preposition(self):
        """Monitor a pending pre-position order on the next window.

        Three phases:
          1. BEFORE the window starts: just wait. The GTC order rests.
          2. EARLY in the new window (first 30s): check if filled. If
             filled, promote to an open position for normal monitoring.
             If not filled and momentum has reversed, cancel.
          3. After 45s unfilled: cancel (the opportunity has passed).
        """
        prepos = self._prepos_pending
        if not prepos:
            return

        now = time.time()
        window_start_ts = prepos["window_start_ts"]
        order_id = prepos["order_id"]
        elapsed_in_window = now - window_start_ts  # negative if window hasn't started

        # Phase 1: Window hasn't started yet -- just wait
        if elapsed_in_window < 0:
            return

        # Phase 2 & 3: Window has started -- check fill status
        if not self.poly_client or not order_id:
            self._prepos_pending = None
            return

        try:
            status = self.poly_client.check_position_status(order_id)
            filled = status.get("filled", 0)
            order_status = status.get("status", "")
        except Exception as e:
            logger.warning(f"Pre-position check error: {e}")
            return

        if order_status.upper() in ("CANCELED", "CANCELLED", "CANCELED_MARKET_RESOLVED"):
            logger.info(f"[GO] Pre-position {order_id} was cancelled externally")
            # Mark trade as CANCELLED
            tid = prepos.get("trade_id")
            if tid:
                self.trade_logger.update_trade_outcome(
                    tid, "CANCELLED", 0.0,
                    self.position_manager.bankroll,
                    self.position_manager.consecutive_wins,
                )
            self._prepos_pending = None
            return

        if filled > 0:
            # Order filled -- promote to a real open position!
            slug = prepos["slug"]
            direction = prepos["direction"]
            entry_price = prepos["limit_price"]
            bet_size = prepos["bet_size"]
            trade_id = prepos.get("trade_id")  # reuse the trade_id from placement

            # Fetch the new window's target price (now available)
            target_price = PolymarketClient.fetch_price_to_beat(slug)

            # ── Direction validation: does BTC still support this trade? ──
            # The preposition was placed based on momentum in the OLD window.
            # BTC may have reversed since then.  If price is now moving
            # against us, immediately sell rather than holding a bad position.
            btc_now = self.price_monitor.current_price
            if btc_now and target_price and target_price > 0:
                new_diff = ((btc_now - target_price) / target_price) * 100
                direction_wrong = (
                    (direction == "DOWN" and new_diff > 0.05)   # BTC above target = UP, we hold DOWN
                    or (direction == "UP" and new_diff < -0.05)  # BTC below target = DOWN, we hold UP
                )
                if direction_wrong:
                    logger.warning(
                        f"[GO][X] PRE-POSITION FILLED but BTC reversed! "
                        f"Holding {direction} but BTC diff is {new_diff:+.4f}% "
                        f"-- direction is WRONG. Marking as orphan for resolution."
                    )
                    # Don't promote -- let it resolve at expiry or get orphaned
                    # This avoids placing a losing limit sell that wastes time
                    if tid := prepos.get("trade_id"):
                        self.trade_logger.update_trade_outcome(
                            tid, "LOSS", -bet_size,
                            self.position_manager.bankroll - bet_size,
                            self.position_manager.consecutive_wins,
                        )
                    self.position_manager.update_after_loss(bet_size)
                    self._prepos_pending = None
                    return

            logger.info(
                f"[GO][OK] PRE-POSITION FILLED: {direction} {prepos['shares']:.1f} "
                f"shares @ {entry_price:.3f} on {slug} | "
                f"Original momentum: {prepos['momentum_diff_pct']:+.4f}% | "
                f"New window target: ${target_price:,.2f}" if target_price else
                f"[GO][OK] PRE-POSITION FILLED: {direction} {prepos['shares']:.1f} "
                f"shares @ {entry_price:.3f} on {slug} (target pending)"
            )

            # Promote to open position for normal monitoring
            # Reuse the trade_id logged at placement time (no duplicate log)
            self._open_position = {
                "trade_id": trade_id,
                "slug": slug,
                "direction": direction,
                "entry_price": entry_price,
                "bet_size": bet_size,
                "target_price": target_price or 0,  # recovery logic in _monitor_open_position will fetch if 0
                "_target_retry_count": 0,
                "diff_pct": prepos["momentum_diff_pct"],
                "edge": 0.05,
                "seconds_left": 300 - elapsed_in_window,
                "prev_diff_pct": None,
                "order_id": order_id,
                "limit_sell_order_id": None,
                "from_preposition": True,
                "entry_ts": time.time(),
            }
            self._traded_window_slugs.add(slug)
            self._window_entry_counts[slug] = self._window_entry_counts.get(slug, 0) + 1
            self._prepos_pending = None

            # Place a limit sell for take-profit (same as normal live trade)
            limit_sell_pct = self.config["strategy"].get("limit_sell_profit", 0.10)
            if limit_sell_pct and limit_sell_pct > 0 and self.poly_client:
                sell_target = round(entry_price * (1.0 + limit_sell_pct), 2)
                sell_target = min(sell_target, 0.99)
                shares = bet_size / entry_price
                next_market = PolymarketClient._fetch_market_by_slug(slug)

                if next_market:
                    task = asyncio.create_task(
                        self._place_limit_sell_bg_task(
                            next_market, direction, shares, sell_target,
                            max_attempts=10, initial_delay=18.0,
                            label="PREPOS LIMIT SELL",
                        )
                    )
                    self._background_tasks.append(task)

            return

        # Not filled yet -- check if we should cancel
        btc_now = self.price_monitor.current_price
        if btc_now and elapsed_in_window > 10:
            # If momentum reversed in the new window, cancel immediately
            new_target = PolymarketClient.fetch_price_to_beat(prepos["slug"])
            if new_target:
                new_diff = ((btc_now - new_target) / new_target) * 100
                expected_dir = prepos["direction"]
                momentum_gone = (
                    (expected_dir == "UP" and new_diff < -0.01)
                    or (expected_dir == "DOWN" and new_diff > 0.01)
                )
                if momentum_gone:
                    logger.info(
                        f"[GO][X] Pre-position: momentum reversed "
                        f"(new diff {new_diff:+.4f}%, expected {expected_dir}) "
                        f"-- cancelling"
                    )
                    try:
                        self.poly_client.cancel_order(order_id)
                    except Exception:
                        pass
                    # Mark trade as CANCELLED
                    tid = prepos.get("trade_id")
                    if tid:
                        self.trade_logger.update_trade_outcome(
                            tid, "CANCELLED", 0.0,
                            self.position_manager.bankroll,
                            self.position_manager.consecutive_wins,
                        )
                    self._prepos_pending = None
                    return

        # Cancel if unfilled after 45s into the new window
        if elapsed_in_window > 45:
            logger.info(
                f"[GO][TIME] Pre-position unfilled after {elapsed_in_window:.0f}s "
                f"-- cancelling {order_id}"
            )
            try:
                self.poly_client.cancel_order(order_id)
            except Exception:
                pass
            # Mark trade as CANCELLED
            tid = prepos.get("trade_id")
            if tid:
                self.trade_logger.update_trade_outcome(
                    tid, "CANCELLED", 0.0,
                    self.position_manager.bankroll,
                    self.position_manager.consecutive_wins,
                )
            self._prepos_pending = None

    # -- Straddle strategy ------------------------------------------

    async def _place_straddle(
        self,
        market: MarketInfo,
        window_slug: str,
        target_price: float,
        diff_pct: float,
        seconds_left: float,
    ):
        """Place GTC limit orders on BOTH sides at cheap price.

        The idea: when BTC is very close to the target in the last 1-2 min,
        set limit orders for both UP and DOWN at $0.05.  If the market
        fluctuates and both get filled, one side pays $1 on win minus the
        $0.05 cost of the losing side = guaranteed profit.

        Even if only one side fills, you get a high-upside cheap directional bet.
        """
        price = self._straddle_price
        shares = self._straddle_shares
        total_cost = shares * price * 2
        if total_cost > self._straddle_max_cost:
            shares = int(self._straddle_max_cost / (price * 2))

        if not self.poly_client:
            logger.warning("[SWAP] STRADDLE: No Polymarket client -- skipping")
            return

        # Enforce min_bankroll floor for straddle too
        can_trade, reason = self.position_manager.can_trade()
        if not can_trade:
            logger.info(f"[SWAP] STRADDLE blocked: {reason}")
            return

        max_straddle_cost = shares * price * 2
        ok, reason = self._check_balance_before_trade(max_straddle_cost)
        if not ok:
            logger.warning(f"[SWAP] STRADDLE blocked: {reason}")
            return

        logger.info(
            f"[SWAP] STRADDLE: BTC near target ({diff_pct:+.4f}%) with {seconds_left:.0f}s left | "
            f"Placing {shares} shares BOTH sides @ ${price:.2f} "
            f"(max cost ${shares * price * 2:.2f})"
        )

        up_result = self.poly_client.place_limit_order(market, "UP", shares, price)
        down_result = self.poly_client.place_limit_order(market, "DOWN", shares, price)

        self._straddle_active = {
            "slug": window_slug,
            "target_price": target_price,
            "price": price,
            "shares": shares,
            "up_order_id": up_result.order_id if up_result.success else None,
            "down_order_id": down_result.order_id if down_result.success else None,
            "up_filled": False,
            "down_filled": False,
            "placed_at": time.time(),
        }
        self._straddle_window_slugs.add(window_slug)

        if up_result.success:
            logger.info(f"  [OK] UP limit: {shares} shares @ ${price:.2f} -> {up_result.order_id}")
        else:
            logger.warning(f"  [X] UP limit failed: {up_result.error}")

        if down_result.success:
            logger.info(f"  [OK] DOWN limit: {shares} shares @ ${price:.2f} -> {down_result.order_id}")
        else:
            logger.warning(f"  [X] DOWN limit failed: {down_result.error}")

        # Log both sides as trades
        bankroll = self.position_manager.bankroll
        for side, res in [("UP", up_result), ("DOWN", down_result)]:
            if res.success:
                cost = shares * price
                trade_id = self.trade_logger.log_trade(
                    btc_price_start=target_price,
                    btc_price_end=self.price_monitor.current_price or 0,
                    delta_pct=diff_pct,
                    market_id=market.condition_id,
                    market_question=f"STRADDLE {side}: {market.question}",
                    odds_yes=market.up_price,
                    odds_no=market.down_price,
                    direction=side,
                    bet_size=cost,
                    fill_price=price,
                    order_id=res.order_id or "",
                    bankroll_before=bankroll,
                    edge=0.0,
                    confidence=0.0,
                )
                # Store trade_id so _monitor_straddle can update outcomes
                tid_key = f"{side.lower()}_trade_id"
                self._straddle_active[tid_key] = trade_id

    async def _monitor_straddle(self):
        """Check straddle order status -- cancel unfilled orders at window end."""
        straddle = self._straddle_active
        if not straddle:
            return

        seconds_left = self.price_monitor.seconds_left_in_window()

        # Check fill status
        for side, key in [("UP", "up_order_id"), ("DOWN", "down_order_id")]:
            oid = straddle.get(key)
            filled_key = f"{side.lower()}_filled"
            if oid and not straddle.get(filled_key):
                try:
                    status = self.poly_client.check_position_status(oid)
                    if status.get("filled", 0) > 0:
                        straddle[filled_key] = True
                        logger.info(f"[SWAP] Straddle {side} FILLED! ({straddle['shares']} shares @ ${straddle['price']:.2f})")
                except Exception as e:
                    logger.warning(f"[SWAP] Straddle {side} status check error: {e}")

        up_filled = straddle.get("up_filled", False)
        down_filled = straddle.get("down_filled", False)

        if up_filled and down_filled:
            cost = straddle["shares"] * straddle["price"] * 2
            payout = straddle["shares"] * 1.0  # winning side pays $1/share
            net = payout - cost
            logger.info(
                f"? STRADDLE BOTH SIDES FILLED! Cost=${cost:.2f} -> "
                f"Guaranteed payout=${payout:.2f} -> Net +${net:.2f}"
            )
            self.position_manager.update_after_win(net)
            # Update both trade outcomes in the DB
            for s in ("up", "down"):
                tid = straddle.get(f"{s}_trade_id")
                if tid:
                    self.trade_logger.update_trade_outcome(
                        tid, "WIN", net / 2,
                        self.position_manager.bankroll,
                        self.position_manager.consecutive_wins,
                    )
            self._straddle_active = None
            return

        # Cancel remaining orders near window end
        if seconds_left <= 5:
            for side, key in [("UP", "up_order_id"), ("DOWN", "down_order_id")]:
                filled_key = f"{side.lower()}_filled"
                oid = straddle.get(key)
                if oid and not straddle.get(filled_key):
                    self.poly_client.cancel_order(oid)
                    logger.info(f"[SWAP] Straddle {side} cancelled (unfilled, window ending)")
                    # Mark unfilled trade as CANCELLED
                    tid = straddle.get(f"{side.lower()}_trade_id")
                    if tid:
                        self.trade_logger.update_trade_outcome(
                            tid, "CANCELLED", 0.0,
                            self.position_manager.bankroll,
                            self.position_manager.consecutive_wins,
                        )

            # Resolve: if one side filled, it becomes a directional bet
            if up_filled or down_filled:
                filled_side = "UP" if up_filled else "DOWN"
                cost = straddle["shares"] * straddle["price"]
                logger.info(
                    f"[SWAP] Straddle single fill: {filled_side} only (cost=${cost:.2f}) -> "
                    f"awaiting resolution as directional bet"
                )
            else:
                logger.info("[SWAP] Straddle: no fills, both cancelled -- no cost")

            self._straddle_active = None

    # -- Dual-side limit-order strategy --------------------------------

    async def _check_dual_side_entry(self):
        """Place limit BUY orders on BOTH Up and Down at prices that
        guarantee profit if both fill.  Hold to resolution -- never sell.

        Budget is split evenly: e.g. $40 total -> $20 on Up + $20 on Down.
        At max_price=0.47 per share, each side gets ~42 shares.
        If both fill, cost=$40, one side pays $1/share = $42 -> +$2 guaranteed.
        If only one fills, ride to resolution (50/50 on the window outcome).
        """
        if not self._dual_enabled or not self.poly_client:
            return
        if self._dual_active is not None:
            return  # already have active dual orders this window

        btc_price = self.price_monitor.current_price
        if btc_price is None:
            return

        seconds_left = self.price_monitor.seconds_left_in_window()
        if seconds_left < self._dual_min_secs_to_place:
            return  # too late in the window to place

        window_start_ts = (int(time.time()) // 300) * 300
        window_slug = f"btc-updown-5m-{window_start_ts}"

        if window_slug in self._dual_window_slugs:
            return  # already placed for this window

        # Evict stale slugs (older than 10 min)
        cutoff = window_start_ts - 600
        self._dual_window_slugs = {
            s for s in self._dual_window_slugs
            if int(s.rsplit("-", 1)[-1]) > cutoff
        }

        # Fetch market
        loop = asyncio.get_event_loop()
        market = await loop.run_in_executor(
            None, PolymarketClient._fetch_market_by_slug, window_slug
        )
        if not market:
            return

        # Register for WS
        self._register_market_for_ws(window_slug, market)

        # Check bankroll
        can_trade, reason = self.position_manager.can_trade()
        if not can_trade:
            logger.info(f"[DUAL] Blocked: {reason}")
            return

        ok, reason = self._check_balance_before_trade(self._dual_budget)
        if not ok:
            logger.warning(f"[DUAL] Blocked: {reason}")
            return

        price = self._dual_max_price
        half_budget = self._dual_budget / 2.0
        shares_per_side = int(half_budget / price)  # whole shares

        if shares_per_side < 1:
            logger.warning(f"[DUAL] Budget too small for price ${price:.2f}")
            return

        actual_cost_per_side = shares_per_side * price
        total_cost = actual_cost_per_side * 2
        guaranteed_payout = shares_per_side * 1.0  # winning side pays $1/share
        guaranteed_profit = guaranteed_payout - total_cost

        logger.info(
            f"[DUAL] Placing both-sides order for {window_slug} | "
            f"{shares_per_side} shares each side @ ${price:.2f} | "
            f"cost=${total_cost:.2f} -> if both fill: guaranteed +${guaranteed_profit:.2f} "
            f"({guaranteed_profit / total_cost * 100:.1f}% ROI)"
        )

        up_result = self.poly_client.place_limit_order(market, "UP", shares_per_side, price)
        down_result = self.poly_client.place_limit_order(market, "DOWN", shares_per_side, price)

        self._dual_active = {
            "slug": window_slug,
            "price": price,
            "shares": shares_per_side,
            "total_cost": total_cost,
            "up_order_id": up_result.order_id if up_result.success else None,
            "down_order_id": down_result.order_id if down_result.success else None,
            "up_filled": False,
            "down_filled": False,
            "both_locked": False,
            "placed_at": time.time(),
        }
        self._dual_window_slugs.add(window_slug)

        bankroll = self.position_manager.bankroll
        for side, res in [("UP", up_result), ("DOWN", down_result)]:
            if res.success:
                cost = shares_per_side * price
                trade_id = self.trade_logger.log_trade(
                    btc_price_start=btc_price,
                    btc_price_end=self.price_monitor.current_price or 0,
                    delta_pct=0.0,
                    market_id=market.condition_id,
                    market_question=f"DUAL {side}: {market.question}",
                    odds_yes=market.up_price,
                    odds_no=market.down_price,
                    direction=side,
                    bet_size=cost,
                    fill_price=price,
                    order_id=res.order_id or "",
                    bankroll_before=bankroll,
                    edge=0.0,
                    confidence=0.0,
                )
                self._dual_active[f"{side.lower()}_trade_id"] = trade_id
                logger.info(f"  [OK] DUAL {side}: {shares_per_side} shares @ ${price:.2f} -> {res.order_id}")
            else:
                logger.warning(f"  [X] DUAL {side} failed: {res.error}")

    async def _monitor_dual_side(self):
        """Monitor dual-side order fills.  If both fill -> guaranteed profit.
        Cancel unfilled orders near window end.  Hold filled positions to resolution."""
        dual = self._dual_active
        if not dual:
            return

        # Skip polling if both sides already locked in
        if dual.get("both_locked"):
            seconds_left = self.price_monitor.seconds_left_in_window()
            if seconds_left <= self._dual_cancel_at:
                self._resolve_dual_side_at_end(dual)
            return

        seconds_left = self.price_monitor.seconds_left_in_window()

        # Check fill status for each side
        for side, key in [("UP", "up_order_id"), ("DOWN", "down_order_id")]:
            oid = dual.get(key)
            filled_key = f"{side.lower()}_filled"
            if oid and not dual.get(filled_key):
                try:
                    status = self.poly_client.check_position_status(oid)
                    if status.get("filled", 0) > 0:
                        dual[filled_key] = True
                        cost = dual["shares"] * dual["price"]
                        logger.info(
                            f"[DUAL] {side} FILLED! ({dual['shares']} shares @ ${dual['price']:.2f}, cost=${cost:.2f})"
                        )
                except Exception as e:
                    logger.warning(f"[DUAL] {side} status check error: {e}")

        up_filled = dual.get("up_filled", False)
        down_filled = dual.get("down_filled", False)

        # Both filled -> guaranteed profit, just wait for resolution
        if up_filled and down_filled and not dual.get("both_locked"):
            cost = dual["total_cost"]
            payout = dual["shares"] * 1.0
            profit = payout - cost
            logger.info(
                f"[DUAL] BOTH SIDES FILLED! Cost=${cost:.2f} | "
                f"Guaranteed payout=${payout:.2f} | Profit=+${profit:.2f} -- holding to resolution"
            )
            dual["both_locked"] = True

        # Cancel unfilled orders and resolve near window end
        if seconds_left <= self._dual_cancel_at:
            self._resolve_dual_side_at_end(dual)

    def _resolve_dual_side_at_end(self, dual: dict):
        """Clean up dual-side orders at window end."""
        up_filled = dual.get("up_filled", False)
        down_filled = dual.get("down_filled", False)

        # Cancel any unfilled orders
        for side, key in [("UP", "up_order_id"), ("DOWN", "down_order_id")]:
            filled_key = f"{side.lower()}_filled"
            oid = dual.get(key)
            if oid and not dual.get(filled_key):
                try:
                    self.poly_client.cancel_order(oid)
                    logger.info(f"[DUAL] {side} cancelled (unfilled, window ending)")
                except Exception:
                    pass
                tid = dual.get(f"{side.lower()}_trade_id")
                if tid:
                    self.trade_logger.update_trade_outcome(
                        tid, "CANCELLED", 0.0,
                        self.position_manager.bankroll,
                        self.position_manager.consecutive_wins,
                    )

        if up_filled and down_filled:
            cost = dual["total_cost"]
            payout = dual["shares"] * 1.0
            profit = payout - cost
            self.position_manager.update_after_win(profit)
            for s in ("up", "down"):
                tid = dual.get(f"{s}_trade_id")
                if tid:
                    self.trade_logger.update_trade_outcome(
                        tid, "WIN", profit / 2,
                        self.position_manager.bankroll,
                        self.position_manager.consecutive_wins,
                    )
            logger.info(f"[DUAL] Both-sides WIN locked in: +${profit:.2f}")
        elif up_filled or down_filled:
            filled_side = "UP" if up_filled else "DOWN"
            cost = dual["shares"] * dual["price"]
            logger.info(
                f"[DUAL] Single fill: {filled_side} (cost=${cost:.2f}) -- "
                f"holding to resolution as directional bet"
            )
        else:
            logger.info("[DUAL] No fills on either side -- no cost")

        self._dual_active = None

    # -- Background resolution (runs as asyncio task) -------------

    async def _resolve_trade_background(self, pos: dict):
        """Determine win/loss for a finished trade. Runs in the background
        so the main loop can keep trading the next window."""
        slug = pos["slug"]
        direction = pos["direction"]
        entry_price = pos["entry_price"]
        bet_size = pos["bet_size"]
        trade_id = pos["trade_id"]
        order_id = pos.get("order_id", "")
        btc_at_close = pos.get("btc_at_close")  # Binance price captured at window end

        # Verify the order actually filled before resolving.
        # If size_matched == 0 on the exchange, it never filled.
        if order_id and self.poly_client:
            try:
                status = self.poly_client.check_position_status(order_id)
                matched = status.get("filled", 0)
                if matched == 0:
                    logger.warning(
                        f"[!] Order {order_id} for [{slug}] never filled -- "
                        f"marking CANCELLED (no cost)"
                    )
                    self.trade_logger.update_trade_outcome(
                        trade_id, "CANCELLED", 0.0,
                        self.position_manager.bankroll,
                        self.position_manager.consecutive_wins,
                    )
                    return
            except Exception as e:
                logger.warning(f"Could not verify fill for {order_id}: {e}")

        # Give Polymarket a moment to publish close data
        await asyncio.sleep(15)

        win = None
        close_price = None
        target = PolymarketClient.fetch_price_to_beat(slug)

        for attempt in range(12):  # retry every 15s for up to ~3 minutes
            close_price = PolymarketClient.fetch_close_price(slug)
            if close_price is not None and target is not None:
                win = (close_price > target) if direction == "UP" else (close_price < target)
                logger.info(
                    f"? Resolution [{slug}]: close=${close_price:,.2f} vs "
                    f"target=${target:,.2f} -> {'UP wins' if close_price > target else 'DOWN wins'}"
                )
                break

            # Fallback: check if gamma-api market prices have settled
            resolved = PolymarketClient._fetch_market_by_slug(slug)
            if resolved:
                if direction == "UP" and resolved.up_price > 0.90:
                    win = True
                    break
                elif direction == "DOWN" and resolved.down_price > 0.90:
                    win = True
                    break
                elif direction == "UP" and resolved.down_price > 0.90:
                    win = False
                    break
                elif direction == "DOWN" and resolved.up_price > 0.90:
                    win = False
                    break

            logger.info(f"Waiting for resolution of {slug} (attempt {attempt + 1}/12)...")
            await asyncio.sleep(15)

        if win is None:
            # Fallback: use the BTC price we captured at window end
            # This is more accurate than current price (which may be from a later window)
            btc_fallback = btc_at_close or self.price_monitor.current_price
            if btc_fallback and target:
                win = (btc_fallback > target) if direction == "UP" else (btc_fallback < target)
                logger.info(
                    f"Using Binance price fallback for {slug}: "
                    f"${btc_fallback:,.2f} vs ${target:,.2f} "
                    f"({'captured at close' if btc_at_close else 'current'})"
                )
            else:
                logger.warning(f"Cannot resolve {slug} -- marking as LOSS")
                win = False

        # Determine the actual close price to record
        actual_close = close_price or btc_at_close

        # Feed outcome to adaptive learner
        trade_diff_pct = pos.get("diff_pct", 0.0)
        trade_edge = pos.get("edge", 0.0)
        trade_secs = pos.get("seconds_left", 150.0)
        trade_prev_diff = pos.get("prev_diff_pct")

        # -- Hedged position: guaranteed profit either way --
        if pos.get("hedge_placed"):
            hedge_cost = pos.get("hedge_cost", 0)
            hedge_shares = pos.get("hedge_shares", 0)
            hedge_side = pos.get("hedge_side", "")

            # The winning side pays $1/share. We hold shares in BOTH sides.
            # One side wins, one loses. Net = winning_payout - total_cost.
            # For a FULL hedge (hedge_shares == our shares):
            #   payout = shares * $1.00 = bet_size / entry_price
            #   cost   = bet_size + hedge_cost
            #   profit = payout - cost
            shares = bet_size / entry_price
            payout = hedge_shares * 1.0  # insured shares always pay $1
            unhedged_shares = shares - hedge_shares

            if win:
                # Our original side won -- unhedged shares also pay out
                unhedged_payout = unhedged_shares * 1.0  # they also win
                total_payout = payout + unhedged_payout
            else:
                # Our original side lost -- only hedged shares pay out
                total_payout = payout
                # Unhedged shares are worthless

            total_cost = bet_size + hedge_cost
            profit = total_payout - total_cost

            self.position_manager.update_after_win(profit)
            self.trade_logger.update_trade_outcome(
                trade_id, "WIN", profit,
                self.position_manager.bankroll,
                self.position_manager.consecutive_wins,
                btc_price_close=actual_close,
            )
            self.learner.record_outcome(
                won=True, pnl=profit, diff_pct=trade_diff_pct,
                edge=trade_edge, seconds_left=trade_secs,
                prev_diff_pct=trade_prev_diff,
            )
            result_emoji = "[LOCK]" if not win else "[LOCK][OK]"
            logger.info(
                f"{result_emoji} HEDGED {'WIN' if win else 'SAVED'} "
                f"+${profit:.2f} [{slug}] | "
                f"{'Original side won' if win else 'Hedge insurance paid out!'} | "
                f"Payout ${total_payout:.2f} - Cost ${total_cost:.2f} | "
                f"Bankroll: ${self.position_manager.bankroll:.2f}"
            )
            return

        if win:
            profit = bet_size * ((1.0 / entry_price) - 1)
            self._record_trade_outcome(pos, won=True, pnl=profit, btc_now=actual_close)
            logger.info(
                f"[OK] WIN +${profit:.2f} [{slug}] | "
                f"Bankroll: ${self.position_manager.bankroll:.2f}"
            )
        else:
            self._record_trade_outcome(pos, won=False, pnl=-bet_size, btc_now=actual_close)
            logger.info(
                f"[X] LOSS -${bet_size:.2f} [{slug}] | "
                f"Bankroll: ${self.position_manager.bankroll:.2f}"
            )

    def _print_session_summary(self):
        """Print end-of-session summary."""
        stats = self.trade_logger.get_session_stats()
        lifetime = self.trade_logger.get_session_stats(all_sessions=True)
        state = self.position_manager.get_state()

        logger.info("=" * 50)
        logger.info(f"SESSION SUMMARY (session {self.trade_logger.session_id})")
        logger.info("=" * 50)
        logger.info(f"This session trades: {stats.get('total_trades', 0)}")
        logger.info(f"Wins: {stats.get('wins', 0)} | Losses: {stats.get('losses', 0)}")
        logger.info(f"Win rate: {stats.get('win_rate', 0):.1%}")
        logger.info(f"Session P&L: ${stats.get('total_pnl', 0):.2f}")
        logger.info(f"Final bankroll: ${state.bankroll:.2f}")
        logger.info("-" * 50)

        # Smart compounding summary
        if self.position_manager.compound_enabled:
            cs = self.position_manager.get_compounding_summary()
            logger.info(f"COMPOUNDING: Gear={cs['gear']} | "
                        f"Floor=${cs['compound_floor']:.2f} | "
                        f"HWM=${cs['high_water_mark']:.2f} | "
                        f"Drawdown={cs['drawdown_pct']:.1%}")
            logger.info(f"  Reinvest rate: {cs['reinvest_pct']:.0%} | "
                        f"Protected profit: ${cs['protected_profit']:.2f} | "
                        f"Streak bonus: {'YES' if cs['streak_bonus_active'] else 'no'}")
            logger.info("-" * 50)

        logger.info(f"LIFETIME: {lifetime.get('total_trades', 0)} trades | "
                    f"P&L: ${lifetime.get('total_pnl', 0):.2f} | "
                    f"Win rate: {lifetime.get('win_rate', 0):.1%}")

        # WebSocket stream health
        mkt_ws = self._market_stream.get_stats()
        logger.info(f"WS MARKET: {'connected' if mkt_ws['connected'] else 'disconnected'} | "
                    f"msgs={mkt_ws['total_messages']} | "
                    f"tokens={mkt_ws['subscribed_tokens']}")
        if self._user_stream:
            usr_ws = self._user_stream.get_stats()
            logger.info(f"WS USER: {'connected' if usr_ws['connected'] else 'disconnected'} | "
                        f"msgs={usr_ws['total_messages']} | "
                        f"tracked_orders={usr_ws['tracked_orders']}")

        logger.info("=" * 50)


def handle_shutdown(signum, frame):
    global _shutdown
    logger.info("Shutdown signal received")
    _shutdown = True


def main():
    load_dotenv()

    # Parse CLI args
    config_path = "config.yaml"

    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]

    config = load_config(config_path)
    setup_logging(config["logging"]["log_level"])

    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    orchestrator = Orchestrator(config)

    logger.info("=" * 50)
    logger.info("POLYBOT - BTC 5-Min Arbitrage Bot")
    logger.info(f"Config: {config_path}")
    logger.info("=" * 50)
    logger.info("Create EMERGENCY_STOP file to halt")
    logger.info("Set TRADING_ENABLED=false in env to pause")
    logger.info("=" * 50)

    if not _acquire_pid_lock():
        old_pid = _PID_FILE.read_text().strip() if _PID_FILE.exists() else "?"
        logger.critical(
            f"Another Polybot instance is already running (PID {old_pid}). "
            f"Kill it first or delete {_PID_FILE} if the file is stale "
            f"(e.g., from a crash) and you're certain no other instance is running."
        )
        sys.exit(1)

    try:
        asyncio.run(orchestrator.run())
    finally:
        _release_pid_lock()


if __name__ == "__main__":
    main()
