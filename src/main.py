"""
Main orchestrator for the Polymarket 5-min BTC Up/Down bot.

Strategy:
  1. Stream real-time BTC price from Binance.
  2. Every scan, fetch the current 5-min market from Polymarket (gamma-api).
  3. Compare live BTC price against the market's "price to beat"
     (BTC price at the start of the 5-min window).
  4. If BTC is above the target → bet UP, below → bet DOWN.
  5. Only bet if the Polymarket odds offer value vs. our confidence.
  6. Wait for resolution, record result.
"""

import asyncio
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.price_monitor import PriceMonitor
from src.polymarket_client import PolymarketClient, MarketInfo
from src.arbitrage_engine import ArbitrageEngine
from src.position_manager import PositionManager
from src.trade_logger import TradeLogger
from src.adaptive_learner import AdaptiveLearner

logger = logging.getLogger("polybot")

# Globals for signal handling
_shutdown = False


def setup_logging(level: str = "INFO"):
    # Log to both console and a rotating file
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    # Rotating file handler — keeps history across restarts
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


class Orchestrator:
    def __init__(self, config: dict, paper_trade: bool = True):
        self.config = config
        self.paper_trade = paper_trade

        # Initialize components
        self.price_monitor = PriceMonitor(config)
        self.arbitrage_engine = ArbitrageEngine(config)
        self.position_manager = PositionManager(config)
        self.trade_logger = TradeLogger(config, paper_mode=paper_trade)
        self.learner = AdaptiveLearner(config)

        # Restore bankroll and stats from DB (survives restarts)
        self.position_manager.load_state_from_db(self.trade_logger)

        # ── Straddle strategy state ──
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

        # Polymarket client (only needed for live trading)
        self.poly_client: PolymarketClient | None = None
        if not paper_trade:
            api_key = os.getenv("POLYMARKET_API_KEY", "")
            api_secret = os.getenv("POLYMARKET_API_SECRET", "")
            passphrase = os.getenv("POLYMARKET_PASSPHRASE", "")
            private_key = os.getenv("POLYMARKET_PRIVATE_KEY", "")
            if not api_key or not api_secret or not passphrase:
                raise ValueError(
                    "POLYMARKET_API_KEY, POLYMARKET_API_SECRET, and "
                    "POLYMARKET_PASSPHRASE are required for live trading"
                )
            if not private_key:
                logger.warning(
                    "POLYMARKET_PRIVATE_KEY not set — order signing will fail. "
                    "Add your Polygon wallet private key to .env for live trades."
                )
            self.poly_client = PolymarketClient(
                config, api_key, api_secret, passphrase, private_key
            )

        self._scan_interval = 3  # seconds between scans (fast for 5-min markets)

        # ── Scalp re-entry config ──
        scalp_cfg = config.get("scalp", {})
        self._scalp_enabled = scalp_cfg.get("enabled", False)
        self._scalp_max_entries = scalp_cfg.get("max_entries_per_window", 3)
        self._scalp_cooldown_tp = scalp_cfg.get("cooldown_after_take_profit", 15)
        self._scalp_cooldown_sl = scalp_cfg.get("cooldown_after_stop_loss", 30)

        # ── Hedge lock-in config ──
        hedge_cfg = config.get("hedge", {})
        self._hedge_enabled = hedge_cfg.get("enabled", False)
        self._hedge_min_prob = hedge_cfg.get("min_prob", 0.80)
        self._hedge_min_profit_pct = hedge_cfg.get("min_profit_pct", 0.10)
        self._hedge_max_cost = hedge_cfg.get("max_hedge_cost", 5.0)
        self._hedge_min_secs = hedge_cfg.get("min_seconds_left", 30)

        # ── Momentum pre-position config ──
        prepos_cfg = config.get("preposition", {})
        self._prepos_enabled = prepos_cfg.get("enabled", False)
        self._prepos_trigger_secs = prepos_cfg.get("trigger_seconds", 30)
        self._prepos_min_momentum = prepos_cfg.get("min_momentum_pct", 0.04)
        self._prepos_min_accel = prepos_cfg.get("min_accel_pct", 0.01)
        self._prepos_max_price = prepos_cfg.get("max_price", 0.55)
        self._prepos_bet_fraction = prepos_cfg.get("bet_fraction", 0.5)
        self._prepos_pending: dict | None = None  # tracks pre-positioned order
        self._prepos_window_slugs: set[str] = set()  # next-window slugs we already pre-positioned

        # ── Position tracking for continuous trading ──
        self._open_position: dict | None = None  # currently held position
        self._background_tasks: list[asyncio.Task] = []  # resolution tasks
        self._traded_window_slugs: set[str] = set()  # windows we already traded
        self._window_entry_counts: dict[str, int] = {}  # slug → entries this window
        self._reentry_cooldown_until: float = 0.0  # timestamp when cooldown expires

    async def run(self):
        """Main event loop — fully non-blocking continuous trading."""
        global _shutdown

        logger.info(f"Starting Polybot ({'PAPER' if self.paper_trade else 'LIVE'} mode)")
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

        logger.info("Price history ready — starting continuous scan loop")

        # Main scan loop — every cycle: monitor position → look for new entry
        try:
            while not _shutdown:
                if check_emergency_stop():
                    logger.warning("EMERGENCY STOP detected, shutting down")
                    break

                if not check_trading_enabled():
                    logger.info("Trading disabled, waiting...")
                    await asyncio.sleep(10)
                    continue

                # 1) Monitor open position (take-profit / window-end)
                await self._monitor_open_position()

                # 2) Monitor active straddle orders
                if self._straddle_active:
                    await self._monitor_straddle()

                # 3) Monitor / promote pre-positioned orders
                if self._prepos_pending:
                    await self._monitor_preposition()

                # 4) Look for new entry if we have no open position
                if self._open_position is None:
                    await self._scan_for_entry()

                # 5) Check for momentum pre-position on next window
                if (
                    self._prepos_enabled
                    and not self.paper_trade
                    and self.poly_client
                    and self._prepos_pending is None
                ):
                    await self._check_preposition()

                # 4) Reap finished background tasks
                self._background_tasks = [
                    t for t in self._background_tasks if not t.done()
                ]

                await asyncio.sleep(self._scan_interval)

        except asyncio.CancelledError:
            logger.info("Orchestrator cancelled")
        finally:
            # Cancel any pending resolution tasks
            for task in self._background_tasks:
                task.cancel()
            await self.price_monitor.stop()
            price_task.cancel()
            self.trade_logger.end_session(self.position_manager.bankroll)
            self._print_session_summary()

    # ── Orphaned trade resolution (runs once on startup) ──────────

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

        logger.info(f"Found {len(orphans)} orphaned PENDING trade(s) — resolving...")

        for orph in orphans:
            tid = orph["trade_id"]
            direction = orph["direction"]
            bet_size = orph["bet_size"]
            fill_price = orph["fill_price"]
            target = orph["btc_price_start"]  # the Chainlink open price
            orph_order_id = orph.get("order_id", "")

            # First, check if the order actually filled on the exchange.
            # If size_matched == 0, the order never filled — mark CANCELLED.
            if orph_order_id and orph_order_id != "paper" and self.poly_client:
                try:
                    status = self.poly_client.check_position_status(orph_order_id)
                    matched = status.get("filled", 0)
                    order_status = status.get("status", "")
                    if matched == 0:
                        logger.info(
                            f"  Orphan #{tid} order never filled "
                            f"(status={order_status}) — marking CANCELLED"
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
            # the trade's timestamp — round down to the 5-min boundary.
            import re as _re
            trade_ts = orph["timestamp"]
            if trade_ts:
                # SQLite datetimes are naive — ensure UTC-aware
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
                    f"  Chainlink: close=${close_price:,.2f} vs open=${open_price:,.2f} → {result_str}"
                )
            else:
                # Last resort: use the Binance price that was captured at entry
                btc_end = orph.get("btc_price_end")
                if btc_end and open_price:
                    win = (btc_end > open_price) if direction == "UP" else (btc_end < open_price)
                    logger.warning(
                        f"  No Chainlink close for {slug}, using entry Binance ${btc_end:,.2f} — unreliable"
                    )
                else:
                    win = False
                    logger.warning(f"  Cannot resolve orphan #{tid} — marking as LOSS")

            if win:
                profit = bet_size * ((1.0 / fill_price) - 1)
                self.position_manager.update_after_win(profit)
                self.trade_logger.update_trade_outcome(
                    tid, "WIN", profit,
                    self.position_manager.bankroll,
                    self.position_manager.consecutive_wins,
                    btc_price_close=close_price,
                )
                logger.info(f"  ✅ Orphan #{tid} resolved: WIN +${profit:.2f}")
            else:
                self.position_manager.update_after_loss(bet_size)
                self.trade_logger.update_trade_outcome(
                    tid, "LOSS", -bet_size,
                    self.position_manager.bankroll,
                    self.position_manager.consecutive_wins,
                    btc_price_close=close_price,
                )
                logger.info(f"  ❌ Orphan #{tid} resolved: LOSS -${bet_size:.2f}")

        logger.info(f"Orphan resolution complete. Bankroll: ${self.position_manager.bankroll:.2f}")

    # ── Position monitoring (called every scan cycle) ──────────────

    async def _monitor_open_position(self):
        """Check the open position for take-profit, stop-loss, or window expiry.

        Non-blocking: runs once per scan and returns immediately.

        Decision logic:
          - STOP-LOSS: If BTC reverses and our probability drops below the
            stop_loss threshold, close immediately to limit damage.
          - TAKE-PROFIT: If our probability is very high (≥ take_profit),
            and we're past the midpoint of the window, hold for resolution
            instead of trying to sell on the thin CLOB book.  In paper mode
            we credit a conservative 90% of full-win value.  In live mode
            we just hold — the market will resolve in our favor.
          - WINDOW END: When the window expires, send to background
            resolution (the normal path for profitable holds).
        """
        pos = self._open_position
        if pos is None:
            return

        btc_now = self.price_monitor.current_price
        if btc_now is None:
            return

        secs_left = self.price_monitor.seconds_left_in_window()
        target_price = pos["target_price"]
        diff_now = ((btc_now - target_price) / target_price) * 100
        direction = pos["direction"]

        our_prob = self.arbitrage_engine._estimate_probability(
            diff_now if direction == "UP" else -diff_now,
            secs_left,
        )

        logger.info(
            f"📈 Position: {direction} | "
            f"BTC=${btc_now:,.2f} vs target=${target_price:,.2f} "
            f"({diff_now:+.3f}%) | prob={our_prob:.3f} | {secs_left:.0f}s left"
        )

        # ── Check if limit sell order filled ──
        limit_sell_id = pos.get("limit_sell_order_id")
        if limit_sell_id and not pos.get("paper") and self.poly_client:
            try:
                sell_status = self.poly_client.check_position_status(limit_sell_id)
                sell_filled = sell_status.get("filled", 0)
                sell_order_status = sell_status.get("status", "")
                if sell_filled > 0:
                    # Limit sell filled — book the profit!
                    entry_price = pos["entry_price"]
                    bet_size = pos["bet_size"]
                    shares = bet_size / entry_price
                    limit_sell_pct = self.config["strategy"].get("limit_sell_profit", 0.10)
                    sell_price = round(entry_price * (1.0 + limit_sell_pct), 2)
                    proceeds = shares * sell_price
                    profit = proceeds - bet_size

                    self.position_manager.update_after_win(profit)
                    self.trade_logger.update_trade_outcome(
                        pos["trade_id"], "WIN", profit,
                        self.position_manager.bankroll,
                        self.position_manager.consecutive_wins,
                        btc_price_close=btc_now,
                    )

                    entries = self._window_entry_counts.get(pos['slug'], 1)
                    can_reenter = (
                        self._scalp_enabled
                        and entries < self._scalp_max_entries
                        and secs_left > 30
                    )

                    logger.info(
                        f"✅ LIMIT SELL FILLED +${profit:.2f} [{pos['slug']}] | "
                        f"Sold {shares:.1f} shares @ {sell_price:.3f} "
                        f"(entry {entry_price:.3f}) | "
                        f"Bankroll: ${self.position_manager.bankroll:.2f}"
                        + (f" | 🔄 Re-entry eligible ({self._scalp_cooldown_tp}s cooldown)" if can_reenter else "")
                    )

                    # Record learner outcome
                    self.learner.record_outcome(
                        won=True, pnl=profit,
                        diff_pct=pos.get("diff_pct", 0),
                        edge=pos.get("edge", 0),
                        seconds_left=pos.get("seconds_left", 0),
                        prev_diff_pct=pos.get("prev_diff_pct"),
                    )

                    self._open_position = None
                    if can_reenter:
                        self._reentry_cooldown_until = time.time() + self._scalp_cooldown_tp
                    return

                if sell_order_status.upper() in ("CANCELED", "CANCELLED",
                                                  "CANCELED_MARKET_RESOLVED"):
                    pos["limit_sell_order_id"] = None  # clear stale ref
            except Exception as e:
                logger.warning(f"Limit sell check error: {e}")

        # ── Hedge lock-in: buy insurance on opposite side ──
        # If we're winning big (high prob) and haven't hedged yet,
        # buy the opposite side cheap to guarantee profit either way.
        if (
            self._hedge_enabled
            and not pos.get("paper")
            and not pos.get("hedge_placed")
            and self.poly_client
            and our_prob >= self._hedge_min_prob
            and secs_left >= self._hedge_min_secs
        ):
            await self._execute_hedge_lockin(pos, our_prob, secs_left)

        stop_loss_threshold = self.config["strategy"].get("stop_loss", 0.30)
        take_profit_threshold = self.config["strategy"].get("take_profit", 0.85)

        # ── Stop-loss: BTC reversed hard against us ──
        if our_prob <= stop_loss_threshold:
            # Cancel any pending limit sell first
            if limit_sell_id and self.poly_client:
                try:
                    self.poly_client.cancel_order(limit_sell_id)
                    logger.info(f"Cancelled limit sell {limit_sell_id} (stop-loss)")
                except Exception:
                    pass

            entry_price = pos["entry_price"]
            bet_size = pos["bet_size"]
            shares = bet_size / entry_price

            # ── LIVE: sell shares on CLOB to recover partial value ──
            recovered = 0.0
            if not pos.get("paper") and self.poly_client:
                from src.polymarket_client import PolymarketClient
                slug = pos["slug"]
                try:
                    live = PolymarketClient.get_live_market_price(slug)
                    if live:
                        if direction == "UP":
                            sell_price = live.get("best_bid", 0)
                        else:
                            sell_price = round(1.0 - live.get("best_ask", 1.0), 2) if live.get("best_ask") else 0

                        if sell_price and sell_price > 0.01:
                            market = PolymarketClient._fetch_market_by_slug(slug)
                            if market:
                                # Drop price 2 ticks below bid for aggressive fill
                                aggressive_price = round(max(sell_price - 0.02, 0.01), 2)
                                logger.info(
                                    f"🛑 STOP-LOSS SELL: selling {shares:.1f} shares "
                                    f"@ {aggressive_price:.3f} (bid={sell_price:.3f}, entry {entry_price:.3f})"
                                )
                                result = self.poly_client.sell_position(
                                    market, direction, shares, aggressive_price,
                                    order_type="GTC",
                                )
                                if result.success and result.order_id:
                                    # Poll for fill (up to 4s)
                                    import asyncio as _aio
                                    sell_filled = False
                                    for _attempt in range(4):
                                        await _aio.sleep(1.0)
                                        try:
                                            _status = self.poly_client.check_position_status(result.order_id)
                                            if _status.get("filled", 0) > 0:
                                                sell_filled = True
                                                break
                                        except Exception:
                                            pass
                                    if sell_filled:
                                        recovered = shares * aggressive_price
                                        logger.info(
                                            f"✅ Stop-loss sell filled — recovered ${recovered:.2f} "
                                            f"of ${bet_size:.2f}"
                                        )
                                    else:
                                        # Cancel unfilled order to prevent stranded shares
                                        try:
                                            self.poly_client.cancel_order(result.order_id)
                                            logger.warning(f"Stop-loss sell not filled after 4s — cancelled to prevent stranded shares")
                                        except Exception:
                                            pass
                                else:
                                    logger.warning(f"Stop-loss sell failed: {result.error}")
                except Exception as e:
                    logger.warning(f"Stop-loss sell error: {e}")

            # Calculate actual loss (full bet minus whatever we recovered)
            loss = bet_size - recovered
            loss_pct = (loss / bet_size) * 100

            logger.info(
                f"🛑 STOP-LOSS: prob={our_prob:.3f} "
                f"(≤{stop_loss_threshold:.0%}) | "
                f"BTC ${btc_now:,.2f} ({diff_now:+.3f}%) | "
                f"Loss -${loss:.2f} ({loss_pct:.0f}% of bet)"
                + (f" | Recovered ${recovered:.2f} from sell" if recovered > 0 else " | No recovery (hold to resolution)")
                + f" | {secs_left:.0f}s left"
            )

            self.position_manager.update_after_loss(loss)
            self.trade_logger.update_trade_outcome(
                pos["trade_id"], "LOSS", -loss,
                self.position_manager.bankroll,
                self.position_manager.consecutive_wins,
                btc_price_close=btc_now,
            )

            # Record learner outcome
            self.learner.record_outcome(
                won=False, pnl=-loss,
                diff_pct=pos.get("diff_pct", 0),
                edge=pos.get("edge", 0),
                seconds_left=pos.get("seconds_left", 0),
                prev_diff_pct=pos.get("prev_diff_pct"),
            )

            entries = self._window_entry_counts.get(pos['slug'], 1)
            can_reenter = (
                self._scalp_enabled
                and entries < self._scalp_max_entries
                and secs_left > 45  # need enough time to recover
            )

            logger.info(
                f"❌ {'PAPER' if pos['paper'] else 'LIVE'} LOSS (stop-loss) "
                f"-${loss:.2f} | Bankroll: ${self.position_manager.bankroll:.2f}"
                + (f" | 🔄 Re-entry eligible ({self._scalp_cooldown_sl}s cooldown)" if can_reenter else "")
            )

            self._open_position = None
            if can_reenter:
                self._reentry_cooldown_until = time.time() + self._scalp_cooldown_sl
            return

        # ── Take-profit: sell shares to lock in profit ──
        # If hedge is active, skip take-profit selling — we're guaranteed profit at resolution
        if pos.get("hedge_placed"):
            hedge_profit = pos.get("hedge_shares", 0) * (1.0 - pos["entry_price"] - pos.get("hedge_price", 0))
            if secs_left > 10 and int(secs_left) % 15 < 4:  # log every ~15s
                logger.info(
                    f"🔒 HEDGED: {pos['direction']}@{pos['entry_price']:.3f} + "
                    f"{pos.get('hedge_side','')}@{pos.get('hedge_price',0):.3f} | "
                    f"Locked profit ${hedge_profit:.2f} | {secs_left:.0f}s left"
                )
            # Cancel the limit sell if we hedged — no need for it
            if limit_sell_id and self.poly_client:
                try:
                    self.poly_client.cancel_order(limit_sell_id)
                    pos["limit_sell_order_id"] = None
                    logger.info(f"Cancelled limit sell (hedge active, no longer needed)")
                except Exception:
                    pass
        elif limit_sell_id:
            # Limit sell is resting — just log and wait
            limit_sell_pct = self.config["strategy"].get("limit_sell_profit", 0.10)
            sell_target = round(pos["entry_price"] * (1.0 + limit_sell_pct), 2)
            if secs_left > 10:
                logger.info(
                    f"📋 Limit sell resting @ {sell_target:.3f} "
                    f"(entry {pos['entry_price']:.3f} + {limit_sell_pct:.0%}) | "
                    f"prob={our_prob:.3f} | {secs_left:.0f}s left"
                )
        elif our_prob >= take_profit_threshold and secs_left > 10 and not pos["paper"]:
            await self._execute_take_profit(pos, our_prob, secs_left, btc_now)
            return

        # Paper mode: credit profit estimate at take-profit
        if our_prob >= take_profit_threshold and secs_left > 10 and pos["paper"]:
            logger.info(
                f"💎 HOLD: prob={our_prob:.3f} ≥ {take_profit_threshold:.0%} | "
                f"Holding to resolution ({secs_left:.0f}s left)"
            )

        # ── Smart pre-expiry exit: trust CLOB market price, not our model ──
        # Polymarket resolves on Chainlink, not Binance.  If our limit sell
        # hasn't filled and the CLOB best_bid is below our entry, the market
        # knows something we don't — sell to recover partial value.
        if secs_left <= 30 and secs_left > 3 and not pos.get("paper") and self.poly_client:
            if not pos.get("_expiry_sell_attempted"):
                from src.polymarket_client import PolymarketClient
                slug = pos["slug"]
                entry_price = pos["entry_price"]
                bet_size = pos["bet_size"]
                shares = bet_size / entry_price
                try:
                    live = PolymarketClient.get_live_market_price(slug)
                    if live:
                        if direction == "UP":
                            market_bid = live.get("best_bid", 0)
                        else:
                            market_bid = round(1.0 - live.get("best_ask", 1.0), 2) if live.get("best_ask") else 0

                        # If market bid is below our entry, the market thinks we're losing
                        # OR if limit sell has been resting and we're < 20s from expiry, sell at bid
                        market_losing = market_bid and market_bid < entry_price
                        near_expiry_unsold = limit_sell_id and secs_left <= 20

                        if (market_losing or near_expiry_unsold) and market_bid and market_bid > 0.01:
                            pos["_expiry_sell_attempted"] = True

                            # Cancel limit sell first
                            if limit_sell_id:
                                try:
                                    self.poly_client.cancel_order(limit_sell_id)
                                    pos["limit_sell_order_id"] = None
                                except Exception:
                                    pass

                            market = PolymarketClient._fetch_market_by_slug(slug)
                            if market:
                                reason = "market losing" if market_losing else "unsold near expiry"
                                # Drop price 2 ticks below bid for aggressive fill on thin books
                                aggressive_price = round(max(market_bid - 0.02, 0.01), 2)
                                logger.info(
                                    f"⏱🛑 SMART EXIT ({reason}): "
                                    f"bid={market_bid:.3f} vs entry={entry_price:.3f} | "
                                    f"selling {shares:.1f} shares @ {aggressive_price:.3f} "
                                    f"({secs_left:.0f}s left)"
                                )
                                result = self.poly_client.sell_position(
                                    market, direction, shares, aggressive_price,
                                    order_type="GTC",
                                )
                                if result.success and result.order_id:
                                    # Poll for fill (up to 4s)
                                    sell_filled = False
                                    for _attempt in range(4):
                                        await asyncio.sleep(1.0)
                                        try:
                                            _status = self.poly_client.check_position_status(result.order_id)
                                            if _status.get("filled", 0) > 0:
                                                sell_filled = True
                                                break
                                        except Exception:
                                            pass
                                    if not sell_filled:
                                        # Cancel unfilled order to prevent stranded shares
                                        try:
                                            self.poly_client.cancel_order(result.order_id)
                                            logger.warning(f"Smart exit sell not filled after 4s — cancelled")
                                        except Exception:
                                            pass
                                        # Fall through to window-end resolution
                                    else:
                                        recovered = shares * aggressive_price
                                        if recovered >= bet_size:
                                            # Actually a profit
                                            profit = recovered - bet_size
                                            self.position_manager.update_after_win(profit)
                                            self.trade_logger.update_trade_outcome(
                                                pos["trade_id"], "WIN", profit,
                                                self.position_manager.bankroll,
                                                self.position_manager.consecutive_wins,
                                                btc_price_close=btc_now,
                                            )
                                            self.learner.record_outcome(
                                                won=True, pnl=profit,
                                                diff_pct=pos.get("diff_pct", 0),
                                                edge=pos.get("edge", 0),
                                                seconds_left=pos.get("seconds_left", 0),
                                                prev_diff_pct=pos.get("prev_diff_pct"),
                                            )
                                            logger.info(
                                                f"✅ SMART EXIT WIN +${profit:.2f} | "
                                                f"Sold @ {aggressive_price:.3f} (entry {entry_price:.3f}) | "
                                                f"Bankroll: ${self.position_manager.bankroll:.2f}"
                                            )
                                        else:
                                            loss = bet_size - recovered
                                            self.position_manager.update_after_loss(loss)
                                            self.trade_logger.update_trade_outcome(
                                                pos["trade_id"], "LOSS", -loss,
                                                self.position_manager.bankroll,
                                                self.position_manager.consecutive_wins,
                                                btc_price_close=btc_now,
                                            )
                                            self.learner.record_outcome(
                                                won=False, pnl=-loss,
                                                diff_pct=pos.get("diff_pct", 0),
                                                edge=pos.get("edge", 0),
                                                seconds_left=pos.get("seconds_left", 0),
                                                prev_diff_pct=pos.get("prev_diff_pct"),
                                            )
                                            logger.info(
                                                f"❌ SMART EXIT LOSS -${loss:.2f} "
                                                f"(recovered ${recovered:.2f} of ${bet_size:.2f}) | "
                                                f"Bankroll: ${self.position_manager.bankroll:.2f}"
                                            )
                                        self._open_position = None
                                        return
                                else:
                                    logger.warning(
                                        f"Smart exit sell failed: {result.error} — "
                                        f"will resolve normally"
                                    )
                except Exception as e:
                    logger.warning(f"Smart exit error: {e}")

        # ── Window ended — move to background resolution ──
        if secs_left <= 3:
            # Cancel any pending limit sell — resolution will handle payout
            if limit_sell_id and self.poly_client:
                try:
                    self.poly_client.cancel_order(limit_sell_id)
                    logger.info(f"Cancelled limit sell {limit_sell_id} (window end)")
                except Exception:
                    pass

            # Capture the Binance price NOW (at window end) for fallback resolution
            pos["btc_at_close"] = btc_now
            logger.info(
                f"⏱ Window ended for {pos['slug']} — sending to background resolution "
                f"(BTC at close: ${btc_now:,.2f})"
            )
            self._open_position = None
            task = asyncio.create_task(self._resolve_trade_background(pos))
            self._background_tasks.append(task)

    # ── Scan for new entry (called every scan cycle) ─────────────

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
        # Pass the bot's own realized P&L from its trade records —
        # this is independent of the account balance, so the user's
        # manual trades don't affect the max-loss check.
        bot_pnl = self.trade_logger.get_bot_live_pnl() if not self.paper_trade else 0.0
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

        # ── Scalp re-entry gate ──
        entries_this_window = self._window_entry_counts.get(window_slug, 0)
        if entries_this_window > 0:
            if not self._scalp_enabled:
                return  # scalp disabled — strict one-trade-per-window
            if entries_this_window >= self._scalp_max_entries:
                return  # hit cap for this window
            if time.time() < self._reentry_cooldown_until:
                return  # still in cooldown after last exit
            logger.info(
                f"🔄 Scalp re-entry scan #{entries_this_window + 1}/{self._scalp_max_entries} "
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

        # Fetch market info and price-to-beat concurrently.
        # These are independent HTTP calls — run them in parallel to
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

        if target_price is None:
            logger.info(f"Could not fetch price to beat for {window_slug}")
            return

        # Analyze: BTC above or below target? Are odds favorable?
        decision = self.arbitrage_engine.analyze_opportunity(
            btc_price, target_price, market, seconds_left
        )

        diff_pct = ((btc_price - target_price) / target_price) * 100

        # Let the learner adjust probability estimate if it has data
        if decision.should_trade:
            naive_prob = decision.price + decision.edge
            adj_prob = self.learner.get_adjusted_probability(
                naive_prob, diff_pct, decision.edge,
                seconds_left, self._prev_diff_pct,
            )
            # Recompute edge with adjusted probability
            if adj_prob != naive_prob:
                decision.edge = adj_prob - decision.price
                decision.confidence = min(abs(diff_pct) / 0.10, 1.0)
                if decision.edge < 0:
                    decision.should_trade = False
                    decision.reason = f"Learner adjusted prob {naive_prob:.3f}→{adj_prob:.3f}, edge now negative"

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
            # ── Straddle check: even if directional trade is rejected,
            #    check if conditions are right for a straddle ──
            if (
                self._straddle_enabled
                and not self.paper_trade
                and self.poly_client
                and seconds_left <= self._straddle_trigger_secs
                and abs(diff_pct) <= self._straddle_max_diff
                and window_slug not in self._straddle_window_slugs
                and self._straddle_active is None
            ):
                await self._place_straddle(market, window_slug, target_price, diff_pct, seconds_left)
            self._prev_diff_pct = diff_pct
            return

        # ── Adaptive learner checks ──
        skip, skip_reason = self.learner.should_skip_conditions(
            diff_pct, decision.edge, seconds_left, self._prev_diff_pct
        )
        if skip:
            logger.info(f"🧠 {skip_reason}")
            self._prev_diff_pct = diff_pct
            return

        adj_min_edge = self.learner.get_adjusted_min_edge(
            diff_pct, decision.edge, seconds_left, self._prev_diff_pct
        )
        if decision.edge < adj_min_edge:
            logger.info(
                f"🧠 Learner raised min edge to {adj_min_edge:.4f} "
                f"(from {self.config['strategy'].get('min_edge', 0.01):.4f}) "
                f"for these conditions — edge {decision.edge:.4f} insufficient"
            )
            self._prev_diff_pct = diff_pct
            return

        logger.info(
            f"📊 BTC ${btc_price:,.2f} vs target ${target_price:,.2f} "
            f"({diff_pct:+.4f}%) → {decision.direction} | "
            f"Market: Up={market.up_price:.3f} Down={market.down_price:.3f} | "
            f"Edge={decision.edge:.4f} | {seconds_left:.0f}s left"
        )

        if self.paper_trade:
            await self._paper_trade(decision, market, btc_price, target_price, diff_pct, seconds_left)
        else:
            await self._live_trade(decision, market, btc_price, target_price, diff_pct, seconds_left)

        self._prev_diff_pct = diff_pct

    # ── Balance safety ────────────────────────────────────────────

    def _sync_bankroll_with_exchange(self, is_startup: bool = False):
        """Sync exchange balance for solvency checks.

        The position manager's ``bankroll`` is the **bot's own**
        accounting — it starts at ``initial_bet`` and grows/shrinks
        with the bot's realised P&L.  We do NOT overwrite it with the
        wallet balance because the user trades manually from the same
        wallet.

        Instead we cache the wallet balance separately for pre-trade
        solvency checks (``_check_balance_before_trade``).
        """
        if not self.poly_client:
            return
        live = self.poly_client.get_live_balance()
        if live is None:
            logger.warning("⚠️  Could not fetch live balance — keeping internal bankroll")
            return
        self._cached_real_balance = live
        self._last_balance_sync = time.time()
        bot_pnl = self.trade_logger.get_bot_live_pnl() if not self.paper_trade else 0.0
        logger.info(
            f"💰 Wallet: ${live:.2f} | Bot P&L: ${bot_pnl:+.2f} | "
            f"Bot bankroll: ${self.position_manager.bankroll:.2f} | "
            f"Max loss limit: ${self.position_manager.max_loss:.2f} | "
            f"Remaining: ${self.position_manager.max_loss + bot_pnl:.2f}"
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
        """Get the real exchange balance, with caching (refresh every 30s)."""
        now = time.time()
        if (
            not self.paper_trade
            and self.poly_client
            and now - getattr(self, '_last_balance_sync', 0) > 30
        ):
            live = self.poly_client.get_live_balance()
            if live is not None:
                self._cached_real_balance = live
                self._last_balance_sync = now
        return getattr(self, '_cached_real_balance', self.position_manager.bankroll)

    # ── Trade execution (instant — no blocking) ──────────────────

    async def _paper_trade(
        self,
        decision,
        market: MarketInfo,
        btc_price: float,
        target_price: float,
        diff_pct: float,
        seconds_left: float = 0,
    ):
        """Open a paper trade and register it for monitoring. Returns immediately."""
        # Kelly-aware sizing: pass edge and probability for optimal bet
        prob = decision.price + decision.edge  # our estimated probability
        bet_size = self.position_manager.calculate_bet_size(
            edge=decision.edge, probability=prob
        )
        bankroll_before = self.position_manager.bankroll

        trade_id = self.trade_logger.log_trade(
            btc_price_start=target_price,
            btc_price_end=btc_price,
            delta_pct=diff_pct,
            market_id=market.condition_id,
            market_question=market.question,
            odds_yes=market.up_price,
            odds_no=market.down_price,
            direction=decision.direction,
            bet_size=bet_size,
            fill_price=decision.price,
            order_id="paper",
            bankroll_before=bankroll_before,
            edge=decision.edge,
            confidence=decision.confidence,
            paper_trade=True,
        )

        secs = seconds_left or self.price_monitor.seconds_left_in_window()

        logger.info(
            f"📝 PAPER {decision.direction} ${bet_size:.2f} @ {decision.price:.3f} "
            f"on {market.slug} | Monitoring for {secs:.0f}s "
            f"(take-profit ≥{self.config['strategy'].get('take_profit', 0.70):.0%})..."
        )

        # Register position for monitoring — no blocking
        self._open_position = {
            "trade_id": trade_id,
            "slug": market.slug,
            "direction": decision.direction,
            "entry_price": decision.price,
            "bet_size": bet_size,
            "target_price": target_price,
            "paper": True,
            "diff_pct": diff_pct,
            "edge": decision.edge,
            "seconds_left": secs,
            "prev_diff_pct": self._prev_diff_pct,
        }
        self._traded_window_slugs.add(market.slug)
        self._window_entry_counts[market.slug] = self._window_entry_counts.get(market.slug, 0) + 1

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

        from src.polymarket_client import PolymarketClient

        secs_left = seconds_left or self.price_monitor.seconds_left_in_window()

        # Re-fetch LIVE prices from gamma-api right before ordering.
        # The gamma-api bestAsk/bestBid include complement matching and
        # represent the real price that fills instantly — unlike the raw
        # CLOB book which only shows one token's native orders.
        live = PolymarketClient.get_live_market_price(market.slug)
        if not live:
            logger.warning("Could not fetch live gamma prices — skipping")
            return

        up_price = live["up_price"]
        down_price = live["down_price"]

        # The fill price is the best ask for our chosen direction.
        # For UP: gamma bestAsk is the cheapest UP shares available
        #         (including complement-matched DOWN sells).
        # For DOWN: gamma bestAsk is for UP, so DOWN ask = 1 - UP bestBid.
        if decision.direction == "UP":
            fill_price = live["best_ask"]
        else:
            # DOWN best ask = 1 - UP best bid
            fill_price = round(1.0 - live["best_bid"], 2) if live["best_bid"] else down_price

        if fill_price <= 0 or fill_price >= 1:
            logger.warning(
                f"Invalid fill price {fill_price} for {decision.direction} — skipping"
            )
            return

        # ── Sanity check: is the market already priced in? ──
        # If our direction's mid-price is > 0.75, the market is already strongly
        # in our favour.  At that point the spread is wide, there's no liquidity
        # to fill against, and a buy would rest unfilled for 5-8s then cancel.
        our_mid = down_price if decision.direction == "DOWN" else up_price
        if our_mid > 0.75:
            logger.info(
                f"⏭️  Market already priced in: {decision.direction} mid={our_mid:.3f} "
                f"(>0.75) — not chasing"
            )
            return

        # ── Price bump for better fill rate ──
        # The best_ask can move between when we fetch it and when our
        # order hits the CLOB.  Adding a small bump (e.g. 2¢) means we
        # sweep slightly deeper into the book and fill instantly instead
        # of resting unfilled.  The edge check below uses the bumped
        # price, so we only trade if the edge justifies the extra cost.
        fill_bump = self.config["strategy"].get("fill_bump", 0.0)
        if fill_bump > 0:
            original_ask = fill_price
            fill_price = round(min(fill_price + fill_bump, 0.95), 2)
            if fill_price != original_ask:
                logger.info(
                    f"💰 Price bump: {original_ask:.3f} → {fill_price:.3f} "
                    f"(+{fill_bump:.3f} for fill priority)"
                )

        # ── Spread / fillability check ──
        # When BTC is surging (or crashing), the book becomes one-sided:
        # nobody is selling at a reasonable price, so the spread blows out.
        # Orders placed into a wide spread sit unfilled for 5s then get
        # cancelled — no gas burned (off-chain CLOB), but it wastes time
        # and clutters logs.  Skip early.
        max_spread = self.config["strategy"].get("max_spread", 0.10)
        if live["spread"] >= max_spread:
            logger.info(
                f"⏭️  Spread too wide ({live['spread']:.3f} ≥ {max_spread}) "
                f"— market one-sided, skipping"
            )
            return

        # Re-estimate probability with latest time
        estimated_prob = self.arbitrage_engine._estimate_probability(
            diff_pct if decision.direction == "UP" else -diff_pct,
            secs_left,
        )

        # Check edge against the real fill price
        real_edge = estimated_prob - fill_price
        min_edge = self.config["strategy"].get("min_edge", 0.01)

        if real_edge < min_edge:
            logger.info(
                f"Edge vs live ask too small: {real_edge:.4f} < {min_edge} "
                f"(prob={estimated_prob:.3f}, ask={fill_price:.3f}, "
                f"spread={live['spread']:.3f}) — skipping"
            )
            return

        logger.info(
            f"Execution: prob={estimated_prob:.3f} fill@={fill_price:.3f} "
            f"edge={real_edge:.4f} spread={live['spread']:.3f}"
        )

        prob = estimated_prob
        bet_size = self.position_manager.calculate_bet_size(
            edge=real_edge, probability=prob
        )

        # Hard safety check: real balance vs floor
        ok, reason = self._check_balance_before_trade(bet_size)
        if not ok:
            logger.warning(f"🛑 {reason}")
            return

        bankroll_before = self.position_manager.bankroll

        # ── Place order: GTC directly ──
        # FOK almost never fills on thin 5-min books and wastes 1-2s per
        # rejected round-trip.  Go straight to GTC — the fill polling
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
            logger.info(f"⚡ GTC matched instantly @ {result.fill_price:.3f}")

        order_id = result.order_id or ""

        # Use the actual cost from the exchange (accounts for share rounding)
        actual_cost = result.actual_cost or bet_size
        if actual_cost != bet_size:
            logger.info(
                f"Actual cost ${actual_cost:.2f} differs from intended "
                f"${bet_size:.2f} — using actual"
            )
            bet_size = actual_cost

        # ── Verify the order actually filled (poll up to 5s) ──
        if not filled and order_id and self.poly_client:
            for attempt in range(5):  # 5 × 1s = 5s max
                await asyncio.sleep(0.5 if attempt == 0 else 1.0)
                try:
                    status = self.poly_client.check_position_status(order_id)
                    matched = status.get("filled", 0)
                    order_status = status.get("status", "")
                    logger.info(
                        f"Fill check [{attempt+1}/5]: status={order_status} "
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
                    f"⚠️ Order {order_id} NOT filled after 5s — cancelling"
                )
                try:
                    self.poly_client.cancel_order(order_id)
                except Exception:
                    pass  # may already be cancelled by exchange
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
            f"🔴 LIVE {decision.direction} ${bet_size:.2f} @ "
            f"{fill_price:.3f} on {market.slug} (FILLED ✓)"
        )

        # Register position for monitoring — no blocking
        actual_entry = result.fill_price or fill_price
        self._open_position = {
            "trade_id": trade_id,
            "slug": market.slug,
            "direction": decision.direction,
            "entry_price": actual_entry,
            "bet_size": bet_size,
            "target_price": target_price,
            "paper": False,
            "diff_pct": diff_pct,
            "edge": real_edge,
            "seconds_left": secs_left,
            "prev_diff_pct": self._prev_diff_pct,
            "order_id": order_id,
            "limit_sell_order_id": None,  # filled in below
        }
        self._traded_window_slugs.add(market.slug)
        self._window_entry_counts[market.slug] = self._window_entry_counts.get(market.slug, 0) + 1

        # ── Place a GTC limit sell for take-profit (with settlement delay) ──
        # Shares need a few seconds to settle on-chain after the buy fills.
        # Without a delay, sell_position fails with "not enough balance".
        # Run this in a background task so it doesn't block the scan loop.
        limit_sell_pct = self.config["strategy"].get("limit_sell_profit", 0.10)
        if limit_sell_pct and limit_sell_pct > 0:
            sell_target = round(actual_entry * (1.0 + limit_sell_pct), 2)
            sell_target = min(sell_target, 0.99)  # can't exceed 0.99
            shares = bet_size / actual_entry

            async def _place_limit_sell_bg():
                """Background task: retry limit sell with longer settlement waits."""
                # Retry up to 6 times with increasing delay for settlement
                for sell_attempt in range(6):
                    wait = 3.0 + sell_attempt * 3.0  # 3s, 6s, 9s, 12s, 15s, 18s
                    await asyncio.sleep(wait)
                    if self._open_position is None:
                        return  # position was closed while we waited
                    sell_result = self.poly_client.sell_position(
                        market, decision.direction, shares, sell_target
                    )
                    if sell_result.success:
                        if self._open_position is not None:
                            self._open_position["limit_sell_order_id"] = sell_result.order_id
                        logger.info(
                            f"📋 LIMIT SELL placed: {shares:.1f} shares @ {sell_target:.3f} "
                            f"(entry {actual_entry:.3f} + {limit_sell_pct:.0%} profit) "
                            f"→ order_id={sell_result.order_id} (attempt {sell_attempt+1})"
                        )
                        return
                    logger.info(
                        f"Limit sell attempt {sell_attempt+1}/6: {sell_result.error} — "
                        f"waiting for settlement..."
                    )

                logger.warning(
                    f"Limit sell failed after 6 attempts — will use polling take-profit"
                )

            task = asyncio.create_task(_place_limit_sell_bg())
            self._background_tasks.append(task)

    # ── Take-profit execution ─────────────────────────────────────

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

        from src.polymarket_client import PolymarketClient

        # Fetch current market prices
        live = PolymarketClient.get_live_market_price(slug)
        if not live:
            logger.warning("Cannot fetch live prices for take-profit — holding")
            return

        # The best_bid is the best price someone will pay for UP shares.
        # For DOWN shares, the sell price = 1 - UP best_ask.
        if direction == "UP":
            sell_price = live["best_bid"]
        else:
            sell_price = round(1.0 - live["best_ask"], 2) if live["best_ask"] else None

        if not sell_price or sell_price <= 0:
            logger.warning("No valid sell price for take-profit — holding")
            return

        # Only take profit if we'd actually make money
        if sell_price <= entry_price:
            logger.info(
                f"💎 HOLD: sell@{sell_price:.3f} ≤ entry@{entry_price:.3f} — "
                f"no profit to take, holding ({secs_left:.0f}s left)"
            )
            return

        # Calculate shares we hold
        shares = bet_size / entry_price

        # Fetch the market object for sell_position
        market = PolymarketClient._fetch_market_by_slug(slug)
        if not market:
            logger.warning("Cannot fetch market for sell — holding")
            return

        logger.info(
            f"💰 TAKE-PROFIT: prob={our_prob:.3f} sell@{sell_price:.3f} "
            f"(entry@{entry_price:.3f}) {shares:.1f} shares | "
            f"{secs_left:.0f}s left"
        )

        result = self.poly_client.sell_position(
            market, direction, shares, sell_price
        )

        if not result.success:
            logger.warning(
                f"Sell failed: {result.error} — will hold to resolution"
            )
            return

        # Verify fill (quick check — 3 attempts × 1s)
        sell_filled = False
        sell_order_id = result.order_id or ""
        if sell_order_id:
            for attempt in range(3):
                await asyncio.sleep(0.5 if attempt == 0 else 1.0)
                try:
                    status = self.poly_client.check_position_status(sell_order_id)
                    if status.get("filled", 0) > 0:
                        sell_filled = True
                        break
                    if status.get("status", "").upper() in (
                        "CANCELED", "CANCELLED", "CANCELED_MARKET_RESOLVED"
                    ):
                        break
                except Exception:
                    pass

        if not sell_filled:
            logger.warning(
                "Sell order not filled — cancelling, will hold to resolution"
            )
            try:
                self.poly_client.cancel_order(sell_order_id)
            except Exception:
                pass
            return

        # Calculate actual profit: proceeds - cost
        actual_sell_price = result.fill_price or sell_price
        proceeds = shares * actual_sell_price
        cost = bet_size
        profit = proceeds - cost

        self.position_manager.update_after_win(profit)
        self.trade_logger.update_trade_outcome(
            trade_id, "WIN", profit,
            self.position_manager.bankroll,
            self.position_manager.consecutive_wins,
            btc_price_close=btc_now,
        )

        entries = self._window_entry_counts.get(slug, 1)
        can_reenter = (
            self._scalp_enabled
            and entries < self._scalp_max_entries
            and secs_left > 30  # need enough time for another trade
        )

        logger.info(
            f"✅ LIVE TAKE-PROFIT +${profit:.2f} [{slug}] | "
            f"Sold {shares:.1f} shares @ {actual_sell_price:.3f} "
            f"(entry {entry_price:.3f}) | "
            f"Bankroll: ${self.position_manager.bankroll:.2f}"
            + (f" | 🔄 Re-entry eligible ({self._scalp_cooldown_tp}s cooldown)" if can_reenter else "")
        )

        self._open_position = None
        if can_reenter:
            self._reentry_cooldown_until = time.time() + self._scalp_cooldown_tp

    # ── Hedge lock-in (insurance strategy) ──────────────────────────

    async def _execute_hedge_lockin(self, pos: dict, our_prob: float, secs_left: float):
        """Buy the opposite side to lock in guaranteed profit.

        The math:
          We hold N shares of side A at entry_price.
          Cost so far = N * entry_price.
          The opposite side B is cheap (≈ 1 - our_prob).
          If we buy N shares of side B at hedge_price:
            Total invested = N * (entry_price + hedge_price)
            One side ALWAYS pays $1/share → payout = N * $1.00
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
        from src.polymarket_client import PolymarketClient
        live = PolymarketClient.get_live_market_price(slug)
        if not live:
            logger.debug("Hedge: cannot fetch live prices — skipping")
            return

        # What does it cost to buy the opposite side right now?
        if opposite == "DOWN":
            # DOWN best ask = 1 - UP best bid
            hedge_price = round(1.0 - live["best_bid"], 2) if live["best_bid"] else live["down_price"]
        else:
            # UP best ask
            hedge_price = live["best_ask"] if live["best_ask"] else live["up_price"]

        if hedge_price <= 0 or hedge_price >= 1:
            logger.debug(f"Hedge: invalid opposite price {hedge_price} — skipping")
            return

        # The key formula: can we lock in a profit?
        combined = entry_price + hedge_price
        if combined >= 1.0:
            # No guaranteed profit possible — the spread eats it
            logger.debug(
                f"Hedge: entry({entry_price:.3f}) + opposite({hedge_price:.3f}) "
                f"= {combined:.3f} ≥ 1.00 — no profit to lock in"
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
                f"— not enough upside"
            )
            return

        # Cap hedge cost
        if hedge_cost > self._hedge_max_cost:
            # Scale down: buy fewer insurance shares
            affordable_shares = self._hedge_max_cost / hedge_price
            hedge_cost = affordable_shares * hedge_price
            # We'll have a partial hedge — only affordable_shares are insured.
            # The rest ride on the original directional bet.
            partial = True
            logger.info(
                f"🛡️ HEDGE: capping insurance to {affordable_shares:.1f} of "
                f"{shares:.1f} shares (max cost ${self._hedge_max_cost:.2f})"
            )
        else:
            affordable_shares = shares
            partial = False

        # Balance check
        ok, reason = self._check_balance_before_trade(hedge_cost)
        if not ok:
            logger.info(f"🛡️ HEDGE blocked: {reason}")
            return

        # Fetch market for order placement
        market = PolymarketClient._fetch_market_by_slug(slug)
        if not market:
            logger.warning("Hedge: cannot fetch market object — skipping")
            return

        # Apply fill bump for better fill rate
        fill_bump = self.config["strategy"].get("fill_bump", 0.0)
        order_price = round(min(hedge_price + fill_bump, 0.95), 2)
        actual_hedge_cost = round(affordable_shares * order_price, 2)

        logger.info(
            f"🛡️ HEDGE LOCK-IN: Buying {affordable_shares:.1f} {opposite} shares "
            f"@ {order_price:.3f} (cost ${actual_hedge_cost:.2f}) | "
            f"Entry {direction}@{entry_price:.3f} + {opposite}@{order_price:.3f} "
            f"= {entry_price + order_price:.3f} | "
            f"💰 Guaranteed profit: ${shares * (1.0 - entry_price - order_price):.2f} "
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
                f"✅ HEDGE FILLED! {affordable_shares:.1f} {opposite} @ {order_price:.3f} | "
                f"🔒 Locked in ${locked_profit:.2f} profit regardless of outcome"
                + (" (partial hedge)" if partial else " (FULL hedge)")
            )
        else:
            # FOK failed — try GTC with quick poll
            logger.info(
                f"Hedge FOK rejected ({result.error}) — trying GTC"
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
                        f"✅ HEDGE FILLED (GTC)! 🔒 Locked ${locked_profit:.2f}"
                    )
                else:
                    try:
                        self.poly_client.cancel_order(result.order_id)
                    except Exception:
                        pass
                    logger.info("Hedge GTC not filled — cancelled, will retry next scan")
            else:
                logger.warning(f"Hedge order failed: {result.error}")

    # ── Momentum pre-position (next-window orders) ─────────────────

    async def _check_preposition(self):
        """In the last 30s of the current window, if BTC is surging/crashing,
        place a GTC limit order on the NEXT window's market in the direction
        of the momentum.

        Rationale:
          - The next window's market is already listed on gamma-api, priced
            near 50/50 because the "price to beat" hasn't been set yet.
          - If BTC is in a strong directional move (e.g. +0.05% and
            accelerating), there's a good chance the move continues into
            the next window — BTC will open ABOVE the new target.
          - By placing a limit order at ~$0.50 before the window starts,
            we get positioned cheaply before the market adjusts.
          - If the move reverses, the order may not fill (thin book at 50/50)
            or we cancel early in the new window when momentum fades.

        This is directional (unlike straddle) — we only bet the momentum side.
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

        # Don't pre-position if we already have a position that will
        # carry into the next window or if we have an open position
        if self._open_position is not None:
            return

        # Can we trade at all?
        bot_pnl = self.trade_logger.get_bot_live_pnl() if not self.paper_trade else 0.0
        can_trade, reason = self.position_manager.can_trade(bot_pnl=bot_pnl)
        if not can_trade:
            return

        # Measure momentum: how far has BTC moved from the current window's open?
        target_price = PolymarketClient.fetch_price_to_beat(current_slug)
        if target_price is None:
            return

        diff_pct = ((btc_price - target_price) / target_price) * 100

        # Check momentum strength
        if abs(diff_pct) < self._prepos_min_momentum:
            return  # move isn't strong enough

        # Check acceleration (is it still moving in the same direction?)
        if self._prev_diff_pct is not None:
            accel = abs(diff_pct) - abs(self._prev_diff_pct)
            # Positive accel means the move is getting bigger
            if accel < self._prepos_min_accel:
                return  # momentum is fading, not accelerating
            # Also check direction hasn't flipped
            if self._prev_diff_pct != 0 and (
                (diff_pct > 0) != (self._prev_diff_pct > 0)
            ):
                return  # direction reversed
        else:
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

        # Use a limit price near 50/50 — the market shouldn't have moved yet
        if direction == "UP":
            limit_price = min(next_market.up_price, self._prepos_max_price)
        else:
            limit_price = min(next_market.down_price, self._prepos_max_price)

        if limit_price <= 0 or limit_price >= 0.60:
            logger.debug(
                f"Pre-position: price {limit_price:.3f} too high "
                f"(next window already priced in?) — skipping"
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
            logger.info(f"🚀 Pre-position blocked: {reason}")
            return

        import math
        shares = math.floor((bet_size / limit_price) * 10000) / 10000

        logger.info(
            f"🚀 PRE-POSITION: BTC momentum {diff_pct:+.4f}% (accel "
            f"+{abs(diff_pct) - abs(self._prev_diff_pct):.4f}%) → {direction} | "
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
                f"  ✅ Pre-position order placed: {direction} {shares:.1f} shares "
                f"@ {limit_price:.3f} on {next_slug} → {result.order_id}"
            )

            # Log the trade
            bankroll = self.position_manager.bankroll
            self.trade_logger.log_trade(
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
                paper_trade=False,
            )
        else:
            logger.warning(f"  ❌ Pre-position failed: {result.error}")

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

        # Phase 1: Window hasn't started yet — just wait
        if elapsed_in_window < 0:
            return

        # Phase 2 & 3: Window has started — check fill status
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
            logger.info(f"🚀 Pre-position {order_id} was cancelled externally")
            self._prepos_pending = None
            return

        if filled > 0:
            # Order filled — promote to a real open position!
            slug = prepos["slug"]
            direction = prepos["direction"]
            entry_price = prepos["limit_price"]
            bet_size = prepos["bet_size"]

            # Fetch the new window's target price (now available)
            target_price = PolymarketClient.fetch_price_to_beat(slug)

            logger.info(
                f"🚀✅ PRE-POSITION FILLED: {direction} {prepos['shares']:.1f} "
                f"shares @ {entry_price:.3f} on {slug} | "
                f"Original momentum: {prepos['momentum_diff_pct']:+.4f}% | "
                f"New window target: ${target_price:,.2f}" if target_price else
                f"🚀✅ PRE-POSITION FILLED: {direction} {prepos['shares']:.1f} "
                f"shares @ {entry_price:.3f} on {slug} (target pending)"
            )

            # Log trade entry for the pre-positioned order
            trade_id = self.trade_logger.log_trade(
                btc_price_start=target_price or 0,
                btc_price_end=self.price_monitor.current_price or 0,
                delta_pct=prepos["momentum_diff_pct"],
                market_id="",
                market_question=f"PREPOS-FILLED {direction}: {slug}",
                odds_yes=0.5,
                odds_no=0.5,
                direction=direction,
                bet_size=bet_size,
                fill_price=entry_price,
                order_id=order_id,
                bankroll_before=self.position_manager.bankroll,
                edge=0.05,  # estimated
                confidence=abs(prepos["momentum_diff_pct"]) / 0.10,
                paper_trade=False,
            )

            # Promote to open position for normal monitoring
            self._open_position = {
                "trade_id": trade_id,
                "slug": slug,
                "direction": direction,
                "entry_price": entry_price,
                "bet_size": bet_size,
                "target_price": target_price or 0,
                "paper": False,
                "diff_pct": prepos["momentum_diff_pct"],
                "edge": 0.05,
                "seconds_left": 300 - elapsed_in_window,
                "prev_diff_pct": None,
                "order_id": order_id,
                "limit_sell_order_id": None,
                "from_preposition": True,
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
                    async def _place_prepos_limit_sell():
                        for attempt in range(6):
                            wait = 3.0 + attempt * 3.0
                            await asyncio.sleep(wait)
                            if self._open_position is None:
                                return
                            sell_result = self.poly_client.sell_position(
                                next_market, direction, shares, sell_target
                            )
                            if sell_result.success:
                                if self._open_position is not None:
                                    self._open_position["limit_sell_order_id"] = sell_result.order_id
                                logger.info(
                                    f"📋 PREPOS LIMIT SELL: {shares:.1f} shares @ {sell_target:.3f} "
                                    f"→ {sell_result.order_id}"
                                )
                                return
                            logger.info(
                                f"Prepos limit sell attempt {attempt+1}/6: {sell_result.error}"
                            )
                        logger.warning("Prepos limit sell failed after 6 attempts")

                    task = asyncio.create_task(_place_prepos_limit_sell())
                    self._background_tasks.append(task)

            return

        # Not filled yet — check if we should cancel
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
                        f"🚀❌ Pre-position: momentum reversed "
                        f"(new diff {new_diff:+.4f}%, expected {expected_dir}) "
                        f"— cancelling"
                    )
                    try:
                        self.poly_client.cancel_order(order_id)
                    except Exception:
                        pass
                    self._prepos_pending = None
                    return

        # Cancel if unfilled after 45s into the new window
        if elapsed_in_window > 45:
            logger.info(
                f"🚀⏱ Pre-position unfilled after {elapsed_in_window:.0f}s "
                f"— cancelling {order_id}"
            )
            try:
                self.poly_client.cancel_order(order_id)
            except Exception:
                pass
            self._prepos_pending = None

    # ── Straddle strategy ──────────────────────────────────────────

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
            logger.warning("🔀 STRADDLE: No Polymarket client — skipping")
            return

        # Enforce min_bankroll floor for straddle too
        can_trade, reason = self.position_manager.can_trade()
        if not can_trade:
            logger.info(f"🔀 STRADDLE blocked: {reason}")
            return

        max_straddle_cost = shares * price * 2
        ok, reason = self._check_balance_before_trade(max_straddle_cost)
        if not ok:
            logger.warning(f"🔀 STRADDLE blocked: {reason}")
            return

        logger.info(
            f"🔀 STRADDLE: BTC near target ({diff_pct:+.4f}%) with {seconds_left:.0f}s left | "
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
            logger.info(f"  ✅ UP limit: {shares} shares @ ${price:.2f} → {up_result.order_id}")
        else:
            logger.warning(f"  ❌ UP limit failed: {up_result.error}")

        if down_result.success:
            logger.info(f"  ✅ DOWN limit: {shares} shares @ ${price:.2f} → {down_result.order_id}")
        else:
            logger.warning(f"  ❌ DOWN limit failed: {down_result.error}")

        # Log both sides as trades
        bankroll = self.position_manager.bankroll
        for side, res in [("UP", up_result), ("DOWN", down_result)]:
            if res.success:
                cost = shares * price
                self.trade_logger.log_trade(
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
                    paper_trade=False,
                )

    async def _monitor_straddle(self):
        """Check straddle order status — cancel unfilled orders at window end."""
        straddle = self._straddle_active
        if not straddle:
            return

        seconds_left = self.price_monitor.seconds_left_in_window()

        # Check fill status
        for side, key in [("UP", "up_order_id"), ("DOWN", "down_order_id")]:
            oid = straddle.get(key)
            filled_key = f"{side.lower()}_filled"
            if oid and not straddle.get(filled_key):
                status = self.poly_client.check_position_status(oid)
                if status.get("filled", 0) > 0:
                    straddle[filled_key] = True
                    logger.info(f"🔀 Straddle {side} FILLED! ({straddle['shares']} shares @ ${straddle['price']:.2f})")

        up_filled = straddle.get("up_filled", False)
        down_filled = straddle.get("down_filled", False)

        if up_filled and down_filled:
            cost = straddle["shares"] * straddle["price"] * 2
            payout = straddle["shares"] * 1.0  # winning side pays $1/share
            net = payout - cost
            logger.info(
                f"🎯 STRADDLE BOTH SIDES FILLED! Cost=${cost:.2f} → "
                f"Guaranteed payout=${payout:.2f} → Net +${net:.2f}"
            )
            self.position_manager.update_after_win(net)
            self._straddle_active = None
            return

        # Cancel remaining orders near window end
        if seconds_left <= 5:
            for side, key in [("UP", "up_order_id"), ("DOWN", "down_order_id")]:
                filled_key = f"{side.lower()}_filled"
                oid = straddle.get(key)
                if oid and not straddle.get(filled_key):
                    self.poly_client.cancel_order(oid)
                    logger.info(f"🔀 Straddle {side} cancelled (unfilled, window ending)")

            # Resolve: if one side filled, it becomes a directional bet
            if up_filled or down_filled:
                filled_side = "UP" if up_filled else "DOWN"
                cost = straddle["shares"] * straddle["price"]
                logger.info(
                    f"🔀 Straddle single fill: {filled_side} only (cost=${cost:.2f}) → "
                    f"awaiting resolution as directional bet"
                )
            else:
                logger.info("🔀 Straddle: no fills, both cancelled — no cost")

            self._straddle_active = None

    # ── Background resolution (runs as asyncio task) ─────────────

    async def _resolve_trade_background(self, pos: dict):
        """Determine win/loss for a finished trade. Runs in the background
        so the main loop can keep trading the next window."""
        slug = pos["slug"]
        direction = pos["direction"]
        entry_price = pos["entry_price"]
        bet_size = pos["bet_size"]
        trade_id = pos["trade_id"]
        is_paper = pos["paper"]
        order_id = pos.get("order_id", "")
        btc_at_close = pos.get("btc_at_close")  # Binance price captured at window end

        # Verify the order actually filled before resolving.
        # If size_matched == 0 on the exchange, it never filled.
        if not is_paper and order_id and self.poly_client:
            try:
                status = self.poly_client.check_position_status(order_id)
                matched = status.get("filled", 0)
                if matched == 0:
                    logger.warning(
                        f"⚠️ Order {order_id} for [{slug}] never filled — "
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
                    f"🔍 Resolution [{slug}]: close=${close_price:,.2f} vs "
                    f"target=${target:,.2f} → {'UP wins' if close_price > target else 'DOWN wins'}"
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
                logger.warning(f"Cannot resolve {slug} — marking as LOSS")
                win = False

        prefix = "PAPER" if is_paper else "LIVE"

        # Determine the actual close price to record
        actual_close = close_price or btc_at_close

        # Feed outcome to adaptive learner
        trade_diff_pct = pos.get("diff_pct", 0.0)
        trade_edge = pos.get("edge", 0.0)
        trade_secs = pos.get("seconds_left", 150.0)
        trade_prev_diff = pos.get("prev_diff_pct")

        # ── Hedged position: guaranteed profit either way ──
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
                # Our original side won — unhedged shares also pay out
                unhedged_payout = unhedged_shares * 1.0  # they also win
                total_payout = payout + unhedged_payout
            else:
                # Our original side lost — only hedged shares pay out
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
            result_emoji = "🔒" if not win else "🔒✅"
            logger.info(
                f"{result_emoji} {prefix} HEDGED {'WIN' if win else 'SAVED'} "
                f"+${profit:.2f} [{slug}] | "
                f"{'Original side won' if win else 'Hedge insurance paid out!'} | "
                f"Payout ${total_payout:.2f} - Cost ${total_cost:.2f} | "
                f"Bankroll: ${self.position_manager.bankroll:.2f}"
            )
            return

        if win:
            profit = bet_size * ((1.0 / entry_price) - 1)
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
            logger.info(
                f"✅ {prefix} WIN +${profit:.2f} [{slug}] | "
                f"Bankroll: ${self.position_manager.bankroll:.2f}"
            )
        else:
            self.position_manager.update_after_loss(bet_size)
            self.trade_logger.update_trade_outcome(
                trade_id, "LOSS", -bet_size,
                self.position_manager.bankroll,
                self.position_manager.consecutive_wins,
                btc_price_close=actual_close,
            )
            self.learner.record_outcome(
                won=False, pnl=-bet_size, diff_pct=trade_diff_pct,
                edge=trade_edge, seconds_left=trade_secs,
                prev_diff_pct=trade_prev_diff,
            )
            logger.info(
                f"❌ {prefix} LOSS -${bet_size:.2f} [{slug}] | "
                f"Bankroll: ${self.position_manager.bankroll:.2f}"
            )

    def _print_session_summary(self):
        """Print end-of-session summary."""
        stats = self.trade_logger.get_session_stats()
        lifetime = self.trade_logger.get_all_stats()
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
        logger.info(f"LIFETIME: {lifetime.get('total_trades', 0)} trades | "
                    f"P&L: ${lifetime.get('total_pnl', 0):.2f} | "
                    f"Win rate: {lifetime.get('win_rate', 0):.1%}")
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
