"""
Real-time BTC price monitoring via Binance WebSocket with CoinGecko fallback.

Provides current BTC price and the ability to look up the price at any past
timestamp within the rolling window.  The Orchestrator uses this to compare
the live BTC price against the Polymarket "price to beat" (the Chainlink BTC
price at the start of each 5-minute market window).
"""

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass

import aiohttp
import websockets

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 300  # 5 minutes

# WebSocket keepalive settings
WS_PING_INTERVAL = 20  # send ping every 20s
WS_PING_TIMEOUT = 10   # wait up to 10s for pong
WS_RECONNECT_DELAY = 2  # initial reconnect delay (doubles on each failure, max 60s)


@dataclass
class PricePoint:
    price: float
    timestamp: float


class PriceMonitor:
    def __init__(self, config: dict):
        self.binance_ws_url = config["price_feeds"]["binance_ws"]
        self.coingecko_url = config["price_feeds"]["coingecko_url"]

        # Rolling window of price points – keep ~20 min @ 1 sample/sec
        self.price_history: deque[PricePoint] = deque(maxlen=1200)
        self.current_price: float | None = None
        self._ws_connected = False
        self._running = False
        self._reconnect_delay = WS_RECONNECT_DELAY
        self._first_price_time: float | None = None  # when we got our first price

    # ── Lifecycle ───────────────────────────────────────────────────

    async def start(self):
        """Start price monitoring — reconnects automatically on failure."""
        self._running = True
        while self._running:
            try:
                await self._connect_binance_ws()
            except Exception as e:
                self._ws_connected = False
                logger.warning(
                    f"Binance WS error: {e} — reconnecting in "
                    f"{self._reconnect_delay}s"
                )
                await asyncio.sleep(self._reconnect_delay)
                # Exponential backoff capped at 60s
                self._reconnect_delay = min(self._reconnect_delay * 2, 60)

    async def stop(self):
        self._running = False

    # ── Data Sources ────────────────────────────────────────────────

    async def _connect_binance_ws(self):
        """Connect to Binance BTC/USDT trade stream with keepalive pings."""
        logger.info("Connecting to Binance WebSocket...")
        async with websockets.connect(
            self.binance_ws_url,
            ping_interval=WS_PING_INTERVAL,
            ping_timeout=WS_PING_TIMEOUT,
        ) as ws:
            self._ws_connected = True
            self._reconnect_delay = WS_RECONNECT_DELAY  # reset on success
            logger.info("Binance WebSocket connected")
            async for message in ws:
                if not self._running:
                    break
                data = json.loads(message)
                price = float(data["p"])
                self._record_price(price)

    async def _poll_coingecko(self):
        """Fallback: poll CoinGecko REST API every 2 seconds."""
        logger.info("Starting CoinGecko polling fallback")
        async with aiohttp.ClientSession() as session:
            while self._running:
                try:
                    price = await self._get_coingecko_price(session)
                    if price:
                        self._record_price(price)
                except Exception as e:
                    logger.error(f"CoinGecko poll error: {e}")
                await asyncio.sleep(2)

    async def _get_coingecko_price(self, session: aiohttp.ClientSession) -> float | None:
        params = {"ids": "bitcoin", "vs_currencies": "usd"}
        async with session.get(self.coingecko_url, params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data["bitcoin"]["usd"]
            elif resp.status == 429:
                logger.warning("CoinGecko rate limited, backing off")
                await asyncio.sleep(10)
            return None

    def _record_price(self, price: float):
        now = time.time()
        self.current_price = price
        if self._first_price_time is None:
            self._first_price_time = now
        # Throttle history to ~1 sample per second (Binance sends 100+/sec)
        if self.price_history and (now - self.price_history[-1].timestamp) < 1.0:
            return
        self.price_history.append(PricePoint(price=price, timestamp=now))

    # ── Price Lookups ───────────────────────────────────────────────

    def get_price_at(self, target_ts: float) -> float | None:
        """
        Return the BTC price closest to *target_ts* from our history.
        Returns None if we have no data near that time.
        """
        if not self.price_history:
            return None

        best: PricePoint | None = None
        best_gap = float("inf")

        for pt in self.price_history:
            gap = abs(pt.timestamp - target_ts)
            if gap < best_gap:
                best_gap = gap
                best = pt

        # Only trust it if within 30 seconds of the target
        if best and best_gap <= 30:
            return best.price
        return None

    def get_window_start_price(self) -> tuple[float | None, float]:
        """
        Return (price_at_window_start, window_start_timestamp) for the
        current 5-minute window.
        """
        window_start = (int(time.time()) // WINDOW_SECONDS) * WINDOW_SECONDS
        price = self.get_price_at(float(window_start))
        return price, float(window_start)

    def seconds_left_in_window(self) -> float:
        """How many seconds remain in the current 5-minute window."""
        now = time.time()
        window_start = (int(now) // WINDOW_SECONDS) * WINDOW_SECONDS
        window_end = window_start + WINDOW_SECONDS
        return max(0.0, window_end - now)

    def has_enough_history(self) -> bool:
        """Do we have at least 30 seconds of price data?"""
        if self._first_price_time is None:
            return False
        return (time.time() - self._first_price_time) >= 30
