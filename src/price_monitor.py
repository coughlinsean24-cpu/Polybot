"""
Real-time BTC price monitoring via Binance WebSocket with CoinGecko fallback.
Tracks 5-minute rolling price deltas and emits arbitrage signals.
"""

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field

import aiohttp
import websockets

logger = logging.getLogger(__name__)


@dataclass
class PricePoint:
    price: float
    timestamp: float


@dataclass
class ArbitrageSignal:
    direction: str  # "UP" or "DOWN"
    delta_pct: float
    confidence: float
    btc_price_start: float
    btc_price_end: float
    timestamp: float


class PriceMonitor:
    def __init__(self, config: dict):
        self.binance_ws_url = config["price_feeds"]["binance_ws"]
        self.coingecko_url = config["price_feeds"]["coingecko_url"]
        self.min_price_delta = config["strategy"]["min_price_delta"]

        # Rolling 5-minute window of price points (1-sec granularity)
        self.price_history: deque[PricePoint] = deque(maxlen=600)
        self.current_price: float | None = None
        self._ws_connected = False
        self._running = False

    async def start(self):
        """Start price monitoring with WebSocket primary + REST fallback."""
        self._running = True
        while self._running:
            try:
                await self._connect_binance_ws()
            except Exception as e:
                logger.warning(f"Binance WS failed: {e}, falling back to CoinGecko")
                await self._poll_coingecko()

    async def stop(self):
        self._running = False

    async def _connect_binance_ws(self):
        """Connect to Binance BTC/USDT trade stream."""
        logger.info("Connecting to Binance WebSocket...")
        async with websockets.connect(self.binance_ws_url) as ws:
            self._ws_connected = True
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
        self.price_history.append(PricePoint(price=price, timestamp=now))

    def calculate_5min_delta(self) -> float | None:
        """Calculate price change over the last 5 minutes as a percentage."""
        if len(self.price_history) < 2:
            return None

        now = time.time()
        cutoff = now - 300  # 5 minutes ago

        # Find the oldest price point within our 5-min window
        oldest = None
        for point in self.price_history:
            if point.timestamp >= cutoff:
                oldest = point
                break

        if oldest is None or self.current_price is None:
            return None

        delta_pct = ((self.current_price - oldest.price) / oldest.price) * 100
        return delta_pct

    def detect_arbitrage_signal(self) -> ArbitrageSignal | None:
        """Check if current price movement creates an arbitrage opportunity."""
        delta = self.calculate_5min_delta()
        if delta is None:
            return None

        abs_delta = abs(delta)
        if abs_delta < self.min_price_delta:
            return None

        # Confidence scales with delta magnitude (capped at 1.0)
        confidence = min(abs_delta / (self.min_price_delta * 3), 1.0)

        # Find 5-min-ago price
        now = time.time()
        cutoff = now - 300
        start_price = self.current_price
        for point in self.price_history:
            if point.timestamp >= cutoff:
                start_price = point.price
                break

        return ArbitrageSignal(
            direction="UP" if delta > 0 else "DOWN",
            delta_pct=delta,
            confidence=confidence,
            btc_price_start=start_price,
            btc_price_end=self.current_price,
            timestamp=now,
        )
