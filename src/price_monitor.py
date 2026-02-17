"""
Real-time BTC price monitoring with **parallel** multi-exchange WebSocket feeds.

Runs ALL configured WebSocket providers simultaneously (Coinbase, Binance,
Kraken, Bybit) so the fastest tick always wins.  Falls back to CoinGecko
REST polling only if every single WS connection is down.

Also exposes per-feed agreement metrics that the edge model can consume.
"""

import asyncio
import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field

import aiohttp
import websockets

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 300  # 5 minutes

# WebSocket keepalive settings
WS_PING_INTERVAL = 20  # send ping every 20s
WS_PING_TIMEOUT = 10   # wait up to 10s for pong
WS_RECONNECT_DELAY = 2  # initial reconnect delay (doubles on each failure, max 60s)

# CoinGecko polling interval when used as fallback
COINGECKO_POLL_INTERVAL = 2  # seconds

# Built-in extra feeds (always tried in parallel with config feeds)
EXTRA_FEEDS = [
    {"url": "wss://ws.kraken.com/v2", "type": "kraken"},
    {"url": "wss://stream.bybit.com/v5/public/spot", "type": "bybit"},
]


@dataclass
class PricePoint:
    price: float
    timestamp: float


@dataclass
class FeedStatus:
    """Tracks the latest state from one exchange feed."""
    name: str
    price: float = 0.0
    last_update: float = 0.0
    connected: bool = False
    tick_count: int = 0


class PriceMonitor:
    def __init__(self, config: dict):
        pf = config["price_feeds"]
        self.ws_endpoints: list[dict] = list(pf.get("ws_endpoints", []))
        self.coingecko_url: str = pf.get("coingecko_url", "")

        # Legacy single-endpoint config support
        if not self.ws_endpoints and "binance_ws" in pf:
            self.ws_endpoints = [{"url": pf["binance_ws"], "type": "binance"}]

        # Merge in extra feeds (Kraken, Bybit) -- skip duplicates
        existing_types = {ep.get("type") for ep in self.ws_endpoints}
        for ef in EXTRA_FEEDS:
            if ef["type"] not in existing_types:
                self.ws_endpoints.append(ef)

        # Rolling window of price points - keep ~20 min @ 1 sample/sec
        self.price_history: deque[PricePoint] = deque(maxlen=1200)
        self.current_price: float | None = None
        self._ws_connected = False
        self._running = False
        self._reconnect_delay = WS_RECONNECT_DELAY
        self._first_price_time: float | None = None
        self._active_provider: str | None = None

        # Per-feed tracking for multi-feed consensus
        self._feeds: dict[str, FeedStatus] = {}
        for ep in self.ws_endpoints:
            name = ep.get("type", "unknown")
            self._feeds[name] = FeedStatus(name=name)

    # -- Lifecycle ---------------------------------------------------

    async def start(self):
        """Run ALL WebSocket feeds in parallel; fastest tick wins."""
        self._running = True

        # Launch every WS feed as a concurrent task
        tasks = []
        for ep in self.ws_endpoints:
            url = ep["url"]
            provider = ep.get("type", "unknown")
            tasks.append(asyncio.create_task(
                self._feed_loop(url, provider), name=f"feed-{provider}"
            ))

        # Also run the CoinGecko fallback poller
        if self.coingecko_url:
            tasks.append(asyncio.create_task(
                self._coingecko_loop(), name="feed-coingecko"
            ))

        logger.info(
            f"[FEEDS] Launching {len(tasks)} parallel price feeds: "
            f"{', '.join(ep.get('type','?') for ep in self.ws_endpoints)}"
            f"{' + coingecko' if self.coingecko_url else ''}"
        )

        # Wait for all (they run forever until stopped)
        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop(self):
        self._running = False

    # -- Per-feed reconnect loop -------------------------------------

    async def _feed_loop(self, url: str, provider: str):
        """Reconnecting loop for a single exchange feed."""
        delay = WS_RECONNECT_DELAY
        while self._running:
            try:
                logger.info(f"[FEED:{provider}] Connecting to {url}")
                await self._connect_ws(url, provider)
            except Exception as e:
                if provider in self._feeds:
                    self._feeds[provider].connected = False
                logger.warning(f"[FEED:{provider}] Disconnected: {e}")
            if self._running:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)

    # -- WebSocket providers -----------------------------------------

    async def _connect_ws(self, url: str, provider: str):
        """Generic WS connect — dispatches to the right parser."""
        if provider == "coinbase":
            await self._connect_coinbase(url)
        elif provider == "binance":
            await self._connect_binance(url)
        elif provider == "kraken":
            await self._connect_kraken(url)
        elif provider == "bybit":
            await self._connect_bybit(url)
        else:
            await self._connect_binance(url)

    async def _connect_coinbase(self, url: str):
        """Connect to Coinbase WebSocket ticker feed."""
        async with websockets.connect(
            url, ping_interval=WS_PING_INTERVAL, ping_timeout=WS_PING_TIMEOUT,
        ) as ws:
            sub = {
                "type": "subscribe",
                "product_ids": ["BTC-USD"],
                "channels": ["ticker"],
            }
            await ws.send(json.dumps(sub))
            self._mark_connected("coinbase")

            async for message in ws:
                if not self._running:
                    break
                data = json.loads(message)
                if data.get("type") == "ticker" and "price" in data:
                    self._record_price(float(data["price"]), "coinbase")

    async def _connect_binance(self, url: str):
        """Connect to Binance (or Binance US) BTC/USDT trade stream."""
        async with websockets.connect(
            url, ping_interval=WS_PING_INTERVAL, ping_timeout=WS_PING_TIMEOUT,
        ) as ws:
            self._mark_connected("binance")

            async for message in ws:
                if not self._running:
                    break
                data = json.loads(message)
                if "p" in data:
                    self._record_price(float(data["p"]), "binance")

    async def _connect_kraken(self, url: str):
        """Connect to Kraken WebSocket v2 ticker feed."""
        async with websockets.connect(
            url, ping_interval=WS_PING_INTERVAL, ping_timeout=WS_PING_TIMEOUT,
        ) as ws:
            sub = {
                "method": "subscribe",
                "params": {"channel": "ticker", "symbol": ["BTC/USD"]},
            }
            await ws.send(json.dumps(sub))
            self._mark_connected("kraken")

            async for message in ws:
                if not self._running:
                    break
                try:
                    data = json.loads(message)
                    # Kraken v2 ticker: {"channel":"ticker","type":"update","data":[{"last":12345.67,...}]}
                    if data.get("channel") == "ticker" and "data" in data:
                        for tick in data["data"]:
                            last = tick.get("last")
                            if last is not None:
                                self._record_price(float(last), "kraken")
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass

    async def _connect_bybit(self, url: str):
        """Connect to Bybit v5 spot BTC/USDT ticker feed."""
        async with websockets.connect(
            url, ping_interval=WS_PING_INTERVAL, ping_timeout=WS_PING_TIMEOUT,
        ) as ws:
            sub = {"op": "subscribe", "args": ["tickers.BTCUSDT"]}
            await ws.send(json.dumps(sub))
            self._mark_connected("bybit")

            async for message in ws:
                if not self._running:
                    break
                try:
                    data = json.loads(message)
                    # Bybit v5: {"topic":"tickers.BTCUSDT","data":{"lastPrice":"12345.67",...}}
                    if data.get("topic", "").startswith("tickers.") and "data" in data:
                        lp = data["data"].get("lastPrice")
                        if lp:
                            self._record_price(float(lp), "bybit")
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass

    # -- CoinGecko REST fallback -------------------------------------

    async def _coingecko_loop(self):
        """Continuous CoinGecko poller — runs in parallel with WS feeds."""
        while self._running:
            # Only actually poll if no WS feed has updated in 5 seconds
            any_ws_alive = any(
                f.connected and (time.time() - f.last_update) < 5
                for f in self._feeds.values()
            )
            if not any_ws_alive and self.coingecko_url:
                try:
                    await self._poll_coingecko_once()
                except Exception as e:
                    logger.debug(f"CoinGecko poll error: {e}")
            await asyncio.sleep(COINGECKO_POLL_INTERVAL)

    async def _poll_coingecko_once(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.coingecko_url,
                params={"ids": "bitcoin", "vs_currencies": "usd"},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    price = data.get("bitcoin", {}).get("usd")
                    if price:
                        self._record_price(float(price), "coingecko")

    # -- Feed helpers ------------------------------------------------

    def _mark_connected(self, provider: str):
        if provider in self._feeds:
            self._feeds[provider].connected = True
        self._ws_connected = True
        self._reconnect_delay = WS_RECONNECT_DELAY
        self._active_provider = provider
        logger.info(f"[FEED:{provider}] Connected")

    # -- Price recording ---------------------------------------------

    def _record_price(self, price: float, provider: str = "unknown"):
        now = time.time()
        self.current_price = price
        if self._first_price_time is None:
            self._first_price_time = now

        # Update per-feed status
        if provider in self._feeds:
            fs = self._feeds[provider]
            fs.price = price
            fs.last_update = now
            fs.connected = True
            fs.tick_count += 1

        # Throttle history to ~1 sample per second
        if self.price_history and (now - self.price_history[-1].timestamp) < 1.0:
            return
        self.price_history.append(PricePoint(price=price, timestamp=now))

    # -- Price Lookups -----------------------------------------------

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

    # -- Multi-feed consensus & analytics ----------------------------

    def active_feed_count(self) -> int:
        """How many WS feeds have updated in the last 10 seconds?"""
        cutoff = time.time() - 10
        return sum(1 for f in self._feeds.values()
                   if f.connected and f.last_update > cutoff)

    def feed_agreement(self, target_price: float) -> float:
        """
        Return 0.0-1.0 showing what fraction of active feeds agree
        on the SAME side (UP or DOWN) relative to target_price.
        Useful for big-move confidence: if all 4 exchanges say DOWN,
        it's very likely real.
        """
        cutoff = time.time() - 10
        active = [f for f in self._feeds.values()
                  if f.connected and f.last_update > cutoff and f.price > 0]
        if not active:
            return 0.5  # no data

        above = sum(1 for f in active if f.price > target_price)
        below = len(active) - above
        return max(above, below) / len(active)

    def feed_summary(self) -> str:
        """One-line summary of all feed states for logging."""
        cutoff = time.time() - 10
        parts = []
        for f in self._feeds.values():
            alive = f.connected and f.last_update > cutoff
            status = f"${f.price:,.2f}" if alive else "DOWN"
            parts.append(f"{f.name}={status}")
        return " | ".join(parts)

    def measured_volatility(self, lookback_seconds: int = 300) -> float:
        """
        Compute ACTUAL realised volatility from recent price history.

        Returns the standard deviation of 1-second log returns, annualised
        to a 5-minute window.  This replaces the hardcoded 0.10% assumption
        in the edge model.

        Returns value in percent (e.g. 0.08 means 0.08%).
        """
        if len(self.price_history) < 30:
            return 0.10  # default when not enough data

        cutoff = time.time() - lookback_seconds
        prices = [pt.price for pt in self.price_history if pt.timestamp >= cutoff]

        if len(prices) < 20:
            return 0.10

        # Log returns between consecutive samples
        log_returns = []
        for i in range(1, len(prices)):
            if prices[i - 1] > 0:
                log_returns.append(math.log(prices[i] / prices[i - 1]))

        if len(log_returns) < 10:
            return 0.10

        # Std of 1-second log returns
        mean_r = sum(log_returns) / len(log_returns)
        variance = sum((r - mean_r) ** 2 for r in log_returns) / len(log_returns)
        std_1s = math.sqrt(variance)

        # Scale to 5-minute window: vol_5min = std_1s * sqrt(300)
        vol_5min_pct = std_1s * math.sqrt(300) * 100  # in percent

        # Clamp to reasonable range
        return max(0.03, min(vol_5min_pct, 0.50))

    def price_velocity(self, lookback_seconds: int = 30) -> float:
        """
        How fast BTC price is moving right NOW (%/sec over last N seconds).
        Positive = rising, negative = falling.
        Big absolute values = strong momentum.
        """
        if len(self.price_history) < 5:
            return 0.0

        cutoff = time.time() - lookback_seconds
        recent = [pt for pt in self.price_history if pt.timestamp >= cutoff]
        if len(recent) < 3:
            return 0.0

        # Linear regression slope of prices over time
        n = len(recent)
        t0 = recent[0].timestamp
        sum_t = sum(pt.timestamp - t0 for pt in recent)
        sum_p = sum(pt.price for pt in recent)
        sum_tp = sum((pt.timestamp - t0) * pt.price for pt in recent)
        sum_t2 = sum((pt.timestamp - t0) ** 2 for pt in recent)

        denom = n * sum_t2 - sum_t ** 2
        if abs(denom) < 1e-12:
            return 0.0

        slope = (n * sum_tp - sum_t * sum_p) / denom  # $/sec
        avg_price = sum_p / n

        if avg_price <= 0:
            return 0.0

        # Return as %/sec
        return (slope / avg_price) * 100

    def price_acceleration(self) -> float:
        """
        Is the move speeding up or slowing down?
        Compares velocity over last 10s vs velocity over last 30s.
        Returns positive if accelerating in the current direction.
        """
        v_short = self.price_velocity(10)
        v_long = self.price_velocity(30)

        if abs(v_long) < 1e-9:
            return 0.0

        # Same direction and short-term is stronger = accelerating
        if v_short * v_long > 0:  # same sign
            return abs(v_short) - abs(v_long)
        else:
            return -abs(v_short)  # reversing
