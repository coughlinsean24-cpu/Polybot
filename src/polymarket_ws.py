"""
Polymarket CLOB WebSocket streaming for real-time market prices and order updates.

Replaces hot-path HTTP polling with persistent WebSocket connections:
  1. Market channel  — streams live price/book updates for subscribed markets
  2. User channel    — streams order lifecycle events (fills, cancels, etc.)

The CLOB WebSocket sends JSON messages with this general shape:
  [{"event_type": "...", "asset_id": "...", ...}]

Market events include: "price_change", "book", "tick_size_change"
User events include:   "trade", "order"

Falls back to HTTP polling if WebSocket disconnects.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass

import websockets

logger = logging.getLogger(__name__)

# Polymarket WebSocket endpoints
WS_MARKET_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
WS_USER_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/user"

# Connection settings
WS_PING_INTERVAL = 20
WS_PING_TIMEOUT = 10
WS_RECONNECT_DELAY = 2
WS_MAX_RECONNECT_DELAY = 30


@dataclass
class LiveMarketData:
    """Latest streamed market data for a single asset/token."""
    token_id: str
    price: float = 0.5
    best_bid: float = 0.0
    best_ask: float = 0.0
    timestamp: float = 0.0


@dataclass
class OrderUpdate:
    """Parsed order lifecycle event from the user channel."""
    order_id: str
    status: str  # "MATCHED", "LIVE", "CANCELLED", etc.
    size_matched: float = 0.0
    original_size: float = 0.0
    timestamp: float = 0.0
    asset_id: str = ""
    side: str = ""
    price: float = 0.0
    event_type: str = ""  # "trade" or "order"


class PolymarketMarketStream:
    """Streams real-time price data from Polymarket's market WebSocket.

    Subscribes to specific token IDs and maintains an in-memory cache
    of latest prices.  The main loop reads from this cache instead of
    hitting the gamma-api REST endpoint.
    """

    def __init__(self):
        self._subscribed_assets: set[str] = set()
        self._market_data: dict[str, LiveMarketData] = {}  # token_id -> data
        # Condensed slug-level view: slug -> {up_price, down_price, ...}
        self._slug_prices: dict[str, dict] = {}
        self._slug_token_map: dict[str, tuple[str, str]] = {}  # slug -> (up_id, down_id)
        self._connected = False
        self._running = False
        self._ws = None
        self._reconnect_delay = WS_RECONNECT_DELAY
        self._last_message_time: float = 0.0
        self._message_count: int = 0

    @property
    def connected(self) -> bool:
        return self._connected

    def register_market(self, slug: str, up_token_id: str, down_token_id: str):
        """Register a market's token IDs for streaming.

        Call this whenever you discover a new 5-min window's market.
        The stream will subscribe to these tokens automatically.
        """
        self._slug_token_map[slug] = (up_token_id, down_token_id)
        new_tokens = {up_token_id, down_token_id} - self._subscribed_assets
        if new_tokens:
            self._subscribed_assets.update(new_tokens)
            for tid in new_tokens:
                self._market_data[tid] = LiveMarketData(token_id=tid)
            logger.info(
                f"[WS-MKT] Registered {slug}: UP={up_token_id[:12]}... "
                f"DOWN={down_token_id[:12]}..."
            )
            # If already connected, send subscription for new tokens
            if self._ws and self._connected:
                asyncio.create_task(self._subscribe_tokens(new_tokens))

    def unregister_market(self, slug: str):
        """Remove a market when its window expires."""
        tokens = self._slug_token_map.pop(slug, None)
        self._slug_prices.pop(slug, None)
        if tokens:
            # Only remove tokens not used by any other slug
            all_active = set()
            for up, down in self._slug_token_map.values():
                all_active.add(up)
                all_active.add(down)
            for tid in tokens:
                if tid not in all_active:
                    self._subscribed_assets.discard(tid)
                    self._market_data.pop(tid, None)

    def get_live_prices(self, slug: str) -> dict | None:
        """Get the latest streamed prices for a market slug.

        Returns the same format as PolymarketClient.get_live_market_price():
            {
                'up_price': float,
                'down_price': float,
                'best_ask': float,
                'best_bid': float,
                'spread': float,
            }

        Returns None if we don't have data for this slug or data is stale (>10s).
        """
        tokens = self._slug_token_map.get(slug)
        if not tokens:
            return None

        up_id, down_id = tokens
        up_data = self._market_data.get(up_id)
        down_data = self._market_data.get(down_id)

        if not up_data or not down_data:
            return None

        # Check staleness -- if no update in 10s, caller should fall back to HTTP
        now = time.time()
        if up_data.timestamp > 0 and (now - up_data.timestamp) > 10:
            return None

        # If we haven't received any WS data yet for these tokens, return None
        if up_data.timestamp == 0 and down_data.timestamp == 0:
            return None

        up_price = up_data.price
        down_price = down_data.price
        best_bid = up_data.best_bid
        best_ask = up_data.best_ask
        spread = best_ask - best_bid if best_ask > best_bid else 1.0

        return {
            "up_price": up_price,
            "down_price": down_price,
            "best_ask": best_ask,
            "best_bid": best_bid,
            "spread": spread,
        }

    async def start(self):
        """Start the market WebSocket stream with auto-reconnect."""
        self._running = True
        while self._running:
            connect_start = time.time()
            try:
                await self._connect()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._connected = False
                logger.warning(
                    f"[WS-MKT] Connection error: {e} -- "
                    f"reconnecting in {self._reconnect_delay}s"
                )
            # If connection lasted < 5s, escalate backoff; otherwise reset
            if (time.time() - connect_start) < 5:
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, WS_MAX_RECONNECT_DELAY
                )
            else:
                self._reconnect_delay = WS_RECONNECT_DELAY
            await asyncio.sleep(self._reconnect_delay)

    async def stop(self):
        """Stop the stream gracefully."""
        self._running = False
        if self._ws:
            await self._ws.close()

    async def _connect(self):
        """Connect to the market WebSocket and process messages."""
        logger.info(f"[WS-MKT] Connecting to {WS_MARKET_URL}...")
        async with websockets.connect(
            WS_MARKET_URL,
            ping_interval=WS_PING_INTERVAL,
            ping_timeout=WS_PING_TIMEOUT,
        ) as ws:
            self._ws = ws
            self._connected = True
            logger.info("[WS-MKT] Connected")

            # Subscribe to all registered tokens
            if self._subscribed_assets:
                await self._subscribe_tokens(self._subscribed_assets)

            async for message in ws:
                if not self._running:
                    break
                try:
                    self._handle_market_message(message)
                except Exception as e:
                    logger.debug(f"[WS-MKT] Message parse error: {e}")

        self._connected = False
        self._ws = None

    async def _subscribe_tokens(self, token_ids: set[str]):
        """Send subscription message for the given token IDs."""
        if not self._ws:
            return

        # Polymarket WS subscription format
        sub_msg = {
            "auth": {},
            "type": "subscribe",
            "markets": list(token_ids),
            "assets_ids": list(token_ids),
        }
        try:
            await self._ws.send(json.dumps(sub_msg))
            logger.info(f"[WS-MKT] Subscribed to {len(token_ids)} token(s)")
        except Exception as e:
            logger.warning(f"[WS-MKT] Subscription failed: {e}")

    def _handle_market_message(self, raw: str):
        """Parse and apply a market WebSocket message."""
        data = json.loads(raw)
        now = time.time()
        self._last_message_time = now
        self._message_count += 1

        # Messages can be a single event dict or a list of events
        events = data if isinstance(data, list) else [data]

        for event in events:
            event_type = event.get("event_type", "")
            asset_id = event.get("asset_id", "")

            if not asset_id or asset_id not in self._market_data:
                continue

            md = self._market_data[asset_id]
            md.timestamp = now

            if event_type in ("book", "price_change"):
                # Extract price updates
                if "price" in event:
                    md.price = float(event["price"])
                if "best_bid" in event:
                    md.best_bid = float(event["best_bid"])
                if "best_ask" in event:
                    md.best_ask = float(event["best_ask"])

                # Also check for nested market data
                market_data = event.get("market", {})
                if market_data:
                    if "price" in market_data:
                        md.price = float(market_data["price"])
                    if "best_bid" in market_data:
                        md.best_bid = float(market_data["best_bid"])
                    if "best_ask" in market_data:
                        md.best_ask = float(market_data["best_ask"])

            elif event_type == "last_trade_price":
                if "price" in event:
                    md.price = float(event["price"])

            # Update condensed slug-level view
            self._update_slug_prices(asset_id, md)

    def _update_slug_prices(self, token_id: str, data: LiveMarketData):
        """Rebuild the slug-level price view after a token update."""
        for slug, (up_id, down_id) in self._slug_token_map.items():
            if token_id in (up_id, down_id):
                up_data = self._market_data.get(up_id)
                down_data = self._market_data.get(down_id)
                if up_data and down_data:
                    self._slug_prices[slug] = {
                        "up_price": up_data.price,
                        "down_price": down_data.price,
                        "best_ask": up_data.best_ask,
                        "best_bid": up_data.best_bid,
                        "spread": (
                            up_data.best_ask - up_data.best_bid
                            if up_data.best_ask > up_data.best_bid
                            else 1.0
                        ),
                    }

    def get_stats(self) -> dict:
        """Return stream health stats."""
        return {
            "connected": self._connected,
            "subscribed_tokens": len(self._subscribed_assets),
            "registered_slugs": len(self._slug_token_map),
            "total_messages": self._message_count,
            "last_message_age": (
                time.time() - self._last_message_time
                if self._last_message_time > 0
                else None
            ),
        }


class PolymarketUserStream:
    """Streams order lifecycle events from Polymarket's user WebSocket.

    Provides real-time order fills, cancellations, and status changes
    instead of polling check_position_status().
    """

    def __init__(self, api_key: str = "", api_secret: str = "",
                 passphrase: str = ""):
        self._api_key = api_key
        self._api_secret = api_secret
        self._passphrase = passphrase
        self._connected = False
        self._running = False
        self._ws = None
        self._reconnect_delay = WS_RECONNECT_DELAY
        self._last_message_time: float = 0.0
        self._message_count: int = 0

        # Order state cache: order_id -> latest status
        self._order_states: dict[str, OrderUpdate] = {}
        # Callbacks for order updates
        self._callbacks: list = []

    @property
    def connected(self) -> bool:
        return self._connected

    def on_order_update(self, callback):
        """Register a callback for order updates: callback(OrderUpdate)."""
        self._callbacks.append(callback)

    def get_order_status(self, order_id: str) -> OrderUpdate | None:
        """Get the latest known status for an order from the WS cache.

        Returns None if the order hasn't been seen via WebSocket.
        """
        return self._order_states.get(order_id)

    def is_order_filled(self, order_id: str) -> bool | None:
        """Quick check if an order has been fully filled.

        Returns True/False, or None if we have no WS data for this order.
        """
        update = self._order_states.get(order_id)
        if update is None:
            return None
        return update.status in ("MATCHED", "FILLED", "CLOSED")

    async def start(self):
        """Start the user WebSocket stream with auto-reconnect."""
        self._running = True
        while self._running:
            connect_start = time.time()
            try:
                await self._connect()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._connected = False
                logger.warning(
                    f"[WS-USR] Connection error: {e} -- "
                    f"reconnecting in {self._reconnect_delay}s"
                )
            # If connection lasted < 5s, escalate backoff; otherwise reset
            if (time.time() - connect_start) < 5:
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, WS_MAX_RECONNECT_DELAY
                )
            else:
                self._reconnect_delay = WS_RECONNECT_DELAY
            await asyncio.sleep(self._reconnect_delay)

    async def stop(self):
        """Stop the stream gracefully."""
        self._running = False
        if self._ws:
            await self._ws.close()

    async def _connect(self):
        """Connect to the user WebSocket and process messages."""
        logger.info(f"[WS-USR] Connecting to {WS_USER_URL}...")

        extra_headers = {}
        if self._api_key:
            extra_headers["Authorization"] = f"Bearer {self._api_key}"

        async with websockets.connect(
            WS_USER_URL,
            ping_interval=WS_PING_INTERVAL,
            ping_timeout=WS_PING_TIMEOUT,
            additional_headers=extra_headers if extra_headers else None,
        ) as ws:
            self._ws = ws
            self._connected = True
            logger.info("[WS-USR] Connected")

            # Send auth + subscribe message
            if self._api_key:
                auth_payload = {
                    "apiKey": self._api_key,
                }
                if self._api_secret:
                    auth_payload["secret"] = self._api_secret
                if self._passphrase:
                    auth_payload["passphrase"] = self._passphrase
                auth_msg = {
                    "auth": auth_payload,
                    "type": "subscribe",
                    "channel": "user",
                }
                await ws.send(json.dumps(auth_msg))
                logger.info("[WS-USR] Auth+subscribe sent")

            async for message in ws:
                if not self._running:
                    break
                try:
                    self._handle_user_message(message)
                except Exception as e:
                    logger.debug(f"[WS-USR] Message parse error: {e}")

        self._connected = False
        self._ws = None

    def _handle_user_message(self, raw: str):
        """Parse and apply a user WebSocket message."""
        data = json.loads(raw)
        now = time.time()
        self._last_message_time = now
        self._message_count += 1

        events = data if isinstance(data, list) else [data]

        for event in events:
            event_type = event.get("event_type", event.get("type", ""))
            order_id = event.get("id", event.get("order_id", ""))

            if not order_id:
                continue

            update = OrderUpdate(
                order_id=order_id,
                status=event.get("status", "unknown").upper(),
                size_matched=float(event.get("size_matched", 0)),
                original_size=float(event.get("original_size", event.get("size", 0))),
                timestamp=now,
                asset_id=event.get("asset_id", ""),
                side=event.get("side", ""),
                price=float(event.get("price", 0)),
                event_type=event_type,
            )

            self._order_states[order_id] = update

            # Fire callbacks
            for cb in self._callbacks:
                try:
                    cb(update)
                except Exception as e:
                    logger.warning(f"[WS-USR] Callback error: {e}")

            logger.debug(
                f"[WS-USR] Order {order_id[:12]}... -> "
                f"{update.status} (matched={update.size_matched})"
            )

    def get_stats(self) -> dict:
        """Return stream health stats."""
        return {
            "connected": self._connected,
            "tracked_orders": len(self._order_states),
            "total_messages": self._message_count,
            "last_message_age": (
                time.time() - self._last_message_time
                if self._last_message_time > 0
                else None
            ),
        }

    def cleanup_old_orders(self, max_age: float = 600):
        """Remove order states older than max_age seconds."""
        now = time.time()
        stale = [
            oid for oid, update in self._order_states.items()
            if (now - update.timestamp) > max_age
        ]
        for oid in stale:
            del self._order_states[oid]
        if stale:
            logger.debug(f"[WS-USR] Cleaned up {len(stale)} stale order states")
