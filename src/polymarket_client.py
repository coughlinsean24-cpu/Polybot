"""
Polymarket CLOB API client for interacting with 5-minute BTC Up/Down markets.
Handles authentication, market discovery via gamma-api, order placement, and
position management.

Market structure (discovered via gamma-api.polymarket.com):
  - Series slug: "btc-up-or-down-5m"
  - Event slugs: "btc-updown-5m-{unix_timestamp}" where timestamp = window start
  - Each event has ONE market with outcomes ["Up", "Down"]
  - clobTokenIds[0] = Up token, clobTokenIds[1] = Down token
  - New markets roll every 300 seconds on exact boundaries
"""

import json
import math
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import requests
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType

logger = logging.getLogger(__name__)

POLYMARKET_HOST = "https://clob.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"
MARKET_WINDOW_SECONDS = 300  # 5 minutes

# Simple in-memory caches keyed by slug -- avoids repeated API/scrape calls
_market_cache: dict[str, "MarketInfo | None"] = {}
_price_to_beat_cache: dict[str, float | None] = {}
_price_to_beat_neg_cache: dict[str, float] = {}   # slug -> time.time() of last failure
_PRICE_NEG_CACHE_TTL = 30.0                        # don't re-scrape a failing slug for 30s
_close_price_cache: dict[str, float | None] = {}

_market_slug_cache: dict[str, tuple["MarketInfo | None", float]] = {}
_MARKET_SLUG_TTL = 15.0  # seconds -- same window, same market info

# Shared requests.Session for connection pooling (reuses TCP connections)
_http_session: requests.Session | None = None

# WebSocket streams (set by Orchestrator on startup)
_market_stream = None  # PolymarketMarketStream instance
_user_stream = None    # PolymarketUserStream instance


def set_ws_streams(market_stream=None, user_stream=None):
    """Inject WebSocket stream references (called by Orchestrator on startup)."""
    global _market_stream, _user_stream
    if market_stream is not None:
        _market_stream = market_stream
    if user_stream is not None:
        _user_stream = user_stream


def _get_http_session() -> requests.Session:
    """Return a shared requests.Session for connection pooling."""
    global _http_session
    if _http_session is None:
        _http_session = requests.Session()
        _http_session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })
    return _http_session


@dataclass
class MarketInfo:
    """Represents a single BTC Up/Down 5-minute market."""
    condition_id: str
    question: str
    up_token_id: str
    down_token_id: str
    up_price: float
    down_price: float
    end_date: str
    volume: float
    event_start_time: str  # ISO string: when the 5-min window opens
    slug: str  # e.g. "btc-updown-5m-1771179600"
    best_bid: float = 0.0
    best_ask: float = 0.0

    # Keep backward-compatible aliases so ArbitrageEngine doesn't break
    @property
    def yes_token_id(self) -> str:
        return self.up_token_id

    @property
    def no_token_id(self) -> str:
        return self.down_token_id

    @property
    def yes_price(self) -> float:
        return self.up_price

    @property
    def no_price(self) -> float:
        return self.down_price


@dataclass
class OrderResult:
    success: bool
    order_id: str | None = None
    fill_price: float | None = None
    error: str | None = None
    actual_cost: float | None = None
    matched: bool = False  # True if order was immediately filled


class PolymarketClient:
    def __init__(
        self,
        config: dict,
        api_key: str,
        api_secret: str,
        passphrase: str,
        private_key: str,
    ):
        self.chain_id = config["polymarket"]["chain_id"]
        self.max_slippage = config["strategy"]["max_slippage"]
        self._api_key = api_key
        self._api_secret = api_secret
        self._passphrase = passphrase

        self.client = ClobClient(
            host=POLYMARKET_HOST,
            key=private_key,
            chain_id=self.chain_id,
            funder=config.get("polymarket", {}).get("funder_address", None),
            signature_type=config.get("polymarket", {}).get("signature_type", None),
        )
        self._authenticated = False

    # -- Authentication ----------------------------------------------

    def authenticate(self) -> bool:
        """Authenticate with Polymarket CLOB API.

        Derives API credentials directly from the wallet private key.
        This is the most reliable approach -- it guarantees the API key,
        secret, and passphrase match the signing wallet, even if the
        values stored in .env are stale or from a different wallet.
        """
        try:
            derived_creds = self.client.derive_api_key()
            self.client.set_api_creds(derived_creds)
            self._authenticated = True
            logger.info("Polymarket authentication successful (derived from wallet)")
            return True
        except Exception as e:
            logger.warning(f"derive_api_key failed ({e}), falling back to stored creds")

        # Fallback: use the creds from .env
        try:
            from py_clob_client.clob_types import ApiCreds

            creds = ApiCreds(
                api_key=self._api_key,
                api_secret=self._api_secret,
                api_passphrase=self._passphrase,
            )
            self.client.set_api_creds(creds)
            self._authenticated = True
            logger.info("Polymarket authentication successful (stored creds)")
            return True
        except Exception as e:
            logger.error(f"Polymarket authentication failed: {e}")
            return False

    # -- Balance ---------------------------------------------------

    def get_live_balance(self) -> float | None:
        """Query the real USDC balance available for trading on Polymarket.

        Returns the collateral balance in dollars, or None on error.
        """
        try:
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            result = self.client.get_balance_allowance(params)
            raw = int(result.get("balance", "0"))
            return raw / 1e6
        except Exception as e:
            logger.error(f"Failed to fetch live balance: {e}")
            return None

    def get_position_balance(self, token_id: str) -> float:
        """Return number of shares held for a specific conditional token.

        Returns 0.0 if no shares or on any error.
        """
        try:
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
            params = BalanceAllowanceParams(
                asset_type=AssetType.CONDITIONAL,
                token_id=token_id,
            )
            resp = self.client.get_balance_allowance(params)
            raw = float(resp.get("balance", 0)) if isinstance(resp, dict) else 0
            # Balance API always returns micro-units (6 decimals),
            # same as collateral.  Always divide by 1e6.
            return raw / 1e6
        except Exception:
            return 0.0

    # -- Market Discovery (gamma-api) --------------------------------

    @staticmethod
    def _current_window_ts() -> int:
        """Return the unix timestamp for the START of the current 5-min window."""
        return int(time.time() // MARKET_WINDOW_SECONDS) * MARKET_WINDOW_SECONDS

    @staticmethod
    def _next_window_ts() -> int:
        """Return the unix timestamp for the START of the next 5-min window."""
        return PolymarketClient._current_window_ts() + MARKET_WINDOW_SECONDS

    @staticmethod
    def _fetch_market_by_slug(slug: str) -> MarketInfo | None:
        """Fetch a single BTC Up/Down market from gamma-api by its slug.

        Cached with a 15s TTL -- same 5-min window returns identical data.
        """
        now = time.time()
        cached = _market_slug_cache.get(slug)
        if cached and (now - cached[1]) < _MARKET_SLUG_TTL:
            return cached[0]

        try:
            resp = _get_http_session().get(
                f"{GAMMA_API}/events",
                params={"slug": slug},
                timeout=5,
            )
            resp.raise_for_status()
            events = resp.json()
            if not events:
                return None

            event = events[0]
            markets = event.get("markets", [])
            if not markets:
                return None

            mkt = markets[0]
            token_ids = json.loads(mkt["clobTokenIds"])
            prices = json.loads(mkt.get("outcomePrices", '["0.5","0.5"]'))

            info = MarketInfo(
                condition_id=mkt["conditionId"],
                question=mkt["question"],
                up_token_id=token_ids[0],
                down_token_id=token_ids[1],
                up_price=float(prices[0]),
                down_price=float(prices[1]),
                end_date=mkt.get("endDate", ""),
                volume=float(mkt.get("volumeNum", 0)),
                event_start_time=mkt.get("eventStartTime", ""),
                slug=slug,
                best_bid=float(mkt.get("bestBid", 0)),
                best_ask=float(mkt.get("bestAsk", 0)),
            )
            _market_slug_cache[slug] = (info, time.time())
            return info
        except Exception as e:
            logger.warning(f"Failed to fetch market {slug}: {e}")
            _market_slug_cache[slug] = (None, time.time())
            return None

    @staticmethod
    def _slug_to_iso_start(slug: str) -> str | None:
        """Convert a slug like 'btc-updown-5m-1771187100' to ISO start time.

        Returns e.g. '2026-02-15T20:25:00Z' -- the key Polymarket uses in its
        SSR React-Query dehydrated state for this specific window.
        """
        match = re.search(r"(\d{10})$", slug)
        if not match:
            return None
        ts = int(match.group(1))
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _scrape_event_page(slug: str) -> str | None:
        """Fetch the full SSR HTML for an event page. Returns page text."""
        try:
            resp = _get_http_session().get(
                f"https://polymarket.com/event/{slug}",
                timeout=8,
            )
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            logger.warning(f"Failed to fetch event page for {slug}: {e}")
            return None

    @staticmethod
    def fetch_price_to_beat(slug: str) -> float | None:
        """
        Fetch the Chainlink-sourced "price to beat" (openPrice) for a market.

        The SSR dehydrated React-Query state contains a window-specific key:
            ["past-results","BTC","fiveminute","2026-02-15T20:25:00Z"]
        with {"openPrice": ..., "closePrice": ...} for that exact window.

        We first look for the window-specific key (most reliable), then
        fall back to matching the results array by startTime, and finally
        fall back to any openPrice on the page.

        Results are cached per slug.
        """
        if slug in _price_to_beat_cache:
            cached = _price_to_beat_cache[slug]
            if cached is not None:
                logger.debug(f"Price to beat for {slug}: ${cached:,.2f} (cached)")
                return cached

        # Negative cache: don't hammer a slug that recently failed
        neg_ts = _price_to_beat_neg_cache.get(slug)
        if neg_ts is not None and (time.time() - neg_ts) < _PRICE_NEG_CACHE_TTL:
            return None

        iso_start = PolymarketClient._slug_to_iso_start(slug)
        page = PolymarketClient._scrape_event_page(slug)
        if page is None:
            _price_to_beat_neg_cache[slug] = time.time()
            return None

        # Strategy 1: Match the window-specific past-results key
        # e.g. "past-results","BTC","fiveminute","2026-02-15T20:25:00Z"
        if iso_start:
            pattern = re.escape(f'"past-results","BTC","fiveminute","{iso_start}"')
            match = re.search(pattern + r'.*?"openPrice":([\d.]+)', page)
            if match:
                price = float(match.group(1))
                _price_to_beat_cache[slug] = price
                logger.info(f"Price to beat for {slug}: ${price:,.2f} (Chainlink)")
                return price

        # Strategy 2: Parse the results array and find entry by startTime
        if iso_start:
            # The results array has entries with startTime matching our window
            # Convert our iso_start (no ms) to match the page format (with ms)
            iso_prefix = iso_start.replace("Z", "")
            results_match = re.search(r'"results":\[(.*?)\]\}', page)
            if results_match:
                try:
                    results = json.loads("[" + results_match.group(1) + "]")
                    for entry in results:
                        if entry.get("startTime", "").startswith(iso_prefix):
                            price = float(entry["openPrice"])
                            _price_to_beat_cache[slug] = price
                            logger.info(
                                f"Price to beat for {slug}: ${price:,.2f} (results array)"
                            )
                            return price
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass

        # Strategy 3: Fallback -- first openPrice near any past-results key
        match = re.search(
            r'"past-results","BTC","fiveminute".*?"openPrice":([\d.]+)', page
        )
        if match:
            price = float(match.group(1))
            _price_to_beat_cache[slug] = price
            logger.info(f"Price to beat for {slug}: ${price:,.2f} (page fallback)")
            return price

        logger.warning(f"No openPrice found in page for {slug}")
        return None

    @staticmethod
    def fetch_close_price(slug: str) -> float | None:
        """
        Fetch the Chainlink-sourced close price for a resolved market.

        IMPORTANT: The page contains multiple windows' data.  We MUST match
        our specific window by its startTime to avoid grabbing the wrong
        closePrice.

        The most reliable source is the window-specific React-Query key:
            ["past-results","BTC","fiveminute","2026-02-15T20:25:00Z"]
        which contains {"openPrice": ..., "closePrice": ...} for our window.

        Returns the closePrice float, or None if not yet available.
        """
        if slug in _close_price_cache:
            cached = _close_price_cache[slug]
            if cached is not None:
                return cached

        iso_start = PolymarketClient._slug_to_iso_start(slug)
        page = PolymarketClient._scrape_event_page(slug)
        if page is None:
            return None

        # Strategy 1: Match the window-specific past-results key
        if iso_start:
            pattern = re.escape(f'"past-results","BTC","fiveminute","{iso_start}"')
            match = re.search(pattern + r'.*?"closePrice":([\d.]+)', page)
            if match:
                price = float(match.group(1))
                _close_price_cache[slug] = price
                logger.info(f"Close price for {slug}: ${price:,.2f} (Chainlink)")
                return price

        # Strategy 2: Parse results array, find our window by startTime
        if iso_start:
            iso_prefix = iso_start.replace("Z", "")
            results_match = re.search(r'"results":\[(.*?)\]\}', page)
            if results_match:
                try:
                    results = json.loads("[" + results_match.group(1) + "]")
                    for entry in results:
                        if entry.get("startTime", "").startswith(iso_prefix):
                            price = float(entry["closePrice"])
                            _close_price_cache[slug] = price
                            logger.info(
                                f"Close price for {slug}: ${price:,.2f} (results array)"
                            )
                            return price
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass

        # NO generic fallback for closePrice -- grabbing a random closePrice
        # from the page is worse than returning None and retrying, because
        # it will resolve our trade against the wrong window's data.
        logger.debug(f"Close price not yet available for {slug}")
        return None

    # -- Live Market Prices (via gamma-api) --------------------------
    #
    # IMPORTANT: The raw CLOB order book per-token only shows that
    # token's native orders (e.g. UP book has bids at 1c, no asks).
    # However Polymarket uses neg-risk complement matching: buying UP
    # at 0.49 is matched against selling DOWN at 0.51 under the hood.
    # The gamma-api's bestAsk/bestBid/outcomePrices reflect the REAL
    # tradeable price including complement matching.  Always use those.

    @staticmethod
    def get_live_market_price(slug: str) -> dict | None:
        """Get the latest market prices -- WebSocket first, HTTP fallback.

        Tries the WS market stream for instant, zero-latency data.
        Falls back to HTTP polling if WS is not connected or data is stale.

        Returns a dict with the real tradeable prices::

            {
                'up_price': float,     # mid-market for UP
                'down_price': float,   # mid-market for DOWN
                'best_ask': float,     # cheapest you can BUY UP (incl. complement)
                'best_bid': float,     # best you can SELL UP
                'spread': float,       # ask - bid
            }

        These prices account for complement matching -- the gamma-api
        aggregates both sides of the book.  A BUY UP at best_ask WILL
        fill immediately.
        """
        # --- Try WebSocket stream first (sub-second latency) ---
        if _market_stream and _market_stream.connected:
            ws_data = _market_stream.get_live_prices(slug)
            if ws_data:
                logger.debug(
                    f"[WS] Live prices [{slug}]: UP={ws_data['up_price']:.3f} "
                    f"ask={ws_data['best_ask']:.3f} bid={ws_data['best_bid']:.3f}"
                )
                return ws_data

        # --- Fall back to HTTP polling ---
        try:
            resp = _get_http_session().get(
                f"{GAMMA_API}/events",
                params={"slug": slug},
                timeout=4,
            )
            resp.raise_for_status()
            events = resp.json()
            if not events:
                return None

            mkt = events[0].get("markets", [{}])[0]
            prices = json.loads(mkt.get("outcomePrices", '["0.5","0.5"]'))

            result = {
                "up_price": float(prices[0]),
                "down_price": float(prices[1]),
                "best_ask": float(mkt.get("bestAsk", 0)),
                "best_bid": float(mkt.get("bestBid", 0)),
                "spread": float(mkt.get("spread", 1.0)),
            }

            logger.debug(
                f"[HTTP] Live prices [{slug}]: UP={result['up_price']:.3f} "
                f"ask={result['best_ask']:.3f} bid={result['best_bid']:.3f} "
                f"spread={result['spread']:.3f}"
            )
            return result

        except Exception as e:
            logger.warning(f"Failed to fetch live prices for {slug}: {e}")
            return None

    @staticmethod
    def check_clob_book(token_id: str, max_price: float = 0.95) -> dict | None:
        """Query the real CLOB order book for a token.

        Returns the best ask and total available size at or below max_price.
        This bypasses the gamma API which can report stale prices.

        Returns::

            {
                'best_ask': float | None,   # lowest ask price, or None if empty
                'ask_size': float,           # total askable shares <= max_price
                'best_bid': float | None,    # highest bid price, or None if empty
                'bid_size': float,           # total biddable shares
                'liquid': bool,              # True if there's real tradeable liquidity
            }
        """
        try:
            import httpx
            resp = httpx.get(
                f"{POLYMARKET_HOST}/book",
                params={"token_id": token_id},
                timeout=4,
            )
            resp.raise_for_status()
            data = resp.json()

            asks = data.get("asks", [])
            bids = data.get("bids", [])

            # Filter asks to those at or below our max willingness to pay
            valid_asks = [
                a for a in asks
                if float(a.get("price", 999)) <= max_price
            ]
            valid_asks.sort(key=lambda a: float(a["price"]))

            best_ask = float(valid_asks[0]["price"]) if valid_asks else None
            ask_size = sum(float(a.get("size", 0)) for a in valid_asks)

            best_bid = float(bids[0]["price"]) if bids else None
            bid_size = sum(float(b.get("size", 0)) for b in bids[:10])

            # "Liquid" means there's at least one ask below 0.90 with at
            # least 5 shares available (the minimum order size)
            liquid = best_ask is not None and best_ask < 0.90 and ask_size >= 5

            logger.debug(
                f"[BOOK] token={token_id[:12]}... "
                f"best_ask={best_ask} ask_size={ask_size:.1f} "
                f"best_bid={best_bid} bid_size={bid_size:.1f} "
                f"liquid={liquid}"
            )
            return {
                "best_ask": best_ask,
                "ask_size": ask_size,
                "best_bid": best_bid,
                "bid_size": bid_size,
                "liquid": liquid,
            }
        except Exception as e:
            logger.warning(f"Failed to check CLOB book: {e}")
            return None

    # -- Order Execution ---------------------------------------------

    def place_order(
        self, market: MarketInfo, side: str, size: float, price: float,
        order_type: str = "GTC",
    ) -> OrderResult:
        """
        Place an order on a Polymarket BTC Up/Down market.

        Uses GTC (Good-Till-Cancelled) by default: the order rests on the
        book until filled.  This is better for thin 5-min market books
        where FOK often fails because there isn't enough liquidity to fill
        instantly.

        Args:
            market: The market to trade on
            side: "UP" (or "YES") to bet Up, "DOWN" (or "NO") to bet Down
            size: Dollar amount to risk
            price: Max price willing to pay (0-1)
            order_type: "GTC" (default), "FOK", or "GTD"
        """
        if not self._authenticated:
            return OrderResult(success=False, error="Not authenticated")

        # Map side to the correct token
        if side in ("UP", "YES"):
            token_id = market.up_token_id
        else:
            token_id = market.down_token_id

        # Round price to valid tick size (0.01 for most markets, 0.001 for some)
        # Use 0.01 as default -- Polymarket will reject invalid ticks
        tick_size = 0.01
        price = round(price / tick_size) * tick_size
        price = max(tick_size, min(price, 1.0 - tick_size))

        try:
            shares = size / price

            # Polymarket enforces a minimum order size of 5 shares
            MIN_SHARES = 5
            if shares < MIN_SHARES:
                shares = MIN_SHARES
                size = round(shares * price, 2)
                logger.info(f"Bumped order to minimum {MIN_SHARES} shares (cost ${size:.2f})")

            shares = math.floor(shares * 10000) / 10000  # truncate to 4 dp
            actual_cost = round(shares * price, 2)

            # If MIN_SHARES bump pushed cost above the intended size,
            # cap shares so we never spend more than requested.
            if actual_cost > size:
                shares = size / price
                shares = math.floor(shares * 10000) / 10000
                actual_cost = round(shares * price, 2)
                if shares < MIN_SHARES:
                    logger.warning(
                        f"Cannot place order: ${size:.2f} at {price:.3f} = "
                        f"{shares:.1f} shares (below minimum {MIN_SHARES})"
                    )
                    return OrderResult(success=False, error="Below minimum order size")

            if order_type == "FOK":
                # FOK uses MarketOrderArgs with amount (USDC) -- the library's
                # create_market_order handles precision correctly (maker to 2dp).
                from py_clob_client.clob_types import MarketOrderArgs
                market_args = MarketOrderArgs(
                    token_id=token_id,
                    amount=round(size, 2),  # USDC to spend
                    side="BUY",
                    price=price,
                )
                signed_order = self.client.create_market_order(market_args)
                result = self.client.post_order(signed_order, OrderType.FOK)
            else:
                # GTC uses OrderArgs with size (shares)
                order_args = OrderArgs(
                    price=price,
                    size=shares,
                    side="BUY",
                    token_id=token_id,
                )
                signed_order = self.client.create_order(order_args)
                result = self.client.post_order(signed_order, OrderType.GTC)

            order_id = result.get("orderID", result.get("id"))

            # Check if FOK was killed (not filled)
            status = result.get("status", "")
            error_msg = result.get("errorMsg", "")
            if error_msg:
                logger.warning(
                    f"Order error: {error_msg} (status={status})"
                )
                return OrderResult(success=False, error=error_msg)

            is_matched = status.upper() in ("MATCHED", "FILLED")
            logger.info(
                f"Order placed: {side} ${actual_cost:.2f} ({shares:.1f} shares) @ {price:.3f} "
                f"({order_type}) -> order_id={order_id} status={status}"
            )

            return OrderResult(
                success=True,
                order_id=order_id,
                fill_price=price,
                actual_cost=actual_cost,
                matched=is_matched,
            )

        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            return OrderResult(success=False, error=str(e))

    # -- Position Management -----------------------------------------

    def check_position_status(self, order_id: str) -> dict:
        """Check the status of an existing order/position.

        Tries WebSocket cache first for instant results, falls back
        to HTTP if no WS data is available.
        """
        # --- Try WebSocket user stream cache first ---
        if _user_stream and _user_stream.connected:
            ws_update = _user_stream.get_order_status(order_id)
            if ws_update:
                logger.debug(
                    f"[WS] Order {order_id[:12]}... -> "
                    f"{ws_update.status} (matched={ws_update.size_matched})"
                )
                return {
                    "status": ws_update.status.lower(),
                    "filled": ws_update.size_matched,
                    "remaining": ws_update.original_size - ws_update.size_matched,
                }

        # --- Fall back to HTTP ---
        try:
            order = self.client.get_order(order_id)
            return {
                "status": order.get("status", "unknown"),
                "filled": float(order.get("size_matched", 0)),
                "remaining": float(order.get("original_size", 0))
                - float(order.get("size_matched", 0)),
            }
        except Exception as e:
            logger.error(f"Failed to check position: {e}")
            return {"status": "error", "error": str(e)}

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        try:
            self.client.cancel(order_id)
            logger.info(f"Order {order_id} cancelled")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def sell_position(
        self, market: MarketInfo, side: str, shares: float, price: float,
        order_type: str = "GTC",
    ) -> OrderResult:
        """Sell shares to take profit or cut losses.

        Places a SELL order at the given price.  For instant fills, use
        the gamma-api best_bid as the price (the best someone will pay
        for your shares).

        Args:
            market: The market to trade on
            side: "UP" or "DOWN" -- which token to sell
            shares: Number of shares to sell
            price: Min price to sell at (0-1)
            order_type: "GTC" (default) or "FOK"
        """
        if not self._authenticated:
            return OrderResult(success=False, error="Not authenticated")

        if side in ("UP", "YES"):
            token_id = market.up_token_id
        else:
            token_id = market.down_token_id

        tick_size = 0.01
        price = round(price / tick_size) * tick_size
        price = max(tick_size, min(price, 1.0 - tick_size))

        try:
            MIN_SHARES = 5
            if shares < MIN_SHARES:
                logger.warning(
                    f"Cannot sell {shares:.2f} shares (min {MIN_SHARES}) -- "
                    f"will hold to resolution"
                )
                return OrderResult(success=False, error="Below minimum shares")

            # Approve conditional token for selling (required after buy settles)
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
            try:
                params = BalanceAllowanceParams(
                    asset_type=AssetType.CONDITIONAL,
                    token_id=token_id,
                )
                self.client.update_balance_allowance(params)
            except Exception as e:
                logger.debug(f"update_balance_allowance: {e}")

            # Check actual conditional token balance before attempting sell.
            # Polymarket conditional tokens use 6 decimals on Polygon, so
            # the raw balance is in micro-units (balance / 1e6 = shares).
            try:
                bal_params = BalanceAllowanceParams(
                    asset_type=AssetType.CONDITIONAL,
                    token_id=token_id,
                )
                bal_resp = self.client.get_balance_allowance(bal_params)
                raw_bal = float(bal_resp.get("balance", 0)) if isinstance(bal_resp, dict) else 0

                # Determine if raw_bal is in micro-units or direct shares
                # If raw_bal is > 1000x shares, it's almost certainly in micro-units
                if raw_bal > shares * 1000:
                    actual_shares = raw_bal / 1e6
                else:
                    actual_shares = raw_bal

                logger.info(
                    f"Conditional token balance: raw={raw_bal:.0f} "
                    f"(~{actual_shares:.4f} shares, need {shares:.4f})"
                )

                if actual_shares < shares:
                    if actual_shares >= 5.0:  # above minimum, sell what we have
                        logger.info(
                            f"Adjusting sell from {shares:.4f} -> {actual_shares:.4f} shares "
                            f"(on-chain balance)"
                        )
                        shares = math.floor(actual_shares * 100) / 100  # round down
                    else:
                        return OrderResult(
                            success=False,
                            error=f"not enough balance / allowance "
                                  f"(have {actual_shares:.4f}, need {shares:.4f})"
                        )
            except Exception as e:
                logger.debug(f"get_balance_allowance: {e}")

            otype = OrderType.GTC if order_type == "GTC" else OrderType.FOK

            order_args = OrderArgs(
                price=price,
                size=shares,
                side="SELL",
                token_id=token_id,
            )

            signed_order = self.client.create_order(order_args)
            result = self.client.post_order(signed_order, otype)

            order_id = result.get("orderID", result.get("id"))
            status = result.get("status", "")
            error_msg = result.get("errorMsg", "")

            if error_msg:
                logger.warning(f"Sell order error: {error_msg} (status={status})")
                return OrderResult(success=False, error=error_msg)

            proceeds = shares * price
            logger.info(
                f"Sell order placed: {side} {shares:.2f} shares @ {price:.3f} "
                f"(proceeds~${proceeds:.2f}) -> order_id={order_id} status={status}"
            )

            return OrderResult(
                success=True,
                order_id=order_id,
                fill_price=price,
            )

        except Exception as e:
            logger.error(f"Sell order failed: {e}")
            return OrderResult(success=False, error=str(e))

    def place_limit_order(
        self, market: MarketInfo, side: str, shares: float, price: float,
    ) -> OrderResult:
        """Place a GTC limit order (resting on the book until filled or cancelled).

        Unlike place_order() which uses FOK (fill-or-kill), this places a
        resting order at a specific price. Used for straddle strategy where
        we want to buy cheap shares that may or may not fill.

        Args:
            market: The market to trade on
            side: "UP" / "YES" or "DOWN" / "NO"
            shares: Number of shares (not dollar amount)
            price: Limit price per share (e.g. 0.05)
        """
        if not self._authenticated:
            return OrderResult(success=False, error="Not authenticated")

        if side in ("UP", "YES"):
            token_id = market.up_token_id
        else:
            token_id = market.down_token_id

        tick_size = 0.01
        price = round(price / tick_size) * tick_size
        price = max(tick_size, min(price, 1.0 - tick_size))

        try:
            order_args = OrderArgs(
                price=price,
                size=shares,
                side="BUY",
                token_id=token_id,
            )

            signed_order = self.client.create_order(order_args)
            result = self.client.post_order(signed_order, OrderType.GTC)

            order_id = result.get("orderID", result.get("id"))
            status = result.get("status", "")
            error_msg = result.get("errorMsg", "")

            if error_msg:
                logger.warning(f"Limit order error: {error_msg}")
                return OrderResult(success=False, error=error_msg)

            cost = shares * price
            logger.info(
                f"Limit order placed: {side} {shares:.0f} shares @ ${price:.2f} "
                f"(cost=${cost:.2f}) -> order_id={order_id} status={status}"
            )

            return OrderResult(
                success=True,
                order_id=order_id,
                fill_price=price,
            )

        except Exception as e:
            logger.error(f"Limit order placement failed: {e}")
            return OrderResult(success=False, error=str(e))

    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count of cancelled orders."""
        try:
            result = self.client.cancel_all()
            cancelled = result.get("canceled", []) if isinstance(result, dict) else []
            count = len(cancelled) if isinstance(cancelled, list) else 0
            logger.info(f"Cancelled {count} open orders")
            return count
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return 0

    def get_open_orders(self) -> list[dict]:
        """Get all open/resting orders."""
        try:
            orders = self.client.get_orders()
            open_orders = [
                o for o in orders
                if o.get("status", "").upper() in ("LIVE", "OPEN", "ACTIVE")
            ]
            return open_orders
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []