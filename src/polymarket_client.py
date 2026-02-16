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

# Simple in-memory caches keyed by slug — avoids repeated API/scrape calls
_market_cache: dict[str, "MarketInfo | None"] = {}
_price_to_beat_cache: dict[str, float | None] = {}
_close_price_cache: dict[str, float | None] = {}

# TTL caches for hot-path calls (keyed by slug → (value, timestamp))
_live_price_cache: dict[str, tuple[dict | None, float]] = {}
_LIVE_PRICE_TTL = 0.0  # Always fetch fresh — 5-min markets move too fast for caching

_market_slug_cache: dict[str, tuple["MarketInfo | None", float]] = {}
_MARKET_SLUG_TTL = 15.0  # seconds — same window, same market info

# Shared requests.Session for connection pooling (reuses TCP connections)
_http_session: requests.Session | None = None


def _get_http_session() -> requests.Session:
    """Return a shared requests.Session for connection pooling."""
    global _http_session
    if _http_session is None:
        _http_session = requests.Session()
        _http_session.headers.update({"User-Agent": "Mozilla/5.0"})
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

    # ── Authentication ──────────────────────────────────────────────

    def authenticate(self) -> bool:
        """Authenticate with Polymarket CLOB API.

        Derives API credentials directly from the wallet private key.
        This is the most reliable approach — it guarantees the API key,
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

    # ── Balance ───────────────────────────────────────────────────

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

    # ── Market Discovery (gamma-api) ────────────────────────────────

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

        Cached with a 15s TTL — same 5-min window returns identical data.
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

    def get_active_5min_markets(self) -> list[MarketInfo]:
        """
        Fetch the current and next BTC Up/Down 5-min markets.

        Returns up to 2 markets: the one currently in-play and the upcoming one.
        The CLOB API get_markets() endpoint does NOT surface these rolling
        markets, so we query gamma-api.polymarket.com using the predictable
        slug pattern:  btc-updown-5m-{unix_timestamp}
        """
        found: list[MarketInfo] = []

        for ts in [self._current_window_ts(), self._next_window_ts()]:
            slug = f"btc-updown-5m-{ts}"
            mkt = self._fetch_market_by_slug(slug)
            if mkt:
                # Only include markets that are still accepting orders
                found.append(mkt)
                logger.debug(
                    f"Found market: {mkt.question}  Up={mkt.up_price:.3f} "
                    f"Down={mkt.down_price:.3f}"
                )

        logger.info(f"Found {len(found)} active 5-min BTC Up/Down markets")
        return found

    def get_next_market(self) -> MarketInfo | None:
        """
        Get the NEXT (upcoming) 5-min market that hasn't started yet.
        This is the one we want to trade on — place orders before it begins.
        """
        slug = f"btc-updown-5m-{self._next_window_ts()}"
        mkt = self._fetch_market_by_slug(slug)
        if mkt:
            logger.info(
                f"Next market: {mkt.question}  Up={mkt.up_price:.3f} "
                f"Down={mkt.down_price:.3f}"
            )
        return mkt

    def get_current_market(self) -> MarketInfo | None:
        """Get the currently in-play 5-min market."""
        slug = f"btc-updown-5m-{self._current_window_ts()}"
        return self._fetch_market_by_slug(slug)

    @staticmethod
    def _slug_to_iso_start(slug: str) -> str | None:
        """Convert a slug like 'btc-updown-5m-1771187100' to ISO start time.

        Returns e.g. '2026-02-15T20:25:00Z' — the key Polymarket uses in its
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

        iso_start = PolymarketClient._slug_to_iso_start(slug)
        page = PolymarketClient._scrape_event_page(slug)
        if page is None:
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

        # Strategy 3: Fallback — first openPrice near any past-results key
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

        # NO generic fallback for closePrice — grabbing a random closePrice
        # from the page is worse than returning None and retrying, because
        # it will resolve our trade against the wrong window's data.
        logger.debug(f"Close price not yet available for {slug}")
        return None

    # ── Live Market Prices (via gamma-api) ──────────────────────────
    #
    # IMPORTANT: The raw CLOB order book per-token only shows that
    # token's native orders (e.g. UP book has bids at 1¢, no asks).
    # However Polymarket uses neg-risk complement matching: buying UP
    # at 0.49 is matched against selling DOWN at 0.51 under the hood.
    # The gamma-api's bestAsk/bestBid/outcomePrices reflect the REAL
    # tradeable price including complement matching.  Always use those.

    @staticmethod
    def get_live_market_price(slug: str) -> dict | None:
        """Re-fetch the latest market prices from gamma-api.

        Cached with a 4s TTL — avoids hitting gamma-api multiple times
        within the same scan cycle (scan interval = 3s).

        Returns a dict with the real tradeable prices::

            {
                'up_price': float,     # mid-market for UP
                'down_price': float,   # mid-market for DOWN
                'best_ask': float,     # cheapest you can BUY UP (incl. complement)
                'best_bid': float,     # best you can SELL UP
                'spread': float,       # ask - bid
            }

        These prices account for complement matching — the gamma-api
        aggregates both sides of the book.  A BUY UP at best_ask WILL
        fill immediately.
        """
        now = time.time()
        cached = _live_price_cache.get(slug)
        if cached and (now - cached[1]) < _LIVE_PRICE_TTL:
            return cached[0]

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
                f"Live prices [{slug}]: UP={result['up_price']:.3f} "
                f"ask={result['best_ask']:.3f} bid={result['best_bid']:.3f} "
                f"spread={result['spread']:.3f}"
            )
            _live_price_cache[slug] = (result, time.time())
            return result

        except Exception as e:
            logger.warning(f"Failed to fetch live prices for {slug}: {e}")
            return None

    def get_current_odds(self, market: MarketInfo) -> tuple[float, float]:
        """Get live Up/Down prices from gamma-api (includes complement matching)."""
        live = self.get_live_market_price(market.slug)
        if live:
            return live["up_price"], live["down_price"]
        return market.up_price, market.down_price

    # ── Order Execution ─────────────────────────────────────────────

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
        # Use 0.01 as default — Polymarket will reject invalid ticks
        tick_size = 0.01
        price = round(price / tick_size) * tick_size
        price = max(tick_size, min(price, 1.0 - tick_size))

        try:
            import math
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
                # FOK uses MarketOrderArgs with amount (USDC) — the library's
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

    # ── Position Management ─────────────────────────────────────────

    def check_position_status(self, order_id: str) -> dict:
        """Check the status of an existing order/position."""
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
            side: "UP" or "DOWN" — which token to sell
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
                    f"Cannot sell {shares:.2f} shares (min {MIN_SHARES}) — "
                    f"will hold to resolution"
                )
                return OrderResult(success=False, error="Below minimum shares")

            # Approve conditional token for selling (required after buy settles)
            try:
                from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
                params = BalanceAllowanceParams(
                    asset_type=AssetType.CONDITIONAL,
                    token_id=token_id,
                )
                self.client.update_balance_allowance(params)
            except Exception as e:
                logger.debug(f"update_balance_allowance: {e}")

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
                f"(proceeds~${proceeds:.2f}) → order_id={order_id} status={status}"
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
                f"(cost=${cost:.2f}) → order_id={order_id} status={status}"
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