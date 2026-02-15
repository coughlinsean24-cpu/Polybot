"""
Polymarket CLOB API client for interacting with 5-minute BTC binary markets.
Handles authentication, market discovery, order placement, and position management.
"""

import logging
import time
from dataclasses import dataclass

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType

logger = logging.getLogger(__name__)

POLYMARKET_HOST = "https://clob.polymarket.com"


@dataclass
class MarketInfo:
    condition_id: str
    question: str
    yes_token_id: str
    no_token_id: str
    yes_price: float
    no_price: float
    end_date: str
    volume: float


@dataclass
class OrderResult:
    success: bool
    order_id: str | None = None
    fill_price: float | None = None
    error: str | None = None


class PolymarketClient:
    def __init__(self, config: dict, api_key: str, private_key: str):
        self.chain_id = config["polymarket"]["chain_id"]
        self.max_slippage = config["strategy"]["max_slippage"]

        self.client = ClobClient(
            host=POLYMARKET_HOST,
            key=api_key,
            chain_id=self.chain_id,
            funder=private_key,
        )
        self._authenticated = False

    def authenticate(self) -> bool:
        """Authenticate with Polymarket CLOB API."""
        try:
            self.client.set_api_creds(self.client.create_or_derive_api_creds())
            self._authenticated = True
            logger.info("Polymarket authentication successful")
            return True
        except Exception as e:
            logger.error(f"Polymarket authentication failed: {e}")
            return False

    def get_active_5min_markets(self) -> list[MarketInfo]:
        """Fetch currently active 5-minute BTC binary markets."""
        if not self._authenticated:
            raise RuntimeError("Not authenticated. Call authenticate() first.")

        try:
            # Search for BTC 5-minute markets
            markets = self.client.get_markets()
            btc_5min = []

            for market in markets:
                question = market.get("question", "").lower()
                # Filter for 5-minute BTC price markets
                if "btc" in question and ("5 min" in question or "5-min" in question or "five min" in question):
                    tokens = market.get("tokens", [])
                    if len(tokens) < 2:
                        continue

                    yes_token = next((t for t in tokens if t["outcome"] == "Yes"), None)
                    no_token = next((t for t in tokens if t["outcome"] == "No"), None)

                    if not yes_token or not no_token:
                        continue

                    btc_5min.append(MarketInfo(
                        condition_id=market["condition_id"],
                        question=market["question"],
                        yes_token_id=yes_token["token_id"],
                        no_token_id=no_token["token_id"],
                        yes_price=float(yes_token.get("price", 0.5)),
                        no_price=float(no_token.get("price", 0.5)),
                        end_date=market.get("end_date_iso", ""),
                        volume=float(market.get("volume", 0)),
                    ))

            logger.info(f"Found {len(btc_5min)} active 5-min BTC markets")
            return btc_5min

        except Exception as e:
            logger.error(f"Failed to fetch markets: {e}")
            return []

    def get_current_odds(self, market: MarketInfo) -> tuple[float, float]:
        """Get live YES/NO prices for a market."""
        try:
            book = self.client.get_order_book(market.yes_token_id)
            best_ask = float(book.asks[0].price) if book.asks else market.yes_price
            best_bid = float(book.bids[0].price) if book.bids else market.yes_price

            spread = best_ask - best_bid
            mid = (best_ask + best_bid) / 2

            yes_price = mid
            no_price = 1.0 - mid

            logger.debug(f"Market odds: YES={yes_price:.3f} NO={no_price:.3f} spread={spread:.4f}")
            return yes_price, no_price

        except Exception as e:
            logger.warning(f"Failed to get live odds, using cached: {e}")
            return market.yes_price, market.no_price

    def place_order(
        self, market: MarketInfo, side: str, size: float, price: float
    ) -> OrderResult:
        """
        Place a limit order on a Polymarket market.

        Args:
            market: The market to trade on
            side: "YES" or "NO"
            size: Dollar amount to risk
            price: Max price willing to pay (0-1)
        """
        if not self._authenticated:
            return OrderResult(success=False, error="Not authenticated")

        token_id = market.yes_token_id if side == "YES" else market.no_token_id

        try:
            # Calculate number of shares: size / price
            shares = size / price

            order_args = OrderArgs(
                price=price,
                size=shares,
                side="BUY",
                token_id=token_id,
            )

            signed_order = self.client.create_order(order_args)
            result = self.client.post_order(signed_order, OrderType.GTC)

            order_id = result.get("orderID", result.get("id"))
            logger.info(f"Order placed: {side} ${size} @ {price} -> order_id={order_id}")

            return OrderResult(
                success=True,
                order_id=order_id,
                fill_price=price,
            )

        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            return OrderResult(success=False, error=str(e))

    def check_position_status(self, order_id: str) -> dict:
        """Check the status of an existing order/position."""
        try:
            order = self.client.get_order(order_id)
            return {
                "status": order.get("status", "unknown"),
                "filled": float(order.get("size_matched", 0)),
                "remaining": float(order.get("size_matched", 0)),
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
