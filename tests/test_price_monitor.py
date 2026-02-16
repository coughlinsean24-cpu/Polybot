"""Tests for PriceMonitor - price tracking and window lookups."""

import time
import unittest

from src.price_monitor import PriceMonitor, PricePoint


def make_config():
    return {
        "price_feeds": {
            "binance_ws": "wss://stream.binance.com:9443/ws/btcusdt@trade",
            "coingecko_url": "https://api.coingecko.com/api/v3/simple/price",
        },
        "strategy": {},
    }


class TestPriceMonitor(unittest.TestCase):
    def setUp(self):
        self.monitor = PriceMonitor(make_config())

    def test_record_price(self):
        self.monitor._record_price(100000)
        self.assertEqual(self.monitor.current_price, 100000)
        self.assertEqual(len(self.monitor.price_history), 1)

    def test_get_price_at_exact(self):
        """Should return the closest price to a target timestamp."""
        now = time.time()
        self.monitor.price_history.append(PricePoint(price=100000, timestamp=now - 10))
        self.monitor.price_history.append(PricePoint(price=100100, timestamp=now - 5))
        self.monitor.price_history.append(PricePoint(price=100200, timestamp=now))

        # Should get the price closest to 5 seconds ago
        price = self.monitor.get_price_at(now - 5)
        self.assertEqual(price, 100100)

    def test_get_price_at_returns_none_if_too_far(self):
        """Should return None if no data within 30 seconds of target."""
        now = time.time()
        self.monitor.price_history.append(PricePoint(price=100000, timestamp=now - 60))
        price = self.monitor.get_price_at(now)
        self.assertIsNone(price)

    def test_get_price_at_empty(self):
        """Should return None with no data."""
        self.assertIsNone(self.monitor.get_price_at(time.time()))

    def test_has_enough_history_false_initially(self):
        """Should be False with too little data."""
        self.assertFalse(self.monitor.has_enough_history())
        self.monitor._record_price(100000)
        self.assertFalse(self.monitor.has_enough_history())

    def test_has_enough_history_true(self):
        """Should be True once 30+ seconds have passed since first price."""
        now = time.time()
        self.monitor._first_price_time = now - 35
        self.monitor.price_history.append(PricePoint(price=100000, timestamp=now - 35))
        self.monitor.price_history.append(PricePoint(price=100100, timestamp=now))
        self.assertTrue(self.monitor.has_enough_history())

    def test_seconds_left_in_window(self):
        """Seconds left should be between 0 and 300."""
        secs = self.monitor.seconds_left_in_window()
        self.assertGreaterEqual(secs, 0)
        self.assertLessEqual(secs, 300)

    def test_get_window_start_price(self):
        """Should return price at the window start if we have data there."""
        now = time.time()
        window_start = (int(now) // 300) * 300

        # Inject a price very close to the window start
        self.monitor.price_history.append(
            PricePoint(price=68500, timestamp=float(window_start) + 1)
        )
        self.monitor._record_price(68600)

        price, ts = self.monitor.get_window_start_price()
        self.assertIsNotNone(price)
        self.assertEqual(price, 68500)
        self.assertEqual(ts, float(window_start))


if __name__ == "__main__":
    unittest.main()
