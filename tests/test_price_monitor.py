"""Tests for PriceMonitor - price tracking and signal detection."""

import time
import unittest

from src.price_monitor import PriceMonitor, PricePoint


def make_config():
    return {
        "price_feeds": {
            "binance_ws": "wss://stream.binance.com:9443/ws/btcusdt@trade",
            "coingecko_url": "https://api.coingecko.com/api/v3/simple/price",
        },
        "strategy": {
            "min_price_delta": 0.15,
        },
    }


class TestPriceMonitor(unittest.TestCase):
    def setUp(self):
        self.monitor = PriceMonitor(make_config())

    def test_record_price(self):
        self.monitor._record_price(100000)
        self.assertEqual(self.monitor.current_price, 100000)
        self.assertEqual(len(self.monitor.price_history), 1)

    def test_delta_insufficient_data(self):
        """Should return None with less than 2 data points."""
        self.assertIsNone(self.monitor.calculate_5min_delta())
        self.monitor._record_price(100000)
        self.assertIsNone(self.monitor.calculate_5min_delta())

    def test_delta_calculation(self):
        """Should correctly calculate percentage delta."""
        now = time.time()
        # Simulate a price 4 minutes ago
        self.monitor.price_history.append(
            PricePoint(price=100000, timestamp=now - 240)
        )
        self.monitor._record_price(100150)

        delta = self.monitor.calculate_5min_delta()
        self.assertIsNotNone(delta)
        self.assertAlmostEqual(delta, 0.15, places=2)

    def test_no_signal_below_threshold(self):
        """Should not emit signal for small price moves."""
        now = time.time()
        self.monitor.price_history.append(
            PricePoint(price=100000, timestamp=now - 240)
        )
        self.monitor._record_price(100050)  # 0.05% move

        signal = self.monitor.detect_arbitrage_signal()
        self.assertIsNone(signal)

    def test_signal_above_threshold(self):
        """Should emit signal for significant price moves."""
        now = time.time()
        self.monitor.price_history.append(
            PricePoint(price=100000, timestamp=now - 240)
        )
        self.monitor._record_price(100200)  # 0.20% move

        signal = self.monitor.detect_arbitrage_signal()
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, "UP")
        self.assertGreater(signal.delta_pct, 0.15)

    def test_down_signal(self):
        now = time.time()
        self.monitor.price_history.append(
            PricePoint(price=100000, timestamp=now - 240)
        )
        self.monitor._record_price(99800)  # -0.20% move

        signal = self.monitor.detect_arbitrage_signal()
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, "DOWN")

    def test_confidence_scaling(self):
        """Larger moves should produce higher confidence."""
        now = time.time()

        # Small move
        self.monitor.price_history.clear()
        self.monitor.price_history.append(
            PricePoint(price=100000, timestamp=now - 240)
        )
        self.monitor._record_price(100200)
        small_signal = self.monitor.detect_arbitrage_signal()

        # Large move
        self.monitor.price_history.clear()
        self.monitor.price_history.append(
            PricePoint(price=100000, timestamp=now - 240)
        )
        self.monitor._record_price(100500)
        large_signal = self.monitor.detect_arbitrage_signal()

        self.assertGreater(large_signal.confidence, small_signal.confidence)


if __name__ == "__main__":
    unittest.main()
