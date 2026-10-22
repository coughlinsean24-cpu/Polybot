"""AI News Trader dashboard tab: API endpoints, settings validation, halts."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

import src.dashboard as main_dash
import src.news_trader.dashboard as nd
from src.news_trader.clock import ManualClock
from src.news_trader.config import load_settings
from src.news_trader.journal import Journal
from src.news_trader.risk import RiskManager
from tests.news_fakes import T0

BASE_CONFIG = {
    "strategy": {"initial_bet": 15.0, "max_bet": 15.0},
    "logging": {"db_path": "./data/trades.db"},
    "news_trader": {"mode": "paper", "risk": {"max_trade_usd": 10.0}},
}


class TestNewsDashboard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "config.yaml").write_text(yaml.safe_dump(BASE_CONFIG))
        self.patches = [mock.patch.object(nd, "PROJECT_ROOT", self.root),
                        mock.patch.object(nd, "CONFIG_PATH", self.root / "config.yaml")]
        for p in self.patches:
            p.start()
        nd._journals.clear()
        self.client = main_dash.app.test_client()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        nd._journals.clear()
        self.tmp.cleanup()

    def cfg(self):
        return yaml.safe_load((self.root / "config.yaml").read_text())

    def test_tab_page_and_main_dashboard_link(self):
        r = self.client.get("/news")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"AI News Trader", r.data)
        self.assertIn(b'href="/news"', self.client.get("/").data)

    def test_status_when_engine_not_running(self):
        d = self.client.get("/api/news/status").get_json()
        self.assertEqual(d["engine"], "not_running")
        self.assertEqual(d["config"]["mode"], "paper")

    def test_status_reads_heartbeat(self):
        status = self.root / "data" / "news_trader_status.json"
        status.parent.mkdir(parents=True, exist_ok=True)
        import time
        status.write_text(json.dumps({"heartbeat": time.time(), "mode": "paper", "providers": []}))
        self.assertEqual(self.client.get("/api/news/status").get_json()["engine"], "running")

    def test_settings_roundtrip_preserves_other_strategies(self):
        r = self.client.post("/api/news/settings", json={
            "max_trade_usd": 25, "min_edge": 0.05, "categories": ["economics", "monetary_policy"],
            "ai_model": "claude-haiku-5-5", "sources_enabled": ["fed_press", "bls_latest"]})
        self.assertEqual(r.status_code, 200, r.get_json())
        c = self.cfg()
        self.assertEqual(c["strategy"], BASE_CONFIG["strategy"])  # BTC config untouched
        s = load_settings(c)
        self.assertEqual(s.risk.max_trade_usd, 25)
        self.assertEqual(s.ai.model, "claude-haiku-5-5")
        self.assertEqual(s.sources.enabled, ["fed_press", "bls_latest"])

    def test_invalid_settings_rejected_and_not_written(self):
        before = self.cfg()
        for bad in ({"min_edge": 5}, {"max_trade_usd": "abc"}, {"categories": ["astrology"]},
                    {"ai_model": "gpt-x"}, {"sources_enabled": ["evil_feed"]}, {"mode": "yolo"}):
            r = self.client.post("/api/news/settings", json=bad)
            self.assertEqual(r.status_code, 400, bad)
        self.assertEqual(self.cfg(), before)

    def test_live_toggle_is_only_a_request(self):
        r = self.client.post("/api/news/settings", json={"mode": "live"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("only a request", r.get_json()["note"])

    def test_halt_and_resume_affect_risk_gate(self):
        self.client.post("/api/news/halt")
        s = nd._settings()
        rm = RiskManager(s, Journal(":memory:"), ManualClock(T0), [Path(s.halt_file)])
        self.assertIsNotNone(rm.halted())
        self.client.post("/api/news/resume")
        self.assertIsNone(rm.halted())

    def test_data_endpoints_and_performance(self):
        for path in ("/api/news/events", "/api/news/signals", "/api/news/positions", "/api/news/orders"):
            r = self.client.get(path)
            self.assertEqual(r.status_code, 200, path)
            self.assertEqual(r.get_json(), [])
        perf = self.client.get("/api/news/performance").get_json()
        self.assertEqual(perf["performance"]["trades"], 0)


if __name__ == "__main__":
    unittest.main()
