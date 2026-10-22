"""Shared fakes and builders for the news-trader tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

from src.news_trader.ai import AIAnalysis, AnalysisResult
from src.news_trader.config import NewsTraderSettings
from src.news_trader.http import NetworkError
from src.news_trader.models import BookLevel, MarketRecord, NewsEvent, OrderBook, Verification

T0 = 1_800_000_000.0


class FakeHttp:
    """Routes by URL substring to a value, an exception, or a callable."""

    def __init__(self, routes: dict | None = None):
        self.routes = routes or {}
        self.calls: list[tuple[str, dict | None]] = []

    def _resolve(self, url: str, params):
        self.calls.append((url, params))
        for key, val in self.routes.items():
            if key in url:
                if callable(val) and not isinstance(val, type):
                    val = val(url, params)
                if isinstance(val, Exception):
                    raise val
                return val
        raise NetworkError(f"no route for {url}")

    async def get_text(self, url, params=None, headers=None) -> str:
        v = self._resolve(url, params)
        return v if isinstance(v, str) else json.dumps(v)

    async def get_json(self, url, params=None, headers=None):
        v = self._resolve(url, params)
        return json.loads(v) if isinstance(v, str) else v


def make_market(market_id="m1", question="Will Acme Corp acquire Widget Inc by December 31, 2026?",
                description=None, yes="ty1", no="tn1", category="corporate", end_ts=T0 + 30 * 86400,
                event_id="ev1", fee_rate=0.04, **kw) -> MarketRecord:
    desc = description if description is not None else (
        'This market will resolve to "Yes" if Acme Corp officially announces a definitive agreement '
        "to acquire Widget Inc by December 31, 2026, 11:59 PM ET. Otherwise, this market will resolve "
        'to "No". The resolution source for this market will be official announcements from Acme Corp.'
    )
    base = dict(
        market_id=market_id, condition_id=f"0xcond-{market_id}", question=question, description=desc,
        slug=f"slug-{market_id}", event_id=event_id, event_title=question, event_slug="evslug",
        tags=["business"], category=category, outcomes=["Yes", "No"], token_ids=[yes, no], end_ts=end_ts,
        active=True, closed=False, archived=False, accepting_orders=True, enable_order_book=True,
        neg_risk=False, liquidity=50_000.0, volume_24h=20_000.0, best_bid=0.30, best_ask=0.31,
        last_trade_price=0.30, tick_size=0.01, min_order_size=5.0, fees_enabled=fee_rate > 0,
        fee_rate=fee_rate, resolution_source="", uma_resolution_status="",
    )
    base.update(kw)
    return MarketRecord(**base)


def make_book(token="ty1", bids=((0.30, 500),), asks=((0.31, 500), (0.33, 500)), at=T0,
              exchange_ts=None, tick=0.01) -> OrderBook:
    return OrderBook(
        token_id=token,
        bids=[BookLevel(p, s) for p, s in sorted(bids, key=lambda x: -x[0])],
        asks=[BookLevel(p, s) for p, s in sorted(asks, key=lambda x: x[0])],
        exchange_ts=exchange_ts if exchange_ts is not None else at,
        fetched_at=at, tick_size=tick, min_order_size=5.0,
    )


def make_event(title="Acme Corp agrees to acquire Widget Inc for $4 billion", summary="",
               category="corporate", provider="sec_8k", verification=Verification.PRIMARY_SOURCE,
               primary=True, credibility=0.97, published=T0 - 30, first_seen=T0 - 20,
               event_id="e1", cluster_id=None, flags=None, **kw) -> NewsEvent:
    return NewsEvent(
        event_id=event_id, cluster_id=cluster_id or event_id, provider=provider, source_name="SEC",
        source_domain="sec.gov", url="https://www.sec.gov/x", title=title,
        summary=summary or f"{title}. The boards of both companies approved the definitive agreement.",
        category=category, credibility=credibility, published_at=published, first_seen_at=first_seen,
        ingested_at=first_seen, is_realtime_source=True, is_primary_source=primary,
        verification=verification, corroborating_sources=["sec.gov"], flags=flags or [], **kw,
    )


class FakeBooks:
    """OrderBookClient stand-in.  Values may be an OrderBook, a list (served in
    sequence, last one repeating), or an exception."""

    def __init__(self, clock, books: dict):
        self.clock = clock
        self.books = books
        self.calls: list[str] = []

    async def get_book(self, token_id: str) -> OrderBook:
        self.calls.append(token_id)
        v = self.books[token_id]
        if isinstance(v, list):
            v = v.pop(0) if len(v) > 1 else v[0]
        if isinstance(v, Exception):
            raise v
        # Serve as freshly fetched at the current clock time.
        return OrderBook(v.token_id, list(v.bids), list(v.asks), self.clock.now(), self.clock.now(),
                         v.tick_size, v.min_order_size, v.neg_risk)


def assessment(market_id="m1", action="BUY_YES", p=0.62, low=0.52, high=0.72, conf=0.8,
               evidence=None, outcome=None, relevance="DIRECTLY_RESOLVES", compat="COMPATIBLE",
               corroborate=False, max_entry=0.60) -> dict:
    return {
        "market_id": market_id, "relevance": relevance, "rule_compatibility": compat,
        "rule_analysis": "The rule requires an official definitive agreement; the filing announces one.",
        "impacted_outcome": outcome or ("YES" if action == "BUY_YES" else "NO" if action == "BUY_NO" else "NONE"),
        "evidence": evidence if evidence is not None else ["agrees to acquire Widget Inc"],
        "fair_probability_yes": p, "probability_low": low, "probability_high": high, "confidence": conf,
        "priced_in_assessment": "Price has not moved in the last hour.", "requires_corroboration": corroborate,
        "action": action, "max_entry_price": max_entry if action != "NO_TRADE" else 0.0,
        "rejection_reasons": [] if action != "NO_TRADE" else ["not relevant"],
    }


def analysis(*assessments, factual=True, new=True) -> dict:
    return {
        "event_classification": "corporate_announcement", "is_new_information": new,
        "is_factual_announcement": factual, "summary": "Acme agreed to buy Widget.",
        "assessments": list(assessments),
    }


class StubAnalyzer:
    def __init__(self, payload: dict | None = None, error: str = ""):
        self.payload = payload
        self.error = error
        self.calls = 0

    async def analyze(self, event, candidates, related_headlines) -> AnalysisResult:
        self.calls += 1
        if self.error:
            return AnalysisResult(ok=False, error=self.error, model="stub")
        return AnalysisResult(ok=True, analysis=AIAnalysis.model_validate(self.payload), model="stub")


def fake_message(text: str | None, stop_reason="end_turn", usage=(1000, 500, 0, 0)):
    content = [SimpleNamespace(type="thinking", thinking="")]
    if text is not None:
        content.append(SimpleNamespace(type="text", text=text))
    return SimpleNamespace(
        content=content, stop_reason=stop_reason, model="claude-opus-5-5", _request_id="req_test",
        usage=SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1],
                              cache_read_input_tokens=usage[2], cache_creation_input_tokens=usage[3]),
    )


def settings(**risk_overrides) -> NewsTraderSettings:
    s = NewsTraderSettings()
    s.execution.paper_latency_ms = 0
    s.execution.order_timeout_seconds = 0.0001
    for k, v in risk_overrides.items():
        setattr(s.risk, k, v)
    return s
