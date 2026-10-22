"""
Point-in-time replay without look-ahead.

Replays journaled events through the real decision pipeline with:
  * a ManualClock set to the moment the engine originally received each
    event (journaled ``received_at``);
  * order books served from recorded snapshots, returning only the latest
    snapshot observed AT OR BEFORE the replay clock — a request for a book
    when none had yet been observed fails rather than borrowing a later one;
  * market metadata as recorded when the event was matched;
  * the AI analysis that was recorded for that event (no new paid calls),
    so threshold/risk changes can be evaluated on history for free.

Fills are simulated by the PaperExecutor against those same point-in-time
books, so a replay can never fill at a price that did not exist yet.
"""

from __future__ import annotations

import bisect
import json
import logging
from dataclasses import dataclass

from sqlalchemy import select

from src.news_trader.ai import AIAnalysis, AnalysisResult
from src.news_trader.catalog import MarketCatalog, MarketIndex
from src.news_trader.clock import ManualClock
from src.news_trader.config import NewsTraderSettings
from src.news_trader.execution import PaperExecutor
from src.news_trader.ingestion import IngestedEvent
from src.news_trader.journal import AnalysisRow, BookSnapshotRow, EventRow, Journal, MatchRow, market_from_json
from src.news_trader.models import BookLevel, NewsEvent, OrderBook, Verification

logger = logging.getLogger(__name__)


class LookAheadError(LookupError):
    pass


class PointInTimeBookStore:
    def __init__(self, snapshots: list[BookSnapshotRow]):
        self._by_token: dict[str, list[BookSnapshotRow]] = {}
        for s in sorted(snapshots, key=lambda r: r.observed_at):
            self._by_token.setdefault(s.token_id, []).append(s)
        self._times = {t: [r.observed_at for r in rows] for t, rows in self._by_token.items()}

    def get(self, token_id: str, at: float) -> OrderBook:
        times = self._times.get(token_id, [])
        i = bisect.bisect_right(times, at) - 1
        if i < 0:
            raise LookAheadError(f"no book for {token_id[:12]}… observed at or before t={at}")
        row = self._by_token[token_id][i]
        return OrderBook(
            token_id=token_id,
            bids=[BookLevel(p, s) for p, s in json.loads(row.bids)],
            asks=[BookLevel(p, s) for p, s in json.loads(row.asks)],
            exchange_ts=row.exchange_ts,
            # Report the book as observed when it was, so staleness checks
            # see its true age at replay time.
            fetched_at=row.observed_at,
            tick_size=row.tick_size or 0.01,
            min_order_size=row.min_order_size or 5.0,
        )


class ReplayBookClient:
    def __init__(self, store: PointInTimeBookStore, clock: ManualClock):
        self.store = store
        self.clock = clock

    async def get_book(self, token_id: str) -> OrderBook:
        return self.store.get(token_id, self.clock.now())


class RecordedAnalyzer:
    """Serves the analysis recorded for an event; never calls a model."""

    def __init__(self, analyses: dict[str, str]):
        self.analyses = analyses

    async def analyze(self, event, candidates, related_headlines) -> AnalysisResult:
        raw = self.analyses.get(event.event_id)
        if raw is None:
            return AnalysisResult(ok=False, error="no recorded analysis for this event")
        analysis = AIAnalysis.model_validate_json(raw)
        offered = {c.market.market_id for c in candidates}
        analysis.assessments = [a for a in analysis.assessments if a.market_id in offered]
        return AnalysisResult(ok=True, analysis=analysis, model="recorded", cached=True)


@dataclass
class ReplayInputs:
    events: list[NewsEvent]
    received_at: dict[str, float]
    markets_by_event: dict[str, list]
    analyses: dict[str, str]
    snapshots: list[BookSnapshotRow]


def load_replay_inputs(journal: Journal) -> ReplayInputs:
    with journal.Session() as s:
        rows = list(s.scalars(select(EventRow).where(EventRow.is_duplicate.is_(False))))
        rows.sort(key=lambda r: r.received_at or r.first_seen_at)
        matches = list(s.scalars(select(MatchRow)))
        analyses = {a.event_id: a.analysis_json for a in s.scalars(select(AnalysisRow))
                    if a.ok and a.analysis_json}
        snaps = list(s.scalars(select(BookSnapshotRow)))
    events = [NewsEvent(
        event_id=r.event_id, cluster_id=r.cluster_id, provider=r.provider, source_name=r.source_name,
        source_domain=r.source_domain, url=r.url, title=r.title, summary=r.summary or "",
        category=r.category, credibility=r.credibility, published_at=r.published_at,
        first_seen_at=r.first_seen_at, ingested_at=r.ingested_at, is_realtime_source=bool(r.realtime_source),
        is_primary_source=bool(r.primary_source), verification=Verification(r.verification),
        corroborating_sources=json.loads(r.corroborating or "[]"), flags=json.loads(r.flags or "[]"),
    ) for r in rows]
    by_event: dict[str, list] = {}
    for m in matches:
        if m.market_json:
            by_event.setdefault(m.event_id, []).append(market_from_json(m.market_json))
    received = {r.event_id: (r.received_at or r.first_seen_at) for r in rows}
    return ReplayInputs(events, received, by_event, analyses, snaps)


async def replay(source: Journal, target: Journal, settings: NewsTraderSettings) -> dict:
    """Re-run recorded events under ``settings``; writes decisions to ``target``."""
    from src.news_trader.analytics import performance_report
    from src.news_trader.engine import NewsTradingEngine
    from src.news_trader.risk import RiskManager

    inputs = load_replay_inputs(source)
    if not inputs.events:
        return {"events": 0}
    clock = ManualClock(inputs.received_at[inputs.events[0].event_id])
    books = ReplayBookClient(PointInTimeBookStore(inputs.snapshots), clock)
    catalog = MarketCatalog(http=None, clock=clock, settings=settings.catalog)
    executor = PaperExecutor(books, clock, settings.execution, settings.risk.max_quote_age_seconds)
    risk = RiskManager(settings, target, clock, halt_files=[], mode="paper")
    engine = NewsTradingEngine(settings, catalog=catalog, books=books,
                               analyzer=RecordedAnalyzer(inputs.analyses), executor=executor,
                               journal=target, risk=risk, clock=clock)
    for ev in inputs.events:
        markets = inputs.markets_by_event.get(ev.event_id)
        if not markets:
            continue
        # The engine saw the event at received_at; it must not see anything later.
        clock.set(max(clock.now(), inputs.received_at[ev.event_id]))
        catalog.markets = {m.market_id: m for m in markets}
        catalog.index = MarketIndex(markets)
        try:
            await engine.process_event(IngestedEvent(ev, "new"))
        except LookAheadError as e:
            logger.info("replay skipped %s: %s", ev.event_id, e)
    return performance_report(target, "paper")
