"""
Asynchronous news ingestion with per-provider health and backoff.

Each provider is polled on its own cadence.  A provider failure (network,
HTTP error, rate limit, malformed payload, even a parser bug) only affects
that provider: it backs off exponentially and the rest keep running.
Normalised events go onto an asyncio.Queue for the engine.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass

from src.news_trader.clock import Clock
from src.news_trader.http import HttpClient, HttpError, RateLimited, SchemaError
from src.news_trader.models import NewsEvent
from src.news_trader.normalize import Normalizer
from src.news_trader.sources.base import NewsProvider

logger = logging.getLogger(__name__)

MAX_BACKOFF_SECONDS = 1800.0
DOWN_AFTER_FAILURES = 5


@dataclass
class IngestedEvent:
    event: NewsEvent
    kind: str  # "new" | "duplicate" | "upgrade"


@dataclass
class ProviderHealth:
    key: str
    name: str
    latency_class: str
    expected_latency_seconds: float
    poll_seconds: float
    state: str = "pending"          # pending | ok | degraded | down | rate_limited
    last_success_at: float | None = None
    last_error: str = ""
    last_error_at: float | None = None
    consecutive_failures: int = 0
    polls: int = 0
    items_seen: int = 0
    events_emitted: int = 0
    next_poll_at: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class IngestionService:
    def __init__(self, providers: list[NewsProvider], http: HttpClient, normalizer: Normalizer,
                 clock: Clock, queue: asyncio.Queue, max_queue_items: int = 5000):
        self.providers = {p.key: p for p in providers}
        self.http = http
        self.normalizer = normalizer
        self.clock = clock
        self.queue = queue
        self.max_queue_items = max_queue_items
        self.health = {
            p.key: ProviderHealth(
                p.key, p.profile.name, p.profile.latency_class,
                p.profile.expected_latency_seconds, p.profile.poll_seconds,
            )
            for p in providers
        }
        self.dropped_events = 0

    async def poll_once(self, provider: NewsProvider) -> int:
        """Poll one provider; returns the number of events emitted."""
        h = self.health[provider.key]
        now = self.clock.now()
        h.polls += 1
        try:
            items = await provider.fetch(self.http, now)
        except RateLimited as e:
            self._fail(h, f"rate limited: {e}", now, rate_limited=True, retry_after=e.retry_after)
            return 0
        except (HttpError, SchemaError) as e:
            self._fail(h, f"{type(e).__name__}: {e}", now)
            return 0
        except Exception as e:  # parser bugs must not kill ingestion
            logger.exception("provider %s crashed", provider.key)
            self._fail(h, f"unexpected {type(e).__name__}: {e}", now)
            return 0

        h.state = "ok"
        h.consecutive_failures = 0
        h.last_success_at = now
        h.items_seen += len(items)
        h.next_poll_at = now + provider.profile.poll_seconds

        emitted = 0
        for raw in items:
            result = self.normalizer.process(
                raw, provider.assess(raw), self.clock.now(), provider.profile.is_realtime,
            )
            if result is None:
                continue
            kind = "duplicate" if result.event.is_duplicate else "new"
            emitted += self._emit(IngestedEvent(result.event, kind))
            if result.upgraded_leader is not None:
                emitted += self._emit(IngestedEvent(result.upgraded_leader, "upgrade"))
        h.events_emitted += emitted
        return emitted

    def _emit(self, item: IngestedEvent) -> int:
        if self.queue.qsize() >= self.max_queue_items:
            self.dropped_events += 1
            logger.warning("ingestion queue full; dropping %s", item.event.event_id)
            return 0
        self.queue.put_nowait(item)
        return 1

    def _fail(self, h: ProviderHealth, msg: str, now: float, rate_limited: bool = False,
              retry_after: float | None = None) -> None:
        h.consecutive_failures += 1
        h.last_error = msg[:300]
        h.last_error_at = now
        backoff = min(MAX_BACKOFF_SECONDS, h.poll_seconds * 2 ** min(h.consecutive_failures, 6))
        if rate_limited:
            backoff = max(backoff, retry_after or 60.0)
            h.state = "rate_limited"
        else:
            h.state = "down" if h.consecutive_failures >= DOWN_AFTER_FAILURES else "degraded"
        h.next_poll_at = now + backoff
        logger.warning("news source %s %s (retry in %.0fs): %s", h.key, h.state, backoff, msg)

    async def run(self, stop: asyncio.Event) -> None:
        in_flight: dict[str, asyncio.Task] = {}
        try:
            while not stop.is_set():
                now = self.clock.now()
                for key, provider in self.providers.items():
                    if key in in_flight and not in_flight[key].done():
                        continue
                    if self.health[key].next_poll_at <= now:
                        in_flight[key] = asyncio.create_task(self.poll_once(provider))
                next_due = min((h.next_poll_at for h in self.health.values()), default=now + 5)
                await self.clock.sleep(max(0.25, min(2.0, next_due - self.clock.now())))
        finally:
            for t in in_flight.values():
                t.cancel()
