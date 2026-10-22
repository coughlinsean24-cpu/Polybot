"""
Small async HTTP layer with typed failures.

Providers and market clients depend on the ``HttpClient`` protocol, so tests
can inject a fake.  The real implementation wraps one shared aiohttp session
(``trust_env=True`` so standard proxy/CA environment variables apply), retries
transient failures with exponential backoff, honours ``Retry-After`` on 429,
and enforces a per-host minimum spacing between requests.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any, Protocol
from urllib.parse import urlparse

import aiohttp

from src.news_trader.clock import Clock, SystemClock

logger = logging.getLogger(__name__)


class HttpError(Exception):
    """Non-retryable HTTP failure (4xx other than 429, or retries exhausted)."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class RateLimited(HttpError):
    def __init__(self, message: str, retry_after: float | None):
        super().__init__(message, status=429)
        self.retry_after = retry_after


class NetworkError(HttpError):
    """Connection failure, DNS error, timeout, or 5xx after retries."""


class SchemaError(Exception):
    """A response parsed, but its structure is not what the docs describe."""


class HttpClient(Protocol):
    async def get_text(self, url: str, params: dict | list | None = None,
                       headers: dict | None = None) -> str: ...

    async def get_json(self, url: str, params: dict | list | None = None,
                       headers: dict | None = None) -> Any: ...


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


class AiohttpClient:
    def __init__(
        self,
        user_agent: str,
        timeout_seconds: float = 15.0,
        max_retries: int = 2,
        clock: Clock | None = None,
        host_min_interval: dict[str, float] | None = None,
    ):
        self._user_agent = user_agent
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._max_retries = max_retries
        self._clock = clock or SystemClock()
        self._session: aiohttp.ClientSession | None = None
        self._host_min_interval = host_min_interval or {}
        self._host_next_ok: dict[str, float] = {}
        self._host_locks: dict[str, asyncio.Lock] = {}

    async def __aenter__(self) -> "AiohttpClient":
        await self._ensure_session()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self._timeout,
                trust_env=True,
                headers={"User-Agent": self._user_agent},
            )
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def _respect_host_spacing(self, host: str) -> None:
        interval = self._host_min_interval.get(host)
        if not interval:
            return
        lock = self._host_locks.setdefault(host, asyncio.Lock())
        async with lock:
            wait = self._host_next_ok.get(host, 0.0) - self._clock.now()
            if wait > 0:
                await self._clock.sleep(wait)
            self._host_next_ok[host] = self._clock.now() + interval

    async def get_text(self, url: str, params: dict | list | None = None,
                       headers: dict | None = None) -> str:
        host = urlparse(url).netloc
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            await self._respect_host_spacing(host)
            session = await self._ensure_session()
            try:
                async with session.get(url, params=params, headers=headers) as resp:
                    body = await resp.text(errors="replace")
                    if resp.status == 429:
                        raise RateLimited(
                            f"429 from {host}",
                            _retry_after_seconds(resp.headers.get("Retry-After")),
                        )
                    if resp.status >= 500:
                        raise NetworkError(f"{resp.status} from {host}", resp.status)
                    if resp.status >= 400:
                        raise HttpError(f"{resp.status} from {host}: {body[:200]}", resp.status)
                    return body
            except RateLimited:
                raise  # caller owns rate-limit backoff for its source
            except HttpError as e:
                if not isinstance(e, NetworkError):
                    raise
                last_exc = e
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exc = NetworkError(f"{type(e).__name__} contacting {host}: {e}")
            if attempt < self._max_retries:
                await self._clock.sleep(min(8.0, 0.5 * 2 ** attempt + random.uniform(0, 0.25)))
        assert last_exc is not None
        raise last_exc

    async def get_json(self, url: str, params: dict | list | None = None,
                       headers: dict | None = None) -> Any:
        text = await self.get_text(url, params=params, headers=headers)
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise SchemaError(f"invalid JSON from {urlparse(url).netloc}: {e}") from e
