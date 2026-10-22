"""
Injectable clocks.

Every component reads time through a Clock so that tests and backtests can
control it.  A backtest that used the wall clock (or a later quote) would
silently introduce look-ahead bias.
"""

from __future__ import annotations

import asyncio
import time
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...

    async def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def now(self) -> float:
        return time.time()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class ManualClock:
    """Deterministic clock for tests and replays; sleep() advances time."""

    def __init__(self, start: float = 1_800_000_000.0):
        self._now = float(start)

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("clock cannot go backwards")
        self._now += seconds

    def set(self, ts: float) -> None:
        if ts < self._now:
            raise ValueError("clock cannot go backwards")
        self._now = ts

    async def sleep(self, seconds: float) -> None:
        self.advance(max(0.0, seconds))
        await asyncio.sleep(0)
