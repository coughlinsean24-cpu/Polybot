"""
Command-line entry point for the AI news-trading engine.

    python -m src.news_trader run        # continuous (paper mode)
    python -m src.news_trader once       # one full pass: catalog, all sources, pipeline
    python -m src.news_trader sources    # poll each news source once, print health
    python -m src.news_trader report     # performance report from the journal
    python -m src.news_trader replay     # re-run recorded events, no look-ahead, no AI spend

Live execution is not reachable from here unless every gate in
execution.LiveAuthorization passes; today LIVE_TRADING_RELEASED is False.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.news_trader.ai import AnthropicAnalyzer, DisabledAnalyzer, UsageBudget
from src.news_trader.analytics import performance_report
from src.news_trader.catalog import MarketCatalog
from src.news_trader.clock import SystemClock
from src.news_trader.config import NewsTraderSettings, global_emergency_stop_path, load_settings
from src.news_trader.engine import NewsTradingEngine
from src.news_trader.execution import AuthorizationStatus, LiveAuthorization, LiveExecutor, PaperExecutor
from src.news_trader.http import AiohttpClient
from src.news_trader.ingestion import IngestionService
from src.news_trader.journal import Journal
from src.news_trader.normalize import Normalizer
from src.news_trader.orderbook import OrderBookClient
from src.news_trader.risk import RiskManager
from src.news_trader.sources.registry import HOST_MIN_INTERVAL, build_providers
from src.news_trader.tracker import OutcomeTracker

logger = logging.getLogger("news_trader")


def setup_logging(level: str, log_path: Path) -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=5)
    fh.setFormatter(fmt)
    root.addHandler(fh)


def load(config_path: str) -> tuple[dict, NewsTraderSettings, Path]:
    cfg_file = Path(config_path).resolve()
    with open(cfg_file) as f:
        config = yaml.safe_load(f) or {}
    settings = load_settings(config)
    root = cfg_file.parent
    for attr in ("db_path", "status_path", "halt_file"):
        p = Path(getattr(settings, attr))
        if not p.is_absolute():
            setattr(settings, attr, str((root / p).resolve()))
    return config, settings, root


def anthropic_credentials_present() -> bool:
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_PROFILE"):
        return True
    return (Path.home() / ".config" / "anthropic").exists()


async def build(config_path: str, live_flag: bool) -> tuple[NewsTradingEngine, AiohttpClient]:
    config, settings, root = load(config_path)
    clock = SystemClock()
    http = AiohttpClient(settings.sources.user_agent, settings.sources.request_timeout_seconds,
                         clock=clock, host_min_interval=HOST_MIN_INTERVAL)
    journal = Journal(settings.db_path)
    catalog = MarketCatalog(http, clock, settings.catalog)
    books = OrderBookClient(http, clock)
    queue: asyncio.Queue = asyncio.Queue()
    ingestion = IngestionService(build_providers(settings), http,
                                 Normalizer(settings.sources.min_credibility), clock, queue)

    if settings.ai.provider == "disabled":
        analyzer = DisabledAnalyzer("ai_disabled: news_trader.ai.provider is 'disabled'")
    elif not anthropic_credentials_present():
        analyzer = DisabledAnalyzer("ai_unavailable: no Anthropic credentials (set ANTHROPIC_API_KEY)")
        logger.warning("No Anthropic credentials: every candidate will be recorded as NO TRADE (ai_unavailable)")
    else:
        budget = UsageBudget(clock, settings.ai.max_calls_per_day, settings.ai.daily_budget_usd)
        analyzer = AnthropicAnalyzer(settings.ai, budget, clock)

    if settings.mode == "live" or live_flag:
        auth = await LiveAuthorization(settings, live_flag, http).evaluate()
    else:
        auth = AuthorizationStatus(False, ["paper mode (news_trader.mode = paper)"])
    if auth.granted:
        from src.polymarket_client import PolymarketClient
        poly = PolymarketClient(config, os.getenv("POLYMARKET_API_KEY", ""), os.getenv("POLYMARKET_API_SECRET", ""),
                                os.getenv("POLYMARKET_PASSPHRASE", ""), os.getenv("POLYMARKET_PRIVATE_KEY", ""))
        if not poly.authenticate():
            raise SystemExit("Polymarket authentication failed; refusing to start live mode")
        executor = LiveExecutor(poly, auth, books, clock, settings.execution)
    else:
        if settings.mode == "live":
            logger.warning("Live mode requested but NOT authorised; running PAPER. Reasons: %s",
                           "; ".join(auth.reasons))
        executor = PaperExecutor(books, clock, settings.execution, settings.risk.max_quote_age_seconds)

    halt_files = [global_emergency_stop_path(config_path), Path(settings.halt_file)]
    risk = RiskManager(settings, journal, clock, halt_files, mode=executor.mode)
    tracker = OutcomeTracker(settings, journal, books, catalog, executor, clock)
    engine = NewsTradingEngine(settings, catalog=catalog, books=books, analyzer=analyzer, executor=executor,
                               journal=journal, risk=risk, clock=clock, ingestion=ingestion, tracker=tracker,
                               live_auth=auth, config_path=str(Path(config_path).resolve()))
    return engine, http


async def cmd_run(args) -> None:
    engine, http = await build(args.config, args.enable_live_news_trading)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass
    try:
        await engine.run(stop)
    finally:
        await http.close()


async def cmd_once(args) -> None:
    engine, http = await build(args.config, False)
    try:
        stats = await engine.catalog.refresh()
        print(f"catalog: {stats.markets} markets ({stats.invalid} invalid) in {stats.duration_seconds:.1f}s")
        ing = engine.ingestion
        await asyncio.gather(*(ing.poll_once(p) for p in ing.providers.values()))
        items = []
        while not ing.queue.empty():
            items.append(ing.queue.get_nowait())
        print(f"ingested {len(items)} items from {len(ing.providers)} sources")
        sem = asyncio.Semaphore(engine.settings.max_concurrent_events)

        async def one(it):
            async with sem:
                return await engine.process_event(it)
        outcomes = await asyncio.gather(*(one(it) for it in items))
        await engine.tracker.run_once()
        engine.write_status()
        counts: dict[str, int] = {}
        for o in outcomes:
            counts[o.status] = counts.get(o.status, 0) + 1
        print("event outcomes:", json.dumps(counts, indent=1))
        for h in ing.health.values():
            print(f"  {h.key:18} {h.state:12} items={h.items_seen:<4} {h.last_error[:90]}")
        interesting = [o for o in outcomes if o.status in ("traded", "signals_rejected", "ai_failed", "markets_rejected")]
        for o in interesting[:15]:
            print(f"  [{o.status}] {o.event_id}: {o.detail[:220]}")
    finally:
        await http.close()


async def cmd_sources(args) -> None:
    _, settings, _ = load(args.config)
    clock = SystemClock()
    async with AiohttpClient(settings.sources.user_agent, settings.sources.request_timeout_seconds,
                             clock=clock, host_min_interval=HOST_MIN_INTERVAL) as http:
        ing = IngestionService(build_providers(settings), http, Normalizer(), clock, asyncio.Queue())
        await asyncio.gather(*(ing.poll_once(p) for p in ing.providers.values()))
        newest: dict[str, tuple[float, str]] = {}
        while not ing.queue.empty():
            ev = ing.queue.get_nowait().event
            ts = ev.published_at or 0
            if ts >= newest.get(ev.provider, (0, ""))[0]:
                newest[ev.provider] = (ts, ev.title)
        now = clock.now()
        for h in ing.health.values():
            ts, title = newest.get(h.key, (0, ""))
            age = f"{(now - ts) / 60:7.1f}m" if ts else "    n/a"
            print(f"{h.key:18} {h.state:12} {h.latency_class:9} items={h.items_seen:<4} newest={age}  "
                  f"{(title or h.last_error)[:80]}")


def cmd_report(args) -> None:
    _, settings, _ = load(args.config)
    print(json.dumps(performance_report(Journal(settings.db_path), args.mode), indent=2, default=str))


async def cmd_replay(args) -> None:
    from src.news_trader.backtest import replay
    _, settings, root = load(args.config)
    target_path = Path(args.target).resolve()
    if target_path == Path(settings.db_path).resolve():
        raise SystemExit("replay target must differ from the live journal")
    if target_path.exists():
        target_path.unlink()
    report = await replay(Journal(settings.db_path), Journal(str(target_path)), settings)
    print(json.dumps(report, indent=2, default=str))


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="python -m src.news_trader")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--enable-live-news-trading", action="store_true",
                       help="one of several required gates for live mode; see execution.LiveAuthorization")
    sub.add_parser("once")
    sub.add_parser("sources")
    p_rep = sub.add_parser("report")
    p_rep.add_argument("--mode", default="paper", choices=["paper", "live"])
    p_replay = sub.add_parser("replay")
    p_replay.add_argument("--target", default="data/news_replay.db")
    args = parser.parse_args(argv)

    _, settings, root = load(args.config)
    setup_logging(args.log_level, root / "data" / "news_trader.log")
    if args.command == "run":
        asyncio.run(cmd_run(args))
    elif args.command == "once":
        asyncio.run(cmd_once(args))
    elif args.command == "sources":
        asyncio.run(cmd_sources(args))
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "replay":
        asyncio.run(cmd_replay(args))


if __name__ == "__main__":
    main(sys.argv[1:])
