# AI News Trader (`src/news_trader`)

An isolated, paper-only engine that turns real-world news into Polymarket
trade decisions and records everything needed to find out whether a
repeatable edge exists. It runs as its own process next to the BTC bot
(`src/main.py`), which it does not modify.

```
news providers ─► normalise / dedupe / corroborate ─► match markets ─► rule screen
     │                                                                      │
     └─ health, backoff                                    order books (YES + NO)
                                                                            │
journal ◄─ paper executor ◄─ deterministic risk gate ◄─ edge/EV ◄─ AI signal (JSON)
   │
   └─► tracker (marks, exits, resolutions) ─► analytics ─► dashboard "AI News Trader" tab
```

## 1. What the audit found (existing repo)

| Area | Existing code | Reused by the news engine |
|---|---|---|
| Entry point | `src/main.py` — one asyncio `Orchestrator` for 5-min BTC Up/Down markets | No (separate process, `python -m src.news_trader`) |
| Market discovery | `polymarket_client.py` — Gamma `/events?slug=btc-updown-5m-…` only | No; new general catalog via Gamma `/markets` |
| Prices | Gamma `bestBid/bestAsk` per BTC slug; Binance WS for BTC | No; CLOB `/book` depth per token |
| Orders/signing | `PolymarketClient` (py-clob-client, derive API key, `place_limit_order`, `check_position_status`, `cancel_order`, `get_live_balance`) | **Yes**, wrapped by `LiveExecutor` (disabled) |
| Paper vs live | `--live` CLI flag; paper trades skip the client entirely | Same idea, plus `LiveAuthorization` gates |
| Persistence | SQLite `data/trades.db` (`trades`, `signals`, `bot_sessions`) | No; separate `data/news_trader.db` so BTC data is untouched |
| Risk | `PositionManager` (Kelly/compound, max loss, consecutive-loss breaker) — BTC-specific state | No; new `RiskManager` |
| Kill switch | `EMERGENCY_STOP` file + `TRADING_ENABLED` env | **Yes**: `EMERGENCY_STOP` halts the news engine too; plus `NEWS_TRADER_HALT` |
| Dashboard | Flask app, one HTML page, `/api/*` | **Yes**: Blueprint registered + tab link (16 added lines) |
| Config | `config.yaml` (edited by the dashboard) + `.env` | **Yes**: new `news_trader:` section; credentials stay in `.env` |
| Tests | `unittest` style, run with pytest | Same style (`IsolatedAsyncioTestCase`) |

Pre-existing test failures (unchanged by this work): `test_rejects_too_close_to_expiry`
(test expects a 60s cutoff, code uses 30s) and `test_circuit_breaker_bankroll`
(test expects a `min_bankroll` breaker the code no longer has).

### Polymarket facts verified against docs and live responses (2026-10-22)

* **Geography.** The international CLOB lists the **United States as close-only on
  both the frontend and the API**; several other countries are close-only or
  blocked (docs: `developers/CLOB/geoblock`). `GET https://polymarket.com/api/geoblock`
  returns `{"blocked", "ip", "country", "region"}`. Live mode refuses to start unless it
  returns `blocked: false`. Nothing in this code attempts to bypass this.
* **Polymarket US** (`docs.polymarket.us`) is a separate venue with its own API, SDKs and
  key-pair auth. It is *not* interchangeable with the international CLOB integration
  in this repo; no Polymarket US execution adapter is implemented.
* **Fees** (`trading/fees`): takers pay `shares × feeRate × p × (1−p)`; makers pay
  nothing. Rates are per market (`feesEnabled`, `feeSchedule.rate` on Gamma); e.g.
  crypto 0.07, politics/finance 0.04, sports 0.05, geopolitics 0. Entries are modelled
  as taker fills.
* **CLOB `/book`**: bids sorted ascending and asks descending (best price is the
  *last* element), string prices, ms `timestamp`, `tick_size`, `min_order_size`.
* **Rate limits**: Gamma `/markets` 300 req/10s, `/events` 500/10s; CLOB `/book` 1500/10s.
* **Gamma `/markets`**: max 100 per page; list fields are JSON-encoded strings;
  `include_tag=true` returns tags; `condition_ids` (repeatable, ≤100) with
  `closed=any` returns resolved markets.

## 2. News sources (all free; each URL and format checked live)

| Key | Source | Latency class | Notes |
|---|---|---|---|
| `fed_press`, `fed_monetary` | federalreserve.gov RSS | poll_fast (~45–60s) | primary official |
| `sec_8k` | SEC EDGAR current 8-K Atom | poll_fast (~60–90s) | primary; set a real contact in `sources.user_agent` (SEC fair-access policy) |
| `bls_latest`, `bea_releases` | BLS / BEA RSS | poll_fast | primary economic releases |
| `whitehouse` | whitehouse.gov RSS | poll | primary for executive announcements |
| `cnbc_top`, `marketwatch_top`, `bbc_world`, `npr_news` | publisher RSS | poll (1–5 min) | secondary |
| `bluesky_official` | Bluesky public AppView: Reuters, AP, Bloomberg, CNBC, NYT, Federal Reserve | poll_fast (~45–60s) | verified accounts only; reposts ignored; free wire headlines |
| `gdelt_markets` | GDELT DOC 2.0 | **delayed (~15 min)** | aggregator; 1 req/5s; frequently 429s from shared IPs |

None of these is a push stream. Realistic detection latency is 30–120s for
official feeds and wire headlines on Bluesky. That is fast enough to *measure*
whether an edge exists, but slow next to paid low-latency wires. Sports results
need a licensed data feed; none is integrated.

## 3. How decisions are made

1. **Normalise** (`normalize.py`): stable event id, canonical URL, near-duplicate
   clustering (headline shingles/token Jaccard), independent-source corroboration
   (Reuters via Bluesky and reuters.com via GDELT count once), conflict detection
   (denials/retractions), verification status, category, speculation flags.
2. **Pre-filter** (`risk.pre_event`, free): halts, category/source filters, credibility,
   verification (conflicting/unverified/awaiting corroboration), speculation, news age.
3. **Match** (`matching.py`): IDF-weighted market-term coverage, entity overlap,
   action-concept agreement (acquisition ≠ resignation, increase ≠ decrease ≠ no change),
   category compatibility. Top 4 candidates.
4. **Rule screen** (`resolution.py` + `catalog.market_tradeable`): explicit resolution
   clause required; discretionary/subjective language scored; markets closed, paused,
   expiring within 1h, or already in UMA resolution are rejected.
5. **Books**: YES and NO books fetched concurrently; stale quotes, wide spreads,
   extreme prices and thin depth rejected — all before any paid AI call.
6. **AI** (`ai.py`): one call per event covering all candidates, JSON-schema output.
   Validated with pydantic and cross-checked: offered market ids only, consistent
   action/outcome/probability vs ask, **evidence quotes must appear verbatim in the
   news text**, factual + new information, DIRECTLY_RESOLVES/MATERIALLY_CHANGES and
   COMPATIBLE rules. Any failure → NO TRADE.
7. **Edge** (`edge.py`): AI probability shrunk toward the mid
   (`p = mid + 0.5·(p_ai − mid)`) until calibration is proven; limit = p − min_edge −
   safety margin, rounded down to tick; fractional Kelly at the limit; book walked for
   VWAP + fees; edge recomputed from the all-in price.
8. **Risk** (`risk.py`): confidence, uncertainty width, corroboration, decision latency,
   pre-trade book recheck, duplicate/cooldown, cash, per-trade/market/daily/correlated
   caps, open-position cap, daily realized+unrealized loss limit (latches a halt).
   Sizing → order → position runs under one lock so concurrent events cannot jointly
   breach a cap.
9. **Execution** (`execution.py`): limit orders only. Paper fills use a book fetched
   *after* simulated latency, consume only 50% of displayed depth, fill resting
   remainder only on new liquidity crossing below the limit, and cancel at timeout.
10. **Tracking** (`tracker.py`): price marks at 1m/5m/30m/1h/6h/24h for every assessed
    signal (traded or not), positions marked at the bid net of exit fees, settlement
    from Gamma resolution (1/0/0.5).

## 4. Running it

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...          # optional; without it every candidate is NO TRADE (ai_unavailable)

python -m src.news_trader sources     # check every feed once
python -m src.news_trader once        # one full pass (catalog + all feeds + pipeline)
python -m src.news_trader run         # continuous paper trading
python -m src.news_trader report      # JSON performance report
python -m src.news_trader replay      # re-run history under current settings (no AI spend, no look-ahead)

python -m src.dashboard               # then open http://localhost:8050/news
```

Halts: dashboard buttons, `touch NEWS_TRADER_HALT` (news engine only),
`touch EMERGENCY_STOP` (every engine), or `news_trader.emergency_stop: true`.

## 5. Costs (first implementation)

| Item | Cost |
|---|---|
| News feeds | $0 (all sources above are free) |
| Polymarket data | $0 (public Gamma/CLOB endpoints) |
| AI, default `claude-opus-5-5` ($4/$20 per MTok) | ≈ $0.05–0.08 per analysed event (≈5–6k input tokens incl. rules for up to 4 markets, 1.5–3k output incl. thinking). Hard caps: 60 calls/day and $3/day by default → ≤ ~$90/month; typical 10–30 qualifying events/day ≈ $0.50–2.40/day |
| AI, `claude-haiku-5-5` ($0.10/$0.50) | ≈ $0.002 per event; selectable in the dashboard. Quality on rule interpretation should be compared before relying on it |
| Hosting | a small VPS (1 vCPU / 1 GB) ≈ $5–10/month; must be in a jurisdiction where Polymarket permits trading *before* any live use |

Only events that survive the free deterministic stages reach the model; results
are cached per story cluster and price bucket, and re-evaluations after
corroboration reuse the cached analysis.

## 6. Evidence required before considering live trading

Live execution is disabled in code (`LIVE_TRADING_RELEASED = False`). Before
changing it, the paper journal should show, at minimum:

1. **Sample size**: ≥ 100 filled paper trades across ≥ 8 weeks and several news
   categories, not dominated by one event.
2. **Positive net P&L after fees** with a 95% bootstrap confidence interval that
   excludes zero, and max drawdown within the planned risk budget.
3. **Calibration**: on resolved markets, the model's Brier score beats the market
   mid at decision time (`calibration.model_beats_market` in the report). If the
   model does not beat the price, there is no information edge.
4. **Speed is not the whole story**: positive P&L in the 30s–2m latency bucket, and
   the `vs_immediate_entry.delay_cost` is small relative to the edge.
5. **Fill realism**: paper fill rate and slippage stable when `paper_depth_haircut`
   is lowered to 0.25 and `paper_latency_ms` raised to 2000 (re-run with `replay`).
6. **Robustness**: P&L not explained by a handful of trades; per-source and
   per-category results reviewed; missed-opportunity list reviewed for systematic
   false rejections.
7. **Operational**: ≥ 2 weeks of continuous running without unhandled errors; the AI
   budget, halts and loss latch exercised; geoblock check passes from the production
   host; a tiny live pilot (e.g. $1–2 per trade, $20 daily loss limit) approved by you.
