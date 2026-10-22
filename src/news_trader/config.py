"""
Typed configuration for the AI news-trading engine.

Settings live under the ``news_trader:`` key of the shared config.yaml so the
existing dashboard can edit them.  Every field has a conservative default, so
an absent section yields a paper-only engine with small limits.

Live trading is NOT authorised by anything in this file.  ``mode: live`` is
only a request; see ``execution.LiveAuthorization`` for the separate gates.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

CATEGORIES = (
    "monetary_policy",
    "economics",
    "corporate",
    "politics",
    "geopolitics",
    "crypto",
    "sports",
    "general",
)

# Models offered in the dashboard selector, with first-party per-MTok prices
# (input, output, cache read) used for budget accounting.
AI_MODEL_PRICES: dict[str, tuple[float, float, float]] = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}


class _Section(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True)


class SourceSettings(_Section):
    # Provider keys from sources.registry.DEFAULT_PROVIDERS
    enabled: list[str] = Field(default_factory=lambda: [
        "fed_press", "fed_monetary", "sec_8k", "bls_latest", "bea_releases",
        "whitehouse", "cnbc_top", "marketwatch_top", "bbc_world", "npr_news",
        "gdelt_markets", "bluesky_official",
    ])
    min_credibility: float = Field(0.6, ge=0.0, le=1.0)
    user_agent: str = "Polybot-NewsTrader/1.0 (contact: set news_trader.sources.user_agent)"
    request_timeout_seconds: float = Field(15.0, gt=0)


class CatalogSettings(_Section):
    refresh_seconds: float = Field(600.0, ge=30)
    max_markets: int = Field(2000, ge=10, le=20000)
    page_size: int = Field(100, ge=10, le=100)  # Gamma caps /markets pages at 100
    min_liquidity: float = Field(1000.0, ge=0)
    min_volume_24h: float = Field(0.0, ge=0)


class MatchingSettings(_Section):
    max_candidates: int = Field(4, ge=1, le=20)
    min_score: float = Field(0.30, ge=0.0, le=1.0)
    max_ambiguity: float = Field(0.5, ge=0.0, le=1.0)


class AISettings(_Section):
    provider: Literal["anthropic", "disabled"] = "anthropic"
    model: str = "claude-opus-5-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    timeout_seconds: float = Field(60.0, gt=0, le=600)
    max_tokens: int = Field(16000, ge=1024, le=64000)
    max_calls_per_day: int = Field(60, ge=0)
    daily_budget_usd: float = Field(3.0, ge=0)
    cache_ttl_seconds: float = Field(900.0, ge=0)
    max_concurrency: int = Field(2, ge=1, le=16)
    # Until resolved outcomes prove the model is calibrated, its probability is
    # shrunk toward the market mid: p = mid + weight * (p_ai - mid).
    probability_weight: float = Field(0.5, ge=0.0, le=1.0)


class RiskSettings(_Section):
    max_trade_usd: float = Field(10.0, gt=0)
    max_market_exposure_usd: float = Field(20.0, gt=0)
    max_daily_exposure_usd: float = Field(100.0, gt=0)
    max_open_positions: int = Field(10, ge=1)
    max_correlated_exposure_usd: float = Field(30.0, gt=0)
    daily_loss_limit_usd: float = Field(30.0, gt=0)
    min_edge: float = Field(0.04, ge=0.0, le=1.0)
    min_confidence: float = Field(0.6, ge=0.0, le=1.0)
    max_uncertainty: float = Field(0.25, ge=0.0, le=1.0)
    max_news_age_seconds: float = Field(900.0, gt=0)
    max_decision_latency_seconds: float = Field(120.0, gt=0)
    max_quote_age_seconds: float = Field(10.0, gt=0)
    max_spread: float = Field(0.06, gt=0, le=1.0)
    min_depth_usd: float = Field(50.0, ge=0)
    min_time_to_close_hours: float = Field(1.0, ge=0)
    min_entry_price: float = Field(0.03, ge=0.0, le=1.0)
    max_entry_price: float = Field(0.95, ge=0.0, le=1.0)
    kelly_fraction: float = Field(0.25, ge=0.0, le=1.0)
    safety_margin: float = Field(0.01, ge=0.0, le=0.5)
    signal_cooldown_seconds: float = Field(3600.0, ge=0)
    require_corroboration_categories: list[str] = Field(
        default_factory=lambda: ["politics", "geopolitics", "crypto", "general", "sports"]
    )
    paper_bankroll_usd: float = Field(500.0, gt=0)


class ExecutionSettings(_Section):
    order_timeout_seconds: float = Field(30.0, gt=0)
    poll_interval_seconds: float = Field(2.0, gt=0)
    # Paper fills only consume this fraction of displayed depth: faster bots
    # and hidden cancels mean the displayed book is never entirely ours.
    paper_depth_haircut: float = Field(0.5, gt=0.0, le=1.0)
    # Simulated wire latency: the paper fill is computed against a book
    # fetched this long after the decision, not the one we decided on.
    paper_latency_ms: int = Field(500, ge=0, le=60000)
    exit_policy: Literal["hold_to_resolution", "take_profit"] = "hold_to_resolution"
    max_hold_hours: float = Field(0.0, ge=0)


class TrackingSettings(_Section):
    mark_horizons_seconds: list[int] = Field(
        default_factory=lambda: [60, 300, 1800, 3600, 21600, 86400]
    )
    loop_seconds: float = Field(30.0, ge=1)
    resolution_check_seconds: float = Field(900.0, ge=30)


class NewsTraderSettings(_Section):
    enabled: bool = True
    mode: Literal["paper", "live"] = "paper"
    emergency_stop: bool = False
    db_path: str = "./data/news_trader.db"
    status_path: str = "./data/news_trader_status.json"
    halt_file: str = "./NEWS_TRADER_HALT"
    categories: list[str] = Field(default_factory=lambda: list(CATEGORIES))
    max_concurrent_events: int = Field(8, ge=1, le=64)
    sources: SourceSettings = Field(default_factory=SourceSettings)
    catalog: CatalogSettings = Field(default_factory=CatalogSettings)
    matching: MatchingSettings = Field(default_factory=MatchingSettings)
    ai: AISettings = Field(default_factory=AISettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)

    @field_validator("categories")
    @classmethod
    def _known_categories(cls, v: list[str]) -> list[str]:
        unknown = [c for c in v if c not in CATEGORIES]
        if unknown:
            raise ValueError(f"unknown categories: {unknown}")
        return v


def load_settings(config: dict | None) -> NewsTraderSettings:
    """Build settings from the full config dict (the ``news_trader`` section)."""
    section = (config or {}).get("news_trader") or {}
    return NewsTraderSettings.model_validate(copy.deepcopy(section))


def load_settings_file(path: str | Path = "config.yaml") -> NewsTraderSettings:
    p = Path(path)
    if not p.exists():
        return NewsTraderSettings()
    with open(p) as f:
        return load_settings(yaml.safe_load(f) or {})


def global_emergency_stop_path(config_path: str | Path = "config.yaml") -> Path:
    """The existing bot-wide kill switch file (shared with the BTC engine)."""
    return Path(config_path).resolve().parent / "EMERGENCY_STOP"
