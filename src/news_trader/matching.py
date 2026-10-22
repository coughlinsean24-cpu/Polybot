"""
Event-to-market matching.

Stage 1 (cheap, deterministic, runs on every event):
  * retrieve candidates from the inverted index using rare (high-IDF) tokens;
  * score by how much of the MARKET's key terms the news covers (so a
    headline about "Microsoft to acquire X" matches "Will Microsoft acquire
    X?" and not "Will Microsoft hit a $5T market cap?");
  * require an action-concept agreement when both sides name one
    (acquisition vs resignation vs rate cut ...);
  * filter by category compatibility.

Stage 2 (the LLM) judges true relevance and rule compatibility, but only for
the handful of candidates that survive stage 1 and the resolution screen.

The scorer is lexical with alias/entity normalisation.  An embedding scorer
can be dropped in behind ``SimilarityScorer`` without touching the engine.
"""

from __future__ import annotations

from typing import Protocol

from src.news_trader import text as T
from src.news_trader.catalog import MarketCatalog, MarketIndex
from src.news_trader.models import MarketMatch, MarketRecord, NewsEvent

# Which market categories an event category may affect.  "general" markets
# (no recognised tags) are always eligible.
CATEGORY_COMPAT: dict[str, set[str]] = {
    "monetary_policy": {"monetary_policy", "economics", "politics"},
    "economics": {"economics", "monetary_policy"},
    "corporate": {"corporate", "crypto"},
    "politics": {"politics", "geopolitics", "monetary_policy", "economics"},
    "geopolitics": {"geopolitics", "politics"},
    "crypto": {"crypto", "corporate"},
    "sports": {"sports"},
    "general": {"politics", "geopolitics", "corporate", "economics", "monetary_policy", "crypto"},
}

# Action concepts: if both the news and the market name one, they must agree.
ACTION_CONCEPTS = {
    "acquisition", "resign", "ceasefire", "increase", "decrease", "nochange", "bankruptcy", "ipo",
    "indict", "indicted", "elected", "impeach", "tariffs", "shutdown", "default", "approve",
    "approved", "ban", "launch", "release", "sanction", "invade", "invasion", "strike",
}


def categories_compatible(event_category: str, market_category: str) -> bool:
    if market_category == "general":
        return True
    return market_category in CATEGORY_COMPAT.get(event_category, set())


class SimilarityScorer(Protocol):
    def score(self, event: NewsEvent, market: MarketRecord, index: MarketIndex) -> tuple[float, list[str], list[str]]:
        """Return (score 0..1, matched terms, reasons)."""


class LexicalScorer:
    def __init__(self, title_weight: float = 1.0, summary_weight: float = 0.4):
        self.title_weight = title_weight
        self.summary_weight = summary_weight

    def event_terms(self, event: NewsEvent) -> dict[str, float]:
        terms: dict[str, float] = {}
        for t in T.tokens(event.summary[:600]):
            terms[t] = max(terms.get(t, 0.0), self.summary_weight)
        for t in T.tokens(event.title):
            terms[t] = self.title_weight
        return terms

    def score(self, event: NewsEvent, market: MarketRecord, index: MarketIndex,
              terms: dict[str, float] | None = None) -> tuple[float, list[str], list[str]]:
        terms = terms if terms is not None else self.event_terms(event)
        m_tokens = index.doc_tokens.get(market.market_id) or T.token_set(market.search_text)
        if not m_tokens:
            return 0.0, [], ["market has no searchable text"]
        overlap = [t for t in m_tokens if t in terms]
        m_weight = sum(index.weight(t) for t in m_tokens)
        o_weight = sum(index.weight(t) * terms[t] for t in overlap)
        e_weight = sum(index.weight(t) * w for t, w in terms.items()) or 1.0
        market_cov = o_weight / m_weight if m_weight else 0.0
        event_cov = o_weight / e_weight
        score = 0.75 * market_cov + 0.25 * min(1.0, event_cov * 2)
        reasons = [f"market term coverage {market_cov:.2f}", f"event term coverage {event_cov:.2f}"]

        ev_entities = T.entities(event.title)
        m_entities = index.doc_entities.get(market.market_id, set())
        shared_entities = {e for e in ev_entities if " " in e and e in m_entities}
        if shared_entities:
            score += 0.1
            reasons.append(f"shared entities: {sorted(shared_entities)}")

        ev_actions = ACTION_CONCEPTS & set(terms)
        m_actions = ACTION_CONCEPTS & m_tokens
        if ev_actions and m_actions:
            if ev_actions & m_actions:
                score += 0.1
                reasons.append(f"action agrees: {sorted(ev_actions & m_actions)}")
            else:
                score *= 0.4
                reasons.append(f"action mismatch: news {sorted(ev_actions)} vs market {sorted(m_actions)}")

        rare = [t for t in overlap if index.weight(t) >= 0.5 * index.max_idf]
        if not rare and not shared_entities:
            score *= 0.3
            reasons.append("only common terms overlap")
        return min(1.0, score), sorted(overlap, key=lambda t: -index.weight(t)), reasons


class EventMarketMatcher:
    def __init__(self, catalog: MarketCatalog, scorer: SimilarityScorer | None = None,
                 max_candidates: int = 4, min_score: float = 0.30, candidate_pool: int = 200):
        self.catalog = catalog
        self.scorer = scorer or LexicalScorer()
        self.max_candidates = max_candidates
        self.min_score = min_score
        self.candidate_pool = candidate_pool
        self.category_filtered = 0

    def match(self, event: NewsEvent) -> list[MarketMatch]:
        index = self.catalog.index
        if not index.postings:
            return []
        lexical = self.scorer if isinstance(self.scorer, LexicalScorer) else LexicalScorer()
        terms = lexical.event_terms(event)
        # Retrieval: accumulate IDF over postings of informative event terms.
        acc: dict[str, float] = {}
        for t, w in terms.items():
            idf = index.idf.get(t)
            if idf is None or idf < 0.25 * index.max_idf:
                continue
            for mid in index.postings[t]:
                acc[mid] = acc.get(mid, 0.0) + idf * w
        pool = sorted(acc, key=lambda m: -acc[m])[: self.candidate_pool]

        matches: list[MarketMatch] = []
        for mid in pool:
            market = self.catalog.markets.get(mid)
            if market is None:
                continue
            if not categories_compatible(event.category, market.category):
                self.category_filtered += 1
                continue
            if isinstance(self.scorer, LexicalScorer):
                score, matched, reasons = self.scorer.score(event, market, index, terms)
            else:
                score, matched, reasons = self.scorer.score(event, market, index)
            if score >= self.min_score:
                matches.append(MarketMatch(market, round(score, 4), matched, reasons))
        matches.sort(key=lambda m: -m.score)
        return matches[: self.max_candidates]
