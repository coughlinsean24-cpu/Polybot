"""
Deterministic resolution-rule validation.

Before any AI analysis, a market's own wording is checked: it must contain an
explicit resolution clause, and must not lean on discretionary or subjective
language.  Markets whose rules cannot be read reliably are rejected; the
score and the specific issues are journaled so rejections are auditable.

This is a screen, not an interpreter: the LLM still receives the full rule
text and must state how the news maps onto it, and nothing trades on an LLM
claim that contradicts these checks.
"""

from __future__ import annotations

import re

from src.news_trader.models import MarketRecord, ResolutionCheck

_EXPLICIT_RULE_RE = re.compile(
    r"(?:will|shall)\s+resolve\s+to\s+[\"“']?[^\"”'\n]{1,80}[\"”']?\s+if\b"
    r"|resolves?\s+(?:to\s+)?[\"“']?(?:yes|no)[\"”']?\s+if\b"
    r"|(?:will|shall)\s+resolve\s+(?:according\s+to|in\s+favou?r\s+of|based\s+on|to\s+the)\b"
    r"|\bif\b[^.]{1,200}?(?:will|shall)\s+resolve\s+to\s+[\"“']",
    re.IGNORECASE,
)
_SOURCE_RE = re.compile(
    r"resolution\s+source\s+(?:for\s+this\s+market\s+)?(?:will\s+be|is)\s+([^.\n]{3,200})",
    re.IGNORECASE,
)
_DEADLINE_RE = re.compile(
    r"\b(?:by|before|on or before|no later than|through|until)\s+"
    r"((?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2}(?:,\s*\d{4})?(?:,?\s*\d{1,2}:\d{2}\s*(?:AM|PM)?\s*(?:ET|UTC|EST|EDT)?)?)",
    re.IGNORECASE,
)

# (pattern, weight, issue label).  Weights for Polymarket boilerplate that
# appears in most well-specified markets (media-consensus fallback, 50-50 on
# cancellation, "if ambiguous" clauses) are deliberately small.
AMBIGUITY_MARKERS: list[tuple[re.Pattern, float, str]] = [
    (re.compile(r"sole discretion|at (?:its|the) discretion", re.I), 0.35, "discretionary resolution"),
    (re.compile(r"in the opinion of|subjective", re.I), 0.30, "subjective judgement"),
    (re.compile(r"\bmay (?:be )?resolve", re.I), 0.20, "conditional/optional resolution path"),
    (re.compile(r"ambigu|unclear", re.I), 0.10, "ambiguity fallback clause"),
    (re.compile(r"consensus of credible reporting", re.I), 0.10, "resolves on media consensus"),
    (re.compile(r"clarification|update(?:d)? on \w+ \d", re.I), 0.15, "rules amended by clarification"),
    (re.compile(r"substantially similar|or similar|or equivalent", re.I), 0.15, "open-ended equivalence"),
    (re.compile(r"50[-/ ]50|fifty[- ]fifty", re.I), 0.10, "can resolve 50-50"),
    (re.compile(r"including but not limited to", re.I), 0.10, "open-ended criteria list"),
    (re.compile(r"\bdeemed\b", re.I), 0.10, "deemed outcome"),
]


def check_resolution(market: MarketRecord, max_ambiguity: float) -> ResolutionCheck:
    desc = market.description or ""
    issues: list[str] = []
    score = 0.0

    has_rule = bool(_EXPLICIT_RULE_RE.search(desc))
    if not has_rule:
        score += 0.4
        issues.append("no explicit 'will resolve to X if' clause")
    if len(desc.strip()) < 80:
        score += 0.3
        issues.append("resolution text too short to interpret")

    source = market.resolution_source.strip()
    if not source:
        m = _SOURCE_RE.search(desc)
        source = m.group(1).strip() if m else ""
    if not source:
        score += 0.1
        issues.append("no named resolution source")

    for pattern, weight, label in AMBIGUITY_MARKERS:
        if pattern.search(desc):
            score += weight
            issues.append(label)

    deadline = ""
    dm = _DEADLINE_RE.search(desc) or _DEADLINE_RE.search(market.question)
    if dm:
        deadline = dm.group(1)

    score = min(1.0, score)
    ok = has_rule and score <= max_ambiguity
    return ResolutionCheck(
        ok=ok,
        ambiguity_score=round(score, 3),
        resolution_source=source[:200],
        has_explicit_rule=has_rule,
        deadline_text=deadline,
        issues=issues,
    )


def rules_excerpt(market: MarketRecord, limit: int = 3500) -> str:
    """Full rule text for the LLM (truncation is flagged, never silent)."""
    desc = market.description or ""
    if len(desc) <= limit:
        return desc
    return desc[:limit] + "\n[...rules truncated; treat any unseen clause as unknown...]"
