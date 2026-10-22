"""
Deterministic text utilities: tokenisation, entity extraction, aliases.

These run on every headline, so they are cheap and dependency-free.  Their
job is recall (find plausible candidates); precision comes from category and
resolution-rule checks downstream and finally the LLM.
"""

from __future__ import annotations

import re

STOPWORDS = frozenset("""
a an the and or but if of to in on at by for with from as is are was were be been
being it its this that these those will would shall should can could may might
must do does did done has have had not no yes than then so such into over under
about after before between during through up down out off again further once
here there when where why how all any both each few more most other some own same
very s t just now new says said say report reports according amid via per vs
what which who whom whose market markets price prices trade trading today week
year month day days time first last next still also announce announces announced
""".split())

# Concept aliases: every surface form maps to a canonical token so that
# "Federal Reserve", "FOMC" and "Powell" can match a market saying "Fed".
ALIASES: dict[str, str] = {
    "federal reserve": "fed", "fomc": "fed", "powell": "fed",
    "interest rate": "rates", "interest rates": "rates",
    "rate cut": "rates decrease", "rate cuts": "rates decrease",
    "rate hike": "rates increase", "rate hikes": "rates increase",
    "basis points": "bps", "basis point": "bps", "bps": "bps",
    # Direction concepts (used for action agreement in matching).
    "cut": "decrease", "cuts": "decrease", "lowers": "decrease", "lowered": "decrease",
    "reduces": "decrease", "reduced": "decrease", "decreases": "decrease", "decrease": "decrease",
    "slashes": "decrease", "hikes": "increase", "hike": "increase", "raises": "increase",
    "raised": "increase", "increases": "increase", "increase": "increase",
    "holds steady": "nochange", "hold steady": "nochange", "holds": "nochange",
    "unchanged": "nochange", "no change": "nochange", "leaves rates": "nochange rates",
    "keeps rates": "nochange rates", "pause": "nochange", "pauses": "nochange",
    "consumer price index": "cpi", "inflation": "cpi",
    "nonfarm payrolls": "payrolls", "non-farm payrolls": "payrolls",
    "jobs report": "payrolls", "unemployment rate": "unemployment",
    "jobless claims": "unemployment",
    "gross domestic product": "gdp", "recession": "gdp",
    "bitcoin": "btc", "ethereum": "eth", "ether": "eth",
    "united states": "us", "u.s.": "us", "usa": "us",
    "white house": "whitehouse", "president trump": "trump",
    "supreme court": "scotus", "securities and exchange commission": "sec",
    "acquire": "acquisition", "acquires": "acquisition", "acquired": "acquisition",
    "acquiring": "acquisition", "merger": "acquisition", "buyout": "acquisition",
    "takeover": "acquisition", "to buy": "acquisition",
    "ceasefire": "ceasefire", "cease-fire": "ceasefire", "truce": "ceasefire",
    "resigns": "resign", "resigned": "resign", "resignation": "resign",
    "steps down": "resign",
    "bankrupt": "bankruptcy", "chapter 11": "bankruptcy",
    "ipo": "ipo", "initial public offering": "ipo",
    "tariff": "tariffs", "tariffs": "tariffs",
    "shutdown": "shutdown", "government shutdown": "shutdown",
}

_ALIAS_PATTERNS = sorted(ALIASES.items(), key=lambda kv: -len(kv[0]))
_ALIAS_RE = re.compile(
    "|".join(r"(?<![\w])" + re.escape(k) + r"(?![\w])" for k, _ in _ALIAS_PATTERNS),
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-\.&']*[a-z0-9]|[a-z0-9]")
_CAP_PHRASE_RE = re.compile(r"\b([A-Z][a-zA-Z0-9&\.\-']+(?:\s+(?:[A-Z][a-zA-Z0-9&\.\-']+|of|de|and))*)")
_TICKER_RE = re.compile(r"(?:\$|\b(?:NYSE|NASDAQ|Nasdaq):\s?)([A-Z]{1,5})\b")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?")

# Words that show the item denies or retracts a claim (conflict detection).
DENIAL_TERMS = re.compile(
    r"\b(denies|denied|deny|false|fake|hoax|retract(?:s|ed|ion)?|debunk(?:s|ed)?|"
    r"not true|no plans? to|rumou?rs? (?:are )?(?:false|unfounded)|misleading)\b",
    re.IGNORECASE,
)

# Words that signal speculation rather than a factual announcement.
SPECULATION_TERMS = re.compile(
    r"\b(could|may|might|considering|weighs|weighing|mulls|explores|exploring|"
    r"reportedly|sources say|people familiar|rumou?r(?:s|ed)?|expected to|"
    r"likely to|plans to|in talks)\b",
    re.IGNORECASE,
)


def stem(tok: str) -> str:
    """Plural stripping only; deliberately conservative."""
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith(("ches", "shes", "xes", "zes", "sses")):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok


def apply_aliases(text: str) -> str:
    return _ALIAS_RE.sub(lambda m: " " + ALIASES[m.group(0).lower()] + " ", text)


def tokens(text: str, keep_stopwords: bool = False) -> list[str]:
    """Lower-cased, alias-normalised, lightly stemmed tokens."""
    text = apply_aliases(text or "").lower()
    out = []
    for tok in _TOKEN_RE.findall(text):
        tok = tok.strip(".'")
        if not tok:
            continue
        if not keep_stopwords and (tok in STOPWORDS or len(tok) < 2):
            continue
        out.append(stem(tok))
    return out


def token_set(text: str) -> set[str]:
    return set(tokens(text))


def entities(text: str) -> set[str]:
    """Capitalised phrases and tickers, normalised to token strings.

    Sentence-initial capitalisation creates some noise; downstream scoring
    weights entities by rarity so common words wash out.
    """
    found: set[str] = set()
    for m in _CAP_PHRASE_RE.finditer(text or ""):
        phrase = " ".join(tokens(m.group(1)))
        if phrase:
            found.add(phrase)
    for m in _TICKER_RE.finditer(text or ""):
        found.add(m.group(1).lower())
    return found


def numbers(text: str) -> set[str]:
    return set(_NUMBER_RE.findall(text or ""))


def shingles(text: str, n: int = 2) -> set[str]:
    toks = tokens(text)
    if len(toks) < n:
        return set(toks)
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def is_denial(text: str) -> bool:
    return bool(DENIAL_TERMS.search(text or ""))


def is_speculative(text: str) -> bool:
    return bool(SPECULATION_TERMS.search(text or ""))
