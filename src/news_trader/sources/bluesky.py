"""
Bluesky official-account provider (public AppView, no key required).

Polls ``app.bsky.feed.getAuthorFeed`` for a fixed list of verified accounts
(wire services and official institutions).  Shape verified live: {"feed":
[{"post": {uri, author{handle, verification{verifiedStatus}}, record{text,
createdAt, embed}, indexedAt}, "reason"?}]}.

Impersonation guard: an item is only attributed to the configured account if
the author handle matches exactly AND Bluesky reports a valid verification.
Reposts (items with ``reason``) are skipped: they are someone else's content.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.news_trader.http import HttpClient, SchemaError
from src.news_trader.models import RawNewsItem
from src.news_trader.sources.base import NewsProvider, SourceAssessment, SourceProfile, parse_timestamp

# Credibility cap for a post whose author is not (or no longer) verified.
UNVERIFIED_ACCOUNT_CREDIBILITY = 0.2

AUTHOR_FEED_URL = "https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed"


@dataclass(frozen=True)
class BlueskyAccount:
    handle: str
    name: str
    credibility: float
    primary_official: bool = False
    default_category: str = "general"


class BlueskyProvider(NewsProvider):
    def __init__(self, profile: SourceProfile, accounts: list[BlueskyAccount], per_account: int = 10):
        super().__init__(profile)
        self.accounts = {a.handle: a for a in accounts}
        self.per_account = per_account

    async def fetch(self, http: HttpClient, now: float) -> list[RawNewsItem]:
        out: list[RawNewsItem] = []
        errors: list[str] = []
        for handle in self.accounts:
            try:
                data = await http.get_json(AUTHOR_FEED_URL, params={
                    "actor": handle, "limit": str(self.per_account),
                    "filter": "posts_no_replies",
                })
                out.extend(self.parse(handle, data, now))
            except SchemaError as e:
                errors.append(str(e))
        if errors and not out:
            raise SchemaError("; ".join(errors))
        return out

    def assess(self, item: RawNewsItem) -> SourceAssessment:
        account = self.accounts.get(item.author_handle)
        if account is None or not item.author_verified:
            return SourceAssessment(f"bsky:{item.author_handle}", UNVERIFIED_ACCOUNT_CREDIBILITY,
                                    False, "general")
        return SourceAssessment(account.handle, account.credibility,
                                account.primary_official, account.default_category)

    def parse(self, handle: str, data, now: float) -> list[RawNewsItem]:
        if not isinstance(data, dict) or not isinstance(data.get("feed"), list):
            raise SchemaError(f"bluesky {handle}: missing feed list")
        account = self.accounts[handle]
        out: list[RawNewsItem] = []
        for entry in data["feed"]:
            if not isinstance(entry, dict) or entry.get("reason"):
                continue
            post = entry.get("post") or {}
            author = post.get("author") or {}
            record = post.get("record") or {}
            if author.get("handle") != handle:
                continue
            text = (record.get("text") or "").strip()
            if not text or not post.get("uri"):
                continue
            verified = (author.get("verification") or {}).get("verifiedStatus") == "valid"
            created = parse_timestamp(record.get("createdAt"))
            indexed = parse_timestamp(post.get("indexedAt"))
            # createdAt is client-supplied; indexedAt is when the AppView saw it.
            published = min(t for t in (created, indexed) if t is not None) if (created or indexed) else None
            out.append(RawNewsItem(
                provider=self.key,
                source_name=account.name,
                url=self._link(post) or self._post_url(handle, post["uri"]),
                guid=post["uri"],
                title=text.split("\n", 1)[0][:300],
                summary=text[:2000],
                published_at=published,
                fetched_at=now,
                raw_category=account.default_category,
                author_handle=handle,
                author_verified=verified,
            ))
        return out

    @staticmethod
    def _link(post: dict) -> str:
        embed = (post.get("record") or {}).get("embed") or {}
        external = embed.get("external") or {}
        return external.get("uri") or ""

    @staticmethod
    def _post_url(handle: str, uri: str) -> str:
        rkey = uri.rsplit("/", 1)[-1]
        return f"https://bsky.app/profile/{handle}/post/{rkey}"
