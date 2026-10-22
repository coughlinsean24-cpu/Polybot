"""
RSS 2.0 / Atom provider (stdlib XML parsing).

Formats were checked against live responses from each default feed; see
``sources.registry`` for the list.  SEC EDGAR's current-filings Atom feed gets
light special handling so that the headline carries the filer name and the
8-K item descriptions (e.g. "Item 2.01: Completion of Acquisition").
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from src.news_trader.http import HttpClient, SchemaError
from src.news_trader.models import RawNewsItem
from src.news_trader.sources.base import NewsProvider, SourceProfile, parse_timestamp, strip_html

ATOM_NS = "{http://www.w3.org/2005/Atom}"
DC_NS = "{http://purl.org/dc/elements/1.1/}"

_SEC_TITLE_RE = re.compile(r"^(?P<form>[\w\-/]+)\s+-\s+(?P<name>.+?)\s+\(\d{6,}\)\s+\((?P<role>[^)]+)\)\s*$")
_SEC_ITEM_RE = re.compile(r"Item\s+\d+\.\d+:\s*[^<\n]+")


def _child_text(el: ET.Element, *names: str) -> str:
    for name in names:
        child = el.find(name)
        if child is not None and (child.text or "").strip():
            return child.text.strip()
    return ""


class RssAtomProvider(NewsProvider):
    def __init__(self, profile: SourceProfile, max_items: int = 50):
        super().__init__(profile)
        self.max_items = max_items

    async def fetch(self, http: HttpClient, now: float) -> list[RawNewsItem]:
        body = await http.get_text(self.profile.url)
        return self.parse(body, now)

    def parse(self, body: str, now: float) -> list[RawNewsItem]:
        try:
            root = ET.fromstring(body.lstrip("﻿").strip())
        except ET.ParseError as e:
            raise SchemaError(f"{self.key}: malformed XML: {e}") from e

        items = root.findall(".//item")
        if items:
            out = [self._parse_rss_item(it, now) for it in items[: self.max_items]]
        else:
            entries = root.findall(f"{ATOM_NS}entry")
            if not entries and root.tag not in ("rss", f"{ATOM_NS}feed"):
                raise SchemaError(f"{self.key}: neither RSS nor Atom (root={root.tag})")
            out = [self._parse_atom_entry(e, now) for e in entries[: self.max_items]]
        return [i for i in out if i is not None]

    def _parse_rss_item(self, it: ET.Element, now: float) -> RawNewsItem | None:
        title = strip_html(_child_text(it, "title"), 500)
        link = _child_text(it, "link")
        guid = _child_text(it, "guid") or link
        if not title or not (link or guid):
            return None
        pub = _child_text(it, "pubDate", f"{DC_NS}date")
        published = parse_timestamp(pub)
        if self.profile.key == "bls_latest" and published is not None:
            # BLS republishes one rolling item; key it by publication time.
            guid = f"{guid}#{int(published)}"
        return RawNewsItem(
            provider=self.key,
            source_name=self.profile.name,
            url=link or guid,
            guid=guid,
            title=title,
            summary=strip_html(_child_text(it, "description")),
            published_at=published,
            fetched_at=now,
            raw_category=_child_text(it, "category"),
        )

    def _parse_atom_entry(self, e: ET.Element, now: float) -> RawNewsItem | None:
        title = strip_html(_child_text(e, f"{ATOM_NS}title"), 500)
        link_el = e.find(f"{ATOM_NS}link")
        link = link_el.attrib.get("href", "") if link_el is not None else ""
        guid = _child_text(e, f"{ATOM_NS}id") or link
        if not title or not guid:
            return None
        summary_raw = _child_text(e, f"{ATOM_NS}summary", f"{ATOM_NS}content")
        summary = strip_html(summary_raw)
        cat_el = e.find(f"{ATOM_NS}category")
        category = cat_el.attrib.get("term", "") if cat_el is not None else ""
        published = parse_timestamp(
            _child_text(e, f"{ATOM_NS}published", f"{ATOM_NS}updated")
        )
        if self.profile.key.startswith("sec_"):
            title, summary = self._sec_headline(title, summary_raw, summary)
        return RawNewsItem(
            provider=self.key,
            source_name=self.profile.name,
            url=link or guid,
            guid=guid,
            title=title,
            summary=summary,
            published_at=published,
            fetched_at=now,
            raw_category=category,
        )

    @staticmethod
    def _sec_headline(title: str, summary_raw: str, summary: str) -> tuple[str, str]:
        m = _SEC_TITLE_RE.match(title)
        items = [strip_html(i) for i in _SEC_ITEM_RE.findall(summary_raw or "")]
        if not m:
            return title, summary
        name = m.group("name").title()
        headline = f"{name} files {m.group('form')}"
        if items:
            headline += ": " + "; ".join(i.split(":", 1)[1].strip() for i in items)
        return headline, summary
