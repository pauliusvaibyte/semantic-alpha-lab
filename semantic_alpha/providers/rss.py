"""Primary-source ingestion: RSS/Atom feeds (exchange blogs, project
announcements, regulators, news wires) normalized into SocialPost rows that flow
through the same semantic + narrative + label pipeline as X posts.

Primary sources are where information enters; X mostly tells us how fast it is
propagating. Items are matched to tracked assets with the same cashtag/name
matching used for stream posts; unmatched items land under symbol "MACRO".
"""
from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from ..schema import SocialPost
from .x_stream import symbols_from_tag_or_text

_ATOM = "{http://www.w3.org/2005/Atom}"


def _dt(v: str | None) -> datetime:
    if not v:
        return datetime.now(timezone.utc)
    v = v.strip()
    try:
        return parsedate_to_datetime(v)
    except (TypeError, ValueError):
        pass
    try:
        d = datetime.fromisoformat(v.replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)


def _text(el) -> str:
    return "".join(el.itertext()).strip() if el is not None else ""


def parse_feed(xml_text: str) -> list[dict[str, Any]]:
    """Parse RSS 2.0 or Atom into normalized item dicts."""
    root = ET.fromstring(xml_text)
    items: list[dict[str, Any]] = []
    for it in root.iter("item"):  # RSS 2.0
        items.append({
            "guid": _text(it.find("guid")) or _text(it.find("link")),
            "link": _text(it.find("link")),
            "title": _text(it.find("title")),
            "summary": _text(it.find("description")),
            "published": _dt(_text(it.find("pubDate")) or _text(it.find("date")) or None),
        })
    for it in root.iter(f"{_ATOM}entry"):  # Atom
        link_el = it.find(f"{_ATOM}link")
        items.append({
            "guid": _text(it.find(f"{_ATOM}id")) or (link_el.get("href") if link_el is not None else ""),
            "link": link_el.get("href") if link_el is not None else "",
            "title": _text(it.find(f"{_ATOM}title")),
            "summary": _text(it.find(f"{_ATOM}summary")) or _text(it.find(f"{_ATOM}content")),
            "published": _dt(_text(it.find(f"{_ATOM}published")) or _text(it.find(f"{_ATOM}updated")) or None),
        })
    return items


def items_to_posts(items: list[dict[str, Any]], source_name: str, symbols: list[str],
                   first_seen: datetime | None = None) -> list[SocialPost]:
    first_seen = first_seen or datetime.now(timezone.utc)
    out: list[SocialPost] = []
    for it in items:
        text = re.sub(r"<[^>]+>", " ", f"{it['title']} {it['summary']}").strip()
        text = re.sub(r"\s+", " ", text)
        if not text:
            continue
        pid = "rss-" + hashlib.sha1((it["guid"] or it["link"] or it["title"]).encode()).hexdigest()[:20]
        matched = symbols_from_tag_or_text(None, text, symbols)
        raw_item = {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in it.items()}
        for sym in (matched or ["MACRO"]):
            out.append(SocialPost(
                post_id=pid, platform="news", symbol=sym,
                author_username=source_name, text=text[:4000],
                created_at=it["published"], first_seen_at=first_seen,
                urls=[it["link"]] if it["link"] else [],
                raw={**raw_item, "_usage": {
                    "provider": "rss", "category": "feed_read", "reference_id": pid,
                    "units": 1.0, "unit_name": "post", "estimated_usd": 0.0,
                    "cost_source": "free",
                }},
            ))
    return out


class RSSClient:
    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout

    async def fetch(self, url: str) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            r = await client.get(url, headers={"User-Agent": "semantic-alpha-lab/0.4"})
            r.raise_for_status()
            return parse_feed(r.text)
