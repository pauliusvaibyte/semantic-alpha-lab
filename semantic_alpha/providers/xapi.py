"""X (Twitter) social data ingestion via the twitterapi.io third-party provider.

Platform terminology is "X"; the vendor is twitterapi.io. Query syntax
(``since_time``/``until_time``/``-filter:retweets``) and response shapes are
the vendor's, not official X API v2. Official X API pay-per-use tiers lack
filtered-stream and full-archive endpoints (Enterprise-only), so ingestion
stays here until an Enterprise contract is justified.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator
import re

import httpx

from ..schema import SocialPost
from ..config import settings


ALIASES = {
    "BTC": ["$BTC", "Bitcoin"], "ETH": ["$ETH", "Ethereum"], "SOL": ["$SOL", "Solana"],
    "DOGE": ["$DOGE", "Dogecoin"], "XRP": ["$XRP", "Ripple"], "SUI": ["$SUI", "Sui"],
    "AVAX": ["$AVAX", "Avalanche"], "LINK": ["$LINK", "Chainlink"], "ADA": ["$ADA", "Cardano"],
    "NEAR": ["$NEAR", "Near"], "APT": ["$APT", "Aptos"], "ARB": ["$ARB", "Arbitrum"],
    "OP": ["$OP", "Optimism"],
}


class XAPIClient:
    def __init__(self, api_key: str, api_url: str = "https://api.twitterapi.io/twitter/tweet/advanced_search", timeout: float = 20.0):
        self.api_key = api_key
        self.api_url = api_url
        self.timeout = timeout

    def query(self, symbol: str, start: datetime, end: datetime) -> str:
        symbol = symbol.upper().replace("$", "")
        names = ALIASES.get(symbol, [f"${symbol}"])
        expr = " OR ".join(names)
        # Historical collection uses second-resolution bounds to avoid date/pagination ambiguity.
        lang = f" lang:{settings.x_query_langs}" if settings.x_query_langs else ""
        return f"({expr}){lang} -filter:retweets since_time:{int(start.timestamp())} until_time:{int(end.timestamp())}"

    async def search_window(self, symbol: str, start: datetime, end: datetime, max_pages: int = 50, *, known_ids: set[str] | None = None) -> list[SocialPost]:
        """Paginated Latest search over [start, end).

        When ``known_ids`` is supplied, posts already stored are skipped —
        and since ``queryType=Latest`` returns reverse-chronological order,
        a page containing only known posts means everything deeper is already
        held, so pagination halts early (the vendor bills per returned tweet;
        trailing pages of duplicates are pure spend). Only leading-edge
        callers should pass it: gap backfills have unknown posts *behind* the
        known leading edge, and engagement refresh deliberately re-reads
        known posts — both must leave ``known_ids=None``."""
        if not self.api_key:
            raise RuntimeError("X_API_KEY is required")
        query = self.query(symbol, start, end)
        headers = {"X-API-Key": self.api_key}
        cursor = None
        seen_cursors: set[str] = set()
        seen_ids: set[str] = set()
        out: list[SocialPost] = []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for _ in range(max_pages):
                params: dict[str, Any] = {"query": query, "queryType": "Latest"}
                if cursor:
                    params["cursor"] = cursor
                r = await _get_with_backoff(client, self.api_url, params=params, headers=headers)
                data = r.json()
                raw = data.get("tweets") or data.get("data") or []
                new_count = 0
                for t in raw:
                    pid = str(t.get("id") or "")
                    if not pid or pid in seen_ids or t.get("isRetweet") is True:
                        continue
                    seen_ids.add(pid)
                    if known_ids is not None and pid in known_ids:
                        continue
                    t_for_storage = dict(t)
                    t_for_storage["_semantic_alpha_acquisition_id"] = f"{pid}:{symbol.upper()}:{int(start.timestamp())}:{int(end.timestamp())}"
                    p = normalize_post(t_for_storage, symbol)
                    if start <= p.created_at < end:
                        out.append(p)
                        new_count += 1
                nxt = data.get("next_cursor") or data.get("nextCursor") or data.get("cursor")
                if not nxt or nxt in seen_cursors or new_count == 0:
                    break
                seen_cursors.add(nxt)
                cursor = nxt
        out.sort(key=lambda p: p.created_at)
        return out

    async def historical_windows(self, symbol: str, start: datetime, end: datetime, window: timedelta = timedelta(minutes=15)) -> AsyncIterator[list[SocialPost]]:
        cur = start
        while cur < end:
            nxt = min(end, cur + window)
            yield await self.search_window(symbol, cur, nxt)
            cur = nxt


async def _get_with_backoff(client: httpx.AsyncClient, url: str, *, params: dict[str, Any], headers: dict[str, str], attempts: int = 4) -> httpx.Response:
    """GET with bounded retry on 429/5xx.

    The vendor rate-limits aggressively on hot cashtags mid-pagination. Honor
    Retry-After / x-rate-limit-reset when present, else exponential backoff.
    """
    last: httpx.Response | None = None
    for attempt in range(attempts):
        try:
            r = await client.get(url, params=params, headers=headers)
        except (httpx.TransportError, httpx.TimeoutException):
            # Network-level failure: nothing to honor but backoff; retry bounded.
            if attempt == attempts - 1:
                raise
            await asyncio.sleep(min(30.0, 2.0 ** attempt + 0.5))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            last = r
            if attempt == attempts - 1:
                break
            wait = None
            ra = r.headers.get("retry-after")
            if ra:
                try:
                    wait = float(ra)
                except ValueError:
                    wait = None
            if wait is None:
                reset = r.headers.get("x-rate-limit-reset")
                if reset:
                    try:
                        wait = max(0.5, float(reset) - time.time())
                    except ValueError:
                        wait = None
            if wait is None:
                wait = min(30.0, 2.0 ** attempt + 0.5)
            await asyncio.sleep(min(wait, 30.0))
            continue
        r.raise_for_status()
        return r
    assert last is not None
    last.raise_for_status()
    return last


def normalize_post(t: dict[str, Any], symbol: str) -> SocialPost:
    author = t.get("author") or {}
    created_raw = t.get("createdAt") or t.get("created_at") or t.get("createdAtTimestamp")
    created_at = parse_dt(created_raw)
    text = str(t.get("text") or "")
    urls = re.findall(r"https?://\S+", text)
    return SocialPost(
        post_id=str(t.get("id")),
        platform="x",
        symbol=symbol.upper().replace("$", ""),
        author_id=str(author.get("id") or author.get("restId") or "") or None,
        author_username=author.get("userName") or author.get("username"),
        text=text,
        created_at=created_at,
        first_seen_at=datetime.now(timezone.utc),
        likes=_i(t.get("likeCount") or t.get("likes")),
        reposts=_i(t.get("retweetCount") or t.get("retweets")),
        replies=_i(t.get("replyCount") or t.get("replies")),
        followers=_i(author.get("followers") or author.get("followersCount")),
        verified=bool(author.get("isBlueVerified") or author.get("verified")),
        urls=urls,
        raw={**t,"_usage":{
            "provider":"twitterapi.io","category":"post_read","reference_id":str(t.get("_semantic_alpha_acquisition_id") or t.get("_semantic_alpha_delivery_id") or t.get("id")),
            "units":1.0,"unit_name":"post","estimated_usd":settings.x_cost_per_post_usd,
            "cost_source":"configured_rate",
        }},
    )


def parse_dt(v: Any) -> datetime:
    if isinstance(v, (int, float)):
        if v > 1e12:
            v /= 1000
        return datetime.fromtimestamp(v, tz=timezone.utc)
    if isinstance(v, str):
        s = v.strip().replace("Z", "+00:00")
        try:
            d = datetime.fromisoformat(s)
            return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
        for fmt in ("%a %b %d %H:%M:%S %z %Y", "%Y-%m-%d %H:%M:%S%z"):
            try:
                return datetime.strptime(v, fmt)
            except ValueError:
                continue
    return datetime.now(timezone.utc)


def _i(v: Any) -> int | None:
    if v in (None, ""):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
