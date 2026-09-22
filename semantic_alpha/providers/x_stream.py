"""X (Twitter) real-time filtered stream via the twitterapi.io vendor.

Endpoint paths and event shapes (``tweets``/``tweet``/``data`` keys,
``oapi/tweet_filter`` routes) are the vendor's wire protocol, not official
X API v2.
"""
from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import httpx

from .xapi import ALIASES


class XFilterRules:
    def __init__(self, api_key: str, base_url: str = "https://api.twitterapi.io", timeout: float = 20.0):
        if not api_key:
            raise ValueError("X_API_KEY required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key}

    async def add_rule(self, tag: str, value: str, interval_seconds: float = 5.0) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            r = await client.post("/oapi/tweet_filter/add_rule", headers=self.headers, json={
                "tag": tag, "value": value, "interval_seconds": interval_seconds,
            })
            r.raise_for_status()
            out = r.json()
            # add_rule creates rules with is_effect=0; a rule that never activates
            # silently delivers nothing. Activate it explicitly.
            rule_id = out.get("rule_id") or (out.get("data") or {}).get("rule_id")
            if rule_id:
                r2 = await client.post("/oapi/tweet_filter/update_rule", headers=self.headers, json={
                    "rule_id": rule_id, "tag": tag, "value": value,
                    "interval_seconds": interval_seconds, "is_effect": 1,
                })
                r2.raise_for_status()
                out["activated"] = r2.json()
            return out

    async def list_rules(self) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            r = await client.get("/oapi/tweet_filter/get_rules", headers=self.headers)
            r.raise_for_status(); return r.json()

    async def delete_rule(self, rule_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            r = await client.request("DELETE", "/oapi/tweet_filter/delete_rule", headers=self.headers, json={"rule_id": rule_id})
            r.raise_for_status(); return r.json()


class XRealtimeStream:
    def __init__(self, api_key: str, url: str = "wss://ws.twitterapi.io/twitter/tweet/websocket"):
        if not api_key:
            raise ValueError("X_API_KEY required")
        self.api_key = api_key
        self.url = url

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        """Yield stream events plus ``{"_control": ...}`` lifecycle markers.

        The vendor sends app-level ping events roughly every 20s, so a healthy
        connection is never silent for long; callers should treat sustained
        silence as a half-dead socket and force a reconnect.
        """
        try:
            import websockets
        except ImportError as e:
            raise RuntimeError("Install semantic-alpha-lab[stream] for X WebSocket capture") from e
        backoff = 2.0
        while True:
            try:
                try:
                    ctx = websockets.connect(self.url, additional_headers={"x-api-key": self.api_key}, ping_interval=20, ping_timeout=120, max_size=8_000_000)
                except TypeError:  # older websockets
                    ctx = websockets.connect(self.url, extra_headers={"x-api-key": self.api_key}, ping_interval=20, ping_timeout=120, max_size=8_000_000)
                async with ctx as ws:
                    backoff = 2.0
                    yield {"_control": "connected", "ts": datetime.now(timezone.utc).isoformat()}
                    async for raw in ws:
                        event=json.loads(raw)
                        yield event
            except asyncio.CancelledError:
                raise
            except Exception as e:
                yield {"_control": "disconnected", "ts": datetime.now(timezone.utc).isoformat(), "error": str(e)[:200]}
                # Back off on repeated failures: the vendor rejects rapid
                # reconnect churn (HTTP 403) and constant-rate retries can
                # turn a short penalty into a persistent one.
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 120.0)


def posts_from_event(event: dict[str, Any], default_symbol: str | None = None) -> list[tuple[str | None, Any]]:
    """Return (rule_tag, raw_post) pairs from several documented stream shapes."""
    tag = event.get("rule_tag") or event.get("tag")
    raws=[]
    if isinstance(event.get("tweets"), list): raws.extend(event["tweets"])
    elif isinstance(event.get("tweet"), dict): raws.append(event["tweet"])
    elif isinstance(event.get("data"), dict) and (event.get("event_type") in ("tweet","fast_tweet",None)):
        raws.append(event["data"])
    elif event.get("id") and event.get("text"):
        raws.append(event)
    return [(tag or default_symbol, x) for x in raws]


def symbols_from_tag_or_text(tag: str | None, text: str, symbols: list[str]) -> list[str]:
    """Return every configured asset materially named by a stream event.

    Rule tags are treated as one source of asset identity, but we also inspect the
    post itself because a single post can express cross-asset views (for example,
    ``long ETH vs BTC``). Cashtags are preferred; unprefixed matching uses known
    full asset names to avoid ambiguous tickers such as OP/NEAR/LINK.
    """
    configured={s.upper().replace("/",""):s.upper().replace("/","") for s in symbols}
    base={s.replace("USDT","").replace("USDC","").replace("USD",""):s for s in configured.values()}
    found: list[str]=[]

    def add(sym: str) -> None:
        if sym not in found:
            found.append(sym)

    if tag:
        u=tag.upper()
        for b,sym in base.items():
            # Tags are controlled by us, so ticker matching is acceptable here.
            if b in u:
                add(sym)

    up=text.upper()
    for b,sym in base.items():
        if re.search(rf"(?<![A-Z0-9])\${re.escape(b)}(?![A-Z0-9])",up):
            add(sym)
            continue
        aliases=ALIASES.get(b,[])
        for alias in aliases:
            if alias.startswith("$"):
                continue
            if re.search(rf"(?<![A-Z0-9]){re.escape(alias.upper())}(?![A-Z0-9])",up):
                add(sym); break
    return found


def symbol_from_tag_or_text(tag: str | None, text: str, symbols: list[str]) -> str | None:
    """Backward-compatible single-symbol helper; live ingestion uses the plural form."""
    found=symbols_from_tag_or_text(tag,text,symbols)
    return found[0] if found else None
