from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx

from ..schema import MarketSnapshot


class BybitPublicClient:
    """Public Bybit V5 market-data client. No API key required.

    Retries transient failures (HTTP 429/5xx, retCode 10006 rate limit, network
    errors) with bounded exponential backoff honoring Retry-After. After
    ``max_retries`` the error propagates — a caller must see provider failure,
    never silently-fresh stale data."""

    RETRYABLE_RETCODES = {10006, 10016}  # rate limit, server error
    RATE_LIMIT_RETCODES = {10006}

    def __init__(self, base_url: str = "https://api.bybit.com", timeout: float = 10.0, max_retries: int = 4):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        delay = 0.5
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
                    r = await client.get(path, params=params)
                if r.status_code in (429,) or r.status_code >= 500:
                    retry_after = _retry_after(r.headers)
                    if attempt >= self.max_retries:
                        r.raise_for_status()
                    await asyncio.sleep(retry_after if retry_after is not None else delay)
                    delay = min(delay * 2, 8.0)
                    continue
                r.raise_for_status()
                data = r.json()
                if data.get("retCode") in self.RETRYABLE_RETCODES:
                    if attempt >= self.max_retries:
                        raise RuntimeError(f"Bybit error {data.get('retCode')}: {data.get('retMsg')}")
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 8.0)
                    continue
                if data.get("retCode") not in (None, 0):
                    raise RuntimeError(f"Bybit error {data.get('retCode')}: {data.get('retMsg')}")
                return data
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last_exc = e
                if attempt >= self.max_retries:
                    raise
                await asyncio.sleep(delay)
                delay = min(delay * 2, 8.0)
        raise last_exc or RuntimeError("Bybit request failed")

    async def ticker(self, symbol: str) -> dict[str, Any]:
        d = await self._get("/v5/market/tickers", {"category": "linear", "symbol": symbol})
        rows = d.get("result", {}).get("list", [])
        if not rows:
            raise ValueError(f"No Bybit linear ticker for {symbol}")
        return rows[0]

    async def orderbook(self, symbol: str, limit: int = 50) -> dict[str, Any]:
        d = await self._get("/v5/market/orderbook", {"category": "linear", "symbol": symbol, "limit": limit})
        return d.get("result", {})

    async def trades(self, symbol: str, limit: int = 200) -> list[dict[str, Any]]:
        d = await self._get("/v5/market/recent-trade", {"category": "linear", "symbol": symbol, "limit": limit})
        return d.get("result", {}).get("list", [])

    async def funding_history(self, symbol: str, limit: int = 200) -> list[dict[str, Any]]:
        d = await self._get("/v5/market/funding/history", {"category": "linear", "symbol": symbol, "limit": limit})
        return d.get("result", {}).get("list", [])

    async def open_interest_history(self, symbol: str, interval: str = "5min", limit: int = 200) -> list[dict[str, Any]]:
        d = await self._get("/v5/market/open-interest", {
            "category": "linear", "symbol": symbol, "intervalTime": interval, "limit": limit
        })
        return d.get("result", {}).get("list", [])


    async def klines(self, symbol: str, start: datetime, end: datetime, interval: str = "1", limit: int = 1000) -> list[list[str]]:
        d = await self._get("/v5/market/kline", {
            "category":"linear", "symbol":symbol.upper(), "interval":interval,
            "start":int(start.timestamp()*1000), "end":int(end.timestamp()*1000), "limit":limit,
        })
        return d.get("result", {}).get("list", [])

    async def open_interest_window(self, symbol: str, start: datetime, end: datetime, interval: str = "5min", limit: int = 200) -> list[dict[str, Any]]:
        d = await self._get("/v5/market/open-interest", {
            "category":"linear", "symbol":symbol.upper(), "intervalTime":interval,
            "startTime":int(start.timestamp()*1000), "endTime":int(end.timestamp()*1000), "limit":limit,
        })
        return d.get("result", {}).get("list", [])

    async def funding_window(self, symbol: str, start: datetime, end: datetime, limit: int = 200) -> list[dict[str, Any]]:
        d = await self._get("/v5/market/funding/history", {
            "category":"linear", "symbol":symbol.upper(),
            "startTime":int(start.timestamp()*1000), "endTime":int(end.timestamp()*1000), "limit":limit,
        })
        return d.get("result", {}).get("list", [])

    async def snapshot(self, symbol: str) -> MarketSnapshot:
        symbol = symbol.upper().replace("/", "")
        ticker, book, trades = await asyncio.gather(
            self.ticker(symbol), self.orderbook(symbol), self.trades(symbol)
        )
        bid = _f(ticker.get("bid1Price"))
        ask = _f(ticker.get("ask1Price"))
        last = _f(ticker.get("lastPrice")) or 0.0
        spread_bps = ((ask - bid) / ((ask + bid) / 2) * 10_000) if bid and ask else None

        bids = [(float(p), float(q)) for p, q in book.get("b", [])]
        asks = [(float(p), float(q)) for p, q in book.get("a", [])]
        mid = ((bid + ask) / 2) if bid and ask else last
        bid_depth = sum(p * q for p, q in bids if mid and p >= mid * (1 - 0.001))
        ask_depth = sum(p * q for p, q in asks if mid and p <= mid * (1 + 0.001))
        denom = bid_depth + ask_depth
        imbalance = ((bid_depth - ask_depth) / denom) if denom else None

        buy_qty = 0.0
        sell_qty = 0.0
        for t in trades:
            qty = _f(t.get("size")) or 0.0
            if str(t.get("side", "")).lower() == "buy":
                buy_qty += qty
            else:
                sell_qty += qty
        tq = buy_qty + sell_qty
        taker_buy_ratio = buy_qty / tq if tq else None
        mark = _f(ticker.get("markPrice"))
        index = _f(ticker.get("indexPrice"))
        # Bybit V5 tickers do not carry a basis field; derive it from mark/index
        # the same way the Binance snapshot does.
        basis = _f(ticker.get("basis")) or (mark / index - 1.0 if mark and index else None)

        return MarketSnapshot(
            source="bybit",
            symbol=symbol,
            ts=datetime.now(timezone.utc),
            last=last,
            bid=bid,
            ask=ask,
            mark=mark,
            index=index,
            funding_rate=_f(ticker.get("fundingRate")),
            open_interest_qty=_f(ticker.get("openInterest")),
            open_interest_value=_f(ticker.get("openInterestValue")),
            volume_24h=_f(ticker.get("volume24h")),
            turnover_24h=_f(ticker.get("turnover24h")),
            basis=basis,
            spread_bps=spread_bps,
            book_imbalance=imbalance,
            depth_bid_10bps=bid_depth,
            depth_ask_10bps=ask_depth,
            taker_buy_ratio=taker_buy_ratio,
            raw={"ticker": ticker, "orderbook_ts": book.get("ts"), "trade_count": len(trades)},
        )


def _retry_after(headers: httpx.Headers) -> float | None:
    v = headers.get("Retry-After") or headers.get("X-Bapi-Limit-Reset-Timestamp")
    if not v:
        return None
    try:
        val = float(v)
    except ValueError:
        return None
    # A millisecond epoch in the future vs a plain seconds count.
    if val > 1e12:
        return max(0.0, val / 1000.0 - datetime.now(timezone.utc).timestamp())
    return min(max(val, 0.0), 30.0)


def _f(x: Any) -> float | None:
    if x in (None, ""):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
