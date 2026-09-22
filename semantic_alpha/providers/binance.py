"""Binance USD-M futures public REST client — second venue for dispersion and
lead-lag features. No API key required for public endpoints.

Four endpoints per snapshot (book ticker, premium index/funding, open interest,
24h ticker). Snapshots are stored with source="binance"; canonical market truth
remains source="bybit" unless callers opt in otherwise.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx

from ..schema import MarketSnapshot


class BinancePublicClient:
    def __init__(self, base_url: str = "https://fapi.binance.com", timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def _get(self, client: httpx.AsyncClient, path: str, symbol: str) -> dict:
        r = await client.get(f"{self.base_url}{path}", params={"symbol": symbol})
        r.raise_for_status()
        return r.json()

    async def snapshot(self, symbol: str) -> MarketSnapshot:
        symbol = symbol.upper().replace("/", "")
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            book, premium, oi, ticker = await asyncio.gather(
                self._get(client, "/fapi/v1/ticker/bookTicker", symbol),
                self._get(client, "/fapi/v1/premiumIndex", symbol),
                self._get(client, "/fapi/v1/openInterest", symbol),
                self._get(client, "/fapi/v1/ticker/24hr", symbol),
            )
        last = _f(ticker.get("lastPrice")) or _f(premium.get("markPrice"))
        bid = _f(book.get("bidPrice"))
        ask = _f(book.get("askPrice"))
        mark = _f(premium.get("markPrice"))
        index = _f(premium.get("indexPrice"))
        oi_qty = _f(oi.get("openInterest"))
        mid = (bid + ask) / 2 if bid and ask else last
        return MarketSnapshot(
            source="binance",
            symbol=symbol,
            ts=datetime.now(timezone.utc),
            last=last or 0.0,
            bid=bid,
            ask=ask,
            mark=mark,
            index=index,
            funding_rate=_f(premium.get("lastFundingRate")),
            open_interest_qty=oi_qty,
            open_interest_value=(oi_qty * mark) if oi_qty and mark else None,
            volume_24h=_f(ticker.get("quoteVolume")),
            basis=(mark / index - 1.0) if mark and index else None,
            spread_bps=((ask - bid) / mid * 10_000) if bid and ask and mid else None,
            raw={"book": book, "premiumIndex": premium, "openInterest": oi, "ticker24hr": ticker},
        )


def _f(v) -> float | None:
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
