from __future__ import annotations

from datetime import datetime, timedelta

from .providers.xapi import XAPIClient
from .storage import Store


def _base_symbol(market_symbol: str) -> str:
    s=market_symbol.upper().replace('/','')
    for q in ('USDT','USDC','USD'):
        if s.endswith(q): return s[:-len(q)]
    return s


async def ingest_historical_social(
    store: Store, client: XAPIClient, symbol: str, start: datetime, end: datetime,
    window_minutes: int = 15,
) -> dict[str, int]:
    """Fetch by base ticker but persist under the canonical market symbol.

    Example: `SOLUSDT` queries X for SOL/Solana while stored posts remain
    `SOLUSDT`, allowing direct joins with Bybit market/features/labels.
    """
    market_symbol=symbol.upper().replace('/','')
    query_symbol=_base_symbol(market_symbol)
    fetched=inserted=windows=0
    async for posts in client.historical_windows(query_symbol,start,end,timedelta(minutes=window_minutes)):
        windows+=1; fetched+=len(posts)
        for p in posts:
            inserted += int(store.save_post(p.model_copy(update={'symbol':market_symbol})))
    return {'windows':windows,'fetched':fetched,'inserted':inserted}
