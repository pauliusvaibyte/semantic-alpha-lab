from __future__ import annotations

import asyncio
from bisect import bisect_right
from datetime import datetime, timedelta, timezone

from .providers.bybit import BybitPublicClient
from .schema import MarketSnapshot
from .storage import Store


def _dt_ms(x: str | int | float) -> datetime:
    return datetime.fromtimestamp(float(x)/1000, tz=timezone.utc)


async def ingest_bybit_history(
    store: Store,
    client: BybitPublicClient,
    symbol: str,
    start: datetime,
    end: datetime,
    interval_minutes: int = 1,
    sleep_seconds: float = 0.02,
) -> dict:
    """Backfill point-in-time price plus carry-forward OI/funding.

    Historical REST data cannot reconstruct spread/order-book/taker flow, so those
    fields remain absent rather than being fabricated. The historical research
    baseline can therefore test price/OI/funding + semantics, while the forward
    recorder supplies richer microstructure features.
    """
    if interval_minutes not in {1,3,5,15,30,60,120,240,360,720}:
        raise ValueError('Unsupported Bybit kline interval')
    symbol=symbol.upper().replace('/','')
    if start.tzinfo is None: start=start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None: end=end.replace(tzinfo=timezone.utc)

    # Pull 5-minute OI in bounded windows and funding in 30-day windows.
    oi=[]; cur=start
    while cur<end:
        nxt=min(end,cur+timedelta(minutes=5*190))
        oi.extend(await client.open_interest_window(symbol,cur,nxt,"5min",200))
        cur=nxt; await asyncio.sleep(sleep_seconds)
    funding=[]; cur=start
    while cur<end:
        nxt=min(end,cur+timedelta(days=30))
        funding.extend(await client.funding_window(symbol,cur,nxt,200))
        cur=nxt; await asyncio.sleep(sleep_seconds)

    oi_points=sorted((_dt_ms(x['timestamp']), float(x['openInterest'])) for x in oi if x.get('timestamp') and x.get('openInterest'))
    fund_points=sorted((_dt_ms(x['fundingRateTimestamp']), float(x['fundingRate'])) for x in funding if x.get('fundingRateTimestamp') and x.get('fundingRate'))
    oi_times=[x[0] for x in oi_points]; fund_times=[x[0] for x in fund_points]

    inserted=0; bars=0; cur=start
    step=timedelta(minutes=interval_minutes)
    max_rows=950
    window=step*max_rows
    while cur<end:
        nxt=min(end,cur+window)
        rows=await client.klines(symbol,cur,nxt,str(interval_minutes),1000)
        parsed=[]
        for r in rows:
            if len(r)<7: continue
            open_ts=_dt_ms(r[0]); close=float(r[4]); volume=float(r[5]); turnover=float(r[6])
            # Bybit r[0] is the candle OPEN time but r[4] is the CLOSE price, only
            # knowable at candle completion. Timestamping the close at open time
            # leaks ~one bar of lookahead into every downstream feature/label;
            # availability ts = open + interval is the honest convention.
            ts = open_ts + step
            # Keep bars *completing* inside the window; a candle opening at
            # end-step is knowable at end and must not be dropped.
            if not (start<ts<=end): continue
            oi_idx=bisect_right(oi_times,ts)-1
            f_idx=bisect_right(fund_times,ts)-1
            oi_qty=oi_points[oi_idx][1] if oi_idx>=0 else None
            fund=fund_points[f_idx][1] if f_idx>=0 else None
            parsed.append(MarketSnapshot(
                source='bybit_history',symbol=symbol,ts=ts,last=close,mark=close,index=close,
                funding_rate=fund,open_interest_qty=oi_qty,
                open_interest_value=(oi_qty*close if oi_qty is not None else None),
                raw={'interval_minutes':interval_minutes,'candle_open_ts':open_ts.isoformat(),'availability_ts':ts.isoformat(),'open':float(r[1]),'high':float(r[2]),'low':float(r[3]),'volume_interval':volume,'turnover_interval':turnover},
            ))
        for snap in sorted(parsed,key=lambda x:x.ts):
            store.save_market(snap); inserted+=1
        bars+=len(parsed); cur=nxt; await asyncio.sleep(sleep_seconds)
    return {'symbol':symbol,'bars':bars,'stored':inserted,'oi_points':len(oi_points),'funding_points':len(fund_points)}
