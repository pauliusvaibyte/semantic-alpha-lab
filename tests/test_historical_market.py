from datetime import datetime, timedelta, timezone
import pytest

from semantic_alpha.historical_market import ingest_bybit_history
from semantic_alpha.storage import Store


class FakeBybit:
    async def open_interest_window(self,symbol,start,end,interval='5min',limit=200):
        return [{'timestamp':str(int(start.timestamp()*1000)),'openInterest':'100'}]
    async def funding_window(self,symbol,start,end,limit=200):
        return [{'fundingRateTimestamp':str(int(start.timestamp()*1000)),'fundingRate':'0.0001'}]
    async def klines(self,symbol,start,end,interval='1',limit=1000):
        out=[]; cur=start
        while cur<end:
            ms=str(int(cur.timestamp()*1000)); out.append([ms,'99','101','98','100','10','1000']); cur+=timedelta(minutes=int(interval))
        return list(reversed(out))


@pytest.mark.asyncio
async def test_historical_market_backfill(tmp_path):
    s=Store(str(tmp_path/'h.db')); st=datetime(2026,1,1,tzinfo=timezone.utc); en=st+timedelta(minutes=10)
    r=await ingest_bybit_history(s,FakeBybit(),'BTCUSDT',st,en,1,0)
    xs=s.market_between('BTCUSDT',st,en)
    assert r['stored']==10
    # Candle close is stored at candle *end* (open+interval), the first instant
    # the close price is actually knowable. A 10-minute window of 1m opens
    # produces availabilities at st+1m..st+10m; the last is outside [st,en).
    assert len(xs)==9
    assert xs[0].ts==st+timedelta(minutes=1)
    assert xs[0].raw['candle_open_ts']==st.isoformat()
    # No lookahead: at the candle's open time its close must not be visible.
    assert s.nearest_market_at_or_before('BTCUSDT',st+timedelta(seconds=30)) is None
    assert s.nearest_market_at_or_before('BTCUSDT',st+timedelta(minutes=1,seconds=30)).last==100.0
    assert xs[0].funding_rate==0.0001
    assert xs[0].open_interest_value==10000
    # Re-ingest must be idempotent on (source,symbol,ts).
    r2=await ingest_bybit_history(s,FakeBybit(),'BTCUSDT',st,en,1,0)
    assert r2['stored']==10
    assert len(s.market_between('BTCUSDT',st,en+timedelta(minutes=1)))==10
