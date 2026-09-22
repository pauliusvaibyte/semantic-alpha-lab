from datetime import datetime, timedelta, timezone
from semantic_alpha.schema import MarketSnapshot
from semantic_alpha.storage import Store


def test_market_compaction_dry_run_and_execute(tmp_path):
    s=Store(str(tmp_path/'s.db')); t=datetime(2026,1,1,tzinfo=timezone.utc)
    for i in range(120):
        s.save_market(MarketSnapshot(symbol='SOLUSDT',ts=t+timedelta(seconds=i),last=100+i*.001))
    dry=s.compact_market_range(t,t+timedelta(minutes=2),bucket_seconds=60,execute=False)
    assert dry['scanned']==120 and dry['would_delete']==118
    assert len(s.market_between('SOLUSDT',t,t+timedelta(minutes=2)))==120
    done=s.compact_market_range(t,t+timedelta(minutes=2),bucket_seconds=60,execute=True)
    assert done['deleted']==118
    assert len(s.market_between('SOLUSDT',t,t+timedelta(minutes=2)))==2
