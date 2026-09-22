from datetime import datetime, timezone
from semantic_alpha.quality import data_quality
from semantic_alpha.schema import MarketSnapshot, SocialPost
from semantic_alpha.storage import Store


def test_quality_report(tmp_path):
    s=Store(str(tmp_path/'q.db')); now=datetime.now(timezone.utc)
    s.save_post(SocialPost(post_id='1',symbol='BTCUSDT',text='x',created_at=now,first_seen_at=now))
    s.save_market(MarketSnapshot(symbol='BTCUSDT',ts=now,last=1))
    r=data_quality(s)
    assert r['totals']['posts']==1
    assert 'BTCUSDT' in r['symbols']
