from datetime import datetime, timezone
from semantic_alpha.schema import SocialPost, PostSemantics, MarketSnapshot
from semantic_alpha.storage import Store

def test_point_in_time_storage(tmp_path):
    s=Store(str(tmp_path/'x.db'))
    t=datetime.now(timezone.utc)
    p=SocialPost(post_id='1',symbol='SOLUSDT',text='buy $SOL',created_at=t,first_seen_at=t)
    assert s.save_post(p)
    assert not s.save_post(p)
    sem=PostSemantics(post_id='1',symbol='SOLUSDT',model='m',trade_intent='LONG')
    s.save_semantics(sem)
    m=MarketSnapshot(symbol='SOLUSDT',ts=t,last=100)
    s.save_market(m)
    assert len(s.posts_between('SOLUSDT',t.replace(microsecond=0), t.replace(microsecond=0))) == 0
    assert s.latest_market('SOLUSDT').last == 100
