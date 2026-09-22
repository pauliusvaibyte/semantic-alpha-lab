import asyncio
from datetime import datetime, timezone, timedelta
from semantic_alpha.storage import Store
from semantic_alpha.schema import SocialPost, MarketSnapshot
from semantic_alpha.semantics.heuristic import HeuristicSemanticEngine
from semantic_alpha.features import FeatureEngine

def test_feature_engine(tmp_path):
    s=Store(str(tmp_path/'f.db')); t=datetime.now(timezone.utc); e=HeuristicSemanticEngine()
    for i in range(6):
        p=SocialPost(post_id=str(i),symbol='SOLUSDT',author_username=f'u{i}',text="I'm buying $SOL partnership announced",created_at=t-timedelta(seconds=i*20),first_seen_at=t)
        s.save_post(p); s.save_semantics(asyncio.run(e.classify(p)))
    m=MarketSnapshot(symbol='SOLUSDT',ts=t,last=100,funding_rate=.0001,open_interest_value=1e9,spread_bps=1,book_imbalance=.1,taker_buy_ratio=.6)
    s.save_market(m)
    # Feature time must be >= classification time: posts only count once observed
    # AND classified (point-in-time availability, fixed in the correctness release).
    f=FeatureEngine(s,e.model).build('SOLUSDT',datetime.now(timezone.utc),m)
    assert f.values['intent_sum_5m']>0
    assert f.values['new_info_rate_5m']>0
    assert 'information_price_gap_proxy' in f.values
