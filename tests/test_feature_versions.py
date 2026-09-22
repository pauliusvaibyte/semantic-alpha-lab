from datetime import datetime, timezone
from semantic_alpha.features import FeatureEngine
from semantic_alpha.schema import MarketSnapshot
from semantic_alpha.storage import Store


def test_feature_engine_preserves_parallel_versions(tmp_path):
    s=Store(str(tmp_path/'v.db'))
    t=datetime(2026,1,1,tzinfo=timezone.utc)
    m=MarketSnapshot(symbol='SOLUSDT',ts=t,last=100,bid=99.9,ask=100.1)
    s.save_market(m)
    a=FeatureEngine(s,semantic_model='jev-x',feature_version='v4__jev').build('SOLUSDT',t,m)
    b=FeatureEngine(s,semantic_model='heuristic-v1',feature_version='v4__heuristic').build('SOLUSDT',t,m)
    s.save_features(a); s.save_features(b)
    assert len(s.all_features('v4__jev'))==1
    assert len(s.all_features('v4__heuristic'))==1
    assert a.feature_version != b.feature_version
