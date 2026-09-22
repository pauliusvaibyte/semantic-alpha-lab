from datetime import datetime, timedelta, timezone
import pandas as pd
from semantic_alpha.models import SemanticResidualModel
from semantic_alpha.paper import paper_step
from semantic_alpha.schema import FeatureSnapshot, MarketSnapshot
from semantic_alpha.storage import Store


def _model(now):
    rows=[]
    for i in range(140):
        x=(i%7)-3
        rows.append({'symbol':'SOLUSDT','ts':now-timedelta(minutes=300-i),'target_return':.001*x,'mkt':x*.1,'sem':x})
    d=pd.DataFrame(rows)
    return SemanticResidualModel.fit(d,['mkt'],['mkt','sem'],model_name='logistic',horizon='15m',calibrate_thresholds=False)


def test_stale_feature_is_not_traded(tmp_path):
    s=Store(str(tmp_path/'p.db')); now=datetime.now(timezone.utc); m=_model(now)
    f=FeatureSnapshot(symbol='SOLUSDT',ts=now-timedelta(minutes=5),values={'mkt':0.0,'sem':3.0,'spread_bps':1,'depth_bid_10bps':1e6,'depth_ask_10bps':1e6})
    s.save_features(f); s.save_market(MarketSnapshot(symbol='SOLUSDT',ts=now,last=100,bid=99.9,ask=100.1))
    r=paper_step(s,m,now=now,max_feature_age_seconds=60)
    assert not r['opened'] and r['skipped']['stale_feature']==1
