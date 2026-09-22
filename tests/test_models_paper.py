from datetime import datetime, timedelta, timezone

import pandas as pd

from semantic_alpha.models import SemanticResidualModel
from semantic_alpha.paper import paper_step, paper_summary
from semantic_alpha.schema import FeatureSnapshot, MarketSnapshot
from semantic_alpha.storage import Store


def test_semantic_residual_model_and_paper_trade(tmp_path):
    start=datetime(2026,1,1,tzinfo=timezone.utc)
    rows=[]
    for i in range(220):
        sem=2.0 if i%2 else -2.0
        rows.append({'symbol':'SOLUSDT','ts':start+timedelta(minutes=i),'target_return':0.01 if sem>0 else -0.01,'spread_bps':1.0,'semantic_shock':sem})
    d=pd.DataFrame(rows)
    m=SemanticResidualModel.fit(d,['spread_bps'],['spread_bps','semantic_shock'],model_name='logistic',horizon='15m',calibrate_thresholds=False)
    m.probability_threshold=.51; m.semantic_edge_threshold=.01
    path=tmp_path/'m.joblib'; m.save(path); m=SemanticResidualModel.load(path)
    score=m.score_row({'spread_bps':1.0,'semantic_shock':3.0})
    assert score['p_full_up'] > score['p_baseline_up']

    s=Store(str(tmp_path/'p.db')); live=start+timedelta(minutes=300)
    s.save_market(MarketSnapshot(symbol='SOLUSDT',ts=live,last=100,bid=99.99,ask=100.01,spread_bps=2,depth_bid_10bps=100000,depth_ask_10bps=100000))
    s.save_features(FeatureSnapshot(symbol='SOLUSDT',ts=live,values={'spread_bps':2.0,'semantic_shock':3.0,'depth_bid_10bps':100000,'depth_ask_10bps':100000}))
    r=paper_step(s,m,now=live,base_notional_usd=1000,max_positions=1)
    assert len(r['opened'])==1
    later=live+timedelta(minutes=16)
    s.save_market(MarketSnapshot(symbol='SOLUSDT',ts=later,last=101,bid=100.99,ask=101.01,spread_bps=2,depth_bid_10bps=100000,depth_ask_10bps=100000))
    s.save_features(FeatureSnapshot(symbol='SOLUSDT',ts=later,values={'spread_bps':2.0,'semantic_shock':3.0,'depth_bid_10bps':100000,'depth_ask_10bps':100000}))
    r2=paper_step(s,m,now=later,base_notional_usd=1000,max_positions=1)
    assert len(r2['closed'])==1
    assert paper_summary(s)['closed_trades']==1
