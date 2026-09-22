from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from semantic_alpha.research import FEATURE_GROUPS
from semantic_alpha.validation import semantic_permutation_placebo, semantic_staleness_test


def frame(n=360):
    rng=np.random.default_rng(22)
    d=pd.DataFrame({
        'symbol':np.array(['BTCUSDT','ETHUSDT','SOLUSDT'])[np.arange(n)%3],
        'ts':[datetime(2026,1,1,tzinfo=timezone.utc)+timedelta(minutes=i) for i in range(n)],
    })
    sem=rng.normal(size=n)
    d['target_return']=.001*sem+rng.normal(scale=.003,size=n)
    cols=sorted({x for xs in FEATURE_GROUPS.values() for x in xs})
    extras={c:rng.normal(scale=.1,size=n) for c in cols if c not in d.columns}
    d=pd.concat([d,pd.DataFrame(extras)],axis=1)
    d['semantic_shock']=sem; d['information_price_gap_proxy']=sem; d['intent_mean_5m']=sem*.8
    d['spread_bps']=2.; d['depth_bid_10bps']=1e5; d['depth_ask_10bps']=1e5
    return d


def test_semantic_placebo_runs():
    r=semantic_permutation_placebo(frame(),iterations=5)
    assert r['ok'] and r['iterations']==5 and 0 <= r['p_value'] <= 1


def test_semantic_staleness_runs():
    r=semantic_staleness_test(frame(),shifts=(1,4))
    assert list(r.stale_rows)==[0,1,4]
