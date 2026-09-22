from datetime import datetime, timedelta, timezone
import pandas as pd
from semantic_alpha.sampling import decision_points, sampling_report


def test_decision_points_reduce_dense_snapshots_without_future_labels():
    t=datetime(2026,1,1,tzinfo=timezone.utc)
    rows=[]
    for i in range(120):
        rows.append({'symbol':'SOLUSDT','ts':t+timedelta(seconds=15*i),'target_return':.001,'posts_1m':1 if i in {20,21,22,80} else 0,'semantic_shock_delta_5m':0.0,'attention_z_delta_5m':0.0})
    d=pd.DataFrame(rows)
    x=decision_points(d,min_event_spacing_minutes=2,background_spacing_minutes=10)
    r=sampling_report(d,x)
    assert len(x) < len(d)/4
    assert r['event_rows'] >= 2 and r['background_rows'] >= 1
    assert set(x.decision_reason) <= {'EVENT','BACKGROUND'}
