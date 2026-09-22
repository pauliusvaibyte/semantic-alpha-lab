from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from semantic_alpha.factor_analysis import cross_sectional_factor_ic, factor_monotonicity


def test_factor_ic_detects_rank_signal():
    rows=[]; t=datetime(2026,1,1,tzinfo=timezone.utc)
    syms=['A','B','C','D','E']
    for i in range(40):
        for j,s in enumerate(syms):
            x=float(j-2)
            rows.append({'ts':t+timedelta(minutes=5*i),'symbol':s,'factor':x,'target_return':.001*x+0.00001*i})
    d=pd.DataFrame(rows)
    r=cross_sectional_factor_ic(d,'factor')
    assert r['ok'] and r['mean_ic'] > .9 and r['mean_top_bottom_return_spread'] > 0


def test_monotonicity():
    x=np.linspace(-2,2,100)
    d=pd.DataFrame({'x':x,'target_return':x*.001})
    r=factor_monotonicity(d,'x',bins=5)
    assert len(r)==5 and r.mean_target_return.iloc[-1] > r.mean_target_return.iloc[0]
