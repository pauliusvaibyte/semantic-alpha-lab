import numpy as np
import pandas as pd

from semantic_alpha.backtest import stateful_trade_vectors


def test_stateful_backtest_blocks_overlapping_same_symbol():
    d=pd.DataFrame({
        'ts':pd.date_range('2026-01-01',periods=20,freq='min',tz='UTC'),
        'symbol':['SOLUSDT']*20,
        'target_return':[.01]*20,
    })
    sig=np.ones(20,dtype=int); costs=np.zeros(20)
    net,accepted=stateful_trade_vectors(d,sig,costs,horizon_minutes=15,max_positions=3)
    assert accepted.sum()==2
    assert net[accepted].sum()==.02
