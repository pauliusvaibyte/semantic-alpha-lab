import pandas as pd
from semantic_alpha.backtest import probabilistic_sharpe_ratio, trade_metrics


def test_psr_positive_series_high():
    r=pd.Series([.002,-.0005,.0015,.001,.0025,-.0003,.0012,.0008,.0017,.0011,.0009,.0013])
    p=probabilistic_sharpe_ratio(r)
    assert p is not None and .5 < p <= 1
    assert trade_metrics(r)['probabilistic_sharpe_gt_zero'] == p
