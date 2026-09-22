from datetime import datetime, timezone, timedelta
from semantic_alpha.storage import Store
from semantic_alpha.schema import MarketSnapshot
from semantic_alpha.labels import build_labels

def test_labels(tmp_path):
    s=Store(str(tmp_path/'l.db')); t=datetime.now(timezone.utc)
    for mins,price in [(0,100),(1,101),(5,102),(15,103),(30,104),(60,105),(240,106),(1440,108)]:
        s.save_market(MarketSnapshot(symbol='SOLUSDT',ts=t+timedelta(minutes=mins),last=price))
        s.save_market(MarketSnapshot(symbol='BTCUSDT',ts=t+timedelta(minutes=mins),last=100))
    l=build_labels(s,'SOLUSDT',t)
    assert l is not None
    assert l.returns['15m'] > 0.02
    assert l.abnormal_returns['15m'] == l.returns['15m']
