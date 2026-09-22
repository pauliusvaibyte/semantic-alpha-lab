from datetime import datetime, timedelta, timezone

from semantic_alpha.labels import build_labels
from semantic_alpha.schema import MarketSnapshot
from semantic_alpha.storage import Store


def test_labels_include_path_excursions(tmp_path):
    s=Store(str(tmp_path/'l.db')); t=datetime(2026,1,1,tzinfo=timezone.utc)
    for minute,price in [(0,100),(1,102),(2,98),(5,101),(15,103),(30,104),(60,105),(240,106),(1440,107)]:
        s.save_market(MarketSnapshot(symbol='SOLUSDT',ts=t+timedelta(minutes=minute),last=price))
        s.save_market(MarketSnapshot(symbol='BTCUSDT',ts=t+timedelta(minutes=minute),last=100+minute*.001))
    l=build_labels(s,'SOLUSDT',t,max_delay_seconds=1)
    assert l is not None
    assert l.max_up['5m'] >= .02
    assert l.max_down['5m'] <= -.02
