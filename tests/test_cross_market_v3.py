from datetime import datetime, timedelta, timezone

from semantic_alpha.cross_asset import enrich_cross_asset_at
from semantic_alpha.features import FeatureEngine
from semantic_alpha.labels import estimate_market_beta
from semantic_alpha.schema import FeatureSnapshot, MarketSnapshot
from semantic_alpha.storage import Store


def test_market_history_features_and_cross_asset(tmp_path):
    s=Store(str(tmp_path/'m.db')); t=datetime(2026,1,1,tzinfo=timezone.utc)
    for i in range(61):
        ts=t+timedelta(minutes=i)
        p=100+i*.1
        s.save_market(MarketSnapshot(symbol='SOLUSDT',ts=ts,last=p,open_interest_value=1_000_000+i*1000,spread_bps=1+i*.001,book_imbalance=.1,taker_buy_ratio=.55,trade_notional_1m=10000,signed_trade_notional_1m=1000))
    m=s.latest_market('SOLUSDT')
    f=FeatureEngine(s).build('SOLUSDT',m.ts,m); s.save_features(f)
    assert f.values['return_5m']>0
    assert f.values['oi_change_30m']>0
    assert 'realized_vol_15m' in f.values

    s.save_features(FeatureSnapshot(symbol='BTCUSDT',ts=m.ts,values={'return_5m':.001,'semantic_shock':.2,'attention_z_5m':.1}))
    n=enrich_cross_asset_at(s,m.ts,symbols=['SOLUSDT','BTCUSDT'])
    assert n==2
    sol=s.latest_feature('SOLUSDT')
    assert 'relative_semantic_z' in sol.values
    assert sol.values['btc_return_5m']==.001


def test_beta_estimator_detects_high_beta_asset(tmp_path):
    s=Store(str(tmp_path/'b.db')); t=datetime(2026,1,1,tzinfo=timezone.utc)
    btc=100.0; alt=50.0
    for i in range(80):
        # Alternating but correlated hourly changes; alt is roughly 2x BTC beta.
        r=.01 if i%2 else -.008
        btc*=1+r; alt*=1+2*r
        ts=t+timedelta(hours=i)
        s.save_market(MarketSnapshot(symbol='BTCUSDT',ts=ts,last=btc))
        s.save_market(MarketSnapshot(symbol='ALTUSDT',ts=ts,last=alt))
    beta=estimate_market_beta(s,'ALTUSDT','BTCUSDT',t+timedelta(hours=79))
    assert beta>1.3
