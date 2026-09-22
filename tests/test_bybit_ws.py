from datetime import datetime, timezone
from semantic_alpha.providers.bybit_ws import MicrostructureAccumulator


def test_microstructure_accumulator():
    a=MicrostructureAccumulator('SOLUSDT')
    a.apply({'topic':'tickers.SOLUSDT','type':'snapshot','data':{
        'symbol':'SOLUSDT','lastPrice':'150','markPrice':'150.1','indexPrice':'150.0','fundingRate':'-0.0002',
        'openInterest':'10000','openInterestValue':'1500000','volume24h':'500000','turnover24h':'75000000',
        'bid1Price':'149.99','ask1Price':'150.01'}})
    a.apply({'topic':'orderbook.50.SOLUSDT','type':'snapshot','data':{'s':'SOLUSDT','b':[['149.99','10'],['149.9','20']],'a':[['150.01','12'],['150.1','15']]}})
    now=int(datetime.now(timezone.utc).timestamp()*1000)
    a.apply({'topic':'publicTrade.SOLUSDT','data':[{'T':now,'S':'Buy','v':'2','p':'150'},{'T':now,'S':'Sell','v':'1','p':'150'}]})
    a.apply({'topic':'allLiquidation.SOLUSDT','data':[{'T':now,'S':'Buy','v':'3','p':'149'}]})
    s=a.snapshot()
    assert s.last == 150
    assert s.spread_bps > 0
    assert 0 < s.taker_buy_ratio < 1
    assert s.trade_notional_1m == 450
    assert s.liquidation_long_usd_1m == 447
