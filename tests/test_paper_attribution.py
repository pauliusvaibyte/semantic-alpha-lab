from datetime import datetime, timezone
from semantic_alpha.paper import paper_attribution, _risk_sized_notional
from semantic_alpha.schema import PaperTrade
from semantic_alpha.storage import Store


def test_risk_sizing_brakes_high_vol():
    low=_risk_sized_notional({'realized_vol_15m':.001},equity_usd=100000,risk_per_trade=.0025,max_notional_usd=20000,signal_strength=1)
    high=_risk_sized_notional({'realized_vol_15m':.03},equity_usd=100000,risk_per_trade=.0025,max_notional_usd=20000,signal_strength=1)
    assert high < low


def test_paper_attribution(tmp_path):
    s=Store(str(tmp_path/'x.db'))
    now=datetime.now(timezone.utc)
    for i,(sym,side,ret,reg) in enumerate([('SOLUSDT','LONG',.01,'FLOW_UP'),('SOLUSDT','LONG',-.005,'FLOW_UP'),('BTCUSDT','SHORT',.02,'QUIET')]):
        s.save_paper_trade(PaperTrade(
            trade_id=str(i),symbol=sym,side=side,opened_at=now,entry_price=100,notional_usd=1000,
            model_version='x',model_fingerprint='f',horizon_minutes=15,p_baseline_up=.5,p_full_up=.7,
            semantic_edge_logodds=.4+i*.1,signal_strength=.5,entry_regime=reg,status='CLOSED',closed_at=now,
            exit_price=101,realized_return=ret,
        ))
    r=paper_attribution(s,'f')
    assert r['closed_trades']==3 and len(r['by_symbol'])==2 and len(r['by_side'])==2
