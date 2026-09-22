from semantic_alpha.registry import promotion_decision


def test_champion_requires_locked_holdout_now():
    report={"evidence_gate":{"status":"PROMISING_FORWARD_VALIDATED"},"leakage":{"ok":True},"drift":{"ok":True,"high_drift_features":0}}
    paper={"closed_trades":120,"net_return":.1,"profit_factor":1.4,"max_drawdown":-.04,"top5_profit_share":.3}
    r=promotion_decision(report,paper)
    assert r['status']=='KEEP_AS_CANDIDATE'
    assert not r['checks']['locked_holdout_pass']
