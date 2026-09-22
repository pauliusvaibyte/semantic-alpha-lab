from semantic_alpha.registry import promotion_decision


def _report(status="PROMISING_FORWARD_OBSERVED"):
    return {
        "evidence_gate":{"status":status},
        "locked_holdout":{"status":"PASS_LOCKED_HOLDOUT"},
        "leakage":{"ok":True},
        "drift":{"ok":True,"high_drift_features":0},
    }


def _paper(n=120):
    return {
        "closed_trades":n,"net_return":.12,"profit_factor":1.4,"max_drawdown":-.05,
        "top5_profit_share":.30,
    }


def test_promotion_requires_forward_evidence_and_paper_record():
    good=promotion_decision(_report(),_paper())
    assert good["status"] == "ELIGIBLE_FOR_CHAMPION"
    validated=promotion_decision(_report("PROMISING_FORWARD_VALIDATED"),_paper())
    assert validated["status"] == "ELIGIBLE_FOR_CHAMPION"
    bad=promotion_decision(_report("PROMISING_HISTORICAL_ONLY"),_paper())
    assert bad["status"] == "KEEP_AS_CANDIDATE"
    assert not bad["checks"]["forward_evidence_gate"]
    sparse=promotion_decision(_report(),_paper(20))
    assert not sparse["checks"]["paper_trade_count"]
