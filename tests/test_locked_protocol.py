from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from semantic_alpha.protocol import expected_calibration_error, locked_split, evaluate_locked_holdout
from semantic_alpha.research import FEATURE_GROUPS


def _frame(n=420):
    rng=np.random.default_rng(4)
    ts=[datetime(2026,1,1,tzinfo=timezone.utc)+timedelta(minutes=i) for i in range(n)]
    symbols=np.array(["BTCUSDT","ETHUSDT","SOLUSDT","SUIUSDT"])
    signal=rng.normal(size=n)
    target=.0015*signal+rng.normal(scale=.003,size=n)
    d=pd.DataFrame({"symbol":symbols[np.arange(n)%4],"ts":ts,"target_return":target})
    all_features=sorted({x for xs in FEATURE_GROUPS.values() for x in xs})
    extras=pd.DataFrame({c:rng.normal(scale=.2,size=n) for c in all_features if c not in d.columns})
    d=pd.concat([d,extras],axis=1)
    # Make semantic block genuinely predictive so the protocol has something testable.
    d["semantic_shock"]=signal
    d["information_price_gap_proxy"]=signal*.8
    d["intent_mean_5m"]=signal*.7
    d["attention_z_5m"]=rng.normal(size=n)
    d["spread_bps"]=2.0
    d["depth_bid_10bps"]=100000.0
    d["depth_ask_10bps"]=100000.0
    return d


def test_locked_split_purges_overlap():
    d=_frame()
    discovery, holdout, start=locked_split(d,"15m",.2)
    assert len(discovery)>200 and len(holdout)>50
    assert pd.to_datetime(discovery.ts,utc=True).max() < start-pd.Timedelta(minutes=15)


def test_ece_bounds():
    y=np.array([0,0,1,1])
    assert 0 <= expected_calibration_error(y,np.array([.1,.2,.8,.9])) <= 1


def test_locked_holdout_runs_and_records_fingerprints():
    d=_frame()
    discovery, holdout, _=locked_split(d,"15m",.2)
    r,m=evaluate_locked_holdout(discovery,holdout,horizon="15m",model_name="logistic",notional_usd=1000,bootstrap_samples=30)
    assert r["status"] in {"PASS_LOCKED_HOLDOUT","FAIL_LOCKED_HOLDOUT"}
    assert len(r["discovery_fingerprint"])==64 and len(r["holdout_fingerprint"])==64
    assert "ece_full" in r["classification"]
    assert m.trained_rows>=100
