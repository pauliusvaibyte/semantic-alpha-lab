from __future__ import annotations

from datetime import datetime

import numpy as np

from .storage import Store


def _z(x: float, arr: list[float]) -> float:
    if len(arr) < 2:
        return 0.0
    sd=float(np.std(arr,ddof=1))
    return float((x-float(np.mean(arr)))/sd) if sd>1e-12 else 0.0


def enrich_cross_asset_at(
    store: Store,
    ts: datetime,
    *,
    symbols: list[str] | None = None,
    feature_version: str = "v4",
    tolerance_seconds: int = 120,
) -> int:
    symbols=symbols or store.symbols()
    feats=[]
    for sym in symbols:
        f=store.feature_at_or_before(sym,ts,feature_version,tolerance_seconds)
        if f is not None:
            feats.append(f)
    if len(feats)<2:
        return 0

    semantic=[float(f.values.get('semantic_shock',0.0)) for f in feats]
    attention=[float(f.values.get('attention_z_5m',0.0)) for f in feats]
    returns=[float(f.values.get('return_5m',0.0)) for f in feats]
    sem_mean=float(np.mean(semantic)); sem_std=float(np.std(semantic,ddof=1)) if len(semantic)>1 else 0.0
    att_mean=float(np.mean(attention)); att_std=float(np.std(attention,ddof=1)) if len(attention)>1 else 0.0
    ret_median=float(np.median(returns)); ret_std=float(np.std(returns,ddof=1)) if len(returns)>1 else 0.0
    market_breadth=float(np.mean(np.asarray(returns)>0))
    social_breadth=float(np.mean(np.asarray(semantic)>0))
    social_extreme=float(np.mean(np.abs(np.asarray(semantic))>2.0))

    lookup={f.symbol:f for f in feats}
    btc=lookup.get('BTCUSDT'); eth=lookup.get('ETHUSDT')
    n=0
    for f in feats:
        v=dict(f.values)
        v['cross_semantic_mean']=sem_mean
        v['cross_semantic_std']=sem_std
        v['cross_attention_mean']=att_mean
        v['cross_attention_std']=att_std
        v['relative_semantic_z']=_z(float(v.get('semantic_shock',0.0)),semantic)
        v['relative_attention_z']=_z(float(v.get('attention_z_5m',0.0)),attention)
        v['market_breadth_5m']=market_breadth
        v['return_dispersion_5m']=ret_std
        v['relative_return_5m']=float(v.get('return_5m',0.0))-ret_median
        v['social_breadth_positive']=social_breadth
        v['social_breadth_extreme']=social_extreme
        v['btc_return_5m']=float(btc.values.get('return_5m',0.0)) if btc else 0.0
        v['btc_semantic_shock']=float(btc.values.get('semantic_shock',0.0)) if btc else 0.0
        v['eth_return_5m']=float(eth.values.get('return_5m',0.0)) if eth else 0.0
        v['eth_semantic_shock']=float(eth.values.get('semantic_shock',0.0)) if eth else 0.0
        store.save_features(f.model_copy(update={'values':v}))
        n+=1
    return n
