from __future__ import annotations

import numpy as np
import pandas as pd

from .storage import Store
from .labels import HORIZONS


def _robust_z(s: pd.Series) -> pd.Series:
    med = s.median(); mad = (s - med).abs().median()
    scale = 1.4826 * mad
    if not scale or np.isnan(scale):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - med) / scale


def cross_sectional_ranks(store: Store, feature_version: str = "v4") -> pd.DataFrame:
    rows=[]
    for f in store.latest_features(feature_version):
        r={"symbol":f.symbol,"ts":f.ts}; r.update(f.values); rows.append(r)
    if not rows:
        return pd.DataFrame()
    d=pd.DataFrame(rows)
    for c in ["information_price_gap_proxy","attention_z_5m","narrative_novelty_15m","semantic_shock","crowding_proxy","panic_squeeze_proxy"]:
        if c not in d: d[c]=0.0
        d[c+"_z"]=_robust_z(d[c].astype(float))
    d["underreaction_rank_score"] = (
        .45*d["information_price_gap_proxy_z"] + .20*d["attention_z_5m_z"] +
        .20*d["narrative_novelty_15m_z"] + .15*d["semantic_shock_z"]
    )
    d["fade_rank_score"] = d["crowding_proxy_z"] - .25*d["narrative_novelty_15m_z"]
    d["squeeze_rank_score"] = d["panic_squeeze_proxy_z"]
    return d.sort_values("underreaction_rank_score", ascending=False).reset_index(drop=True)


def cross_sectional_backtest(df: pd.DataFrame, score: str = "information_price_gap_proxy", top_k: int = 1,
                             notional_usd: float = 1_000.0, taker_fee_bps: float = 5.5, horizon: str = "15m") -> tuple[pd.DataFrame, dict]:
    """Simple long-top / short-bottom cross-sectional research test by minute bucket."""
    from .backtest import estimated_execution_cost_bps, trade_metrics
    if df.empty or score not in df.columns:
        return pd.DataFrame(), trade_metrics(pd.Series(dtype=float))
    d=df.copy(); d['bucket']=pd.to_datetime(d['ts'],utc=True).dt.floor('min')
    trades=[]
    next_rebalance=None
    hold=pd.Timedelta(minutes=HORIZONS.get(horizon,15))
    for bucket,g in d.groupby('bucket'):
        if next_rebalance is not None and bucket < next_rebalance:
            continue
        g=g.dropna(subset=[score,'target_return'])
        if len(g)<max(3,top_k*2): continue
        next_rebalance=bucket+hold
        ordered=g.sort_values(score)
        picks=[]
        for _,r in ordered.head(top_k).iterrows(): picks.append((-1,r))
        for _,r in ordered.tail(top_k).iterrows(): picks.append((1,r))
        for side,r in picks:
            cost=estimated_execution_cost_bps(r,notional_usd,taker_fee_bps)/10_000
            gross=side*float(r.target_return); net=gross-cost
            trades.append({'ts':bucket,'symbol':r.symbol,'side':side,'score':float(r[score]),'gross_return':gross,'cost':cost,'net_return':net})
    t=pd.DataFrame(trades)
    return t, trade_metrics(t['net_return'] if not t.empty else pd.Series(dtype=float))
