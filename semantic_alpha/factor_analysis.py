from __future__ import annotations

import math
import numpy as np
import pandas as pd


def cross_sectional_factor_ic(
    df: pd.DataFrame,
    factor: str,
    *,
    bucket: str = "5min",
    min_assets: int = 3,
) -> dict:
    """Cross-sectional Spearman IC and top-minus-bottom abnormal-return spread."""
    if df.empty or factor not in df.columns or "target_return" not in df.columns:
        return {"factor": factor, "ok": False, "reason": "missing_data"}
    d=df[["ts","symbol",factor,"target_return"]].dropna().copy()
    if d.empty:
        return {"factor": factor, "ok": False, "reason": "no_complete_rows"}
    d["bucket_ts"]=pd.to_datetime(d.ts,utc=True).dt.floor(bucket)
    ics=[]; spreads=[]; assets=[]
    for _,g in d.groupby("bucket_ts"):
        # Keep one observation per asset per bucket to prevent fast recorders from overweighting a coin.
        g=g.sort_values("ts").groupby("symbol",as_index=False).last()
        if len(g)<min_assets or g[factor].nunique()<2 or g.target_return.nunique()<2:
            continue
        rx=g[factor].rank(method="average")
        ry=g.target_return.rank(method="average")
        ic=float(rx.corr(ry))
        if np.isfinite(ic):
            ics.append(ic); assets.append(len(g))
        q=max(1,len(g)//4)
        order=g.sort_values(factor)
        spreads.append(float(order.tail(q).target_return.mean()-order.head(q).target_return.mean()))
    if not ics:
        return {"factor":factor,"ok":False,"reason":"insufficient_cross_sections"}
    a=np.asarray(ics,dtype=float); sp=np.asarray(spreads,dtype=float)
    sd=float(a.std(ddof=1)) if len(a)>1 else 0.0
    ir=float(a.mean()/sd*math.sqrt(len(a))) if sd>0 else None
    # Normal approximation for a quick diagnostic; event/holdout tests remain the stronger evidence.
    tstat=float(a.mean()/(sd/math.sqrt(len(a)))) if sd>0 else None
    return {
        "factor":factor,"ok":True,"cross_sections":len(a),"mean_assets":float(np.mean(assets)),
        "mean_ic":float(a.mean()),"median_ic":float(np.median(a)),"ic_positive_rate":float(np.mean(a>0)),
        "ic_information_ratio":ir,"ic_tstat_approx":tstat,
        "mean_top_bottom_return_spread":float(sp.mean()) if len(sp) else None,
        "spread_positive_rate":float(np.mean(sp>0)) if len(sp) else None,
    }


def factor_ic_report(
    df: pd.DataFrame,
    factors: list[str] | None = None,
    *,
    bucket: str = "5min",
    min_assets: int = 3,
) -> pd.DataFrame:
    default=[
        "semantic_shock","information_price_gap_proxy","attention_z_5m","intent_sum_5m",
        "new_info_rate_5m","narrative_novelty_15m","diffusion_velocity_15m",
        "author_alpha_signal_5m","crowding_proxy","panic_squeeze_proxy",
        "relative_semantic_z","relative_attention_z",
    ]
    rows=[cross_sectional_factor_ic(df,f,bucket=bucket,min_assets=min_assets) for f in (factors or default) if f in df.columns]
    out=pd.DataFrame([x for x in rows if x.get("ok")])
    if out.empty:
        return out
    return out.sort_values(["mean_ic","mean_top_bottom_return_spread"],ascending=False).reset_index(drop=True)


def factor_monotonicity(
    df: pd.DataFrame,
    factor: str,
    *,
    bins: int = 5,
) -> pd.DataFrame:
    """Global quantile return curve for interpretability; not used as evidence alone."""
    if factor not in df or df.empty:
        return pd.DataFrame()
    d=df[[factor,"target_return"]].dropna().copy()
    if len(d)<max(50,bins*10) or d[factor].nunique()<bins:
        return pd.DataFrame()
    try:
        d["quantile"]=pd.qcut(d[factor],q=bins,labels=False,duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    rows=[]
    for q,g in d.groupby("quantile"):
        rows.append({"quantile":int(q),"rows":len(g),"factor_mean":float(g[factor].mean()),"mean_target_return":float(g.target_return.mean()),"positive_rate":float((g.target_return>0).mean())})
    return pd.DataFrame(rows)


PROB_HEADS = [
    "new_information_prob", "reactive_to_price_prob", "original_information_prob",
    "explicit_recommendation_prob", "evidence_prob", "promotional_prob",
    "coordinated_shill_prob", "market_impact_score", "confidence",
]


def semantic_reliability(store, horizon: str = "15m", bins: int = 5) -> pd.DataFrame:
    """Binned reliability of semantic probability heads vs realized |abnormal return|.

    Jev outputs are scores, not guaranteed calibrated probabilities — this shows
    whether each head's value actually orders realized magnitude/direction on our
    own data instead of trusting the nominal scale.
    """
    rows = []
    for sym in store.symbols():
        for p in store.all_posts(sym):
            label = store.label_for(sym, p.created_at)
            if not label:
                continue
            abn = label.abnormal_returns.get(horizon)
            if abn is None:
                continue
            for s in store.semantics_for_post(p.post_id, symbol=sym):
                rec = {"symbol": sym, "abs_abn": abs(abn), "abn": abn}
                for h in PROB_HEADS:
                    rec[h] = getattr(s, h, 0.0)
                rows.append(rec)
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    out = []
    for h in PROB_HEADS:
        dd = d[[h, "abs_abn", "abn"]].dropna()
        if len(dd) < 30 or dd[h].nunique() < bins:
            continue
        try:
            dd["q"] = pd.qcut(dd[h], q=bins, labels=False, duplicates="drop")
        except ValueError:
            continue
        means = dd.groupby("q")["abs_abn"].mean()
        ic = float(dd[h].rank().corr(dd["abs_abn"].rank()))
        out.append({
            "head": h, "rows": len(dd),
            "spearman_vs_abs_move": ic,
            "bottom_bin_abs_abn": float(means.iloc[0]),
            "top_bin_abs_abn": float(means.iloc[-1]),
            "monotone": bool(means.is_monotonic_increasing),
        })
    return pd.DataFrame(out).sort_values("spearman_vs_abs_move", ascending=False).reset_index(drop=True) if out else pd.DataFrame()
