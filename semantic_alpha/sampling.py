from __future__ import annotations

import pandas as pd


def decision_points(
    df: pd.DataFrame,
    *,
    min_event_spacing_minutes: int = 3,
    background_spacing_minutes: int = 30,
    semantic_delta_threshold: float = 0.15,
    attention_delta_threshold: float = 0.75,
    include_background: bool = True,
) -> pd.DataFrame:
    """Reduce dense recorder snapshots to plausible independent decision points.

    Event rows are selected when fresh posts arrive or semantic/attention state changes
    materially. Quiet background rows are sparsely retained as negative/control states.
    Selection uses only contemporaneous features and timestamps, never future labels.
    """
    if df.empty:
        return df.copy()
    d=df.sort_values(["symbol","ts"]).copy()
    d["ts"]=pd.to_datetime(d.ts,utc=True)
    for c in ("posts_1m","semantic_shock_delta_5m","attention_z_delta_5m"):
        if c not in d: d[c]=0.0
    d["_event_trigger"]=(
        (d.posts_1m.fillna(0)>0)
        | (d.semantic_shock_delta_5m.fillna(0).abs()>=semantic_delta_threshold)
        | (d.attention_z_delta_5m.fillna(0).abs()>=attention_delta_threshold)
    )
    keep=[]; reasons=[]
    event_gap=pd.Timedelta(minutes=max(1,min_event_spacing_minutes))
    bg_gap=pd.Timedelta(minutes=max(min_event_spacing_minutes,background_spacing_minutes))
    for _,g in d.groupby("symbol",sort=False):
        last_event=None; last_bg=None
        for idx,row in g.iterrows():
            ts=row.ts
            if bool(row._event_trigger):
                if last_event is None or ts-last_event>=event_gap:
                    keep.append(idx); reasons.append("EVENT"); last_event=ts
            elif include_background and (last_bg is None or ts-last_bg>=bg_gap):
                keep.append(idx); reasons.append("BACKGROUND"); last_bg=ts
    out=d.loc[keep].copy()
    # loc preserves the order of keep; assign matching reasons then restore chronology.
    out["decision_reason"]=reasons
    out=out.drop(columns=["_event_trigger"]).sort_values("ts").reset_index(drop=True)
    return out


def sampling_report(raw: pd.DataFrame, sampled: pd.DataFrame) -> dict:
    if raw.empty:
        return {"raw_rows":0,"sampled_rows":0,"retention_rate":0.0}
    event_rows=int((sampled.get("decision_reason",pd.Series(dtype=str))=="EVENT").sum()) if not sampled.empty else 0
    bg_rows=int((sampled.get("decision_reason",pd.Series(dtype=str))=="BACKGROUND").sum()) if not sampled.empty else 0
    return {
        "raw_rows":int(len(raw)),"sampled_rows":int(len(sampled)),"retention_rate":float(len(sampled)/len(raw)),
        "event_rows":event_rows,"background_rows":bg_rows,"symbols":int(sampled.symbol.nunique()) if not sampled.empty and 'symbol' in sampled else 0,
    }
