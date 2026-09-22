from __future__ import annotations

import pandas as pd

from .research import dataset, incremental_semantic_test
from .storage import Store
from .validation import purged_walk_forward_ablation


def compare_feature_versions(
    store: Store,
    versions: list[str],
    *,
    horizon: str = "15m",
    model_name: str = "logistic",
    notional_usd: float = 1_000.0,
) -> pd.DataFrame:
    """Compare semantic engines/feature builds on exactly the same labeled observations.

    Each semantic engine must have been materialized into a distinct feature_version.
    We intersect (symbol, ts) before evaluation so one engine cannot win merely because
    it happened to have easier/more observations.
    """
    frames: dict[str, pd.DataFrame] = {}
    keysets: list[set[tuple[str, str]]] = []
    for v in versions:
        d = dataset(store, horizon=horizon, feature_version=v)
        if d.empty:
            continue
        d = d.copy()
        d["_key"] = list(zip(d.symbol.astype(str), pd.to_datetime(d.ts, utc=True).astype(str), strict=True))
        frames[v] = d
        keysets.append(set(d._key.tolist()))
    if len(frames) < 2:
        return pd.DataFrame()
    common = set.intersection(*keysets) if keysets else set()
    if len(common) < 100:
        return pd.DataFrame()
    rows = []
    for v, d in frames.items():
        x = d[d._key.isin(common)].drop(columns=["_key"]).sort_values("ts").reset_index(drop=True)
        inc = incremental_semantic_test(x, model_name=model_name, notional_usd=notional_usd, bootstrap_samples=200, horizon=horizon)
        wf = purged_walk_forward_ablation(x, horizon=horizon, folds=4, min_train=max(100, len(x)//3), model_name=model_name, notional_usd=notional_usd)
        m4 = wf[wf.stage == "M4_plus_gap"] if not wf.empty else pd.DataFrame()
        rows.append({
            "feature_version": v,
            "common_rows": len(x),
            "auc_delta": inc.get("auc_delta") if inc.get("ok") else None,
            "net_delta": inc.get("mean_net_delta") if inc.get("ok") else None,
            "auc_supported": inc.get("semantic_auc_improvement_supported") if inc.get("ok") else False,
            "net_supported": inc.get("semantic_net_improvement_supported") if inc.get("ok") else False,
            "m4_positive_fold_rate": float((m4.mean_net_return > 0).mean()) if not m4.empty else None,
            "m4_mean_net_return": float(m4.mean_net_return.mean()) if not m4.empty else None,
            "m4_mean_auc": float(m4.auc.mean()) if not m4.empty else None,
        })
    return pd.DataFrame(rows).sort_values(["net_supported", "m4_mean_net_return", "auc_delta"], ascending=False).reset_index(drop=True)
