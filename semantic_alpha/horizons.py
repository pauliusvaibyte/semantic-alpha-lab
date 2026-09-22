from __future__ import annotations

import pandas as pd

from .labels import HORIZONS
from .research import dataset, incremental_semantic_test
from .storage import Store


def horizon_scan(
    store: Store,
    *,
    model_name: str = "logistic",
    min_rows: int = 100,
    notional_usd: float = 1_000.0,
    bootstrap_samples: int = 150,
) -> pd.DataFrame:
    rows=[]
    for horizon in HORIZONS:
        d=dataset(store,horizon=horizon)
        if len(d)<min_rows:
            rows.append({"horizon":horizon,"minutes":HORIZONS[horizon],"rows":len(d),"ok":False})
            continue
        r=incremental_semantic_test(
            d,model_name=model_name,notional_usd=notional_usd,
            bootstrap_samples=bootstrap_samples, horizon=horizon,
        )
        rows.append({
            "horizon":horizon,"minutes":HORIZONS[horizon],"rows":len(d),"ok":bool(r.get("ok")),
            "auc_m2":r.get("auc_m2"),"auc_m4":r.get("auc_m4"),"auc_delta":r.get("auc_delta"),
            "auc_ci_low": (r.get("auc_delta_ci95") or [None,None])[0],
            "auc_ci_high": (r.get("auc_delta_ci95") or [None,None])[1],
            "mean_net_m2":r.get("mean_net_all_rows_m2"),"mean_net_m4":r.get("mean_net_all_rows_m4"),
            "mean_net_delta":r.get("mean_net_delta"),
            "net_ci_low": (r.get("mean_net_delta_ci95") or [None,None])[0],
            "net_ci_high": (r.get("mean_net_delta_ci95") or [None,None])[1],
            "semantic_auc_supported":r.get("semantic_auc_improvement_supported"),
            "semantic_net_supported":r.get("semantic_net_improvement_supported"),
        })
    return pd.DataFrame(rows).sort_values("minutes").reset_index(drop=True)
