from __future__ import annotations

import numpy as np
import pandas as pd

from .research import incremental_semantic_test
from .validation import leave_one_symbol_out, placebo_test, purged_walk_forward_ablation


def evidence_gate(
    df: pd.DataFrame,
    *,
    horizon: str = "15m",
    model_name: str = "histgb",
    notional_usd: float = 1_000.0,
    placebo_iterations: int = 30,
) -> dict:
    """Conservative research gate. PASS means 'worth forward paper testing', not profitable."""
    rows = len(df)
    symbols = int(df.symbol.nunique()) if rows and "symbol" in df else 0
    if rows < 500 or symbols < 3:
        return {"status": "INSUFFICIENT_DATA", "rows": rows, "symbols": symbols, "minimum_rows": 500, "minimum_symbols": 3}

    inc = incremental_semantic_test(df, model_name=model_name, notional_usd=notional_usd, bootstrap_samples=300, horizon=horizon)
    purged = purged_walk_forward_ablation(df, horizon=horizon, folds=4, min_train=max(150, rows // 3), model_name=model_name, notional_usd=notional_usd)
    loso = leave_one_symbol_out(df, model_name=model_name, notional_usd=notional_usd, horizon=horizon)
    placebo = placebo_test(df, iterations=placebo_iterations, model_name="logistic")

    m4 = purged[purged.stage == "M4_plus_gap"] if not purged.empty else pd.DataFrame()
    m2 = purged[purged.stage == "M2_plus_attention"] if not purged.empty else pd.DataFrame()
    paired = []
    if not m4.empty and not m2.empty:
        z = m4.merge(m2, on="fold", suffixes=("_m4", "_m2"))
        paired = (z.mean_net_return_m4 - z.mean_net_return_m2).dropna().tolist()

    symbol_positive = 0.0
    if not loso.empty and "mean_net_return" in loso:
        vals = loso.mean_net_return.dropna()
        symbol_positive = float((vals > 0).mean()) if len(vals) else 0.0

    active = df[df.get("posts_15m", 0) > 0] if "posts_15m" in df.columns else pd.DataFrame()
    forward_rate = float(active["forward_observation_rate_15m"].mean()) if len(active) and "forward_observation_rate_15m" in active.columns else None
    semantic_live_rate = float(active["semantic_live_rate_15m"].mean()) if len(active) and "semantic_live_rate_15m" in active.columns else None
    forward_observed = bool(forward_rate is not None and semantic_live_rate is not None and forward_rate >= 0.80 and semantic_live_rate >= 0.80)

    checks = {
        "semantic_auc_ci_positive": bool(inc.get("semantic_auc_improvement_supported", False)),
        "semantic_net_ci_positive": bool(inc.get("semantic_net_improvement_supported", False)),
        "purged_increment_positive_majority": bool(paired and np.mean(np.array(paired) > 0) >= 0.60),
        "placebo_p_below_10pct": bool(placebo.get("ok") and placebo.get("p_value", 1.0) < 0.10),
        "cross_asset_positive_majority": bool(len(loso) >= 2 and symbol_positive >= 0.60),
    }
    # Require net improvement plus at least two independent robustness checks.
    robustness = sum(checks[k] for k in ["purged_increment_positive_majority", "placebo_p_below_10pct", "cross_asset_positive_majority"])
    promising = checks["semantic_net_ci_positive"] and robustness >= 2
    # Terminology: FORWARD_OBSERVED = the feature rows were produced by live
    # (non-backfilled) collection. FORWARD_VALIDATED is reserved for evidence
    # observed strictly after model freeze (the paper-trading record itself).
    if promising and forward_observed:
        status="PROMISING_FORWARD_OBSERVED"
    elif promising:
        status="PROMISING_HISTORICAL_ONLY"
    else:
        status="REJECT_OR_KEEP_RESEARCHING"
    return {
        "status": status,
        "rows": rows, "symbols": symbols, "checks": checks,
        "provenance":{"forward_observation_rate":forward_rate,"semantic_live_rate":semantic_live_rate,"forward_observed":forward_observed},
        "semantic_increment": inc,
        "purged_walk_forward": purged.to_dict(orient="records"),
        "leave_one_symbol_out": loso.to_dict(orient="records"),
        "placebo": placebo,
        "note": "PROMISING is only a gate to forward paper trading; it is not evidence of future profitability.",
    }
