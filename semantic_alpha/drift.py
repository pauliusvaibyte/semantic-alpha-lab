from __future__ import annotations

import numpy as np
import pandas as pd


def population_stability_index(reference: pd.Series, recent: pd.Series, bins: int = 10) -> float | None:
    a = reference.dropna().astype(float).to_numpy()
    b = recent.dropna().astype(float).to_numpy()
    if len(a) < 30 or len(b) < 20:
        return None
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        # A reference distribution collapsed to one value is itself informative:
        # moving away from it is maximal drift, not zero drift.
        return 0.0 if np.allclose(b, a[0]) else 10.0
    edges[0], edges[-1] = -np.inf, np.inf
    ah, _ = np.histogram(a, bins=edges)
    bh, _ = np.histogram(b, bins=edges)
    ap = np.maximum(ah / max(1, ah.sum()), 1e-6)
    bp = np.maximum(bh / max(1, bh.sum()), 1e-6)
    return float(np.sum((bp - ap) * np.log(bp / ap)))


def drift_report(df: pd.DataFrame, recent_fraction: float = 0.25) -> dict:
    if len(df) < 120:
        return {"ok": False, "reason": "need_at_least_120_rows", "rows": len(df)}
    d = df.sort_values("ts").reset_index(drop=True)
    cut = max(60, int(len(d) * (1 - recent_fraction)))
    ref, recent = d.iloc[:cut], d.iloc[cut:]
    ignore = {"ts", "symbol", "target_return", "market_regime"}
    rows = []
    for c in d.columns:
        if c in ignore or not pd.api.types.is_numeric_dtype(d[c]):
            continue
        psi = population_stability_index(ref[c], recent[c])
        if psi is None:
            continue
        rows.append({"feature": c, "psi": psi, "severity": "HIGH" if psi >= 0.25 else ("WATCH" if psi >= 0.10 else "OK")})
    rows.sort(key=lambda x: x["psi"], reverse=True)
    return {
        "ok": True, "reference_rows": len(ref), "recent_rows": len(recent),
        "high_drift_features": sum(x["severity"] == "HIGH" for x in rows),
        "watch_features": sum(x["severity"] == "WATCH" for x in rows),
        "features": rows,
    }
