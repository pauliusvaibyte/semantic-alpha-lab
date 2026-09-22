from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .backtest import dynamic_costs, stateful_trade_vectors, trade_metrics
from .labels import HORIZONS
from .research import FEATURE_GROUPS, _fit_predict, _evaluate_split


def horizon_minutes(horizon: str) -> int:
    if horizon not in HORIZONS:
        raise ValueError(f"unknown horizon {horizon}; expected one of {sorted(HORIZONS)}")
    return HORIZONS[horizon]


def market_regime(row: pd.Series | dict) -> str:
    funding = float(row.get("funding_rate", 0.0) or 0.0)
    oi = float(row.get("oi_change_30m", 0.0) or 0.0)
    ret = float(row.get("return_30m", 0.0) or 0.0)
    flow = float(row.get("signed_trade_notional_1m", 0.0) or 0.0)
    if funding > 0.0002 and oi > 0.01:
        return "CROWDED_LONG"
    if funding < -0.0002 and oi > 0.01:
        return "CROWDED_SHORT"
    if ret >= 0.01:
        return "VOLATILE_UP"
    if ret <= -0.01:
        return "VOLATILE_DOWN"
    if ret >= 0.003 or flow > 0:
        return "FLOW_UP"
    if ret <= -0.003 or flow < 0:
        return "FLOW_DOWN"
    return "QUIET"


def add_regime_column(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["market_regime"] = [market_regime(r) for _, r in d.iterrows()]
    return d


def _stage_features(stage: str, df: pd.DataFrame) -> list[str]:
    groups = {
        "M1_price_derivatives": ["price_derivatives"],
        "M2_plus_attention": ["price_derivatives", "attention"],
        "M3_plus_semantics": ["price_derivatives", "attention", "semantics"],
        "M4_plus_gap": ["price_derivatives", "attention", "semantics", "gap"],
    }[stage]
    return [x for g in groups for x in FEATURE_GROUPS[g] if x in df.columns]


def purged_walk_forward_ablation(
    df: pd.DataFrame,
    *,
    horizon: str = "15m",
    folds: int = 4,
    min_train: int = 150,
    model_name: str = "histgb",
    cost_bps: float = 10.0,
    dynamic_cost: bool = True,
    notional_usd: float = 1_000.0,
) -> pd.DataFrame:
    """Expanding walk-forward with label-overlap purging.

    Any training row whose outcome horizon reaches the test start is removed.
    This is essential when feature snapshots are frequent and labels overlap.
    """
    if df.empty:
        return pd.DataFrame()
    d = df.sort_values("ts").reset_index(drop=True).copy()
    d["ts"] = pd.to_datetime(d.ts, utc=True)
    if len(d) < min_train + folds * 15:
        return pd.DataFrame()
    test_size = max(15, (len(d) - min_train) // folds)
    purge = timedelta(minutes=horizon_minutes(horizon))
    rows: list[dict] = []
    stages = ["M1_price_derivatives", "M2_plus_attention", "M3_plus_semantics", "M4_plus_gap"]
    for fold in range(folds):
        test_i = min_train + fold * test_size
        test_end = min(len(d), test_i + test_size)
        if test_end <= test_i:
            continue
        test = d.iloc[test_i:test_end].copy()
        test_start = test.ts.min()
        train = d[d.ts < test_start - purge].copy()
        for stage in stages:
            feats = _stage_features(stage, d)
            tr = train.dropna(subset=[*feats, "target_return"])
            te = test.dropna(subset=[*feats, "target_return"])
            if len(tr) < 50 or len(te) < 10:
                continue
            r = _evaluate_split(
                tr, te, feats, model_name=model_name, cost_bps=cost_bps,
                dynamic_cost=dynamic_cost, notional_usd=notional_usd,
                horizon_minutes=horizon_minutes(horizon),
            )
            rows.append({
                "fold": fold, "stage": stage, "horizon": horizon,
                "train_rows": len(tr), "test_rows": len(te),
                "test_start": test_start, "purge_minutes": horizon_minutes(horizon),
                **{k: v for k, v in asdict(r).items() if k != "rows"},
            })
    return pd.DataFrame(rows)


def leave_one_symbol_out(
    df: pd.DataFrame,
    *,
    model_name: str = "histgb",
    notional_usd: float = 1_000.0,
    horizon: str = "15m",
) -> pd.DataFrame:
    if df.empty or "symbol" not in df:
        return pd.DataFrame()
    feats = _stage_features("M4_plus_gap", df)
    rows = []
    for sym in sorted(df.symbol.unique()):
        tr = df[df.symbol != sym].dropna(subset=[*feats, "target_return"])
        te = df[df.symbol == sym].dropna(subset=[*feats, "target_return"])
        if len(tr) < 100 or len(te) < 20:
            continue
        r = _evaluate_split(tr, te, feats, model_name=model_name, cost_bps=10.0, dynamic_cost=True, notional_usd=notional_usd, horizon_minutes=horizon_minutes(horizon))
        rows.append({"heldout_symbol": sym, "train_rows": len(tr), "test_rows": len(te), **{k: v for k, v in asdict(r).items() if k != "rows"}})
    return pd.DataFrame(rows)


def heldout_regime_report(
    df: pd.DataFrame,
    *,
    model_name: str = "histgb",
    notional_usd: float = 1_000.0,
    horizon: str = "15m",
) -> pd.DataFrame:
    feats = _stage_features("M4_plus_gap", df)
    d = add_regime_column(df.dropna(subset=[*feats, "target_return"]).sort_values("ts").reset_index(drop=True))
    if len(d) < 100:
        return pd.DataFrame()
    split = int(len(d) * 0.7)
    tr, te = d.iloc[:split], d.iloc[split:].copy()
    if (tr.target_return > 0).nunique() < 2:
        return pd.DataFrame()
    p = _fit_predict(tr, te, feats, model_name)
    sig = np.where(p >= 0.60, 1, np.where(p <= 0.40, -1, 0))
    costs = dynamic_costs(te, notional_usd=notional_usd)
    net, accepted = stateful_trade_vectors(te, sig, costs, horizon_minutes=horizon_minutes(horizon), max_positions=3)
    te["signal"] = np.where(accepted, sig, 0)
    te["net"] = net
    rows = []
    for regime, g in te.groupby("market_regime"):
        active = g[g.signal != 0]
        met = trade_metrics(active.net) if len(active) else trade_metrics(pd.Series(dtype=float))
        rows.append({"regime": regime, "rows": len(g), "trades": len(active), "hit_rate": float((active.signal * active.target_return > 0).mean()) if len(active) else None, **met})
    return pd.DataFrame(rows)


def placebo_test(
    df: pd.DataFrame,
    *,
    iterations: int = 40,
    model_name: str = "logistic",
    seed: int = 17,
) -> dict:
    """Circular-shift null test per symbol to preserve rough target autocorrelation."""
    feats = _stage_features("M4_plus_gap", df)
    d = df.dropna(subset=[*feats, "target_return"]).sort_values("ts").reset_index(drop=True)
    if len(d) < 150:
        return {"ok": False, "reason": "need_at_least_150_rows", "rows": len(d)}
    split = int(len(d) * 0.7)
    tr, te = d.iloc[:split].copy(), d.iloc[split:].copy()
    if (tr.target_return > 0).nunique() < 2 or (te.target_return > 0).nunique() < 2:
        return {"ok": False, "reason": "single_class_split"}
    p = _fit_predict(tr, te, feats, model_name)
    actual_auc = float(roc_auc_score((te.target_return > 0).astype(int), p))
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(iterations):
        shuffled = tr.copy()
        pieces = []
        for _, g in shuffled.groupby("symbol", sort=False):
            g = g.copy()
            n = len(g)
            if n > 4:
                shift = int(rng.integers(max(1, n // 5), max(2, n - 1)))
                g["target_return"] = np.roll(g.target_return.to_numpy(), shift)
            pieces.append(g)
        sh = pd.concat(pieces).sort_index()
        if (sh.target_return > 0).nunique() < 2:
            continue
        pp = _fit_predict(sh, te, feats, model_name)
        nulls.append(float(roc_auc_score((te.target_return > 0).astype(int), pp)))
    if not nulls:
        return {"ok": False, "reason": "no_valid_null_iterations"}
    pvalue = float((1 + sum(x >= actual_auc for x in nulls)) / (len(nulls) + 1))
    return {
        "ok": True, "actual_auc": actual_auc, "null_mean_auc": float(np.mean(nulls)),
        "null_p95_auc": float(np.quantile(nulls, 0.95)), "p_value": pvalue,
        "iterations": len(nulls),
    }


def semantic_staleness_test(
    df: pd.DataFrame,
    *,
    horizon: str = "15m",
    model_name: str = "histgb",
    shifts: tuple[int, ...] = (1, 4, 12),
) -> pd.DataFrame:
    """Evaluate whether semantic value decays when only older social features are supplied.

    This is a timing sanity check, not a causal proof. If semantics allegedly create
    short-horizon alpha but 12-snapshot stale semantics perform identically, either the
    feature is really a slow regime variable or the timing thesis is suspect.
    """
    baseline = _stage_features("M2_plus_attention", df)
    full = _stage_features("M4_plus_gap", df)
    semantic_cols = [c for c in full if c not in baseline]
    d = df.dropna(subset=[*full, "target_return"]).sort_values("ts").reset_index(drop=True).copy()
    if len(d) < 150 or not semantic_cols:
        return pd.DataFrame()
    split = int(len(d) * .70)
    te = d.iloc[split:].copy()
    start = pd.to_datetime(te.ts, utc=True).min()
    purge = pd.Timedelta(minutes=horizon_minutes(horizon))
    tr = d[pd.to_datetime(d.ts, utc=True) < start - purge].copy()
    if len(tr) < 80 or len(te) < 30 or (tr.target_return > 0).nunique() < 2 or (te.target_return > 0).nunique() < 2:
        return pd.DataFrame()
    from .models import _make_model
    model = _make_model(model_name)
    model.fit(tr[full].fillna(0.0), (tr.target_return > 0).astype(int))
    y = (te.target_return > 0).astype(int)
    actual = model.predict_proba(te[full].fillna(0.0))[:, 1]
    actual_auc = float(roc_auc_score(y, actual))
    rows = [{"stale_rows": 0, "auc": actual_auc, "auc_delta_vs_fresh": 0.0, "mean_abs_probability_change": 0.0}]
    for k in shifts:
        stale = te.copy()
        shifted = stale.groupby("symbol", sort=False)[semantic_cols].shift(int(k)).fillna(0.0)
        stale.loc[:, semantic_cols] = shifted
        p = model.predict_proba(stale[full].fillna(0.0))[:, 1]
        rows.append({
            "stale_rows": int(k),
            "auc": float(roc_auc_score(y, p)),
            "auc_delta_vs_fresh": float(roc_auc_score(y, p) - actual_auc),
            "mean_abs_probability_change": float(np.mean(np.abs(p - actual))),
        })
    # Estimate typical snapshot cadence for interpretation.
    ts = pd.to_datetime(te.ts, utc=True).sort_values()
    med = ts.diff().dropna().dt.total_seconds().median() if len(ts) > 1 else np.nan
    for r in rows:
        r["approx_stale_minutes"] = float(r["stale_rows"] * med / 60.0) if pd.notna(med) else None
    return pd.DataFrame(rows)


def semantic_permutation_placebo(
    df: pd.DataFrame,
    *,
    horizon: str = "15m",
    model_name: str = "logistic",
    iterations: int = 30,
    seed: int = 19,
) -> dict:
    """Null test that destroys semantic alignment in training but preserves market features.

    The held-out period is fixed. Only semantic/gap columns are circularly shifted
    independently within each asset in the training set. True aligned semantics should
    beat this null distribution if their timing/content adds information.
    """
    baseline = _stage_features("M2_plus_attention", df)
    full = _stage_features("M4_plus_gap", df)
    semantic_cols = [c for c in full if c not in baseline]
    d = df.dropna(subset=[*full, "target_return"]).sort_values("ts").reset_index(drop=True).copy()
    if len(d) < 180 or not semantic_cols:
        return {"ok": False, "reason": "insufficient_rows_or_semantic_features", "rows": len(d)}
    split = int(len(d) * .70)
    te = d.iloc[split:].copy()
    start = pd.to_datetime(te.ts, utc=True).min()
    purge = pd.Timedelta(minutes=horizon_minutes(horizon))
    tr = d[pd.to_datetime(d.ts, utc=True) < start - purge].copy()
    if len(tr) < 100 or (tr.target_return > 0).nunique() < 2 or (te.target_return > 0).nunique() < 2:
        return {"ok": False, "reason": "invalid_split"}
    from .models import _make_model
    ytr = (tr.target_return > 0).astype(int)
    yte = (te.target_return > 0).astype(int)
    actual_model = _make_model(model_name)
    actual_model.fit(tr[full].fillna(0.0), ytr)
    actual_auc = float(roc_auc_score(yte, actual_model.predict_proba(te[full].fillna(0.0))[:, 1]))
    rng = np.random.default_rng(seed)
    nulls: list[float] = []
    for _ in range(iterations):
        sh = tr.copy()
        for _, idx in sh.groupby("symbol", sort=False).groups.items():
            idx = np.asarray(list(idx), dtype=int)
            n = len(idx)
            if n < 4:
                continue
            offset = int(rng.integers(max(1, n // 5), max(2, n - 1)))
            vals = sh.loc[idx, semantic_cols].to_numpy()
            sh.loc[idx, semantic_cols] = np.roll(vals, offset, axis=0)
        m = _make_model(model_name)
        m.fit(sh[full].fillna(0.0), ytr)
        nulls.append(float(roc_auc_score(yte, m.predict_proba(te[full].fillna(0.0))[:, 1])))
    if not nulls:
        return {"ok": False, "reason": "no_null_iterations"}
    pvalue = float((1 + sum(x >= actual_auc for x in nulls)) / (len(nulls) + 1))
    return {
        "ok": True,
        "actual_auc": actual_auc,
        "null_mean_auc": float(np.mean(nulls)),
        "null_p95_auc": float(np.quantile(nulls, .95)),
        "p_value": pvalue,
        "iterations": len(nulls),
        "semantic_columns": len(semantic_cols),
    }
