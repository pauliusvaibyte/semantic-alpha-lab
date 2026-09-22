"""Anti-overfit statistics for strategy selection.

Two complementary controls on top of the locked holdout:

- Deflated Sharpe Ratio (Bailey & Lopez de Prado): the P(true Sharpe > 0)
  adjusted for the number of selection trials. Every run recorded in the
  experiment_trials ledger raises the bar.
- Combinatorial Purged Cross-Validation / PBO: C(G, G/2) contiguous-group
  splits; measure how often the in-sample best variant lands at or below the
  median out-of-sample variant. PBO near 0 = selection is informative;
  near 0.5 = the search was noise.
"""
from __future__ import annotations

import math
from itertools import combinations
from statistics import NormalDist

import numpy as np
import pandas as pd

from .backtest import dynamic_costs, trade_metrics
from .research import FEATURE_GROUPS, _fit_predict

_N = NormalDist()
_EULER_GAMMA = 0.5772156649015329

DEFAULT_VARIANTS = {
    "M1_market": FEATURE_GROUPS["price_derivatives"],
    "M2_attention": FEATURE_GROUPS["price_derivatives"] + FEATURE_GROUPS["attention"],
    "M3_semantics": FEATURE_GROUPS["price_derivatives"] + FEATURE_GROUPS["attention"] + FEATURE_GROUPS["semantics"],
    "M4_gap": FEATURE_GROUPS["price_derivatives"] + FEATURE_GROUPS["attention"] + FEATURE_GROUPS["semantics"] + FEATURE_GROUPS["gap"],
}


def deflated_sharpe_ratio(observed_sharpe: float, n_trials: int, n_obs: int, skew: float = 0.0, kurt: float = 3.0) -> float | None:
    """P(true SR > 0) given observed SR, sample size, and selection trials."""
    if n_obs < 10 or n_trials < 1:
        return None
    sr = float(observed_sharpe)
    var = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    if var <= 0:
        return None
    sr_std = math.sqrt(var / (n_obs - 1))
    if sr_std <= 0:
        return None
    n = max(1, int(n_trials))
    if n == 1:
        # No selection happened: the expected max Sharpe under the null is 0.
        expected_max_under_null = 0.0
    else:
        p1 = _N.inv_cdf(1.0 - 1.0 / n)
        p2 = _N.inv_cdf(1.0 - 1.0 / (n * math.e))
        expected_max_under_null = sr_std * ((1.0 - _EULER_GAMMA) * p1 + _EULER_GAMMA * p2)
    z = (sr - expected_max_under_null) / sr_std
    return float(_N.cdf(z))


def combinatorial_folds(n_rows: int, n_groups: int = 8, test_groups: int | None = None):
    """(train_idx, test_idx) for each C(n_groups, test_groups) contiguous split."""
    test_groups = test_groups or n_groups // 2
    bounds = np.linspace(0, n_rows, n_groups + 1).astype(int)
    groups = [np.arange(bounds[g], bounds[g + 1]) for g in range(n_groups)]
    for combo in combinations(range(n_groups), test_groups):
        test = np.concatenate([groups[g] for g in combo])
        train = np.concatenate([groups[g] for g in range(n_groups) if g not in combo])
        yield np.sort(train), np.sort(test)


def _purged_train_idx(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, horizon_minutes: int) -> np.ndarray:
    """Drop train rows whose [ts, ts+horizon] label window overlaps any test block."""
    h = pd.Timedelta(minutes=horizon_minutes)
    ts = df.ts
    keep = np.ones(len(train_idx), dtype=bool)
    order = np.sort(test_idx)
    runs = np.split(order, np.where(np.diff(order) > 1)[0] + 1)
    train_ts = ts.iloc[train_idx].to_numpy()
    for run in runs:
        lo = ts.iloc[run[0]] - h
        hi = ts.iloc[run[-1]]
        keep &= ~((train_ts >= lo) & (train_ts <= hi))
    return train_idx[keep]


def _sharpe_of(prob: np.ndarray, target: pd.Series, costs: np.ndarray) -> float | None:
    sig = np.where(prob >= 0.60, 1.0, np.where(prob <= 0.40, -1.0, 0.0))
    net = sig * target.to_numpy(dtype=float) - np.abs(sig) * costs
    m = trade_metrics(pd.Series(net[net != 0.0]))
    return m["sharpe"]


def cpcv_pbo_report(
    df: pd.DataFrame,
    variants: dict[str, list[str]] | None = None,
    horizon_minutes: int = 15,
    n_groups: int = 8,
    model_name: str = "logistic",
    notional_usd: float = 1000.0,
) -> dict:
    """CPCV across model/feature variants; reports PBO and per-variant OOS medians."""
    variants = variants or DEFAULT_VARIANTS
    names = list(variants)
    df = df.sort_values("ts").reset_index(drop=True)
    combos = list(combinatorial_folds(len(df), n_groups))
    is_m = np.full((len(combos), len(names)), np.nan)
    oos_m = np.full_like(is_m, np.nan)
    used = 0
    for ci, (tr, te) in enumerate(combos):
        tr = _purged_train_idx(df, tr, te, horizon_minutes)
        if len(tr) < 50 or len(te) < 10:
            continue
        train, test = df.iloc[tr], df.iloc[te]
        used += 1
        costs_te = dynamic_costs(test, notional_usd=notional_usd)
        costs_tr = dynamic_costs(train, notional_usd=notional_usd)
        for vi, v in enumerate(names):
            cols = [c for c in variants[v] if c in df.columns]
            if not cols:
                continue
            try:
                oos_m[ci, vi] = _sharpe_of(_fit_predict(train, test, cols, model_name), test.target_return, costs_te)
                is_m[ci, vi] = _sharpe_of(_fit_predict(train, train, cols, model_name), train.target_return, costs_tr)
            except Exception:
                continue
    valid = ~np.isnan(oos_m).all(axis=1)
    is_m, oos_m = is_m[valid], oos_m[valid]
    pbo = None
    if len(oos_m) >= 4 and len(names) >= 2:
        hits = 0
        for c in range(len(oos_m)):
            row_is, row_oos = is_m[c], oos_m[c]
            if np.isnan(row_is).all() or np.isnan(row_oos).all():
                continue
            hits += int(row_oos[int(np.nanargmax(row_is))] <= np.nanmedian(row_oos))
        pbo = hits / len(oos_m)
    med = {v: (float(np.nanmedian(oos_m[:, i])) if not np.isnan(oos_m[:, i]).all() else None) for i, v in enumerate(names)}
    return {
        "pbo": pbo,
        "combos_evaluated": int(used),
        "variants": names,
        "oos_sharpe_median": med,
        "note": "PBO = fraction of splits where the in-sample best variant was <= median OOS. ~0 good, ~0.5 means selection found noise.",
    }


def overfit_report(store, df: pd.DataFrame, *, horizon: str = "15m", horizon_minutes: int = 15,
                   model_name: str = "logistic", n_groups: int = 8, notional_usd: float = 1000.0) -> dict:
    rep = cpcv_pbo_report(df, horizon_minutes=horizon_minutes, n_groups=n_groups, model_name=model_name, notional_usd=notional_usd)
    trials = store.trial_count()
    best = None
    best_name = None
    for v, s in rep["oos_sharpe_median"].items():
        if s is not None and (best is None or s > best):
            best, best_name = s, v
    dsr = None
    if best is not None:
        rets = df.target_return.dropna()
        dsr = deflated_sharpe_ratio(best, n_trials=max(1, trials), n_obs=len(rets),
                                  skew=float(rets.skew()), kurt=float(rets.kurtosis() + 3.0))
    return {
        "horizon": horizon, "model": model_name, "effective_trials_recorded": trials,
        "best_variant_oos": best_name, "best_oos_sharpe_median": best,
        "deflated_sharpe_p": dsr, **rep,
    }
