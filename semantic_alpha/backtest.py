from __future__ import annotations

import math
import numpy as np
import pandas as pd


def estimated_execution_cost_bps(
    row: pd.Series | dict,
    notional_usd: float = 1_000.0,
    taker_fee_bps: float = 5.5,
    fallback_slippage_bps: float = 3.0,
    include_spread: bool = True,
) -> float:
    """Conservative one-way execution-cost approximation.

    Cost = taker fee + half spread + depth-sensitive impact. This is not a fill
    simulator; it exists to stop tiny apparent edges from surviving on a flat,
    unrealistically cheap cost assumption.
    """
    spread = max(0.0, float(row.get("spread_bps", 0.0) or 0.0))
    bid_depth = float(row.get("depth_bid_10bps", 0.0) or 0.0)
    ask_depth = float(row.get("depth_ask_10bps", 0.0) or 0.0)
    depth = max(1.0, min(x for x in (bid_depth, ask_depth) if x > 0)) if (bid_depth > 0 and ask_depth > 0) else 0.0
    if depth > 0:
        participation = max(0.0, notional_usd / depth)
        impact = min(75.0, 10.0 * math.sqrt(participation))
    else:
        impact = fallback_slippage_bps
    return taker_fee_bps + (spread / 2.0 if include_spread else 0.0) + impact


def probabilistic_sharpe_ratio(returns: pd.Series, benchmark_sharpe: float = 0.0) -> float | None:
    """Approximate probability that the per-observation Sharpe exceeds a benchmark.

    Uses the Bailey/Lopez de Prado PSR approximation with sample skew/kurtosis.
    It is a diagnostic, not a guarantee and not a substitute for an untouched holdout.
    """
    r=returns.dropna().astype(float)
    n=len(r)
    if n < 10:
        return None
    sd=float(r.std(ddof=1))
    if sd <= 0:
        return None
    sr=float(r.mean()/sd)
    skew=float(r.skew())
    kurt=float(r.kurtosis()+3.0)  # pandas returns excess kurtosis
    denom=1.0 - skew*sr + ((kurt-1.0)/4.0)*(sr**2)
    if denom <= 0:
        return None
    z=(sr-benchmark_sharpe)*math.sqrt(n-1)/math.sqrt(denom)
    return float(0.5*(1.0+math.erf(z/math.sqrt(2.0))))


def dynamic_costs(df: pd.DataFrame, notional_usd: float = 1_000.0, taker_fee_bps: float = 5.5) -> np.ndarray:
    return np.array([estimated_execution_cost_bps(r, notional_usd, taker_fee_bps) / 10_000 for _, r in df.iterrows()])


def trade_metrics(returns: pd.Series, periods_per_year: int = 365 * 24 * 4) -> dict[str, float | int | None]:
    r = returns.dropna().astype(float)
    if r.empty:
        return {"trades":0, "net_return":0.0, "sharpe":None, "probabilistic_sharpe_gt_zero":None, "max_drawdown":None, "profit_factor":None, "median_return":None, "largest_trade_return":None, "top5_profit_share":None}
    equity = (1 + r).cumprod()
    dd = equity / equity.cummax() - 1
    sd = r.std(ddof=1)
    sharpe = (r.mean() / sd * np.sqrt(periods_per_year)) if sd and sd > 0 else None
    pos = r[r > 0]
    gains = pos.sum(); losses = -r[r < 0].sum()
    top5 = pos.nlargest(5).sum() if len(pos) else 0.0
    return {
        "trades": int((r != 0).sum()), "net_return": float(equity.iloc[-1] - 1),
        "sharpe": float(sharpe) if sharpe is not None else None,
        "probabilistic_sharpe_gt_zero": probabilistic_sharpe_ratio(r, 0.0),
        "max_drawdown": float(dd.min()),
        "profit_factor": float(gains / losses) if losses > 0 else None,
        "median_return": float(r.median()),
        "largest_trade_return": float(r.abs().max()),
        "top5_profit_share": float(top5 / gains) if gains > 0 else None,
    }


def stateful_trade_vectors(
    df: pd.DataFrame,
    signals: np.ndarray,
    costs: np.ndarray,
    *,
    horizon_minutes: int = 15,
    max_positions: int = 3,
    exit_costs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply signals with realistic non-overlap constraints and round-trip costs.

    Returns `(net_return_vector, accepted_mask)` aligned to `df`. A symbol cannot
    open a new position until its prior horizon expires, and the portfolio cannot
    exceed `max_positions`. This prevents high-frequency feature snapshots from
    masquerading as independent trades on the same underlying event.

    ``costs`` are per-row *one-way* costs; each accepted trade is charged the
    entry cost at signal time plus the exit cost at the realized exit timestamp
    (``exit_costs`` overrides the exit-time lookup, e.g. for cost stress tests).
    """
    if len(df) != len(signals) or len(df) != len(costs):
        raise ValueError("df/signals/costs length mismatch")
    if len(df) == 0:
        return np.array([], dtype=float), np.array([], dtype=bool)
    times=pd.to_datetime(df['ts'],utc=True).to_numpy()
    symbols=df['symbol'].astype(str).to_numpy() if 'symbol' in df else np.array(['ASSET']*len(df))
    target=df['target_return'].astype(float).to_numpy()
    order=np.argsort(times)
    next_available: dict[str, pd.Timestamp] = {}
    active_exits: list[pd.Timestamp] = []
    net=np.zeros(len(df),dtype=float); accepted=np.zeros(len(df),dtype=bool)
    horizon=pd.Timedelta(minutes=max(1,horizon_minutes))

    # Exit liquidity: the one-way cost observed at the row nearest the realized
    # exit time on the same symbol (falls back to entry cost when no later row).
    if exit_costs is None:
        horizon_ns = np.timedelta64(int(horizon_minutes) * 60, 's')
        exit_costs = np.asarray(costs, dtype=float).copy()
        for s in pd.unique(symbols):
            idx = np.flatnonzero(symbols == s)
            srt = np.argsort(times[idx])
            ts_arr = times[idx][srt]
            c_arr = np.asarray(costs, dtype=float)[idx][srt]
            for i in idx:
                if int(signals[i]) == 0:
                    continue
                j = int(np.searchsorted(ts_arr, times[i] + horizon_ns))
                exit_costs[i] = c_arr[min(j, len(c_arr) - 1)]

    for i in order:
        side=int(signals[i])
        if side==0: continue
        ts=pd.Timestamp(times[i])
        active_exits=[x for x in active_exits if x>ts]
        sym=symbols[i]
        if sym in next_available and ts < next_available[sym]:
            continue
        if len(active_exits)>=max(1,max_positions):
            continue
        exit_at=ts+horizon
        accepted[i]=True
        net[i]=side*target[i]-float(costs[i])-float(exit_costs[i])
        next_available[sym]=exit_at
        active_exits.append(exit_at)
    return net, accepted


def portfolio_metrics(
    trades: pd.DataFrame,
    *,
    horizon_minutes: int = 15,
    initial_capital: float = 100_000.0,
) -> dict[str, float | int | None]:
    """Portfolio-level metrics from a ledger of completed trades.

    Unlike :func:`trade_metrics` (per-trade statistics), this builds an equity
    curve on a fixed time grid: each period's return is the PnL realized during
    that period divided by equity at period start. Idle periods count as 0%
    returns, so a strategy that trades rarely is not annualized as if every
    trade were a periodic observation.

    ``trades`` needs columns: entry_ts, exit_ts, net_return, notional_usd.
    """
    cols = {"entry_ts", "exit_ts", "net_return", "notional_usd"}
    if trades.empty or not cols.issubset(set(trades.columns)):
        return {"portfolio_periods": 0, "portfolio_net_return": 0.0, "portfolio_sharpe": None,
                "portfolio_max_drawdown": None, "avg_concurrent_positions": None,
                "capital_utilization": None}
    d = trades.copy()
    d["entry_ts"] = pd.to_datetime(d["entry_ts"], utc=True)
    d["exit_ts"] = pd.to_datetime(d["exit_ts"], utc=True)
    d["pnl_usd"] = d["net_return"].astype(float) * d["notional_usd"].astype(float)
    start = d["entry_ts"].min()
    end = d["exit_ts"].max()
    step = pd.Timedelta(minutes=max(1, horizon_minutes))
    n_periods = max(1, int(np.ceil((end - start) / step)))
    edges = [start + i * step for i in range(n_periods + 1)]
    equity = float(initial_capital)
    curve = [equity]
    exposure = []
    returns = []
    for i in range(n_periods):
        lo, hi = edges[i], edges[i + 1]
        realized = float(d[(d["exit_ts"] > lo) & (d["exit_ts"] <= hi)]["pnl_usd"].sum())
        open_notional = float(d[(d["entry_ts"] < hi) & (d["exit_ts"] > lo)]["notional_usd"].sum())
        n_open = int(((d["entry_ts"] < hi) & (d["exit_ts"] > lo)).sum())
        r = realized / equity if equity > 0 else 0.0
        equity += realized
        returns.append(r)
        curve.append(equity)
        exposure.append((n_open, open_notional))
    r = np.asarray(returns, dtype=float)
    sd = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    periods_per_year = 365.0 * 24.0 * 60.0 / max(1, horizon_minutes)
    sharpe = float(r.mean() / sd * math.sqrt(periods_per_year)) if sd > 0 else None
    eq = np.asarray(curve, dtype=float)
    dd = eq / np.maximum.accumulate(eq) - 1.0
    mean_open = float(np.mean([x[0] for x in exposure]))
    mean_notional = float(np.mean([x[1] for x in exposure]))
    return {
        "portfolio_periods": int(n_periods),
        "portfolio_net_return": float(equity / initial_capital - 1.0),
        "portfolio_sharpe": sharpe,
        "portfolio_max_drawdown": float(dd.min()),
        "avg_concurrent_positions": mean_open,
        "capital_utilization": float(mean_notional / initial_capital) if initial_capital > 0 else None,
    }
