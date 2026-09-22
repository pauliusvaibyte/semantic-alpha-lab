from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone

from .config import settings
from .storage import Store
from .backtest import estimated_execution_cost_bps


def usage_summary(store: Store, *, days: int | None = 30, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    start = now - timedelta(days=days) if days is not None else None
    rows = store.usage_records(start=start, end=now + timedelta(seconds=1))
    by = defaultdict(lambda: {"records": 0, "units": 0.0, "estimated_usd": 0.0})
    for r in rows:
        key = f"{r.provider}:{r.category}"
        by[key]["records"] += 1
        by[key]["units"] += float(r.units)
        by[key]["estimated_usd"] += float(r.estimated_usd)
    variable = float(sum(r.estimated_usd for r in rows))
    fixed = float(settings.fixed_monthly_data_usd * (days / 30.0 if days is not None else 0.0))
    return {
        "days": days,
        "records": len(rows),
        "variable_usd": variable,
        "fixed_prorated_usd": fixed,
        "total_usd": variable + fixed,
        "by_provider_category": dict(by),
    }


def operating_economics(store: Store, *, days: int = 30, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    usage = usage_summary(store, days=days, now=now)
    closed = [
        t for t in store.paper_trades("CLOSED")
        if t.closed_at is not None and start <= t.closed_at <= now
    ]
    paper_pnl = float(sum((t.realized_return or 0.0) * t.notional_usd for t in closed))
    net_after_data = paper_pnl - float(usage["total_usd"])
    return {
        "days": days,
        "closed_paper_trades": len(closed),
        "paper_pnl_after_exchange_costs_usd": paper_pnl,
        "data_and_model_cost_usd": usage["total_usd"],
        "paper_pnl_after_data_cost_usd": net_after_data,
        "data_cost_share_of_positive_pnl": (float(usage["total_usd"]) / paper_pnl if paper_pnl > 0 else None),
        "usage": usage,
        "note": "Paper P&L is simulated. Data/model costs are provider/configured estimates and should be reconciled to invoices before deployment.",
    }


def capacity_report(store: Store, notionals: list[float] | None = None, days: int = 30,
                    source: str = "bybit") -> dict:
    """Execution-cost curve across position sizes from stored book depth.

    Answers "what size does the edge survive at" before trusting gross alpha:
    for each notional, the median/p90 estimated one-way cost in bps per symbol.
    """
    import numpy as np
    notionals = notionals or [500.0, 1_000.0, 5_000.0, 10_000.0, 25_000.0, 50_000.0]
    start = datetime.now(timezone.utc) - timedelta(days=days)
    out: dict[str, dict] = {}
    for sym in store.symbols():
        snaps = [m for m in store.market_between(sym, start, datetime.now(timezone.utc), source=source)
                 if m.spread_bps is not None or m.depth_bid_10bps or m.depth_ask_10bps]
        if not snaps:
            continue
        per_notional = {}
        for n in notionals:
            costs = [
                estimated_execution_cost_bps(
                    {"spread_bps": m.spread_bps or 0.0,
                     "depth_bid_10bps": m.depth_bid_10bps or 0.0,
                     "depth_ask_10bps": m.depth_ask_10bps or 0.0},
                    notional_usd=n)
                for m in snaps
            ]
            per_notional[str(int(n))] = {
                "median_cost_bps": float(np.median(costs)),
                "p90_cost_bps": float(np.quantile(costs, 0.9)),
            }
        out[sym] = {"snapshots": len(snaps), "cost_bps_by_notional": per_notional}
    return {"days": days, "source": source, "symbols": out,
            "note": "One-way cost = taker fee + half spread + depth impact; not a fill simulator."}
