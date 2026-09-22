from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pandas as pd

from .backtest import estimated_execution_cost_bps, trade_metrics
from .models import SemanticResidualModel
from .labels import HORIZONS
from .schema import PaperTrade
from .storage import Store


def _entry_regime(values: dict[str, float]) -> str:
    funding=float(values.get("funding_rate",0.0) or 0.0)
    oi=float(values.get("oi_change_30m",0.0) or 0.0)
    ret=float(values.get("return_30m",0.0) or 0.0)
    flow=float(values.get("signed_trade_notional_1m",0.0) or 0.0)
    if funding > 0.0002 and oi > 0.01: return "CROWDED_LONG"
    if funding < -0.0002 and oi > 0.01: return "CROWDED_SHORT"
    if ret >= 0.01: return "VOLATILE_UP"
    if ret <= -0.01: return "VOLATILE_DOWN"
    if ret >= 0.003 or flow > 0: return "FLOW_UP"
    if ret <= -0.003 or flow < 0: return "FLOW_DOWN"
    return "QUIET"


def _risk_sized_notional(values: dict[str, float], *, equity_usd: float, risk_per_trade: float, max_notional_usd: float, signal_strength: float) -> float:
    # Volatility is used only as a sizing brake. We deliberately avoid pretending
    # this proxy is an executable stop-loss distance.
    # Caps dominate floors: the cap is an upper bound, never a minimum. A trade
    # whose risk size falls below the minimum viable notional is skipped by the
    # caller rather than inflated past the cap.
    vol=max(0.0,float(values.get("realized_vol_15m",0.0) or 0.0))
    risk_distance=min(0.05,max(0.005,3.0*vol))
    risk_budget=max(0.0,equity_usd*risk_per_trade)
    raw=(risk_budget/risk_distance) if risk_budget>0 else max_notional_usd
    strength=min(1.0,max(0.35,float(signal_strength)))
    return min(max_notional_usd,raw*strength)


def _row_from_feature(f) -> dict:
    return {"symbol": f.symbol, "ts": f.ts, **f.values}


def _exit_price(store: Store, trade: PaperTrade) -> float | None:
    m = store.latest_market(trade.symbol)
    if not m or not m.last:
        return None
    if trade.side == "LONG":
        return float(m.bid or m.last)
    return float(m.ask or m.last)


def close_trade(store: Store, trade: PaperTrade, now: datetime, reason: str, max_market_age_seconds: int = 10) -> PaperTrade | None:
    m = store.latest_market(trade.symbol)
    px = _exit_price(store, trade)
    if px is None or m is None:
        return None
    if abs((now - m.ts).total_seconds()) > max_market_age_seconds:
        return None
    # Exit economics come from the fresh market snapshot, never a stale feature row.
    row = {"spread_bps": m.spread_bps or 0.0, "depth_bid_10bps": m.depth_bid_10bps or 0.0, "depth_ask_10bps": m.depth_ask_10bps or 0.0}
    exit_cost = estimated_execution_cost_bps(row, notional_usd=trade.notional_usd, include_spread=False)
    side = 1.0 if trade.side == "LONG" else -1.0
    gross = side * (px / trade.entry_price - 1.0)
    net = gross - (trade.entry_cost_bps + exit_cost) / 10_000.0
    closed = trade.model_copy(update={
        "status": "CLOSED", "closed_at": now, "exit_price": px,
        "exit_cost_bps": exit_cost, "realized_return": float(net), "close_reason": reason,
    })
    store.save_paper_trade(closed)
    return closed


def paper_step(
    store: Store,
    model: SemanticResidualModel,
    *,
    now: datetime | None = None,
    base_notional_usd: float = 1_000.0,
    equity_usd: float = 100_000.0,
    risk_per_trade: float = 0.0025,
    max_positions: int = 3,
    min_probability: float | None = None,
    min_semantic_edge_logodds: float | None = None,
    max_spread_bps: float = 15.0,
    depth_participation: float = 0.02,
    min_notional_usd: float = 50.0,
    max_feature_age_seconds: int = 60,
    max_market_age_seconds: int = 10,
) -> dict:
    now = now or datetime.now(timezone.utc)
    min_probability = float(model.probability_threshold if min_probability is None else min_probability)
    min_semantic_edge_logodds = float(model.semantic_edge_threshold if min_semantic_edge_logodds is None else min_semantic_edge_logodds)
    closed: list[PaperTrade] = []
    open_trades = store.paper_trades("OPEN")

    # Time-based exits are intentionally primary for event-driven alpha.
    for t in open_trades:
        if now >= t.opened_at + timedelta(minutes=t.horizon_minutes):
            x = close_trade(store, t, now, "HORIZON", max_market_age_seconds=max_market_age_seconds)
            if x:
                closed.append(x)
    open_trades = store.paper_trades("OPEN")

    # Scope to the model's own feature universe — a candidate trained on a
    # different feature_version must never silently score foreign rows.
    latest = store.latest_features(version=model.feature_version or "v4")
    skipped={"stale_feature":0,"stale_market":0,"feature_already_traded":0,"spread":0,"model_unscorable":0,"size_below_minimum":0}
    trained_through = None
    if model.trained_through:
        try:
            trained_through = datetime.fromisoformat(model.trained_through.replace("Z", "+00:00"))
        except ValueError:
            trained_through = None
    candidates = []
    for f in latest:
        if trained_through is not None and f.ts <= trained_through:
            continue
        if abs((now-f.ts).total_seconds()) > max_feature_age_seconds:
            skipped["stale_feature"]+=1; continue
        if store.paper_feature_already_used(f.symbol,f.ts,model.training_fingerprint):
            skipped["feature_already_traded"]+=1; continue
        if store.open_paper_trade_for_symbol(f.symbol):
            continue
        m = store.latest_market(f.symbol)
        if not m or not m.last or abs((now-m.ts).total_seconds()) > max_market_age_seconds:
            skipped["stale_market"]+=1; continue
        # Execution checks read the fresh market snapshot, not the (possibly
        # minute-old) feature row.
        spread = float(m.spread_bps if m.spread_bps is not None else (f.values.get("spread_bps", 0.0) or 0.0))
        if spread > max_spread_bps:
            skipped["spread"]+=1; continue
        try:
            score = model.score_row(f.values)
        except ValueError:
            skipped["model_unscorable"]+=1; continue
        p = score["p_full_up"]
        edge = score["semantic_edge_logodds"]
        side = None
        if p >= min_probability and edge >= min_semantic_edge_logodds:
            side = "LONG"
        elif p <= 1.0 - min_probability and edge <= -min_semantic_edge_logodds:
            side = "SHORT"
        if side is None:
            continue
        strength = abs(edge) * score["direction_confidence"]
        candidates.append((strength, side, f, m, score))

    candidates.sort(key=lambda x: x[0], reverse=True)
    capacity = max(0, max_positions - len(open_trades))
    opened = []
    for _, side, f, m, score in candidates[:capacity]:
        bid_depth = float(m.depth_bid_10bps or 0.0)
        ask_depth = float(m.depth_ask_10bps or 0.0)
        usable_depth = min([x for x in (bid_depth, ask_depth) if x > 0], default=0.0)
        notional = _risk_sized_notional(
            f.values, equity_usd=equity_usd, risk_per_trade=risk_per_trade,
            max_notional_usd=base_notional_usd, signal_strength=min(1.0, strength),
        )
        if usable_depth > 0:
            notional = min(notional, usable_depth * depth_participation)
        if notional < min_notional_usd:
            skipped["size_below_minimum"] += 1
            continue
        entry_px = float((m.ask if side == "LONG" else m.bid) or m.last)
        entry_cost = estimated_execution_cost_bps(
            {"spread_bps": m.spread_bps or 0.0, "depth_bid_10bps": bid_depth, "depth_ask_10bps": ask_depth},
            notional_usd=notional, include_spread=False,
        )
        t = PaperTrade(
            trade_id=str(uuid4()), symbol=f.symbol, side=side, opened_at=now,
            entry_price=entry_px, notional_usd=notional, entry_cost_bps=entry_cost,
            model_version=model.version, model_fingerprint=model.training_fingerprint, horizon_minutes=HORIZONS.get(model.horizon, 15),
            p_baseline_up=score["p_baseline_up"], p_full_up=score["p_full_up"],
            semantic_edge_logodds=score["semantic_edge_logodds"], signal_strength=float(strength),
            entry_regime=_entry_regime(f.values), entry_feature_ts=f.ts,
            entry_context={k: float(f.values.get(k, 0.0) or 0.0) for k in (
                "semantic_shock","information_price_gap_proxy","attention_z_5m","narrative_novelty_15m",
                "realized_vol_15m","spread_bps","funding_rate","oi_change_30m","return_30m"
            )},
        )
        store.save_paper_trade(t)
        opened.append(t)
    return {
        "opened": [x.model_dump(mode="json") for x in opened],
        "closed": [x.model_dump(mode="json") for x in closed],
        "open_positions": len(store.paper_trades("OPEN")),
        "skipped": skipped,
    }


def paper_summary(store: Store, model_fingerprint: str | None = None) -> dict:
    trades = store.paper_trades()
    if model_fingerprint is not None:
        trades = [t for t in trades if t.model_fingerprint == model_fingerprint]
    closed = [t for t in trades if t.status == "CLOSED" and t.realized_return is not None]
    returns = pd.Series([t.realized_return for t in closed], dtype=float)
    metrics = trade_metrics(returns, periods_per_year=365 * 24 * 4)
    pnl_usd = float(sum((t.realized_return or 0.0) * t.notional_usd for t in closed))
    return {
        "total_trades": len(trades), "open_trades": len([t for t in trades if t.status == "OPEN"]),
        "closed_trades": len(closed), "realized_pnl_usd": pnl_usd, **metrics,
    }


def paper_attribution(store: Store, model_fingerprint: str | None = None) -> dict:
    trades=store.paper_trades()
    if model_fingerprint is not None:
        trades=[t for t in trades if t.model_fingerprint==model_fingerprint]
    closed=[t for t in trades if t.status=="CLOSED" and t.realized_return is not None]
    if not closed:
        return {"closed_trades":0,"by_symbol":[],"by_side":[],"by_regime":[],"by_edge_bucket":[]}
    rows=[]
    for t in closed:
        rows.append({
            "symbol":t.symbol,"side":t.side,"regime":t.entry_regime or "UNKNOWN",
            "return":float(t.realized_return or 0.0),"pnl_usd":float((t.realized_return or 0.0)*t.notional_usd),
            "semantic_edge":float(t.semantic_edge_logodds),"signal_strength":float(t.signal_strength or 0.0),
            "notional_usd":float(t.notional_usd),
        })
    d=pd.DataFrame(rows)
    def grouped(col: str) -> list[dict]:
        out=[]
        for key,g in d.groupby(col,dropna=False):
            met=trade_metrics(g["return"])
            out.append({
                col:str(key),"trades":int(len(g)),"mean_return":float(g["return"].mean()),
                "win_rate":float((g["return"]>0).mean()),"pnl_usd":float(g["pnl_usd"].sum()),
                **{k:v for k,v in met.items() if k not in {"trades"}},
            })
        return sorted(out,key=lambda x:x["pnl_usd"],reverse=True)
    buckets=[]
    if len(d)>=8 and d.semantic_edge.nunique()>=4:
        q=min(4,int(d.semantic_edge.nunique()))
        d["edge_bucket"]=pd.qcut(d.semantic_edge.abs(),q=q,duplicates="drop").astype(str)
        buckets=grouped("edge_bucket")
    return {
        "closed_trades":len(closed),"pnl_usd":float(d.pnl_usd.sum()),
        "by_symbol":grouped("symbol"),"by_side":grouped("side"),"by_regime":grouped("regime"),"by_edge_bucket":buckets,
    }
