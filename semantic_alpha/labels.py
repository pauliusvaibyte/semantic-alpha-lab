from __future__ import annotations
from datetime import datetime, timedelta, timezone

import numpy as np

from .schema import LabelRecord
from .storage import Store

HORIZONS = {"1m":1, "5m":5, "15m":15, "30m":30, "1h":60, "4h":240, "24h":1440}
_BETA_CACHE: dict[tuple[str, str, str, str], float] = {}


def _hour_key(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts=ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).replace(minute=0,second=0,microsecond=0).isoformat()


def _hourly_last(snaps) -> dict[datetime,float]:
    out={}
    for s in snaps:
        if not s.last: continue
        k=s.ts.replace(minute=0,second=0,microsecond=0)
        out[k]=float(s.last)
    return out


def estimate_market_beta(
    store: Store,
    symbol: str,
    benchmark_symbol: str,
    ts: datetime,
    *,
    lookback_hours: int = 72,
    prior_beta: float = 1.0,
    prior_strength: int = 20,
) -> float:
    if symbol == benchmark_symbol:
        return 1.0
    key=(store.path,symbol,benchmark_symbol,_hour_key(ts))
    if key in _BETA_CACHE:
        return _BETA_CACHE[key]
    start=ts-timedelta(hours=lookback_hours)
    a=_hourly_last(store.market_between(symbol,start,ts))
    b=_hourly_last(store.market_between(benchmark_symbol,start,ts))
    keys=sorted(set(a)&set(b))
    if len(keys)<8:
        beta=prior_beta
    else:
        ar=[]; br=[]
        for k0,k1 in zip(keys,keys[1:],strict=False):
            if a[k0] and b[k0]:
                ar.append(a[k1]/a[k0]-1.0); br.append(b[k1]/b[k0]-1.0)
        if len(ar)<7 or float(np.var(br,ddof=1))<1e-12:
            beta=prior_beta
        else:
            raw=float(np.cov(ar,br,ddof=1)[0,1]/np.var(br,ddof=1))
            n=len(ar)
            beta=(raw*n+prior_beta*prior_strength)/(n+prior_strength)
            beta=float(np.clip(beta,-0.5,3.5))
    _BETA_CACHE[key]=beta
    return beta


def post_actionable_ts(store: Store, post, symbol: str, model: str | None = None) -> datetime:
    """When this post's information became tradable for ``symbol``:
    max(observation time, classification completion for the given model).

    All post-anchored labels and event studies must share this anchor — keying
    a label at created_at measures returns that accrued before anyone could
    have seen (let alone classified) the post."""
    anchor = post.first_seen_at or post.created_at
    classified = [s.classified_at for s in store.semantics_for_post(post.post_id, model=model, symbol=symbol)]
    if classified:
        anchor = max(anchor, max(classified))
    return anchor


def build_labels(store: Store, symbol: str, ts: datetime, benchmark_symbol: str = "BTCUSDT", max_delay_seconds: int = 120) -> LabelRecord | None:
    """Entry-anchored forward labels.

    The entry price is the first canonical market snapshot at/after ``ts`` —
    the moment a signal could actually trade. Every horizon target is searched
    relative to that *realized* base timestamp, so sparse data delays the exit
    search instead of silently shortening the holding period. ``realized_minutes``
    records the true entry->exit duration per horizon."""
    base = store.nearest_market_at_or_after(symbol, ts, max_delay_seconds=max_delay_seconds)
    if not base or not base.last:
        return None
    entry_delay = (base.ts - ts).total_seconds()
    bench0 = store.nearest_market_at_or_after(benchmark_symbol, base.ts, max_delay_seconds=max_delay_seconds) if symbol != benchmark_symbol else None
    beta=estimate_market_beta(store,symbol,benchmark_symbol,base.ts) if bench0 else 1.0
    returns: dict[str, float | None] = {}
    abnormal: dict[str, float | None] = {}
    realized: dict[str, float | None] = {}
    max_up: dict[str, float | None] = {}
    max_down: dict[str, float | None] = {}
    abs_move: dict[str, float | None] = {}
    range_move: dict[str, float | None] = {}
    fade_ratio: dict[str, float | None] = {}
    for name, mins in HORIZONS.items():
        future_t = base.ts + timedelta(minutes=mins)
        # Exit tolerance is horizon-proportional: a "5m" label measured over
        # 6.75 minutes is a different measurement, not a disclosed nuance.
        # Cap at 25% of the horizon (floor 10s) or the caller's bound — a late
        # exit leaves the horizon None until a qualifying row exists.
        exit_tol = min(float(max_delay_seconds), max(10.0, mins * 60 * 0.25))
        f = store.nearest_market_at_or_after(symbol, future_t, max_delay_seconds=int(exit_tol))
        r = (f.last / base.last - 1.0) if f and f.last else None
        returns[name] = r
        realized[name] = (f.ts - base.ts).total_seconds() / 60.0 if f and f.last else None
        # Path excursion starts at the realized entry, not the requested ts —
        # prices before base.ts were never tradeable for this label. When the
        # exit realized late the excursion covers the *realized* window so the
        # path stats describe the same duration the return measures.
        path_end = f.ts if (f and f.last) else future_t + timedelta(seconds=exit_tol)
        path = store.market_between(symbol, base.ts, path_end)
        prices = [x.last for x in path if x.last]
        if prices:
            max_up[name] = max(prices) / base.last - 1.0
            max_down[name] = min(prices) / base.last - 1.0
            # Magnitude head: largest absolute excursion regardless of direction.
            abs_move[name] = max(abs(max_up[name]), abs(max_down[name]))
            # Volatility head: full path range within the horizon.
            range_move[name] = max_up[name] - max_down[name]
            # Crowding/reversal head: fraction of the dominant excursion retained
            # at horizon end. ~1.0 = move held; <0.3 = spike mostly faded.
            if r is not None and max_up[name] is not None and max_up[name] > 1e-9 and (max_down[name] or 0.0) > -max_up[name] * 0.5:
                fade_ratio[name] = r / max_up[name]
            elif r is not None and max_down[name] is not None and max_down[name] < -1e-9:
                fade_ratio[name] = r / max_down[name]
            else:
                fade_ratio[name] = None
        else:
            max_up[name] = None
            max_down[name] = None
            abs_move[name] = None
            range_move[name] = None
            fade_ratio[name] = None
        if r is None:
            abnormal[name] = None
        elif bench0 and bench0.last:
            bf = store.nearest_market_at_or_after(benchmark_symbol, future_t, max_delay_seconds=int(exit_tol))
            br = (bf.last / bench0.last - 1.0) if bf and bf.last else 0.0
            abnormal[name] = r - beta*br
        else:
            abnormal[name] = r
    return LabelRecord(
        symbol=symbol, ts=ts, returns=returns, abnormal_returns=abnormal,
        benchmark_symbol=benchmark_symbol, benchmark_beta=beta,
        base_ts=base.ts, entry_delay_seconds=entry_delay, realized_minutes=realized,
        max_up=max_up, max_down=max_down,
        abs_move=abs_move, range_move=range_move, fade_ratio=fade_ratio,
    )
