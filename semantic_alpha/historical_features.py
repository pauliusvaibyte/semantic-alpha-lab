from __future__ import annotations

from datetime import datetime, timedelta

from .clustering import EventClusterer
from .cross_asset import enrich_cross_asset_at
from .features import FeatureEngine
from .storage import Store


def rebuild_historical_features(
    store: Store,
    symbols: list[str],
    start: datetime,
    end: datetime,
    *,
    interval_minutes: int = 5,
    semantic_model: str | None = None,
    feature_version: str = "v4",
    cluster_window_minutes: int = 30,
) -> dict:
    """Reconstruct feature snapshots from already-ingested point-in-time history.

    Designed for research, not speed. The interval defaults to 5 minutes to keep
    large historical rebuilds tractable in SQLite. All lookbacks end at the feature
    timestamp, preserving point-in-time semantics.
    """
    symbols=[s.upper().replace('/','') for s in symbols]
    generated=0; timestamps=set(); clusters_saved=0
    clusterer=EventClusterer(max_gap_minutes=cluster_window_minutes)
    for sym in symbols:
        markets=store.market_between(sym,start,end)
        if not markets:
            continue
        next_ts=start
        for m in markets:
            if m.ts < next_ts:
                continue
            posts=store.posts_between(sym,m.ts-timedelta(minutes=cluster_window_minutes),m.ts,as_of=m.ts)
            for c in clusterer.cluster(posts):
                store.save_event(c); clusters_saved+=1
            f=FeatureEngine(store,semantic_model,feature_version=feature_version).build(sym,m.ts,m)
            store.save_features(f); generated+=1; timestamps.add(m.ts)
            next_ts=m.ts+timedelta(minutes=max(1,interval_minutes))
    enriched=0
    for ts in sorted(timestamps):
        enriched += enrich_cross_asset_at(store,ts,symbols=symbols,feature_version=feature_version,tolerance_seconds=max(120,interval_minutes*60))
    return {'symbols':symbols,'feature_version':feature_version,'semantic_model':semantic_model,'features_generated':generated,'cross_asset_updates':enriched,'clusters_saved':clusters_saved}
