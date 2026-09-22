from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .clustering import assign_posts_to_events
from .features import FeatureEngine
from .providers.bybit import BybitPublicClient
from .providers.xapi import XAPIClient
from .schema import EngagementSnapshot, FeatureSnapshot
from .storage import Store


@dataclass
class CycleResult:
    symbol: str
    posts_fetched: int
    posts_inserted: int
    posts_semanticized: int
    events: int
    feature: FeatureSnapshot


def base_symbol(market_symbol: str) -> str:
    s = market_symbol.upper()
    for q in ("USDT", "USDC", "USD"):
        if s.endswith(q):
            return s[:-len(q)]
    return s


async def collect_cycle(
    store: Store,
    market_symbol: str,
    market_client: BybitPublicClient,
    social_client: XAPIClient,
    semantic_engine,
    social_lookback_minutes: int = 5,
    lag_overlap_seconds: int = 60,
    extra_market_clients: list | None = None,
    feature_version: str = "v4",
) -> CycleResult:
    now = datetime.now(timezone.utc)
    x_symbol = base_symbol(market_symbol)
    # Watermark-driven windows: steady state resumes at the previous window edge
    # plus a small overlap for posts indexed late by the provider. First run or a
    # gap longer than the lookback falls back to the full lookback, so we never
    # silently issue a huge backfill window after downtime.
    lookback_start = now - timedelta(minutes=social_lookback_minutes)
    watermark = store.get_social_cursor(market_symbol)
    window_start = watermark - timedelta(seconds=lag_overlap_seconds) if watermark and watermark >= lookback_start else lookback_start
    market_task = market_client.snapshot(market_symbol)
    posts_task = social_client.search_window(x_symbol, window_start, now, known_ids=store.post_ids_between(window_start, now))
    market, posts = await asyncio.gather(market_task, posts_task)
    store.save_market(market)
    # Secondary venues (binance/okx/...) feed cross-venue dispersion features;
    # canonical market truth stays on the primary client's source.
    for extra in extra_market_clients or []:
        try:
            store.save_market(await extra.snapshot(market_symbol))
        except Exception:
            pass  # venue redundancy must not kill the primary cycle
    store.set_social_cursor(market_symbol, now)
    normalized_posts = [p.model_copy(update={"symbol": market_symbol}) for p in posts]
    inserted = 0
    for p in normalized_posts:
        inserted += int(store.save_post(p))
        # Every observation is stored separately; this creates a forward-only
        # engagement curve without mutating the original post snapshot.
        store.save_engagement(EngagementSnapshot(
            post_id=p.post_id, symbol=market_symbol, observed_at=now,
            likes=int(p.likes or 0), reposts=int(p.reposts or 0), replies=int(p.replies or 0),
        ))

    # Prefer the provider-reported effective model for coverage/feature identity.
    model_name = getattr(semantic_engine, "query_model", getattr(semantic_engine, "model", semantic_engine.__class__.__name__))
    unclassified = store.unsemanticized_posts(market_symbol, model_name, limit=2000)
    for p in unclassified:
        store.save_semantics(await semantic_engine.classify(p))

    # Persistent narrative lifecycle: attach this cycle's posts to still-open
    # events (stable event_id) and only mint new events for unmatched posts.
    ev = assign_posts_to_events(store, market_symbol, normalized_posts, max_gap_minutes=30)

    # The first classify may resolve an alias to the provider-reported effective
    # model (jev-latest -> jev-1.13.0); re-read so feature reads address the
    # model identity under which semantics were actually stored.
    model_name = getattr(semantic_engine, "query_model", getattr(semantic_engine, "model", semantic_engine.__class__.__name__))
    feature = FeatureEngine(store, model_name, feature_version=feature_version).build(market_symbol, market.ts, market)
    # event-level factors, added after base feature build
    recent_events = store.events_between(market_symbol, now - timedelta(minutes=15), now)
    feature.values["events_15m"] = float(len(recent_events))
    feature.values["event_author_breadth_15m"] = float(sum(e.get("unique_authors",0) for e in recent_events))
    store.save_features(feature)
    return CycleResult(market_symbol, len(posts), inserted, len(unclassified), int(ev["attached"] + ev["new_events"]), feature)
