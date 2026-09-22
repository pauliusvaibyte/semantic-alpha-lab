from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone

from .clustering import assign_posts_to_events
from .config import settings
from .cross_asset import enrich_cross_asset_at
from .features import FeatureEngine
from .pipeline import base_symbol
from .providers.binance import BinancePublicClient
from .providers.bybit_ws import BybitLinearStream, MicrostructureAccumulator
from .providers.x_stream import XRealtimeStream, posts_from_event, symbols_from_tag_or_text
from .providers.xapi import normalize_post
from .schema import ApiUsageRecord, EngagementSnapshot
from .storage import Store


def delivery_usage_record(raw: dict, tag: str | None, frame_seq: int, index: int, ts: datetime) -> ApiUsageRecord:
    """Usage identity for one stream-delivered tweet.

    The vendor bills per delivered tweet per matched rule, so the reference
    must be unique per frame — keying by post_id alone would let re-deliveries
    and second-rule matches collapse onto one ledger row (billed, unlogged).
    Stamping the id onto the raw payload also lets ``save_post`` reuse the same
    identity when the post is stored normally."""
    ref = str(raw.get("_semantic_alpha_delivery_id") or f"{raw.get('id')}:{tag or 'stream'}:{frame_seq}:{index}")
    return ApiUsageRecord(
        provider="twitterapi.io", category="post_read", reference_id=ref,
        ts=ts, units=1.0, unit_name="post",
        estimated_usd=settings.x_cost_per_post_usd, cost_source="configured_rate",
        metadata={"source": "x_ws"},
    )


async def run_live_stack(
    store: Store,
    symbols: list[str],
    semantic_engine,
    x_api_key: str,
    x_ws_url: str,
    bybit_ws_url: str,
    market_snapshot_seconds: int = 1,
    feature_seconds: int = 15,
    semantic_workers: int = 4,
    duration_seconds: int = 0,
    feature_version: str = "v4",
    social_stall_seconds: float = 90.0,
    delivery_recycle_seconds: float = 300.0,
    classify_timeout: float = 60.0,
    binance_poll_seconds: float = 20.0,
):
    symbols=[s.upper().replace('/','') for s in symbols]
    acc={s:MicrostructureAccumulator(s) for s in symbols}
    q: asyncio.Queue = asyncio.Queue(maxsize=5000)
    stop=asyncio.Event()
    def _model_name() -> str:
        # Resolved lazily: Jev reports its effective model only after the first
        # classify call. Before that, fall back to the most recently *stored*
        # model — coverage checks under the bare alias ('jev-latest') treat
        # already-classified posts as pending → billed reclassification.
        return (getattr(semantic_engine,'effective_model',None)
                or store.latest_semantic_model()
                or getattr(semantic_engine,'query_model',getattr(semantic_engine,'model',semantic_engine.__class__.__name__)))
    counters={"social_events":0,"posts":0,"semanticized":0,"market_snapshots":0,"feature_snapshots":0,"dropped":0,"deduped":0}
    # Prevent duplicate semantic calls when the same post matches multiple filter
    # rules or is redelivered by the stream. Bounded so a long-running collector
    # does not leak memory.
    recently_queued: OrderedDict[tuple[str,str], None] = OrderedDict()
    max_recent_keys=100_000

    async def market_reader():
        async for msg in BybitLinearStream(symbols,bybit_ws_url).messages():
            ctrl=msg.get('_control')
            if ctrl:
                ts=datetime.fromisoformat(msg['ts'])
                for a in acc.values():
                    if ctrl=='connected': a.note_connected(ts)
                    elif ctrl=='disconnected': a.note_disconnected(ts)
                for sym in acc:
                    store.save_feed_health(sym,ctrl,{'source':'bybit_ws'},ts)
                counters.setdefault('reconnects',0)
                if ctrl=='disconnected': counters['reconnects']+=1
                continue
            topic=str(msg.get('topic') or '')
            sym=None; data=msg.get('data')
            if isinstance(data,dict): sym=data.get('symbol') or data.get('s')
            if not sym:
                parts=topic.split('.'); sym=parts[-1] if parts else None
            if sym in acc: acc[sym].apply(msg)
            if stop.is_set(): break

    async def social_reader():
        stream = XRealtimeStream(x_api_key, x_ws_url)
        it = stream.events()
        last_delivery = datetime.now(timezone.utc)
        gap_flagged = False
        last_queue_warn = 0.0
        while not stop.is_set():
            try:
                event = await asyncio.wait_for(anext(it), timeout=social_stall_seconds)
            except asyncio.TimeoutError:
                # The vendor sends ping events roughly every 20s, so a healthy
                # socket is never quiet this long — silence means a half-dead
                # connection TCP keepalive won't catch quickly. Force reconnect.
                counters['social_silent'] = counters.get('social_silent', 0) + 1
                store.save_feed_health('STREAM', 'silent', {'source': 'x_ws', 'seconds': social_stall_seconds}, datetime.now(timezone.utc))
                try: await it.aclose()
                except Exception: pass
                it = stream.events()
                last_delivery = datetime.now(timezone.utc); gap_flagged = False
                continue
            except StopAsyncIteration:
                it = stream.events()
                continue
            ctrl = event.get('_control')
            if ctrl:
                store.save_feed_health('STREAM', ctrl, {'source': 'x_ws', 'error': event.get('error')}, datetime.now(timezone.utc))
                if ctrl == 'connected':
                    last_delivery = datetime.now(timezone.utc); gap_flagged = False
                continue
            now = datetime.now(timezone.utc)
            pairs = posts_from_event(event)
            delivered = False
            for idx,(tag,raw) in enumerate(pairs):
                # The vendor bills every delivered tweet (~15cr/instance), even
                # ones our symbol matcher discards — stamp + log usage for ALL
                # deliveries before any filtering so billed spend is never
                # invisible. save_post replays the same delivery_id later, so
                # normal processing does not double-count.
                post_id=str(raw.get("id") or "")
                if isinstance(raw, dict) and post_id:
                    raw['_semantic_alpha_delivery_id']=f"{post_id}:{tag or 'stream'}:{counters['social_events']}:{idx}"
                    store.save_usage(delivery_usage_record(raw, tag, counters['social_events'], idx, now))
                matched=symbols_from_tag_or_text(tag,str(raw.get('text') or ''),symbols)
                if not matched:
                    counters['unmatched']=counters.get('unmatched',0)+1
                    continue
                delivered = True
                first_seen=now
                for sym in matched:
                    key=(post_id,sym)
                    if post_id and key in recently_queued:
                        counters['deduped']+=1
                        continue
                    if post_id:
                        recently_queued[key]=None
                        recently_queued.move_to_end(key)
                        if len(recently_queued)>max_recent_keys:
                            recently_queued.popitem(last=False)
                    try: q.put_nowait((sym,raw,first_seen))
                    except asyncio.QueueFull:
                        counters['dropped']+=1
                        if time.monotonic()-last_queue_warn>60:
                            last_queue_warn=time.monotonic()
                            store.save_feed_health('STREAM','queue_saturated',{'source':'x_ws','dropped':counters['dropped'],'qsize':q.qsize()},now)
            if pairs:
                counters['social_events']+=1
            if delivered:
                last_delivery = now; gap_flagged = False
            else:
                gap = (now - last_delivery).total_seconds()
                if gap > delivery_recycle_seconds:
                    # Pings alive but no tweets for minutes on active rules: the
                    # vendor binds deliveries to a single socket per key and a
                    # newer connection elsewhere can steal it. Recycle to rebind.
                    counters['delivery_recycles'] = counters.get('delivery_recycles', 0) + 1
                    store.save_feed_health('STREAM', 'delivery_recycle', {'source': 'x_ws', 'gap_seconds': gap}, now)
                    try: await it.aclose()
                    except Exception: pass
                    it = stream.events()
                    last_delivery = now; gap_flagged = False
                elif gap > delivery_recycle_seconds / 2 and not gap_flagged:
                    gap_flagged = True
                    store.save_feed_health('STREAM', 'delivery_gap', {'source': 'x_ws', 'gap_seconds': gap}, now)
            if stop.is_set(): break

    async def semantic_worker():
        last_warn = 0.0
        def warn(kind: str, payload: dict) -> None:
            nonlocal last_warn
            if time.monotonic() - last_warn > 60:
                last_warn = time.monotonic()
                store.save_feed_health('STREAM', kind, payload, datetime.now(timezone.utc))
        while not stop.is_set():
            try: sym,raw,first_seen=await asyncio.wait_for(q.get(),timeout=1)
            except asyncio.TimeoutError: continue
            try:
                p=normalize_post(raw,base_symbol(sym)).model_copy(update={'symbol':sym,'first_seen_at':first_seen})
                inserted=store.save_post(p); counters['posts']+=int(inserted)
                store.save_engagement(EngagementSnapshot(post_id=p.post_id,symbol=sym,observed_at=first_seen,likes=int(p.likes or 0),reposts=int(p.reposts or 0),replies=int(p.replies or 0)))
                if not store.semantics_for_post(p.post_id,_model_name(),p.symbol):
                    try:
                        sem = await asyncio.wait_for(semantic_engine.classify(p), timeout=classify_timeout)
                    except asyncio.TimeoutError:
                        counters['classify_timeouts']=counters.get('classify_timeouts',0)+1
                        store.record_classification_failure(p.post_id,p.symbol,_model_name(),'timeout',counted=False)
                        warn('classify_timeout',{'post_id':p.post_id,'timeout_s':classify_timeout})
                    except Exception as e:
                        counters['classify_errors']=counters.get('classify_errors',0)+1
                        from .semantics.jev import is_provider_error
                        store.record_classification_failure(p.post_id,p.symbol,_model_name(),str(e)[:300],counted=not is_provider_error(str(e)))
                        warn('classify_error',{'post_id':p.post_id,'error':str(e)[:200]})
                    else:
                        store.save_semantics(sem); counters['semanticized']+=1
            except Exception as e:
                counters['worker_errors']=counters.get('worker_errors',0)+1
                warn('worker_error',{'error':str(e)[:200]})
            finally: q.task_done()

    async def snapshotter():
        last_feature=datetime.min.replace(tzinfo=timezone.utc)
        stale_marked: set[str]=set()
        while not stop.is_set():
            now=datetime.now(timezone.utc)
            for sym,a in acc.items():
                # Hard freshness gate: a disconnected feed keeps last-known state
                # in the accumulator but must not write fresh-looking rows.
                if not a.is_fresh(now):
                    if a.last_update and sym not in stale_marked:
                        stale_marked.add(sym)
                        store.save_feed_health(sym,'stale_state',{'source':'bybit_ws',**a.freshness(now)},now)
                        counters['stale_skipped']=counters.get('stale_skipped',0)+1
                    continue
                if sym in stale_marked:
                    stale_marked.discard(sym)
                    store.save_feed_health(sym,'recovered',{'source':'bybit_ws',**a.freshness(now)},now)
                snap=a.snapshot(now)
                if snap.last>0:
                    store.save_market(snap); counters['market_snapshots']+=1
            if (now-last_feature).total_seconds()>=feature_seconds:
                built=[]
                for sym,a in acc.items():
                    if not a.is_fresh(now): continue
                    snap=a.snapshot(now)
                    if snap.last<=0: continue
                    posts=store.posts_between(sym,now-timedelta(minutes=30),now,as_of=now)
                    assign_posts_to_events(store,sym,posts)
                    f=FeatureEngine(store,_model_name(),feature_version=feature_version).build(sym,now,snap); store.save_features(f); counters['feature_snapshots']+=1; built.append(sym)
                if len(built)>=2:
                    enrich_cross_asset_at(store,now,symbols=built,feature_version=feature_version,tolerance_seconds=max(30,feature_seconds*2))
                last_feature=now
            await asyncio.sleep(max(1,market_snapshot_seconds))

    async def binance_poller():
        """Free public REST snapshots from the secondary venue — revives the
        venue-dispersion / lead-lag features that are dead under a Bybit-only
        stream. Errors are throttled into feed_health, never fatal."""
        client = BinancePublicClient()
        last_warn = 0.0
        while not stop.is_set():
            for sym in symbols:
                try:
                    snap = await client.snapshot(sym)
                    if snap and snap.last > 0:
                        store.save_market(snap); counters['binance_snapshots']=counters.get('binance_snapshots',0)+1
                except Exception as e:
                    counters['binance_errors']=counters.get('binance_errors',0)+1
                    if time.monotonic()-last_warn>60:
                        last_warn=time.monotonic()
                        store.save_feed_health(sym,'venue_poll_error',{'source':'binance','error':(str(e) or repr(e))[:200]},datetime.now(timezone.utc))
            await asyncio.sleep(max(5.0,binance_poll_seconds))

    tasks=[asyncio.create_task(market_reader()),asyncio.create_task(social_reader()),asyncio.create_task(snapshotter())]
    if binance_poll_seconds>0:
        tasks.append(asyncio.create_task(binance_poller()))
    tasks += [asyncio.create_task(semantic_worker()) for _ in range(max(1,semantic_workers))]
    try:
        if duration_seconds>0:
            await asyncio.sleep(duration_seconds); stop.set()
        else:
            await asyncio.Event().wait()
    finally:
        stop.set()
        for t in tasks: t.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
    return counters
