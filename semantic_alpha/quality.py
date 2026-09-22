from __future__ import annotations

from datetime import datetime
import json
import numpy as np

from .storage import Store


def data_quality(store: Store) -> dict:
    symbols={}
    with store.conn() as c:
        syms=[r[0] for r in c.execute("SELECT symbol FROM posts UNION SELECT symbol FROM market_snapshots UNION SELECT symbol FROM feature_snapshots ORDER BY symbol").fetchall()]
        def pct(x,q): return float(np.quantile(x,q)) if x else None
        for sym in syms:
            # Multi-asset correct: count post/asset mappings, not canonical rows,
            # so a post mapped to BTC+ETH counts toward both symbols' coverage.
            posts=c.execute("SELECT p.post_id,p.created_at,p.first_seen_at FROM post_assets a JOIN posts p ON p.post_id=a.post_id WHERE a.symbol=? ORDER BY p.created_at",(sym,)).fetchall()
            sem_n=c.execute("SELECT COUNT(DISTINCT post_id) FROM post_semantics_asset WHERE symbol=?",(sym,)).fetchone()[0]
            markets=c.execute("SELECT source,ts FROM market_snapshots WHERE symbol=? ORDER BY source,ts",(sym,)).fetchall()
            features=c.execute("SELECT ts FROM feature_snapshots WHERE symbol=?",(sym,)).fetchall()
            labels=c.execute("SELECT COUNT(*) FROM labels WHERE symbol=?",(sym,)).fetchone()[0]
            events=c.execute("SELECT COUNT(*) FROM events WHERE symbol=?",(sym,)).fetchone()[0]
            multi_eng=c.execute("SELECT COUNT(*) FROM (SELECT e.post_id,COUNT(*) n FROM engagement_snapshots e JOIN post_assets a ON a.post_id=e.post_id WHERE a.symbol=? GROUP BY e.post_id HAVING n>=2)",(sym,)).fetchone()[0]
            seen_lags=[]
            for p in posts:
                seen_lags.append(max(0.0,(datetime.fromisoformat(p['first_seen_at'])-datetime.fromisoformat(p['created_at'])).total_seconds()))
            # Gap stats per venue source — blending sources would manufacture fake
            # ~0s interleavings between venues and hide a dead feed's real gaps.
            gaps=[]; per_source={}
            src_ts: dict[str, list] = {}
            for r in markets:
                src_ts.setdefault(r['source'],[]).append(datetime.fromisoformat(r['ts']))
            for src,mts in src_ts.items():
                g=[(b-a).total_seconds() for a,b in zip(mts,mts[1:],strict=False)]
                per_source[src]={'snapshots':len(mts),'gap_median_s':pct(g,.5),'gap_p95_s':pct(g,.95),'gap_max_s':max(g) if g else None}
                if src in ('bybit','bybit_history','bybit_ws'):
                    gaps.extend(g)
            symbols[sym]={
                'posts':len(posts),'semantic_posts':sem_n,'semantic_coverage':(sem_n/len(posts) if posts else None),
                'first_seen_lag_median_s':pct(seen_lags,.5),'first_seen_lag_p95_s':pct(seen_lags,.95),
                'posts_with_2plus_engagement_snapshots':multi_eng,'engagement_reobservation_rate':(multi_eng/len(posts) if posts else None),
                'market_snapshots':len(markets),'market_gap_median_s':pct(gaps,.5),'market_gap_p95_s':pct(gaps,.95),'market_gap_max_s':max(gaps) if gaps else None,
                'market_sources':per_source,
                'feature_rows':len(features),'labels':labels,'events':events,
            }
    totals={
        'symbols':len(symbols),
        'posts':sum(v['posts'] for v in symbols.values()),
        'market_snapshots':sum(v['market_snapshots'] for v in symbols.values()),
        'feature_rows':sum(v['feature_rows'] for v in symbols.values()),
        'labels':sum(v['labels'] for v in symbols.values()),
    }
    warnings=[]
    for sym,v in symbols.items():
        if v['posts'] and (v['semantic_coverage'] or 0)<.95: warnings.append(f"{sym}: semantic coverage below 95%")
        if v['market_gap_p95_s'] is not None and v['market_gap_p95_s']>120: warnings.append(f"{sym}: market p95 gap >120s")
        if v['posts']>=20 and (v['engagement_reobservation_rate'] or 0)<.2: warnings.append(f"{sym}: low engagement re-observation coverage")
    return {'ok':not warnings,'totals':totals,'symbols':symbols,'warnings':warnings}


def latency_report(store: Store) -> dict:
    """Ingestion and semantic latency — the Attack-3 instrumentation.

    ingest_lag = first_seen - created_at (provider indexing delay).
    semantic_lag = classified_at - first_seen (our pipeline cost).
    """
    def _stats(xs):
        if not xs:
            return {"n": 0}
        a = np.asarray(xs, dtype=float)
        return {"n": len(xs), "mean_s": float(a.mean()), "p50_s": float(np.quantile(a, .5)),
                "p90_s": float(np.quantile(a, .9)), "p99_s": float(np.quantile(a, .99)), "max_s": float(a.max())}

    ingest: list[float] = []
    for p in store.all_posts():
        ingest.append(max(0.0, (p.first_seen_at - p.created_at).total_seconds()))

    semantic: list[float] = []
    with store.conn() as c:
        rows = c.execute(
            """SELECT s.payload_json, p.first_seen_at FROM post_semantics_asset s
               JOIN posts p ON p.post_id=s.post_id"""
        ).fetchall()
    for payload, first_seen in rows:
        s = json.loads(payload)
        lag = (datetime.fromisoformat(s["classified_at"]) - datetime.fromisoformat(first_seen)).total_seconds()
        semantic.append(max(0.0, lag))
    return {"ingest_lag": _stats(ingest), "semantic_lag": _stats(semantic)}


def feed_health_report(store: Store, hours: int = 6) -> dict:
    """Feed liveness: per-symbol topic ages, reconnects, stale-state events.

    The quality trap this prevents: a disconnected WS keeps last-known state, so
    a naive 'latest market ts' check sees fresh rows while the feed is dead.
    Freshness must be judged on exchange/topic timestamps, not row timestamps."""
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours)
    symbols = store.symbols()
    out = {"window_hours": hours, "symbols": {}}
    ok = True
    for sym in symbols:
        events = store.feed_health_between(sym, start, now)
        kinds: dict[str, int] = {}
        last_ts = None
        for e in events:
            kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
            last_ts = e["ts"]
        reconnects = kinds.get("disconnected", 0)
        stale_events = kinds.get("stale_state", 0)
        latest = store.latest_market(sym)
        row = {
            "events": len(events), "reconnects": reconnects, "stale_events": stale_events,
            "connected_events": kinds.get("connected", 0), "recovered_events": kinds.get("recovered", 0),
            "last_event_ts": last_ts,
            "latest_market_age_s": round((now - latest.ts).total_seconds(), 1) if latest else None,
        }
        if stale_events > 0 and kinds.get("recovered", 0) == 0:
            row["status"] = "STALE_UNRECOVERED"
            ok = False
        elif latest is None or (now - latest.ts).total_seconds() > 300:
            row["status"] = "NO_RECENT_MARKET_DATA"
            ok = False
        else:
            row["status"] = "OK"
        out["symbols"][sym] = row
    out["ok"] = ok
    return out
