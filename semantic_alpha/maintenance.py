from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from .author_alpha import snapshot_author_reputations
from .labels import build_labels
from .pipeline import base_symbol
from .schema import EngagementSnapshot
from .storage import Store, _iso

if TYPE_CHECKING:
    from .providers.xapi import XAPIClient


def _has_any_value(l) -> bool:
    return any(v is not None for v in l.returns.values())


def maintenance_step(
    store: Store,
    *,
    benchmark_symbol: str = "BTCUSDT",
    semantic_model: str | None = None,
    reputation_horizon: str = "15m",
    reputation_min_posts: int = 3,
    max_delay_seconds: int = 120,
    label_version: str = "v4",
    scan_window_hours: float = 72,
) -> dict:
    """Fill labels for unlabeled anchors AND refill still-immature label rows.

    Bounded scan: only anchors inside ``scan_window_hours`` are considered — an
    anchor older than the max label horizon plus grace that still has no label
    is missing market data that live mode never backfills (historical windows
    use ``backfill-labels`` instead). Label existence and actionable anchors are
    resolved from three bulk loads (label ts set, max-classified map, bounded
    post scan) rather than per-post queries — full-table scans made ticks take
    ~80min once posts passed ~30k.

    save_label merges per-horizon, so refilling cannot erase realized values.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=scan_window_hours)
    labeled = store.label_ts_set(label_version)
    sem_map = store.max_classified_map(semantic_model)
    feature_labels = 0
    post_labels = 0
    refilled = 0
    for f in store.features_since(cutoff):
        if (f.symbol, _iso(f.ts)) in labeled:
            continue
        l = build_labels(store, f.symbol, f.ts, benchmark_symbol=benchmark_symbol, max_delay_seconds=max_delay_seconds)
        if l and _has_any_value(l):
            store.save_label(l)
            feature_labels += 1
    for p in store.posts_with_symbols_since(cutoff):
        # Per-asset anchors at *actionable* time (max of observation and
        # classification), never created_at — returns before we could have seen
        # or classified the post are not tradeable evidence.
        sym = p.symbol  # posts_with_symbols_since yields one row per asset mapping
        anchor_str = _iso(p.first_seen_at or p.created_at)
        classified = sem_map.get((p.post_id, sym))
        if classified and classified > anchor_str:
            anchor_str = classified
        if (sym, anchor_str) in labeled:
            continue
        anchor = datetime.fromisoformat(anchor_str)
        l = build_labels(store, sym, anchor, benchmark_symbol=benchmark_symbol, max_delay_seconds=max_delay_seconds)
        if l and _has_any_value(l):
            store.save_label(l)
            post_labels += 1
    # Second pass: re-evaluate rows that exist but still have open horizons.
    # Gate on maturity: a label can only gain data once its earliest open
    # horizon's exit window has opened (horizon - proportional tolerance).
    # Without this, every tick rebuilt up to 2k labels whose 24h leg could not
    # possibly have matured — the dominant per-tick cost at ~35k immature rows.
    from .labels import HORIZONS
    finalized = 0
    max_horizon_deadline = now - timedelta(minutes=max(HORIZONS.values()) + 120)
    for old in store.immature_labels(label_version):
        open_mins = [m for name, m in HORIZONS.items() if old.returns.get(name) is None]
        if open_mins:
            earliest_open = min(m - min(max_delay_seconds / 60.0, max(10.0 / 60.0, m * 0.25)) for m in open_mins)
            if old.ts + timedelta(minutes=earliest_open) > now:
                continue
        l = build_labels(store, old.symbol, old.ts, benchmark_symbol=benchmark_symbol, max_delay_seconds=max_delay_seconds)
        # Save when anything moved: horizons filled, path excursions extended
        # into newly-arrived data, or the row completed so its matured flag
        # flips and it stops being rescanned.
        if l is not None and l != old:
            store.save_label(l)
            refilled += 1
        elif open_mins and old.ts < max_horizon_deadline:
            # Every exit window has closed and data still isn't there — the gap
            # is permanent in live mode. Release the row instead of rescanning
            # it every tick forever.
            store.mark_label_matured(old.symbol, old.ts, label_version)
            finalized += 1
    reps = snapshot_author_reputations(
        store, now, horizon=reputation_horizon,
        semantic_model=semantic_model, min_posts=reputation_min_posts,
    )
    repaired = repair_event_actionable_at(store)
    return {
        "ts": now.isoformat(), "feature_labels_created": feature_labels,
        "post_labels_created": post_labels, "labels_refilled": refilled,
        "labels_finalized": finalized,
        "author_reputations_created": len(reps),
        "events_repaired": repaired["events_repaired"],
    }


async def refresh_engagement(
    store: Store,
    social: "XAPIClient",
    *,
    symbols: list[str] | None = None,
    lookback_minutes: int = 14,
    slice_minutes: int = 8,
    max_pages: int = 2,
) -> dict:
    """Re-observe recently seen posts so engagement velocity has a second point.

    The stream writes exactly one engagement observation per post at receipt;
    velocity needs two. Re-fetching a delayed trailing slice of the search index
    returns the same posts with *current* counts — a bounded-cost second
    observation. ``max_pages`` caps spend; posts not previously seen are saved
    too (idempotent insert, honest late first_seen).

    The slice defaults to posts aged 6–14 minutes: the velocity feature reads
    posts *created* in the trailing 15 minutes, so a second observation must
    land while the post is still inside that window. Refreshing 15–25-minute-old
    posts can never produce a measurable velocity — the post ages out of the
    feature window before its second observation exists."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=lookback_minutes)
    end = now - timedelta(minutes=max(0, lookback_minutes - slice_minutes))
    if symbols is None:
        symbols = store.recent_post_symbols(now - timedelta(hours=2))
    refreshed = 0
    for sym in symbols:
        for p in await social.search_window(base_symbol(sym), start, end, max_pages=max_pages):
            # search_window normalizes to the base ticker; re-map to the tracked
            # market symbol so post_assets/engagement stay consistent with the
            # live path instead of creating phantom base-ticker mappings.
            p = p.model_copy(update={"symbol": sym})
            store.save_post(p)
            store.save_engagement(EngagementSnapshot(
                post_id=p.post_id, symbol=sym, observed_at=now,
                likes=int(p.likes or 0), reposts=int(p.reposts or 0), replies=int(p.replies or 0),
            ))
            refreshed += 1
    return {"ts": now.isoformat(), "engagement_refreshed": refreshed, "symbols": list(symbols)}


async def backfill_semantics(store: Store, engine, *, limit: int = 200, classify_timeout: float = 90.0) -> dict:
    """Classify posts that have no semantics row for the engine's current model.

    Live ingestion can drop or time out individual classifications; those posts
    are never redelivered, so without this pass they would stay permanently
    unclassified. Bounded per call so a backlog drains gradually without
    stalling label maturity work.

    Per-post failures are recorded (content failures count toward a cap so
    deterministic junk stops being retried/billed; provider failures like 402
    do not). A run of consecutive provider errors stops the pass early — the
    provider is down — but a single bad post no longer kills the whole batch,
    and a hung call cannot stall the maintenance loop forever."""
    import asyncio
    from .semantics.jev import is_provider_error
    # Coverage model = what the provider actually writes: resolved effective
    # model, else the most recently stored model, else the configured alias.
    # Querying under the bare alias pre-resolution makes every already-
    # classified post look pending → billed reclassification (burned 1,500
    # calls on a fresh process once).
    model = (getattr(engine, "effective_model", None)
             or store.latest_semantic_model()
             or getattr(engine, "query_model", getattr(engine, "model", engine.__class__.__name__)))
    pending = store.unsemanticized_posts(None, model, limit=limit)
    done = 0
    errors = 0
    consecutive_provider_errors = 0
    for p in pending:
        try:
            store.save_semantics(await asyncio.wait_for(engine.classify(p), timeout=classify_timeout))
            done += 1
            consecutive_provider_errors = 0
        except Exception as e:
            errors += 1
            prov = is_provider_error(str(e)) or isinstance(e, asyncio.TimeoutError)
            store.record_classification_failure(p.post_id, p.symbol, model, str(e)[:300], counted=not prov)
            if prov:
                consecutive_provider_errors += 1
                if consecutive_provider_errors >= 5:
                    break  # provider down/out of credits — leave backlog for next pass
    with store.conn() as c:
        true_pending = c.execute(
            """SELECT COUNT(*) FROM post_assets a
               LEFT JOIN post_semantics_asset s ON a.post_id=s.post_id AND a.symbol=s.symbol AND s.model=?
               LEFT JOIN classification_failures f ON f.post_id=a.post_id AND f.symbol=a.symbol AND f.model=? AND f.attempts>=3
               WHERE s.post_id IS NULL AND a.symbol!='MACRO' AND f.post_id IS NULL""",
            (model, model)).fetchone()[0]
    return {"ts": datetime.now(timezone.utc).isoformat(), "backfilled": done, "errors": errors,
            "pending_after": true_pending}


def repair_event_actionable_at(store: Store) -> dict:
    """Backfill ``actionable_at`` on events written before the field existed.

    ``actionable_at`` is derived cluster state — the earliest member's observed
    time (``first_seen_at``, falling back to ``created_at``) — so recomputing it
    from member posts repairs the record without touching raw evidence."""
    import json
    fixed = 0
    with store.conn() as c:
        rows = c.execute("SELECT event_id, payload_json FROM events").fetchall()
        for eid, pj in rows:
            payload = json.loads(pj)
            if payload.get("actionable_at") is not None:
                continue
            earliest = c.execute(
                """SELECT MIN(COALESCE(p.first_seen_at, p.created_at))
                   FROM event_members m JOIN posts p ON p.post_id=m.post_id
                   WHERE m.event_id=?""",
                (eid,),
            ).fetchone()[0]
            if earliest is None:
                continue
            payload["actionable_at"] = earliest
            c.execute("UPDATE events SET payload_json=? WHERE event_id=?", (json.dumps(payload), eid))
            fixed += 1
    return {"ts": datetime.now(timezone.utc).isoformat(), "events_repaired": fixed}


# Vendor billing is ~15 credits per delivered tweet-instance: webhook rows
# show credits == 15 x data_count exactly, and checks that deliver nothing
# produce no billable row. Delivery cost is ledgered as post_read at receipt
# (live.py stamps + logs every delivered tweet before symbol filtering), so a
# separate per-check estimate would double-count. The former
# estimate_rule_polling() was removed for that reason.
