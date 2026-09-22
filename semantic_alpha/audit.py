from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta

from .storage import Store


def _parse(dt: str) -> datetime:
    return datetime.fromisoformat(dt)


def leakage_audit(store: Store) -> dict:
    """Verify point-in-time discipline across the whole store.

    Two layers:
    1. Record hygiene: first_seen >= created, classified >= first_seen,
       engagement observed >= first_seen.
    2. Feature availability: a stored feature row may only reflect posts that
       were observed AND semantics that were classified at the feature's ts.
       This catches the historical-lookahead failure mode where a post created
       at 10:00 but first seen at 13:00 appeared in the 10:05 feature.
    """
    issues = []
    with store.conn() as c:
        posts = c.execute("SELECT post_id,created_at,first_seen_at FROM posts").fetchall()
        for r in posts:
            created = _parse(r["created_at"]); seen = _parse(r["first_seen_at"])
            if seen < created:
                issues.append({"type": "first_seen_before_created", "post_id": r["post_id"]})
        sem = c.execute(
            "SELECT s.post_id,s.classified_at,p.first_seen_at FROM post_semantics_asset s JOIN posts p ON p.post_id=s.post_id"
        ).fetchall()
        for r in sem:
            if _parse(r["classified_at"]) < _parse(r["first_seen_at"]):
                issues.append({"type": "classified_before_first_seen", "post_id": r["post_id"]})
        eng = c.execute(
            "SELECT e.post_id,e.observed_at,p.first_seen_at FROM engagement_snapshots e JOIN posts p ON p.post_id=e.post_id"
        ).fetchall()
        for r in eng:
            if _parse(r["observed_at"]) < _parse(r["first_seen_at"]):
                issues.append({"type": "engagement_before_first_seen", "post_id": r["post_id"]})
        feature_count = c.execute("SELECT COUNT(*) n FROM feature_snapshots").fetchone()[0]
        label_count = c.execute("SELECT COUNT(*) n FROM labels").fetchone()[0]

        # --- Feature availability check ---
        # Per (symbol): sorted arrays of (created, first_seen) and
        # (created, classified) so availability counts are O(log n) per feature.
        post_rows = c.execute(
            """SELECT a.symbol, p.created_at, p.first_seen_at
               FROM post_assets a JOIN posts p ON p.post_id=a.post_id"""
        ).fetchall()
        sem_rows = c.execute(
            """SELECT s.symbol, p.created_at, p.first_seen_at, s.classified_at
               FROM post_semantics_asset s JOIN posts p ON p.post_id=s.post_id"""
        ).fetchall()
        feat_rows = c.execute("SELECT symbol,ts,payload_json FROM feature_snapshots").fetchall()

    posts_by_sym: dict[str, list[tuple[datetime, datetime]]] = {}
    for r in post_rows:
        posts_by_sym.setdefault(r["symbol"], []).append((_parse(r["created_at"]), _parse(r["first_seen_at"])))
    sems_by_sym: dict[str, list[tuple[datetime, datetime, datetime]]] = {}
    for r in sem_rows:
        sems_by_sym.setdefault(r["symbol"], []).append(
            (_parse(r["created_at"]), _parse(r["first_seen_at"]), _parse(r["classified_at"]))
        )
    for v in posts_by_sym.values():
        v.sort(key=lambda x: x[0])
    for v in sems_by_sym.values():
        v.sort(key=lambda x: x[0])

    checked = 0
    for r in feat_rows:
        sym = r["symbol"]
        ts = _parse(r["ts"])
        try:
            vals = json.loads(r["payload_json"]).get("values", {})
        except (ValueError, AttributeError):
            continue
        p_list = posts_by_sym.get(sym, [])
        s_list = sems_by_sym.get(sym, [])
        p_created = [x[0] for x in p_list]
        for minutes in (1, 5, 15):
            lo = ts - timedelta(minutes=minutes)
            a, b = bisect_left(p_created, lo), bisect_right(p_created, ts)
            # Posts created in-window that were observable at ts.
            available = sum(1 for _, seen in p_list[a:b] if seen <= ts)
            stored = int(float(vals.get(f"posts_{minutes}m", 0.0) or 0.0))
            if stored > available:
                issues.append({
                    "type": "feature_social_lookahead", "symbol": sym, "ts": r["ts"],
                    "field": f"posts_{minutes}m", "stored": stored, "available": available,
                })
                break  # one lookahead flag per feature row is enough
        else:
            # Semantic availability: any nonzero semantic field requires at least
            # one classification completed by ts on a post observed by ts.
            s_created = [x[0] for x in s_list]
            a, b = bisect_left(s_created, ts - timedelta(minutes=15)), bisect_right(s_created, ts)
            sem_available = any(seen <= ts and cls <= ts for _, seen, cls in s_list[a:b])
            semantic_nonzero = any(
                float(vals.get(k, 0.0) or 0.0) != 0.0
                for k in ("intent_sum_5m", "new_info_rate_5m", "impact_mean_5m", "intent_mean_15m")
            )
            if semantic_nonzero and not sem_available:
                issues.append({
                    "type": "feature_semantic_lookahead", "symbol": sym, "ts": r["ts"],
                    "detail": "semantic fields nonzero but no classification was available at feature time",
                })
        checked += 1

    return {
        "ok": not issues,
        "issues": issues[:200],
        "issue_count": len(issues),
        "feature_rows": feature_count,
        "feature_rows_checked": checked,
        "label_rows": label_count,
    }
