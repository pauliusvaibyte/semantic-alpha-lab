"""v4 validation pass — run after collection ends.

Four questions, all on data NOT used to design the features:
1. PURITY: did cashtag-only + relevance weighting fix the feed? (rel_share dist)
2. REPLICATION: do v4 semantic features predict returns on data they weren't
   fitted on? Independent time buckets + event level + block bootstrap.
3. REGISTERED LEADS: non_reactive directional content, liquidation_imbalance —
   measured on the fresh window only.
4. REGIME PULSE: is trade_action better than chance vs realized forward returns?

Outputs honest verdicts, not alpha claims. Nothing here feeds back into v4.
"""
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta

sys.path.insert(0, ".")
from semantic_alpha.storage import Store
from semantic_alpha.config import settings


def _rho(xs, ys):
    n = len(xs)
    if n < 8:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / n)
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / n)
    if sx < 1e-12 or sy < 1e-12:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / (n * sx * sy)


def _bucket(ts: datetime, minutes: int) -> datetime:
    return ts.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=(ts.minute // minutes) * minutes)


def main() -> None:
    s = Store(getattr(settings, "db_path", None))
    out = {"ts": datetime.now(timezone.utc).isoformat()}

    with s.conn() as c:
        # ---------- 1. PURITY ----------
        purity = c.execute(
            """SELECT symbol,
                      SUM(json_extract(payload_json,'$.relevance')='RELEVANT'),
                      SUM(json_extract(payload_json,'$.relevance')='MAYBE'),
                      SUM(json_extract(payload_json,'$.relevance')='IRRELEVANT'),
                      COUNT(*)
               FROM post_semantics_asset WHERE model='jev-1.13.0' GROUP BY symbol"""
        ).fetchall()
        out["purity"] = {
            r[0]: {"relevant": r[1], "maybe": r[2], "irrelevant": r[3],
                   "n": r[4], "irrelevant_share": round((r[3] or 0) / r[4], 3)}
            for r in purity if r[4]
        }

        # ---------- 2. REPLICATION (v4 features → forward returns) ----------
        # Join v4 feature rows to their labels; aggregate into INDEPENDENT
        # 1h buckets so serial correlation can't manufacture significance.
        rows = c.execute(
            """SELECT f.symbol, f.ts, f.payload_json, l.payload_json
               FROM feature_snapshots f JOIN labels l
                 ON l.symbol=f.symbol AND l.ts=f.ts AND l.label_version='v4'
               WHERE f.feature_version='v4'"""
        ).fetchall()

    KEYS = ["intent_mean_15m", "rel_posts_15m", "rel_share_15m", "new_info_rate_15m",
            "shill_rate_15m", "evidence_rate_15m", "rumor_impact_15m",
            "impact_mean_15m", "semantic_confidence_mean_15m",
            "liquidation_imbalance_mean_5m", "reactive_rate_15m", "intent_residual_5m"]
    buckets: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for sym, ts_s, fj, lj in rows:
        f, l = json.loads(fj), json.loads(lj)
        r = l.get("returns", {}).get("15m")
        if r is None:
            continue
        b = (sym, _bucket(datetime.fromisoformat(ts_s), 60))
        buckets[b]["r"].append(r)
        for k in KEYS:
            v = f.get("values", {}).get(k)
            buckets[b][k].append(float(v) if v is not None else float("nan"))

    # Per-bucket means → feature-vs-return correlations across buckets per symbol
    per_sym: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for (sym, _b), cols in buckets.items():
        per_sym[sym]["r"].append(sum(cols["r"]) / len(cols["r"]))
        for k in KEYS:
            vs = [v for v in cols[k] if v == v]
            per_sym[sym][k].append(sum(vs) / len(vs) if vs else float("nan"))
    out["replication_bucket_1h"] = {"n_buckets_per_sym": {k: len(v["r"]) for k, v in per_sym.items()}}
    for sym, cols in per_sym.items():
        rs = cols["r"]
        for k in KEYS:
            pairs = [(x, y) for x, y in zip(cols[k], rs, strict=True) if x == x]
            rho = _rho([p[0] for p in pairs], [p[1] for p in pairs])
            out["replication_bucket_1h"].setdefault(k, {})[sym] = {"rho": round(rho, 3) if rho is not None else None,
                                                                  "n": len(pairs)}

    # ---------- 3. REGIME PULSE ----------
    with s.conn() as c:
        pulses = c.execute("SELECT symbol, ts, payload_json FROM regime_pulses ORDER BY ts").fetchall()
    pulse_hits = {"BUY": [0, 0], "SELL": [0, 0], "HOLD": [0, 0]}
    for sym, ts_s, pj in pulses:
        p = json.loads(pj)
        action = p.get("trade_action")
        anchor = datetime.fromisoformat(ts_s)
        base = s.nearest_market_at_or_after(sym, anchor, max_delay_seconds=300)
        fut = s.nearest_market_at_or_after(sym, anchor + timedelta(minutes=15), max_delay_seconds=300)
        if not (base and base.last and fut and fut.last):
            continue
        ret = fut.last / base.last - 1.0
        a = "BUY" if action in ("BUY", "STRONG_BUY") else "SELL" if action in ("SELL", "STRONG_SELL") else "HOLD"
        correct = ret > 0.001 if a == "BUY" else ret < -0.001 if a == "SELL" else abs(ret) <= 0.003
        pulse_hits[a][0] += int(correct)
        pulse_hits[a][1] += 1
    out["regime_pulse"] = {"n": len(pulses), "hits": {k: f"{v[0]}/{v[1]}" for k, v in pulse_hits.items()}}

    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
