"""v4 validation battery — one-shot honest inference on the collected window.

Discipline: every test runs on independent 1h buckets (serial correlation at
feature level manufactures fake significance). Effect must survive: split-half
replication, block bootstrap, BH-FDR, and partialling out post volume.
"""
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta

sys.path.insert(0, ".")
from semantic_alpha.storage import Store
from semantic_alpha.config import settings

random.seed(7)

KEYS = ["intent_mean_15m", "intent_residual_5m", "rel_posts_15m", "rel_share_15m",
        "new_info_rate_15m", "shill_rate_15m", "evidence_rate_15m", "rumor_impact_15m",
        "impact_mean_15m", "semantic_confidence_mean_15m", "reactive_rate_15m",
        "liquidation_imbalance_mean_5m"]
HORIZONS = ["15m", "30m", "1h", "4h"]


def _rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(xs, ys):
    n = len(xs)
    if n < 8:
        return None
    rx, ry = _rank(xs), _rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx)); dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx > 1e-12 and dy > 1e-12 else None


def block_boot_rho(xs, ys, iters=2000, block=6):
    n = len(xs)
    stats = []
    for _ in range(iters):
        idx = []
        while len(idx) < n:
            start = random.randrange(n)
            idx.extend(range(start, min(start + block, n)))
        idx = idx[:n]
        r = spearman([xs[i] for i in idx], [ys[i] for i in idx])
        if r is not None:
            stats.append(r)
    if len(stats) < 50:
        return (None, None)
    stats.sort()
    return stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats))]


def residualize(y, x):
    """Remove linear dependence on control x."""
    mx, my = sum(x) / len(x), sum(y) / len(y)
    denom = sum((v - mx) ** 2 for v in x)
    if denom < 1e-12:
        return y
    b = sum((a - mx) * (c - my) for a, c in zip(x, y, strict=True)) / denom
    return [c - b * (a - mx) for a, c in zip(x, y, strict=True)]


def bucket(ts, mins):
    return ts.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=(ts.minute // mins) * mins)


def main():
    s = Store(getattr(settings, "db_path", None))
    out = {"ts": datetime.now(timezone.utc).isoformat()}

    with s.conn() as c:
        rows = c.execute(
            """SELECT f.symbol, f.ts, f.payload_json, l.payload_json
               FROM feature_snapshots f JOIN labels l
                 ON l.symbol=f.symbol AND l.ts=f.ts AND l.label_version='v4'
               WHERE f.feature_version='v4'""").fetchall()

    # bucket means: {(sym, hour): {key: [vals]}, 'ret_15m': [...], ...}
    bk = defaultdict(lambda: defaultdict(list))
    for sym, ts_s, fj, lj in rows:
        f, l = json.loads(fj), json.loads(lj)
        rets = l.get("returns", {})
        if rets.get("15m") is None:
            continue
        b = (sym, bucket(datetime.fromisoformat(ts_s), 60))
        for h in HORIZONS:
            bk[b][f"r_{h}"].append(rets.get(h))
        for k in KEYS:
            v = f.get("values", {}).get(k)
            bk[b][k].append(float(v) if v is not None else float("nan"))

    # per-symbol bucket-level series
    series = defaultdict(lambda: defaultdict(list))
    for (sym, _b), cols in bk.items():
        for k in KEYS + [f"r_{h}" for h in HORIZONS]:
            vs = [v for v in cols[k] if v is not None and v == v]
            series[sym][k].append(sum(vs) / len(vs) if vs else float("nan"))

    report = {}
    all_tests = []  # (sym, key, rho, n) for FDR
    for sym in sorted(series):
        cols = series[sym]
        r15 = cols["r_15m"]
        vol = cols["rel_posts_15m"]
        for k in KEYS:
            pairs = [(x, y, v) for x, y, v in zip(cols[k], r15, vol, strict=True)
                     if x == x and y == y and v == v]
            if len(pairs) < 12:
                continue
            xs = [p[0] for p in pairs]; ys = [p[1] for p in pairs]; vs = [p[2] for p in pairs]
            rho = spearman(xs, ys)
            lo, hi = block_boot_rho(xs, ys)
            # split-half
            half = len(xs) // 2
            rho1 = spearman(xs[:half], ys[:half]); rho2 = spearman(xs[half:], ys[half:])
            # partial rho controlling for volume
            pr = None
            rx_ = residualize(xs, vs); ry_ = residualize(ys, vs)
            pr = spearman(rx_, ry_)
            # horizon decay for this feature
            decay = {}
            for h in HORIZONS:
                hp = [(x, y) for x, y in zip(cols[k], cols[f"r_{h}"], strict=True) if x == x and y == y]
                if len(hp) >= 12:
                    decay[h] = spearman([p[0] for p in hp], [p[1] for p in hp])
            entry = {"rho": round(rho, 3) if rho is not None else None,
                     "ci95": [round(lo, 3), round(hi, 3)] if lo is not None else None,
                     "split": [round(rho1, 2) if rho1 else None, round(rho2, 2) if rho2 else None],
                     "partial_vol": round(pr, 3) if pr is not None else None,
                     "decay": {h: round(v, 2) if v else None for h, v in decay.items()},
                     "n": len(pairs)}
            report.setdefault(k, {})[sym] = entry
            if rho is not None:
                all_tests.append((sym, k, rho, len(pairs)))

    # family-wise permutation: max |rho| under shuffled bucket returns
    max_abs_null = []
    for _ in range(500):
        for sym in sorted(series):
            cols = series[sym]
            rs = cols["r_15m"][:]
            random.shuffle(rs)
            for k in KEYS:
                pairs = [(x, y) for x, y in zip(cols[k], rs, strict=True) if x == x and y == y]
                if len(pairs) >= 12:
                    r = spearman([p[0] for p in pairs], [p[1] for p in pairs])
                    if r is not None:
                        max_abs_null.append(abs(r))
    max_abs_null.sort(reverse=True)
    fw_threshold = max_abs_null[int(0.05 * len(max_abs_null))] if max_abs_null else None

    out["battery_15m_returns"] = report
    out["familywise_5pct_threshold"] = round(fw_threshold, 3) if fw_threshold else None
    out["survivors"] = [
        {"sym": s_, "feature": k_, "rho": round(r_, 3), "n": n_}
        for s_, k_, r_, n_ in all_tests
        if fw_threshold and abs(r_) > fw_threshold
    ]
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
