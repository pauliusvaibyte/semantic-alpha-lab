"""Pre-registered holdout test — v4_btc_intent_holdout (fingerprint 71f98143c326f2f3).

Evaluates ONE hypothesis on data collected strictly AFTER the lock timestamp:
  1h-bucket Spearman rho(intent_mean_15m -> fwd 15m return) > 0 for BTCUSDT.

Criteria (locked 2026-09-22T10:43:54Z, see holdout_v4_btc_intent.json):
  rho > 0.15, bootstrap CI95 lower > 0, partial-after-momentum+vol > 0.10,
  >= 36 independent buckets required for any verdict.
"""
import json
import random
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, ".")
from battery_v4 import spearman, residualize, block_boot_rho, bucket
from semantic_alpha.storage import Store
from semantic_alpha.config import settings

random.seed(11)


def main():
    spec = json.loads(open("holdout_v4_btc_intent.json").read())
    lock_ts = spec["locked_at"]
    s = Store(getattr(settings, "db_path", None))

    with s.conn() as c:
        rows = c.execute(
            """SELECT f.ts, f.payload_json, l.payload_json
               FROM feature_snapshots f JOIN labels l
                 ON l.symbol=f.symbol AND l.ts=f.ts AND l.label_version='v4'
               WHERE f.feature_version='v4' AND f.symbol=? AND f.ts>=?""",
            (spec["symbol"], lock_ts)).fetchall()

    bk = defaultdict(lambda: defaultdict(list))
    for ts_s, fj, lj in rows:
        f, l = json.loads(fj), json.loads(lj)
        r = l.get("returns", {}).get(spec["horizon"])
        if r is None:
            continue
        b = bucket(datetime.fromisoformat(ts_s), 60)
        bk[b]["r"].append(r)
        for k in [spec["feature"], "return_15m", "realized_vol_15m"]:
            v = f.get("values", {}).get(k)
            bk[b][k].append(float(v) if v is not None else float("nan"))

    xs, ys, mom, vol = [], [], [], []
    for cols in bk.values():
        cells = {k: [v for v in cols[k] if v == v] for k in cols}
        if not cells["r"]:
            continue
        ys.append(sum(cells["r"]) / len(cells["r"]))
        for k, acc in ((spec["feature"], xs), ("return_15m", mom), ("realized_vol_15m", vol)):
            acc.append(sum(cells[k]) / len(cells[k]) if cells[k] else float("nan"))

    pairs = [(x, y, m, v) for x, y, m, v in zip(xs, ys, mom, vol, strict=True)
             if x == x and m == m and v == v]
    n = len(pairs)
    xs = [p[0] for p in pairs]; ys = [p[1] for p in pairs]
    mom = [p[2] for p in pairs]; vol = [p[3] for p in pairs]

    rho = spearman(xs, ys) if n >= 12 else None
    lo, hi = block_boot_rho(xs, ys) if n >= 12 else (None, None)
    pmv = spearman(residualize(residualize(xs, mom), vol),
                   residualize(residualize(ys, mom), vol)) if n >= 12 else None

    crit = spec["criteria"]
    verdict = "INSUFFICIENT"
    if n >= crit["min_buckets"]:
        ok = (rho is not None and rho > crit["rho_threshold"]
              and lo is not None and lo > 0
              and pmv is not None and pmv > 0.10)
        verdict = "PASS" if ok else "FAIL"

    print(json.dumps({
        "holdout": spec["name"], "fingerprint": spec["fingerprint"],
        "locked_at": lock_ts, "n_buckets": n,
        "rho": round(rho, 3) if rho is not None else None,
        "ci95": [round(lo, 3), round(hi, 3)] if lo is not None else None,
        "partial_mom_vol": round(pmv, 3) if pmv is not None else None,
        "verdict": verdict,
    }, indent=2))


if __name__ == "__main__":
    main()
