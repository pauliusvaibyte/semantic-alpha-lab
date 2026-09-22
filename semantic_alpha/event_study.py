from __future__ import annotations

import hashlib
from statistics import NormalDist

import numpy as np
import pandas as pd

from .author_alpha import DIRECTION
from .labels import HORIZONS, build_labels, post_actionable_ts
from .schema import EventCluster
from .storage import Store


def _mode(values: list[str], default: str = "UNKNOWN") -> str:
    if not values:
        return default
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]


def _stable_seed(*parts: object) -> int:
    raw = "|".join(str(x) for x in parts).encode()
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def _normal_p_from_t(t_stat: float | None) -> float | None:
    """Two-sided normal approximation used only as a fast descriptive diagnostic."""
    if t_stat is None or not np.isfinite(t_stat):
        return None
    return float(2.0 * (1.0 - NormalDist().cdf(abs(float(t_stat)))))


def _cluster_keys(g: pd.DataFrame) -> list[tuple]:
    """Economic dependence units: (symbol, calendar day). Rows sharing a symbol
    and day move through the same regime — resampling them independently
    understates uncertainty."""
    keys = []
    for _, r in g.iterrows():
        ts = r.get("ts")
        day = pd.Timestamp(ts).strftime("%Y-%m-%d") if ts is not None and pd.notna(ts) else ""
        keys.append((str(r.get("symbol", "")), day))
    return keys


def _clustered_draw(x: np.ndarray, keys: list[tuple], rng: np.random.Generator) -> np.ndarray:
    units: dict[tuple, list[int]] = {}
    for i, k in enumerate(keys):
        units.setdefault(k, []).append(i)
    unit_keys = list(units.keys())
    picked = rng.choice(len(unit_keys), size=len(unit_keys), replace=True)
    idx = [i for u in picked for i in units[unit_keys[u]]]
    return x[idx]


def _bootstrap_ci(values: np.ndarray, *, seed: int, samples: int = 500, cluster_keys: list[tuple] | None = None) -> tuple[float | None, float | None]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (None, None)
    # For large event sets the CLT interval is both stable and much cheaper than
    # repeatedly resampling tens of thousands of events across many hypotheses.
    if len(x) >= 1000 and cluster_keys is None:
        se=float(np.std(x,ddof=1)/np.sqrt(len(x)))
        mean=float(np.mean(x)); return (mean-1.96*se, mean+1.96*se)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    if cluster_keys is not None and len(set(cluster_keys)) >= 2:
        for i in range(samples):
            means[i] = float(np.mean(_clustered_draw(x, cluster_keys, rng)))
    else:
        for i in range(samples):
            means[i] = float(np.mean(rng.choice(x, size=len(x), replace=True)))
    return (float(np.quantile(means, .025)), float(np.quantile(means, .975)))


def _bootstrap_mean_p(values: np.ndarray, *, seed: int, samples: int = 2000, cluster_keys: list[tuple] | None = None) -> float | None:
    """Shifted-null bootstrap p for mean != 0; replaces the normal-approx t as the
    inferential p-value (t retained as descriptive diagnostic)."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return None
    rng = np.random.default_rng(seed)
    if cluster_keys is not None and len(set(cluster_keys)) >= 2:
        means = np.array([np.mean(_clustered_draw(x, cluster_keys, rng)) for _ in range(samples)])
    else:
        means = rng.choice(x, size=(samples, len(x)), replace=True).mean(axis=1)
    return float(min(1.0, 2.0 * min(float((means <= 0).mean()), float((means >= 0).mean()))))


def _two_sample_test(hi: np.ndarray, lo: np.ndarray, *, seed: int, n_perm: int = 2000, n_boot: int = 2000) -> dict:
    """High-vs-low contrast propagating uncertainty from BOTH sides: permutation
    p on the mean difference plus a two-sample bootstrap CI."""
    rng = np.random.default_rng(seed)
    obs = float(np.mean(hi) - np.mean(lo))
    pooled = np.concatenate([hi, lo])
    perms = np.empty(n_perm)
    for b in range(n_perm):
        perm = rng.permutation(pooled)
        perms[b] = perm[: len(hi)].mean() - perm[len(hi):].mean()
    p_perm = float(min(1.0, 2.0 * min(float((perms >= obs).mean()), float((perms <= obs).mean()))))
    diffs = rng.choice(hi, size=(n_boot, len(hi)), replace=True).mean(axis=1) - rng.choice(
        lo, size=(n_boot, len(lo)), replace=True).mean(axis=1)
    return {
        "p_value": p_perm,
        "mean_return_ci95_low": float(np.quantile(diffs, .025)),
        "mean_return_ci95_high": float(np.quantile(diffs, .975)),
    }


def benjamini_hochberg(p_values: list[float | None]) -> list[float | None]:
    """Benjamini-Hochberg FDR correction, preserving None positions."""
    valid = [(i, float(p)) for i, p in enumerate(p_values) if p is not None and np.isfinite(p)]
    out: list[float | None] = [None] * len(p_values)
    if not valid:
        return out
    ordered = sorted(valid, key=lambda x: x[1])
    m = len(ordered)
    q_sorted = [0.0] * m
    running = 1.0
    for rank_from_end in range(m - 1, -1, -1):
        _, p = ordered[rank_from_end]
        rank = rank_from_end + 1
        running = min(running, p * m / rank)
        q_sorted[rank_from_end] = min(1.0, running)
    for (idx, _), q in zip(ordered, q_sorted, strict=True):
        out[idx] = float(q)
    return out


def _label_at(store: Store, symbol: str, ts, version: str = "v4"):
    """Label anchored at ``ts``; build+persist on miss so studies anchored at
    actionable timestamps don't silently drop rows maintenance hasn't reached."""
    lab = store.label_for(symbol, ts, version)
    if lab is None:
        lab = build_labels(store, symbol, ts)
        if lab is not None:
            store.save_label(lab)
    return lab


def post_event_frame(store: Store, horizon: str = "15m", semantic_model: str | None = None) -> pd.DataFrame:
    """Legacy post-level event frame (diagnostic only — duplicates inflate N).

    Rows are keyed by (post_id, asset symbol): a pair-trade post yields one row
    per asset with THAT asset's semantics, anchored at the actionable timestamp
    (max of first_seen and classification completion) rather than created_at.
    """
    rows=[]
    for p in store.all_posts():
        for symbol in store.post_asset_symbols(p.post_id) or [p.symbol]:
            sems=store.semantics_for_post(p.post_id,semantic_model,symbol)
            if not sems: continue
            s=sems[-1]
            anchor=post_actionable_ts(store,p,symbol,model=semantic_model)
            lab=_label_at(store,symbol,anchor)
            if not lab: continue
            ret=lab.abnormal_returns.get(horizon)
            if ret is None: continue
            direction=DIRECTION.get(s.trade_intent,0)
            rows.append({
                'post_id':p.post_id,'symbol':symbol,'ts':anchor,'author':p.author_username,
                'seen_lag_s':max(0.0,((p.first_seen_at or p.created_at)-p.created_at).total_seconds()),
                'communication_type':s.communication_type,'trade_intent':s.trade_intent,'catalyst_type':s.catalyst_type,'source_role':s.source_role,'claim_status':s.claim_status,
                'new_information_prob':s.new_information_prob,'reactive_prob':s.reactive_to_price_prob,
                'explicit_prob':s.explicit_recommendation_prob,'promo_prob':s.promotional_prob,
                'impact':s.market_impact_score,'return':ret,'signed_return':direction*ret if direction else np.nan,
                'max_up':lab.max_up.get(horizon) if hasattr(lab,'max_up') else None,
                'max_down':lab.max_down.get(horizon) if hasattr(lab,'max_down') else None,
            })
    return pd.DataFrame(rows)


def narrative_event_frame(store: Store, horizon: str = "15m", semantic_model: str | None = None) -> pd.DataFrame:
    """One row per stored narrative event.

    Semantics are aggregated across member posts using (post_id, event.symbol)
    lookups — a BTC/ETH pair-trade post contributes its ETH leg to an ETH event,
    never the BTC interpretation. The outcome is anchored to the event's
    actionable time (earliest member observation, pushed past classification
    completion), so returns before anyone could have traded never enter.
    """
    if horizon not in HORIZONS:
        raise ValueError(f"unknown horizon: {horizon}")
    posts = {p.post_id: p for p in store.all_posts()}
    rows: list[dict] = []
    for payload in store.all_events():
        event = EventCluster.model_validate(payload)
        members = [
            p for pid in event.post_ids if (p := posts.get(pid)) is not None
            and event.symbol in (store.post_asset_symbols(pid) or [p.symbol])
        ]
        if not members:
            continue
        members.sort(key=lambda p: (p.first_seen_at or p.created_at))
        anchor = event.actionable_at or (members[0].first_seen_at or members[0].created_at)
        sems = []
        for p in members:
            found = store.semantics_for_post(p.post_id, semantic_model, event.symbol)
            if found:
                sems.append(found[-1])
        if not sems:
            continue
        # The signal is tradable only once the member semantics exist.
        classified = [s.classified_at for s in sems if getattr(s, "classified_at", None)]
        if classified:
            anchor = max(anchor, max(classified))
        lab = _label_at(store, event.symbol, anchor)
        if not lab:
            continue
        ret = lab.abnormal_returns.get(horizon)
        if ret is None:
            continue
        direction_score = float(np.mean([DIRECTION.get(s.trade_intent, 0) for s in sems]))
        source_roles = [s.source_role for s in sems]
        claim_statuses = [s.claim_status for s in sems]
        communication_types = [s.communication_type for s in sems]
        catalyst_types = [s.catalyst_type for s in sems]
        trade_intents = [s.trade_intent for s in sems]
        rows.append({
            "event_id": event.event_id,
            "symbol": event.symbol,
            "ts": anchor,
            "last_seen_at": event.last_seen_at,
            "n_posts": len(members),
            "unique_authors": event.unique_authors,
            "source_domains": event.source_domains,
            "communication_type": _mode(communication_types, "OTHER"),
            "trade_intent": _mode(trade_intents, "NEUTRAL"),
            "catalyst_type": _mode(catalyst_types, "NONE"),
            "source_role": _mode(source_roles, "UNKNOWN"),
            "claim_status": _mode(claim_statuses, "NONE"),
            "new_information_prob": float(np.mean([s.new_information_prob for s in sems])),
            "reactive_prob": float(np.mean([s.reactive_to_price_prob for s in sems])),
            "explicit_prob": float(np.mean([s.explicit_recommendation_prob for s in sems])),
            "promo_prob": float(np.mean([s.promotional_prob for s in sems])),
            "evidence_prob": float(np.mean([s.evidence_prob for s in sems])),
            "impact": float(np.mean([s.market_impact_score for s in sems])),
            "semantic_confidence": float(np.mean([s.confidence for s in sems])),
            "direction_score": direction_score,
            "return": float(ret),
            "signed_return": float(direction_score * ret) if direction_score else np.nan,
            "max_up": lab.max_up.get(horizon),
            "max_down": lab.max_down.get(horizon),
        })
    return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True) if rows else pd.DataFrame()


def _bucket_row(dimension: str, bucket: str, g: pd.DataFrame, *, horizon: str, seed: int) -> dict:
    r = g["return"].astype(float).to_numpy()
    sr = g["signed_return"].dropna().astype(float).to_numpy()
    std = float(np.std(r, ddof=1)) if len(r) > 1 else np.nan
    se = std / np.sqrt(len(r)) if len(r) > 1 and std > 0 else np.nan
    t_stat = float(np.mean(r) / se) if np.isfinite(se) and se > 0 else None
    keys = _cluster_keys(g) if {"symbol", "ts"} <= set(g.columns) else None
    ci_low, ci_high = _bootstrap_ci(r, seed=seed, cluster_keys=keys)
    return {
        "horizon": horizon,
        "dimension": dimension,
        "bucket": str(bucket),
        "n_events": int(len(g)),
        "n_posts": int(g["n_posts"].sum()) if "n_posts" in g else int(len(g)),
        "mean_return": float(np.mean(r)),
        "median_return": float(np.median(r)),
        "mean_return_ci95_low": ci_low,
        "mean_return_ci95_high": ci_high,
        "t_stat": t_stat,
        "p_value": _bootstrap_mean_p(r, seed=seed + 1, cluster_keys=keys),
        "p_value_normal": _normal_p_from_t(t_stat),
        "positive_rate": float(np.mean(r > 0)),
        "mean_signed_return": float(np.mean(sr)) if len(sr) else None,
        "mean_max_up": float(g["max_up"].dropna().mean()) if "max_up" in g and g["max_up"].notna().any() else None,
        "mean_max_down": float(g["max_down"].dropna().mean()) if "max_down" in g and g["max_down"].notna().any() else None,
        "unique_symbols": int(g["symbol"].nunique()) if "symbol" in g else None,
        "cluster_unit": "(symbol,day)" if keys else "row",
    }


def event_study(
    store: Store,
    horizon: str = "15m",
    semantic_model: str | None = None,
    min_n: int = 5,
    *,
    inferential_unit: str = "event",
) -> pd.DataFrame:
    """Semantic event study with event de-duplication and FDR control.

    `inferential_unit="event"` is the research default. `post` is retained only for
    diagnostics/backward compatibility and should not be used for headline claims.
    """
    d = narrative_event_frame(store, horizon, semantic_model) if inferential_unit == "event" else post_event_frame(store, horizon, semantic_model)
    if d.empty:
        return d
    # Normalize column naming for post-level fallback.
    if "n_posts" not in d:
        d = d.copy(); d["n_posts"] = 1
    blocks: list[dict] = []
    for col in ["communication_type", "trade_intent", "catalyst_type", "source_role", "claim_status"]:
        for key, g in d.groupby(col):
            if len(g) < min_n:
                continue
            blocks.append(_bucket_row(col, str(key), g, horizon=horizon, seed=_stable_seed(horizon, col, key)))

    # Explicit high-vs-low contrasts. Use quartile tails rather than comparing the
    # upper quartile with the other 75%, which tends to dilute effects.
    for col in ["new_information_prob", "explicit_prob", "reactive_prob", "promo_prob", "impact"]:
        if col not in d:
            continue
        q25 = float(d[col].quantile(.25)); q75 = float(d[col].quantile(.75))
        hi = d[d[col] >= q75]; lo = d[d[col] <= q25]
        if len(hi) < min_n or len(lo) < min_n:
            continue
        row = _bucket_row(col, "high_minus_low", hi, horizon=horizon, seed=_stable_seed(horizon, col, "contrast"))
        two = _two_sample_test(
            hi["return"].astype(float).to_numpy(), lo["return"].astype(float).to_numpy(),
            seed=_stable_seed(horizon, col, "contrast2"),
        )
        row["p_value"] = two["p_value"]
        row["mean_return_ci95_low"] = two["mean_return_ci95_low"]
        row["mean_return_ci95_high"] = two["mean_return_ci95_high"]
        row["high_n_events"] = int(len(hi)); row["low_n_events"] = int(len(lo))
        row["high_mean_return"] = float(hi["return"].mean()); row["low_mean_return"] = float(lo["return"].mean())
        row["mean_return"] = float(hi["return"].mean() - lo["return"].mean())
        row["effect_definition"] = "upper_quartile_minus_lower_quartile"
        blocks.append(row)

    out = pd.DataFrame(blocks)
    if out.empty:
        return out
    out["q_value_fdr"] = benjamini_hochberg(out["p_value"].tolist())
    out["fdr_10pct"] = out["q_value_fdr"].apply(lambda x: bool(x <= .10) if pd.notna(x) else False)
    out["inferential_unit"] = inferential_unit
    return out.sort_values(["q_value_fdr", "dimension", "n_events"], ascending=[True, True, False], na_position="last").reset_index(drop=True)


def event_study_grid(
    store: Store,
    *,
    horizons: list[str] | None = None,
    semantic_model: str | None = None,
    min_n: int = 5,
) -> pd.DataFrame:
    """Run event-level studies across horizons and control FDR globally."""
    horizons = horizons or list(HORIZONS)
    frames = [event_study(store, h, semantic_model, min_n, inferential_unit="event") for h in horizons]
    frames = [x for x in frames if not x.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # Recompute BH globally across all semantic hypotheses *and* horizons.
    out["q_value_fdr_global"] = benjamini_hochberg(out["p_value"].tolist())
    out["fdr_global_10pct"] = out["q_value_fdr_global"].apply(lambda x: bool(x <= .10) if pd.notna(x) else False)
    return out.sort_values(["q_value_fdr_global", "horizon", "dimension"], na_position="last").reset_index(drop=True)
