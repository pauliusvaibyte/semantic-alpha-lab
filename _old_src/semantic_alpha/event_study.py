from __future__ import annotations

import hashlib
from statistics import NormalDist

import numpy as np
import pandas as pd

from .author_alpha import DIRECTION
from .labels import HORIZONS
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


def _bootstrap_ci(values: np.ndarray, *, seed: int, samples: int = 500) -> tuple[float | None, float | None]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (None, None)
    # For large event sets the CLT interval is both stable and much cheaper than
    # repeatedly resampling tens of thousands of events across many hypotheses.
    if len(x) >= 1000:
        se=float(np.std(x,ddof=1)/np.sqrt(len(x)))
        mean=float(np.mean(x)); return (mean-1.96*se, mean+1.96*se)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for i in range(samples):
        means[i] = float(np.mean(rng.choice(x, size=len(x), replace=True)))
    return (float(np.quantile(means, .025)), float(np.quantile(means, .975)))


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
    for (idx, _), q in zip(ordered, q_sorted):
        out[idx] = float(q)
    return out


def post_event_frame(store: Store, horizon: str = "15m", semantic_model: str | None = None) -> pd.DataFrame:
    """Legacy post-level event frame.

    Useful for debugging, but not the default inferential unit because duplicate/copy
    posts can make one narrative look like many independent observations.
    """
    rows=[]
    for p in store.all_posts():
        sems=store.semantics_for_post(p.post_id,semantic_model,p.symbol)
        if not sems: continue
        s=sems[-1]
        lab=store.label_for(p.symbol,p.created_at)
        if not lab: continue
        ret=lab.abnormal_returns.get(horizon)
        if ret is None: continue
        direction=DIRECTION.get(s.trade_intent,0)
        rows.append({
            'post_id':p.post_id,'symbol':p.symbol,'ts':p.created_at,'author':p.author_username,
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

    Semantics are aggregated across member posts, while the outcome is anchored to
    the first observed post in the cluster. This prevents copied/reposted narratives
    from inflating sample size and apparent statistical significance.
    """
    if horizon not in HORIZONS:
        raise ValueError(f"unknown horizon: {horizon}")
    posts = {p.post_id: p for p in store.all_posts()}
    rows: list[dict] = []
    for payload in store.all_events():
        event = EventCluster.model_validate(payload)
        members = [posts[pid] for pid in event.post_ids if pid in posts]
        if not members:
            continue
        members.sort(key=lambda p: p.created_at)
        anchor = members[0]
        lab = store.label_for(event.symbol, anchor.created_at)
        if not lab:
            continue
        ret = lab.abnormal_returns.get(horizon)
        if ret is None:
            continue
        sems = []
        for p in members:
            found = store.semantics_for_post(p.post_id, semantic_model, p.symbol)
            if found:
                sems.append(found[-1])
        if not sems:
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
            "ts": anchor.created_at,
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
    ci_low, ci_high = _bootstrap_ci(r, seed=seed)
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
        "p_value": _normal_p_from_t(t_stat),
        "positive_rate": float(np.mean(r > 0)),
        "mean_signed_return": float(np.mean(sr)) if len(sr) else None,
        "mean_max_up": float(g["max_up"].dropna().mean()) if "max_up" in g and g["max_up"].notna().any() else None,
        "mean_max_down": float(g["max_down"].dropna().mean()) if "max_down" in g and g["max_down"].notna().any() else None,
        "unique_symbols": int(g["symbol"].nunique()) if "symbol" in g else None,
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
        diff = hi["return"].astype(float).to_numpy() - np.mean(lo["return"].astype(float).to_numpy())
        row = _bucket_row(col, "high_minus_low", hi.assign(**{"return": diff}), horizon=horizon, seed=_stable_seed(horizon, col, "contrast"))
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
