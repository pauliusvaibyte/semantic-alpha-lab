from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from .labels import HORIZONS, post_actionable_ts
from .storage import Store

DIRECTION = {"STRONG_LONG":1, "LONG":1, "SHORT":-1, "STRONG_SHORT":-1, "REDUCE_LONG":-1}


@dataclass
class AuthorAlpha:
    author: str
    symbol: str
    horizon: str
    samples: int
    mean_signed_return: float
    hit_rate: float
    posterior_edge: float


def estimate_author_alpha(
    store: Store, author: str, symbol: str, horizon: str = "15m",
    before: datetime | None = None, semantic_model: str | None = None,
    prior_strength: int = 20, prior_mean: float = 0.0,
    label_version: str = "v4",
) -> AuthorAlpha:
    """Forward-only author edge estimate.

    When `before` is supplied, a historical call is usable only if its *entire*
    outcome horizon had elapsed before `before`. This prevents subtle reputation
    leakage from a call made 2 minutes ago using a 15-minute future label.
    """
    if horizon not in HORIZONS:
        raise ValueError(f"unknown horizon: {horizon}")
    maturity = timedelta(minutes=HORIZONS[horizon])
    signed = []
    hits = []
    for p in store.posts_for_author(author, symbol, before):
        # The outcome is anchored at the actionable timestamp — a call seen hours
        # late cannot earn returns that accrued before we observed it.
        anchor = post_actionable_ts(store, p, symbol, model=semantic_model)
        if before is not None and anchor + maturity > before:
            continue
        sems = store.semantics_for_post(p.post_id, semantic_model, symbol)
        if not sems:
            continue
        s = sems[-1]
        d = DIRECTION.get(s.trade_intent, 0)
        if d == 0 or s.explicit_recommendation_prob < 0.5:
            continue
        label = store.label_for(symbol, anchor, version=label_version)
        if not label:
            continue
        r = label.abnormal_returns.get(horizon)
        if r is None:
            continue
        sr = d * r
        signed.append(sr)
        hits.append(sr > 0)
    n = len(signed)
    raw_mean = float(np.mean(signed)) if signed else 0.0
    posterior = (raw_mean*n + prior_mean*prior_strength) / (n+prior_strength)
    return AuthorAlpha(
        author=author, symbol=symbol, horizon=horizon, samples=n,
        mean_signed_return=raw_mean, hit_rate=float(np.mean(hits)) if hits else 0.0,
        posterior_edge=float(posterior),
    )


def snapshot_author_reputations(
    store: Store,
    as_of: datetime,
    *,
    horizon: str = "15m",
    semantic_model: str | None = None,
    min_posts: int = 1,
) -> list:
    """Materialize point-in-time author reputations for cheap live feature lookup."""
    from .schema import AuthorReputationRecord

    pairs = sorted({(p.author_username, p.symbol) for p in store.all_posts() if p.author_username and p.created_at < as_of})
    out = []
    for author, symbol in pairs:
        a = estimate_author_alpha(store, author, symbol, horizon=horizon, before=as_of, semantic_model=semantic_model)
        if a.samples < min_posts:
            continue
        r = AuthorReputationRecord(
            author=author, symbol=symbol, horizon=horizon, as_of=as_of,
            samples=a.samples, hit_rate=a.hit_rate, posterior_edge=a.posterior_edge,
        )
        store.save_author_reputation(r)
        out.append(r)
    return out
