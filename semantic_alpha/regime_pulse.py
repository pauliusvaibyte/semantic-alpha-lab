"""Regime pulse: one holistic Jev `system_one` call per symbol per maintenance
cycle over a fixed digest of market state + aggregate social stats + a few
representative posts.

Per-post classification answers "what is this post"; this answers "what is the
current state" — a different measurement axis (joint reading vs summed micro-
readings) at ~1/1000th the cost per observation. The `trade_action` choice is
recorded as a falsifiable directional call with a timestamp.

Research discipline: the digest spec and question set are fixed (pulse v1) and
deliberately not wired into feature versions. This is a raw sensor channel —
a registered candidate that must validate on untouched data before any
feature may depend on it.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from .config import settings
from .storage import Store

PULSE_VERSION = "v1"


def build_digest(store: Store, symbol: str, ts: datetime) -> dict[str, Any]:
    """Fixed-spec digest: latest market state + 15m social aggregates +
    representative posts (top-5 by engagement, latest-5)."""
    market = store.latest_market(symbol)
    market_state: dict[str, Any] = {}
    if market is not None:
        market_state = {
            "price": market.last,
            "funding_rate_pct": market.funding_rate,
            "open_interest_value": market.open_interest_value,
            "volume_24h": market.volume_24h,
            "basis_bps": market.basis,
            "liquidation_long_usd_1m": market.liquidation_long_usd_1m,
            "liquidation_short_usd_1m": market.liquidation_short_usd_1m,
            "snapshot_age_seconds": max(0.0, (ts - market.ts).total_seconds()),
        }
    posts = store.posts_between(symbol, ts - timedelta(minutes=15), ts, as_of=ts)
    authors = {p.author_username for p in posts if p.author_username}
    rep = sorted(posts, key=lambda p: (p.likes or 0) + 2 * (p.reposts or 0), reverse=True)[:5]
    rep += [p for p in posts[-5:] if p.post_id not in {r.post_id for r in rep}]
    return {
        "asset": symbol,
        "ts": ts.isoformat(),
        "market": market_state,
        "social_stats": {
            "posts_15m": len(posts),
            "unique_authors_15m": len(authors),
            "total_likes_15m": sum(p.likes or 0 for p in posts),
            "total_reposts_15m": sum(p.reposts or 0 for p in posts),
        },
        "representative_posts": [
            {"author": p.author_username, "likes": p.likes or 0, "text": (p.text or "")[:280]}
            for p in rep
        ],
    }


async def regime_pulse(store: Store, engine, symbol: str) -> dict[str, Any]:
    """One holistic state reading. Stored raw; usage ledgered like any Jev call."""
    from typesafe_sdk import Choice, Noul, Score

    ts = datetime.now(timezone.utc)
    state = build_digest(store, symbol, ts)
    questions = {
        "market_regime": Choice(
            instructions="What is the current market regime for `asset` given `market` and `social_stats`?",
            criteria={
                "TREND_UP": "Sustained directional advance.",
                "TREND_DOWN": "Sustained directional decline.",
                "RANGE": "Mean-reverting / consolidating.",
                "VOL_EXPANSION": "Volatility expanding, direction unresolved.",
                "QUIET": "Low activity, no regime shift.",
            },
        ),
        "sentiment_spectrum": Score(
            instructions="Rate prevailing social mood across `social_stats` and `representative_posts`.",
            criteria=["Extreme Panic / Capitulation", "Bearish", "Neutral / Mixed", "Bullish", "Euphoric / Greedy"],
        ),
        "squeeze_risk": Noul(
            instructions="Does funding/positioning in `market` clash with social mood in a way that suggests a squeeze risk?"
        ),
        "catalyst_significance": Score(
            instructions="Rate the significance of any events described in `representative_posts`.",
            criteria=["No news or pure noise", "Minor routine update or rumor", "Moderate catalyst", "Major market-shifting catalyst"],
        ),
        "trade_action": Choice(
            instructions="Best immediate action for `asset` given the whole state. Recorded as a falsifiable call, not executed.",
            criteria={
                "STRONG_BUY": "High-conviction long.", "BUY": "Favorable long.",
                "HOLD": "No clear edge.", "TAKE_PROFIT": "Secure gains.",
                "SELL": "Bearish continuation risk.", "STRONG_SELL": "High-conviction short.",
            },
        ),
    }
    client = await engine._get_client()
    requested = getattr(engine, "model", "jev-latest")
    try:
        response = await client.system_one(state=state, questions=questions, model=requested)
    except TypeError:
        response = await client.system_one(state=state, questions=questions)
    effective = str(getattr(response, "model", None) or getattr(response, "model_used", None) or requested)

    answers = response.answers
    out = {
        "ts": ts.isoformat(), "symbol": symbol, "pulse_version": PULSE_VERSION,
        "requested_model": requested, "effective_model": effective,
        "market_regime": getattr(answers.get("market_regime"), "choice", None),
        "sentiment_score": getattr(answers.get("sentiment_spectrum"), "score", None),
        "squeeze_risk_prob": getattr(answers.get("squeeze_risk"), "noul", None),
        "catalyst_significance": getattr(answers.get("catalyst_significance"), "score", None),
        "trade_action": getattr(answers.get("trade_action"), "choice", None),
        "trade_action_probs": getattr(answers.get("trade_action"), "probabilities", None),
    }
    est_tokens = max(1, int((len(json.dumps(state, default=str)) + 2200) / 4))
    store.save_regime_pulse(symbol, ts, effective, {
        **out, "digest": state,
        "_usage": {
            "provider": "typesafe", "category": "regime_pulse",
            "reference_id": f"regime_pulse:{symbol}:{int(ts.timestamp())}:{PULSE_VERSION}",
            "units": float(est_tokens), "unit_name": "input_token",
            "estimated_usd": est_tokens * settings.jev_input_usd_per_m / 1_000_000,
            "cost_source": "estimated_tokens_configured_rate",
        },
    })
    return out
