from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SocialPost(BaseModel):
    platform: str = "x"
    post_id: str
    symbol: str
    author_id: str | None = None
    author_username: str | None = None
    text: str
    created_at: datetime
    first_seen_at: datetime = Field(default_factory=utcnow)
    likes: int | None = None
    reposts: int | None = None
    replies: int | None = None
    followers: int | None = None
    verified: bool | None = None
    urls: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class ApiUsageRecord(BaseModel):
    provider: str
    category: str
    reference_id: str
    ts: datetime = Field(default_factory=utcnow)
    units: float = 0.0
    unit_name: str = "unit"
    estimated_usd: float = 0.0
    cost_source: str = "estimated"
    metadata: dict[str, Any] = Field(default_factory=dict)


class EngagementSnapshot(BaseModel):
    post_id: str
    symbol: str
    observed_at: datetime = Field(default_factory=utcnow)
    likes: int = 0
    reposts: int = 0
    replies: int = 0


class PostSemantics(BaseModel):
    post_id: str
    symbol: str
    model: str
    question_version: str = "v1"
    classified_at: datetime = Field(default_factory=utcnow)

    relevance: Literal["RELEVANT", "MAYBE", "IRRELEVANT"] = "MAYBE"
    communication_type: str = "OTHER"
    trade_intent: str = "NEUTRAL"
    catalyst_type: str = "NONE"
    horizon: str = "NONE"
    source_role: str = "UNKNOWN"
    claim_status: str = "NONE"

    new_information_prob: float = 0.0
    reactive_to_price_prob: float = 0.0
    original_information_prob: float = 0.0
    explicit_recommendation_prob: float = 0.0
    evidence_prob: float = 0.0
    promotional_prob: float = 0.0
    coordinated_shill_prob: float = 0.0
    market_impact_score: float = 0.0
    confidence: float = 0.0

    probabilities: dict[str, dict[str, float]] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)


class MarketSnapshot(BaseModel):
    source: str = "bybit"
    symbol: str
    ts: datetime = Field(default_factory=utcnow)

    last: float
    bid: float | None = None
    ask: float | None = None
    mark: float | None = None
    index: float | None = None

    funding_rate: float | None = None
    open_interest_qty: float | None = None
    open_interest_value: float | None = None
    volume_24h: float | None = None
    turnover_24h: float | None = None
    basis: float | None = None

    spread_bps: float | None = None
    book_imbalance: float | None = None
    depth_bid_10bps: float | None = None
    depth_ask_10bps: float | None = None
    taker_buy_ratio: float | None = None
    trade_notional_1m: float | None = None
    signed_trade_notional_1m: float | None = None
    liquidation_long_usd_1m: float | None = None
    liquidation_short_usd_1m: float | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class FeatureSnapshot(BaseModel):
    symbol: str
    ts: datetime
    feature_version: str = "v4"
    values: dict[str, float]


class LabelRecord(BaseModel):
    symbol: str
    ts: datetime
    label_version: str = "v4"
    returns: dict[str, float | None]
    abnormal_returns: dict[str, float | None] = Field(default_factory=dict)
    benchmark_symbol: str = "BTCUSDT"
    benchmark_beta: float = 1.0
    # Actual entry snapshot the returns are measured from. Horizons are anchored
    # to base_ts (not ts), so a delayed first tick can never silently shorten the
    # realized holding period.
    base_ts: datetime | None = None
    entry_delay_seconds: float | None = None
    realized_minutes: dict[str, float | None] = Field(default_factory=dict)
    max_up: dict[str, float | None] = Field(default_factory=dict)
    max_down: dict[str, float | None] = Field(default_factory=dict)
    # v4 secondary target heads: magnitude / volatility / crowding-reversal.
    abs_move: dict[str, float | None] = Field(default_factory=dict)
    range_move: dict[str, float | None] = Field(default_factory=dict)
    fade_ratio: dict[str, float | None] = Field(default_factory=dict)


class AuthorReputationRecord(BaseModel):
    author: str
    symbol: str
    horizon: str
    as_of: datetime
    samples: int
    hit_rate: float
    posterior_edge: float


class PaperTrade(BaseModel):
    trade_id: str
    symbol: str
    side: Literal["LONG", "SHORT"]
    opened_at: datetime
    entry_price: float
    notional_usd: float
    entry_cost_bps: float = 0.0
    model_version: str
    model_fingerprint: str | None = None
    horizon_minutes: int
    p_baseline_up: float
    p_full_up: float
    semantic_edge_logodds: float
    signal_strength: float = 0.0
    entry_regime: str | None = None
    entry_feature_ts: datetime | None = None
    entry_context: dict[str, float] = Field(default_factory=dict)
    status: Literal["OPEN", "CLOSED"] = "OPEN"
    closed_at: datetime | None = None
    exit_price: float | None = None
    exit_cost_bps: float | None = None
    realized_return: float | None = None
    close_reason: str | None = None


class EventCluster(BaseModel):
    event_id: str
    symbol: str
    started_at: datetime
    last_seen_at: datetime
    representative_text: str
    post_ids: list[str]
    unique_authors: int
    source_domains: int = 0
    similarity_mean: float = 0.0
    # Earliest time any member was observable (first_seen_at): the tradable
    # anchor, as opposed to started_at which is narrative creation time.
    actionable_at: datetime | None = None
