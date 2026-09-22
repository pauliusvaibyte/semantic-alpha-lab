"""Correctness-release tests: every audit finding gets a regression test.

Covered: time travel (social availability), candle-close availability, partial
label maturity + refill, market idempotency/source isolation, WS stale gating,
provider retries, t.co clustering, persistent event lifecycle, cross-asset
(post_id, symbol) semantics, model/report identity binding, holdout identity,
round-trip costs, portfolio-native accounting, campaign data-gate hard-stop.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import httpx
import numpy as np
import pandas as pd
import pytest

from semantic_alpha.audit import leakage_audit
from semantic_alpha.backtest import portfolio_metrics, stateful_trade_vectors
from semantic_alpha.campaign import run_campaign
from semantic_alpha.clustering import EventClusterer, assign_posts_to_events
from semantic_alpha.event_study import narrative_event_frame
from semantic_alpha.features import FeatureEngine
from semantic_alpha.historical_market import ingest_bybit_history
from semantic_alpha.labels import build_labels
from semantic_alpha.maintenance import maintenance_step
from semantic_alpha.manifest import holdout_identity
from semantic_alpha.models import SemanticResidualModel
from semantic_alpha.protocol import run_locked_protocol
from semantic_alpha.providers.bybit import BybitPublicClient
from semantic_alpha.providers.bybit_ws import MicrostructureAccumulator
from semantic_alpha.providers.xapi import _get_with_backoff
from semantic_alpha.quality import feed_health_report
from semantic_alpha.registry import promote_candidate
from semantic_alpha.schema import (
    EventCluster, FeatureSnapshot, LabelRecord, MarketSnapshot, PostSemantics, SocialPost,
)
from semantic_alpha.storage import Store

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _post(pid: str, symbol: str = "SOLUSDT", created=None, seen=None, text: str = "test post", author: str = "u1") -> SocialPost:
    import re
    created = created or T0
    return SocialPost(
        post_id=pid, symbol=symbol, author_username=author, text=text,
        created_at=created, first_seen_at=seen or created,
        urls=re.findall(r"https?://\S+", text),
    )


def _mkt(symbol: str, ts: datetime, last: float = 100.0, source: str = "bybit") -> MarketSnapshot:
    return MarketSnapshot(
        source=source, symbol=symbol, ts=ts, last=last, bid=last - 0.01, ask=last + 0.01,
        spread_bps=2.0, depth_bid_10bps=50_000.0, depth_ask_10bps=50_000.0,
        funding_rate=0.0001, open_interest_value=1_000_000.0,
    )


# ---------- TIME: social availability ----------

def test_social_reads_respect_as_of(tmp_path):
    """The audit's reproduced case: created 10:00, seen 13:00, classified 13:01.
    A 10:05 feature must not contain this post — and now doesn't."""
    s = Store(str(tmp_path / "t.db"))
    created = T0.replace(hour=10, minute=0)
    seen = T0.replace(hour=13, minute=0)
    classified = T0.replace(hour=13, minute=1)
    s.save_post(_post("p1", created=created, seen=seen))
    s.save_semantics(PostSemantics(
        post_id="p1", symbol="SOLUSDT", model="m", classified_at=classified,
        trade_intent="LONG", new_information_prob=0.9, market_impact_score=3.0,
    ))
    feat_ts = T0.replace(hour=10, minute=5)
    assert s.posts_between("SOLUSDT", feat_ts - timedelta(minutes=15), feat_ts)  # raw window sees it
    assert not s.posts_between("SOLUSDT", feat_ts - timedelta(minutes=15), feat_ts, as_of=feat_ts)
    assert s.posts_between("SOLUSDT", feat_ts - timedelta(minutes=15), feat_ts, as_of=T0.replace(hour=13, minute=5))
    assert not s.semantics_between("SOLUSDT", feat_ts - timedelta(minutes=15), feat_ts, "m", as_of=feat_ts)
    assert s.semantics_between("SOLUSDT", feat_ts - timedelta(minutes=15), feat_ts, "m", as_of=T0.replace(hour=13, minute=5))
    f = FeatureEngine(s, "m").build("SOLUSDT", feat_ts, _mkt("SOLUSDT", feat_ts))
    assert f.values["posts_15m"] == 0.0
    assert f.values["intent_sum_15m"] == 0.0


def test_semantic_features_are_relevance_weighted(tmp_path):
    """Jev's own relevance verdict gates feature contribution: an IRRELEVANT
    post (e.g. Solana-chain memecoin spam that mentions 'Solana') adds zero to
    semantic rates — unweighted means let the spam flood pose as signal."""
    s = Store(str(tmp_path / "t.db"))
    s.save_post(_post("good", created=T0, seen=T0))
    s.save_post(_post("spam", created=T0, seen=T0))
    s.save_post(_post("mid", created=T0, seen=T0))
    s.save_semantics(PostSemantics(post_id="good", symbol="SOLUSDT", model="m", classified_at=T0,
        relevance="RELEVANT", trade_intent="LONG", new_information_prob=0.8, coordinated_shill_prob=0.1))
    s.save_semantics(PostSemantics(post_id="spam", symbol="SOLUSDT", model="m", classified_at=T0,
        relevance="IRRELEVANT", trade_intent="STRONG_LONG", new_information_prob=0.9, coordinated_shill_prob=0.95))
    s.save_semantics(PostSemantics(post_id="mid", symbol="SOLUSDT", model="m", classified_at=T0,
        relevance="MAYBE", trade_intent="SHORT", new_information_prob=0.4, coordinated_shill_prob=0.2))
    f = FeatureEngine(s, "m").build("SOLUSDT", T0 + timedelta(minutes=5), _mkt("SOLUSDT", T0 + timedelta(minutes=5)))
    v = f.values
    # weights: 1.0 (good) + 0 (spam) + 0.5 (mid) = 1.5 of 3 classified posts
    assert v["rel_posts_15m"] == pytest.approx(1.5)
    assert v["rel_share_15m"] == pytest.approx(0.5)
    # weighted means: spam's extreme values contribute nothing
    assert v["new_info_rate_15m"] == pytest.approx((0.8 + 0.5 * 0.4) / 1.5)
    assert v["shill_rate_15m"] == pytest.approx((0.1 + 0.5 * 0.2) / 1.5)
    # weighted intent: (1.0*1.0 + 0*2.0 + 0.5*-1.0) = 0.5
    assert v["intent_sum_15m"] == pytest.approx(0.5)


def test_author_features_ignore_future_classifications(tmp_path):
    """_author_features reads per-post semantics — without a classified_at<=ts
    gate, a historical rebuild would see classifications made AFTER the feature
    timestamp (post stored 10:00, classified 13:00, feature at 10:05 must not
    use that intent). Reputation exists; only the semantics are late."""
    s = Store(str(tmp_path / "t.db"))
    s.save_post(_post("p1", created=T0, seen=T0, author="alice"))
    s.save_semantics(PostSemantics(
        post_id="p1", symbol="SOLUSDT", model="m",
        classified_at=T0 + timedelta(hours=3),  # classified AFTER feature ts
        trade_intent="LONG", explicit_recommendation_prob=0.9,
    ))
    from semantic_alpha.schema import AuthorReputationRecord
    s.save_author_reputation(AuthorReputationRecord(
        author="alice", symbol="SOLUSDT", horizon="15m", as_of=T0,
        samples=10, hit_rate=0.8, posterior_edge=0.05,
    ))
    f = FeatureEngine(s, "m").build("SOLUSDT", T0 + timedelta(minutes=3), _mkt("SOLUSDT", T0 + timedelta(minutes=3)))
    assert f.values["author_alpha_signal_5m"] == 0.0


def test_leakage_audit_catches_social_lookahead(tmp_path):
    """A feature row claiming social content before availability is flagged."""
    s = Store(str(tmp_path / "t.db"))
    created = T0.replace(hour=10, minute=0)
    seen = T0.replace(hour=13, minute=0)
    s.save_post(_post("p1", created=created, seen=seen))
    feat_ts = T0.replace(hour=10, minute=5)
    # Forged dirty feature: counts a post that wasn't observable at feat_ts.
    s.save_features(FeatureSnapshot(
        symbol="SOLUSDT", ts=feat_ts, feature_version="v4",
        values={"posts_15m": 1.0, "intent_sum_5m": 1.0, "new_info_rate_5m": 0.9},
    ))
    r = leakage_audit(s)
    assert not r["ok"]
    assert any(i["type"] == "feature_social_lookahead" for i in r["issues"])


def test_leakage_audit_passes_clean_feature(tmp_path):
    s = Store(str(tmp_path / "t.db"))
    s.save_post(_post("p1", created=T0, seen=T0))
    s.save_semantics(PostSemantics(post_id="p1", symbol="SOLUSDT", model="m", classified_at=T0, trade_intent="LONG"))
    f = FeatureEngine(s, "m").build("SOLUSDT", T0 + timedelta(minutes=5), _mkt("SOLUSDT", T0 + timedelta(minutes=5)))
    s.save_features(f)
    assert leakage_audit(s)["ok"]


# ---------- TIME: candle-close availability ----------

def test_candle_close_timestamped_at_completion(tmp_path):
    """Bybit kline r[0]=open, r[4]=close — the close is only knowable at candle
    end, so availability ts = open + interval and the raw open ts is preserved."""
    s = Store(str(tmp_path / "t.db"))
    open_ms = int(T0.timestamp() * 1000)
    bars = [[str(open_ms + i * 60_000), "100", "101", "99", "100.5", "10", "1000"] for i in range(3)]

    class FakeBybit:
        async def klines(self, symbol, start, end, interval, limit):
            return bars

        async def open_interest_window(self, symbol, start, end, interval, limit):
            return []

        async def funding_window(self, symbol, start, end, limit):
            return []

    r = asyncio.run(ingest_bybit_history(s, FakeBybit(), "SOLUSDT", T0, T0 + timedelta(minutes=3), 1, sleep_seconds=0))
    assert r["stored"] == 3
    snaps = s.market_between("SOLUSDT", T0, T0 + timedelta(minutes=4))
    assert [x.ts for x in snaps] == [T0 + timedelta(minutes=i) for i in (1, 2, 3)]
    assert snaps[0].raw["candle_open_ts"] == T0.isoformat()
    # The candle completing exactly at `end` is kept, not dropped.
    assert snaps[-1].ts == T0 + timedelta(minutes=3)


# ---------- TIME: partial label maturity ----------

def test_partial_labels_mature_and_refill(tmp_path):
    """1m/5m labels exist while 1h is open; a later maintenance pass fills the
    newly matured horizons instead of skipping the row forever."""
    s = Store(str(tmp_path / "t.db"))
    for i in range(0, 6):
        s.save_market(_mkt("SOLUSDT", T0 + timedelta(minutes=i)))
        s.save_market(_mkt("BTCUSDT", T0 + timedelta(minutes=i)))
    l = build_labels(s, "SOLUSDT", T0)
    assert l is not None
    assert l.returns["1m"] is not None and l.returns["5m"] is not None
    assert l.returns["1h"] is None
    s.save_label(l)
    assert any(x.ts == T0 for x in s.immature_labels())
    # Market data arrives through +90min; the row must refill, not stay frozen.
    for i in range(6, 91):
        s.save_market(_mkt("SOLUSDT", T0 + timedelta(minutes=i)))
        s.save_market(_mkt("BTCUSDT", T0 + timedelta(minutes=i)))
    maintenance_step(s)
    refilled = s.label_for("SOLUSDT", T0)
    assert refilled.returns["1h"] is not None
    assert refilled.returns["1m"] == l.returns["1m"]  # realized values preserved


# ---------- DATA: market idempotency + source lineage ----------

def test_market_upsert_idempotent(tmp_path):
    s = Store(str(tmp_path / "t.db"))
    m = _mkt("SOLUSDT", T0)
    s.save_market(m)
    s.save_market(m.model_copy(update={"last": 999.0}))  # same identity, new price
    rows = s.market_between("SOLUSDT", T0 - timedelta(minutes=1), T0 + timedelta(minutes=1))
    assert len(rows) == 1
    assert rows[0].last == 999.0  # upsert wins


def test_market_sources_do_not_collapse(tmp_path):
    s = Store(str(tmp_path / "t.db"))
    s.save_market(_mkt("SOLUSDT", T0, source="bybit"))
    s.save_market(_mkt("SOLUSDT", T0, last=101.0, source="binance"))
    canonical = s.market_between("SOLUSDT", T0 - timedelta(minutes=1), T0 + timedelta(minutes=1))
    assert len(canonical) == 1 and canonical[0].source == "bybit"
    venue = s.market_between("SOLUSDT", T0 - timedelta(minutes=1), T0 + timedelta(minutes=1), source="binance")
    assert len(venue) == 1 and venue[0].last == 101.0


def test_bybit_ws_is_canonical_market_truth(tmp_path):
    """Live WS ticks are Bybit data — excluding them left a live-only database
    with zero market history: returns, realized vol, and forward label outcomes
    all read as missing even though fresh data existed."""
    s = Store(str(tmp_path / "t.db"))
    s.save_market(_mkt("SOLUSDT", T0, last=100.0, source="bybit_ws"))
    s.save_market(_mkt("SOLUSDT", T0 + timedelta(minutes=5), last=101.0, source="bybit_ws"))
    xs = s.market_between("SOLUSDT", T0 - timedelta(minutes=1), T0 + timedelta(minutes=6))
    assert [x.source for x in xs] == ["bybit_ws", "bybit_ws"]
    prev = s.nearest_market_at_or_before("SOLUSDT", T0 + timedelta(minutes=5), max_delay_seconds=180)
    assert prev is not None and prev.source == "bybit_ws"
    # Cross-venue isolation is unchanged.
    s.save_market(_mkt("SOLUSDT", T0, last=99.0, source="binance"))
    xs = s.market_between("SOLUSDT", T0 - timedelta(minutes=1), T0 + timedelta(minutes=6))
    assert all(x.source != "binance" for x in xs)


# ---------- DATA: WebSocket freshness ----------

def _ws_msg(topic: str, data, ts_ms: int) -> dict:
    return {"topic": topic, "type": "snapshot", "ts": ts_ms, "data": data}


def test_ws_stale_state_not_fresh(tmp_path):
    """A disconnected feed keeps last state but must fail the freshness gate —
    the audit's 14:00-disconnect-producing-14:04-rows scenario. Freshness ages
    are measured on the local receive clock, so the test uses real now."""
    a = MicrostructureAccumulator("SOLUSDT")
    now = datetime.now(timezone.utc)
    t0_ms = int(now.timestamp() * 1000)
    a.apply(_ws_msg("tickers.SOLUSDT", {"lastPrice": "100", "bid1Price": "99.9", "ask1Price": "100.1"}, t0_ms))
    a.apply(_ws_msg("orderbook.50.SOLUSDT", {"b": [["99.9", "10"]], "a": [["100.1", "10"]]}, t0_ms))
    assert a.is_fresh(now)
    assert not a.is_fresh(now + timedelta(seconds=61))
    a.note_disconnected(now + timedelta(seconds=61))
    f = a.freshness(now + timedelta(seconds=62))
    assert f["connected"] is False
    assert f["ticker_age_s"] == pytest.approx(62.0, abs=2.0)
    assert "tickers" in f["last_exchange_ts"]


def test_feed_health_report_marks_unrecovered_stale(tmp_path):
    s = Store(str(tmp_path / "t.db"))
    s.save_market(_mkt("SOLUSDT", T0))
    s.save_feed_health("SOLUSDT", "stale_state", {"source": "bybit_ws"}, T0)
    # Report window anchored at the event time; symbol has a stale event and no recovery.
    r = feed_health_report(s, hours=24 * 365)
    assert "SOLUSDT" in r["symbols"]
    assert r["symbols"]["SOLUSDT"]["stale_events"] == 1
    assert r["symbols"]["SOLUSDT"]["status"] in {"STALE_UNRECOVERED", "NO_RECENT_MARKET_DATA"}


# ---------- OPERATIONS: provider retries ----------

class _FakeResponse:
    def __init__(self, status: int, payload: dict | None = None, headers: dict | None = None):
        self.status_code = status
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError("err", request=None, response=None)


class _FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    async def get(self, url, params=None, headers=None):
        self.calls += 1
        return self.responses.pop(0)


def test_xapi_retries_429_then_succeeds():
    c = _FakeClient([
        _FakeResponse(429, headers={"Retry-After": "0"}),
        _FakeResponse(200, {"tweets": []}),
    ])
    r = asyncio.run(_get_with_backoff(c, "http://x", params={}, headers={}))
    assert r.status_code == 200 and c.calls == 2


def test_xapi_gives_up_after_bounded_retries():
    c = _FakeClient([_FakeResponse(429, headers={"Retry-After": "0"})] * 10)
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(_get_with_backoff(c, "http://x", params={}, headers={}, attempts=3))
    assert c.calls == 3


def test_bybit_retries_retcode_and_5xx(monkeypatch):
    import httpx
    responses = iter([
        _FakeResponse(500),
        _FakeResponse(200, {"retCode": 10006, "retMsg": "rate limit"}),
        _FakeResponse(200, {"retCode": 0, "result": {"list": [{"x": 1}]}}),
    ])

    class FakeAsyncClient:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, path, params=None):
            return next(responses)

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    client = BybitPublicClient()
    d = asyncio.run(client._get("/v5/market/tickers", {"symbol": "SOLUSDT"}))
    assert d["result"]["list"] == [{"x": 1}]


# ---------- MULTI-ASSET: (post_id, symbol) semantics ----------

def test_pair_trade_post_gets_asset_specific_semantics(tmp_path):
    """BTC LONG / ETH SHORT legs must not cross-contaminate the event frame."""
    s = Store(str(tmp_path / "t.db"))
    p = _post("pair", symbol="BTCUSDT", created=T0, seen=T0, text="long BTC short ETH pair")
    s.save_post(p)
    s.save_post(p.model_copy(update={"symbol": "ETHUSDT"}))  # second asset mapping
    s.save_semantics(PostSemantics(post_id="pair", symbol="BTCUSDT", model="m", classified_at=T0,
                                   trade_intent="LONG", market_impact_score=3.0, confidence=0.9,
                                   new_information_prob=0.9, communication_type="CATALYST_NEWS"))
    s.save_semantics(PostSemantics(post_id="pair", symbol="ETHUSDT", model="m", classified_at=T0,
                                   trade_intent="SHORT", market_impact_score=3.0, confidence=0.9,
                                   new_information_prob=0.9, communication_type="CATALYST_NEWS"))
    for sym in ("BTCUSDT", "ETHUSDT"):
        for i in range(0, 20):
            s.save_market(_mkt(sym, T0 + timedelta(minutes=i)))
    s.save_event(EventCluster(
        event_id="ev-eth", symbol="ETHUSDT", started_at=T0, last_seen_at=T0,
        representative_text="long BTC short ETH pair", post_ids=["pair"],
        unique_authors=1, actionable_at=T0,
    ))
    frame = narrative_event_frame(s, "15m", "m")
    assert len(frame) == 1
    assert frame.iloc[0].symbol == "ETHUSDT"
    assert frame.iloc[0].trade_intent == "SHORT"  # ETH leg, not the BTC LONG


# ---------- MULTI-ASSET: t.co redirector clustering ----------

def test_tco_links_do_not_cluster_unrelated_posts():
    t = T0
    posts = [
        _post("a", created=t, text="Solana validator outage investigation https://t.co/aaa"),
        _post("b", created=t + timedelta(seconds=30), text="Huge NFT giveaway mint live https://t.co/bbb"),
    ]
    clusters = EventClusterer(similarity=0.36).cluster(posts)
    assert len(clusters) == 2


def test_exact_url_match_still_clusters():
    t = T0
    url = "https://blog.example.com/post/1"
    posts = [
        _post("a", created=t, text=f"breaking: listing announced {url}", author="u1"),
        _post("b", created=t + timedelta(seconds=30), text=f"confirmed listing details {url}", author="u2"),
    ]
    # Exact normalized URL match is decisive even with weak text overlap.
    clusters = EventClusterer(similarity=0.99).cluster(posts)
    assert len(clusters) == 1


# ---------- MULTI-ASSET: persistent event lifecycle ----------

def test_events_attach_instead_of_recreate(tmp_path):
    s = Store(str(tmp_path / "t.db"))
    p1 = _post("p1", created=T0, seen=T0, text="Coinbase lists ABC tomorrow official")
    r1 = assign_posts_to_events(s, "SOLUSDT", [p1])
    assert r1["new_events"] == 1
    first_id = s.all_events("SOLUSDT")[0]["event_id"]
    p2 = _post("p2", created=T0 + timedelta(minutes=5), seen=T0 + timedelta(minutes=5),
               text="Coinbase lists ABC tomorrow official confirmed", author="u2")
    r2 = assign_posts_to_events(s, "SOLUSDT", [p2])
    assert r2["attached"] == 1 and r2["new_events"] == 0
    events = s.all_events("SOLUSDT")
    assert len(events) == 1
    assert events[0]["event_id"] == first_id
    assert set(events[0]["post_ids"]) == {"p1", "p2"}


def test_multi_asset_post_not_blocked_by_other_symbol_event(tmp_path):
    """A 'long ETH vs BTC' post claimed by a BTC event must still be attachable
    to an ETH event — event membership claims are symbol-scoped."""
    s = Store(str(tmp_path / "t.db"))
    p = _post("pair1", created=T0, seen=T0, text="long ETH vs BTC pair trade")
    r_btc = assign_posts_to_events(s, "BTCUSDT", [p])
    assert r_btc["new_events"] == 1
    # Same post under a different symbol universe must NOT be treated as claimed.
    r_eth = assign_posts_to_events(s, "ETHUSDT", [p])
    assert r_eth["new_events"] == 1
    # But the same symbol reprocessing the same post must see it as claimed.
    r_btc2 = assign_posts_to_events(s, "BTCUSDT", [p])
    assert r_btc2["new_events"] == 0


# ---------- PROVENANCE: model identity ----------

def _train_df(n: int = 120, f2_value: float = 0.0) -> pd.DataFrame:
    ts = pd.date_range(T0, periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(7)
    x1 = rng.normal(size=n)
    return pd.DataFrame({
        "symbol": "SOLUSDT", "ts": ts,
        "f1": x1,
        "f2": np.full(n, f2_value),
        "target_return": np.where(x1 + rng.normal(scale=0.5, size=n) > 0, 0.01, -0.01),
    })


def test_model_fingerprint_binds_dataset_values(tmp_path):
    """Two models with identical timestamps/labels but different feature values
    must not share a fingerprint (the audit's identity collision)."""
    a = SemanticResidualModel.fit(_train_df(f2_value=0.0), ["f1"], ["f1", "f2"],
                                  model_name="logistic", horizon="15m", calibrate_thresholds=False)
    b = SemanticResidualModel.fit(_train_df(f2_value=1.0), ["f1"], ["f1", "f2"],
                                  model_name="logistic", horizon="15m", calibrate_thresholds=False)
    assert a.training_fingerprint != b.training_fingerprint
    assert a.manifest["dataset_fingerprint"] != b.manifest["dataset_fingerprint"]
    c = SemanticResidualModel.fit(_train_df(f2_value=0.0), ["f1"], ["f1", "f2"],
                                  model_name="logistic", horizon="15m", calibrate_thresholds=False)
    assert a.training_fingerprint == c.training_fingerprint  # deterministic


def test_promotion_rejects_mismatched_report(tmp_path):
    model = SemanticResidualModel.fit(_train_df(), ["f1"], ["f1", "f2"],
                                      model_name="logistic", horizon="15m", calibrate_thresholds=False)
    mp = tmp_path / "m.joblib"
    model.save(mp)
    report = {"model_fingerprint": "deadbeef" * 8, "locked_holdout": {"status": "PASS_LOCKED_HOLDOUT"}}
    rp = tmp_path / "r.json"
    rp.write_text(json.dumps(report))
    s = Store(str(tmp_path / "t.db"))
    out = promote_candidate(candidate_model=mp, research_report=rp, store=s, registry_dir=tmp_path / "reg")
    assert out["status"] == "REJECTED_IDENTITY_MISMATCH"
    assert out["promoted"] is False


# ---------- HOLDOUT: raw-window identity + reopen ----------

def test_holdout_identity_binds_raw_window_not_features():
    a = holdout_identity(symbols=["SOLUSDT"], holdout_start="2026-01-01", holdout_end="2026-02-01",
                         horizon="15m", label_version="v4", sampling_mode="event")
    b = holdout_identity(symbols=["SOLUSDT"], holdout_start="2026-01-01", holdout_end="2026-02-01",
                         horizon="15m", label_version="v4", sampling_mode="event")
    c = holdout_identity(symbols=["SOLUSDT"], holdout_start="2026-01-02", holdout_end="2026-02-01",
                         horizon="15m", label_version="v4", sampling_mode="event")
    assert a == b and a != c  # feature rebuild over same window collides


def _labeled_store(path, n: int = 200) -> Store:
    s = Store(str(path))
    ts = T0
    for i in range(n):
        t = ts + timedelta(minutes=5 * i)
        s.save_features(FeatureSnapshot(
            symbol="SOLUSDT", ts=t, feature_version="v4",
            values={"return_5m": float(np.sin(i)), "posts_5m": 1.0,
                    "semantic_shock": float(np.cos(i)), "information_price_gap_proxy": float(np.sin(i) * 0.1)},
        ))
        s.save_label(LabelRecord(
            symbol="SOLUSDT", ts=t, label_version="v4",
            returns={"15m": 0.01 if i % 3 else -0.008},
            abnormal_returns={"15m": 0.01 if i % 3 else -0.008},
        ))
    return s


def test_locked_holdout_cannot_reopen(tmp_path):
    s = _labeled_store(tmp_path / "t.db")
    r1 = run_locked_protocol(
        s, tmp_path / "run1", horizons=["15m"], model_names=["logistic"],
        holdout_fraction=0.20, bootstrap_samples=50, sampling_mode="row",
    )
    assert r1["status"] in {"PASS_LOCKED_HOLDOUT", "FAIL_LOCKED_HOLDOUT"}
    r2 = run_locked_protocol(
        s, tmp_path / "run2", horizons=["15m"], model_names=["logistic"],
        holdout_fraction=0.20, bootstrap_samples=50, sampling_mode="row",
    )
    assert r2["status"] == "HOLDOUT_ALREADY_OPENED"
    r3 = run_locked_protocol(
        s, tmp_path / "run3", horizons=["15m"], model_names=["logistic"],
        holdout_fraction=0.20, bootstrap_samples=50, sampling_mode="row",
        allow_reopen=True,
    )
    assert r3["governance"]["holdout_reopened"] is True


def test_campaign_hard_stops_before_burning_holdout(tmp_path):
    """Leakage/quality failure must block the campaign before the holdout opens."""
    s = _labeled_store(tmp_path / "t.db")
    # Poison the well: a feature counting a post that wasn't observable.
    s.save_post(_post("late", created=T0, seen=T0 + timedelta(hours=5)))
    s.save_features(FeatureSnapshot(
        symbol="SOLUSDT", ts=T0 + timedelta(minutes=5), feature_version="v4",
        values={"posts_15m": 1.0, "intent_sum_5m": 1.0, "new_info_rate_5m": 0.9},
    ))
    r = run_campaign(s, tmp_path / "camp", horizons=["15m"], model_names=["logistic"], sampling_mode="row")
    assert r["status"] == "BLOCKED_INVALID_DATA"
    assert r["locked_protocol"]["status"] == "NOT_OPENED_DATA_GATES_FAILED"


# ---------- ECONOMICS: round-trip + portfolio ----------

def test_round_trip_costs_entry_plus_exit():
    """The audit's example: 50bps gross, 8.5bps one-way cost -> ~33bps net."""
    df = pd.DataFrame({
        "symbol": ["SOLUSDT", "SOLUSDT"],
        "ts": pd.to_datetime([T0, T0 + timedelta(minutes=15)]),
        "target_return": [0.005, 0.005],
    })
    costs = np.array([0.00085, 0.00085])
    sig = np.array([1, 0])
    net, mask = stateful_trade_vectors(df, sig, costs, horizon_minutes=15)
    assert mask.tolist() == [True, False]
    assert net[0] == pytest.approx(0.005 - 2 * 0.00085, abs=1e-9)


def test_portfolio_metrics_equity_native():
    """Idle periods count as 0% — a strategy trading once is not annualized as
    if every period were a trade."""
    trades = pd.DataFrame({
        "entry_ts": pd.to_datetime([T0, T0 + timedelta(hours=1)]),
        "exit_ts": pd.to_datetime([T0 + timedelta(minutes=15), T0 + timedelta(hours=1, minutes=15)]),
        "net_return": [0.01, -0.005],
        "notional_usd": [1000.0, 1000.0],
    })
    m = portfolio_metrics(trades, horizon_minutes=15, initial_capital=100_000.0)
    # $10 win then $5 loss on $100k: equity-curve net = 5/100000.
    assert m["portfolio_net_return"] == pytest.approx(0.00005, abs=1e-9)
    assert m["portfolio_periods"] >= 4  # idle periods exist between the trades
    assert m["avg_concurrent_positions"] is not None


# ---------- OPERATIONS: stream lifecycle telemetry ----------

def test_bybit_ws_basis_derived_from_mark_index():
    """Bybit V5 tickers never send a basis field — leaving it unread made basis
    (and venue_basis_dispersion) permanently zero. Derive mark/index like the
    Binance snapshot does."""
    a = MicrostructureAccumulator("SOLUSDT")
    now = datetime.now(timezone.utc)
    t0_ms = int(now.timestamp() * 1000)
    a.apply(_ws_msg("tickers.SOLUSDT", {"lastPrice": "100", "markPrice": "101", "indexPrice": "100"}, t0_ms))
    snap = a.snapshot(now)
    assert snap.basis == pytest.approx(0.01, abs=1e-9)


def test_x_stream_emits_disconnected_control_event():
    """A failed X WS connect must surface as telemetry, not silent retry."""
    from semantic_alpha.providers.x_stream import XRealtimeStream

    async def first():
        it = XRealtimeStream("k", "ws://127.0.0.1:1/").events()
        try:
            return await asyncio.wait_for(anext(it), timeout=15)
        finally:
            await it.aclose()

    ev = asyncio.run(first())
    assert ev["_control"] == "disconnected"
    assert "error" in ev


def test_stream_delivery_usage_unique_per_frame(tmp_path):
    """Vendor bills per delivered tweet per frame. Two deliveries of the same
    post (re-delivery, second rule match) must produce two usage rows, not
    collapse onto one reference_id."""
    from semantic_alpha.live import delivery_usage_record
    from semantic_alpha.storage import Store

    store = Store(str(tmp_path / "t.db"))
    now = datetime.now(timezone.utc)
    raw = {"id": "p1", "text": "$SOL up"}
    raw["_semantic_alpha_delivery_id"] = "p1:SOL:0:0"
    u1 = delivery_usage_record(raw, "SOL", 0, 0, now)
    raw["_semantic_alpha_delivery_id"] = "p1:SOL:1:0"
    u2 = delivery_usage_record(raw, "SOL", 1, 0, now)
    assert u1.reference_id != u2.reference_id
    store.save_usage(u1)
    store.save_usage(u2)
    with store.conn() as c:
        n = c.execute(
            "select count(*) from api_usage where category='post_read'").fetchone()[0]
    assert n == 2


def test_dropped_delivery_still_records_usage(tmp_path):
    """A post dropped on QueueFull was still billed by the vendor — the ledger
    must carry it (marked) rather than show zero."""
    from semantic_alpha.live import delivery_usage_record
    from semantic_alpha.storage import Store

    store = Store(str(tmp_path / "t.db"))
    now = datetime.now(timezone.utc)
    raw = {"id": "p9", "_semantic_alpha_delivery_id": "p9:BTC:3:0"}
    u = delivery_usage_record(raw, "BTC", 3, 0, now)
    store.save_usage(u.model_copy(update={"metadata": {**u.metadata, "drop_reason": "queue_full"}}))
    with store.conn() as c:
        row = c.execute(
            "select reference_id, json_extract(payload_json,'$.metadata.drop_reason') from api_usage").fetchone()
    assert tuple(row) == ("p9:BTC:3:0", "queue_full")


def test_receipt_usage_dedupes_save_post_replay(tmp_path):
    """Usage is logged at WS receipt for EVERY delivered tweet (the vendor
    bills ~15cr per delivery even when our symbol matcher discards it), then
    save_post replays the same delivery_id from raw._usage. The ledger must
    hold exactly one row per billed delivery — no double count, no gap."""
    from semantic_alpha.live import delivery_usage_record
    from semantic_alpha.providers.xapi import normalize_post
    from semantic_alpha.storage import Store

    store = Store(str(tmp_path / "t.db"))
    now = datetime.now(timezone.utc)
    # Receipt-time log for a tweet that fails symbol matching (unmatched path).
    raw_unmatched = {"id": "u1"}
    raw_unmatched["_semantic_alpha_delivery_id"] = "u1:all:7:0"
    store.save_usage(delivery_usage_record(raw_unmatched, "all", 7, 0, now))
    # Receipt-time log + save_post replay for a matched tweet.
    raw_matched = {"id": "m1", "text": "$BTC moon", "createdAt": now.isoformat()}
    raw_matched["_semantic_alpha_delivery_id"] = "m1:all:8:0"
    store.save_usage(delivery_usage_record(raw_matched, "all", 8, 0, now))
    post = normalize_post(raw_matched, "BTCUSDT")
    store.save_post(post)
    store.save_post(post)  # dedupe path replays _usage again — still one row
    with store.conn() as c:
        refs = [r[0] for r in c.execute("select reference_id from api_usage order by reference_id")]
    assert refs == ["m1:all:8:0", "u1:all:7:0"]


def test_label_exit_tolerance_is_horizon_proportional(tmp_path):
    """A '5m' label measured over 6.75 minutes is a different measurement:
    exits may lag at most 25% of the horizon (bounded by max_delay) or the
    horizon stays unrealized instead of silently overshooting."""
    s = Store(str(tmp_path / "t.db"))
    s.save_market(_mkt("SOLUSDT", T0))
    s.save_market(_mkt("SOLUSDT", T0 + timedelta(minutes=5, seconds=100)))
    s.save_market(_mkt("SOLUSDT", T0 + timedelta(minutes=16)))
    l = build_labels(s, "SOLUSDT", T0)
    assert l is not None
    # 5m tol = 75s: the +100s row must not be accepted as the 5m exit.
    assert l.returns["5m"] is None
    # 15m tol = 120s: the +16m row is +60s into the window → realized.
    assert l.returns["15m"] == pytest.approx(0.0)
    assert l.realized_minutes["15m"] == pytest.approx(16.0)


def test_partial_path_stats_recompute_on_refill(tmp_path):
    """Path excursions are only final once their horizon realized: a '15m'
    max_down measured over a 2-minute partial window must be recomputed when
    the window completes — previously the first non-None value froze forever.
    realized_minutes must fill alongside returns on refill too."""
    s = Store(str(tmp_path / "t.db"))
    s.save_market(_mkt("SOLUSDT", T0))
    s.save_market(_mkt("SOLUSDT", T0 + timedelta(minutes=2), last=95.0))
    l1 = build_labels(s, "SOLUSDT", T0)
    assert l1.returns["15m"] is None
    assert l1.max_down["15m"] == pytest.approx(-0.05)
    s.save_label(l1)
    s.save_market(_mkt("SOLUSDT", T0 + timedelta(minutes=10), last=90.0))
    s.save_market(_mkt("SOLUSDT", T0 + timedelta(minutes=15), last=100.0))
    s.save_label(build_labels(s, "SOLUSDT", T0))
    merged = s.label_for("SOLUSDT", T0)
    assert merged.returns["15m"] == pytest.approx(0.0)
    assert merged.max_down["15m"] == pytest.approx(-0.10)
    assert merged.realized_minutes["15m"] == pytest.approx(15.0)


def test_repair_event_actionable_at_backfills_legacy(tmp_path):
    """Events written before actionable_at existed get it repaired from member
    posts' earliest observed time — derived state repaired, evidence untouched."""
    from semantic_alpha.clustering import EventCluster
    from semantic_alpha.maintenance import repair_event_actionable_at

    s = Store(str(tmp_path / "t.db"))
    s.save_post(_post("ev1", created=T0, seen=T0 + timedelta(seconds=30)))
    s.save_event(EventCluster(
        event_id="e1", symbol="SOLUSDT", started_at=T0, last_seen_at=T0,
        representative_text="x", post_ids=["ev1"], unique_authors=1,
        source_domains=0, similarity_mean=1.0, actionable_at=None,
    ))
    out = repair_event_actionable_at(s)
    assert out["events_repaired"] == 1
    with s.conn() as c:
        pj = json.loads(c.execute("select payload_json from events where event_id='e1'").fetchone()[0])
    assert pj["actionable_at"] == (T0 + timedelta(seconds=30)).isoformat()
    assert repair_event_actionable_at(s)["events_repaired"] == 0  # idempotent


def test_engagement_refresh_maps_to_market_symbol(tmp_path):
    """refresh_engagement queries X with the base ticker ($SOL) but must store
    posts/engagement under the tracked perp (SOLUSDT) — otherwise it creates
    phantom 'SOL' asset mappings and wrong-symbol semantics."""
    from semantic_alpha.maintenance import refresh_engagement

    s = Store(str(tmp_path / "t.db"))
    s.save_post(_post("r1", created=T0, seen=T0))

    class FakeX:
        async def search_window(self, symbol, start, end, max_pages=2):
            assert symbol == "SOL"  # query must use the base ticker
            return [_post("r1", symbol="SOL", created=start + timedelta(minutes=1),
                          seen=start + timedelta(minutes=2))]

    out = asyncio.run(refresh_engagement(s, FakeX(), symbols=["SOLUSDT"],
                                         lookback_minutes=14, slice_minutes=8))
    assert out["engagement_refreshed"] == 1
    with s.conn() as c:
        syms = {r[0] for r in c.execute("select symbol from post_assets where post_id='r1'")}
        obs_syms = {r[0] for r in c.execute("select symbol from engagement_snapshots where post_id='r1'")}
    assert syms == {"SOLUSDT"} and obs_syms == {"SOLUSDT"}
