"""Coverage for the v4 research-hardening layer: new label heads, venue-aware
market reads, residualized features, adversarial benchmark, overfit controls,
RSS ingestion, and capacity/latency reports."""
import asyncio
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from semantic_alpha.schema import MarketSnapshot, SocialPost, FeatureSnapshot
from semantic_alpha.storage import Store
from semantic_alpha.labels import build_labels
from semantic_alpha.semantics.heuristic import HeuristicSemanticEngine


def _mkt(sym, ts, last, source="bybit", **kw):
    return MarketSnapshot(source=source, symbol=sym, ts=ts, last=last, **kw)


def _path(s, sym, t0, prices_minutes, source="bybit"):
    for mins, price in prices_minutes:
        s.save_market(_mkt(sym, t0 + timedelta(minutes=mins), price, source))


# ---------- labels v4 ----------

def test_labels_v4_heads(tmp_path):
    s = Store(str(tmp_path / 'l.db')); t = datetime.now(timezone.utc)
    # Spike to 110 within 5m then settle at 102 by 15m -> fade_ratio < 1.
    _path(s, 'SOLUSDT', t, [(0, 100), (1, 105), (3, 110), (5, 108), (15, 102), (30, 102), (60, 102), (240, 102), (1440, 102)])
    _path(s, 'BTCUSDT', t, [(0, 100), (1440, 100)])
    l = build_labels(s, 'SOLUSDT', t)
    assert l is not None
    assert l.abs_move['15m'] == pytest.approx(0.10)
    assert l.range_move['15m'] == pytest.approx(0.10 - (-0.0)) or l.range_move['15m'] > 0.09
    # Return at 15m is +2% but excursion was +10% -> most of the move faded.
    assert l.fade_ratio['15m'] == pytest.approx(0.02 / 0.10, abs=1e-6)


def test_dataset_exposes_v4_targets(tmp_path):
    s = Store(str(tmp_path / 'd.db')); t = datetime.now(timezone.utc) - timedelta(hours=2)
    _path(s, 'SOLUSDT', t, [(m, 100 + 0.1 * m) for m in [0, 1, 5, 15, 30, 60, 240, 1440]])
    _path(s, 'BTCUSDT', t, [(m, 100) for m in [0, 1, 5, 15, 30, 60, 240, 1440]])
    s.save_features(FeatureSnapshot(symbol='SOLUSDT', ts=t, feature_version='v4',
                                    values={'return_1m': 0.0, 'posts_5m': 1.0}))
    l = build_labels(s, 'SOLUSDT', t)
    assert l is not None and l.label_version == 'v4'
    s.save_label(l)
    from semantic_alpha.research import dataset
    d = dataset(s, horizon='15m')
    assert len(d) == 1
    assert 'target_abs_move' in d.columns and 'target_range_move' in d.columns and 'target_fade_ratio' in d.columns
    assert d.iloc[0]['target_abs_move'] > 0


# ---------- venue-aware reads ----------

def test_canonical_reads_exclude_other_venues(tmp_path):
    s = Store(str(tmp_path / 'v.db')); t = datetime.now(timezone.utc)
    s.save_market(_mkt('SOLUSDT', t, 100.0, source='bybit'))
    s.save_market(_mkt('SOLUSDT', t, 101.0, source='binance'))
    xs = s.market_between('SOLUSDT', t - timedelta(minutes=1), t + timedelta(minutes=1))
    assert len(xs) == 1 and xs[0].source == 'bybit' and xs[0].last == 100.0
    # Explicit venue opt-in still works.
    xs = s.market_between('SOLUSDT', t - timedelta(minutes=1), t + timedelta(minutes=1), source='binance')
    assert len(xs) == 1 and xs[0].last == 101.0
    assert s.latest_market('SOLUSDT').source == 'bybit'
    # Cross-venue view returns the latest per source.
    near = s.markets_near('SOLUSDT', t, max_delay_seconds=60)
    assert set(near) == {'bybit', 'binance'}


def test_canonical_reads_include_bybit_history(tmp_path):
    s = Store(str(tmp_path / 'h.db')); t = datetime.now(timezone.utc)
    s.save_market(_mkt('SOLUSDT', t, 100.0, source='bybit_history'))
    xs = s.market_between('SOLUSDT', t - timedelta(minutes=1), t + timedelta(minutes=1))
    assert len(xs) == 1 and xs[0].source == 'bybit_history'


def test_binance_snapshot_normalization():
    from semantic_alpha.providers.binance import BinancePublicClient
    payloads = {
        "/fapi/v1/ticker/bookTicker": {"bidPrice": "99.90", "askPrice": "100.10"},
        "/fapi/v1/premiumIndex": {"markPrice": "100.05", "indexPrice": "100.00", "lastFundingRate": "0.0001"},
        "/fapi/v1/openInterest": {"openInterest": "5000"},
        "/fapi/v1/ticker/24hr": {"lastPrice": "100.02", "quoteVolume": "123456.0"},
    }

    class Fake(BinancePublicClient):
        async def _get(self, client, path, symbol):
            return payloads[path]

    m = asyncio.run(Fake().snapshot('SOLUSDT'))
    assert m.source == 'binance' and m.symbol == 'SOLUSDT'
    assert m.bid == 99.90 and m.ask == 100.10
    assert m.open_interest_value == pytest.approx(5000 * 100.05)
    assert m.spread_bps == pytest.approx((0.20 / 100.0) * 10_000, rel=1e-3)


# ---------- residualized + venue features ----------

def test_residual_features_warmup_and_value(tmp_path):
    from semantic_alpha.features import FeatureEngine, RESIDUAL_MIN_SAMPLES
    s = Store(str(tmp_path / 'f.db')); t0 = datetime.now(timezone.utc) - timedelta(hours=6)
    sym = 'SOLUSDT'
    # Trailing feature history: social activity perfectly tracks |return|.
    for i in range(RESIDUAL_MIN_SAMPLES + 10):
        ts = t0 + timedelta(minutes=i)
        ret = 0.001 * (1 + i % 5)
        s.save_features(FeatureSnapshot(symbol=sym, ts=ts, feature_version='v4', values={
            'return_1m': ret, 'return_5m': ret, 'return_15m': ret,
            'trade_notional_1m': 1000.0, 'liquidation_long_usd_1m': 0.0, 'liquidation_short_usd_1m': 0.0,
            'oi_change_5m': 0.0, 'realized_vol_15m': 0.0,
            'attention_z_5m': 2.0 * ret, 'posts_5m': 10.0 * ret, 'intent_sum_5m': ret, 'semantic_shock': ret,
        }))
    now = t0 + timedelta(hours=6)
    s.save_market(_mkt(sym, now, 100.0))
    fe = FeatureEngine(s)
    f = fe.build(sym, now, s.latest_market(sym))
    for k in ('attention_residual_5m', 'posts_residual_5m', 'intent_residual_5m', 'semantic_shock_residual'):
        assert k in f.values
    # Current row: attention_z exactly what the market state predicts -> residual ~ 0.
    f2 = fe.build(sym, now, s.latest_market(sym))
    assert abs(f2.values['attention_residual_5m']) < 0.5


def test_venue_dispersion_features(tmp_path):
    from semantic_alpha.features import FeatureEngine
    s = Store(str(tmp_path / 'vd.db')); t = datetime.now(timezone.utc)
    s.save_market(_mkt('SOLUSDT', t, 100.00, source='bybit'))
    s.save_market(_mkt('SOLUSDT', t, 100.30, source='binance'))
    f = FeatureEngine(s).build('SOLUSDT', t, s.latest_market('SOLUSDT'))
    assert f.values['venue_price_dispersion_bps'] == pytest.approx(30.0, rel=0.05)


# ---------- adversarial benchmark ----------

def test_adversarial_cases_deterministic_and_evaluated(tmp_path):
    from semantic_alpha.adversarial import generate_cases, evaluate_case, run_benchmark
    from semantic_alpha.schema import PostSemantics
    cases = generate_cases('SOL')
    assert cases == generate_cases('SOL')
    assert len(cases) >= 15
    classes = {c['case_class'] for c in cases}
    assert {'prompt_injection', 'fake_official', 'negation', 'benign_control'} <= classes
    # A fooled classifier registers violations.
    fooled = PostSemantics(post_id='x', symbol='SOLUSDT', model='t',
                           claim_status='VERIFIED', source_role='PRIMARY_OFFICIAL', trade_intent='STRONG_LONG')
    v = evaluate_case(fooled, {'claim_status_neq': 'VERIFIED', 'source_role_neq': 'PRIMARY_OFFICIAL',
                               'trade_intent_not_in': ['STRONG_LONG']})
    assert set(v) == {'claim_status', 'source_role', 'trade_intent'}
    r = asyncio.run(run_benchmark(HeuristicSemanticEngine(), symbol='SOL'))
    assert r['cases'] == len(cases) and 'manipulability_score' in r


# ---------- overfit controls ----------

def test_deflated_sharpe_selection_penalty():
    from semantic_alpha.overfit import deflated_sharpe_ratio
    p1 = deflated_sharpe_ratio(0.15, n_trials=1, n_obs=500)
    p100 = deflated_sharpe_ratio(0.15, n_trials=100, n_obs=500)
    assert p1 is not None and p100 is not None
    assert p100 < p1  # more trials -> higher bar -> lower confidence
    assert deflated_sharpe_ratio(1.0, n_trials=1, n_obs=5) is None


def test_combinatorial_folds_and_purge():
    from semantic_alpha.overfit import combinatorial_folds, _purged_train_idx
    import pandas as pd
    folds = list(combinatorial_folds(200, n_groups=8))
    assert len(folds) == math.comb(8, 4)
    tr, te = folds[0]
    assert len(tr) + len(te) == 200 and not set(tr) & set(te)
    # Purging drops train rows whose label window overlaps the test block.
    # Put the test block in the middle: train rows in the 15min before it must go.
    df = pd.DataFrame({'ts': pd.date_range('2026-01-01', periods=200, freq='min')})
    te = np.arange(95, 115)
    tr = np.array([i for i in range(200) if i not in set(te)])
    kept = _purged_train_idx(df, tr, te, horizon_minutes=15)
    assert set(kept) == set(range(0, 80)) | set(range(115, 200))  # rows 80-94 purged
    assert all(df.ts.iloc[k] + pd.Timedelta(minutes=15) <= df.ts.iloc[95]
               or df.ts.iloc[k] > df.ts.iloc[114] for k in kept)


def test_trial_ledger_immutable_and_dedup(tmp_path):
    s = Store(str(tmp_path / 't.db'))
    assert s.trial_count() == 0
    assert s.record_trial('', 'research-suite', {'rows': 100}) is True
    # Same payload -> same fingerprint -> INSERT OR IGNORE dedupes.
    assert s.record_trial('', 'research-suite', {'rows': 100}) is False
    assert s.record_trial('', 'research-suite', {'rows': 200}) is True
    assert s.trial_count() == 2
    recent = s.trials(10)
    assert len(recent) == 2 and all('fingerprint' in r and 'command' in r for r in recent)


def test_cpcv_pbo_report_smoke(tmp_path):
    import pandas as pd
    from semantic_alpha.overfit import cpcv_pbo_report
    rng = np.random.default_rng(0)
    n = 400
    d = pd.DataFrame({
        'ts': pd.date_range('2026-01-01', periods=n, freq='min'),
        'target_return': rng.normal(0, 0.001, n),
        'f1': rng.normal(0, 1, n), 'f2': rng.normal(0, 1, n),
        'spread_bps': np.full(n, 2.0), 'depth_bid_10bps': np.full(n, 1e6), 'depth_ask_10bps': np.full(n, 1e6),
    })
    variants = {'a': ['f1'], 'b': ['f1', 'f2']}
    r = cpcv_pbo_report(d, variants=variants, horizon_minutes=1, n_groups=6)
    assert r['combos_evaluated'] > 0 and r['pbo'] is not None
    assert 0.0 <= r['pbo'] <= 1.0 and set(r['oos_sharpe_median']) == {'a', 'b'}


# ---------- rss ----------

RSS_XML = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>Exchange Blog</title>
<item><title>SOLUSDT staking launch</title><link>https://example.com/1</link>
<guid>g1</guid><pubDate>Mon, 01 Jun 2026 12:00:00 GMT</pubDate>
<description>Solana staking is live.</description></item>
<item><title>Macro note</title><link>https://example.com/2</link>
<guid>g2</guid><pubDate>Mon, 01 Jun 2026 12:05:00 GMT</pubDate>
<description>Fed minutes today.</description></item>
</channel></rss>"""

ATOM_XML = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
<entry><title>BTC maintenance</title><id>a1</id>
<link href="https://example.com/a1"/><published>2026-06-01T12:10:00Z</published>
<summary>Bitcoin wallet maintenance window.</summary></entry>
</feed>"""


def test_rss_parse_and_symbol_mapping(tmp_path):
    from semantic_alpha.providers.rss import parse_feed, items_to_posts
    items = parse_feed(RSS_XML)
    assert len(items) == 2 and items[0]['title'].startswith('SOLUSDT')
    posts = items_to_posts(items, 'exchange_blog', ['SOLUSDT'])
    syms = {p.symbol for p in posts}
    assert 'SOLUSDT' in syms and 'MACRO' in syms
    assert all(p.platform == 'news' and p.post_id.startswith('rss-') for p in posts)
    atoms = parse_feed(ATOM_XML)
    assert len(atoms) == 1 and 'Bitcoin' in atoms[0]['summary']


def test_rss_posts_dedup_in_storage(tmp_path):
    from semantic_alpha.providers.rss import parse_feed, items_to_posts
    s = Store(str(tmp_path / 'r.db'))
    posts = items_to_posts(parse_feed(RSS_XML), 'exchange_blog', ['SOLUSDT'])
    n1 = sum(int(s.save_post(p)) for p in posts)
    n2 = sum(int(s.save_post(p)) for p in posts)
    assert n1 >= 2 and n2 == 0


# ---------- reports ----------

def test_capacity_and_latency_reports(tmp_path):
    from semantic_alpha.economics import capacity_report
    from semantic_alpha.quality import latency_report
    s = Store(str(tmp_path / 'c.db')); t = datetime.now(timezone.utc)
    for i in range(3):
        s.save_market(_mkt('SOLUSDT', t - timedelta(minutes=i), 100.0,
                           spread_bps=2.0, depth_bid_10bps=1e6, depth_ask_10bps=1e6))
    r = capacity_report(s, notionals=[1000.0, 10000.0])
    assert 'SOLUSDT' in r['symbols']
    costs = r['symbols']['SOLUSDT']['cost_bps_by_notional']
    assert costs['10000']['median_cost_bps'] >= costs['1000']['median_cost_bps']
    # Latency report on one post + one semantic classification.
    p = SocialPost(post_id='lp', symbol='SOLUSDT', author_username='a', text='x',
                   created_at=t - timedelta(seconds=100), first_seen_at=t)
    s.save_post(p)
    s.save_semantics(asyncio.run(HeuristicSemanticEngine().classify(p)))
    lr = latency_report(s)
    assert lr['ingest_lag']['n'] == 1 and lr['ingest_lag']['mean_s'] == pytest.approx(100.0, abs=1.0)
    assert lr['semantic_lag']['n'] == 1


def test_semantic_reliability_report(tmp_path):
    from semantic_alpha.factor_analysis import semantic_reliability
    from semantic_alpha.schema import PostSemantics
    s = Store(str(tmp_path / 'sr.db')); t0 = datetime.now(timezone.utc) - timedelta(hours=3)
    # Rising market so labels exist at every post timestamp.
    _path(s, 'SOLUSDT', t0, [(m, 100 + 0.05 * m) for m in range(0, 200)])
    _path(s, 'BTCUSDT', t0, [(m, 100) for m in range(0, 200)])
    for i in range(45):
        ts = t0 + timedelta(minutes=i)
        p = SocialPost(post_id=f'p{i}', symbol='SOLUSDT', author_username='a', text=f'post {i}',
                       created_at=ts, first_seen_at=ts)
        s.save_post(p)
        s.save_semantics(PostSemantics(post_id=p.post_id, symbol='SOLUSDT', model='h',
                                       new_information_prob=i / 45.0, confidence=0.5))
        l = build_labels(s, 'SOLUSDT', ts)
        if l:
            s.save_label(l)
    r = semantic_reliability(s, horizon='15m', bins=4)
    assert not r.empty and 'spearman_vs_abs_move' in r.columns


def test_query_langs_configurable():
    import dataclasses
    from unittest.mock import patch
    from semantic_alpha.providers.xapi import XAPIClient
    from semantic_alpha.config import settings
    c = XAPIClient('k')
    st = datetime(2026, 1, 1, tzinfo=timezone.utc); en = datetime(2026, 1, 2, tzinfo=timezone.utc)
    with patch('semantic_alpha.providers.xapi.settings', dataclasses.replace(settings, x_query_langs='en')):
        assert 'lang:en' in c.query('SOL', st, en)
    with patch('semantic_alpha.providers.xapi.settings', dataclasses.replace(settings, x_query_langs='')):
        assert 'lang:' not in c.query('SOL', st, en)


# ---------- X search early-stop + regime pulse ----------

def _tweet(pid, created_iso, text="$BTC test"):
    return {"id": pid, "text": text, "createdAt": created_iso,
            "author": {"userName": "u1", "followersCount": 10},
            "likeCount": 5, "retweetCount": 1, "replyCount": 0}


def test_xapi_search_early_stop_on_known_page(monkeypatch, tmp_path):
    """A page whose posts are all already stored halts pagination — trailing
    pages of duplicates are billed spend we can skip."""
    pages = [
        {"tweets": [_tweet("k1", "2026-09-21T10:00:00Z"), _tweet("k2", "2026-09-21T10:01:00Z")], "next_cursor": "c2"},
        {"tweets": [_tweet("n1", "2026-09-21T09:59:00Z")], "next_cursor": None},
    ]
    calls = []

    class FakeClient:
        def __init__(self, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, params=None, headers=None):
            calls.append(1)
            return xapi_resp(pages.pop(0))

    import semantic_alpha.providers.xapi as xm
    monkeypatch.setattr(xm.httpx, "AsyncClient", FakeClient)
    c = xm.XAPIClient("k")
    st = datetime(2026, 9, 21, 9, 0, tzinfo=timezone.utc); en = datetime(2026, 9, 21, 11, 0, tzinfo=timezone.utc)
    out = asyncio.run(c.search_window("BTC", st, en, known_ids={"k1", "k2"}))
    assert out == [] and len(calls) == 1  # stopped after the all-known first page


def test_xapi_search_known_then_new_page_continues(monkeypatch):
    """New posts on a page keep pagination alive; stop only fires when a page
    contributes nothing new."""
    import semantic_alpha.providers.xapi as xm
    pages = [
        {"tweets": [_tweet("n1", "2026-09-21T10:05:00Z"), _tweet("k1", "2026-09-21T10:04:00Z")], "next_cursor": "c2"},
        {"tweets": [_tweet("k2", "2026-09-21T10:02:00Z")], "next_cursor": "c3"},
        {"tweets": [_tweet("n2", "2026-09-21T10:00:00Z")], "next_cursor": None},
    ]
    calls = []

    class FakeClient:
        def __init__(self, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, params=None, headers=None):
            calls.append(1)
            return xapi_resp(pages.pop(0))

    monkeypatch.setattr(xm.httpx, "AsyncClient", FakeClient)
    c = xm.XAPIClient("k")
    st = datetime(2026, 9, 21, 9, 0, tzinfo=timezone.utc); en = datetime(2026, 9, 21, 11, 0, tzinfo=timezone.utc)
    out = asyncio.run(c.search_window("BTC", st, en, known_ids={"k1", "k2"}))
    assert [p.post_id for p in out] == ["n1"] and len(calls) == 2  # known page 2 ended the walk


class _R:
    status_code = 200
    headers: dict = {}

    def __init__(self, payload): self._p = payload
    def raise_for_status(self): pass
    def json(self): return self._p


def xapi_resp(payload):
    return _R(payload)


def test_regime_pulse_digest_and_storage(tmp_path):
    """Digest carries market state + social aggregates + representative posts;
    the stored pulse ledgers its usage like any billed call."""
    from semantic_alpha.regime_pulse import build_digest
    s = Store(str(tmp_path / "p.db"))
    t = datetime.now(timezone.utc)
    s.save_market(_mkt("BTCUSDT", t, 67000.0, funding_rate=0.01, open_interest_value=1e9))
    p = SocialPost(post_id="pp1", platform="x", symbol="BTCUSDT", author_username="alice",
                   text="$BTC breakout", created_at=t - timedelta(minutes=3),
                   first_seen_at=t - timedelta(minutes=2), likes=42, reposts=3, replies=1,
                   urls=[], raw={})
    s.save_post(p)
    d = build_digest(s, "BTCUSDT", t)
    assert d["asset"] == "BTCUSDT"
    assert d["market"]["price"] == 67000.0
    assert d["social_stats"]["posts_15m"] == 1
    assert d["representative_posts"][0]["text"].startswith("$BTC")
    # storage + usage ledgering
    s.save_regime_pulse("BTCUSDT", t, "jev-1.13.0", {
        "trade_action": "BUY", "digest": d,
        "_usage": {"provider": "typesafe", "category": "regime_pulse",
                   "reference_id": "regime_pulse:BTCUSDT:1:v1", "units": 900.0,
                   "unit_name": "input_token", "estimated_usd": 0.001,
                   "cost_source": "estimated_tokens_configured_rate"}})
    rows = s.regime_pulses_between("BTCUSDT", t - timedelta(minutes=1), t + timedelta(minutes=1))
    assert len(rows) == 1 and rows[0]["trade_action"] == "BUY"
    with s.conn() as c:
        n = c.execute("select count(*) from api_usage where category='regime_pulse'").fetchone()[0]
    assert n == 1


# ---------- classification failure tracking ----------

def _post2(pid, sym="BTCUSDT", minutes_ago=30):
    t = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return SocialPost(post_id=pid, platform="x", symbol=sym, author_username="u",
                      text=f"$BTC test {pid}", created_at=t, first_seen_at=t,
                      likes=0, reposts=0, replies=0, urls=[], raw={})


def test_classification_failure_cap_excludes_junk(tmp_path):
    """Posts that deterministically fail classification stop being retried
    after 3 counted failures — otherwise each backfill tick re-bills them."""
    s = Store(str(tmp_path / "f.db"))
    s.save_post(_post2("bad1")); s.save_post(_post2("ok1"))
    for _ in range(3):
        s.record_classification_failure("bad1", "BTCUSDT", "jev-1.13.0", "400 invalid content")
    pending = s.unsemanticized_posts(None, "jev-1.13.0", limit=50)
    assert {p.post_id for p in pending} == {"ok1"}
    # provider-side errors do NOT count toward the cap
    for _ in range(5):
        s.record_classification_failure("ok1", "BTCUSDT", "jev-1.13.0", "402 no credits", counted=False)
    pending2 = s.unsemanticized_posts(None, "jev-1.13.0", limit=50)
    assert {p.post_id for p in pending2} == {"ok1"}  # still retryable


def test_backfill_continues_past_single_failure(tmp_path):
    """One bad post must not kill the whole backfill pass; consecutive provider
    errors still stop early (provider down)."""
    from semantic_alpha.maintenance import backfill_semantics
    s = Store(str(tmp_path / "b.db"))
    s.save_post(_post2("f1")); s.save_post(_post2("g1")); s.save_post(_post2("g2"))

    class Flakey:
        model = "jev-x"
        async def classify(self, p):
            if p.post_id == "f1":
                raise ValueError("400 content rejected")
            from semantic_alpha.schema import PostSemantics
            return PostSemantics(post_id=p.post_id, symbol=p.symbol, model="jev-x",
                                 relevance="RELEVANT", trade_intent="LONG")

    out = asyncio.run(backfill_semantics(s, Flakey()))
    assert out["backfilled"] == 2 and out["errors"] == 1
    # f1 stays retryable (1 counted failure < cap 3) but g1/g2 got classified
    assert [p.post_id for p in s.unsemanticized_posts(None, "jev-x", limit=50)] == ["f1"]

    class Down:
        model = "jev-x"
        async def classify(self, p):
            raise RuntimeError("402 no available TypeSafe API credits")

    s.save_post(_post2("x1")); s.save_post(_post2("x2")); s.save_post(_post2("x3"))
    s.save_post(_post2("x4")); s.save_post(_post2("x5")); s.save_post(_post2("x6"))
    out2 = asyncio.run(backfill_semantics(s, Down()))
    assert out2["backfilled"] == 0 and out2["errors"] == 5  # stopped after 5 consecutive provider errors
    # provider errors must not count toward the cap — posts stay retryable
    # (6 x-posts + f1 still pending from part one)
    assert len(s.unsemanticized_posts(None, "jev-x", limit=50)) == 7


# ---------- maintenance label maturity ----------

def _label(sym, ts, ret_15m=0.01, ret_24h=None):
    from semantic_alpha.schema import LabelRecord
    return LabelRecord(symbol=sym, ts=ts, label_version="v4",
                       returns={"1m": 0.001, "5m": 0.004, "15m": ret_15m, "30m": 0.01,
                                "1h": 0.02, "4h": 0.03, "24h": ret_24h},
                       abnormal_returns={}, benchmark_symbol="BTCUSDT",
                       benchmark_beta=1.0, base_ts=ts, entry_delay_seconds=0,
                       realized_minutes={}, max_up={}, max_down={}, abs_move={},
                       range_move={}, fade_ratio={})


def test_maintenance_finalizes_permanently_open_labels(tmp_path):
    """A label whose 24h exit window closed >26h ago with no market data must be
    marked matured — otherwise it rescans every tick forever."""
    from semantic_alpha.maintenance import maintenance_step
    s = Store(str(tmp_path / "m.db"))
    now = datetime.now(timezone.utc)
    old_ts = now - timedelta(hours=30)
    s.save_label(_label("BTCUSDT", old_ts, ret_24h=None))
    assert s.immature_labels("v4")  # starts immature
    out = maintenance_step(s, semantic_model=None)
    assert out["labels_finalized"] == 1
    assert not s.immature_labels("v4")  # finalized — no longer rescanned
    # the None horizon stays None (honest missing data, never fabricated)
    assert s.label_for("BTCUSDT", old_ts, "v4").returns["24h"] is None


def test_maintenance_skips_not_yet_due_labels(tmp_path):
    """A label whose earliest open horizon hasn't arrived must NOT be rebuilt —
    rebuilding it is pure waste."""
    from semantic_alpha.maintenance import maintenance_step
    s = Store(str(tmp_path / "m2.db"))
    now = datetime.now(timezone.utc)
    recent_ts = now - timedelta(hours=1)  # 24h horizon not due for ~23h
    s.save_label(_label("BTCUSDT", recent_ts, ret_24h=None))
    out = maintenance_step(s, semantic_model=None)
    assert out["labels_finalized"] == 0 and out["labels_refilled"] == 0
    assert s.immature_labels("v4")  # still immature, untouched


def test_coverage_model_prefers_stored_over_alias(tmp_path):
    """Before Jev resolves its effective model, coverage checks must use the
    most recently stored model — otherwise every already-classified post looks
    pending under the 'jev-latest' alias and gets billed reclassification."""
    from semantic_alpha.maintenance import backfill_semantics
    s = Store(str(tmp_path / "c.db"))
    s.save_post(_post2("p1"))
    from semantic_alpha.schema import PostSemantics
    s.save_semantics(PostSemantics(post_id="p1", symbol="BTCUSDT", model="jev-1.13.0",
                                   relevance="RELEVANT", trade_intent="LONG"))

    class FreshEngine:
        model = "jev-latest"          # configured alias
        effective_model = None        # not yet resolved
        query_model = "jev-latest"    # property would return alias pre-resolution
        calls = 0
        async def classify(self, p):
            self.calls += 1
            raise RuntimeError("unreachable — nothing is actually pending")

    e = FreshEngine()
    out = asyncio.run(backfill_semantics(s, e))
    assert e.calls == 0 and out["backfilled"] == 0  # saw stored model, not the alias
