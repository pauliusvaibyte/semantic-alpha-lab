import asyncio
from datetime import datetime, timedelta, timezone
from semantic_alpha.pipeline import collect_cycle
from semantic_alpha.schema import MarketSnapshot, PostSemantics, SocialPost
from semantic_alpha.semantics.heuristic import HeuristicSemanticEngine
from semantic_alpha.storage import Store

class M:
    async def snapshot(self,symbol):
        return MarketSnapshot(symbol=symbol,ts=datetime.now(timezone.utc),last=100,spread_bps=1,open_interest_value=1e9)
class T:
    async def search_window(self,symbol,start,end,max_pages=50,*,known_ids=None):
        return [SocialPost(post_id='x',symbol=symbol,text="I'm buying $SOL after partnership",author_username='a',created_at=end,first_seen_at=end)]

def test_collect_cycle(tmp_path):
    s=Store(str(tmp_path/'p.db'))
    r=asyncio.run(collect_cycle(s,'SOLUSDT',M(),T(),HeuristicSemanticEngine(),5))
    assert r.posts_inserted==1
    assert r.posts_semanticized==1
    assert 'events_15m' in r.feature.values


def test_collect_cycle_watermark_prevents_refetch(tmp_path):
    s=Store(str(tmp_path/'w.db'))
    calls=[]
    class W:
        async def search_window(self,symbol,start,end,max_pages=50,*,known_ids=None):
            calls.append((start,end))
            # Provider returns the same post again inside the lag-overlap buffer.
            return [SocialPost(post_id='x',symbol=symbol,text="I'm buying $SOL",author_username='a',created_at=end-timedelta(seconds=1),first_seen_at=end)]
    asyncio.run(collect_cycle(s,'SOLUSDT',M(),W(),HeuristicSemanticEngine(),5,60))
    r2=asyncio.run(collect_cycle(s,'SOLUSDT',M(),W(),HeuristicSemanticEngine(),5,60))
    # Steady state: second window starts at prior window end minus lag buffer,
    # not the full 5-minute lookback.
    assert abs((calls[1][0]-(calls[0][1]-timedelta(seconds=60))).total_seconds())<2
    assert calls[1][0]>calls[0][0]
    # Same post id is not stored twice even though the provider re-returned it.
    assert r2.posts_fetched==1 and r2.posts_inserted==0
    with s.conn() as c:
        assert c.execute("SELECT COUNT(*) FROM posts").fetchone()[0]==1


def test_collect_cycle_gap_falls_back_to_full_lookback(tmp_path):
    s=Store(str(tmp_path/'g.db'))
    calls=[]
    class W:
        async def search_window(self,symbol,start,end,max_pages=50,*,known_ids=None):
            calls.append((start,end))
            return []
    # Simulate a collector that stopped an hour ago: resuming must not silently
    # fetch a huge backfill window; it reverts to the configured lookback.
    s.set_social_cursor('SOLUSDT',datetime.now(timezone.utc)-timedelta(hours=1))
    asyncio.run(collect_cycle(s,'SOLUSDT',M(),W(),HeuristicSemanticEngine(),5,60))
    assert abs((calls[0][1]-calls[0][0]).total_seconds()-300)<2


class _AliasResolvingEngine:
    """Simulates Jev: queried under an alias before the first call, stores
    semantics under the provider-reported effective model afterwards."""
    model = "jev-latest"

    def __init__(self):
        self.effective_model = None

    @property
    def query_model(self):
        return self.effective_model or self.model

    async def classify(self, p):
        self.effective_model = "jev-1.13.0"
        return PostSemantics(
            post_id=p.post_id, symbol=p.symbol, model="jev-1.13.0",
            relevance="RELEVANT", trade_intent="LONG", market_impact_score=0.6,
            promotional_prob=0.1, new_information_prob=0.9,
        )


def test_collect_cycle_features_use_resolved_model(tmp_path):
    """Semantics are stored under the effective model (jev-1.13.0), not the
    requested alias (jev-latest). If the feature build keeps the stale alias,
    every semantic feature silently reads zero despite data existing. The same
    cycle's post is still correctly as_of-excluded (classified after market.ts);
    the next cycle must see it."""
    s=Store(str(tmp_path/'m.db'))
    e=_AliasResolvingEngine()
    r1=asyncio.run(collect_cycle(s,'SOLUSDT',M(),T(),e,5))
    assert r1.posts_semanticized==1
    r2=asyncio.run(collect_cycle(s,'SOLUSDT',M(),T(),e,5))
    # Feature reads must address the model the provider actually served.
    assert r2.feature.values["intent_mean_15m"] > 0
    assert r2.feature.values["new_info_rate_15m"] > 0
