import asyncio
from datetime import datetime, timezone, timedelta
from semantic_alpha.storage import Store
from semantic_alpha.schema import SocialPost, MarketSnapshot
from semantic_alpha.semantics.heuristic import HeuristicSemanticEngine
from semantic_alpha.labels import build_labels, post_actionable_ts
from semantic_alpha.author_alpha import estimate_author_alpha

def test_author_alpha_forward_only(tmp_path):
    s=Store(str(tmp_path/'a.db')); e=HeuristicSemanticEngine(); t=datetime.now(timezone.utc)
    for i in range(5):
        ts=t+timedelta(minutes=i*20)
        p=SocialPost(post_id=str(i),symbol='SOLUSDT',author_username='alice',text="I'm buying $SOL here",created_at=ts,first_seen_at=ts)
        # Pin classification to the post timeline — the anchor convention treats
        # max(first_seen, classified_at) as the tradable instant.
        s.save_post(p); s.save_semantics((asyncio.run(e.classify(p))).model_copy(update={"classified_at":p.first_seen_at}))
        s.save_market(MarketSnapshot(symbol='SOLUSDT',ts=ts,last=100+i))
        s.save_market(MarketSnapshot(symbol='SOLUSDT',ts=ts+timedelta(minutes=15),last=101+i))
        s.save_market(MarketSnapshot(symbol='BTCUSDT',ts=ts,last=100))
        s.save_market(MarketSnapshot(symbol='BTCUSDT',ts=ts+timedelta(minutes=15),last=100))
        # Labels anchor at the actionable timestamp (max of observation and
        # classification), matching estimate_author_alpha's lookup convention.
        l=build_labels(s,'SOLUSDT',post_actionable_ts(s,p,'SOLUSDT',model=e.model)); s.save_label(l)
    a=estimate_author_alpha(s,'alice','SOLUSDT','15m',semantic_model=e.model,prior_strength=5)
    assert a.samples==5
    assert a.posterior_edge>0
