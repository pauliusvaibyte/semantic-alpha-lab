import asyncio
from datetime import datetime, timezone
from semantic_alpha.schema import SocialPost
from semantic_alpha.semantics.heuristic import HeuristicSemanticEngine

def test_explicit_intent():
    p=SocialPost(post_id='1',symbol='SOLUSDT',text="I'm buying $SOL here after the partnership",created_at=datetime.now(timezone.utc))
    s=asyncio.run(HeuristicSemanticEngine().classify(p))
    assert s.trade_intent == 'LONG'
    assert s.explicit_recommendation_prob > .7
    assert s.catalyst_type == 'PARTNERSHIP'
