from datetime import datetime, timezone, timedelta
from semantic_alpha.schema import SocialPost
from semantic_alpha.clustering import EventClusterer

def test_similar_posts_cluster():
    t=datetime.now(timezone.utc)
    ps=[
      SocialPost(post_id='1',symbol='SOLUSDT',text='Solana major exchange listing announced today',created_at=t),
      SocialPost(post_id='2',symbol='SOLUSDT',text='Major Solana exchange listing announced today',created_at=t+timedelta(seconds=20)),
      SocialPost(post_id='3',symbol='SOLUSDT',text='I like the weather',created_at=t+timedelta(seconds=30)),
    ]
    cs=EventClusterer(similarity=.3).cluster(ps)
    assert len(cs)==2
    assert len(cs[0].post_ids)==2
