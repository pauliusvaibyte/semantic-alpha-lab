from datetime import datetime, timedelta, timezone

from semantic_alpha.author_alpha import estimate_author_alpha, snapshot_author_reputations
from semantic_alpha.schema import LabelRecord, PostSemantics, SocialPost
from semantic_alpha.storage import Store


def test_author_reputation_excludes_unmatured_call(tmp_path):
    s=Store(str(tmp_path/'a.db')); t=datetime(2026,1,1,tzinfo=timezone.utc)
    for i,minute in enumerate((0,10)):
        p=SocialPost(post_id=str(i),symbol='SOLUSDT',author_username='alice',text='long sol',created_at=t+timedelta(minutes=minute),first_seen_at=t+timedelta(minutes=minute))
        s.save_post(p)
        s.save_semantics(PostSemantics(post_id=p.post_id,symbol=p.symbol,model='x',trade_intent='LONG',explicit_recommendation_prob=.9,classified_at=p.first_seen_at))
        s.save_label(LabelRecord(symbol=p.symbol,ts=p.created_at,abnormal_returns={'15m':.01},returns={'15m':.01}))
    a=estimate_author_alpha(s,'alice','SOLUSDT','15m',before=t+timedelta(minutes=20),semantic_model='x')
    assert a.samples==1
    reps=snapshot_author_reputations(s,t+timedelta(minutes=20),horizon='15m',semantic_model='x',min_posts=1)
    assert len(reps)==1
    assert s.author_reputation_for('alice','SOLUSDT',t+timedelta(minutes=21),'15m').samples==1
