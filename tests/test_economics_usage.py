from datetime import datetime, timezone
from semantic_alpha.schema import ApiUsageRecord, SocialPost, PostSemantics
from semantic_alpha.storage import Store
from semantic_alpha.economics import usage_summary


def test_usage_records_deduplicate_and_sum(tmp_path):
    s=Store(str(tmp_path/'u.db'))
    now=datetime.now(timezone.utc)
    u=ApiUsageRecord(provider='typesafe',category='semantic_classification',reference_id='x',ts=now,units=1000,unit_name='input_token',estimated_usd=.000042)
    s.save_usage(u); s.save_usage(u)
    r=usage_summary(s,days=30,now=now)
    assert r['records']==1
    assert abs(r['variable_usd']-.000042)<1e-12


def test_post_and_semantics_auto_record_usage(tmp_path):
    s=Store(str(tmp_path/'a.db')); now=datetime.now(timezone.utc)
    p=SocialPost(post_id='p1',symbol='SOLUSDT',text='x',created_at=now,first_seen_at=now,raw={'_usage':{'provider':'twitterapi.io','category':'post_read','reference_id':'p1','units':1,'unit_name':'post','estimated_usd':.00015,'cost_source':'configured_rate'}})
    assert s.save_post(p)
    sem=PostSemantics(post_id='p1',symbol='SOLUSDT',model='jev-x',classified_at=now,raw={'_usage':{'provider':'typesafe','category':'semantic_classification','reference_id':'p1:jev-x:v1','units':100,'unit_name':'input_token','estimated_usd':.0000042,'cost_source':'estimated'}})
    s.save_semantics(sem)
    assert len(s.usage_records())==2
