from datetime import datetime, timezone

from semantic_alpha.audit import leakage_audit
from semantic_alpha.clustering import EventClusterer
from semantic_alpha.features import FeatureEngine
from semantic_alpha.ranking import cross_sectional_ranks
from semantic_alpha.schema import EngagementSnapshot, FeatureSnapshot, MarketSnapshot, SocialPost
from semantic_alpha.storage import Store


def dt(m=0):
    return datetime(2026, 9, 20, 0, m, tzinfo=timezone.utc)


def test_engagement_velocity_and_narrative_features(tmp_path):
    s=Store(str(tmp_path/'x.db'))
    p1=SocialPost(post_id='1',symbol='SOLUSDT',author_username='a',text='Breaking Solana exchange listing confirmed',created_at=dt(1),first_seen_at=dt(1))
    p2=SocialPost(post_id='2',symbol='SOLUSDT',author_username='b',text='Solana exchange listing is now confirmed',created_at=dt(2),first_seen_at=dt(2))
    for p in (p1,p2): s.save_post(p)
    s.save_engagement(EngagementSnapshot(post_id='1',symbol='SOLUSDT',observed_at=dt(2),likes=1,reposts=0,replies=0))
    s.save_engagement(EngagementSnapshot(post_id='1',symbol='SOLUSDT',observed_at=dt(4),likes=21,reposts=3,replies=2))
    for e in EventClusterer().cluster([p1,p2]): s.save_event(e)
    m=MarketSnapshot(symbol='SOLUSDT',ts=dt(5),last=150,bid=149.99,ask=150.01)
    s.save_market(m)
    f=FeatureEngine(s).build('SOLUSDT',dt(5),m)
    assert f.values['engagement_velocity_max_15m'] > 0
    assert f.values['events_15m'] == 1
    assert f.values['event_author_breadth_15m'] == 2
    assert 0 <= f.values['narrative_novelty_15m'] <= 1


def test_cross_sectional_ranking(tmp_path):
    s=Store(str(tmp_path/'r.db'))
    for sym,gap in [('AUSDT',3.0),('BUSDT',0.0),('CUSDT',-2.0)]:
        s.save_features(FeatureSnapshot(symbol=sym,ts=dt(1),values={
            'information_price_gap_proxy':gap,'attention_z_5m':gap/2,'narrative_novelty_15m':.8 if gap>0 else .1,
            'semantic_shock':gap,'crowding_proxy':0,'panic_squeeze_proxy':0,
        }))
    d=cross_sectional_ranks(s)
    assert d.iloc[0].symbol == 'AUSDT'
    assert d.iloc[-1].symbol == 'CUSDT'


def test_leakage_audit_clean(tmp_path):
    s=Store(str(tmp_path/'a.db'))
    p=SocialPost(post_id='x',symbol='BTCUSDT',text='x',created_at=dt(0),first_seen_at=dt(1))
    s.save_post(p)
    s.save_engagement(EngagementSnapshot(post_id='x',symbol='BTCUSDT',observed_at=dt(1),likes=1))
    assert leakage_audit(s)['ok']
