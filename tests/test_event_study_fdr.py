from datetime import datetime, timedelta, timezone

from semantic_alpha.clustering import EventClusterer
from semantic_alpha.event_study import benjamini_hochberg, narrative_event_frame, event_study
from semantic_alpha.schema import LabelRecord, PostSemantics, SocialPost
from semantic_alpha.storage import Store


def _post(i: int, text: str, ts: datetime) -> SocialPost:
    return SocialPost(
        platform="x", post_id=f"p{i}", symbol="SOLUSDT", author_username=f"a{i}",
        text=text, created_at=ts, first_seen_at=ts,
    )


def test_bh_is_monotone_and_bounded():
    q=benjamini_hochberg([.001,.02,.03,None,.8])
    assert q[0] <= q[1] <= q[2] <= q[4]
    assert q[3] is None
    assert all(0 <= x <= 1 for x in q if x is not None)


def test_event_frame_deduplicates_copies(tmp_path):
    s=Store(str(tmp_path/'x.db'))
    t=datetime(2026,1,1,tzinfo=timezone.utc)
    posts=[
        _post(1,"Coinbase lists ABC tomorrow",t),
        _post(2,"Coinbase lists ABC tomorrow",t+timedelta(seconds=10)),
        _post(3,"Coinbase lists ABC tomorrow",t+timedelta(seconds=20)),
        _post(4,"SOL breakout looks strong",t+timedelta(minutes=10)),
    ]
    for p in posts:
        s.save_post(p)
        s.save_semantics(PostSemantics(
            post_id=p.post_id,symbol=p.symbol,model="m",communication_type="CATALYST_NEWS" if p.post_id!="p4" else "MARKET_ANALYSIS",
            trade_intent="LONG",catalyst_type="LISTING" if p.post_id!="p4" else "TECHNICAL",classified_at=p.first_seen_at,
            new_information_prob=.9,explicit_recommendation_prob=.7,market_impact_score=3.0,confidence=.9,
        ))
        s.save_label(LabelRecord(symbol=p.symbol,ts=p.created_at,returns={"15m":.01},abnormal_returns={"15m":.01},max_up={"15m":.02},max_down={"15m":-.004}))
    events=EventClusterer(similarity=.30).cluster(posts)
    for e in events: s.save_event(e)
    frame=narrative_event_frame(s,"15m","m")
    # Three duplicated listing posts must be one inferential observation.
    assert len(frame) == 2
    listing=frame[frame.catalyst_type=="LISTING"].iloc[0]
    assert listing.n_posts == 3
    result=event_study(s,"15m","m",min_n=1)
    assert "q_value_fdr" in result.columns
    assert "n_events" in result.columns
