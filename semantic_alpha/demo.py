from __future__ import annotations
from datetime import datetime, timedelta, timezone
import math
import random

from .features import FeatureEngine
from .clustering import EventClusterer
from .labels import build_labels
from .schema import EngagementSnapshot, MarketSnapshot, SocialPost
from .semantics.heuristic import HeuristicSemanticEngine
from .storage import Store


async def seed_demo(store: Store, minutes: int = 360) -> None:
    random.seed(7)
    start = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    sem = HeuristicSemanticEngine()
    price = 150.0
    for i in range(minutes + 250):
        ts = start + timedelta(minutes=i)
        drift = 0.00015 * math.sin(i / 25) + random.gauss(0, 0.0012)
        price *= (1 + drift)
        shock = i in {100, 220, 310}
        if shock:
            for j in range(12):
                p = SocialPost(
                    post_id=f"demo-{i}-{j}", symbol="SOLUSDT", author_username=f"u{j}",
                    text=("Breaking Solana protocol partnership announced, I'm buying $SOL here" if j < 8 else "SOL bullish breakout"),
                    created_at=ts + timedelta(seconds=j), first_seen_at=ts + timedelta(seconds=j+1),
                )
                store.save_post(p); store.save_engagement(EngagementSnapshot(post_id=p.post_id,symbol=p.symbol,observed_at=p.first_seen_at,likes=j,reposts=0,replies=0)); store.save_engagement(EngagementSnapshot(post_id=p.post_id,symbol=p.symbol,observed_at=p.first_seen_at+timedelta(minutes=2),likes=j+10,reposts=2,replies=1)); store.save_semantics((await sem.classify(p)).model_copy(update={"classified_at":p.first_seen_at}))
            price *= 1.004
        elif random.random() < 0.12:
            p = SocialPost(post_id=f"demo-{i}", symbol="SOLUSDT", author_username=f"u{random.randrange(20)}", text="Watching $SOL, market looks mixed", created_at=ts, first_seen_at=ts)
            store.save_post(p); store.save_engagement(EngagementSnapshot(post_id=p.post_id,symbol=p.symbol,observed_at=p.first_seen_at,likes=0,reposts=0,replies=0)); store.save_engagement(EngagementSnapshot(post_id=p.post_id,symbol=p.symbol,observed_at=p.first_seen_at+timedelta(minutes=2),likes=2,reposts=0,replies=0)); store.save_semantics((await sem.classify(p)).model_copy(update={"classified_at":p.first_seen_at}))
        m = MarketSnapshot(
            symbol="SOLUSDT", ts=ts, last=price, bid=price*0.99995, ask=price*1.00005,
            funding_rate=random.gauss(0.0001,0.00005), open_interest_value=1_000_000_000*(1+i/10000),
            spread_bps=1.0, book_imbalance=random.uniform(-0.2,0.2), taker_buy_ratio=random.uniform(0.4,0.6), volume_24h=2e9,
        )
        store.save_market(m)
        # market benchmark to support abnormal labels
        store.save_market(MarketSnapshot(symbol="BTCUSDT", ts=ts, last=65000*(1+i*0.00001), spread_bps=0.5))
        if i < minutes:
            f = FeatureEngine(store, semantic_model=sem.model).build("SOLUSDT", ts + timedelta(seconds=59), m)
            store.save_features(f)
    for f in store.all_features():
        l = build_labels(store, f.symbol, f.ts)
        if l:
            store.save_label(l)

    for p in store.all_posts():
        l = build_labels(store, p.symbol, p.created_at)
        if l:
            store.save_label(l)

    # Exercise the same narrative-event abstraction used by live/historical paths.
    for event in EventClusterer().cluster(store.all_posts("SOLUSDT")):
        store.save_event(event)
