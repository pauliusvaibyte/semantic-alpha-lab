from datetime import datetime, timedelta, timezone

from semantic_alpha.schema import PostSemantics, SocialPost
from semantic_alpha.storage import Store


def test_one_external_post_can_map_to_multiple_assets_with_asset_specific_semantics(tmp_path):
    store = Store(str(tmp_path / "multi.db"))
    t = datetime.now(timezone.utc)
    raw = {"_usage": {"provider": "twitterapi.io", "category": "post_read", "reference_id": "post-1", "units": 1, "unit_name": "post", "estimated_usd": 0.00015}}

    btc = SocialPost(post_id="post-1", symbol="BTCUSDT", author_username="pairtrader", text="Long ETH vs BTC", created_at=t, first_seen_at=t, raw=raw)
    eth = btc.model_copy(update={"symbol": "ETHUSDT"})

    assert store.save_post(btc)
    assert store.save_post(eth)  # new asset mapping, even though canonical post already exists
    assert not store.save_post(eth)

    with store.conn() as c:
        assert c.execute("SELECT COUNT(*) FROM posts").fetchone()[0] == 1
        assert c.execute("SELECT COUNT(*) FROM post_assets").fetchone()[0] == 2
        # One provider read, not double charged because the post is shared across assets.
        assert c.execute("SELECT COUNT(*) FROM api_usage WHERE provider='twitterapi.io'").fetchone()[0] == 1

    start, end = t - timedelta(seconds=1), t + timedelta(seconds=1)
    assert [p.symbol for p in store.posts_between("BTCUSDT", start, end)] == ["BTCUSDT"]
    assert [p.symbol for p in store.posts_between("ETHUSDT", start, end)] == ["ETHUSDT"]

    store.save_semantics(PostSemantics(post_id="post-1", symbol="BTCUSDT", model="jev-x", question_version="v2", trade_intent="SHORT"))
    store.save_semantics(PostSemantics(post_id="post-1", symbol="ETHUSDT", model="jev-x", question_version="v2", trade_intent="LONG"))

    assert store.semantics_for_post("post-1", "jev-x", "BTCUSDT")[0].trade_intent == "SHORT"
    assert store.semantics_for_post("post-1", "jev-x", "ETHUSDT")[0].trade_intent == "LONG"
    assert len(store.semantics_for_post("post-1", "jev-x")) == 2
    assert store.unsemanticized_posts("BTCUSDT", "jev-x") == []
    assert store.unsemanticized_posts("ETHUSDT", "jev-x") == []


def test_legacy_rows_migrate_to_asset_mapping_and_asset_semantics(tmp_path):
    # Build a DB with the legacy tables, then initialize Store and verify migration.
    import sqlite3, json
    path = tmp_path / "legacy.db"
    c = sqlite3.connect(path)
    c.executescript("""
        CREATE TABLE posts (
            post_id TEXT PRIMARY KEY, platform TEXT NOT NULL, symbol TEXT NOT NULL,
            author_id TEXT, author_username TEXT, text TEXT NOT NULL,
            created_at TEXT NOT NULL, first_seen_at TEXT NOT NULL,
            likes INTEGER, reposts INTEGER, replies INTEGER, followers INTEGER,
            verified INTEGER, urls_json TEXT NOT NULL, raw_json TEXT NOT NULL
        );
        CREATE TABLE post_semantics (
            post_id TEXT NOT NULL, symbol TEXT NOT NULL, model TEXT NOT NULL,
            question_version TEXT NOT NULL, classified_at TEXT NOT NULL,
            payload_json TEXT NOT NULL, PRIMARY KEY(post_id, model, question_version)
        );
    """)
    t = datetime.now(timezone.utc)
    p = SocialPost(post_id="legacy-1", symbol="SOLUSDT", text="long SOL", created_at=t, first_seen_at=t)
    sem = PostSemantics(post_id="legacy-1", symbol="SOLUSDT", model="legacy", trade_intent="LONG")
    c.execute("INSERT INTO posts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
        p.post_id,p.platform,p.symbol,p.author_id,p.author_username,p.text,p.created_at.isoformat(),p.first_seen_at.isoformat(),
        p.likes,p.reposts,p.replies,p.followers,None,json.dumps(p.urls),json.dumps(p.raw),
    ))
    c.execute("INSERT INTO post_semantics VALUES (?,?,?,?,?,?)", (sem.post_id,sem.symbol,sem.model,sem.question_version,sem.classified_at.isoformat(),sem.model_dump_json()))
    c.commit(); c.close()

    store = Store(str(path))
    with store.conn() as c2:
        assert c2.execute("SELECT COUNT(*) FROM post_assets WHERE post_id='legacy-1' AND symbol='SOLUSDT'").fetchone()[0] == 1
        assert c2.execute("SELECT COUNT(*) FROM post_semantics_asset WHERE post_id='legacy-1' AND symbol='SOLUSDT'").fetchone()[0] == 1
    assert store.semantics_for_post("legacy-1", "legacy", "SOLUSDT")[0].trade_intent == "LONG"
