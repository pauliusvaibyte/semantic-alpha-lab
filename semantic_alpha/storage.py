from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .schema import ApiUsageRecord, AuthorReputationRecord, EngagementSnapshot, FeatureSnapshot, LabelRecord, MarketSnapshot, PaperTrade, PostSemantics, SocialPost


# Venue sources that count as canonical market truth. Backfilled Bybit rows are
# tagged 'bybit_history' and live Bybit stream rows 'bybit_ws' — all are Bybit
# data; other venues (binance, ...) are context only and never leak into
# canonical reads unless explicitly requested. Without 'bybit_ws' here, a
# live-only database would have zero canonical market history: returns,
# realized vol, and label outcomes would all read as missing.
CANONICAL_MARKET_SOURCES = ("bybit", "bybit_history", "bybit_ws")


def _src_clause(source: str | None) -> tuple[str, tuple]:
    if source is None:
        return f"source IN ({','.join('?' * len(CANONICAL_MARKET_SOURCES))})", CANONICAL_MARKET_SOURCES
    return "source=?", (source,)


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _merge_label(existing: LabelRecord, new: LabelRecord) -> LabelRecord:
    """Fill still-unrealized horizons of ``existing`` from ``new``.

    Realized values in ``existing`` win: a rebuild must never replace a matured
    label with None because of a transient data gap. Horizon keys present only
    on one side are kept."""
    out = existing.model_copy()
    for attr in ("returns", "abnormal_returns", "realized_minutes"):
        merged = dict(getattr(new, attr, {}) or {})
        for k, v in (getattr(existing, attr, {}) or {}).items():
            if v is not None or k not in merged:
                merged[k] = v
        setattr(out, attr, merged)
    # Path statistics are only final once their horizon realized: a value
    # computed over a partial window (e.g. a "24h" max_down measured on 30
    # minutes of data) must be recomputed as the window completes, not frozen.
    for attr in ("max_up", "max_down", "abs_move", "range_move", "fade_ratio"):
        merged = dict(getattr(new, attr, {}) or {})
        for k, v in (getattr(existing, attr, {}) or {}).items():
            horizon_realized = (existing.returns or {}).get(k) is not None
            if (v is not None and horizon_realized) or k not in merged:
                merged[k] = v
        setattr(out, attr, merged)
    # Prefer the earliest established entry anchor; keep newest when old lacked it.
    if new.base_ts is not None and (existing.base_ts is None or new.base_ts <= existing.base_ts):
        out.base_ts = new.base_ts
        out.entry_delay_seconds = new.entry_delay_seconds
    return out


class Store:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.init()

    @contextmanager
    def conn(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    def init(self) -> None:
        with self.conn() as c:
            c.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS api_usage (
                    provider TEXT NOT NULL,
                    category TEXT NOT NULL,
                    reference_id TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    estimated_usd REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY(provider, category, reference_id)
                );
                CREATE INDEX IF NOT EXISTS idx_api_usage_time ON api_usage(ts, provider);
                CREATE TABLE IF NOT EXISTS posts (
                    post_id TEXT PRIMARY KEY,
                    platform TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    author_id TEXT,
                    author_username TEXT,
                    text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    first_seen_at TEXT NOT NULL,
                    likes INTEGER,
                    reposts INTEGER,
                    replies INTEGER,
                    followers INTEGER,
                    verified INTEGER,
                    urls_json TEXT NOT NULL,
                    raw_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_posts_symbol_time ON posts(symbol, created_at);
                CREATE TABLE IF NOT EXISTS post_assets (
                    post_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    PRIMARY KEY(post_id, symbol)
                );
                CREATE INDEX IF NOT EXISTS idx_post_assets_symbol ON post_assets(symbol, post_id);

                CREATE TABLE IF NOT EXISTS engagement_snapshots (
                    post_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    likes INTEGER NOT NULL,
                    reposts INTEGER NOT NULL,
                    replies INTEGER NOT NULL,
                    PRIMARY KEY(post_id, symbol, observed_at)
                );
                CREATE INDEX IF NOT EXISTS idx_engagement_symbol_time ON engagement_snapshots(symbol, observed_at);

                CREATE TABLE IF NOT EXISTS post_semantics (
                    post_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    model TEXT NOT NULL,
                    question_version TEXT NOT NULL,
                    classified_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY(post_id, model, question_version)
                );
                CREATE INDEX IF NOT EXISTS idx_sem_symbol ON post_semantics(symbol);
                CREATE TABLE IF NOT EXISTS post_semantics_asset (
                    post_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    model TEXT NOT NULL,
                    question_version TEXT NOT NULL,
                    classified_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY(post_id, symbol, model, question_version)
                );
                CREATE INDEX IF NOT EXISTS idx_sem_asset_symbol ON post_semantics_asset(symbol, model, classified_at);

                CREATE TABLE IF NOT EXISTS market_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_market_symbol_time ON market_snapshots(symbol, ts);

                CREATE TABLE IF NOT EXISTS feature_snapshots (
                    symbol TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    feature_version TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY(symbol, ts, feature_version)
                );

                CREATE TABLE IF NOT EXISTS regime_pulses (
                    symbol TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    model TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY(symbol, ts, model)
                );

                CREATE TABLE IF NOT EXISTS classification_failures (
                    post_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    model TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    last_ts TEXT NOT NULL,
                    PRIMARY KEY(post_id, symbol, model)
                );

                CREATE TABLE IF NOT EXISTS labels (
                    symbol TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    label_version TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    matured INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(symbol, ts, label_version)
                );

                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    representative_text TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS event_members (
                    event_id TEXT NOT NULL,
                    post_id TEXT NOT NULL,
                    PRIMARY KEY(event_id, post_id)
                );
                CREATE TABLE IF NOT EXISTS paper_trades (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    opened_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_paper_trades_status ON paper_trades(status, opened_at);
                CREATE TABLE IF NOT EXISTS holdout_registry (
                    holdout_fingerprint TEXT PRIMARY KEY,
                    horizon TEXT NOT NULL,
                    opened_at TEXT NOT NULL,
                    protocol_hash TEXT NOT NULL,
                    reopened INTEGER NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS author_reputation (
                    author TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    horizon TEXT NOT NULL,
                    as_of TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY(author, symbol, horizon, as_of)
                );
                CREATE INDEX IF NOT EXISTS idx_author_rep_lookup ON author_reputation(author, symbol, horizon, as_of);
                CREATE TABLE IF NOT EXISTS social_cursors (
                    symbol TEXT PRIMARY KEY,
                    watermark TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                -- Append-only experiment ledger: every research run is a counted
                -- trial for Deflated Sharpe / PBO. No delete path exists.
                CREATE TABLE IF NOT EXISTS experiment_trials (
                    fingerprint TEXT PRIMARY KEY,
                    ts TEXT NOT NULL,
                    command TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                -- Feed liveness telemetry: one row per symbol per observed
                -- transition (connect/disconnect/stale/recovered) so a dead
                -- feed can never masquerade as a healthy one.
                CREATE TABLE IF NOT EXISTS feed_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_feed_health_symbol_time ON feed_health(symbol, ts);
                -- Small key/value state (maintenance watermarks, etc.).
                CREATE TABLE IF NOT EXISTS meta_kv (
                    k TEXT PRIMARY KEY,
                    v TEXT NOT NULL
                );
                """
            )
            # Backward-compatible migrations: canonical post rows remain one-per external
            # post, while asset mappings and semantic interpretations are many-per-post.
            c.execute("INSERT OR IGNORE INTO post_assets(post_id,symbol) SELECT post_id,symbol FROM posts")
            c.execute(
                """INSERT OR IGNORE INTO post_semantics_asset
                   (post_id,symbol,model,question_version,classified_at,payload_json)
                   SELECT post_id,symbol,model,question_version,classified_at,payload_json FROM post_semantics"""
            )
            # labels.matured: 1 when every configured horizon has a realized value.
            # Immature rows are revisited by maintenance until they fill.
            lcols = {r[1] for r in c.execute("PRAGMA table_info(labels)").fetchall()}
            if "matured" not in lcols:
                c.execute("ALTER TABLE labels ADD COLUMN matured INTEGER NOT NULL DEFAULT 0")
            # Engagement is keyed per (post, asset, observation): a post mapped to
            # multiple symbols keeps a row per asset context.
            ecols = c.execute("PRAGMA table_info(engagement_snapshots)").fetchall()
            pk_cols = sorted(r[1] for r in ecols if r[5])
            if pk_cols == ["observed_at", "post_id"]:
                c.execute(
                    """CREATE TABLE IF NOT EXISTS engagement_snapshots_v2 (
                        post_id TEXT NOT NULL, symbol TEXT NOT NULL, observed_at TEXT NOT NULL,
                        likes INTEGER NOT NULL, reposts INTEGER NOT NULL, replies INTEGER NOT NULL,
                        PRIMARY KEY(post_id, symbol, observed_at))"""
                )
                c.execute(
                    """INSERT OR IGNORE INTO engagement_snapshots_v2
                       SELECT post_id,symbol,observed_at,likes,reposts,replies FROM engagement_snapshots"""
                )
                c.execute("DROP TABLE engagement_snapshots")
                c.execute("ALTER TABLE engagement_snapshots_v2 RENAME TO engagement_snapshots")
                c.execute("CREATE INDEX IF NOT EXISTS idx_engagement_symbol_time ON engagement_snapshots(symbol, observed_at)")
            # Market rows are idempotent on (source, symbol, ts). Older DBs may
            # hold duplicates; collapse them once, then enforce uniqueness.
            if not c.execute("SELECT 1 FROM sqlite_master WHERE type='index' AND name='uq_market_source_symbol_ts'").fetchone():
                c.execute(
                    """DELETE FROM market_snapshots WHERE id NOT IN (
                        SELECT MAX(id) FROM market_snapshots GROUP BY source,symbol,ts)"""
                )
                c.execute("CREATE UNIQUE INDEX uq_market_source_symbol_ts ON market_snapshots(source,symbol,ts)")

    def save_usage(self, u: ApiUsageRecord) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO api_usage(provider,category,reference_id,ts,estimated_usd,payload_json) VALUES (?,?,?,?,?,?)",
                (u.provider,u.category,u.reference_id,_iso(u.ts),u.estimated_usd,u.model_dump_json()),
            )

    def usage_records(self, start: datetime | None = None, end: datetime | None = None) -> list[ApiUsageRecord]:
        sql="SELECT payload_json FROM api_usage WHERE 1=1"; args=[]
        if start is not None:
            sql += " AND ts>=?"; args.append(_iso(start))
        if end is not None:
            sql += " AND ts<?"; args.append(_iso(end))
        sql += " ORDER BY ts"
        with self.conn() as c:
            rows=c.execute(sql,args).fetchall()
        return [ApiUsageRecord.model_validate_json(r[0]) for r in rows]

    def save_post(self, p: SocialPost) -> bool:
        with self.conn() as c:
            cur = c.execute(
                """INSERT OR IGNORE INTO posts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    p.post_id, p.platform, p.symbol, p.author_id, p.author_username, p.text,
                    _iso(p.created_at), _iso(p.first_seen_at), p.likes, p.reposts, p.replies,
                    p.followers, int(p.verified) if p.verified is not None else None,
                    json.dumps(p.urls), json.dumps(p.raw),
                ),
            )
            canonical_inserted = cur.rowcount > 0
            map_cur = c.execute(
                "INSERT OR IGNORE INTO post_assets(post_id,symbol) VALUES (?,?)",
                (p.post_id, p.symbol),
            )
            mapping_inserted = map_cur.rowcount > 0
        usage=(p.raw or {}).get("_usage") if isinstance(p.raw,dict) else None
        # Usage is keyed by provider acquisition identity, not by whether the
        # canonical post was new. Separate paid searches can legitimately return
        # the same post; save_usage deduplicates a single delivery/reference.
        if isinstance(usage,dict):
            self.save_usage(ApiUsageRecord(
                provider=str(usage.get("provider","twitterapi.io")),category=str(usage.get("category","post_read")),
                reference_id=str(usage.get("reference_id",p.post_id)),ts=p.first_seen_at,units=float(usage.get("units",1.0)),
                unit_name=str(usage.get("unit_name","post")),estimated_usd=float(usage.get("estimated_usd",0.0)),
                cost_source=str(usage.get("cost_source","estimated")),metadata={k:v for k,v in usage.items() if k not in {"provider","category","reference_id","units","unit_name","estimated_usd","cost_source"}},
            ))
        return canonical_inserted or mapping_inserted


    def get_social_cursor(self, symbol: str) -> datetime | None:
        """High-water mark for forward social collection: end of the last fetched window."""
        with self.conn() as c:
            r = c.execute("SELECT watermark FROM social_cursors WHERE symbol=?", (symbol,)).fetchone()
        return datetime.fromisoformat(r[0]) if r else None

    def set_social_cursor(self, symbol: str, watermark: datetime) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO social_cursors(symbol,watermark,updated_at) VALUES (?,?,?)",
                (symbol, _iso(watermark), _iso(datetime.now(timezone.utc))),
            )

    def meta_get(self, key: str) -> str | None:
        with self.conn() as c:
            r = c.execute("SELECT v FROM meta_kv WHERE k=?", (key,)).fetchone()
        return str(r[0]) if r else None

    def meta_set(self, key: str, value: str) -> None:
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO meta_kv(k,v) VALUES (?,?)", (key, value))

    def save_feed_health(self, symbol: str, kind: str, payload: dict, ts: datetime | None = None) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT INTO feed_health(symbol,ts,kind,payload_json) VALUES (?,?,?,?)",
                (symbol, _iso(ts or datetime.now(timezone.utc)), kind, json.dumps(payload, default=str)),
            )

    def feed_health_between(self, symbol: str, start: datetime, end: datetime) -> list[dict]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT ts,kind,payload_json FROM feed_health WHERE symbol=? AND ts>=? AND ts<? ORDER BY ts",
                (symbol, _iso(start), _iso(end)),
            ).fetchall()
        return [{"ts": r[0], "kind": r[1], **json.loads(r[2])} for r in rows]

    def record_trial(self, fingerprint: str, command: str, payload: dict) -> bool:
        """Append one experiment to the trial ledger. Returns False if this exact
        fingerprint was already recorded — reruns do not inflate the count."""
        import hashlib as _h
        fp = fingerprint or _h.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:24]
        with self.conn() as c:
            cur = c.execute(
                "INSERT OR IGNORE INTO experiment_trials(fingerprint,ts,command,payload_json) VALUES (?,?,?,?)",
                (fp, _iso(datetime.now(timezone.utc)), command, json.dumps(payload, default=str)),
            )
            return cur.rowcount > 0

    def trial_count(self) -> int:
        with self.conn() as c:
            return int(c.execute("SELECT COUNT(*) FROM experiment_trials").fetchone()[0])

    def trials(self, limit: int = 500) -> list[dict]:
        with self.conn() as c:
            rows = c.execute("SELECT fingerprint,ts,command,payload_json FROM experiment_trials ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [{"fingerprint": r[0], "ts": r[1], "command": r[2], **json.loads(r[3])} for r in rows]

    def save_engagement(self, e: EngagementSnapshot) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO engagement_snapshots(post_id,symbol,observed_at,likes,reposts,replies) VALUES (?,?,?,?,?,?)",
                (e.post_id, e.symbol, _iso(e.observed_at), e.likes, e.reposts, e.replies),
            )

    def engagement_for_post(self, post_id: str, end: datetime | None = None) -> list[EngagementSnapshot]:
        sql = "SELECT * FROM engagement_snapshots WHERE post_id=?"
        args: list[object] = [post_id]
        if end is not None:
            sql += " AND observed_at<=?"
            args.append(_iso(end))
        sql += " ORDER BY observed_at"
        with self.conn() as c:
            rows = c.execute(sql, args).fetchall()
        return [EngagementSnapshot(
            post_id=r["post_id"], symbol=r["symbol"], observed_at=datetime.fromisoformat(r["observed_at"]),
            likes=r["likes"], reposts=r["reposts"], replies=r["replies"]
        ) for r in rows]

    def latest_engagement_for_posts(self, post_ids: list[str], end: datetime) -> dict[str, EngagementSnapshot]:
        out: dict[str, EngagementSnapshot] = {}
        for pid in post_ids:
            rows = self.engagement_for_post(pid, end=end)
            if rows:
                out[pid] = rows[-1]
        return out

    def save_semantics(self, s: PostSemantics) -> None:
        with self.conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO post_semantics_asset
                   (post_id,symbol,model,question_version,classified_at,payload_json)
                   VALUES (?,?,?,?,?,?)""",
                (s.post_id, s.symbol, s.model, s.question_version, _iso(s.classified_at), s.model_dump_json()),
            )
        usage=(s.raw or {}).get("_usage") if isinstance(s.raw,dict) else None
        if isinstance(usage,dict):
            self.save_usage(ApiUsageRecord(
                provider=str(usage.get("provider","typesafe")),category=str(usage.get("category","semantic_classification")),
                reference_id=str(usage.get("reference_id",f"{s.post_id}:{s.symbol}:{s.model}:{s.question_version}")),ts=s.classified_at,
                units=float(usage.get("units",0.0)),unit_name=str(usage.get("unit_name","input_token")),
                estimated_usd=float(usage.get("estimated_usd",0.0)),cost_source=str(usage.get("cost_source","estimated")),
                metadata={k:v for k,v in usage.items() if k not in {"provider","category","reference_id","units","unit_name","estimated_usd","cost_source"}},
            ))


    def save_market(self, m: MarketSnapshot) -> None:
        """Idempotent on (source, symbol, ts): re-ingesting the same observation
        updates the row in place rather than duplicating it."""
        with self.conn() as c:
            c.execute(
                """INSERT INTO market_snapshots(source,symbol,ts,payload_json) VALUES (?,?,?,?)
                   ON CONFLICT(source,symbol,ts) DO UPDATE SET payload_json=excluded.payload_json""",
                (m.source, m.symbol, _iso(m.ts), m.model_dump_json()),
            )

    def save_regime_pulse(self, symbol: str, ts: datetime, model: str, payload: dict) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO regime_pulses VALUES (?,?,?,?)",
                (symbol, _iso(ts), model, json.dumps(payload, default=str)),
            )
        usage = payload.get("_usage")
        if isinstance(usage, dict):
            self.save_usage(ApiUsageRecord(
                provider=str(usage.get("provider","typesafe")),category=str(usage.get("category","regime_pulse")),
                reference_id=str(usage.get("reference_id",f"regime_pulse:{symbol}:{_iso(ts)}")),ts=ts,
                units=float(usage.get("units",0.0)),unit_name=str(usage.get("unit_name","input_token")),
                estimated_usd=float(usage.get("estimated_usd",0.0)),cost_source=str(usage.get("cost_source","estimated")),
                metadata={k:v for k,v in usage.items() if k not in {"provider","category","reference_id","units","unit_name","estimated_usd","cost_source"}},
            ))

    def regime_pulses_between(self, symbol: str, start: datetime, end: datetime, model: str | None = None) -> list[dict]:
        sql = "SELECT payload_json FROM regime_pulses WHERE symbol=? AND ts>=? AND ts<?"
        args: list[object] = [symbol, _iso(start), _iso(end)]
        if model:
            sql += " AND model=?"; args.append(model)
        sql += " ORDER BY ts"
        with self.conn() as c:
            return [json.loads(r[0]) for r in c.execute(sql, args).fetchall()]

    def save_features(self, f: FeatureSnapshot) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO feature_snapshots VALUES (?,?,?,?)",
                (f.symbol, _iso(f.ts), f.feature_version, f.model_dump_json()),
            )

    def save_label(self, l: LabelRecord) -> None:
        """Upsert a label row, merging per-horizon values with any stored row.

        Partial labels must never freeze forever: an existing value that is still
        None gets filled when later market data makes the horizon mature, while
        already-realized values are preserved (a rebuild must not erase them).
        """
        existing = self.label_for(l.symbol, l.ts, l.label_version)
        if existing is not None:
            l = _merge_label(existing, l)
        matured = int(all(v is not None for v in l.returns.values()))
        with self.conn() as c:
            c.execute(
                """INSERT INTO labels(symbol,ts,label_version,payload_json,matured) VALUES (?,?,?,?,?)
                   ON CONFLICT(symbol,ts,label_version) DO UPDATE SET
                   payload_json=excluded.payload_json, matured=excluded.matured""",
                (l.symbol, _iso(l.ts), l.label_version, l.model_dump_json(), matured),
            )

    def mark_label_matured(self, symbol: str, ts: datetime, label_version: str = "v4") -> None:
        """Finalize a label whose open horizons can never realize — the market
        data for those exits is permanently absent (gap, downtime). The horizon
        stays None (honest missing data); only the rescan flag is released."""
        with self.conn() as c:
            c.execute(
                "UPDATE labels SET matured=1 WHERE symbol=? AND ts=? AND label_version=?",
                (symbol, _iso(ts), label_version),
            )

    def immature_labels(self, label_version: str = "v4", limit: int = 2000) -> list[LabelRecord]:
        """Label rows that still have unrealized horizons; maintenance re-evaluates
        only these plus never-labeled anchors instead of scanning everything."""
        with self.conn() as c:
            rows = c.execute(
                "SELECT payload_json FROM labels WHERE label_version=? AND matured=0 ORDER BY ts LIMIT ?",
                (label_version, limit),
            ).fetchall()
        return [LabelRecord.model_validate_json(r[0]) for r in rows]

    def recent_post_symbols(self, since: datetime) -> list[str]:
        """Distinct asset symbols observed since ``since`` (MACRO excluded — it
        is a symbol bucket for unmatched RSS posts, not a tradable asset)."""
        with self.conn() as c:
            return [r[0] for r in c.execute(
                "SELECT DISTINCT symbol FROM posts WHERE first_seen_at>=? AND symbol!='MACRO'",
                (_iso(since),),
            ).fetchall()]

    def posts_between(self, symbol: str, start: datetime, end: datetime, *, as_of: datetime | None = None) -> list[SocialPost]:
        """Posts created in [start, end). With ``as_of``, additionally require the
        post to have been *available* then (first_seen_at <= as_of) — point-in-time
        correct for historical reconstruction."""
        sql = """SELECT p.*, a.symbol AS asset_symbol
                 FROM post_assets a JOIN posts p ON p.post_id=a.post_id
                 WHERE a.symbol=? AND p.created_at>=? AND p.created_at<?"""
        args: list[object] = [symbol, _iso(start), _iso(end)]
        if as_of is not None:
            sql += " AND p.first_seen_at<=?"
            args.append(_iso(as_of))
        sql += " ORDER BY p.created_at"
        with self.conn() as c:
            rows = c.execute(sql, args).fetchall()
        return [self._post(r) for r in rows]

    def post_ids_between(self, start: datetime, end: datetime) -> set[str]:
        """Post ids created in [start, end) — cheap early-stop set for
        leading-edge searches that would otherwise re-buy known pages."""
        with self.conn() as c:
            rows = c.execute(
                "SELECT post_id FROM posts WHERE created_at>=? AND created_at<?",
                (_iso(start), _iso(end)),
            ).fetchall()
        return {r[0] for r in rows}

    def semantics_between(self, symbol: str, start: datetime, end: datetime, model: str | None = None, *, as_of: datetime | None = None) -> list[PostSemantics]:
        """Semantics for posts created in [start, end). With ``as_of``, both the
        underlying post must have been observed (first_seen_at <= as_of) and the
        classification completed (classified_at <= as_of) at that instant."""
        sql = """
            SELECT s.payload_json FROM post_semantics_asset s
            JOIN posts p ON p.post_id=s.post_id
            WHERE s.symbol=? AND p.created_at>=? AND p.created_at<?
        """
        args: list[object] = [symbol, _iso(start), _iso(end)]
        if model:
            sql += " AND s.model=?"
            args.append(model)
        if as_of is not None:
            sql += " AND p.first_seen_at<=? AND s.classified_at<=?"
            args += [_iso(as_of), _iso(as_of)]
        sql += " ORDER BY p.created_at"
        with self.conn() as c:
            rows = c.execute(sql, args).fetchall()
        return [PostSemantics.model_validate_json(r[0]) for r in rows]


    def market_between(self, symbol: str, start: datetime, end: datetime, source: str | None = None) -> list[MarketSnapshot]:
        clause, sp = _src_clause(source)
        with self.conn() as c:
            rows = c.execute(
                f"SELECT payload_json FROM market_snapshots WHERE symbol=? AND {clause} AND ts>=? AND ts<? ORDER BY ts",
                (symbol, *sp, _iso(start), _iso(end)),
            ).fetchall()
        return [MarketSnapshot.model_validate_json(r[0]) for r in rows]

    def nearest_market_at_or_after(self, symbol: str, ts: datetime, max_delay_seconds: int = 120, source: str | None = None) -> MarketSnapshot | None:
        end = ts + timedelta(seconds=max_delay_seconds)
        clause, sp = _src_clause(source)
        with self.conn() as c:
            row = c.execute(
                f"SELECT payload_json FROM market_snapshots WHERE symbol=? AND {clause} AND ts>=? AND ts<=? ORDER BY ts LIMIT 1",
                (symbol, *sp, _iso(ts), _iso(end)),
            ).fetchone()
        return MarketSnapshot.model_validate_json(row[0]) if row else None


    def nearest_market_at_or_before(self, symbol: str, ts: datetime, max_delay_seconds: int = 120, source: str | None = None) -> MarketSnapshot | None:
        start = ts - timedelta(seconds=max_delay_seconds)
        clause, sp = _src_clause(source)
        with self.conn() as c:
            row = c.execute(
                f"SELECT payload_json FROM market_snapshots WHERE symbol=? AND {clause} AND ts>=? AND ts<=? ORDER BY ts DESC LIMIT 1",
                (symbol, *sp, _iso(start), _iso(ts)),
            ).fetchone()
        return MarketSnapshot.model_validate_json(row[0]) if row else None

    def markets_near(self, symbol: str, ts: datetime, max_delay_seconds: int = 180) -> dict[str, MarketSnapshot]:
        """Latest snapshot per venue source within tolerance of ts — cross-venue view."""
        start = ts - timedelta(seconds=max_delay_seconds)
        end = ts + timedelta(seconds=max_delay_seconds)
        with self.conn() as c:
            rows = c.execute(
                "SELECT source, payload_json FROM market_snapshots WHERE symbol=? AND ts>=? AND ts<=? ORDER BY ts",
                (symbol, _iso(start), _iso(end)),
            ).fetchall()
        out: dict[str, MarketSnapshot] = {}
        for src, payload in rows:
            out[src] = MarketSnapshot.model_validate_json(payload)
        return out

    def feature_at_or_before(self, symbol: str, ts: datetime, version: str = "v4", max_delay_seconds: int = 120) -> FeatureSnapshot | None:
        start = ts - timedelta(seconds=max_delay_seconds)
        with self.conn() as c:
            row = c.execute(
                "SELECT payload_json FROM feature_snapshots WHERE symbol=? AND feature_version=? AND ts>=? AND ts<=? ORDER BY ts DESC LIMIT 1",
                (symbol, version, _iso(start), _iso(ts)),
            ).fetchone()
        return FeatureSnapshot.model_validate_json(row[0]) if row else None

    def symbols(self) -> list[str]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT symbol FROM market_snapshots UNION SELECT symbol FROM post_assets UNION SELECT symbol FROM feature_snapshots ORDER BY symbol"
            ).fetchall()
        return [r[0] for r in rows]

    def latest_market(self, symbol: str, source: str | None = None) -> MarketSnapshot | None:
        clause, sp = _src_clause(source)
        with self.conn() as c:
            row = c.execute(
                f"SELECT payload_json FROM market_snapshots WHERE symbol=? AND {clause} ORDER BY ts DESC LIMIT 1",
                (symbol, *sp),
            ).fetchone()
        return MarketSnapshot.model_validate_json(row[0]) if row else None

    def features_before(self, symbol: str, ts: datetime, version: str = "v4", limit: int = 2000) -> list[FeatureSnapshot]:
        """Trailing feature rows strictly before ts — the point-in-time history a
        residual model may fit on without seeing the present."""
        with self.conn() as c:
            rows = c.execute(
                "SELECT payload_json FROM feature_snapshots WHERE symbol=? AND feature_version=? AND ts<? ORDER BY ts DESC LIMIT ?",
                (symbol, version, _iso(ts), limit),
            ).fetchall()
        return [FeatureSnapshot.model_validate_json(r[0]) for r in reversed(rows)]

    def all_features(self, version: str = "v4") -> list[FeatureSnapshot]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT payload_json FROM feature_snapshots WHERE feature_version=? ORDER BY ts",
                (version,),
            ).fetchall()
        return [FeatureSnapshot.model_validate_json(r[0]) for r in rows]

    def features_since(self, since: datetime, version: str = "v4") -> list[FeatureSnapshot]:
        """Feature snapshots anchored at/after ``since`` — maintenance's bounded scan."""
        with self.conn() as c:
            rows = c.execute(
                "SELECT payload_json FROM feature_snapshots WHERE feature_version=? AND ts>=? ORDER BY ts",
                (version, _iso(since)),
            ).fetchall()
        return [FeatureSnapshot.model_validate_json(r[0]) for r in rows]

    def label_ts_set(self, label_version: str = "v4") -> set[tuple[str, str]]:
        """{(symbol, ts_iso)} for every stored label — one bulk load so label
        existence checks are set-membership instead of per-anchor queries."""
        with self.conn() as c:
            return {(r[0], r[1]) for r in c.execute(
                "SELECT symbol, ts FROM labels WHERE label_version=?", (label_version,)).fetchall()}

    def latest_semantic_model(self) -> str | None:
        """Most recently written semantics model — the best available guess for
        what the provider will write *before* the engine's alias resolves
        (query_model starts as 'jev-latest'; coverage checks against the alias
        treat every already-classified post as pending → billed reclassification)."""
        with self.conn() as c:
            r = c.execute(
                "SELECT model FROM post_semantics_asset ORDER BY classified_at DESC LIMIT 1"
            ).fetchone()
        return r[0] if r else None

    def max_classified_map(self, model: str | None = None) -> dict[tuple[str, str], str]:
        """{(post_id, symbol): latest classified_at iso} — lets maintenance compute
        actionable anchors in memory instead of one semantics query per post."""
        sql = "SELECT post_id, symbol, MAX(classified_at) FROM post_semantics_asset"
        args: list[object] = []
        if model:
            sql += " WHERE model=?"
            args.append(model)
        sql += " GROUP BY post_id, symbol"
        with self.conn() as c:
            return {(r[0], r[1]): r[2] for r in c.execute(sql, args).fetchall()}

    def posts_with_symbols_since(self, since: datetime) -> list[SocialPost]:
        """(post, asset_symbol) rows created at/after ``since`` — the bounded
        recency window maintenance actually needs to label."""
        with self.conn() as c:
            rows = c.execute(
                """SELECT p.*, a.symbol AS asset_symbol
                   FROM post_assets a JOIN posts p ON p.post_id=a.post_id
                   WHERE p.created_at>=? ORDER BY p.created_at""",
                (_iso(since),)).fetchall()
        return [self._post(r) for r in rows]

    def label_for(self, symbol: str, ts: datetime, version: str = "v4") -> LabelRecord | None:
        with self.conn() as c:
            row = c.execute(
                "SELECT payload_json FROM labels WHERE symbol=? AND ts=? AND label_version=?",
                (symbol, _iso(ts), version),
            ).fetchone()
        return LabelRecord.model_validate_json(row[0]) if row else None

    def post(self, post_id: str) -> SocialPost | None:
        with self.conn() as c:
            r = c.execute("SELECT * FROM posts WHERE post_id=?", (post_id,)).fetchone()
        return self._post(r) if r else None

    def _post(self, r: sqlite3.Row) -> SocialPost:
        return SocialPost(
            post_id=r["post_id"], platform=r["platform"], symbol=(r["asset_symbol"] if "asset_symbol" in r.keys() else r["symbol"]),
            author_id=r["author_id"], author_username=r["author_username"], text=r["text"],
            created_at=datetime.fromisoformat(r["created_at"]), first_seen_at=datetime.fromisoformat(r["first_seen_at"]),
            likes=r["likes"], reposts=r["reposts"], replies=r["replies"], followers=r["followers"],
            verified=bool(r["verified"]) if r["verified"] is not None else None,
            urls=json.loads(r["urls_json"]), raw=json.loads(r["raw_json"]),
        )

    def save_event(self, event) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO events(event_id,symbol,started_at,last_seen_at,representative_text,payload_json) VALUES (?,?,?,?,?,?)",
                (event.event_id, event.symbol, _iso(event.started_at), _iso(event.last_seen_at), event.representative_text, event.model_dump_json()),
            )
            for pid in event.post_ids:
                c.execute("INSERT OR IGNORE INTO event_members(event_id,post_id) VALUES (?,?)", (event.event_id, pid))

    def events_between(self, symbol: str, start: datetime, end: datetime) -> list[dict]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT payload_json FROM events WHERE symbol=? AND last_seen_at>=? AND started_at<? ORDER BY started_at",
                (symbol, _iso(start), _iso(end)),
            ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def event_memberships(self, post_ids: list[str], symbol: str | None = None) -> set[str]:
        """Which of these posts are already claimed by an event.

        Claims are symbol-scoped: one external post may legitimately belong to
        both a BTC and an ETH narrative event, so membership is checked against
        events of the queried symbol only."""
        if not post_ids:
            return set()
        out: set[str] = set()
        with self.conn() as c:
            for i in range(0, len(post_ids), 500):
                chunk = post_ids[i:i + 500]
                q = ",".join("?" for _ in chunk)
                if symbol is None:
                    out.update(r[0] for r in c.execute(f"SELECT post_id FROM event_members WHERE post_id IN ({q})", chunk).fetchall())
                else:
                    out.update(r[0] for r in c.execute(
                        f"SELECT m.post_id FROM event_members m JOIN events e ON e.event_id=m.event_id WHERE e.symbol=? AND m.post_id IN ({q})",
                        [symbol, *chunk],
                    ).fetchall())
        return out

    def all_events(self, symbol: str | None = None) -> list[dict]:
        sql = "SELECT payload_json FROM events"
        args: list[object] = []
        if symbol is not None:
            sql += " WHERE symbol=?"
            args.append(symbol)
        sql += " ORDER BY started_at"
        with self.conn() as c:
            rows = c.execute(sql, args).fetchall()
        return [json.loads(r[0]) for r in rows]

    def record_classification_failure(self, post_id: str, symbol: str, model: str, error: str, *, counted: bool = True) -> None:
        """Track a failed classify attempt. ``counted=False`` for provider-side
        failures (402 credits, 429, timeouts, network): those say nothing about
        the post, so they must not push it toward the permanent-failure cap —
        otherwise a provider outage poisons every attempted post."""
        with self.conn() as c:
            c.execute(
                """INSERT INTO classification_failures(post_id,symbol,model,attempts,last_error,last_ts)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(post_id,symbol,model) DO UPDATE SET
                     attempts = attempts + ?,
                     last_error = excluded.last_error,
                     last_ts = excluded.last_ts""",
                (post_id, symbol, model, 1 if counted else 0, error[:300],
                 datetime.now(timezone.utc).isoformat(), 1 if counted else 0),
            )

    def unsemanticized_posts(self, symbol: str | None, model: str, limit: int = 1000, *, max_failures: int = 3) -> list[SocialPost]:
        """(post_id, asset_symbol) pairs missing a semantics row for ``model``.
        ``symbol=None`` scans every asset mapping — used by maintenance backfill.
        Pairs with >= ``max_failures`` *counted* failures are excluded: retrying
        deterministic content failures forever burns billed calls each tick."""
        sql = """SELECT p.*, a.symbol AS asset_symbol
                 FROM post_assets a JOIN posts p ON p.post_id=a.post_id
                 LEFT JOIN post_semantics_asset s
                   ON p.post_id=s.post_id AND a.symbol=s.symbol AND s.model=?
                 WHERE s.post_id IS NULL
                   AND NOT EXISTS (SELECT 1 FROM classification_failures f
                     WHERE f.post_id=p.post_id AND f.symbol=a.symbol AND f.model=?
                     AND f.attempts>=?)"""
        args: list[object] = [model, model, max_failures]
        if symbol is not None:
            sql += " AND a.symbol=?"
            args.append(symbol)
        else:
            sql += " AND a.symbol!='MACRO'"
        sql += " ORDER BY p.created_at LIMIT ?"
        args.append(limit)
        with self.conn() as c:
            rows = c.execute(sql, args).fetchall()
        return [self._post(r) for r in rows]

    def post_asset_symbols(self, post_id: str) -> list[str]:
        """All assets a post maps to. A pair-trade post has one row in posts but
        N rows in post_assets; per-asset labels/semantics must iterate this."""
        with self.conn() as c:
            return [r[0] for r in c.execute("SELECT symbol FROM post_assets WHERE post_id=?", (post_id,)).fetchall()]

    def all_posts(self, symbol: str | None = None) -> list[SocialPost]:
        sql = """SELECT p.*, a.symbol AS asset_symbol
                 FROM post_assets a JOIN posts p ON p.post_id=a.post_id"""
        args: list[object] = []
        if symbol is not None:
            sql += " WHERE a.symbol=?"
            args.append(symbol)
        sql += " ORDER BY p.created_at, a.symbol"
        with self.conn() as c:
            rows = c.execute(sql, args).fetchall()
        return [self._post(r) for r in rows]

    def posts_for_author(self, author_username: str, symbol: str, before: datetime | None = None) -> list[SocialPost]:
        sql = """SELECT p.*, a.symbol AS asset_symbol
                 FROM post_assets a JOIN posts p ON p.post_id=a.post_id
                 WHERE lower(p.author_username)=lower(?) AND a.symbol=?"""
        args: list[object] = [author_username, symbol]
        if before is not None:
            sql += " AND p.created_at<?"
            args.append(_iso(before))
        sql += " ORDER BY p.created_at"
        with self.conn() as c:
            rows = c.execute(sql, args).fetchall()
        return [self._post(r) for r in rows]

    def semantics_for_post(self, post_id: str, model: str | None = None, symbol: str | None = None) -> list[PostSemantics]:
        sql = "SELECT payload_json FROM post_semantics_asset WHERE post_id=?"
        args: list[object] = [post_id]
        if model:
            sql += " AND model=?"
            args.append(model)
        if symbol:
            sql += " AND symbol=?"
            args.append(symbol)
        sql += " ORDER BY symbol, classified_at"
        with self.conn() as c:
            rows = c.execute(sql, args).fetchall()
        return [PostSemantics.model_validate_json(r[0]) for r in rows]


    def latest_feature(self, symbol: str, version: str = "v4") -> FeatureSnapshot | None:
        with self.conn() as c:
            row = c.execute(
                "SELECT payload_json FROM feature_snapshots WHERE symbol=? AND feature_version=? ORDER BY ts DESC LIMIT 1",
                (symbol, version),
            ).fetchone()
        return FeatureSnapshot.model_validate_json(row[0]) if row else None

    def latest_features(self, version: str = "v4") -> list[FeatureSnapshot]:
        with self.conn() as c:
            rows = c.execute(
                """SELECT f.payload_json FROM feature_snapshots f
                   JOIN (SELECT symbol, MAX(ts) ts FROM feature_snapshots WHERE feature_version=? GROUP BY symbol) x
                   ON f.symbol=x.symbol AND f.ts=x.ts
                   WHERE f.feature_version=? ORDER BY f.symbol""",
                (version, version),
            ).fetchall()
        return [FeatureSnapshot.model_validate_json(r[0]) for r in rows]

    def save_paper_trade(self, trade: PaperTrade) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO paper_trades(trade_id,symbol,opened_at,status,payload_json) VALUES (?,?,?,?,?)",
                (trade.trade_id, trade.symbol, _iso(trade.opened_at), trade.status, trade.model_dump_json()),
            )

    def paper_trades(self, status: str | None = None) -> list[PaperTrade]:
        sql = "SELECT payload_json FROM paper_trades"
        args: list[object] = []
        if status is not None:
            sql += " WHERE status=?"
            args.append(status)
        sql += " ORDER BY opened_at"
        with self.conn() as c:
            rows = c.execute(sql, args).fetchall()
        return [PaperTrade.model_validate_json(r[0]) for r in rows]

    def open_paper_trade_for_symbol(self, symbol: str) -> PaperTrade | None:
        with self.conn() as c:
            row = c.execute(
                "SELECT payload_json FROM paper_trades WHERE symbol=? AND status='OPEN' ORDER BY opened_at DESC LIMIT 1",
                (symbol,),
            ).fetchone()
        return PaperTrade.model_validate_json(row[0]) if row else None

    def paper_feature_already_used(self, symbol: str, feature_ts: datetime, model_fingerprint: str | None) -> bool:
        # SQLite JSON portability is intentionally avoided here; the number of paper trades is small.
        for t in reversed(self.paper_trades()):
            if t.symbol != symbol or t.model_fingerprint != model_fingerprint or t.entry_feature_ts is None:
                continue
            if _iso(t.entry_feature_ts) == _iso(feature_ts):
                return True
        return False

    def holdout_opening(self, fingerprint: str) -> dict | None:
        with self.conn() as c:
            row=c.execute("SELECT payload_json FROM holdout_registry WHERE holdout_fingerprint=?",(fingerprint,)).fetchone()
        return json.loads(row[0]) if row else None

    def save_holdout_opening(self, fingerprint: str, horizon: str, protocol_hash: str, payload: dict, reopened: bool = False) -> None:
        body={**payload,"holdout_fingerprint":fingerprint,"horizon":horizon,"protocol_hash":protocol_hash,"reopened":bool(reopened)}
        with self.conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO holdout_registry(holdout_fingerprint,horizon,opened_at,protocol_hash,reopened,payload_json) VALUES (?,?,?,?,?,?)",
                (fingerprint,horizon,_iso(datetime.now(timezone.utc)),protocol_hash,int(bool(reopened)),json.dumps(body,default=str)),
            )

    def storage_health(self, now: datetime | None = None) -> dict:
        now=now or datetime.now(timezone.utc)
        tables=["posts","post_assets","post_semantics_asset","engagement_snapshots","market_snapshots","feature_snapshots","labels","events","paper_trades","api_usage"]
        counts={}
        with self.conn() as c:
            for t in tables:
                counts[t]=int(c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
            day_ago=_iso(now-timedelta(days=1))
            market_24h=int(c.execute("SELECT COUNT(*) FROM market_snapshots WHERE ts>=?",(day_ago,)).fetchone()[0])
            first=c.execute("SELECT MIN(ts),MAX(ts) FROM market_snapshots").fetchone()
            page_count=int(c.execute("PRAGMA page_count").fetchone()[0]); page_size=int(c.execute("PRAGMA page_size").fetchone()[0])
        db_bytes=page_count*page_size
        total_rows=max(1,sum(counts.values()))
        bytes_per_row=float(db_bytes/total_rows)
        projected_rows_30d=int(market_24h*30)
        return {
            "db_path":self.path,"db_bytes":db_bytes,"db_mb":db_bytes/1024/1024,"counts":counts,
            "market_first_ts":first[0] if first else None,"market_last_ts":first[1] if first else None,
            "market_rows_last_24h":market_24h,"projected_market_rows_30d_at_current_rate":projected_rows_30d,
            "rough_bytes_per_db_row":bytes_per_row,"rough_projected_market_storage_30d_mb":projected_rows_30d*bytes_per_row/1024/1024,
        }

    def compact_market_range(self, start: datetime, end: datetime, bucket_seconds: int = 60, execute: bool = False) -> dict:
        if bucket_seconds < 1:
            raise ValueError("bucket_seconds must be >=1")
        with self.conn() as c:
            rows=c.execute("SELECT id,source,symbol,ts FROM market_snapshots WHERE ts>=? AND ts<? ORDER BY source,symbol,ts,id",(_iso(start),_iso(end))).fetchall()
        keep: dict[tuple[str,str,int], tuple[int,datetime]]={}
        ids=[]
        for r in rows:
            ts=datetime.fromisoformat(r["ts"]); key=(str(r["source"]),str(r["symbol"]),int(ts.timestamp())//bucket_seconds)
            ids.append(int(r["id"])); prev=keep.get(key)
            if prev is None or ts>=prev[1]: keep[key]=(int(r["id"]),ts)
        keep_ids={v[0] for v in keep.values()}; delete_ids=[i for i in ids if i not in keep_ids]
        if execute and delete_ids:
            with self.conn() as c:
                for i in range(0,len(delete_ids),500):
                    chunk=delete_ids[i:i+500]; q=','.join('?' for _ in chunk)
                    c.execute(f"DELETE FROM market_snapshots WHERE id IN ({q})",chunk)
        return {
            "start":_iso(start),"end":_iso(end),"bucket_seconds":bucket_seconds,"scanned":len(rows),
            "kept":len(keep_ids),"deleted":len(delete_ids) if execute else 0,"would_delete":len(delete_ids),"executed":bool(execute),
        }

    def save_author_reputation(self, record: AuthorReputationRecord) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO author_reputation(author,symbol,horizon,as_of,payload_json) VALUES (?,?,?,?,?)",
                (record.author.lower(), record.symbol, record.horizon, _iso(record.as_of), record.model_dump_json()),
            )

    def author_reputation_for(self, author: str, symbol: str, ts: datetime, horizon: str = "15m") -> AuthorReputationRecord | None:
        with self.conn() as c:
            row = c.execute(
                "SELECT payload_json FROM author_reputation WHERE author=? AND symbol=? AND horizon=? AND as_of<=? ORDER BY as_of DESC LIMIT 1",
                (author.lower(), symbol, horizon, _iso(ts)),
            ).fetchone()
        return AuthorReputationRecord.model_validate_json(row[0]) if row else None
