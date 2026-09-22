from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import typer

from . import __version__

from .config import settings
from .demo import seed_demo
from .features import FeatureEngine
from .labels import build_labels, post_actionable_ts, HORIZONS
from .providers.bybit import BybitPublicClient
from .providers.xapi import XAPIClient
from .research import FEATURE_GROUPS, ablation_ladder, dataset, incremental_semantic_test, model_ladder
from .semantics.heuristic import HeuristicSemanticEngine
from .semantics.jev import JevSemanticEngine
from .storage import Store
from .pipeline import collect_cycle, base_symbol
from .historical import ingest_historical_social
from .historical_market import ingest_bybit_history
from .event_study import event_study, event_study_grid
from .ranking import cross_sectional_backtest, cross_sectional_ranks
from .audit import leakage_audit
from .providers.x_stream import XFilterRules, XRealtimeStream, posts_from_event
from .providers.xapi import parse_dt as _parse_x_ts
from .live import run_live_stack
from .quality import data_quality, feed_health_report, latency_report
from .models import SemanticResidualModel
from .paper import paper_step, paper_summary, paper_attribution
from .validation import heldout_regime_report, leave_one_symbol_out, placebo_test, purged_walk_forward_ablation, semantic_permutation_placebo, semantic_staleness_test
from .evidence import evidence_gate
from .drift import drift_report
from .author_alpha import snapshot_author_reputations
from .cross_asset import enrich_cross_asset_at
from .historical_features import rebuild_historical_features
from .maintenance import maintenance_step, refresh_engagement, backfill_semantics
from .regime_pulse import regime_pulse
from .horizons import horizon_scan
from .explain import feature_importance_report
from .registry import champion_status, promote_candidate, research_manifest, source_semantic_models
from .protocol import run_locked_protocol
from .engine_compare import compare_feature_versions
from .campaign import run_campaign
from .factor_analysis import factor_ic_report, factor_monotonicity, semantic_reliability
from .economics import operating_economics, usage_summary, capacity_report
from .adversarial import run_benchmark as adversarial_benchmark
from .overfit import overfit_report

app = typer.Typer(no_args_is_help=True, help="Semantic Alpha Lab research CLI")


def store(db: str | None) -> Store:
    return Store(db or settings.db_path)


@app.command("init-db")
def init_db(db: str | None = typer.Option(None)):
    s = store(db)
    typer.echo(f"Initialized {s.path}")


@app.command("doctor")
def doctor():
    try:
        import websockets  # noqa: F401
        has_stream=True
    except ImportError:
        has_stream=False
    try:
        import typesafe_sdk  # noqa: F401
        has_jev_sdk=True
    except ImportError:
        has_jev_sdk=False
    out = {
        "version": __version__,
        "db": settings.db_path,
        "typesafe_key": bool(settings.typesafe_api_key),
        "x_api_key": bool(settings.x_api_key),
        "jev_sdk_installed": has_jev_sdk,
        "stream_extra_installed": has_stream,
        "bybit_base_url": settings.bybit_base_url,
        "bybit_ws_url": settings.bybit_ws_url,
        "x_ws_url": settings.x_ws_url,
        "jev_model": settings.jev_model,
    }
    typer.echo(json.dumps(out, indent=2))


@app.command("market-snapshot")
def market_snapshot(symbol: str = "SOLUSDT", db: str | None = typer.Option(None)):
    async def run():
        s = store(db)
        m = await BybitPublicClient(settings.bybit_base_url).snapshot(symbol)
        s.save_market(m)
        typer.echo(m.model_dump_json(indent=2))
    asyncio.run(run())


@app.command("x-window")
def x_window(symbol: str, minutes: int = 15, db: str | None = typer.Option(None)):
    async def run():
        s = store(db)
        end = datetime.now(timezone.utc); start = end - timedelta(minutes=minutes)
        c = XAPIClient(settings.x_api_key, settings.x_api_url)
        posts = await c.search_window(symbol, start, end, known_ids=s.post_ids_between(start, end))
        n = sum(s.save_post(p) for p in posts)
        typer.echo(f"Fetched {len(posts)}, inserted {n}")
    asyncio.run(run())


@app.command("semanticize")
def semanticize(symbol: str, minutes: int = 60, engine: str = "jev", db: str | None = typer.Option(None)):
    async def run():
        s = store(db)
        end = datetime.now(timezone.utc); start = end - timedelta(minutes=minutes)
        posts = s.posts_between(symbol, start, end)
        sem = JevSemanticEngine(settings.typesafe_api_key, settings.jev_model) if engine == "jev" else HeuristicSemanticEngine()
        for p in posts:
            s.save_semantics(await sem.classify(p))
        typer.echo(f"Semanticized {len(posts)} posts with {getattr(sem,'model',engine)}")
    asyncio.run(run())


@app.command("features")
def features(symbol: str, db: str | None = typer.Option(None), semantic_model: str | None = typer.Option(None), feature_version: str = "v4"):
    s = store(db)
    m = s.latest_market(symbol)
    if not m:
        raise typer.BadParameter("No market snapshot for symbol")
    f = FeatureEngine(s, semantic_model, feature_version=feature_version).build(symbol, m.ts, m)
    s.save_features(f)
    typer.echo(f.model_dump_json(indent=2))


@app.command("label")
def label(symbol: str, ts: str, db: str | None = typer.Option(None)):
    s = store(db)
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    l = build_labels(s, symbol, dt)
    if not l:
        raise typer.BadParameter("Insufficient market snapshots")
    s.save_label(l); typer.echo(l.model_dump_json(indent=2))


@app.command("ablation")
def ablation(db: str | None = typer.Option(None), horizon: str = "15m", cost_bps: float = 10.0):
    s = store(db); d = dataset(s, horizon=horizon)
    if d.empty:
        typer.echo("No labeled feature rows")
        raise typer.Exit(2)
    typer.echo(ablation_ladder(d, cost_bps=cost_bps, horizon=horizon).to_string(index=False))


@app.command("demo")
def demo(db: str = typer.Option("./data/demo.db"), minutes: int = 360):
    async def run():
        p = Path(db)
        if p.exists(): p.unlink()
        s = Store(db)
        await seed_demo(s, minutes=minutes)
        d = dataset(s, horizon="15m")
        typer.echo(f"Rows: {len(d)}")
        typer.echo(ablation_ladder(d, cost_bps=8).to_string(index=False))
    asyncio.run(run())


def _venue_clients(venues: str):
    """Extra venue clients for cross-venue features (canonical truth stays bybit)."""
    out = []
    for v in [x.strip().lower() for x in venues.split(",") if x.strip()]:
        if v == "bybit":
            continue
        if v == "binance":
            from .providers.binance import BinancePublicClient
            out.append(BinancePublicClient(settings.binance_base_url))
        else:
            raise typer.BadParameter(f"Unknown venue '{v}' (supported: binance)")
    return out


@app.command("collect-once")
def collect_once(symbol: str = "SOLUSDT", engine: str = "jev", lookback_minutes: int = 5, lag_overlap_seconds: int = 60, venues: str = "bybit", feature_version: str = "v4", db: str | None = typer.Option(None)):
    """One forward-data collection cycle: market + X + semantics + events + features."""
    async def run():
        s=store(db)
        sem=JevSemanticEngine(settings.typesafe_api_key, settings.jev_model) if engine=="jev" else HeuristicSemanticEngine()
        if not settings.x_api_key:
            raise typer.BadParameter("X_API_KEY required for live collection")
        result=await collect_cycle(s, symbol, BybitPublicClient(settings.bybit_base_url), XAPIClient(settings.x_api_key, settings.x_api_url), sem, lookback_minutes, lag_overlap_seconds, _venue_clients(venues), feature_version)
        typer.echo(json.dumps({**result.__dict__, "feature":result.feature.model_dump(mode="json")}, indent=2, default=str))
    asyncio.run(run())

@app.command("historical-social")
def historical_social(symbol: str, start: str, end: str, window_minutes: int = 15, db: str | None = typer.Option(None)):
    """Ingest historical X data using bounded time windows to reduce cursor failure risk."""
    async def run():
        s=store(db)
        if not settings.x_api_key:
            raise typer.BadParameter("X_API_KEY required")
        st=datetime.fromisoformat(start.replace("Z","+00:00")); en=datetime.fromisoformat(end.replace("Z","+00:00"))
        r=await ingest_historical_social(s, XAPIClient(settings.x_api_key, settings.x_api_url), symbol, st, en, window_minutes)
        typer.echo(json.dumps(r, indent=2))
    asyncio.run(run())


@app.command("historical-market")
def historical_market(symbol: str, start: str, end: str, interval_minutes: int = 1, db: str | None = typer.Option(None)):
    """Backfill Bybit historical price + carry-forward OI/funding for research labels."""
    async def run():
        s=store(db); st=datetime.fromisoformat(start.replace("Z","+00:00")); en=datetime.fromisoformat(end.replace("Z","+00:00"))
        r=await ingest_bybit_history(s,BybitPublicClient(settings.bybit_base_url),symbol,st,en,interval_minutes)
        typer.echo(json.dumps(r,indent=2))
    asyncio.run(run())

@app.command("event-study")
def event_study_cmd(db: str | None = typer.Option(None), horizon: str = "15m", semantic_model: str | None = typer.Option(None), min_n: int = 5, unit: str = "event"):
    """Event-deduplicated semantic study with FDR-corrected q-values."""
    if unit not in {"event", "post"}:
        raise typer.BadParameter("unit must be event or post")
    r=event_study(store(db),horizon=horizon,semantic_model=semantic_model,min_n=min_n,inferential_unit=unit)
    if r.empty:
        typer.echo("No labeled semantic events"); raise typer.Exit(2)
    typer.echo(r.to_string(index=False))


@app.command("event-study-grid")
def event_study_grid_cmd(
    db: str | None = typer.Option(None),
    horizons: str = "1m,5m,15m,30m,1h,4h,24h",
    semantic_model: str | None = typer.Option(None),
    min_n: int = 5,
    output: str | None = typer.Option(None),
):
    """Run narrative-event studies across horizons with global FDR control."""
    hs=[x.strip() for x in horizons.split(',') if x.strip()]
    r=event_study_grid(store(db),horizons=hs,semantic_model=semantic_model,min_n=min_n)
    if r.empty:
        typer.echo("No labeled narrative events"); raise typer.Exit(2)
    if output:
        Path(output).parent.mkdir(parents=True,exist_ok=True); r.to_csv(output,index=False)
        typer.echo(json.dumps({"output":output,"tests":len(r),"global_fdr_10pct":int(r.fdr_global_10pct.sum())},indent=2))
    else:
        typer.echo(r.to_string(index=False))

@app.command("walk-forward")
def walk_forward(db: str | None = typer.Option(None), horizon: str = "15m", cost_bps: float = 10.0, model_name: str = "logistic", notional_usd: float = 1000.0):
    """Purged expanding walk-forward; training labels cannot overlap the test window."""
    s=store(db); d=dataset(s,horizon=horizon); r=purged_walk_forward_ablation(d,horizon=horizon,cost_bps=cost_bps,model_name=model_name,notional_usd=notional_usd)
    if r.empty:
        typer.echo("Not enough labeled data for purged walk-forward evaluation"); raise typer.Exit(2)
    typer.echo(r.to_string(index=False))


@app.command("collect-forward")
def collect_forward(
    symbols: str = "BTCUSDT,ETHUSDT,SOLUSDT", engine: str = "jev", interval_seconds: int = 60,
    iterations: int = 0, lookback_minutes: int = 5, lag_overlap_seconds: int = 60, venues: str = "bybit", feature_version: str = "v4", db: str | None = typer.Option(None),
):
    """Continuously build a point-in-time forward dataset. iterations=0 means run until Ctrl-C.

    Each symbol keeps a persistent watermark; steady-state windows overlap only by
    lag-overlap-seconds (for late-indexed posts) instead of the full lookback, so
    the provider does not repeatedly re-bill the same posts. Set 0 for strict
    contiguous windows at the cost of dropping late-indexed posts."""
    async def run():
        s=store(db)
        if not settings.x_api_key:
            raise typer.BadParameter("X_API_KEY required")
        sem=JevSemanticEngine(settings.typesafe_api_key, settings.jev_model) if engine=="jev" else HeuristicSemanticEngine()
        market=BybitPublicClient(settings.bybit_base_url)
        social=XAPIClient(settings.x_api_key, settings.x_api_url)
        extra_venues=_venue_clients(venues)
        syms=[x.strip().upper() for x in symbols.split(",") if x.strip()]
        n=0
        while iterations==0 or n<iterations:
            started=datetime.now(timezone.utc)
            for sym in syms:
                try:
                    r=await collect_cycle(s,sym,market,social,sem,lookback_minutes,lag_overlap_seconds,extra_venues,feature_version)
                    typer.echo(f"{datetime.now(timezone.utc).isoformat()} {sym} posts={r.posts_fetched}/{r.posts_inserted} semanticized={r.posts_semanticized} events={r.events}")
                except Exception as e:
                    typer.echo(f"{datetime.now(timezone.utc).isoformat()} {sym} ERROR {e}", err=True)
            enrich_cross_asset_at(s,datetime.now(timezone.utc),symbols=syms,feature_version=feature_version,tolerance_seconds=max(120,interval_seconds*2))
            n+=1
            elapsed=(datetime.now(timezone.utc)-started).total_seconds()
            if iterations==0 or n<iterations:
                await asyncio.sleep(max(0, interval_seconds-elapsed))
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        typer.echo("Stopped forward collector")

@app.command("backfill-labels")
def backfill_labels(db: str | None = typer.Option(None), benchmark_symbol: str = "BTCUSDT", include_posts: bool = True, max_delay_seconds: int = 120):
    """Create labels for feature rows and optionally post timestamps once future market data exists."""
    s=store(db); created=0; skipped=0; post_created=0
    for f in s.all_features():
        if s.label_for(f.symbol,f.ts): continue
        l=build_labels(s,f.symbol,f.ts,benchmark_symbol=benchmark_symbol,max_delay_seconds=max_delay_seconds)
        if l and any(v is not None for v in l.returns.values()): s.save_label(l); created+=1
        else: skipped+=1
    if include_posts:
        for p in s.all_posts():
            for sym in s.post_asset_symbols(p.post_id) or [p.symbol]:
                anchor=post_actionable_ts(s,p,sym)
                if s.label_for(sym,anchor): continue
                l=build_labels(s,sym,anchor,benchmark_symbol=benchmark_symbol,max_delay_seconds=max_delay_seconds)
                if l and any(v is not None for v in l.returns.values()): s.save_label(l); post_created+=1
    typer.echo(json.dumps({"feature_labels_created":created,"post_labels_created":post_created,"skipped":skipped},indent=2))

@app.command("export-dataset")
def export_dataset(output: str = "./data/research.csv", db: str | None = typer.Option(None), horizon: str = "15m"):
    s=store(db); d=dataset(s,horizon=horizon)
    Path(output).parent.mkdir(parents=True,exist_ok=True); d.to_csv(output,index=False)
    typer.echo(f"Exported {len(d)} rows to {output}")

@app.command("research-report")
def research_report(output_dir: str = "./runs/research", db: str | None = typer.Option(None), horizon: str = "15m", cost_bps: float = 10.0, notional_usd: float = 1000.0):
    s=store(db); d=dataset(s,horizon=horizon); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    if d.empty:
        typer.echo("No labeled rows"); raise typer.Exit(2)
    abl=ablation_ladder(d,cost_bps=cost_bps,dynamic_cost=True,notional_usd=notional_usd,horizon=horizon)
    wf=purged_walk_forward_ablation(d,horizon=horizon,cost_bps=cost_bps,dynamic_cost=True,notional_usd=notional_usd,model_name="logistic")
    models=model_ladder(d,cost_bps=cost_bps,dynamic_cost=True,notional_usd=notional_usd,horizon=horizon)
    increment=incremental_semantic_test(d,cost_bps=cost_bps,notional_usd=notional_usd,horizon=horizon)
    xs_trades,xs_metrics=cross_sectional_backtest(d,notional_usd=notional_usd,horizon=horizon)
    audit=leakage_audit(s)
    quality=data_quality(s)
    drift=drift_report(d)
    d.to_csv(out/'dataset.csv',index=False); abl.to_csv(out/'ablation.csv',index=False); wf.to_csv(out/'purged_walk_forward.csv',index=False); models.to_csv(out/'model_ladder.csv',index=False); xs_trades.to_csv(out/'cross_sectional_trades.csv',index=False)
    summary={
      "rows":len(d), "symbols":sorted(d.symbol.unique().tolist()), "horizon":horizon, "cost_bps_fallback":cost_bps, "notional_usd":notional_usd,
      "leakage_audit":audit, "data_quality":quality, "drift":drift, "incremental_semantic_test":increment, "cross_sectional_metrics":xs_metrics, "ablation":abl.to_dict(orient="records"), "model_ladder":models.to_dict(orient="records"), "purged_walk_forward_folds":wf.to_dict(orient="records"),
    }
    (out/'report.json').write_text(json.dumps(summary,indent=2,default=str))
    typer.echo(f"Wrote {out/'report.json'}")

@app.command("rank")
def rank(db: str | None = typer.Option(None), top: int = 20):
    d=cross_sectional_ranks(store(db))
    if d.empty:
        typer.echo("No feature snapshots"); raise typer.Exit(2)
    cols=[c for c in ["symbol","underreaction_rank_score","fade_rank_score","squeeze_rank_score","information_price_gap_proxy","narrative_novelty_15m","attention_z_5m"] if c in d.columns]
    typer.echo(d[cols].head(top).to_string(index=False))

@app.command("semantic-increment-test")
def semantic_increment_test_cmd(db: str | None = typer.Option(None), horizon: str = "15m", notional_usd: float = 1000.0, bootstrap_samples: int = 500):
    d=dataset(store(db),horizon=horizon)
    typer.echo(json.dumps(incremental_semantic_test(d,notional_usd=notional_usd,bootstrap_samples=bootstrap_samples,horizon=horizon),indent=2,default=str))

@app.command("cross-sectional-backtest")
def cross_sectional_backtest_cmd(db: str | None = typer.Option(None), horizon: str = "15m", score: str = "information_price_gap_proxy", top_k: int = 1, notional_usd: float = 1000.0):
    d=dataset(store(db),horizon=horizon); trades,metrics=cross_sectional_backtest(d,score,top_k,notional_usd,horizon=horizon)
    typer.echo(json.dumps({"metrics":metrics,"trades":len(trades)},indent=2,default=str))

@app.command("data-quality")
def data_quality_cmd(db: str | None = typer.Option(None)):
    r=data_quality(store(db)); typer.echo(json.dumps(r,indent=2,default=str))
    if not r['ok']: raise typer.Exit(3)

@app.command("leakage-audit")
def leakage_audit_cmd(db: str | None = typer.Option(None)):
    r=leakage_audit(store(db)); typer.echo(json.dumps(r,indent=2,default=str))
    if not r["ok"]: raise typer.Exit(3)

@app.command("model-ladder")
def model_ladder_cmd(db: str | None = typer.Option(None), horizon: str = "15m", notional_usd: float = 1000.0):
    d=dataset(store(db),horizon=horizon)
    if d.empty:
        typer.echo("No labeled feature rows"); raise typer.Exit(2)
    typer.echo(model_ladder(d,dynamic_cost=True,notional_usd=notional_usd,horizon=horizon).to_string(index=False))


@app.command("x-add-rules")
def x_add_rules(symbols: str = "BTCUSDT,ETHUSDT,SOLUSDT", interval_seconds: float = 5.0):
    """Create X filter rules (via twitterapi.io) tagged by asset for real-time streaming."""
    async def run():
        if not settings.x_api_key: raise typer.BadParameter("X_API_KEY required")
        c=XFilterRules(settings.x_api_key,settings.x_base_url)
        out=[]
        for market_symbol in [x.strip().upper() for x in symbols.split(',') if x.strip()]:
            b=base_symbol(market_symbol)
            names={"BTC":"Bitcoin","ETH":"Ethereum","SOL":"Solana","SUI":"Sui","AVAX":"Avalanche","LINK":"Chainlink","XRP":"Ripple","DOGE":"Dogecoin"}
            full=names.get(b)
            lang=f" lang:{settings.x_query_langs}" if settings.x_query_langs else ""
            value=f"(${b} OR {full}){lang} -filter:retweets" if full else f"${b}{lang} -filter:retweets"
            out.append(await c.add_rule(f"semantic-alpha-{b}",value,interval_seconds))
        typer.echo(json.dumps(out,indent=2,default=str))
    asyncio.run(run())

@app.command("x-rules")
def x_rules():
    async def run():
        if not settings.x_api_key: raise typer.BadParameter("X_API_KEY required")
        typer.echo(json.dumps(await XFilterRules(settings.x_api_key,settings.x_base_url).list_rules(),indent=2,default=str))
    asyncio.run(run())

@app.command("live")
def live(symbols: str = "BTCUSDT,ETHUSDT,SOLUSDT", engine: str = "jev", market_snapshot_seconds: int = 1, feature_seconds: int = 15, semantic_workers: int = 4, duration_seconds: int = 0, feature_version: str = "v4", social_stall_seconds: float = 90.0, delivery_recycle_seconds: float = 300.0, classify_timeout: float = 60.0, binance_poll_seconds: float = 20.0, db: str | None = typer.Option(None)):
    """Run synchronized Bybit + X real-time collection. Requires filter rules already configured."""
    async def run():
        if not settings.x_api_key: raise typer.BadParameter("X_API_KEY required")
        sem=JevSemanticEngine(settings.typesafe_api_key,settings.jev_model) if engine=='jev' else HeuristicSemanticEngine()
        syms=[x.strip().upper() for x in symbols.split(',') if x.strip()]
        r=await run_live_stack(store(db),syms,sem,settings.x_api_key,settings.x_ws_url,settings.bybit_ws_url,market_snapshot_seconds,feature_seconds,semantic_workers,duration_seconds,feature_version,social_stall_seconds,delivery_recycle_seconds,classify_timeout,binance_poll_seconds)
        typer.echo(json.dumps(r,indent=2))
    try: asyncio.run(run())
    except KeyboardInterrupt: typer.echo("Stopped live stack")

@app.command("stream-market")
def stream_market(symbols: str = "BTCUSDT,ETHUSDT,SOLUSDT", snapshot_seconds: int = 1, duration_seconds: int = 0, db: str | None = typer.Option(None)):
    """Capture Bybit ticker/L2/trade/liquidation WebSockets into fixed-interval snapshots."""
    async def run():
        from .providers.bybit_ws import BybitLinearStream, MicrostructureAccumulator
        syms=[x.strip().upper() for x in symbols.split(",") if x.strip()]
        s=store(db); acc={x:MicrostructureAccumulator(x) for x in syms}; stream=BybitLinearStream(syms)
        started=datetime.now(timezone.utc); next_emit=started
        async for msg in stream.messages():
            topic=str(msg.get("topic") or "")
            sym=(msg.get("data") or {}).get("symbol") if isinstance(msg.get("data"),dict) else None
            if not sym:
                parts=topic.split("."); sym=parts[-1] if parts else None
            if sym in acc: acc[sym].apply(msg)
            now=datetime.now(timezone.utc)
            if now>=next_emit:
                for a in acc.values():
                    snap=a.snapshot(now);
                    if snap.last>0: s.save_market(snap)
                typer.echo(f"{now.isoformat()} snapshots={sum(1 for a in acc.values() if a.snapshot(now).last>0)}")
                next_emit=now+timedelta(seconds=max(1,snapshot_seconds))
            if duration_seconds>0 and (now-started).total_seconds()>=duration_seconds: break
    try: asyncio.run(run())
    except KeyboardInterrupt: typer.echo("Stopped market stream")


@app.command("author-reputation-snapshot")
def author_reputation_snapshot(
    db: str | None = typer.Option(None),
    as_of: str | None = typer.Option(None),
    horizon: str = "15m",
    semantic_model: str | None = typer.Option(None),
    min_posts: int = 3,
):
    """Materialize point-in-time author skill estimates for leak-safe live features."""
    ts = datetime.fromisoformat(as_of.replace("Z", "+00:00")) if as_of else datetime.now(timezone.utc)
    rows = snapshot_author_reputations(store(db), ts, horizon=horizon, semantic_model=semantic_model, min_posts=min_posts)
    typer.echo(json.dumps({"as_of": ts.isoformat(), "horizon": horizon, "records": len(rows)}, indent=2))


@app.command("train-model")
def train_model(
    output: str = "./models/semantic_residual.joblib",
    db: str | None = typer.Option(None),
    horizon: str = "15m",
    model_name: str = "histgb",
):
    """Train paired market-only/full models used for the learned semantic residual."""
    d = dataset(store(db), horizon=horizon)
    if len(d) < 100:
        typer.echo(f"Need >=100 labeled rows, found {len(d)}"); raise typer.Exit(2)
    baseline = [x for g in ["price_derivatives", "attention"] for x in FEATURE_GROUPS[g] if x in d.columns]
    full = [x for g in ["price_derivatives", "attention", "semantics", "gap"] for x in FEATURE_GROUPS[g] if x in d.columns]
    m = SemanticResidualModel.fit(d, baseline, full, model_name=model_name, horizon=horizon)
    m.save(output)
    typer.echo(json.dumps({"output": output, "model": model_name, "horizon": horizon, "rows": m.trained_rows, "trained_through": m.trained_through, "baseline_features": len(baseline), "full_features": len(full), "probability_threshold":m.probability_threshold, "semantic_edge_threshold":m.semantic_edge_threshold, "calibration":m.calibration, "training_fingerprint":m.training_fingerprint}, indent=2))


@app.command("score-latest")
def score_latest(
    model_path: str = "./models/semantic_residual.joblib",
    db: str | None = typer.Option(None),
    top: int = 20,
):
    """Score latest cross-section and show what semantics adds beyond market+attention."""
    m = SemanticResidualModel.load(model_path)
    rows=[]
    for f in store(db).latest_features():
        try: sc=m.score_row(f.values)
        except ValueError: continue
        rows.append({"symbol": f.symbol, "ts": str(f.ts), **sc})
    rows.sort(key=lambda x: abs(x["semantic_edge_logodds"]), reverse=True)
    typer.echo(json.dumps(rows[:top], indent=2))


@app.command("robustness-report")
def robustness_report(
    output_dir: str = "./runs/robustness",
    db: str | None = typer.Option(None),
    horizon: str = "15m",
    model_name: str = "histgb",
    notional_usd: float = 1000.0,
    placebo_iterations: int = 40,
):
    """Purged CV, cross-asset, regime and placebo stress suite."""
    d=dataset(store(db),horizon=horizon); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    if len(d)<100: typer.echo("Need >=100 labeled rows"); raise typer.Exit(2)
    purged=purged_walk_forward_ablation(d,horizon=horizon,model_name=model_name,notional_usd=notional_usd)
    loso=leave_one_symbol_out(d,model_name=model_name,notional_usd=notional_usd,horizon=horizon)
    regimes=heldout_regime_report(d,model_name=model_name,notional_usd=notional_usd,horizon=horizon)
    placebo=placebo_test(d,iterations=placebo_iterations,model_name="logistic")
    drift=drift_report(d)
    purged.to_csv(out/'purged_walk_forward.csv',index=False); loso.to_csv(out/'leave_one_symbol_out.csv',index=False); regimes.to_csv(out/'regimes.csv',index=False)
    summary={"rows":len(d),"horizon":horizon,"placebo":placebo,"drift":drift,"purged":purged.to_dict(orient='records'),"leave_one_symbol_out":loso.to_dict(orient='records'),"regimes":regimes.to_dict(orient='records')}
    (out/'robustness.json').write_text(json.dumps(summary,indent=2,default=str))
    typer.echo(f"Wrote {out/'robustness.json'}")


@app.command("evidence-gate")
def evidence_gate_cmd(
    db: str | None = typer.Option(None),
    horizon: str = "15m",
    model_name: str = "histgb",
    notional_usd: float = 1000.0,
    output: str = "./runs/EVIDENCE_GATE.json",
):
    d=dataset(store(db),horizon=horizon)
    result=evidence_gate(d,horizon=horizon,model_name=model_name,notional_usd=notional_usd)
    p=Path(output); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(result,indent=2,default=str))
    typer.echo(json.dumps({"status":result.get("status"),"rows":result.get("rows"),"symbols":result.get("symbols"),"output":str(p)},indent=2))


@app.command("drift-report")
def drift_report_cmd(db: str | None = typer.Option(None), horizon: str = "15m", output: str | None = None):
    r=drift_report(dataset(store(db),horizon=horizon))
    if output:
        p=Path(output); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(r,indent=2,default=str))
    typer.echo(json.dumps(r,indent=2,default=str))


@app.command("paper-step")
def paper_step_cmd(
    model_path: str = "./models/semantic_residual.joblib",
    db: str | None = typer.Option(None),
    notional_usd: float = 1000.0,
    equity_usd: float = 100000.0,
    risk_per_trade: float = 0.0025,
    max_positions: int = 3,
    max_feature_age_seconds: int = 60,
    max_market_age_seconds: int = 10,
):
    r=paper_step(store(db),SemanticResidualModel.load(model_path),base_notional_usd=notional_usd,equity_usd=equity_usd,risk_per_trade=risk_per_trade,max_positions=max_positions,max_feature_age_seconds=max_feature_age_seconds,max_market_age_seconds=max_market_age_seconds)
    typer.echo(json.dumps(r,indent=2,default=str))


@app.command("paper-status")
def paper_status_cmd(db: str | None = typer.Option(None), model_path: str | None = typer.Option(None)):
    fingerprint=None
    if model_path:
        fingerprint=SemanticResidualModel.load(model_path).training_fingerprint
    typer.echo(json.dumps(paper_summary(store(db),model_fingerprint=fingerprint),indent=2,default=str))


@app.command("paper-attribution")
def paper_attribution_cmd(db: str | None = typer.Option(None), model_path: str | None = typer.Option(None), output: str | None = None):
    """Attribute closed paper P&L by symbol, side, entry regime and semantic-edge strength."""
    fingerprint=SemanticResidualModel.load(model_path).training_fingerprint if model_path else None
    r=paper_attribution(store(db),model_fingerprint=fingerprint)
    if output:
        p=Path(output); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(r,indent=2,default=str))
    typer.echo(json.dumps(r,indent=2,default=str))


@app.command("paper-run")
def paper_run(
    model_path: str = "./models/semantic_residual.joblib",
    db: str | None = typer.Option(None),
    interval_seconds: int = 15,
    iterations: int = 0,
    notional_usd: float = 1000.0,
    equity_usd: float = 100000.0,
    risk_per_trade: float = 0.0025,
    max_positions: int = 3,
    max_feature_age_seconds: int = 60,
    max_market_age_seconds: int = 10,
):
    """Continuously paper-trade latest recorded features. Never sends exchange orders."""
    async def run():
        m=SemanticResidualModel.load(model_path); s=store(db); n=0
        while iterations==0 or n<iterations:
            r=paper_step(s,m,base_notional_usd=notional_usd,equity_usd=equity_usd,risk_per_trade=risk_per_trade,max_positions=max_positions,max_feature_age_seconds=max_feature_age_seconds,max_market_age_seconds=max_market_age_seconds)
            typer.echo(json.dumps({"ts":datetime.now(timezone.utc).isoformat(),"opened":len(r['opened']),"closed":len(r['closed']),"open_positions":r['open_positions']}))
            n+=1
            if iterations==0 or n<iterations: await asyncio.sleep(max(1,interval_seconds))
    try: asyncio.run(run())
    except KeyboardInterrupt: typer.echo("Stopped paper trader")


@app.command("maintenance-step")
def maintenance_step_cmd(
    db: str | None = typer.Option(None),
    benchmark_symbol: str = "BTCUSDT",
    semantic_model: str | None = typer.Option(None),
):
    """Backfill matured labels and materialize point-in-time author reputation."""
    typer.echo(json.dumps(maintenance_step(store(db),benchmark_symbol=benchmark_symbol,semantic_model=semantic_model),indent=2,default=str))


@app.command("maintenance-run")
def maintenance_run(
    db: str | None = typer.Option(None),
    interval_minutes: int = 15,
    iterations: int = 0,
    benchmark_symbol: str = "BTCUSDT",
    semantic_model: str | None = typer.Option(None),
    engagement_refresh: bool = True,
    engine: str = "jev",
    backfill_limit: int = 200,
):
    """Continuously maintain labels/reputation while the live recorder is running.

    With ``engagement_refresh``, each cycle also re-observes a small trailing
    slice of posts via X search so engagement-velocity features have a second
    observation point (bills per returned post; bounded to 2 pages/symbol).
    ``engine`` backfills up to 200 missing semantic classifications per cycle —
    posts dropped or timed out during live ingestion would otherwise stay
    permanently unclassified."""
    async def run():
        s=store(db)
        social = XAPIClient(settings.x_api_key, settings.x_api_url) if engagement_refresh else None
        sem = None
        if engine == 'jev' and settings.typesafe_api_key:
            sem = JevSemanticEngine(settings.typesafe_api_key, settings.jev_model)
        elif engine == 'heuristic':
            sem = HeuristicSemanticEngine()
        pulse_engine = sem if isinstance(sem, JevSemanticEngine) else None
        n=0
        while iterations==0 or n<iterations:
            typer.echo(json.dumps(maintenance_step(s,benchmark_symbol=benchmark_symbol,semantic_model=semantic_model),default=str))
            if social is not None:
                try:
                    typer.echo(json.dumps(await refresh_engagement(s, social),default=str))
                except Exception as e:
                    typer.echo(json.dumps({"engagement_refresh_error": str(e)[:200]}))
            if sem is not None:
                try:
                    typer.echo(json.dumps(await backfill_semantics(s, sem, limit=backfill_limit),default=str))
                except Exception as e:
                    typer.echo(json.dumps({"backfill_error": str(e)[:200]}))
            if pulse_engine is not None:
                for sym in s.recent_post_symbols(datetime.now(timezone.utc) - timedelta(hours=2)):
                    try:
                        typer.echo(json.dumps(await regime_pulse(s, pulse_engine, sym),default=str))
                    except Exception as e:
                        typer.echo(json.dumps({"regime_pulse_error": sym, "error": str(e)[:200]}))
            n+=1
            if iterations==0 or n<iterations: await asyncio.sleep(max(60,interval_minutes*60))
    try: asyncio.run(run())
    except KeyboardInterrupt: typer.echo("Stopped maintenance loop")


@app.command("horizon-scan")
def horizon_scan_cmd(
    db: str | None = typer.Option(None),
    model_name: str = "logistic",
    notional_usd: float = 1000.0,
    output: str | None = None,
):
    """Scan 1m-24h to find semantic alpha half-life and possible sign reversals."""
    d=horizon_scan(store(db),model_name=model_name,notional_usd=notional_usd)
    if output:
        p=Path(output); p.parent.mkdir(parents=True,exist_ok=True); d.to_csv(p,index=False)
    typer.echo(d.to_string(index=False))


@app.command("feature-importance")
def feature_importance_cmd(
    db: str | None = typer.Option(None),
    horizon: str = "15m",
    model_name: str = "histgb",
    top: int = 30,
    output: str | None = None,
):
    """Held-out permutation importance for the full model."""
    d=feature_importance_report(dataset(store(db),horizon=horizon),model_name=model_name)
    if output:
        p=Path(output); p.parent.mkdir(parents=True,exist_ok=True); d.to_csv(p,index=False)
    typer.echo(d.head(top).to_string(index=False) if not d.empty else "Insufficient data")


@app.command("research-suite")
def research_suite(
    output_dir: str = "./runs/research-suite",
    db: str | None = typer.Option(None),
    horizon: str = "15m",
    model_name: str = "histgb",
    notional_usd: float = 1000.0,
    placebo_iterations: int = 30,
    feature_version: str = "v4",
):
    """Run the main anti-overfit research battery and create a candidate model artifact."""
    s=store(db); d=dataset(s,horizon=horizon,feature_version=feature_version); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    if len(d)<100: typer.echo(f"Need >=100 labeled rows, found {len(d)}"); raise typer.Exit(2)
    quality=data_quality(s); audit=leakage_audit(s)
    inc=incremental_semantic_test(d,model_name=model_name,notional_usd=notional_usd,bootstrap_samples=300,horizon=horizon)
    purged=purged_walk_forward_ablation(d,horizon=horizon,model_name=model_name,notional_usd=notional_usd)
    loso=leave_one_symbol_out(d,model_name=model_name,notional_usd=notional_usd,horizon=horizon)
    regimes=heldout_regime_report(d,model_name=model_name,notional_usd=notional_usd,horizon=horizon)
    placebo=placebo_test(d,iterations=placebo_iterations,model_name="logistic")
    drift=drift_report(d); horizons=horizon_scan(s,model_name="logistic",notional_usd=notional_usd,bootstrap_samples=100)
    importance=feature_importance_report(d,model_name=model_name)
    event_grid=event_study_grid(s,min_n=5)
    gate=evidence_gate(d,horizon=horizon,model_name=model_name,notional_usd=notional_usd,placebo_iterations=placebo_iterations)
    purged.to_csv(out/'purged_walk_forward.csv',index=False); loso.to_csv(out/'leave_one_symbol_out.csv',index=False)
    regimes.to_csv(out/'regimes.csv',index=False); horizons.to_csv(out/'horizon_scan.csv',index=False); importance.to_csv(out/'feature_importance.csv',index=False)
    if not event_grid.empty: event_grid.to_csv(out/'event_study_grid.csv',index=False)
    d.to_csv(out/'dataset.csv',index=False)
    baseline=[x for g in ["price_derivatives","attention"] for x in FEATURE_GROUPS[g] if x in d.columns]
    full=[x for g in ["price_derivatives","attention","semantics","gap"] for x in FEATURE_GROUPS[g] if x in d.columns]
    model=SemanticResidualModel.fit(d,baseline,full,model_name=model_name,horizon=horizon,notional_usd=notional_usd)
    model.save(out/'candidate_model.joblib')
    manifest=research_manifest(
        d,horizon=horizon,semantic_models=source_semantic_models(s),
        config={"model_name":model_name,"notional_usd":notional_usd,"placebo_iterations":placebo_iterations,"feature_version":feature_version,"label_version":"v4"},
    )
    (out/'EXPERIMENT.json').write_text(json.dumps(manifest,indent=2,default=str))
    report={
        "rows":len(d),"symbols":sorted(d.symbol.unique().tolist()),"horizon":horizon,"quality":quality,"leakage":audit,"experiment":manifest,
        "incremental_semantic_test":inc,"placebo":placebo,"drift":drift,"evidence_gate":gate,
        "event_study":{"tests":int(len(event_grid)),"global_fdr_10pct":int(event_grid.fdr_global_10pct.sum()) if not event_grid.empty else 0},
        "candidate_model":{"version":model.version,"model_name":model.model_name,"trained_rows":model.trained_rows,"training_fingerprint":model.training_fingerprint,"probability_threshold":model.probability_threshold,"semantic_edge_threshold":model.semantic_edge_threshold,"calibration":model.calibration},
    }
    (out/'REPORT.json').write_text(json.dumps(report,indent=2,default=str))
    s.record_trial("", "research-suite", {"rows":len(d),"horizon":horizon,"model":model_name,"gate":gate.get("status"),"output":str(out)})
    typer.echo(json.dumps({"output":str(out),"rows":len(d),"symbols":len(report['symbols']),"gate":gate.get('status'),"model":str(out/'candidate_model.joblib')},indent=2))


@app.command("locked-research")
def locked_research_cmd(
    output_dir: str = "./runs/locked-research",
    db: str | None = typer.Option(None),
    horizons: str = "5m,15m,30m,1h",
    models: str = "logistic,histgb",
    holdout_fraction: float = 0.20,
    notional_usd: float = 1000.0,
    bootstrap_samples: int = 500,
    sampling_mode: str = "event",
    allow_reopen: bool = False,
    feature_version: str = "v4",
):
    """Search on discovery data, then open one sealed temporal holdout exactly once."""
    hs=[x.strip() for x in horizons.split(',') if x.strip()]
    ms=[x.strip() for x in models.split(',') if x.strip()]
    s=store(db)
    r=run_locked_protocol(
        s, output_dir, horizons=hs, model_names=ms, holdout_fraction=holdout_fraction,
        notional_usd=notional_usd, bootstrap_samples=bootstrap_samples, sampling_mode=sampling_mode, allow_reopen=allow_reopen,
        feature_version=feature_version,
    )
    s.record_trial("", "locked-research", {"horizons":hs,"models":ms,"status":r.get("status"),"output":output_dir})
    typer.echo(json.dumps({
        "status":r.get("status"), "output":output_dir,
        "selected":r.get("selected_on_discovery_only"),
        "forward_candidate":r.get("forward_candidate"),
    },indent=2,default=str))
    if r.get("status") == "INSUFFICIENT_DATA": raise typer.Exit(2)


@app.command("semantic-placebo")
def semantic_placebo_cmd(
    db: str | None = typer.Option(None), horizon: str = "15m", model_name: str = "logistic", iterations: int = 30,
):
    """Destroy semantic timing in training and compare against the aligned semantic model."""
    r=semantic_permutation_placebo(dataset(store(db),horizon=horizon),horizon=horizon,model_name=model_name,iterations=iterations)
    typer.echo(json.dumps(r,indent=2,default=str))


@app.command("semantic-staleness")
def semantic_staleness_cmd(
    db: str | None = typer.Option(None), horizon: str = "15m", model_name: str = "histgb", shifts: str = "1,4,12",
):
    """Check whether semantic predictive value decays when only older social state is supplied."""
    vals=tuple(int(x.strip()) for x in shifts.split(',') if x.strip())
    r=semantic_staleness_test(dataset(store(db),horizon=horizon),horizon=horizon,model_name=model_name,shifts=vals)
    typer.echo(r.to_string(index=False) if not r.empty else "Insufficient data")


@app.command("storage-health")
def storage_health_cmd(db: str | None = typer.Option(None)):
    """Show database growth and 30-day market-row/storage projection."""
    typer.echo(json.dumps(store(db).storage_health(),indent=2,default=str))


@app.command("compact-market")
def compact_market_cmd(
    db: str | None = typer.Option(None), older_than_days: int = 7, bucket_seconds: int = 60, execute: bool = False,
):
    """Downsample old market snapshots to the last observation per bucket. Dry-run by default."""
    s=store(db); health=s.storage_health(); first=health.get("market_first_ts")
    if not first:
        typer.echo(json.dumps({"status":"NO_MARKET_DATA"},indent=2)); return
    start=datetime.fromisoformat(str(first)).replace(hour=0,minute=0,second=0,microsecond=0)
    cutoff=(datetime.now(timezone.utc)-timedelta(days=max(1,older_than_days))).replace(hour=0,minute=0,second=0,microsecond=0)
    totals={"days":0,"scanned":0,"kept":0,"would_delete":0,"deleted":0,"executed":execute}
    cur=start
    while cur<cutoff:
        nxt=min(cutoff,cur+timedelta(days=1)); r=s.compact_market_range(cur,nxt,bucket_seconds=bucket_seconds,execute=execute)
        totals["days"]+=1
        for k in ("scanned","kept","would_delete","deleted"): totals[k]+=int(r[k])
        cur=nxt
    totals["warning"]="Compaction is destructive when --execute is used. Materialize features/labels first if sub-minute raw history may be needed."
    typer.echo(json.dumps(totals,indent=2,default=str))


@app.command("cost-report")
def cost_report_cmd(db: str | None = typer.Option(None), days: int = 30):
    """Estimate X/Jev operating cost and compare it with simulated paper P&L."""
    typer.echo(json.dumps(operating_economics(store(db),days=days),indent=2,default=str))


@app.command("usage-report")
def usage_report_cmd(db: str | None = typer.Option(None), days: int = 30):
    """Show recorded provider usage/cost estimates."""
    typer.echo(json.dumps(usage_summary(store(db),days=days),indent=2,default=str))


@app.command("factor-ic-report")
def factor_ic_report_cmd(
    db: str | None = typer.Option(None), horizon: str = "15m", bucket: str = "5min",
    min_assets: int = 3, factors: str | None = None, output: str | None = None,
):
    """Cross-sectional Spearman IC for raw semantic/social factors before ML."""
    fs=[x.strip() for x in factors.split(',') if x.strip()] if factors else None
    r=factor_ic_report(dataset(store(db),horizon=horizon),fs,bucket=bucket,min_assets=min_assets)
    if r.empty:
        typer.echo("Insufficient synchronized multi-asset data"); raise typer.Exit(2)
    if output:
        p=Path(output); p.parent.mkdir(parents=True,exist_ok=True); r.to_csv(p,index=False)
    typer.echo(r.to_string(index=False))


@app.command("factor-monotonicity")
def factor_monotonicity_cmd(
    factor: str, db: str | None = typer.Option(None), horizon: str = "15m", bins: int = 5,
):
    """Show future abnormal returns across quantiles of one raw factor."""
    r=factor_monotonicity(dataset(store(db),horizon=horizon),factor,bins=bins)
    typer.echo(r.to_string(index=False) if not r.empty else "Insufficient data")


@app.command("campaign")
def campaign_cmd(
    output_dir: str = "./runs/campaign",
    db: str | None = typer.Option(None),
    horizons: str = "5m,15m,30m,1h",
    models: str = "logistic,histgb",
    holdout_fraction: float = 0.20,
    notional_usd: float = 1000.0,
    placebo_iterations: int = 30,
    sampling_mode: str = "event",
    allow_reopen: bool = False,
    feature_version: str = "v4",
):
    """Run the full research -> locked-holdout -> forward-paper evidence campaign."""
    r=run_campaign(
        store(db),output_dir,horizons=[x.strip() for x in horizons.split(',') if x.strip()],
        model_names=[x.strip() for x in models.split(',') if x.strip()],holdout_fraction=holdout_fraction,
        notional_usd=notional_usd,placebo_iterations=placebo_iterations,sampling_mode=sampling_mode,allow_reopen=allow_reopen,
        feature_version=feature_version,
    )
    typer.echo(json.dumps({
        "status":r.get("status"),"output":output_dir,"selected_horizon":r.get("selected_horizon"),
        "locked_holdout":r.get("locked_holdout",{}).get("status"),"evidence_gate":r.get("evidence_gate",{}).get("status"),
        "closed_paper_trades":r.get("paper",{}).get("closed_trades",0),
    },indent=2,default=str))


@app.command("promote-model")
def promote_model_cmd(
    model_path: str,
    research_report: str,
    registry_dir: str = "./models/registry",
    db: str | None = typer.Option(None),
    min_closed_trades: int = 100,
    min_profit_factor: float = 1.20,
    max_drawdown: float = -0.10,
    max_top5_profit_share: float = 0.50,
    max_high_drift_features: int = 2,
    force: bool = False,
):
    """Register a candidate and promote only if forward evidence + paper gates pass.

    This never enables exchange execution. `--force` exists for research/admin use and
    is recorded in champion metadata.
    """
    r=promote_candidate(
        candidate_model=model_path,research_report=research_report,store=store(db),registry_dir=registry_dir,force=force,
        min_closed_trades=min_closed_trades,min_profit_factor=min_profit_factor,max_drawdown=max_drawdown,
        max_top5_profit_share=max_top5_profit_share,max_high_drift_features=max_high_drift_features,
    )
    typer.echo(json.dumps(r,indent=2,default=str))
    if not r.get("promoted") and not force:
        raise typer.Exit(3)


@app.command("champion-status")
def champion_status_cmd(registry_dir: str = "./models/registry"):
    typer.echo(json.dumps(champion_status(registry_dir),indent=2,default=str))


@app.command("semantic-engine-compare")
def semantic_engine_compare_cmd(
    versions: str,
    db: str | None = typer.Option(None),
    horizon: str = "15m",
    model_name: str = "logistic",
    notional_usd: float = 1000.0,
    output: str | None = None,
):
    """Compare Jev/heuristic/other semantic builds on the exact same timestamps."""
    vs=[x.strip() for x in versions.split(',') if x.strip()]
    if len(vs)<2: raise typer.BadParameter("provide at least two feature versions")
    r=compare_feature_versions(store(db),vs,horizon=horizon,model_name=model_name,notional_usd=notional_usd)
    if r.empty:
        typer.echo("Need >=100 common labeled rows across at least two feature versions"); raise typer.Exit(2)
    if output:
        p=Path(output); p.parent.mkdir(parents=True,exist_ok=True); r.to_csv(p,index=False)
    typer.echo(r.to_string(index=False))


@app.command("semanticize-range")
def semanticize_range(
    symbols: str,
    start: str,
    end: str,
    engine: str = "jev",
    db: str | None = typer.Option(None),
):
    """Semanticize an already-ingested historical time range."""
    async def run():
        s=store(db); st=datetime.fromisoformat(start.replace('Z','+00:00')); en=datetime.fromisoformat(end.replace('Z','+00:00'))
        sem=JevSemanticEngine(settings.typesafe_api_key,settings.jev_model) if engine=='jev' else HeuristicSemanticEngine()
        model_name=getattr(sem,'model',engine); done=0; skipped=0
        for sym in [x.strip().upper() for x in symbols.split(',') if x.strip()]:
            for p in s.posts_between(sym,st,en):
                if s.semantics_for_post(p.post_id,model_name,sym): skipped+=1; continue
                s.save_semantics((await sem.classify(p)).model_copy(update={"symbol":sym})); done+=1
        typer.echo(json.dumps({'semanticized':done,'already_done':skipped,'model':model_name},indent=2))
    asyncio.run(run())


@app.command("rebuild-historical-features")
def rebuild_historical_features_cmd(
    symbols: str,
    start: str,
    end: str,
    interval_minutes: int = 5,
    semantic_model: str | None = typer.Option(None),
    feature_version: str = "v4",
    db: str | None = typer.Option(None),
):
    """Build point-in-time features from historical data; use distinct versions for semantic-engine comparisons."""
    st=datetime.fromisoformat(start.replace('Z','+00:00')); en=datetime.fromisoformat(end.replace('Z','+00:00'))
    syms=[x.strip().upper() for x in symbols.split(',') if x.strip()]
    r=rebuild_historical_features(store(db),syms,st,en,interval_minutes=interval_minutes,semantic_model=semantic_model,feature_version=feature_version)
    typer.echo(json.dumps(r,indent=2,default=str))


@app.command("enrich-cross-asset")
def enrich_cross_asset_cmd(
    ts: str | None = None,
    symbols: str | None = None,
    feature_version: str = "v4",
    db: str | None = typer.Option(None),
):
    """Add universe-relative context to feature snapshots at a point in time."""
    t=datetime.fromisoformat(ts.replace('Z','+00:00')) if ts else datetime.now(timezone.utc)
    syms=[x.strip().upper() for x in symbols.split(',') if x.strip()] if symbols else None
    n=enrich_cross_asset_at(store(db),t,symbols=syms,feature_version=feature_version)
    typer.echo(json.dumps({'ts':t.isoformat(),'updated':n},indent=2))


@app.command("adversarial-benchmark")
def adversarial_benchmark_cmd(
    engine: str = "heuristic", symbol: str = "SOL",
    output: str | None = typer.Option(None),
):
    """Probe the semantic engine with manipulated posts: injection, spam, fake sources, negation."""
    async def run():
        sem = JevSemanticEngine(settings.typesafe_api_key, settings.jev_model) if engine == "jev" else HeuristicSemanticEngine()
        r = await adversarial_benchmark(sem, symbol=symbol)
        if output:
            p = Path(output); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(r, indent=2, default=str))
        typer.echo(json.dumps({k: v for k, v in r.items() if k != "rows"}, indent=2, default=str))
    asyncio.run(run())


@app.command("overfit-report")
def overfit_report_cmd(
    db: str | None = typer.Option(None), horizon: str = "15m",
    model_name: str = "logistic", groups: int = 8, notional_usd: float = 1000.0,
    output: str | None = typer.Option(None),
):
    """CPCV/PBO + deflated-Sharpe audit of the model-selection process itself."""
    s = store(db); d = dataset(s, horizon=horizon)
    if len(d) < 200:
        typer.echo("Need >=200 labeled rows for CPCV"); raise typer.Exit(2)
    hm = HORIZONS.get(horizon, 15)
    r = overfit_report(s, d, horizon=horizon, horizon_minutes=hm, model_name=model_name, n_groups=groups, notional_usd=notional_usd)
    s.record_trial("", "overfit-report", {"horizon": horizon, "model": model_name, "pbo": r.get("pbo")})
    if output:
        p = Path(output); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(r, indent=2, default=str))
    typer.echo(json.dumps(r, indent=2, default=str))


@app.command("trials")
def trials_cmd(db: str | None = typer.Option(None), limit: int = 50):
    """Append-only experiment ledger — every recorded run counts toward effective trials."""
    s = store(db)
    typer.echo(json.dumps({"trial_count": s.trial_count(), "recent": s.trials(limit)}, indent=2, default=str))


@app.command("collect-news")
def collect_news_cmd(
    feeds: str | None = typer.Option(None),
    symbols: str = "BTCUSDT,ETHUSDT,SOLUSDT",
    db: str | None = typer.Option(None),
):
    """Poll RSS/Atom primary sources (NEWS_FEEDS=name=url,...) into the post pipeline."""
    async def run():
        from .providers.rss import RSSClient, items_to_posts
        s = store(db)
        feed_str = feeds or settings.news_feeds
        if not feed_str:
            raise typer.BadParameter("NEWS_FEEDS empty — set name=url pairs or pass --feeds")
        pairs = {}
        for kv in feed_str.split(","):
            if "=" in kv:
                n, u = kv.split("=", 1)
                pairs[n.strip()] = u.strip()
        syms = [x.strip().upper() for x in symbols.split(",") if x.strip()]
        client = RSSClient(); fetched = inserted = 0
        for name, url in pairs.items():
            try:
                items = await client.fetch(url)
            except Exception as e:
                typer.echo(f"{name}: fetch failed {e}", err=True); continue
            fetched += len(items)
            for p in items_to_posts(items, name, syms):
                inserted += int(s.save_post(p))
        typer.echo(json.dumps({"feeds": len(pairs), "items": fetched, "inserted": inserted}, indent=2))
    asyncio.run(run())


@app.command("capacity-report")
def capacity_report_cmd(
    db: str | None = typer.Option(None), days: int = 30,
    notionals: str | None = typer.Option(None),
):
    """Estimated one-way execution-cost curve across position sizes from stored book depth."""
    ns = [float(x) for x in notionals.split(",")] if notionals else None
    typer.echo(json.dumps(capacity_report(store(db), ns, days=days), indent=2, default=str))


@app.command("latency-report")
def latency_report_cmd(db: str | None = typer.Option(None)):
    """Ingestion + semantic classification latency distributions (ingest lag, classify lag)."""
    typer.echo(json.dumps(latency_report(store(db)), indent=2, default=str))


@app.command("feed-health")
def feed_health_cmd(hours: int = 6, db: str | None = typer.Option(None)):
    """Per-symbol feed liveness: reconnects, stale-state events, latest ages.

    A dead WS feed shows up here as stale_state events and absent rows — never
    as fresh-looking market data."""
    typer.echo(json.dumps(feed_health_report(store(db), hours=hours), indent=2, default=str))


@app.command("stream-probe")
def stream_probe_cmd(
    seconds: int = 90, max_tweets: int = 100, db: str | None = typer.Option(None),
):
    """Measure real X stream latency: connect the filter-rule WebSocket and record
    created_at -> delivery lag for every tweet that arrives. Writes the result to
    feed_health (kind='stream_latency_probe') so it is auditable.

    Requires an active filter rule (x-add-rules activates on creation). Zero
    deliveries usually means the API key is out of credits — connection is free,
    delivery is billed per tweet."""
    import numpy as np
    from collections import Counter
    from email.utils import parsedate_to_datetime
    import httpx

    async def clock_offset_s() -> float | None:
        """server_utc - local_utc via the vendor's HTTP Date header."""
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                t0 = datetime.now(timezone.utc).timestamp()
                r = await c.get(f"{settings.x_base_url}/oapi/my/info",
                                headers={"x-api-key": settings.x_api_key})
                t1 = datetime.now(timezone.utc).timestamp()
                d = r.headers.get("date")
                if not d:
                    return None
                return parsedate_to_datetime(d).timestamp() - (t0 + t1) / 2
        except Exception:
            return None

    async def probe():
        offset = await clock_offset_s()  # positive => local clock behind UTC
        stream = XRealtimeStream(settings.x_api_key, settings.x_ws_url)
        lags: list[float] = []
        kinds: Counter[str] = Counter()
        samples: list[dict] = []
        t0 = datetime.now(timezone.utc)

        async def collect():
            async for ev in stream.events():
                if not isinstance(ev, dict):
                    continue
                kinds[ev.get("event_type", "unknown")] += 1
                pairs = posts_from_event(ev)
                if not pairs:
                    continue
                now = datetime.now(timezone.utc)
                for tag, raw in pairs:
                    created = _parse_x_ts(
                        raw.get("created_at") or raw.get("createdAt")
                        or raw.get("createdAtTimestamp")
                        or (raw.get("tweet") or {}).get("created_at")
                    )
                    lag = (now - created).total_seconds()
                    lags.append(lag)
                    if len(samples) < 10:
                        samples.append({"rule_tag": tag, "lag_s": round(lag, 3),
                                        "created_at": created.isoformat()})
                if len(lags) >= max_tweets:
                    return
        try:
            await asyncio.wait_for(collect(), timeout=seconds)
        except asyncio.TimeoutError:
            pass
        corrected = [l - offset for l in lags] if offset is not None else None
        return {
            "probe_seconds": (datetime.now(timezone.utc) - t0).total_seconds(),
            "delivered": len(lags),
            "event_kinds": dict(kinds),
            "local_clock_offset_s": round(offset, 2) if offset is not None else None,
            "lag_s": (
                {"min": round(min(lags), 3), "p50": round(float(np.percentile(lags, 50)), 3),
                 "p90": round(float(np.percentile(lags, 90)), 3),
                 "max": round(max(lags), 3), "mean": round(float(np.mean(lags)), 3)}
                if lags else None
            ),
            "lag_s_clock_corrected": (
                {"min": round(min(corrected), 3), "p50": round(float(np.percentile(corrected, 50)), 3),
                 "p90": round(float(np.percentile(corrected, 90)), 3),
                 "max": round(max(corrected), 3), "mean": round(float(np.mean(corrected)), 3)}
                if corrected else None
            ),
            "delivery_per_min": round(len(lags) / max((datetime.now(timezone.utc) - t0).total_seconds() / 60, 1e-9), 2),
            "samples": samples,
        }

    result = asyncio.run(probe())
    result["note"] = (
        "no deliveries — check credit balance (oapi/my/info) and that a rule is active (x-rules)"
        if result["delivered"] == 0 else None
    )
    store(db).save_feed_health("STREAM", "stream_latency_probe", result)
    typer.echo(json.dumps(result, indent=2, default=str))


@app.command("semantic-reliability")
def semantic_reliability_cmd(
    db: str | None = typer.Option(None), horizon: str = "15m", bins: int = 5,
):
    """Binned reliability of semantic probability heads vs realized |abnormal return|."""
    r = semantic_reliability(store(db), horizon=horizon, bins=bins)
    typer.echo(r.to_string(index=False) if not r.empty else "Insufficient labeled semantic data")


if __name__ == "__main__":
    app()
