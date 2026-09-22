# Changelog

## 0.5.2
Data-quality release driven by the first deep audit of live-collected data: the X rules match ecosystem-keyword noise (60.7% of the SOL feed classifies IRRELEVANT — Solana-chain memecoin spam), and semantic aggregates averaged over every classified post let that spam pose as signal. `feature_version` bumps `v3`→`v4`; v3 rows remain as the pilot dataset, version-scoped.

- FEATURES: all semantic aggregates are now relevance-weighted by Jev's own verdict — `RELEVANT`=1.0, `MAYBE`=0.5, `IRRELEVANT`=0 — so an irrelevant post contributes nothing to `intent_*`, `new_info_rate`, `shill_rate`, `evidence_rate`, catalyst/type/source-role rates, `impact_mean`, `rumor_impact`, `official_new_info_impact`, `semantic_confidence_mean`, and `intent_agreement` (now weight-aware). `intent_sum` is a weighted conviction mass. `semantic_live_rate` stays unweighted (it is a pipeline-latency metric, not signal).
- FEATURES: new `rel_posts_{1,5,15}m` (effective relevant-post count = Σweights) and `rel_share_*` (relevant fraction of classified posts) expose the feed-purity dimension directly.
- OPS: three per-symbol X rules consolidated into one `semantic-alpha-all` rule — vendor bills ~15 credits per delivered tweet-instance *per matching rule*, so multi-asset tweets billed 2–3×; one rule bills once and local text-matching still assigns all symbols. Usage is logged at receipt for every delivery including symbol-unmatched and queue-dropped ones (billed spend was invisible on those paths); the `filter_rule_poll` estimate was removed — the dashboard showed billing is per delivered tweet, so it double-counted.
- OPS: `--delivery-recycle-seconds` must exceed the rule interval — the 300s default recycled healthy sockets mid-window at 600s cadence (1200s is right for 600s rules).
- AUDIT (docs only): ~17h pilot dataset — 9,471 posts / 10,616 semantics / 14,380 market snaps / 11,000 labels / 7,017 events. Zero negative lags; PIT chain verified. Semantic-rate features co-move at rho≈0.97 (one latent conviction-density factor). Feature-level partial rho +0.19 vs 15m returns did **not** replicate at event level (n=6,999, rho≈0) or at independent bucket level (n=125, rho≈0) — treated as spam-intensity artifact, not alpha. See STATUS.md.
- PERF/ECONOMICS: `_narrative_features` fitted a fresh `TfidfVectorizer` per (recent×prior) event pair — ~140k fits per build once events accumulated → multi-minute event-loop stalls → X keepalive timeouts (`1011` drops every 2–4 min) → vendor replayed buffered tweets and **billed them again** (~26% of deliveries were replays). Now uses `batch_similarity` (one fit per build, prior capped at the 2k most recent events): builds went ~140s→~2.7s and the socket stayed connected with zero drops after restart.
- OPS: X socket `ping_timeout` 20s→120s so residual stalls can't sever the connection.
- OPS: run `live` with `--feature-seconds 60` — three symbol builds ≈8s/min leaves the loop mostly idle; the 15s default saturated it as tables grew.
- ECONOMICS: maintenance `--no-engagement-refresh` disables the trailing-slice re-observation: it was ~85% of `post_read` spend (~1,635cr per 15-min tick) for `engagement_velocity`, which showed ρ=+0.127 on n=57 (noise). Search-fetched posts were already ledgered via `_semantic_alpha_acquisition_id`.
- LINT: `pyproject` uses `extend-exclude` for `_old_src`/`build` — a plain `exclude` replaces ruff's defaults and pulled `.venv` into scope.
- ECONOMICS: `search_window` accepts `known_ids` — posts already stored are skipped, and since `queryType=Latest` is reverse-chronological, an all-known page halts pagination (billed-per-returned-tweet, so trailing duplicate pages were pure spend). Wired into leading-edge collect paths (`collect_cycle`, `x-window`) only — gap backfills and engagement refresh deliberately keep `known_ids=None` (they need posts behind/beyond the known edge).
- FEATURES (candidate channel, not wired into v4): `regime_pulse` — one holistic Jev `system_one` call per symbol per maintenance cycle over a fixed digest (market state + 15m social aggregates + top-5-engaged/latest-5 posts), answering `market_regime`/`sentiment_spectrum`/`squeeze_risk`/`catalyst_significance`/`trade_action`. ~$0.02/day. Stored raw in `regime_pulses` with usage ledgered; must validate on untouched data before any feature may depend on it.
- OPS/ECONOMICS: classification-failure bookkeeping (`classification_failures` table). Failures split into *counted* (content/model-side: count toward a cap of 3, then `unsemanticized_posts` excludes the pair — deterministic junk no longer burns a billed classify call every maintenance tick forever) and *uncounted* (provider-side 402/429/timeout/transport: recorded but never capped, so a provider outage can't poison posts into permanent failure and backlog resumes automatically when credits return). Both the live worker and `backfill_semantics` record failures.
- OPS: `backfill_semantics` reworked — per-post `try/except` with a 90s `wait_for` (a hung classify no longer stalls the maintenance loop), continues past single content failures instead of dying on the first exception, and stops only after 5 *consecutive* provider errors (provider down/out-of-credits detection — during the 2026-09-21 17:35→01:00 TypeSafe credit outage the old `break-on-first-error` pass did exactly one futile billed call per ~50min tick). `maintenance-run --backfill-limit` raises the per-tick cap (default 200) to drain accumulated backlog faster.
- OPS: `drain_semantics.py` — bounded parallel backlog drain (6 concurrent classifies, 90s per-call timeout, same failure bookkeeping, stops after 5 consecutive provider errors). Re-resolves `query_model` every batch — the alias→effective-model resolution only happens after the first response, and a frozen alias makes every stored row look pending → infinite billed reclassification (bit this once: ~1,740 calls reclassified the same 60 posts before the fix).
- PERF: `maintenance_step` no longer full-table-scans every tick (~80min at ~30k posts). Label existence and actionable anchors now resolve from three bulk loads — `label_ts_set`, `max_classified_map`, `posts_with_symbols_since` — and the anchor scan is bounded to `scan_window_hours=72` (anchors older than max horizon + grace that still lack labels are missing market data live mode never backfills; historical windows use `backfill-labels`). The immature-label pass gates on each row's earliest open horizon (a label can't newly mature before its first unrealized exit window opens) and *finalizes* labels anchored >26h whose rebuild still finds nothing — rows stuck open on permanent market-data gaps no longer rescan forever (`labels_finalized` counter; horizons stay `None` — honest missing data, never fabricated).
- OPS: `drain_semantics.py` — bounded parallel backlog drain (6 concurrent classifies, 90s per-call timeout, same failure bookkeeping, stops after 5 consecutive provider errors). Re-resolves `query_model` every batch — the alias→effective-model resolution only happens after the first response, and a frozen alias makes every stored row look pending → infinite billed reclassification (bit this once: ~1,740 calls reclassified the same 60 posts before the fix).
- ANALYSIS: `analyze_v4.py` — the post-collection validation pass: feed purity by symbol, v4 feature→15m-return correlations aggregated into independent 1h buckets (serial correlation can't manufacture significance), regime-pulse `trade_action` scored against realized forward returns.
- ECONOMICS: coverage model resolution hardened everywhere — `Store.latest_semantic_model()` returns the most recently *written* model, and `backfill_semantics`/`live._model_name`/`drain_semantics` now resolve coverage as `effective_model → latest_stored → alias`. Pre-resolution, the bare alias `jev-latest` matched no stored rows (`jev-1.13.0`), so every already-classified post looked pending → **billed reclassification**: 1,740 calls in the first drain run, then 1,500 more on a maintenance restart before this landed. Now restart-safe.
- INCIDENT (2026-09-21): TypeSafe/Jev API credits exhausted ~17:35 UTC → every classify returned `402 no available credits` for ~7.5h → ~5.7k-post semantic backlog accumulated while X collection kept running. Credits auto-reloaded ~01:00 UTC; classification resumed. The 17:35→01:00 window is an honest semantic-coverage hole — posts classified after the fact carry late `classified_at` and are correctly excluded from PIT features anchored inside the outage (v4 `semantic_live_rate`/`rel_share` reflect the degradation).

## 0.5.1
Live-operations hardening from the first sustained collection run: feed health, semantic worker resilience, ledger completeness, and label-measurement honesty. No new alpha features.

- OPS: `XRealtimeStream` emits `_control` connected/disconnected lifecycle events; `live` logs them plus `silent` (>90s no frames), `delivery_gap` (>150s pings-alive-no-tweets) and `delivery_recycle` (>300s auto-reconnect) to `feed_health` — vendor zombie sockets and `1011 keepalive` drops are now visible and self-healing.
- OPS: Jev uses one persistent `AsyncTypeSafeClient` (was per-call handshake), classifications carry a timeout, worker exceptions are caught and logged (`classify_error`/`worker_error`), and `queue_saturated` telemetry records post drops instead of losing them silently.
- PROVENANCE: semantic coverage/features resolve `query_model` lazily — the configured alias (`jev-latest`) resolves to the provider's effective model (`jev-1.13.0`) only after the first response, so startup-captured names silently excluded all stored semantics and re-billed redeliveries.
- ECONOMICS: stream deliveries stamp `_semantic_alpha_delivery_id` per vendor frame, and usage is now logged at receipt for **every** delivered tweet — including ones the symbol matcher discards and queue-full drops (billed spend was invisible on both paths); Jev usage `reference_id` distinguishes requested→effective model per call so reclassifications no longer collapse onto the first call's ledger row.
- ECONOMICS: vendor billing model corrected from the per-line-item dashboard — `webhook:tweet_filter` rows bill **exactly 15 credits × delivered tweet count** and `advanced_search` bills ~15 credits per returned tweet, so spend is driven by delivery volume × rule matches, not check cadence. The `estimate_rule_polling`/`filter_rule_poll` estimate (added in this release, premise disproven by the dashboard) was removed — it double-counted deliveries already ledgered as `post_read`. Operationally: the three per-symbol rules were consolidated into one `semantic-alpha-all` rule, because a tweet matching N rules bills N times.
- OPS: `--delivery-recycle-seconds` default (300s) assumed fast delivery cadence; at 600s rule intervals the socket is legitimately quiet ~10min between batches and the watchdog recycled healthy sockets every cycle. Pass `--delivery-recycle-seconds 1200` for 600s rules (recycle only after two missed windows).
- LABELS: exit tolerance is horizon-proportional (`min(max_delay, max(10s, horizon×25%))`) — a "5m" return can no longer silently measure 6.75 minutes; path excursion windows end at the realized exit, and benchmark exits use the same tolerance.
- LABELS: `_merge_label` treats path statistics (`max_up`/`max_down`/`abs_move`/`range_move`/`fade_ratio`) as final only once their horizon realized — partial-window excursions recompute on refill instead of freezing — and `realized_minutes` now merges alongside returns.
- DATA: `bybit_ws` joins canonical market sources (returns/realized-vol/labels were dead in live-only mode); Bybit basis derives from mark/index when the API omits a direct field (matches the Binance snapshot convention).
- OPS: `live` gained a free Binance public-REST poller (`--binance-poll-seconds`, default 20s) reviving `venue_*_dispersion`; `bybit_vs_venues_return_gap_5m` resolves the present canonical Bybit source instead of keying `source="bybit"` literally (live-only mode writes `bybit_ws`, leaving the gap permanently zero); `maintenance-run` re-observes a 6–14-minute-old post slice each cycle (bounded 2 pages/symbol, normalized to base ticker for the query but stored under the tracked perp) so engagement velocity's second observation lands while posts are still inside the 15-minute feature window, and `backfill_semantics` drains posts that were stored but never classified.
- OPS: `repair_event_actionable_at` backfills `actionable_at` on events written before the field existed (derived state repaired from member posts; raw evidence untouched).
- `OFFICIAL_X_ACCOUNTS` populated for SOL/ETH (`@bitcoin` deliberately excluded — BCH-affiliated, unsafe as Bitcoin's official voice).
- Suite: 102 tests, all green; ruff clean.

## 0.5.0
Correctness release: makes time, identity, and economics trustworthy before further alpha work. No new alpha features.

- TIME: `posts_between`/`semantics_between` accept `as_of` and filter on `first_seen_at`/`classified_at`; `FeatureEngine.build` passes `as_of=ts` everywhere, and `leakage_audit` verifies stored features only contain socially-available records — historical social lookahead is now caught, not certified.
- TIME: historical Bybit candles are timestamped at candle close availability (`open_ts + interval`), open time preserved in `raw`; a 12:00 feature no longer knows the 12:00:59 close.
- TIME: labels anchor to the first canonical market snapshot at/after the signal (`base_ts`), record `entry_delay_seconds`/`realized_minutes`, and merge per-horizon — immature rows are refilled by maintenance instead of freezing 1h/4h/24h labels forever.
- DATA: market storage is `(source, symbol, ts)` unique with dedupe-on-write; compaction and data-quality stay per-source so venues never blend.
- DATA: WebSocket accumulator tracks per-topic freshness (ticker/book/trade/liquidation ages, exchange vs receive ts); live snapshotting refuses stale state and records reconnect/feed-health telemetry (`feed-health` command).
- ECONOMICS: `stateful_trade_vectors` charges entry and exit execution costs; `portfolio_metrics` computes portfolio-native equity/Sharpe on a fixed time grid; paper exits price against fresh market snapshots and risk caps dominate floors.
- PROVENANCE: `manifest.py` builds an immutable manifest (dataset hash, feature columns/version, label version, model family+params, thresholds, semantic models, market sources, training window, code hash); `training_fingerprint` is the manifest hash, so materially different models can no longer share an identity. Reports, holdouts and paper trades bind to the same fingerprint.
- PROVENANCE: the configured Jev model is actually sent to the SDK and the returned `effective_model` is recorded as provenance.
- HOLDOUT: holdout identity derives from raw interval/universe/protocol/sampling/label-version/sources — not the feature matrix — locks are written before evaluation, and campaigns hard-stop on quality/leakage failures instead of burning the holdout.
- MULTI-ASSET: event-study semantics join on `(post_id, symbol)`; redirector domains (t.co etc.) no longer cluster narratives, while an exact normalized destination-URL match is decisive; events anchor to actionable observation time (`first_seen_at`/`classified_at`) and persist with stable lifecycle IDs via `event_memberships`.
- STATISTICS: event studies default to event-level inferential units, Benjamini-Hochberg FDR and purged walk-forward preserved; `requirements-lock.txt` pins the tested environment.
- OPERATIONS: X pagination/dedupe/strict-bounds plus transport-level retry with Retry-After/backoff; Bybit REST retry; `ruff` correctness lint gate and GitHub Actions CI (`ci.yml`); wheel + `SHA256SUMS.txt` ship under `dist/`.
- Fixes caught by the new lint gate: `x-add-rules` crashed on an unimported `base_symbol`; dead retry state in `xapi`.
- `tests/test_correctness_release.py` adds 23 regression tests covering every finding; full suite is 91 tests.
- Still no exchange order-placement path.

Ships the previously unreleased post-0.4.0 hardening:

- Platform terminology renamed from Twitter to X throughout code, CLI (`x-window`, `x-add-rules`, `x-rules`), config (`X_API_KEY`, `X_QUERY_LANGS`, ...) and docs; `twitterapi.io` remains only as vendor/billing identity and in wire-level paths/response keys.
- Labels v4: `abs_move` (magnitude), `range_move` (volatility), `fade_ratio` (crowding/reversal) heads alongside direction; `dataset()` exposes them as `target_abs_move`/`target_range_move`/`target_fade_ratio`.
- Residualized social features (`*_residual_5m`, `semantic_shock_residual`): ridge fit of E[social|market] on strictly-trailing feature history, controlling price→posts reverse causality.
- Deterministic source features: `verified_author_rate_*`, `official_account_rate_*` from account metadata.
- Binance USD-M public provider (`providers/binance.py`) + `--venues bybit,binance` on collectors; venue-aware canonical market reads (`bybit`/`bybit_history` only) plus `venue_*_dispersion` and `bybit_vs_venues_return_gap_5m` features.
- `providers/rss.py` + `collect-news`: RSS/Atom primary sources normalized into the post pipeline (unmatched items land under `MACRO`).
- `adversarial.py` + `adversarial-benchmark`: prompt-injection/fake-attribution/spam/negation battery with per-class manipulability scoring.
- `overfit.py` + `overfit-report`/`trials`: Deflated Sharpe, combinatorial purged CV with PBO, and an append-only `experiment_trials` ledger written by research-suite/locked-research/overfit runs.
- Per-symbol social watermarks (`social_cursors`) stop the forward recorder re-fetching overlapping windows; X search retries 429 with backoff.
- `latency-report` (ingest vs semantic lag), `capacity-report` (cost curve by notional from stored depth), `semantic-reliability` (binned prob-head vs realized |abnormal return|).
- Config: `X_QUERY_LANGS`, `BINANCE_BASE_URL`, `NEWS_FEEDS`, `OFFICIAL_X_ACCOUNTS`.

## 0.4.0
Anti-overfit, economics and operational-validity release.

- One-shot persistent locked holdout registry; repeated peeking is refused unless explicitly marked compromised.
- Discovery-only horizon/model search followed by one sealed temporal holdout.
- Event-triggered decision sampling to remove dense-snapshot pseudo-replication.
- Holdout cost/threshold fragility stress tests and probability-calibration diagnostics.
- Semantic permutation/staleness controls.
- Parallel feature versions for Jev-vs-heuristic/other semantic-engine comparisons on common timestamps.
- Cross-sectional raw-factor IC and monotonicity diagnostics.
- Probabilistic-Sharpe diagnostic in trading metrics.
- Risk/volatility/depth-aware paper sizing, stale-data gates and one-trade-per-feature enforcement.
- Paper attribution by asset, side, regime and semantic-edge bucket.
- X/Jev provider usage ledger and operating cost report.
- Storage-health projections and safe dry-run market compaction.
- Full campaign orchestrator and stricter champion gate requiring uncompromised locked holdout + forward evidence.
- Multi-asset social identity: canonical posts map to many assets; Jev semantics are asset-specific, and live X ingestion detects all configured assets in one post.
- Provider usage accounting is keyed by acquisition/delivery identity rather than storage novelty, reducing cost undercounting when paid searches return the same post.
- Still no exchange execution path.

## 0.3.0
Research-governance and alpha-validation release.

- Learned paired semantic-residual model: market+attention baseline versus full semantic model, with semantic log-odds residual.
- Stronger market-only baseline: multi-horizon returns/OI changes, realized volatility, flow/book/liquidation aggregates and cross-asset context.
- Rolling beta-adjusted abnormal-return labels plus MFE/MAE-style path labels.
- Purged/embargoed chronological evaluation at the target horizon; no train label may mature inside the test window.
- Stateful non-overlapping portfolio simulation so dense feature snapshots cannot manufacture independent trades.
- Point-in-time author reputation snapshots with maturity checks and Bayesian shrinkage.
- Cross-asset context and historical v3 feature reconstruction.
- Provenance factors distinguishing genuinely forward-observed posts/semantics from historical reconstruction.
- Evidence gate separates `PROMISING_HISTORICAL_ONLY` from `PROMISING_FORWARD_VALIDATED`.
- Horizon scan, leave-one-symbol-out, market-regime, placebo/null, drift and feature-importance diagnostics.
- Event-level semantic studies: copied posts are collapsed to narrative events; bootstrap CIs and Benjamini-Hochberg FDR q-values are reported across buckets/horizons.
- Experiment manifests fingerprint the exact research matrix and configuration.
- Paper trades are bound to the exact model training fingerprint.
- Candidate/champion registry: promotion requires forward evidence, clean leakage audit, acceptable drift and a minimum diversified paper-trading record.
- Still no exchange order execution path.

## 0.2.0
Research/live-data expansion: engagement curves, provenance semantics, narrative diffusion, Bybit/X WebSockets, historical Bybit backfill, event studies, dynamic costs, nonlinear model baseline, cross-sectional tests, paired incremental-value bootstrap, leakage and data-quality gates.

## 0.1.0
Initial research-first rebuild: point-in-time storage, Jev semantic vectors, Bybit REST market state, X bounded historical ingestion, basic features, labels, ablations and forward recorder.
