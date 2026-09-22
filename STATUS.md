# Semantic Alpha Lab — build status

## Current version
v0.5.2 — data-quality release: semantic aggregates are relevance-weighted (Jev `RELEVANT`=1 / `MAYBE`=0.5 / `IRRELEVANT`=0), `rel_posts`/`rel_share` expose feed purity, and `feature_version` bumps v3→v4. Live X collection runs on a single `semantic-alpha-all` rule (vendor bills ~15cr per delivered tweet-instance per matching rule — N rules meant N× billing); usage is ledgered at receipt for every delivery. Collection **resumed 2026-09-21 ~03:26 UTC** on the fixed stack: narrative-similarity batching removed the multi-minute event-loop stalls that caused keepalive drops and billed redeliveries, `ping_timeout` raised to 120s, features build every 60s, and engagement-refresh search is disabled (it was ~85% of `post_read` spend for a noise-level feature).

## Pilot-dataset audit (2026-09-20 08:39 → 2026-09-21 01:50 UTC, ~17h)
Facts, not alpha claims:
- 9,471 posts / 10,616 semantics / 14,380 market snaps / 2,171 v3 features / 11,000 labels / 7,017 events / 220 author reputations. Zero negative lags; created→seen p50=407s (600s rule batch), seen→classified p50=1037s; only 19 asset rows await semantics.
- Jev output verified by reading raw pairs: pump-spam → PROMOTION+shill≈0.9+IRRELEVANT, TA commentary → MARKET_ANALYSIS+reactive≈0.88. Prob fields show real variance (82–270 distinct values).
- Feed purity is the binding data issue: SOL 60.7% IRRELEVANT (memecoin-spam keyword matches), ETH 25.7%, BTC 21.1%; ~14–16% of BTC/SOL posts don't mention the asset (thread-context deliveries).
- Semantic rates co-move at rho≈0.97 — one latent conviction-density factor, effective test count ≈1.
- The +0.19 feature-level partial rho vs 15m returns did NOT replicate: event level n=6,999 rho≈0; independent bucket level n=125 rho≈0; relevance-filtered also ≈0. Best current explanation: spam-intensity co-moving with microstructure noise (strongest on the spammiest feed, weakest on the cleanest). No tradeable effect demonstrated.

## Verdict (2026-09-22 — collection stopped by decision)

The core research question — *does X semantic content provide incremental tradeable alpha at this pipeline's latency* — is answered **no, not demonstrably**. Battery on 45 independent 1h buckets of v4 data: one weak survivor (`intent_mean_15m`→BTCUSDT, raw ρ=+0.39, ~+0.19–0.33 after momentum/vol/time-detrend/within-regime controls), plausibly real but likely sub-cost at 15m horizon and regime-dependent (window was a +7% rally). Both earlier leads were falsified on contact (v3's +0.19 → spam artifact; `liquidation_imbalance` → momentum echo with a sign flip). The X rule is deleted, all processes stopped, ~$1.1 vendor balance preserved. A locked holdout spec (`holdout_v4_btc_intent.json`, fp `71f98143c326f2f3`) and `holdout_test.py`/`battery_v4.py`/`analyze_v4.py` remain in-repo — restarting collection under the same rule and firing the holdout on ≥36 fresh buckets is the resume path if the question is ever reopened.

## Green checks
- 104 regression tests pass, including `tests/test_correctness_release.py` (one test per audit finding).
- v4 semantic features are relevance-weighted: IRRELEVANT posts contribute zero to every semantic aggregate; `rel_posts`/`rel_share` measure feed purity directly.
- live stack is self-healing: X stream emits connect/disconnect lifecycle telemetry, zombie sockets and delivery gaps auto-recycle, semantic workers survive provider errors/timeouts, queue saturation is logged; Binance public REST keeps venue-dispersion features fed; maintenance re-observes trailing posts so engagement-velocity features resolve; `bybit_ws` rows are canonical market truth (live-only DBs retain full history/labels).
- package compiles cleanly; `ruff check` (correctness rules E9/F/B) is clean; CI runs lint + compile + tests on Python 3.11/3.12.
- `requirements-lock.txt` pins the tested environment; `dist/` ships the wheel + `SHA256SUMS.txt`.
- historical features only contain posts/semantics available at feature time (`as_of` on `first_seen_at`/`classified_at`); the leakage audit verifies this.
- historical candles are timestamped at close availability; a feature can no longer know a bar's close before it completes.
- labels merge per-horizon and refill when data matures; long horizons can't freeze as `None` forever. Exit tolerance is horizon-proportional (≤25%), partial-window path stats recompute instead of freezing, `realized_minutes` fills on refill, and legacy events missing `actionable_at` are repaired from member posts' observed times.
- market writes are idempotent per `(source, symbol, ts)`; venue timelines never blend.
- dead WebSocket feeds cannot masquerade as fresh data (per-topic freshness, exchange vs receive ts, feed-health telemetry).
- backtests charge round-trip costs; portfolio metrics are portfolio-native.
- models, reports, holdouts and paper trades bind to one immutable manifest fingerprint.
- holdout identity derives from the raw window/universe/protocol; locks precede evaluation; campaigns hard-stop on quality/leakage failures.
- the configured Jev model is actually sent and the effective model is recorded.
- event semantics join on `(post_id, symbol)`; redirector URLs don't over-cluster; exact destination-URL matches are decisive.
- maintenance is bounded: label existence/anchors resolve from bulk loads (no per-post queries), the scan window is 72h, immature labels rebuild only once their earliest open horizon can have realized, and labels stuck open past the 26h deadline finalize as `matured` with `None` horizons — the full-table scan that made ticks ~80min is gone.
- classification failures are ledgered per (post, symbol, model): content-side failures cap at 3 attempts then stop being retried (no infinite billed churn on deterministic junk); provider-side failures (402 credits, 429, timeouts) never count toward the cap — the 2026-09-21 17:35→01:00 TypeSafe credit outage produced a ~5.7k-post backlog that drains automatically once credits return, and `backfill_semantics` survives single-post errors plus hangs (per-call timeout, stops only after 5 consecutive provider errors).
- the usage ledger keys by *billed event*: every delivered tweet logs
  `post_read` at receipt via a per-frame `_semantic_alpha_delivery_id`
  (symbol-unmatched, deduped, and queue-dropped deliveries included — the
  vendor bills ~15 credits per delivery), and Jev usage refs distinguish
  requested→effective model per call. Vendor billing is per delivered
  tweet-instance (~15cr; checks themselves are free), so the earlier
  `filter_rule_poll` estimate was removed — it double-counted deliveries.
  One combined `semantic-alpha-all` rule now feeds all three assets:
  a tweet matching N per-symbol rules bills N times.
- leading-edge X searches early-stop on already-known pages (`known_ids` in
  `search_window`) — billed-per-tweet duplicates are no longer re-bought;
  backfill/refresh paths keep full pagination by design.
- `regime_pulses` stores a holistic Jev `system_one` reading per symbol per
  maintenance cycle (fixed digest v1, ~$0.02/day, usage ledgered) — a
  registered candidate channel, deliberately not wired into any feature
  version pending validation on untouched data.
- no exchange order execution path exists.

## Evidence ladder (unchanged)
1. immutable point-in-time data + data-quality/leakage gates;
2. raw factor IC/monotonicity and event-level FDR-controlled studies;
3. event-triggered decision sampling;
4. discovery-only horizon/model search;
5. one sealed temporal holdout with 2× cost + threshold-fragility stress;
6. semantic alignment/staleness controls and cross-asset/regime robustness;
7. forward-observation evidence gate;
8. frozen forward candidate fingerprint;
9. fresh-data-only, one-feature-once paper trading;
10. paper P&L attribution and API/data cost accounting;
11. uncompromised holdout + forward evidence + paper/drift gates before champion eligibility.

## What remains intentionally absent
- no live exchange keys/order placement;
- no claim of profitability;
- no automatic strategy promotion from historical results alone;
- no dependence on a single semantic model: Jev must beat simpler semantic controls on common observations.

## Next evidence milestone
Run the forward recorder (`collect-forward --venues bybit,binance`) long enough to accumulate multiple assets/regimes — the dataset is the moat and only accrues forward. Then execute one `semantic-alpha campaign`. If the locked holdout fails or Jev semantics do not beat simpler semantic/attention controls after costs, stop or pivot rather than adding model complexity.
